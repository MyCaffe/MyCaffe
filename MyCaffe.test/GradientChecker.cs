using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe;
using MyCaffe.common;
using MyCaffe.basecode;
using System.Threading;
using MyCaffe.layers;
using System.Diagnostics;

namespace MyCaffe.test
{
    /// <summary>
    /// The gradient checker adds a L2 normalization loss function on top of the
    /// top blobs, andn checks the gradient.
    /// </summary>
    public class GradientChecker<T>
    {
        CudaDnn<T> m_cuda;
        Log m_log;
        protected double m_dfStepsize;
        protected double m_dfThreshold;
        protected uint m_uiSeed;
        protected double m_dfKink;
        protected double m_dfKinkRange;
        protected EventWaitHandle m_evtCancel = new EventWaitHandle(false, EventResetMode.AutoReset, "__GRADIENT_CHECKER_CancelEvent__");
        protected string m_strBaseType;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CUDA connection</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="dfStepSize">Step size of the gradient test.</param>
        /// <param name="dfThreshold">Error threshold.</param>
        /// <param name="nSeed">Specifies the random seed (default = 1701)</param>
        /// <param name="dfKink">Specifies the kink (default = 0)</param>
        /// <param name="dfKinkRange">Specifies the kink range (default = -1)</param>
        /// <remarks>
        /// The kink and kink_range specify an ignored nonsmooth region of the form
        /// kink - kink_range less than or equal |feature value| less than or equal kink + kink_range,
        /// which accounts for all nonsmoothness in use by caffe.
        /// </remarks>
        public GradientChecker(CudaDnn<T> cuda, Log log, double dfStepSize = 1e-2, double dfThreshold = 1e-3, uint nSeed = 1701, double dfKink = 0.0, double dfKinkRange = -1)
        {
            m_cuda = cuda;
            m_log = log;
            m_dfStepsize = dfStepSize;
            m_dfThreshold = dfThreshold;
            m_uiSeed = nSeed;
            m_dfKink = dfKink;
            m_dfKinkRange = dfKinkRange;
            m_strBaseType = (typeof(T) == typeof(double)) ? "DOUBLE" : "FLOAT";
        }

        /// <summary>
        /// Checks the gradient of a layer, with provided bottom layers and top layers.
        /// </summary>
        /// <param name="layer">Specifies the layer to test.</param>
        /// <param name="colBottom">Specifies the bottom (inputs)</param>
        /// <param name="colTop">Specifies the top (outputs)</param>
        /// <param name="nCheckBottom">Specifies to check the bottom at this index, or all if -1.</param>
        /// <param name="nFeatureStep">Specifies the feature step (default = 1)</param>
        /// <param name="dfDynamicFeatureStepPct">Specifies a dynamic percentage based feature step that when > 0 is applied to the blob count to get the step (default = 0)</param>
        /// <param name="nFirstFeature">Specifies the first feature to check (default = 0)</param>
        /// <remarks>
        /// Note that after the gradient check, we do not guarantee that the data stored
        /// in the layer parameters and the blobs are unchanged.
        /// </remarks>
        public void CheckGradient(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, int nCheckBottom = -1, int nFeatureStep = 1, double dfDynamicFeatureStepPct = 0, int nFirstFeature = 0)
        {
            layer.Setup(colBottom, colTop);
            CheckGradientSingle(layer, colBottom, colTop, nCheckBottom, -1, -1, false, nFeatureStep, dfDynamicFeatureStepPct, nFirstFeature);
        }

        /// <summary>
        /// Checks the gradient of a single output with respect to particular input
        /// blob(s).  If check_bottom = i >= 0, check only the ith bottom Blob<T>.
        /// If check_bottom == -1, check everything -- all bottom Blobs and all
        /// param Blobs.  Otherwise (if check_bottom less than -1), check only param Blobs.
        /// </summary>
        /// <param name="layer">Specifies the layer to test.</param>
        /// <param name="colBottom">Specifies the bottom (inputs)</param>
        /// <param name="colTop">Specifies the top (outputs)</param>
        /// <param name="nCheckBottom">Specifies to check the bottom at this index, or all if -1.</param>
        /// <param name="nTopID">Specifies the id of the top to check.</param>
        /// <param name="nTopDataID">Specifies to check the top data to check.</param>
        /// <param name="bElementwise">Specifies to check elementwise (default = false).</param>
        /// <param name="nFeatureStep">Specifies the feature step (default = 1)</param>
        /// <param name="dfDynamicFeatureStepPct">Specifies a dynamic percentage based feature step that when > 0 is applied to the blob count to get the step (default = 0)</param>
        /// <param name="nFirstFeature">Specifies the first feature to check (default = 0)</param>
        public void CheckGradientSingle(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, int nCheckBottom, int nTopID, int nTopDataID, bool bElementwise = false, int nFeatureStep = 1, double dfDynamicFeatureStepPct = 0, int nFirstFeature = 0)
        {
            // Store computed gradients for all checked blobs
            BlobCollection<T> colComputedGradientBlobs = new BlobCollection<T>();

            try
            {
                if (bElementwise)
                {
                    m_log.CHECK_EQ(0, layer.blobs.Count(), "Cannot have blobs in the layer checked for element-wise checking.");
                    m_log.CHECK_LE(0, nTopID, "The top ID '" + nTopID.ToString() + "' must be zero or greater with element-wise checking.");
                    m_log.CHECK_LE(0, nTopDataID, "The top data ID '" + nTopDataID.ToString() + "' must be zero or greater with element-wise checking.");

                    int nTopCount = colTop[nTopID].count();

                    for (int nBlobID = 0; nBlobID < colBottom.Count(); nBlobID++)
                    {
                        m_log.CHECK_EQ(nTopCount, colBottom[nBlobID].count(), "The top count and blob counts must be equal for element-wise checking.");
                    }
                }

                // First, figure out what blobs we need to check against, and zero init
                // parameter blobs.
                BlobCollection<T> colBlobsToCheck = new BlobCollection<T>();
                List<bool> rgPropagateDown = new List<bool>();

                for (int i = 0; i < colBottom.Count; i++)
                {
                    rgPropagateDown.Add((nCheckBottom == -1) ? true : false);
                }

                for (int i = 0; i < layer.blobs.Count; i++)
                {
                    Blob<T> blob = layer.blobs[i];

                    blob.SetDiff(0);
                    colBlobsToCheck.Add(blob);
                }

                if (nCheckBottom == -1)
                {
                    for (int i = 0; i < colBottom.Count; i++)
                    {
                        colBlobsToCheck.Add(colBottom[i]);
                    }
                }
                else if (nCheckBottom >= 0)
                {
                    m_log.CHECK_LT(nCheckBottom, colBottom.Count, "The check bottom value '" + nCheckBottom.ToString() + "' must be less than the number of bottom blobs.");
                    colBlobsToCheck.Add(colBottom[nCheckBottom]);
                    rgPropagateDown[nCheckBottom] = true;
                }

                m_log.CHECK_GT(colBlobsToCheck.Count, 0, "No blobs to check!");

                // Compute the gradient analytically using Backward.
                m_cuda.rng_setseed(m_uiSeed);

                // Ignore the loss from the layer (it's just the weighted sum of the losses
                // from the top blobs, whose gradients we may want to test individually).
                layer.Forward(colBottom, colTop);
                layer.ConvertToBase(colTop);

                // Get additional loss from the objective.
                GetObjAndGradient(layer, colTop, nTopID, nTopDataID);
                layer.Backward(colTop, rgPropagateDown, colBottom);
                layer.ConvertToBase(colBlobsToCheck);

                for (int nBlobID = 0; nBlobID < colBlobsToCheck.Count; nBlobID++)
                {
                    Blob<T> current_blob = colBlobsToCheck[nBlobID];
                    Blob<T> new_blob = new Blob<T>(m_cuda, m_log);

                    if (current_blob.DiffExists)
                    {
                        new_blob.ReshapeLike(current_blob);
                        m_cuda.copy(current_blob.count(), current_blob.gpu_diff, new_blob.mutable_gpu_data);
                    }

                    colComputedGradientBlobs.Add(new_blob);
                }

                // Compute derivative of top w.r.t. each bottom and parameter input using
                // finite differencing.
                long lTotal = 0;
                for (int nBlobID = 0; nBlobID < colBlobsToCheck.Count; nBlobID++)
                {
                    lTotal += colBlobsToCheck[nBlobID].count();
                }

                Stopwatch sw = new Stopwatch();
                sw.Start();

                long lIdx = nFirstFeature;
                for (int nBlobID = 0; nBlobID < colBlobsToCheck.Count; nBlobID++)
                {
                    Blob<T> current_blob = colBlobsToCheck[nBlobID];

                    if (!current_blob.DiffExists)
                        continue;

                    T[] rgdfComputedGradients = colComputedGradientBlobs[nBlobID].update_cpu_data();

                    Trace.WriteLine("** BLOB " + nBlobID.ToString() + " of " + colBlobsToCheck.Count.ToString() + " **");

                    if (dfDynamicFeatureStepPct > 0)
                    {
                        nFeatureStep = (int)(current_blob.count() * dfDynamicFeatureStepPct);
                        if (nFeatureStep <= 0)
                            nFeatureStep = 1;
                        Trace.WriteLine("Using dynamic feature step of " + nFeatureStep.ToString() + " for blob " + nBlobID.ToString() + ".");
                    }

                    for (int nFeatID = nFirstFeature; nFeatID < current_blob.count(); nFeatID += nFeatureStep)
                    {
                        if (m_evtCancel.WaitOne(0))
                            throw new Exception("Aborted!");

                        // For an element-wise layer, we only need to do finite differencing to
                        // compute the derivative of top[nTopID][nTopDataID] w.r.t.
                        // bottom[nBlobID][i] only for i == nTopDataID.  For any otehr
                        // i != nTopDataID, we know the derivative is 0 by definition, and simply
                        // check that that's true.
                        double dfEstimateGradient = 0;
                        double dfPositiveObjective = 0;
                        double dfNegativeObjective = 0;

                        if (!bElementwise || (nFeatID == nTopDataID))
                        {
                            // Do finite differencing.
                            // Compute loss with stepwise added to input.
                            set_at(current_blob, nFeatID, m_dfStepsize, true);

                            m_cuda.rng_setseed(m_uiSeed);

                            layer.Forward(colBottom, colTop);
                            layer.ConvertToBase(colTop);
                            dfPositiveObjective = GetObjAndGradient(layer, colTop, nTopID, nTopDataID);

                            // Compute loss with stepsize subtracted from input.
                            set_at(current_blob, nFeatID, m_dfStepsize * -2, true);

                            m_cuda.rng_setseed(m_uiSeed);

                            layer.Forward(colBottom, colTop);
                            layer.ConvertToBase(colTop);
                            dfNegativeObjective = GetObjAndGradient(layer, colTop, nTopID, nTopDataID);

                            // Recover original input value.
                            set_at(current_blob, nFeatID, m_dfStepsize, true);

                            dfEstimateGradient = (dfPositiveObjective - dfNegativeObjective) / m_dfStepsize / 2.0;
                        }

                        double dfComputedGradient = (double)Convert.ChangeType(rgdfComputedGradients[nFeatID], typeof(double));
                        double dfFeature = (double)Convert.ChangeType(current_blob.GetData(nFeatID), typeof(double));

                        if (m_dfKink - m_dfKinkRange > Math.Abs(dfFeature) ||
                            Math.Abs(dfFeature) > m_dfKink + m_dfKinkRange)
                        {
                            // We check the relative accuracy, but for too small values, we threshold
                            // the scale factor by 1.
                            double dfScale = Math.Max(Math.Max(Math.Abs(dfComputedGradient), Math.Abs(dfEstimateGradient)), 1.0);

                            m_log.EXPECT_NEAR(dfComputedGradient, dfEstimateGradient, m_dfThreshold * dfScale, "DEBUG: (nTopID, nTopDataID, nBlobID, nFeatID)=" + nTopID.ToString() + ", " + nTopDataID.ToString() + ", " + nBlobID.ToString() + ", " + nFeatID.ToString() + "; feat = " + dfFeature.ToString() + "; objective+ = " + dfPositiveObjective.ToString() + "; objective- = " + dfNegativeObjective.ToString());
                        }

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            sw.Restart();
                            double dfPct = (double)lIdx / lTotal;
                            double dfPctBlob = (double)nFeatID / current_blob.count();
                            Trace.WriteLine("Checking BLOB " + nBlobID.ToString() + " '" + current_blob.Name + "' gradient at " + lIdx.ToString("N0") + " of " + lTotal.ToString("N0") + " - blob " + dfPctBlob.ToString("P") + " global " + dfPct.ToString("P") + "...");
                        }

                        lIdx += nFeatureStep;
                    }
                }
            }
            finally
            {
                colComputedGradientBlobs.Dispose();
            }
        }

        private void set_at(Blob<T> blob, int nFeatID, double dfStep, bool bVerify = false)
        {
            T val = blob.GetData(nFeatID);

            if (typeof(T) == typeof(float))
            {
                float fVal = (float)Convert.ChangeType(val, typeof(float));
                fVal += (float)dfStep;
                blob.SetData(fVal, nFeatID);

                if (bVerify)
                {
                    val = blob.GetData(nFeatID);
                    float fAct = (float)Convert.ChangeType(val, typeof(float));

                    m_log.CHECK_EQ(fVal, fAct, "The values are not the same!");
                }
            }
            else
            {
                double fVal = (double)Convert.ChangeType(val, typeof(double));
                fVal += (double)dfStep;
                blob.SetData(fVal, nFeatID);

                if (bVerify)
                {
                    val = blob.GetData(nFeatID);
                    double fAct = (double)Convert.ChangeType(val, typeof(double));

                    m_log.CHECK_EQ(fVal, fAct, "The values are not the same!");
                }
            }
        }

        public void CheckGradientExhaustive(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, int nCheckBottom = -1)
        {
            Stopwatch sw = new Stopwatch();
            layer.Setup(colBottom, colTop);

            m_log.CHECK_GT(colTop.Count, 0, "Exhaustive mode requires at least one top blob.");
            sw.Start();

            int nTotal = 0;
            int nIdx = 0;

            for (int i = 0; i < colTop.Count; i++)
            {
                nTotal += colTop[i].count();
            }

            TestingProgressSet progress = new TestingProgressSet();

            for (int i = 0; i < colTop.Count; i++)
            {
                for (int j = 0; j < colTop[i].count(); j++)
                {
                    CheckGradientSingle(layer, colBottom, colTop, nCheckBottom, i, j);
                    nIdx++;

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)nIdx / (double)nTotal;
                        Trace.WriteLine(m_strBaseType + ": Check gradient exhaustive at " + dfPct.ToString("P") + "...");

                        progress.SetProgress(dfPct);

                        sw.Restart();
                    }
                }
            }

            progress.SetProgress(0);
            progress.Dispose();
        }

        /// <summary>
        /// CheckGradientEltwise can be used to test layers that perform element-wise
        /// computation only (e.g. neuron layers) -- where (d y_i)/(d x_j) = 0 when
        /// i != j.
        /// </summary>
        public void CheckGradientEltwise(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Stopwatch sw = new Stopwatch();
            layer.Setup(colBottom, colTop);

            m_log.CHECK_GT(colTop.Count, 0, "Eltwise mode requires at least one top blob.");
            sw.Start();

            int nCheckBottom = -1;
            bool bElementWise = true;

            int nTotal = 0;
            int nIdx = 0;

            for (int i = 0; i < colTop.Count; i++)
            {
                nTotal += colTop[i].count();
            }

            TestingProgressSet progress = new TestingProgressSet();

            for (int i = 0; i < colTop.Count; i++)
            {
                for (int j = 0; j < colTop[i].count(); j++)
                {
                    CheckGradientSingle(layer, colBottom, colTop, nCheckBottom, i, j, bElementWise);
                    nIdx++;

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)nIdx / (double)nTotal;
                        Trace.WriteLine(m_strBaseType + ": Check gradient eltwise at " + dfPct.ToString("P") + "...");

                        progress.SetProgress(dfPct);

                        sw.Restart();
                    }
                }
            }
        }

        public void CheckGradientNet(Net<T> net, BlobCollection<T> colInput)
        {
            List<Layer<T>> colLayers = net.layers;
            List<BlobCollection<T>> rgrgBottomVecs = net.bottom_vecs;
            List<BlobCollection<T>> rgrgTopVecs = net.top_vecs;

            for (int i = 0; i < colLayers.Count; i++)
            {
                double dfLoss;
                net.Forward(colInput, out dfLoss);

                m_log.WriteLine("Checking gradient for " + colLayers[i].layer_param.name);
                CheckGradientExhaustive(colLayers[i], rgrgBottomVecs[i], rgrgTopVecs[i]);
            }
        }

        public double GetObjAndGradient(Layer<T> layer, BlobCollection<T> colTop, int nTopID = -1, int nTopDataID = -1)
        {
            double dfLoss = 0;

            // The loss will be half of the sum of squares of all outputs.
            if (nTopID < 0)
            {
                for (int i = 0; i < colTop.Count; i++)
                {
                    Blob<T> top_blob = colTop[i];
                    T[] rgTopData = top_blob.update_cpu_data();
                    int nCount = top_blob.count();

                    for (int j = 0; j < nCount; j++)
                    {
                        double dfTopData = (double)Convert.ChangeType(rgTopData[j], typeof(double));
                        dfLoss += dfTopData * dfTopData;
                    }

                    m_cuda.copy(nCount, top_blob.gpu_data, top_blob.mutable_gpu_diff);
                }

                dfLoss /= 2.0;
            }

            // The loss will be the nTopDataID'th element in the nTopIDth blob.
            else
            {
                for (int i = 0; i < colTop.Count; i++)
                {
                    colTop[i].SetDiff(0);
                }

                double dfLossWeight = 2.0;
                T[] rgData = colTop[nTopID].mutable_cpu_data;
                double dfTopVal = (double)Convert.ChangeType(colTop[nTopID].GetData(nTopDataID), typeof(double));

                if ((double)Convert.ChangeType(rgData[nTopDataID], typeof(double)) != dfTopVal)
                    throw new Exception("Top data did not set correctly!");

                dfLoss = dfTopVal * dfLossWeight;
                colTop[nTopID].SetDiff(dfLossWeight, nTopDataID);

                T[] rgDiff = colTop[nTopID].update_cpu_diff();
                if ((double)Convert.ChangeType(rgDiff[nTopDataID], typeof(double)) != dfLossWeight)
                    throw new Exception("Top diff did not set correctly!");
            }

            return dfLoss;
        }
    }
}
