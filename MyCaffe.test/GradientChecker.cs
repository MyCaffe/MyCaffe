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

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="dfStepSize">Step size of the gradient test.</param>
        /// <param name="dfThreshold">Error threshold.</param>
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
        }

        /// <summary>
        /// Checks the gradient of a layer, with provided bottom layers and top layers.
        /// </summary>
        /// <remarks>
        /// Note that after the gradient check, we do not guarantee that the data stored
        /// in the layer parameters and the blobs are unchanged.
        /// </remarks>
        public void CheckGradient(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, int nCheckBottom = -1)
        {
            layer.Setup(colBottom, colTop);
            CheckGradientSingle(layer, colBottom, colTop, nCheckBottom, -1, -1);
        }

        /// <summary>
        /// Checks the gradient of a single output with respect to particular input
        /// blob(s).  If check_bottom = i >= 0, check only the ith bottom Blob<T>.
        /// If check_bottom == -1, check everything -- all bottom Blobs and all
        /// param Blobs.  Otherwise (if check_bottom less than -1), check only param Blobs.
        /// </summary>
        public void CheckGradientSingle(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, int nCheckBottom, int nTopID, int nTopDataID, bool bElementwise = false)
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

            // Get additional loss from the objective.
            GetObjAndGradient(layer, colTop, nTopID, nTopDataID);
            layer.Backward(colTop, rgPropagateDown, colBottom);

            // Store computed gradients for all checked blobs
            BlobCollection<T> colComputedGradientBlobs = new BlobCollection<T>();

            for (int nBlobID = 0; nBlobID < colBlobsToCheck.Count; nBlobID++)
            {
                Blob<T> current_blob = colBlobsToCheck[nBlobID];
                Blob<T> new_blob = new Blob<T>(m_cuda, m_log);

                new_blob.ReshapeLike(current_blob);
                m_cuda.copy(current_blob.count(), current_blob.gpu_diff, new_blob.mutable_gpu_data);

                colComputedGradientBlobs.Add(new_blob);
            }

            // Compute derivative of top w.r.t. each bottom and parameter input using
            // finite differencing.

            for (int nBlobID = 0; nBlobID < colBlobsToCheck.Count; nBlobID++)
            {
                Blob<T> current_blob = colBlobsToCheck[nBlobID];
                T[] rgdfComputedGradients = colComputedGradientBlobs[nBlobID].update_cpu_data();
                double dfData;

                for (int nFeatID=0; nFeatID<current_blob.count(); nFeatID++)
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
                        dfData = (double)Convert.ChangeType(current_blob.GetData(nFeatID), typeof(double));
                        dfData += m_dfStepsize;
                        current_blob.SetData(dfData, nFeatID);
                        m_cuda.rng_setseed(m_uiSeed);

                        layer.Forward(colBottom, colTop);
                        dfPositiveObjective = GetObjAndGradient(layer, colTop, nTopID, nTopDataID);

                        // Compute loss with stepsize subtracted from input.
                        dfData = (double)Convert.ChangeType(current_blob.GetData(nFeatID), typeof(double));
                        dfData -= (m_dfStepsize * 2);
                        current_blob.SetData(dfData, nFeatID);
                        m_cuda.rng_setseed(m_uiSeed);

                        layer.Forward(colBottom, colTop);
                        dfNegativeObjective = GetObjAndGradient(layer, colTop, nTopID, nTopDataID);

                        // Recover original input value.
                        dfData = (double)Convert.ChangeType(current_blob.GetData(nFeatID), typeof(double));
                        dfData += m_dfStepsize;
                        current_blob.SetData(dfData, nFeatID);

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
                }
            }
        }

        public void CheckGradientExhaustive(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop, int nCheckBottom = -1)
        {
            layer.Setup(colBottom, colTop);

            m_log.CHECK_GT(colTop.Count, 0, "Exhaustive mode requires at least one top blob.");

            for (int i = 0; i < colTop.Count; i++)
            {
                for (int j = 0; j < colTop[i].count(); j++)
                {
                    CheckGradientSingle(layer, colBottom, colTop, nCheckBottom, i, j);
                }
            }
        }

        /// <summary>
        /// CheckGradientEltwise can be used to test layers that perform element-wise
        /// computation only (e.g. neuron layers) -- where (d y_i)/(d x_j) = 0 when
        /// i != j.
        /// </summary>
        public void CheckGradientEltwise(Layer<T> layer, BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            layer.Setup(colBottom, colTop);

            m_log.CHECK_GT(colTop.Count, 0, "Eltwise mode requires at least one top blob.");
            
            int nCheckBottom = -1;
            bool bElementWise = true;

            for (int i = 0; i < colTop.Count; i++)
            {
                for (int j = 0; j < colTop[i].count(); j++)
                {
                    CheckGradientSingle(layer, colBottom, colTop, nCheckBottom, i, j, bElementWise);
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

        protected double GetObjAndGradient(Layer<T> layer, BlobCollection<T> colTop, int nTopID = -1, int nTopDataID = -1)
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
