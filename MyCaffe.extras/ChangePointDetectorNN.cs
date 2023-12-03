using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param.beta;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.solvers;
using System.Diagnostics;

///WORK IN PROGRESS
namespace MyCaffe.extras
{
    /// <summary>
    /// The ChangePointDetector is used to detect change points in a time series using a simple neural network.
    /// </summary>
    /// <remarks>
    /// @see [A Contrastive Approach to Online Change Point Detection](https://arxiv.org/abs/2206.10143) by Artur Goldman, Nikita Puchkin, Valeriia Shcherbakova, and Uliana Vinogradova, 2022, arXiv
    /// @see [Numerical experiments on the WISDM data set described in the paper "A Contrastive Approach to Online Change Point Detection"](https://github.com/npuchkin/contrastive_change_point_detection/blob/main/WISDM_experiments.ipynb) by npuchkin, GitHub 2023
    /// </remarks>
    /// <typeparam name="T">Specifies the base type of <i>float</i> or <i>double</i>.</typeparam>
    public class ChangePointDetectorNN<T> : IDisposable
    {
        CudaDnn<T> m_cuda;
        Log m_log;
        Blob<T> m_blobTSlice = null;
        Blob<T> m_blobT = null;
        ChangePointCandidateCollection<T> m_colCandidates = new ChangePointCandidateCollection<T>();
        int m_nN;
        int m_nB;
        List<int> m_rgGpuID = new List<int>();

        /// <summary>
        /// Optionally, specifies the status event fired to show progress.
        /// </summary>
        public event EventHandler<LogArg> OnStatus;

        /// <summary>
        /// The ChangePointDetector constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to CUDA.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="strGpus">Optionally, specifies the GPU's to run the internal threads on specified as a comma delinated list (ex. '0,1').</param>
        public ChangePointDetectorNN(CudaDnn<T> cuda, Log log, string strGpus = null)
        {
            m_cuda = cuda;
            m_log = log;

            m_blobTSlice = new Blob<T>(cuda, log);
            m_blobT = new Blob<T>(cuda, log);

            if (!string.IsNullOrEmpty(strGpus))
            {
                string[] rgstr = strGpus.Split(',');
                foreach (string str in rgstr)
                {
                    int nGpuID = -1;
                    if (int.TryParse(str, out nGpuID))
                        m_rgGpuID.Add(nGpuID);
                }
            }

            if (m_rgGpuID.Count == 0)   
                m_rgGpuID.Add(0);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobT != null)
            {
                m_blobT.Dispose();
                m_blobT = null;
            }

            if (m_blobTSlice != null)
            {
                m_blobTSlice.Dispose();
                m_blobTSlice = null;
            }

            m_colCandidates.Dispose();
        }

        /// <summary>
        /// Initialize the change point detector.
        /// </summary>
        /// <param name="nN">Specifies the number of items in the sequence.</param>
        /// <param name="blobTSlice">Specifies a signle slice of the T matrix, and should have shape (nN, 1, 1, 1).</param>
        /// <param name="blobT">Specifies the matrix of T values with shape (nN, nN, 1, 1).</param>
        /// <param name="blobX">Specifies the input data to be analyzed with shape (nN, 1, 1, 1)</param>
        /// <param name="nOutMin">Optionally, specifies the out min which provides a margin around the data (default = 10).</param>
        /// <param name="nEpochs">Optionally, specifies the training epochs spent on each candidate (default = 50).</param>
        /// <param name="nB">Optionally, specifies the bounding value for numerical stability (default = 10).</param>
        /// <returns>If successfully initialized, 'True' is returned.</returns>
        /// <exception cref="Exception">Exceptions are thrown when the blobT or blobTSlice are not properly shaped.</exception>
        public bool Initialize(int nN, Blob<T> blobX, int nOutMin = 10, int nEpochs = 50, int nB = 10)
        {
            if (nN > 64 * 3)
                throw new Exception("The maximum nN is 64 * 3, or 192.  Ideally nN should be a factor of 64.");

            m_blobTSlice.Reshape(nN, 1, 1, 1);
            m_blobT.Reshape(nN, nN, 1, 1);

            m_nN = nN;
            m_nB = nB;

            return m_colCandidates.Initialize(nOutMin, nN, nB, nEpochs, blobX, m_rgGpuID);
        }

        private void onProgress(string strSrc, string strMsg, double dfPct = 0)
        {
            if (m_log != null)
            {
                m_log.Progress = dfPct;
                m_log.WriteLine(strSrc + ": " + strMsg);
            }

            if (OnStatus != null)
                OnStatus(this, new LogArg(strSrc, strMsg, dfPct));
        }

        private void onProgress(string strSrc, string strMsg, int i, int nCount)
        {
            double dfPct = (double)i / (double)nCount;
            onProgress(strSrc, strMsg, dfPct);
        }

        private void onError(string strSrc, Exception excpt)
        {
            if (m_log != null)
                m_log.WriteError(excpt);

            if (OnStatus != null)
                OnStatus(this, new LogArg(strSrc, excpt.Message, 0, true));
        }

        /// <summary>
        /// Calculate the S values from the internal T matrix of candidate change point values.
        /// </summary>
        /// <param name="nTmin">Optionally, specifies the t margin (default = 10).</param>
        /// <param name="bAsync">Optionally, specifies to run in async mode (default = true).</param>
        /// <returns>A blob containing the S values with shape (nN, 1, 1, 1) is returned.  A threshold is used to determine the chainge point.</returns>
        public Blob<T> ComputeSvalues(int nTmin = 10, bool bAsync = true)
        {
            Blob<T> blobS = null;
            long hCpd = 0;

            try
            {
                m_blobT.SetData(0);

                onProgress("CPD", "Starting T-value calculations...");
                
                Stopwatch sw = new Stopwatch();
                List<double> rgTiming = new List<double>();

                sw.Start();

                for (int nT = nTmin; nT < m_nN; nT++)
                {
                    string strAveTime = "";

                    if (m_colCandidates.CalculateTvalues(nT, m_blobTSlice, bAsync))
                    {
                        m_cuda.copy(m_blobTSlice.count(), m_blobTSlice.gpu_data, m_blobT.mutable_gpu_data, 0, nT * m_blobTSlice.count());
                        rgTiming.Add(sw.Elapsed.TotalMilliseconds);
                        double dfAve = rgTiming.Average();
                        strAveTime = " (ave: " + dfAve.ToString("N2") + " ms)";

                        sw.Restart();
                    }

                    onProgress("CPD", "Calculating T-values " + strAveTime + "...", nT, m_nN);
                }

                m_cuda.transposeHW(1, 1, m_nN, m_nN, m_blobT.gpu_data, m_blobT.mutable_gpu_diff);

                double dfTotal = rgTiming.Sum();
                string strTotal = " (total: " + (dfTotal/1000).ToString("N2") + " sec)";
                onProgress("CPD", "All t-values created " + strTotal + ".");

                blobS = new Blob<T>(m_cuda, m_log);
                blobS.ReshapeLike(m_blobTSlice);

                onProgress("CPD", "Calculating S-values...");
                hCpd = m_cuda.CreateCpd();
                m_cuda.SetCpd(hCpd, m_nN, m_nB);
                m_cuda.ComputeCpdSvalues(hCpd, blobS.count(), blobS.mutable_gpu_data, m_blobT.count(), m_blobT.gpu_diff);
                blobS.Name = "NN CPD";

                onProgress("CPD", "Done.");
            }
            catch (Exception excpt)
            {
                onError("CPD", excpt);
                if (blobS != null)
                {
                    blobS.Dispose();
                    blobS = null;
                }

                throw excpt;
            }
            finally
            {

                if (hCpd != 0)
                    m_cuda.FreeCpd(hCpd);
            }

            return blobS;
        }
    }

    /// <summary>
    /// The ChangePointCandidateCollection is used to manage a collection of ChangePointCandidate objects by
    /// calcullating one slice of t values.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of <i>float</i> or <i>double</i>.</typeparam>
    public class ChangePointCandidateCollection<T> : IDisposable
    {
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        List<ChangePointCandidate<T>> m_rgItems = new List<ChangePointCandidate<T>>();
        int m_nOutMin;
        int m_nN;

        /// <summary>
        /// The ChangePointCandidateCollection constructor.
        /// </summary>
        public ChangePointCandidateCollection()
        {
        }

        private bool wait(List<WaitHandle> rgWait, int nWait = 1000)
        {
            if (rgWait.Count <= 64)
            {
                while (!WaitHandle.WaitAll(rgWait.ToArray(), nWait))
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                return true;
            }
            else if (rgWait.Count > 64 && rgWait.Count <= 128)
            {
                while (!WaitHandle.WaitAll(rgWait.Take(64).ToArray(), nWait))
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                while (!WaitHandle.WaitAll(rgWait.Skip(64).ToArray(), nWait))
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                return true;
            }
            else
            {
                while (!WaitHandle.WaitAll(rgWait.Take(64).ToArray(), nWait))
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                while (!WaitHandle.WaitAll(rgWait.Skip(64).Take(64).ToArray(), nWait))
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                while (!WaitHandle.WaitAll(rgWait.Skip(128).ToArray(), nWait))
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                return true;
            }
        }

        /// <summary>
        /// Initialize the ChangePointCandidateCollection.
        /// </summary>
        /// <param name="nOutMin">Specifies the margin for the tau values.</param>
        /// <param name="nN">Specifies the number of items in the sequence.</param>
        /// <param name="nB">Specifies a bounding value used for numerical stability.</param>
        /// <param name="nEpochs">Specifies the number of training epochs used.</param>
        /// <param name="blobX">Specifies the input X data.</param>
        /// <param name="rgGpuId">Specifies the GPU ID's to run on.</param>
        /// <returns>On successful initialization, 'True' is returned.</returns>
        public bool Initialize(int nOutMin, int nN, int nB, int nEpochs, Blob<T> blobX, List<int> rgGpuId)
        {
            List<WaitHandle> rgWait = new List<WaitHandle>();

            m_nOutMin = nOutMin;
            m_nN = nN;

            if (rgGpuId == null)
                rgGpuId = new List<int>() { 0 };
            if (rgGpuId.Count == 0)
                rgGpuId.Add(0);

            for (int nTau = nOutMin; nTau < nN - nOutMin; nTau++)
            {
                int nGpuIdx = (nTau % rgGpuId.Count);
                int nGpuID = rgGpuId[nGpuIdx];

                ChangePointCandidate<T> item = new ChangePointCandidate<T>(nN, nB, nEpochs, nGpuID);
                rgWait.Add(item.Initialize(blobX));
                m_rgItems.Add(item);
            }

            return wait(rgWait);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            List<WaitHandle> rgWait = new List<WaitHandle>();

            m_evtCancel.Set();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgWait.Add(m_rgItems[i].CleanUp());
            }

            wait(rgWait);

            m_rgItems.Clear();
        }

        /// <summary>
        /// Calculate all t-values in the slice.
        /// </summary>
        /// <param name="nT">Specifies the current t for the slice.</param>
        /// <param name="blobTSlice">Specifies the blob where all t values calculated for the slice are placed.</param>
        /// <param name="bAsync">Optionally, specifies to run in async mode (default = true).</param>
        /// <returns>Upon success, 'True' is returned.</returns>
        /// <exception cref="Exception"></exception>
        public bool CalculateTvalues(int nT, Blob<T> blobTSlice, bool bAsync = true)
        {
            List<WaitHandle> rgWait = new List<WaitHandle>();
            int nTau = m_nOutMin;

            if (blobTSlice.count() != m_nN)
                throw new Exception("The blobTSlice must have a count of " + m_nN.ToString() + ".");

            for (int i=0; i < m_rgItems.Count; i++)
            {
                if (nTau >= nT - m_nOutMin)
                    break;

                rgWait.Add(m_rgItems[i].CalculateTvalueAtAsync(nT, nTau, !bAsync));
                nTau++;
            }

            if (rgWait.Count == 0)
                return false;

            if (!wait(rgWait))
                return false;

            nTau = m_nOutMin;
            float[] rgSlice = new float[m_nN];
            Array.Clear(rgSlice, 0, rgSlice.Length);

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                int nIdx = nTau + i;
                rgSlice[nIdx] = (float)m_rgItems[i].CalculatedTValue;
            }

            blobTSlice.mutable_cpu_data = Utility.ConvertVec<T>(rgSlice);

            return true;
        }
    }

    /// <summary>
    /// The ChangePointCandidate is used to calculate a single t-value for a given tau value.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of <i>float</i> or <i>double</i>.</typeparam>
    public class ChangePointCandidate<T> : IDisposable
    {
        ManualResetEvent m_evtReady = new ManualResetEvent(false);
        ManualResetEvent m_evtDone = new ManualResetEvent(false);
        ManualResetEvent m_evtReleased = new ManualResetEvent(false);
        AutoResetEvent m_evtRun = new AutoResetEvent(false);
        CancelEvent m_evtCancel = new CancelEvent();
        Blob<T> m_blobInput;
        int m_nT;
        int m_nTau;
        int m_nN;
        int m_nB;
        int m_nEpochs;
        double m_dfTval;
        int m_nGPUID = 0;
        static object m_objSync = new object();

        /// <summary>
        /// The ChangePointCandidate constructor.
        /// </summary>
        /// <param name="nN">Specifies the number of items in the sequence.</param>
        /// <param name="nB">Specifies a bounding value used for numerical stability.</param>
        /// <param name="nEpochs">Specifies the number of training epochs.</param>
        /// <param name="nGPUID">Specifies the GPU ID to run on.</param>
        public ChangePointCandidate(int nN, int nB, int nEpochs, int nGPUID)
        {
            m_nN = nN;
            m_nB = nB;
            m_nEpochs = nEpochs;
            m_nGPUID = nGPUID;
        }

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            m_evtCancel.Set();
        }

        /// <summary>
        /// Initialize the ChangePointCandidate and start its internal thread.
        /// </summary>
        /// <param name="blobX">Specifies the input data.</param>
        /// <returns>A wait handle is returned for the ready event.</returns>
        public WaitHandle Initialize(Blob<T> blobX)
        {
            m_evtReleased.Reset();
            m_evtDone.Reset();
            m_evtCancel.Reset();
            m_evtReady.Reset();

            m_blobInput = blobX;
            Thread th = new Thread(new ThreadStart(computeCandidateThread));
            th.Start();

            return m_evtReady;
        }

        /// <summary>
        /// Clean up the ChangePointCandidate and terminate its internal thread.
        /// </summary>
        /// <returns>The released wait handle is returned.</returns>
        public WaitHandle CleanUp()
        {
            m_evtCancel.Set();
            return m_evtReleased;
        }

        /// <summary>
        /// Returns the calculated t-value for the candidate.
        /// </summary>
        public double CalculatedTValue
        {
            get { return m_dfTval; }
        }

        /// <summary>
        /// Asynchronously start calculating the t-value for the candidate. 
        /// </summary>
        /// <param name="nT">Specifies the current t-index for the candidate.</param>
        /// <param name="nTau">Specifies the change point candidate location.</param>
        /// <param name="bBlock">Optionally, specifies to block and wait for completion (default = false).</param>
        /// <returns>A WaitHandle is returned for the ready event.</returns>
        public WaitHandle CalculateTvalueAtAsync(int nT, int nTau, bool bBlock=false)
        {
            // Contains data on the external kernel of size nT x 1
            m_nT = nT;
            m_nTau = nTau;
            m_dfTval = 0;
            m_evtCancel.Reset();
            m_evtDone.Reset();
            m_evtRun.Set();

            if (bBlock)
            {
                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.Add(m_evtDone);
                rgWait.AddRange(m_evtCancel.Handles);
                WaitHandle.WaitAny(rgWait.ToArray());
            }

            return m_evtDone;
        }

        private string build_solver()
        {
            SolverParameter p = new SolverParameter();

            p.type = SolverParameter.SolverType.ADAM;
            p.base_lr = 1e-1;
            p.lr_policy = "fixed";

            return p.ToProto("root").ToString();
        }

        private string build_model(int nIn, int nOut)
        {
            NetParameter p = new NetParameter();
            p.name = "cpd";

            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { 1, nIn }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { 1, nIn }));
            data.top.Add("input");
            data.top.Add("target");
            p.layer.Add(data);

            LayerParameter fc1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "fc1");
            fc1.inner_product_param.axis = 1;
            fc1.inner_product_param.num_output = (uint)nIn * 2;
            fc1.inner_product_param.bias_term = true;
            fc1.inner_product_param.weight_filler = new FillerParameter("xavier");
            fc1.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            fc1.bottom.Add("input");
            fc1.top.Add("fc1");
            p.layer.Add(fc1);

            LayerParameter relu1 = new LayerParameter(LayerParameter.LayerType.RELU, "act1");
            relu1.bottom.Add("fc1");
            relu1.top.Add("fc1");
            p.layer.Add(relu1);

            LayerParameter fc2 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "fc2");
            fc1.inner_product_param.axis = 1;
            fc2.inner_product_param.num_output = (uint)nIn * 3;
            fc2.inner_product_param.bias_term = true;
            fc2.inner_product_param.weight_filler = new FillerParameter("xavier");
            fc2.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            fc2.bottom.Add("fc1");
            fc2.top.Add("fc2");
            p.layer.Add(fc2);

            LayerParameter relu2 = new LayerParameter(LayerParameter.LayerType.RELU, "act2");
            relu2.bottom.Add("fc2");
            relu2.top.Add("fc2");
            p.layer.Add(relu2);

            LayerParameter fc3 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "fc3");
            fc1.inner_product_param.axis = 1;
            fc3.inner_product_param.num_output = (uint)nOut;
            fc3.inner_product_param.bias_term = true;
            fc3.inner_product_param.weight_filler = new FillerParameter("xavier");
            fc3.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            fc3.bottom.Add("fc2");
            fc3.top.Add("out");
            p.layer.Add(fc3);

            LayerParameter loss = new LayerParameter(LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "loss");
            loss.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            loss.bce_with_logits_loss_param.reduction = BCEWithLogitsLossParameter.REDUCTION.MEAN;
            loss.bottom.Add("out");
            loss.bottom.Add("target");
            loss.top.Add("loss");
            loss.include.Add(new NetStateRule(Phase.TRAIN));
            p.layer.Add(loss);

            return p.ToProto("root").ToString();
        }

        private void computeCandidateThread()
        {
            List<WaitHandle> rgWait = new List<WaitHandle>();
            Log log = new Log("ChangePointCandidate");
            SettingsCaffe settings = new SettingsCaffe();
            settings.GpuIds = m_nGPUID.ToString();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, log, m_evtCancel);
            Blob<T> blobX;
            Blob<T> blobY;
            Blob<T> blobZ;
            long hCpd = 0;

            string strModel = build_model(1, 1);
            string strSolver = build_solver();

            lock (m_objSync)
            {
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
            }

            Solver<T> solver = mycaffe.GetInternalSolver();
            Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
            CudaDnn<T> cuda = mycaffe.Cuda;
            long hHostBuffer = cuda.AllocHostBuffer(m_nN * m_nN);
            Blob<T> blobWts = new Blob<T>(cuda, log);
            Blob<T> blobLocalX = new Blob<T>(cuda, log);
            blobX = net.FindBlob("input");
            blobY = net.FindBlob("target");
            blobZ = net.FindBlob("out");

            blobLocalX.ReshapeLike(m_blobInput);

            // Each instance of MyCaffe manages a separate kernel with independent memory and handles.
            // When using two instances, we must first copy the data from the external kernel
            // to the kernel managed on this thread by the instance of MyCaffe on this thread.
            cuda.KernelCopy(blobLocalX.count(), m_blobInput.gpu_data, 0, blobLocalX.Cuda.KernelHandle, blobLocalX.mutable_gpu_data, 0, hHostBuffer, cuda.KernelHandle, -1, m_blobInput.Cuda.KernelHandle);

            try
            {
                hCpd = cuda.CreateCpd();
                cuda.SetCpd(hCpd, m_nN, m_nB);

                rgWait.Add(m_evtRun);
                rgWait.AddRange(m_evtCancel.Handles);

                m_evtReady.Set();

                while (!m_evtCancel.WaitOne(0))
                {
                    int nWait = WaitHandle.WaitAny(rgWait.ToArray());
                    if (nWait != 0)
                        return;

                    blobX.Reshape(m_nT, 1, 1, 1);
                    blobY.Reshape(m_nT, 1, 1, 1);
                    blobWts.Reshape(m_nT, 1, 1, 1);

                    cuda.copy(blobX.count(), blobLocalX.gpu_data, blobX.mutable_gpu_data);

                    // Set the weights to the current tau value.
                    T fVal = (T)Convert.ChangeType((m_nT - m_nTau), typeof(T));
                    cuda.set(m_nTau, blobWts.mutable_gpu_data, fVal, -1, 0);
                    fVal = (T)Convert.ChangeType(m_nTau, typeof(T));
                    cuda.set(m_nT - m_nTau, blobWts.mutable_gpu_data, fVal, -1, m_nTau);

                    // Set the current target value to virtual labels.
                    cuda.set(m_nTau, blobY.mutable_gpu_data, (T)Convert.ChangeType(1, typeof(T)), -1, 0);
                    cuda.set(m_nT - m_nTau, blobY.mutable_gpu_data, (T)Convert.ChangeType(0, typeof(T)), -1, m_nTau);

                    // Reset the weights to random values.
                    net.Reshape();
                    net.ReInitializeParameters(WEIGHT_TARGET.BOTH);

                    // Train the network.
                    for (int i=0; i<m_nEpochs; i++)
                    {
                        double dfLoss;
                        net.Forward(out dfLoss);
                        net.Backward();
                        solver.Step(1);
                        solver.ApplyUpdate(i);
                    }

                    net.ForwardFromTo(0, net.layers.Count - 2);
                    m_dfTval = cuda.ComputeCpdTvalueAt(hCpd, m_nT, m_nTau, blobZ.count(), blobZ.gpu_data);
                    m_evtDone.Set();
                }
            }
            catch (Exception excpt)
            {
                log.FAIL("ComputeCandidate failed with " + excpt.Message);
            }
            finally
            {
                if (hCpd != 0)
                    cuda.FreeCpd(hCpd);

                if (hHostBuffer != 0)
                    cuda.FreeHostBuffer(hHostBuffer);

                blobLocalX.Dispose();
                blobWts.Dispose();
                mycaffe.Dispose();
                m_evtDone.Set();
                m_evtReleased.Set();
            }
        }
    }
}
