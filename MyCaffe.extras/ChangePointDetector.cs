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
    public class ChangePointDetector<T> : IDisposable
    {
        ChangePointCandidateCollection<T> m_colCandidates = new ChangePointCandidateCollection<T>();
        Blob<T> m_blobTSlice;
        Blob<T> m_blobT;
        int m_nN;
        int m_nB;

        /// <summary>
        /// The ChangePointDetector constructor.
        /// </summary>
        public ChangePointDetector()
        {
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
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
        public bool Initialize(int nN, Blob<T> blobTSlice, Blob<T> blobT, Blob<T> blobX, int nOutMin = 10, int nEpochs = 50, int nB = 10)
        {
            if (blobTSlice.count() != nN)
                throw new Exception("The blobTSlice must have a count of " + nN.ToString() + ".");

            if (blobT.num != nN && blobT.channels != nN)
                throw new Exception("The blobT must have a num and channels of " + nN.ToString() + ".");

            m_nN = nN;
            m_nB = nB;
            m_blobTSlice = blobTSlice;
            m_blobT = blobT;

            return m_colCandidates.Initialize(nOutMin, nN, nB, nEpochs, blobX);
        }

        /// <summary>
        /// Calculate the S values from the internal T matrix of candidate change point values.
        /// </summary>
        /// <param name="nTmin">Optionally, specifies the t margin (default = 10).</param>
        /// <returns>A blob containing the S values with shape (nN, 1, 1, 1) is returned.  A threshold is used to determine the chainge point.</returns>
        public Blob<T> ComputeSvalues(int nTmin = 10)
        {
            Blob<T> blobS = null;
            CudaDnn<T> cuda = m_blobT.Cuda;
            long hCpd = 0;

            try
            {
                m_blobT.SetData(0);

                blobS = new Blob<T>(m_blobTSlice.Cuda, m_blobTSlice.Log);
                blobS.ReshapeLike(m_blobTSlice);

                for (int nT = nTmin; nT < m_nN; nT++)
                {
                    m_colCandidates.CalculateTvalues(nT, m_blobTSlice);
                    cuda.copy(m_blobTSlice.count(), m_blobTSlice.gpu_data, m_blobT.mutable_gpu_data, 0, nT * m_blobTSlice.count());
                }

                hCpd = cuda.CreateCpd();
                cuda.SetCpd(hCpd, m_nN, m_nB);
                cuda.ComputeCpdSvalues(hCpd, blobS.count(), blobS.mutable_gpu_data);
            }
            catch (Exception excpt)
            {
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
                    cuda.FreeCpd(hCpd);
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

        /// <summary>
        /// Initialize the ChangePointCandidateCollection.
        /// </summary>
        /// <param name="nOutMin">Specifies the margin for the tau values.</param>
        /// <param name="nN">Specifies the number of items in the sequence.</param>
        /// <param name="nB">Specifies a bounding value used for numerical stability.</param>
        /// <param name="nEpochs">Specifies the number of training epochs used.</param>
        /// <param name="blobX">Specifies the input X data.</param>
        /// <returns>On successful initialization, 'True' is returned.</returns>
        public bool Initialize(int nOutMin, int nN, int nB, int nEpochs, Blob<T> blobX)
        {
            List<WaitHandle> rgWait = new List<WaitHandle>();

            m_nOutMin = nOutMin;
            m_nN = nN;

            for (int nTau = nOutMin; nTau < nN - nOutMin; nTau++)
            {
                ChangePointCandidate<T> item = new ChangePointCandidate<T>(nN, nB, nEpochs);
                rgWait.Add(item.Initialize(blobX));
                m_rgItems.Add(item);
            }

            while (!WaitHandle.WaitAll(rgWait.ToArray(), 1000))
            {
                if (m_evtCancel.WaitOne(0))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            m_evtCancel.Set();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                m_rgItems[i].Dispose();
            }

            m_rgItems.Clear();
        }

        /// <summary>
        /// Calculate all t-values in the slice.
        /// </summary>
        /// <param name="nT">Specifies the current t for the slice.</param>
        /// <param name="blobTSlice">Specifies the blob where all t values calculated for the slice are placed.</param>
        /// <returns>Upon success, 'True' is returned.</returns>
        /// <exception cref="Exception"></exception>
        public bool CalculateTvalues(int nT, Blob<T> blobTSlice)
        {
            List<WaitHandle> rgWait = new List<WaitHandle>();
            int nTau = m_nOutMin;

            if (blobTSlice.count() != m_nN)
                throw new Exception("The blobTSlice must have a count of " + m_nN.ToString() + ".");

            for (int i=0; i < m_rgItems.Count; i++)
            {
                rgWait.Add(m_rgItems[i].CalculateTvalueAtAsync(nT, nTau));
                nTau++;
            }

            while (!WaitHandle.WaitAll(rgWait.ToArray(), 1000))
            {
                if (m_evtCancel.WaitOne(0))
                    return false;
            }

            blobTSlice.SetData(0);

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                int nIdx = m_nOutMin + i;
                T fVal = (T)Convert.ChangeType(m_rgItems[i].CalculatedTValue, typeof(T));
                blobTSlice.SetData(fVal, nIdx);
            }

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
        AutoResetEvent m_evtRun = new AutoResetEvent(false);
        CancelEvent m_evtCancel = new CancelEvent();
        Blob<T> m_blobInput;
        int m_nT;
        int m_nTau;
        int m_nN;
        int m_nB;
        int m_nEpochs;
        double m_dfTval;

        /// <summary>
        /// The ChangePointCandidate constructor.
        /// </summary>
        /// <param name="nN">Specifies the number of items in the sequence.</param>
        /// <param name="nB">Specifies a bounding value used for numerical stability.</param>
        /// <param name="nEpochs">Specifies the number of training epochs.</param>
        public ChangePointCandidate(int nN, int nB, int nEpochs)
        {
            m_nN = nN;
            m_nB = nB;
            m_nEpochs = nEpochs;
        }

        /// <summary>
        /// Initialize the ChangePointCandidate and start its internal thread.
        /// </summary>
        /// <param name="blobX">Specifies the input data.</param>
        /// <returns>A wait handle is returned for the ready event.</returns>
        public WaitHandle Initialize(Blob<T> blobX)
        {
            m_blobInput = blobX;
            Thread th = new Thread(new ThreadStart(computeCandidateThread));
            th.Start();
            return m_evtReady;
        }

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            m_evtCancel.Set();
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
        /// <returns>A WaitHandle is returned for the ready event.</returns>
        public WaitHandle CalculateTvalueAtAsync(int nT, int nTau)
        {
            // Contains data on the external kernel of size nT x 1
            m_nT = nT;
            m_nTau = nTau;
            m_dfTval = 0;
            m_evtCancel.Reset();
            m_evtDone.Reset();
            m_evtRun.Set();
            return m_evtReady;
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
            settings.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, log, m_evtCancel);
            Blob<T> blobX;
            Blob<T> blobY;
            Blob<T> blobZ;
            long hCpd = 0;

            rgWait.Add(m_evtRun);
            rgWait.AddRange(m_evtCancel.Handles);

            string strModel = build_model(1, 1);
            string strSolver = build_solver();
            mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);

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

            // Copy data from the external kernel to the kernel managed on this thread.
            cuda.KernelCopy(blobLocalX.count(), m_blobInput.gpu_data, 0, blobLocalX.Cuda.KernelHandle, blobLocalX.mutable_gpu_data, 0, hHostBuffer, cuda.KernelHandle, m_blobInput.Cuda.KernelHandle);

            m_evtReady.Set();

            try
            {
                hCpd = cuda.CreateCpd();
                cuda.SetCpd(hCpd, m_nN, m_nB);

                while (!m_evtCancel.WaitOne(0))
                {
                    int nWait = WaitHandle.WaitAny(rgWait.ToArray(), 1000);
                    if (nWait != 0)
                        return;

                    blobX.Reshape(m_nT, 1, 1, 1);
                    blobY.Reshape(m_nT, 1, 1, 1);

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
                    net.ReInitializeParameters(WEIGHT_TARGET.BOTH);

                    // Train the network.
                    for (int i=0; i<m_nB; i++)
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
            }
        }
    }
}
