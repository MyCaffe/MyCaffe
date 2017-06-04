using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.solvers;

namespace MyCaffe.common
{
    /// <summary>
    /// The Params contains the base parameters used in multi-GPU training.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Params<T>
    {
        /// <summary>
        /// size of the buffers (in items).
        /// </summary>
        protected long m_lCount;  

        /// <summary>
        /// size of the padding added to the memory buffers.
        /// </summary>
        protected long m_lExtra;  

        /// <summary>
        /// Handle to GPU memory containing the Net parameters.
        /// </summary>
        protected long m_hData;
        /// <summary>
        /// Handle to GPU memory containing the Net gradient.
        /// </summary>
        protected long m_hDiff;
        /// <summary>
        /// The Device ID.
        /// </summary>
        protected int m_nDeviceID;

        /// <summary>
        /// The Param constructor.
        /// </summary>
        /// <param name="root_solver">Specifies the root Solver.</param>
        public Params(Solver<T> root_solver)
        {
            m_lCount = total_size(root_solver.net.learnable_parameters);
            m_lExtra = 1000;
            m_hData = 0;
            m_hDiff = 0;

            m_lCount += m_lExtra;
        }

        /// <summary>
        /// Returns the size of the buffers (in items).
        /// </summary>
        public long count
        {
            get { return m_lCount; }
        }

        /// <summary>
        /// Returns the handle to the GPU memory containing the Net parameters. 
        /// </summary>
        public long data
        {
            get { return m_hData; }
        }

        /// <summary>
        /// Returns the handle to the GPU memory containing the Net gradients. 
        /// </summary>
        public long diff
        {
            get { return m_hDiff; }
        }

        private long total_size(BlobCollection<T> rgParam)
        {
            long nSize = 0;

            for (int i = 0; i < rgParam.Count; i++)
            {
                nSize += (long)rgParam[i].count();               
            }

            // Size should have at least one byte, otherwise malloc fails
            //  if net has no learnable parameters.
            if (nSize == 0)
                nSize++;

            return nSize;
        }
    }

    /// <summary>
    /// The GPUParams contains the connection to the low-level Cuda, and the stream associated with this instance.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GPUParams<T> : Params<T>, IDisposable
    {
        /// <summary>
        /// The instance of CudaDnn that provides the connection to Cuda.
        /// </summary>
        protected CudaDnn<T> m_cuda;
        /// <summary>
        /// The Log used for output.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// The handle to the Cuda stream used for synchronization.
        /// </summary>
        protected long m_hStream;

        /// <summary>
        /// Defines the memory operation to perform. 
        /// </summary>
        public enum Op
        {
            /// <summary>
            /// Copy over the buffer.
            /// </summary>
            copy,
            /// <summary>
            /// Replace the GPU portion of the data buffer.
            /// </summary>
            replace_gpu,
            /// <summary>
            /// Replace the GPU protion of the diff buffer.
            /// </summary>
            replace_gpu_diff
        }

        /// <summary>
        /// The GPUParams constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="root_solver">Specifies the root Solver.</param>
        /// <param name="nDeviceID">Specifies the device ID to use for this instance.</param>
        public GPUParams(CudaDnn<T> cuda, Log log, Solver<T> root_solver, int nDeviceID)
            : base(root_solver)
        {
            m_cuda = cuda;
            m_log = log;

            m_nDeviceID = m_cuda.GetDeviceID();

            if (nDeviceID != m_nDeviceID)
                m_cuda.SetDeviceID(nDeviceID);

            // Allocate device buffers
            m_hData = m_cuda.AllocMemory(m_lCount);

            // Copy blob values
            BlobCollection<T> net = root_solver.net.learnable_parameters;
            apply_buffers(net, m_hData, m_lCount, Op.copy);

            m_hDiff = m_cuda.AllocMemory(m_lCount);
            m_cuda.set((int)m_lCount, m_hDiff, 0);

            m_hStream = m_cuda.CreateStream();

            if (m_nDeviceID != nDeviceID)
                m_cuda.SetDeviceID(m_nDeviceID);
        }

        /// <summary>
        /// Release all GPU and Host resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_hData != 0)
            {
                m_cuda.FreeMemory(m_hData);
                m_hData = 0;
            }

            if (m_hDiff != 0)
            {
                m_cuda.FreeMemory(m_hDiff);
                m_hDiff = 0;
            }

            if (m_hStream != 0)
            {
                m_cuda.FreeStream(m_hStream);
                m_hStream = 0;
            }
        }

        /// <summary>
        /// Synchronize with the Cuda stream.
        /// </summary>
        public void SynchronizeStream()
        {
            m_cuda.SynchronizeStream(m_hStream);
        }

        /// <summary>
        /// Configure the GPU Params by copying the Solver training Net parameters into the data and diff buffers.
        /// </summary>
        /// <param name="solver"></param>
        public void Configure(Solver<T> solver)
        {
            BlobCollection<T> net = solver.net.learnable_parameters;
            apply_buffers(net, m_hData, m_lCount, Op.replace_gpu);
            apply_buffers(net, m_hDiff, m_lCount, Op.replace_gpu_diff);
        }

        /// <summary>
        /// Transfer between the data/diff buffers and a collection of Blobs (e.g. the learnable parameters).
        /// </summary>
        /// <param name="rgBlobs">Specifies the collection of Blobs to transfer data with.</param>
        /// <param name="hBuffer">Specifies a handle to the memory on the GPU to transfer with the Blob collection.</param>
        /// <param name="lTotalSize">Specifies the number of items to transfer.</param>
        /// <param name="op">Specifies the type of transfer to perform.</param>
        public void apply_buffers(BlobCollection<T> rgBlobs, long hBuffer, long lTotalSize, Op op)
        {
            long lOffset = 0;

            for (int i = 0; i < rgBlobs.Count; i++)
            {
                int nCount = rgBlobs[i].count();

                switch (op)
                {
                    // Init buffer to current values of blobs
                    case Op.copy:
                        m_cuda.copy(nCount, rgBlobs[i].data.gpu_data, hBuffer, 0, (int)lOffset);
                        break;

                    case Op.replace_gpu:
                        rgBlobs[i].data.set_gpu_data(hBuffer, nCount, lOffset);
                        break;

                    case Op.replace_gpu_diff:
                        rgBlobs[i].diff.set_gpu_data(hBuffer, nCount, lOffset);
                        break;
                }

                lOffset += nCount;
            }

            // total_size is at least one byte
            // We allocate extra items past the items used as a pad.
            m_log.CHECK_EQ(lTotalSize - m_lExtra, (lOffset == 0) ? 1 : lOffset, "The total memory doesn't match.");
        }
    }

    /// <summary>
    /// The NCCL class manages the multi-GPU operations using the low-level NCCL functionality provided by the low-level Cuda Dnn DLL.
    /// </summary>
    /// <remarks>
    /// [NVIDIA's NCCL 'Nickel'](https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/) is an NVIDIA library designed to 
    /// optimize communication between multiple GPUs.
    /// 
    /// When using multi-GPU training, it is highly recommended to only train on TCC enabled drivers, otherwise driver timeouts may occur on large models.
    /// @see [NVIDIA Tesla Compute Cluster (TCC) Help](http://docs.nvidia.com/gameworks/content/developertools/desktop/tesla_compute_cluster.htm)
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class NCCL<T> : GPUParams<T>, IDisposable 
    {
        long m_hNccl;
        Solver<T> m_solver;
        ManualResetEvent m_evtGradientsReady = new ManualResetEvent(false);
        List<ManualResetEvent> m_rgGradientReady = new List<ManualResetEvent>();

        /// <summary>
        /// The NCCL constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="root_solver">Specifies the root Solver.</param>
        /// <param name="nDeviceID">Specifies the device ID to use for this instance.</param>
        /// <param name="hNccl">Specifies the handle to NCCL created using CudaDnn::CreateNCCL, or 0 for the root solver as this is set-up in NCCL::Run.</param>
        /// <param name="rgGradientReadyEvents">Specifies a list of events used to synchronize with the other Solvers.</param>
        public NCCL(CudaDnn<T> cuda, Log log, Solver<T> root_solver, int nDeviceID, long hNccl, List<ManualResetEvent> rgGradientReadyEvents)
            : base(cuda, log, root_solver, nDeviceID)
        {
            m_rgGradientReady = rgGradientReadyEvents;
            if (rgGradientReadyEvents != null && rgGradientReadyEvents.Count > 0)
                m_evtGradientsReady = rgGradientReadyEvents[root_solver.solver_rank];

            m_solver = root_solver;
            m_hNccl = hNccl;
            Configure(root_solver);

            root_solver.OnGradientsReady += Solver_OnGradientsReady;
        }

        /// <summary>
        /// Release all GPU and Host resources used.
        /// </summary>
        public new void Dispose()
        {
            base.Dispose();

            if (m_hNccl != 0)
            {
                m_cuda.FreeNCCL(m_hNccl);
                m_hNccl = 0;
            }
        }

        /// <summary>
        /// Broadcast the data to all other solvers participating in the multi-GPU session.
        /// </summary>
        public void Broadcast()
        {
            m_cuda.NcclBroadcast(m_hNccl, m_hStream, m_hData, (int)m_lCount);
            m_cuda.SynchronizeStream(m_hStream);
        }

        private void Solver_OnGradientsReady(object sender, GradientsReadyArgs e)
        {
            try
            {
                m_cuda.SynchronizeStream();
                m_evtGradientsReady.Set();

                while (!WaitHandle.WaitAll(m_rgGradientReady.ToArray(), 250))
                {
                    if (m_solver.CancelEvent.WaitOne(0))
                        return;
                }

                double dfScale = 1.0 / m_solver.solver_count;
                m_cuda.NcclAllReduce(m_hNccl, m_hStream, m_hDiff, (int)m_lCount, NCCL_REDUCTION_OP.SUM, dfScale);
                m_cuda.SynchronizeStream(m_hStream);
            }
            finally
            {
                m_evtGradientsReady.Reset();
            }
        }

        /// <summary>
        /// Run the root Solver and coordinate with all other Solver's participating in the multi-GPU training.
        /// </summary>
        /// <remarks><b>IMPORTANT</b>: When performing multi-GPU training only GPU's that DO NOT have a monitor connected can be used.  Using the GPU
        /// with the monitor connected will cause an error.  However, you can use GPU's that are configured in either TCC mode or WDM mode, BUT,
        /// all GPU's participating must be configured to use the same mode, otherwise you may experience upredictable behavior, the NVIDIA driver
        /// may crash, or you may experience the infamouse BSOD ("Blue Screen of Death") - so as the saying goes, "Beware to all who enter here..."</remarks>
        /// <param name="rgGpus">Specifies all GPU ID's to use.</param>
        /// <param name="nIterationOverride">Optionally, specifies a training iteration override to use.</param>
        public void Run(List<int> rgGpus, int nIterationOverride = -1)
        {
            List<long> rghNccl = new List<long>();
            Guid guid = Guid.NewGuid();

            m_rgGradientReady = new List<ManualResetEvent>();

            for (int i = 0; i < rgGpus.Count; i++)
            {
                long hNccl = m_cuda.CreateNCCL(rgGpus[i], m_solver.solver_count, i, guid);
                rghNccl.Add(hNccl);
                m_rgGradientReady.Add(new ManualResetEvent(false));
            }

            m_cuda.NcclInitializeSingleProcess(rghNccl.ToArray());
            m_hNccl = rghNccl[0];
            m_evtGradientsReady = m_rgGradientReady[0];

            List<WaitHandle> rgWaitAllInit = new List<WaitHandle>();
            List<Worker<T>> rgWorkers = new List<common.Worker<T>>();
            ManualResetEvent evtAllCreated = new ManualResetEvent(false);

            for (int i = 1; i < rghNccl.Count; i++)
            {
                Worker<T> worker = new Worker<T>();

                SolverInfo<T> info = new common.SolverInfo<T>(m_solver, m_cuda.KernelHandle, rghNccl[i], i, nIterationOverride, m_cuda.Path, m_rgGradientReady, evtAllCreated);
                worker.StartInternalThread(null, null, rgGpus[i], info);
                int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_solver.CancelEvent.Handle, info.ErrorEvent, info.StartedEvent });
                if (nWait == 0)
                    return;

                if (nWait == 1)
                {
                    if (info.Error != null)
                        throw info.Error;
                    else
                        throw new Exception("Error starting the solver.");
                }

                rgWaitAllInit.Add(info.InitializedEvent);
                rgWorkers.Add(worker);
            }

            // Wait for all worksers to initialize
            while (!WaitHandle.WaitAll(rgWaitAllInit.ToArray(), 250))
            {
                if (m_solver.CancelEvent.WaitOne(0))
                    return;
            }

            m_cuda.SynchronizeDevice();
            evtAllCreated.Set();

            // Run first solver on current thread.
            Broadcast();

            m_solver.Solve(nIterationOverride);

            // Wait for shutdown
            for (int i = 0; i < rgWorkers.Count; i++)
            {
                rgWorkers[i].StopInternalThread();
            }
        }
    }

    /// <summary>
    /// The Worker manages each 'non' root sover running, where each Worker operates on a different GPU.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Worker<T> : InternalThread<T>
    {
        CudaDnn<T> m_cuda;

        /// <summary>
        /// The Worker constructor.
        /// </summary>
        public Worker()
        {
            this.DoWork += Worker_DoWork;
        }

        private void Worker_DoWork(object sender, ActionStateArgs<T> e)
        {
            SolverInfo<T> info = e.Arg as SolverInfo<T>;
            NCCL<T> nccl = null;

            m_cuda = new common.CudaDnn<T>(e.DeviceID, DEVINIT.CUBLAS | DEVINIT.CURAND, null, info.CudaPath);

            try
            {
                Solver<T> rank0 = info.Rank0;
                Log log = new Log("Worker solver for DeviceID = " + e.DeviceID.ToString());

                //-----------------------------------------
                //  Transfer the NCCL handle from the 
                //  main kernel that created it to the
                //  one used by the CudaDnn on this thread.
                //
                //  After the copy, this thread will 'own'
                //  the nccl and be responsible for its 
                //  destruction.
                //-----------------------------------------
                long hNccl = m_cuda.KernelCopyNccl(info.KernelHandle, info.NcclHandle);

                // Create solver and install callbacks
                SolverParameter param = rank0.parameter.Clone();
                param.device_id = e.DeviceID;
                param.type = rank0.parameter.type;
                Solver<T> solver = Solver<T>.Create(m_cuda, log, param, rank0.CancelEvent, null, null, rank0.Database, null, rank0.solver_count, info.SolverRank);
                info.StartedEvent.Set();
                log.CHECK_EQ((int)solver.type, (int)rank0.type, "The solver types should be the same.");

                //-----------------------------------------
                //  Turn off logging for all other 
                //  operations on the worker thread.
                //-----------------------------------------
                log.Enable = false;

                nccl = new NCCL<T>(m_cuda, log, solver, e.DeviceID, hNccl, info.GradientReadyEvents);

                info.InitializedEvent.Set();
                m_cuda.SynchronizeDevice();

                int nWait = WaitHandle.WaitAny(new WaitHandle[] { rank0.CancelEvent.Handle, info.AllCreatedEvent });
                if (nWait == 0)
                    return;

                nccl.Broadcast();

                int nIterations = param.max_iter - solver.iter;
                if (info.IterationOverride > 0)
                    nIterations = info.IterationOverride;

                solver.Step(nIterations);
            }
            catch (Exception excpt)
            {
                info.Error = excpt;
                info.ErrorEvent.Set();
            }
            finally
            {
                if (nccl != null)
                    nccl.Dispose();

                m_cuda.Dispose();
                m_cuda = null;
            }
        }
    }

    /// <summary>
    /// The SolverInfo defines the user supplied arguments passed to each Worker.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SolverInfo<T>
    {
        string m_strCudaPath;
        Solver<T> m_rank0;
        long m_hSrcKernel;
        long m_hSrcNccl;
        int m_nSolverRank;
        int m_nIterationOverride;
        ManualResetEvent m_evtInitialized = new ManualResetEvent(false);
        ManualResetEvent m_evtStarted = new ManualResetEvent(false);
        ManualResetEvent m_evtAllCreated = new ManualResetEvent(false);
        AutoResetEvent m_evtError = new AutoResetEvent(false);
        List<ManualResetEvent> m_rgGradientReadyEvents = null;
        Exception m_error = null;

        /// <summary>
        /// The SolverInfo constructor.
        /// </summary>
        /// <param name="rank0">Specifies rank Solver that will run in the Worker.</param>
        /// <param name="hSrcKernel">Specifies a handle to the kernel where the NCCL for this Solver was created (typically this is the kernel that also created the root Solver).</param>
        /// <param name="hSrcNccl">Specifies the handle to the NCCL instance for this Solver (typically this is created on the kernel that also created the root Solver, and must be transferred to the kernel of the CudaDnn instance running in the Worker).</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver.</param>
        /// <param name="nIterationOverride">Specifies the training iteration override to use.</param>
        /// <param name="strCudaPath">Specifies the file path to the low-level CudaDnnDll.DLL file to use.  Note, when <i>null</i> or emtpy, the path of the executing Assembly is used.</param>
        /// <param name="rgGradientReadyEvents">Specifies the list of events used to coordinate with other Solvers.</param>
        /// <param name="evtAllCreated">Specifies an event used to coordinate the creation of all participating Workers.</param>
        public SolverInfo(Solver<T> rank0, long hSrcKernel, long hSrcNccl, int nSolverRank, int nIterationOverride, string strCudaPath, List<ManualResetEvent> rgGradientReadyEvents, ManualResetEvent evtAllCreated)
        {
            m_strCudaPath = strCudaPath;
            m_rank0 = rank0;
            m_hSrcKernel = hSrcKernel;
            m_hSrcNccl = hSrcNccl;
            m_nSolverRank = nSolverRank;
            m_nIterationOverride = nIterationOverride;
            m_rgGradientReadyEvents = rgGradientReadyEvents;
            m_evtAllCreated = evtAllCreated;
        }

        /// <summary>
        /// Returns rank Solver that will run in the Worker.
        /// </summary>
        public Solver<T> Rank0
        {
            get { return m_rank0; }
        }

        /// <summary>
        /// Returns the file path to the low-level CudaDnnDll.DLL file to use.  Note, when <i>null</i> or emtpy, the path of the executing Assembly is used.
        /// </summary>
        public string CudaPath
        {
            get { return m_strCudaPath; }
        }

        /// <summary>
        /// Returns the training iteration override to use.
        /// </summary>
        public int IterationOverride
        {
            get { return m_nIterationOverride; }
        }

        /// <summary>
        /// Returns a handle to the kernel where the NCCL for this Solver was created (typically this is the kernel that also created the root Solver)
        /// </summary>
        public long KernelHandle
        {
            get { return m_hSrcKernel; }
        }

        /// <summary>
        /// Returns the handle to the NCCL instance for this Solver (typically this is created on the kernel that also created the root Solver, and must be transferred to the kernel of the CudaDnn instance running in the Worker).
        /// </summary>
        public long NcclHandle
        {
            get { return m_hSrcNccl; }
        }

        /// <summary>
        /// Returns the rank of this Solver.
        /// </summary>
        public int SolverRank
        {
            get { return m_nSolverRank; }
        }

        /// <summary>
        /// Returns the event that is set after the Worker has completed initializing.
        /// </summary>
        public ManualResetEvent InitializedEvent
        {
            get { return m_evtInitialized; }
        }

        /// <summary>
        /// Returns the event that is set after the Worker has started running.
        /// </summary>
        public ManualResetEvent StartedEvent
        {
            get { return m_evtStarted; }
        }

        /// <summary>
        /// Returns the event that is set after all Workers have been created.
        /// </summary>
        public ManualResetEvent AllCreatedEvent
        {
            get { return m_evtAllCreated; }
        }

        /// <summary>
        /// Returns the event that is set after the gradients of the Solver in this Worker are ready.
        /// </summary>
        public List<ManualResetEvent> GradientReadyEvents
        {
            get { return m_rgGradientReadyEvents; }
        }

        /// <summary>
        /// Returns the error (if any) that occured when running the solver thread.
        /// </summary>
        public Exception Error
        {
            get { return m_error; }
            set { m_error = value; }
        }

        /// <summary>
        /// Returns the event that is set when an error occurs.
        /// </summary>
        public AutoResetEvent ErrorEvent
        {
            get { return m_evtError; }
        }
    }
}
