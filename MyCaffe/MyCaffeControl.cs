﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Drawing;
using System.Threading.Tasks;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using MyCaffe.solvers;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.data;
using MyCaffe.layers;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Net;
using static MyCaffe.param.beta.DecodeParameter;

/// <summary>
/// The MyCaffe namespace contains the main body of MyCaffe code that closesly tracks the C++ Caffe open-source project.  
/// </summary>
namespace MyCaffe
{
    /// <summary>
    /// The MyCaffeControl is the main object used to manage all training, testing and running of the MyCaffe system.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public partial class MyCaffeControl<T> : Component, IXMyCaffeState<T>, IXMyCaffe<T>, IXMyCaffeNoDb<T>, IXMyCaffeExtension<T>, IDisposable
    {
        /// <summary>
        /// The settings used to configure the control.
        /// </summary>
        protected SettingsCaffe m_settings;
        /// <summary>
        /// The log used for output.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// The image database.
        /// </summary>
        protected IXDatabaseBase m_db = null;
        /// <summary>
        /// Whether or not the control owns the image database.
        /// </summary>
        protected bool m_bDbOwner = true;
        /// <summary>
        /// The CancelEvent used to cancel training and testing operations.
        /// </summary>
        protected CancelEvent m_evtCancel;
        /// <summary>
        /// An auto-reset event used to force a snapshot.
        /// </summary>
        protected AutoResetEvent m_evtForceSnapshot;
        /// <summary>
        /// An auto-reset event used to force a test cycle.
        /// </summary>
        protected AutoResetEvent m_evtForceTest;
        /// <summary>
        /// An auto-reset event used to pause training.
        /// </summary>
        protected ManualResetEvent m_evtPause;
        /// <summary>
        /// The data transformer used to transform data.
        /// </summary>
        protected DataTransformer<T> m_dataTransformer = null;
        /// <summary>
        /// The active project (if any).
        /// </summary>
        protected ProjectEx m_project = null;
        /// <summary>
        /// The dataset descriptor of the dataset used in the image database.
        /// </summary>
        protected DatasetDescriptor m_dataSet = null;
        /// <summary>
        /// The low-level path of the underlying CudaDnn DLL.
        /// </summary>
        protected string m_strCudaPath = null;
        /// <summary>
        /// A list of the Device ID's used for training.
        /// </summary>
        protected List<int> m_rgGpu;
        CudaDnn<T> m_cuda;
        Solver<T> m_solver;
        Net<T> m_net;
        bool m_bOwnRunNet = true;
        MemoryStream m_msWeights = new MemoryStream();
        Guid m_guidUser;
        PersistCaffe<T> m_persist;
        BlobShape m_inputShape = null;
        Phase m_lastPhaseRun = Phase.NONE;
        long m_hCopyBuffer = 0;
        string m_strStage = null;
        bool m_bLoadLite = false;
        string m_strSolver = null;  // Used with LoadLite.
        string m_strModel = null;   // Used with LoadLite.
        ManualResetEvent m_evtSyncUnload = new ManualResetEvent(false);
        ManualResetEvent m_evtSyncMain = new ManualResetEvent(false);
        ConnectInfo m_dsCi = null;
        bool m_bEnableVerboseStatus = false;
        T[] m_rgRunData = null;
        BlobShape m_loadToRunShape = null;

        /// <summary>
        /// The OnSnapshot event fires each time a snap-shot is taken.
        /// </summary>
        public event EventHandler<SnapshotArgs> OnSnapshot;
        /// <summary>
        /// The OnTrainingIteration event fires at the end of each training iteration.
        /// </summary>
        public event EventHandler<TrainingIterationArgs<T>> OnTrainingIteration;
        /// <summary>
        /// The OnTestingIteration event fires at the end of each testing iteration.
        /// </summary>
        public event EventHandler<TestingIterationArgs<T>> OnTestingIteration;


        /// <summary>
        /// The MyCaffeControl constructor.
        /// </summary>
        /// <param name="settings">Specifies the setting used to configure the MyCaffeControl.</param>
        /// <param name="log">Specifies the Log for all output.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to abort training and testing operations.</param>
        /// <param name="evtSnapshot">Optionally, specifies an auto reset event used to force a snap-shot.</param>
        /// <param name="evtForceTest">Optionally, specifies an auto reset event used to force a test cycle.</param>
        /// <param name="evtPause">Optionally, specifies an auto reset event used to pause training.</param>
        /// <param name="rgGpuId">Optionally, specfies a set of GPU ID's that override those specified in the SettingsCaffe object.</param>
        /// <param name="strCudaPath">Optionally, specifies the path to the low-lever CudaDnnDll.DLL file.  Note, when not set, the system looks in the same directory of the executing assembly for the low-level DLL.</param>
        /// <param name="bCreateCudaDnn">Optionally, specififies create the connection to CUDA (default = false, causing the creation to occur during Load).</param>
        /// <param name="ci">Optionally, specifies the connection information used to connect to the dataset.</param>
        public MyCaffeControl(SettingsCaffe settings, Log log, CancelEvent evtCancel, AutoResetEvent evtSnapshot = null, AutoResetEvent evtForceTest = null, ManualResetEvent evtPause = null, List<int> rgGpuId = null, string strCudaPath = "", bool bCreateCudaDnn = false, ConnectInfo ci = null)
        {
            m_dsCi = ci;
            m_guidUser = Guid.NewGuid();

            InitializeComponent();

            if (evtCancel == null)
                throw new ArgumentNullException("The cancel event must be specified!");

            if (evtSnapshot == null)
                evtSnapshot = new AutoResetEvent(false);

            if (evtForceTest == null)
                evtForceTest = new AutoResetEvent(false);

            if (evtPause == null)
                evtPause = new ManualResetEvent(false);

            m_log = log;
            m_settings = settings;
            m_evtCancel = evtCancel;
            m_evtForceSnapshot = evtSnapshot;
            m_evtForceTest = evtForceTest;
            m_evtPause = evtPause;

            if (rgGpuId == null)
            {
                m_rgGpu = new List<int>();
                string[] rgstrGpuId = settings.GpuIds.Split(',');

                foreach (string str in rgstrGpuId)
                {
                    string strGpuId = str.Trim(' ', '\t', '\n', '\r');
                    m_rgGpu.Add(int.Parse(strGpuId));
                }
            }
            else
            {
                m_rgGpu = Utility.Clone<int>(rgGpuId);
            }

            if (m_rgGpu.Count == 0)
                m_rgGpu.Add(0);

            m_strCudaPath = strCudaPath;
            m_persist = new common.PersistCaffe<T>(m_log, false);

            if (bCreateCudaDnn)
                m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath, false);
        }

        /// <summary>
        /// Releases all GPU and Host resources used by the CaffeControl.
        /// </summary>
        public void dispose()
        {
            if (m_evtSyncMain.WaitOne(0))
                return;

            m_evtSyncMain.Set();

            try            
            {
                if (m_evtCancel != null)
                    m_evtCancel.Set();

                if (m_hCopyBuffer != 0)
                {
                    try
                    {
                        m_cuda.FreeHostBuffer(m_hCopyBuffer);
                    }
                    catch
                    {
                    }

                    m_hCopyBuffer = 0;
                }

                Unload(true, true);

                if (m_cuda != null)
                {
                    try
                    {
                        m_cuda.Dispose();
                    }
                    catch
                    {
                    }

                    m_cuda = null;
                }

                if (m_msWeights != null)
                {
                    m_msWeights.Dispose();
                    m_msWeights = null;
                }

                if (m_dataTransformer != null)
                {
                    m_dataTransformer.Dispose();
                    m_dataTransformer = null;
                }
            }
            finally
            {
                m_evtSyncMain.Reset();
            }
        }

        /// <summary>
        /// Get the file version of the MyCaffe assembly running.
        /// </summary>
        public static FileVersionInfo Version
        {
            get
            {
                string strLocation = Assembly.GetExecutingAssembly().Location;
                return FileVersionInfo.GetVersionInfo(strLocation);
            }
        }

        /// <summary>
        /// Returns the dataset connection information, if used (default = <i>null</i>).
        /// </summary>
        public ConnectInfo DatasetConnectInfo
        {
            get { return m_dsCi; }
        }

        /// <summary>
        /// Returns the stage under which the project was loaded, if any.
        /// </summary>
        public string CurrentStage
        {
            get { return m_strStage; }
        }

        /// <summary>
        /// Clone the current instance of the MyCaffeControl creating a second instance.
        /// </summary>
        /// <remarks>
        /// The second instance has the same project loaded and a copy of the first instance's weights.
        /// </remarks>
        /// <param name="nGpuID">Specifies the GPUID on which to load the second instance.</param>
        /// <returns>The new MyCaffeControl instance is returned.</returns>
        public MyCaffeControl<T> Clone(int nGpuID)
        {
            SettingsCaffe s = m_settings.Clone();
            s.GpuIds = nGpuID.ToString();

            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, m_evtCancel, null, null, null, null, m_strCudaPath);

            if (m_bLoadLite)
                mycaffe.LoadLite(Phase.TRAIN, m_strSolver, m_strModel, null);
            else
                mycaffe.Load(Phase.TRAIN, m_project, null, null, false, m_db, (m_db == null) ? false : true, true, m_strStage);

            Net<T> netSrc = GetInternalNet(Phase.TRAIN);
            Net<T> netDst = mycaffe.GetInternalNet(Phase.TRAIN);

            m_log.CHECK_EQ(netSrc.learnable_parameters.Count, netDst.learnable_parameters.Count, "The src and dst networks do not have the same number of learnable parameters!");

            for (int i = 0; i < netSrc.learnable_parameters.Count; i++)
            {
                Blob<T> bSrc = netSrc.learnable_parameters[i];
                Blob<T> bDst = netDst.learnable_parameters[i];

                mycaffe.m_hCopyBuffer = bDst.CopyFrom(bSrc, false, false, mycaffe.m_hCopyBuffer);
            }

            return mycaffe;
        }

        /// <summary>
        /// Copy the learnable parameter diffs from the source MyCaffeControl into this one.
        /// </summary>
        /// <param name="src">Specifies the source MyCaffeControl whos gradients (blob diffs) are to be copied.</param>
        public void CopyGradientsFrom(MyCaffeControl<T> src)
        {
            Net<T> netSrc = src.GetInternalNet(Phase.TRAIN);
            Net<T> netDst = GetInternalNet(Phase.TRAIN);

            m_log.CHECK_EQ(netSrc.learnable_parameters.Count, netDst.learnable_parameters.Count, "The src and dst networks do not have the same number of learnable parameters!");

            for (int i = 0; i < netSrc.learnable_parameters.Count; i++)
            {
                Blob<T> bSrc = netSrc.learnable_parameters[i];
                Blob<T> bDst = netDst.learnable_parameters[i];

                m_hCopyBuffer = bDst.CopyFrom(bSrc, true, false, m_hCopyBuffer);
            }
        }

        /// <summary>
        /// Copy the learnable parameter data from the source MyCaffeControl into this one.
        /// </summary>
        /// <param name="src">Specifies the source MyCaffeControl whos gradients (blob data) are to be copied.</param>
        public void CopyWeightsFrom(MyCaffeControl<T> src)
        {
            Net<T> netSrc = src.GetInternalNet(Phase.TRAIN);
            Net<T> netDst = GetInternalNet(Phase.TRAIN);

            m_log.CHECK_EQ(netSrc.learnable_parameters.Count, netDst.learnable_parameters.Count, "The src and dst networks do not have the same number of learnable parameters!");

            for (int i = 0; i < netSrc.learnable_parameters.Count; i++)
            {
                Blob<T> bSrc = netSrc.learnable_parameters[i];
                Blob<T> bDst = netDst.learnable_parameters[i];

                m_hCopyBuffer = bDst.CopyFrom(bSrc, false, false, m_hCopyBuffer);
            }
        }

        /// <summary>
        /// Directs the solver to apply the leanred blob diffs to the weights using the solver's learning rate and
        /// update algorithm.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <returns>The learning rate used is returned.</returns>
        public double ApplyUpdate(int nIteration)
        {
            return m_solver.ApplyUpdate(nIteration);
        }

        /// <summary>
        /// Enable/disable testing.  For example reinforcement learning does not use testing.
        /// </summary>
        public bool EnableTesting
        {
            get { return m_solver.EnableTesting; }
            set { m_solver.EnableTesting = value; }
        }

        /// <summary>
        /// Get/set whether or not to use verbose status.  When enabled, the full status is output when loading a project, otherwise a more minimum (faster) set is output (default = false for disabled).
        /// </summary>
        public bool EnableVerboseStatus
        {
            get { return m_bEnableVerboseStatus; }
            set { m_bEnableVerboseStatus = value; }
        }

        /// <summary>
        /// Unload the currently loaded project, if any.
        /// </summary>
        /// <param name="bUnloadImageDb">Optionally, specifies to unload the image database (default = true).</param>
        /// <param name="bIgnoreExceptions">Optionally, specifies to ignore exceptions on error (default = false).</param>
        public void Unload(bool bUnloadImageDb = true, bool bIgnoreExceptions = false)
        {
            if (m_solver == null && m_net == null)
                return;

            if (m_evtSyncUnload.WaitOne(0))
                return;

            m_evtSyncUnload.Set();

            try
            {
                if (m_solver != null)
                {
                    m_solver.Dispose();
                    m_solver = null;
                }

                if (m_net != null)
                {
                    if (m_bOwnRunNet)
                        m_net.Dispose();
                    m_net = null;
                }

                if (m_db != null && bUnloadImageDb)
                {
                    if (m_bDbOwner)
                    {
                        if (m_dataSet != null)
                            m_db.CleanUp(m_dataSet.ID);

                        IDisposable idisp = m_db as IDisposable;
                        if (idisp != null)
                            idisp.Dispose();
                    }

                    m_db = null;
                }

                m_project = null;
            }
            catch (Exception excpt)
            {
                if (!bIgnoreExceptions)
                    throw excpt;
            }
            finally
            {
                m_evtSyncUnload.Reset();
            }
        }

        /// <summary>
        /// Re-initializes each of the specified layers by re-running the filler (if any) specified by the layer.  
        /// When the 'rgstr' parameter is <i>null</i> or otherwise empty, the blobs of all layers are re-initialized. 
        /// </summary>
        /// <param name="target">Specifies the weights to target (e.g. weights, bias or both).</param>
        /// <param name="rgstrLayers">Specifies the layers to reinitialize, when <i>null</i> or empty, all layers are re-initialized</param>
        /// <returns>If a layer is specified and found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        /// <remarks>This method causes the OnTrainingIteration event to fire with the updated values from the re-init.</remarks>
        public bool ReInitializeParameters(WEIGHT_TARGET target, params string[] rgstrLayers)
        {
            Net<T> net = GetInternalNet(Phase.TRAIN);
            net.ReInitializeParameters(target, rgstrLayers);
            return m_solver.ForceOnTrainingIterationEvent();
        }

        /// <summary>
        /// Sets the root solver's onTest event function.
        /// </summary>
        /// <param name="onTest">Specifies the event handler called when testing.</param>
        public void SetOnTestOverride(EventHandler<TestArgs> onTest)
        {
            m_solver.OnTest += onTest;
        }

        /// <summary>
        /// Sets the root solver's onStart event function triggered on the start of each training pass.
        /// </summary>
        /// <param name="onTrainingStart">Specifies the event handler called when testing.</param>
        public void SetOnTrainingStartOverride(EventHandler onTrainingStart)
        {
            m_solver.OnStart += onTrainingStart;
        }

        /// <summary>
        /// Sets the root solver's onTestingStart event function triggered on the start of each testing pass.
        /// </summary>
        /// <param name="onTestingStart">Specifies the event handler called when testing.</param>
        public void SetOnTestingStartOverride(EventHandler onTestingStart)
        {
            m_solver.OnTestStart += onTestingStart;
        }


        /// <summary>
        /// Adds a cancel override.
        /// </summary>
        /// <param name="strEvtCancel">Specifies the new name of the cancel event to add.</param>
        public void AddCancelOverrideByName(string strEvtCancel)
        {
            m_evtCancel.AddCancelOverride(strEvtCancel);
        }

        /// <summary>
        /// Adds a cancel override.
        /// </summary>
        /// <param name="evtCancel">Specifies the new name of the cancel event to add.</param>
        public void AddCancelOverride(CancelEvent evtCancel)
        {
            m_evtCancel.AddCancelOverride(evtCancel);
        }

        /// <summary>
        /// Remove a cancel override.
        /// </summary>
        /// <param name="strEvtCancel">Specifies the new name of the cancel event to remove.</param>
        public void RemoveCancelOverrideByName(string strEvtCancel)
        {
            m_evtCancel.RemoveCancelOverride(strEvtCancel);
        }

        /// <summary>
        /// Remove a cancel override.
        /// </summary>
        /// <param name="evtCancel">Specifies the new name of the cancel event to remove.</param>
        public void RemoveCancelOverride(CancelEvent evtCancel)
        {
            m_evtCancel.RemoveCancelOverride(evtCancel);
        }

        /// <summary>
        /// Enable/disable blob debugging.
        /// </summary>
        /// <remarks>
        /// Note, when enabled, training will dramatically slow down.
        /// </remarks>
        public bool EnableBlobDebugging
        {
            get { return (m_solver == null) ? false : m_solver.EnableBlobDebugging; }
            set
            {
                if (m_solver != null)
                    m_solver.EnableBlobDebugging = value;
            }
        }

        /// <summary>
        /// Enable/disable break training after first detecting a NaN.
        /// </summary>
        /// <remarks>
        /// This option requires that EnableBlobDebugging == <i>true</i>.
        /// </remarks>
        public bool EnableBreakOnFirstNaN
        {
            get { return (m_solver == null) ? false : m_solver.EnableBreakOnFirstNaN; }
            set
            {
                if (m_solver != null)
                    m_solver.EnableBreakOnFirstNaN = value;
            }
        }

        /// <summary>
        /// When enabled (requires EnableBlobDebugging = <i>true</i>), the detailed Nan (and Infinity) detection is perofmed on each blob when training Net.
        /// </summary>
        public bool EnableDetailedNanDetection
        {
            get { return (m_solver == null) ? false : m_solver.EnableDetailedNanDetection; }
            set
            {
                if (m_solver != null)
                    m_solver.EnableDetailedNanDetection = value;
            }
        }

        /// <summary>
        /// Enable/disable layer debugging which causes each layer to check for NAN/INF on each forward/backward pass and throw an exception when found.
        /// </summary>
        /// <remarks>
        /// This option dramatically slows down training and is only recommended during debugging.
        /// </remarks>
        public bool EnableLayerDebugging
        {
            get { return (m_solver == null) ? false : m_solver.EnableLayerDebugging; }
            set
            {
                if (m_solver != null)
                    m_solver.EnableLayerDebugging = value;
            }
        }

        /// <summary>
        /// Enable/disable single step training.
        /// </summary>
        /// <remarks>
        /// This option requires that EnableBlobDebugging == true.
        /// </remarks>
        public bool EnableSingleStep
        {
            get { return (m_solver == null) ? false : m_solver.EnableSingleStep; }
            set
            {
                if (m_solver != null)
                    m_solver.EnableSingleStep = value;
            }
        }

        /// <summary>
        /// Returns the DataTransormer used.
        /// </summary>
        public DataTransformer<T> DataTransformer
        {
            get { return m_dataTransformer; }
        }

        /// <summary>
        /// Returns the settings used to create the control.
        /// </summary>
        public SettingsCaffe Settings
        {
            get { return m_settings; }
        }

        /// <summary>
        /// Returns the CudaDnn connection used.
        /// </summary>
        public CudaDnn<T> Cuda
        {
            get { return m_cuda; }
        }

        /// <summary>
        /// Returns the Log (for output) used.
        /// </summary>
        public Log Log
        {
            get { return m_log; }
        }

        /// <summary>
        /// Returns the persist used to load and save weights.
        /// </summary>
        public IXPersist<T> Persist
        {
            get { return m_persist; }
        }

        /// <summary>
        /// Returns the MyCaffeImageDatabase or MyCaffeTemporalDatabase used.
        /// </summary>
        public IXDatabaseBase Database
        {
            get { return m_db; }
        }

        /// <summary>
        /// Returns the CancelEvent used.
        /// </summary>
        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        /// <summary>
        /// Returns a list of Active GPU's used by the control.
        /// </summary>
        public List<int> ActiveGpus
        {
            get { return m_rgGpu; }
        }

        /// <summary>
        /// Returns a string describing the active label counts observed during training.
        /// </summary>
        /// <remarks>
        /// This string can help diagnose label balancing issue.
        /// </remarks>
        public string ActiveLabelCounts
        {
            get { return m_solver.ActiveLabelCounts; }
        }

        /// <summary>
        /// Returns a string describing the label query hit percentages observed during training.
        /// </summary>
        /// <remarks>
        /// This string can help diagnose label balancing issue.
        /// </remarks>
        public string LabelQueryHitPercents
        {
            get { return m_solver.LabelQueryHitPercents; }
        }

        /// <summary>
        /// Returns a string describing the label query epochs observed during training.
        /// </summary>
        /// <remarks>
        /// This string can help diagnose label balancing issue.
        /// </remarks>
        public string LabelQueryEpochs
        {
            get { return m_solver.LabelQueryEpochs; }
        }

        /// <summary>
        /// Returns the name of the current device used.
        /// </summary>
        public string CurrentDevice
        {
            get
            {
                int nId = m_cuda.GetDeviceID();
                return "GPU #" + nId.ToString() + "  " + m_cuda.GetDeviceName(nId);
            }
        }

        /// <summary>
        /// Returns the name of the currently loaded project.
        /// </summary>
        public ProjectEx CurrentProject
        {
            get { return m_project; }
        }

        /// <summary>
        /// Returns the current iteration.
        /// </summary>
        public int CurrentIteration
        {
            get { return m_solver.CurrentIteration; }
        }

        /// <summary>
        /// Returns the maximum iteration.
        /// </summary>
        public int MaximumIteration
        {
            get { return m_solver.MaximumIteration; }
        }

        /// <summary>
        /// Returns the total number of devices installed on this computer.
        /// </summary>
        /// <returns></returns>
        public int GetDeviceCount()
        {
            return m_cuda.GetDeviceCount();
        }

        /// <summary>
        /// Returns the device name of a given device ID.
        /// </summary>
        /// <param name="nDeviceID">Specifies the device ID.</param>
        /// <returns></returns>
        public string GetDeviceName(int nDeviceID)
        {
            return m_cuda.GetDeviceName(nDeviceID);
        }

        /// <summary>
        /// Returns the last phase run (TRAIN, TEST or RUN).
        /// </summary>
        public Phase LastPhase
        {
            get { return m_lastPhaseRun; }
        }

        /// <summary>
        /// Creates a net parameter for the RUN phase.
        /// </summary>
        /// <remarks>
        /// This function transforms the training net parameter into a new net parameter suitable to run in the RUN phase.
        /// </remarks>
        /// <param name="p">Specifies a project.</param>
        /// <param name="transform_param">Specifies the TransformationParameter to use.</param>
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        protected NetParameter createNetParameterForRunning(ProjectEx p, out TransformationParameter transform_param)
        {
            return createNetParameterForRunning(p.Dataset, p.ModelDescription, out transform_param, p.Stage);
        }

        /// <summary>
        /// Creates a net parameter for the RUN phase.
        /// </summary>
        /// <remarks>
        /// This function transforms the training net parameter into a new net parameter suitable to run in the RUN phase.
        /// </remarks>
        /// <param name="ds">Specifies a DatasetDescriptor for the dataset used.</param>
        /// <param name="strModel">Specifies the model descriptor.</param>
        /// <param name="transform_param">Specifies the TransformationParameter to use.</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        protected NetParameter createNetParameterForRunning(DatasetDescriptor ds, string strModel, out TransformationParameter transform_param, Stage stage = Stage.NONE)
        {
            BlobShape shape = datasetToShape(ds);
            return createNetParameterForRunning(shape, strModel, out transform_param, stage);
        }

        /// <summary>
        /// Creates a net parameter for the RUN phase.
        /// </summary>
        /// <remarks>
        /// This function transforms the training net parameter into a new net parameter suitable to run in the RUN phase.
        /// </remarks>
        /// <param name="shape">Specifies the shape of the images that will be used.</param>
        /// <param name="strModel">Specifies the model descriptor.</param>
        /// <param name="transform_param">Specifies the TransformationParameter to use.</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        protected NetParameter createNetParameterForRunning(BlobShape shape, string strModel, out TransformationParameter transform_param, Stage stage = Stage.NONE)
        {
            NetParameter param = CreateNetParameterForRunning(shape, strModel, out transform_param, stage);
            m_inputShape = shape;
            return param;
        }

        /// <summary>
        /// Creates a net parameter for the RUN phase.
        /// </summary>
        /// <remarks>
        /// This function transforms the training net parameter into a new net parameter suitable to run in the RUN phase.
        /// </remarks>
        /// <param name="sdMean">Specifies the mean image data used to size the network and as the mean image when used in the transformation parameter.</param>
        /// <param name="strModel">Specifies the model descriptor.</param>
        /// <param name="transform_param">Returns the TransformationParameter to use.</param>
        /// <param name="nC">Returns the discovered channel sizing to use.</param>
        /// <param name="nH">Returns the discovered height sizing to use.</param>
        /// <param name="nW">Returns the discovered width sizing to use.</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        protected NetParameter createNetParameterForRunning(SimpleDatum sdMean, string strModel, out TransformationParameter transform_param, out int nC, out int nH, out int nW, Stage stage = Stage.NONE)
        {
            nC = 0;
            nH = 0;
            nW = 0;

            if (sdMean != null)
            {
                nC = sdMean.Channels;
                nH = sdMean.Height;
                nW = sdMean.Width;
            }
            else
            {
                RawProto protoModel = RawProto.Parse(strModel);
                RawProtoCollection layers = protoModel.FindChildren("layer");

                foreach (RawProto layer in layers)
                {
                    RawProto type = layer.FindChild("type");
                    if (type != null && type.Value == "Input")
                    {
                        RawProto input_param = layer.FindChild("input_param");
                        if (input_param != null)
                        {
                            RawProto shape1 = input_param.FindChild("shape");
                            if (shape1 != null)
                            {
                                RawProtoCollection rgDim = shape1.FindChildren("dim");
                                int nNum = (rgDim.Count > 0) ? int.Parse(rgDim[0].Value) : 1;
                                nC = (rgDim.Count > 1) ? int.Parse(rgDim[1].Value) : 1;
                                nH = (rgDim.Count > 2) ? int.Parse(rgDim[2].Value) : 1;
                                nW = (rgDim.Count > 3) ? int.Parse(rgDim[3].Value) : 1;
                                break;
                            }
                        }
                    }
                }
            }

            if (nC == 0 && nH == 0 && nW == 0)
                throw new Exception("Could not dicern the shape to use for no 'sdMean' parameter was supplied and the model does not contain an 'Input' layer!");

            BlobShape shape = new BlobShape(1, nC, nH, nW);
            NetParameter param = CreateNetParameterForRunning(shape, strModel, out transform_param, stage);
            m_inputShape = shape;

            return param;
        }

        /// <summary>
        /// Creates a net parameter for the RUN phase.
        /// </summary>
        /// <remarks>
        /// This function transforms the training net parameter into a new net parameter suitable to run in the RUN phase.
        /// </remarks>
        /// <param name="shape">Specifies the shape of the images that will be used.</param>
        /// <param name="strModel">Specifies the model descriptor.</param>
        /// <param name="transform_param">Specifies the TransformationParameter to use.</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <param name="bSkipLossLayer">Optionally, specifies to skip the loss layer and not output a converted layer to replace it (default = false).</param>
        /// <param name="bMaintainBatchSize">Optionally, specifies to keep the batch size, otherwise batch size is set to 1 (default = false).</param>
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        public static NetParameter CreateNetParameterForRunning(BlobShape shape, string strModel, out TransformationParameter transform_param, Stage stage = Stage.NONE, bool bSkipLossLayer = false, bool bMaintainBatchSize = false)
        {
            int nNum = (bMaintainBatchSize) ? shape.dim[0] : 1;
            int nImageChannels = shape.dim[1];
            int nImageHeight = (shape.dim.Count > 2) ? shape.dim[2] : 1;
            int nImageWidth = (shape.dim.Count > 3) ? shape.dim[3] : 1;

            transform_param = null;

            RawProto protoTransform = null;
            bool bSkipTransformParam = false;
            RawProto protoModel = ProjectEx.CreateModelForRunning(strModel, "data", nNum, nImageChannels, nImageHeight, nImageWidth, out protoTransform, out bSkipTransformParam, stage, bSkipLossLayer);

            if (!bSkipTransformParam)
            {
                if (protoTransform != null)
                    transform_param = TransformationParameter.FromProto(protoTransform);
                else
                    transform_param = new param.TransformationParameter();

                if (transform_param.resize_param != null && transform_param.resize_param.Active)
                {
                    shape.dim[2] = (int)transform_param.resize_param.height;
                    shape.dim[3] = (int)transform_param.resize_param.width;
                }
            }

            NetParameter np = NetParameter.FromProto(protoModel);

            string strInput = "";
            foreach (LayerParameter layer in np.layer)
            {
                layer.PrepareRunModel();

                string strInput1 = layer.PrepareRunModelInputs();
                if (!string.IsNullOrEmpty(strInput1))
                    strInput += strInput1;
            }

            if (!string.IsNullOrEmpty(strInput))
            {
                RawProto proto = RawProto.Parse(strInput);
                Dictionary<string, BlobShape> rgInput = NetParameter.InputFromProto(proto);

                if (rgInput.Count > 0)
                {
                    np.input = new List<string>();
                    np.input_dim = new List<int>();
                    np.input_shape = new List<BlobShape>();

                    foreach (KeyValuePair<string, BlobShape> kv in rgInput)
                    {
                        np.input.Add(kv.Key);
                        np.input_shape.Add(kv.Value);
                    }
                }
            }
            

            np.ProjectID = 0;
            np.state.phase = Phase.RUN;

            return np;
        }

        private BlobShape datasetToShape(DatasetDescriptor ds)
        {
            int nH = 1;
            int nW = 1;
            int nC = 1;

            if (!ds.IsModelData)
            {
                nH = ds.TestingSource.Height;
                nW = ds.TestingSource.Width;
                nC = ds.TestingSource.Channels;
            }

            List<int> rgShape = new List<int>() { 1, nC, nH, nW };
            return new BlobShape(rgShape);
        }

        private Stage getStage(string strStage)
        {
            if (strStage == Stage.RNN.ToString())
                return Stage.RNN;

            if (strStage == Stage.RL.ToString())
                return Stage.RL;

            return Stage.NONE;
        }

        private string addStage(string strModel, Phase phase, string strStage)
        {
            if (string.IsNullOrEmpty(strStage))
                return strModel;

            RawProto proto = RawProto.Parse(strModel);
            NetParameter param = NetParameter.FromProto(proto);

            param.state.stage.Clear();
            param.state.phase = phase;
            param.state.stage.Add(strStage);

            return param.ToProto("root", true).ToString();
        }

        /// <summary>
        /// Prepare the testing image mean by copying the training image mean if the testing image mean is missing.
        /// </summary>
        /// <param name="prj">Specifies the project whos image mean is to be prepared.</param>
        public void PrepareImageMeans(ProjectEx prj)
        {
            if (prj.Dataset.IsModelData || prj.Dataset.IsGym)
                return;

            DatasetFactory factory = new DatasetFactory();

            // Copy the training image mean to the testing source if it does not have a mean.
            // NOTE: This this will not impact a service based image database that is already loaded,
            // - it must be reloaded.
            int nDstID = factory.GetRawImageMeanID(prj.Dataset.TestingSource.ID);
            if (nDstID == 0)
            {
                int nSrcID = factory.GetRawImageMeanID(prj.Dataset.TrainingSource.ID);
                if (nSrcID != 0)
                    factory.CopyImageMean(prj.Dataset.TrainingSourceName, prj.Dataset.TestingSourceName);
            }

            if (prj.DatasetTarget != null)
            {
                // Copy the training image mean to the testing source if it does not have a mean.
                // NOTE: This this will not impact a service based image database that is already loaded,
                // - it must be reloaded.
                nDstID = factory.GetRawImageMeanID(prj.DatasetTarget.TestingSource.ID);
                if (nDstID == 0)
                {
                    int nSrcID = factory.GetRawImageMeanID(prj.DatasetTarget.TrainingSource.ID);
                    if (nSrcID != 0)
                        factory.CopyImageMean(prj.DatasetTarget.TrainingSourceName, prj.DatasetTarget.TestingSourceName);
                }
            }
        }

        private bool verifySharedWeights()
        {
            Net<T> netTest = m_solver.TestingNet;
            if (netTest != null)
            {
                Net<T> netTrain = m_solver.TrainingNet;

                if (netTrain.parameters.Count != netTest.parameters.Count)
                {
                    m_log.WriteLine("WARNING: Training net has a different number of parameters than the testing net!");
                    return false;
                }

                for (int i = 0; i < netTrain.parameters.Count; i++)
                {
                    if (netTrain.parameters[i].gpu_data != netTest.parameters[i].gpu_data)
                    {
                        m_log.WriteLine("WARNING: Training net parameter " + i.ToString() + " is not shared with the testing net!");
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Load a project and optionally the MyCaffeImageDatabase.
        /// </summary>
        /// <remarks>
        /// This load function uses the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="phase">Specifies the Phase for which the load should focus.</param>
        /// <param name="p">Specifies the Project to load.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection override (overides the label selection in SettingsCaffe).  The label selection dictates how the label sets are selected.</param>
        /// <param name="itemSelectionOverride">Optionally, specifies the item (e.g., image or temporal item) selection override (overides the item selection in SettingsCaffe).  The item selection dictates how the items are selected from each label set.</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="db">Optionally, specifies the MyCaffeImageDatabase or MyCaffeTemporalDatabase to use.  When <i>null</i>, an instance if the in-memory database is created internally.</param>
        /// <param name="bUseDb">Optionally, specifies to use the in-memory database based on the DB_VERSION specified in the settings.</param>
        /// <param name="bCreateRunNet">Optionally, specifies whether or not to create the Run net.</param>
        /// <param name="strStage">Optionally, specifies a stage under which to load the model.</param>
        /// <param name="bEnableMemTrace">Optionally, specifies to enable the memory tracing (only available in debug builds).</param>
        /// <returns>If the project is loaded the function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        public bool Load(Phase phase, ProjectEx p, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? itemSelectionOverride = null, bool bResetFirst = false, IXDatabaseBase db = null, bool bUseDb = true, bool bCreateRunNet = true, string strStage = null, bool bEnableMemTrace = false)
        {
            try
            {
                m_log.Enable = m_bEnableVerboseStatus;

                DatasetFactory factory = new DatasetFactory();
                m_strStage = strStage;
                m_db = db;
                m_bDbOwner = false;

                if (db != null && bUseDb)
                {
                    if (m_settings.DbVersion != db.GetVersion())
                        throw new Exception("The database version in the settings (" + m_settings.DbVersion.ToString() + ") must match the database version (" + db.GetVersion().ToString() + ") of the 'db' parameter!");
                }

                if (m_db == null && bUseDb)
                {
                    switch (m_settings.DbVersion)
                    {
                        case DB_VERSION.IMG_V1:
                            m_db = new MyCaffeImageDatabase(m_log);
                            m_log.WriteLine("Loading primary images...", true);
                            m_bDbOwner = true;
                            m_log.Enable = true;
                            ((IXImageDatabase1)m_db).InitializeWithDs1(m_settings, p.Dataset, m_evtCancel.Name);
                            break;

                        case DB_VERSION.TEMPORAL:
                            throw new NotImplementedException("The temporal database is not yet supported.");

                        default:
                            m_db = new MyCaffeImageDatabase2(m_log);
                            m_log.WriteLine("Loading primary images...", true);
                            m_bDbOwner = true;
                            m_log.Enable = true;
                            ((IXImageDatabase2)m_db).InitializeWithDs(m_settings, p.Dataset, m_evtCancel.Name);
                            break;
                    }

                    if (m_evtCancel.WaitOne(0))
                        return false;

                    // m_db.UpdateLabelBoosts(p.ID, p.Dataset.TrainingSource.ID);

                    Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> selMethod = MyCaffeImageDatabase.GetSelectionMethod(p);
                    DB_LABEL_SELECTION_METHOD lblSel = selMethod.Item1;
                    DB_ITEM_SELECTION_METHOD imgSel = selMethod.Item2;

                    if (labelSelectionOverride.HasValue)
                        lblSel = labelSelectionOverride.Value;

                    if (itemSelectionOverride.HasValue)
                        imgSel = itemSelectionOverride.Value;

                    m_db.SetSelectionMethod(lblSel, imgSel);
                    m_db.QueryItemMean(p.Dataset.TrainingSource.ID);
                    m_log.WriteLine("Images loaded.");

                    if (p.TargetDatasetID > 0)
                    {
                        DatasetDescriptor dsTarget = factory.LoadDataset(p.TargetDatasetID);

                        m_log.WriteLine("Loading target dataset '" + dsTarget.Name + "' images using " + m_settings.DbLoadMethod.ToString() + " loading method.");
                        string strType = "images";

                        switch (m_settings.DbVersion)
                        {
                            case DB_VERSION.IMG_V1:
                                ((IXImageDatabase1)m_db).LoadDatasetByID1(dsTarget.ID);
                                break;

                            case DB_VERSION.TEMPORAL:
                                strType = "items";
                                throw new NotImplementedException("The temporal database is not yet supported.");

                            default:
                                ((IXImageDatabase2)m_db).LoadDatasetByID(dsTarget.ID);
                                break;
                        }

                        m_db.QueryItemMean(dsTarget.TrainingSource.ID);
                        m_log.WriteLine("Target dataset " + strType + " loaded.");
                    }

                    m_log.Enable = m_bEnableVerboseStatus;
                }

                p.ModelDescription = addStage(p.ModelDescription, phase, strStage);
                m_project = p;
                m_project.Stage = getStage(m_strStage);

                if (m_project == null)
                    throw new Exception("You must specify a project.");

                m_dataSet = m_project.Dataset;

                if (m_cuda != null)
                    m_cuda.Dispose();

                m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath, bResetFirst, bEnableMemTrace);

                m_log.WriteLine("Cuda Connection created using '" + m_cuda.Path + "'.", true);

                if (phase == Phase.TEST || phase == Phase.TRAIN)
                {
                    m_log.WriteLine("Creating solver...", true);

                    m_solver = Solver<T>.Create(m_cuda, m_log, p, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist, m_rgGpu.Count, 0);
                    m_solver.SnapshotWeightUpdateMethod = m_settings.SnapshotWeightUpdateMethod;
                    if (p.WeightsState != null || p.SolverState != null)
                    {
                        string strSkipBlobType = null;

                        ParameterDescriptor param = p.Parameters.Find("ModelResized");
                        if (param != null && param.Value == "True")
                            strSkipBlobType = BLOB_TYPE.IP_WEIGHT.ToString();

                        if (p.SolverState != null)
                            m_solver.Restore(p.WeightsState, p.SolverState, strSkipBlobType);
                        else
                            m_solver.TrainingNet.LoadWeights(p.WeightsState, m_persist);
                    }

                    m_solver.OnSnapshot += new EventHandler<SnapshotArgs>(m_solver_OnSnapshot);
                    m_solver.OnTrainingIteration += new EventHandler<TrainingIterationArgs<T>>(m_solver_OnTrainingIteration);
                    m_solver.OnTestingIteration += new EventHandler<TestingIterationArgs<T>>(m_solver_OnTestingIteration);
                    m_log.WriteLine("Solver created.", true);

                    verifySharedWeights();
                }

                if (m_db is IXImageDatabase1)
                {
#warning ImageDatabase V1 only
                    if (phase == Phase.TRAIN && m_db != null)
                        ((IXImageDatabase1)m_db).UpdateLabelBoosts(p.ID, m_dataSet.TrainingSource.ID);

                    if (phase == Phase.TEST && m_db != null)
                        ((IXImageDatabase1)m_db).UpdateLabelBoosts(p.ID, m_dataSet.TestingSource.ID);
                }

                if (phase == Phase.RUN && !bCreateRunNet)
                    throw new Exception("You cannot opt out of creating the Run net when using the RUN phase.");

                if (p == null || !bCreateRunNet)
                    return true;

                TransformationParameter tp = null;
                NetParameter netParam = createNetParameterForRunning(p, out tp);

                m_dataTransformer = null;

                if (tp != null && !p.Dataset.IsModelData && !p.Dataset.IsGym && m_settings.DbVersion != DB_VERSION.TEMPORAL)
                {
                    SimpleDatum sdMean = (m_db == null) ? null : m_db.QueryItemMean(m_dataSet.TrainingSource.ID);
                    int nC = m_project.Dataset.TrainingSource.Channels;
                    int nH = m_project.Dataset.TrainingSource.Height;
                    int nW = m_project.Dataset.TrainingSource.Width;

                    if (sdMean != null)
                    {
                        m_log.CHECK_EQ(nC, sdMean.Channels, "The mean channel count does not match the datasets channel count.");
                        m_log.CHECK_EQ(nH, sdMean.Height, "The mean height count does not match the datasets height count.");
                        m_log.CHECK_EQ(nW, sdMean.Width, "The mean width count does not match the datasets width count.");
                    }

                    m_dataTransformer = new DataTransformer<T>(m_cuda, m_log, tp, Phase.RUN, nC, nH, nW, sdMean);
                }

                m_log.WriteLine("Creating run net...", true);

                if (phase == Phase.RUN)
                {
                    m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_db);

                    if (p.WeightsState != null)
                    {
                        m_log.WriteLine("Loading run weights...", true);
                        loadWeights(m_net, p.WeightsState);
                    }
                }
                else if (phase == Phase.TEST || phase == Phase.TRAIN)
                {
                    try
                    {
                        m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_db, Phase.RUN, null, m_solver.TrainingNet);
                    }
                    catch (Exception excpt)
                    {
                        m_log.WriteLine("WARNING: Failed to create run net - using Test net instead for running.  Error = " + excpt.Message);
                        m_net = m_solver.TestingNet;
                        m_bOwnRunNet = false;
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_log.Enable = true;
            }

            return true;
        }

        /// <summary>
        /// Load a project and optionally the MyCaffeImageDatabase.
        /// </summary>
        /// <remarks>
        /// This load function uses the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="phase">Specifies the Phase for which the load should focus.</param>
        /// <param name="strSolver">Specifies the solver descriptor.</param>
        /// <param name="strModel">Specifies the model desciptor.</param>
        /// <param name="rgWeights">Optionally, specifies the weights to load, or <i>null</i> to ignore.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection override (overides the label selection in SettingsCaffe).  The label selection dictates how the label sets are selected.</param>
        /// <param name="itemSelectionOverride">Optionally, specifies the item (e.g., image or temporal item) selection override (overides the item selection in SettingsCaffe).  The item selection dictates how the item are selected from each label set.</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="db">Optionally, specifies the in-memory MyCaffeDatabase to use.  When <i>null</i>, an instance if the MyCaffeImageDatabase or MyCaffeTemporalDatabase is created internally depending on the DB_VERSION used.</param>
        /// <param name="bUseDb">Optionally, specifies to use the in-memory database based on the DB_VERSION specified in the settings.</param>
        /// <param name="bCreateRunNet">Optionally, specifies whether or not to create the Run net (default = true).</param>
        /// <param name="strStage">Optionally, specifies a stage under which to load the model.</param>
        /// <param name="bEnableMemTrace">Optionally, specifies to enable the memory tracing (only available in debug builds).</param>
        /// <returns>If the project is loaded the function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        public bool Load(Phase phase, string strSolver, string strModel, byte[] rgWeights, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? itemSelectionOverride = null, bool bResetFirst = false, IXDatabaseBase db = null, bool bUseDb = true, bool bCreateRunNet = true, string strStage = null, bool bEnableMemTrace = false)
        {
            try
            {
                m_log.Enable = m_bEnableVerboseStatus;

                m_strStage = strStage;
                m_db = db;
                m_bDbOwner = false;

                RawProto protoSolver = RawProto.Parse(strSolver);
                SolverParameter solverParam = SolverParameter.FromProto(protoSolver);

                strModel = addStage(strModel, phase, strStage);

                RawProto protoModel = RawProto.Parse(strModel);
                solverParam.net_param = NetParameter.FromProto(protoModel);

                m_dataSet = findDataset(solverParam.net_param);

                if (db != null && bUseDb)
                {
                    if (m_settings.DbVersion != db.GetVersion())
                        throw new Exception("The database version in the settings (" + m_settings.DbVersion.ToString() + ") must match the database version (" + db.GetVersion().ToString() + ") of the 'db' parameter!");
                }

                if (m_db == null && bUseDb)
                {
                    string strType = "images";
                    switch (m_settings.DbVersion)
                    {
                        case DB_VERSION.IMG_V1:
                            m_db = new MyCaffeImageDatabase(m_log);
                            ((IXImageDatabase1)m_db).InitializeWithDs1(m_settings, m_dataSet, m_evtCancel.Name);
                            m_log.WriteLine("Loading primary images...", true);
                            m_log.Enable = true;
                            break;

                        case DB_VERSION.IMG_V2:
                            m_db = new MyCaffeImageDatabase2(m_log);
                            ((IXImageDatabase2)m_db).InitializeWithDs(m_settings, m_dataSet, m_evtCancel.Name);
                            m_log.WriteLine("Loading primary images...", true);
                            m_log.Enable = true;
                            break;

                        case DB_VERSION.TEMPORAL:
                            throw new NotImplementedException("The temporal database is not yet implemented!");
                    }

                    m_bDbOwner = true;
                    if (m_evtCancel.WaitOne(0))
                        return false;

                    Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> selMethod = MyCaffeImageDatabase.GetSelectionMethod(m_settings);
                    DB_LABEL_SELECTION_METHOD lblSel = selMethod.Item1;
                    DB_ITEM_SELECTION_METHOD imgSel = selMethod.Item2;

                    if (labelSelectionOverride.HasValue)
                        lblSel = labelSelectionOverride.Value;

                    if (itemSelectionOverride.HasValue)
                        imgSel = itemSelectionOverride.Value;

                    m_db.SetSelectionMethod(lblSel, imgSel);
                    m_db.QueryItemMean(m_dataSet.TrainingSource.ID);
                    m_log.WriteLine("Images loaded.", true);

                    DatasetDescriptor dsTarget = findDataset(solverParam.net_param, m_dataSet);
                    if (dsTarget != null)
                    {
                        m_log.WriteLine("Loading target dataset '" + dsTarget.Name + "' " + strType + " using " + m_settings.DbLoadMethod.ToString() + " loading method.", true);

                        switch (m_settings.DbVersion)
                        {
                            case DB_VERSION.IMG_V1:
                                ((IXImageDatabase1)m_db).LoadDatasetByID1(dsTarget.ID);
                                break;

                            case DB_VERSION.IMG_V2:
                                ((IXImageDatabase2)m_db).LoadDatasetByID(dsTarget.ID);
                                break;

                            case DB_VERSION.TEMPORAL:
                                throw new NotImplementedException("The temporal database is not yet implemented!");
                        }

                        m_db.QueryItemMean(dsTarget.TrainingSource.ID);
                        m_log.WriteLine("Target dataset " + strType + " loaded.", true);
                    }

                    m_log.Enable = m_bEnableVerboseStatus;
                }

                m_project = null;

                if (m_cuda != null)
                    m_cuda.Dispose();

                m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath, bResetFirst, bEnableMemTrace);
                m_log.WriteLine("Cuda Connection created using '" + m_cuda.Path + "'.", true);

                if (phase == Phase.TEST || phase == Phase.TRAIN)
                {
                    m_log.WriteLine("Creating solver...", true);

                    m_solver = Solver<T>.Create(m_cuda, m_log, solverParam, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist, m_rgGpu.Count, 0);

                    if (rgWeights != null)
                    {
                        m_log.WriteLine("Restoring weights...", true);
                        m_solver.Restore(rgWeights, null);
                    }

                    m_solver.OnSnapshot += new EventHandler<SnapshotArgs>(m_solver_OnSnapshot);
                    m_solver.OnTrainingIteration += new EventHandler<TrainingIterationArgs<T>>(m_solver_OnTrainingIteration);
                    m_solver.OnTestingIteration += new EventHandler<TestingIterationArgs<T>>(m_solver_OnTestingIteration);
                    m_log.WriteLine("Solver created.", true);
                }

                if (!bCreateRunNet)
                {
                    if (phase == Phase.RUN)
                        throw new Exception("You cannot opt out of creating the Run net when using the RUN phase.");

                    return true;
                }

                TransformationParameter tp = null;
                NetParameter netParam = createNetParameterForRunning(m_dataSet, strModel, out tp);

                m_dataTransformer = null;

                if (tp != null)
                {
                    SimpleDatum sdMean = (m_db == null) ? null : m_db.QueryItemMean(m_dataSet.TrainingSource.ID);
                    int nC = 0;
                    int nH = 0;
                    int nW = 0;

                    if (sdMean != null)
                    {
                        nC = sdMean.Channels;
                        nH = sdMean.Height;
                        nW = sdMean.Width;
                    }
                    else if (m_project != null)
                    {
                        nC = m_project.Dataset.TrainingSource.Channels;
                        nH = m_project.Dataset.TrainingSource.Height;
                        nW = m_project.Dataset.TrainingSource.Width;
                    }

                    if (nC == 0 || nH == 0 || nW == 0)
                        throw new Exception("Unable to size the Data Transformer for there is no Mean or Project to gather the sizing information from.");

                    m_dataTransformer = new DataTransformer<T>(m_cuda, m_log, tp, Phase.RUN, nC, nH, nW, sdMean);
                }

                m_log.WriteLine("Creating run net...", true);

                if (phase == Phase.RUN)
                {
                    m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_db);

                    if (rgWeights != null)
                    {
                        m_log.WriteLine("Loading run weights...", true);
                        loadWeights(m_net, rgWeights);
                    }
                }
                else if (phase == Phase.TEST || phase == Phase.TRAIN)
                {
                    netParam.force_backward = true;

                    try
                    {
                        m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_db, Phase.RUN, null, m_solver.TrainingNet);
                    }
                    catch (Exception excpt)
                    {
                        m_log.WriteLine("WARNING: Failed to create run net - using Test net instead for running.  Error = " + excpt.Message);
                        m_net = m_solver.TestingNet;
                        m_bOwnRunNet = false;
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_log.Enable = true;
            }

            return true;
        }

        /// <summary>
        /// Load a solver and model without using the MyCaffeImageDatabase.
        /// </summary>
        /// <remarks>
        /// This load function is a load lite that does not use the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="phase">Specifies the Phase for which the load should focus.</param>
        /// <param name="strSolver">Specifies the solver descriptor.</param>
        /// <param name="strModel">Specifies the model desciptor.</param>
        /// <param name="rgWeights">Optionally, specifies the weights to load, or <i>null</i> to ignore (default = null).</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="bCreateRunNet">Optionally, specifies whether or not to create the Run net (default = true).</param>
        /// <param name="sdMean">Optionally, specifies the image mean to use (default = null).</param>
        /// <param name="strStage">Optionally, specifies a stage under which to load the model.</param>
        /// <param name="bEnableMemTrace">Optionally, specifies to enable the memory tracing (only available in debug builds).</param>
        /// <param name="db">Optionally, specifies the in-memory MyCaffeDatabase to use.  When <i>null</i>, an instance if the MyCaffeImageDatabase or MyCaffeTemporalDatabase is created internally depending on the DB_VERSION used.</param>
        /// <returns>If the project is loaded the function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        public bool LoadLite(Phase phase, string strSolver, string strModel, byte[] rgWeights = null, bool bResetFirst = false, bool bCreateRunNet = true, SimpleDatum sdMean = null, string strStage = null, bool bEnableMemTrace = false, IXDatabaseBase db = null)
        {
            try
            {
                m_log.Enable = m_bEnableVerboseStatus;

                m_bLoadLite = true;
                m_strSolver = strSolver;
                m_strModel = strModel;

                m_strStage = strStage;
                m_db = db;
                m_bDbOwner = false;

                RawProto protoSolver = RawProto.Parse(strSolver);
                SolverParameter solverParam = SolverParameter.FromProto(protoSolver);

                strModel = addStage(strModel, phase, strStage);

                RawProto protoModel = RawProto.Parse(strModel);
                solverParam.net_param = NetParameter.FromProto(protoModel);

                m_dataSet = null;
                m_project = null;

                if (m_cuda != null)
                    m_cuda.Dispose();

                m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath, bResetFirst, bEnableMemTrace);
                m_log.WriteLine("Cuda Connection created using '" + m_cuda.Path + "'.", true);

                if (phase == Phase.TEST || phase == Phase.TRAIN)
                {
                    m_log.WriteLine("Creating solver...", true);

                    m_solver = Solver<T>.Create(m_cuda, m_log, solverParam, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_db, m_persist, m_rgGpu.Count, 0);

                    if (rgWeights != null)
                        m_solver.Restore(rgWeights, null);

                    m_solver.OnSnapshot += new EventHandler<SnapshotArgs>(m_solver_OnSnapshot);
                    m_solver.OnTrainingIteration += new EventHandler<TrainingIterationArgs<T>>(m_solver_OnTrainingIteration);
                    m_solver.OnTestingIteration += new EventHandler<TestingIterationArgs<T>>(m_solver_OnTestingIteration);
                    m_log.WriteLine("Solver created.");
                }

                if (!bCreateRunNet)
                {
                    if (phase == Phase.RUN)
                        throw new Exception("You cannot opt out of creating the Run net when using the RUN phase.");

                    return true;
                }

                TransformationParameter tp = null;
                int nC = 0;
                int nH = 0;
                int nW = 0;
                NetParameter netParam = createNetParameterForRunning(sdMean, strModel, out tp, out nC, out nH, out nW);

                m_dataTransformer = null;

                if (tp != null)
                {
                    if (nC == 0 || nH == 0 || nW == 0)
                        throw new Exception("Unable to size the Data Transformer for no Mean image was provided as the 'sdMean' parameter which is used to gather the sizing information.");

                    m_dataTransformer = new DataTransformer<T>(m_cuda, m_log, tp, Phase.RUN, nC, nH, nW, sdMean);
                }

                m_log.WriteLine("Creating run net...", true);

                if (phase == Phase.RUN)
                {
                    m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, null);

                    if (rgWeights != null)
                    {
                        m_log.WriteLine("Loading run weights...", true);
                        loadWeights(m_net, rgWeights);
                    }
                }
                else if (phase == Phase.TEST || phase == Phase.TRAIN)
                {
                    netParam.force_backward = true;

                    try
                    {
                        m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, null, Phase.RUN, null, m_solver.TrainingNet);
                    }
                    catch (Exception excpt)
                    {
                        m_log.WriteLine("WARNING: Failed to create run net - using Test net instead for running.  Error = " + excpt.Message);
                        m_net = m_solver.TestingNet;
                        m_bOwnRunNet = false;
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_log.Enable = true;
            }

            return true;
        }

        /// <summary>
        /// The LoadToRun method loads the MyCaffeControl for running only (e.g. deployment).
        /// </summary>
        /// <remarks>
        /// This method does not use the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="strModel">Specifies the model description to load.</param>
        /// <param name="rgWeights">Specifies the trained weights to load.</param>
        /// <param name="shape">Specifies the expected shape to run on.</param>
        /// <param name="sdMean">Optionally, specifies the simple datum mean to subtract from input images that are run.</param>
        /// <param name="transParam">Optionally, specifies the TransformationParameter to use.  When using a 'deployment' model that has no data layers, you should supply a transformation parameter
        /// that matches the transformation used during training.</param>
        /// <param name="bForceBackward">Optionally, specifies to force backward propagation in the event that a backward pass is to be run on the Run net - The DeepDraw functionality
        /// uses this setting so that it can view what the trained weights actually see.</param>
        /// <param name="bConvertToRunNet">When <i>true</i>, the 'strModel' is converted from a training model to a run model, otherwise the model is used unaltered (default = <i>true</i>)</param>
        public void LoadToRun(string strModel, byte[] rgWeights, BlobShape shape, SimpleDatum sdMean = null, TransformationParameter transParam = null, bool bForceBackward = false, bool bConvertToRunNet = true)
        {
            try
            {
                m_log.Enable = m_bEnableVerboseStatus;
                m_dataSet = null;
                m_project = null;
                m_loadToRunShape = shape;

                if (m_cuda != null)
                    m_cuda.Dispose();

                m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath);
                m_log.WriteLine("Cuda Connection created using '" + m_cuda.Path + "'.", true);

                TransformationParameter tp = null;
                NetParameter netParam = null;

                if (bConvertToRunNet)
                {
                    netParam = createNetParameterForRunning(shape, strModel, out tp);
                }
                else
                {
                    netParam = NetParameter.FromProto(RawProto.Parse(strModel));

                    foreach (LayerParameter layer in netParam.layer)
                    {
                        if (layer.transform_param != null)
                        {
                            tp = layer.transform_param;
                            break;
                        }
                    }
                }

                netParam.force_backward = bForceBackward;

                if (transParam != null)
                    tp = transParam;

                if (tp != null)
                {
                    if (tp.use_imagedb_mean && sdMean == null)
                        throw new Exception("The transformer expects an image mean, yet the sdMean parameter is null!");

                    m_dataTransformer = new DataTransformer<T>(m_cuda, m_log, tp, Phase.RUN, shape.dim[1], shape.dim[2], shape.dim[3], sdMean);
                }
                else
                {
                    m_dataTransformer = null;
                }

                m_log.WriteLine("Creating run net...", true);
                m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, null, Phase.RUN);

                m_log.WriteLine("Loading weights...", true);
                loadWeights(m_net, rgWeights);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_log.Enable = true;
            }
        }

        private SimpleDatum getMeanImage(NetParameter p)
        {
            string strSrc = null;

            foreach (LayerParameter lp in p.layer)
            {
                if (lp.type == LayerParameter.LayerType.TRANSFORM)
                {
                    if (!lp.transform_param.use_imagedb_mean)
                        return null;
                }
                else if (lp.type == LayerParameter.LayerType.DATA)
                {
                    switch (lp.type)
                    {
                        case LayerParameter.LayerType.DATA:
                            strSrc = lp.data_param.source;
                            break;
                    }
                }
            }

            if (strSrc == null)
                throw new Exception("Could not find the data source in the model!");

            DatasetFactory factory = new DatasetFactory();
            SourceDescriptor sd = factory.LoadSource(strSrc);

            if (sd == null)
                throw new Exception("Could not find the data source '" + strSrc + "' in the database.");

            return factory.QueryImageMean(sd.ID);
        }

        private DatasetDescriptor findDataset(NetParameter p, DatasetDescriptor dsPrimary = null)
        {
            string strTestSrc = null;
            string strTrainSrc = null;

            foreach (LayerParameter lp in p.layer)
            {
                if (lp.type == LayerParameter.LayerType.DATA)
                {
                    string strSrc = null;

                    switch (lp.type)
                    {
                        case LayerParameter.LayerType.DATA:
                            strSrc = lp.data_param.source;
                            break;
                    }

                    foreach (NetStateRule rule in lp.include)
                    {
                        if (rule.phase == Phase.TRAIN)
                            strTrainSrc = strSrc;
                        else if (rule.phase == Phase.TEST)
                            strTestSrc = strSrc;
                    }
                }

                if (strTrainSrc != null && strTestSrc != null)
                {
                    if (dsPrimary == null || (strTrainSrc != dsPrimary.TrainingSourceName && strTestSrc != dsPrimary.TestingSourceName))
                        break;
                }
            }

            if (strTrainSrc == null || strTestSrc == null)
                return null;

            if (dsPrimary != null && (strTrainSrc == dsPrimary.TrainingSourceName && strTestSrc == dsPrimary.TestingSourceName))
                return null;

            DatasetFactory factory = new DatasetFactory();
            DatasetDescriptor ds = factory.LoadDataset(strTestSrc, strTrainSrc);

            if (ds == null)
                throw new Exception("The datset sources '" + strTestSrc + "' and '" + strTrainSrc + "' do not exist in the database - do you need to load them?");

            return ds;
        }

        private void loadWeights(Net<T> net, byte[] rgWeights)
        {
            net.LoadWeights(rgWeights, m_persist);
        }

        /// <summary>
        /// The CompareWeights method compares the weights held in two different Net objects.
        /// </summary>
        /// <param name="net1">Specifies the first Net to compare.</param>
        /// <param name="net2">Specifies the second Net to compare.</param>
        /// <returns>If the weights are identical, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool CompareWeights(Net<T> net1, Net<T> net2)
        {
            if (net1.learnable_parameters.Count != net2.learnable_parameters.Count)
            {
                m_log.WriteLine("WARNING: The number of learnable parameters differs between the two nets!");
                return false;
            }

            Blob<T> blobWork = new Blob<T>(m_cuda, m_log, false);

            try
            {
                for (int i = 0; i < net1.learnable_parameters.Count; i++)
                {
                    Blob<T> blob1 = net1.learnable_parameters[i];
                    Blob<T> blob2 = net2.learnable_parameters[i];

                    if (blob1.Name != blob2.Name)
                    {
                        m_log.WriteLine("WARNING: The name of the blobs at index " + i.ToString() + " differ: net1 - " + blob1.Name + " vs. net2 - " + blob2.Name);
                        return false;
                    }

                    if (blob1.shape_string != blob2.shape_string)
                    {
                        m_log.WriteLine("WARNING: The shape of the blobs at index " + i.ToString() + " differ: net1 - " + blob1.Name + " " + blob1.shape_string + " vs. net2 - " + blob2.Name + " " + blob2.shape_string);
                        return false;
                    }

                    blobWork.ReshapeLike(blob1);
                    m_cuda.sub(blob1.count(), blob1.gpu_data, blob2.gpu_data, blobWork.mutable_gpu_data);
                    double dfSum = Utility.ConvertVal<T>(blobWork.asum_data());
                    if (dfSum != 0)
                    {
                        m_log.WriteLine("WARNING: The data of the blobs at index " + i.ToString() + " differ: net1 - " + blob1.Name + " " + blob1.shape_string + " vs. net2 - " + blob2.Name + " " + blob2.shape_string);
                        return false;
                    }
                }
            }
            finally
            {
                blobWork.Dispose();
            }

            return true;
        }

        void m_solver_OnTestingIteration(object sender, TestingIterationArgs<T> e)
        {
            if (OnTestingIteration != null)
                OnTestingIteration(sender, e);
        }

        void m_solver_OnTrainingIteration(object sender, TrainingIterationArgs<T> e)
        {
            if (OnTrainingIteration != null)
                OnTrainingIteration(sender, e);
        }

        void m_solver_OnSnapshot(object sender, SnapshotArgs e)
        {
            if (OnSnapshot != null)
                OnSnapshot(sender, e);
        }

        /// <summary>
        /// Train the network a set number of iterations.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies number of iterations to run that override the iterations specified in the solver desctiptor.</param>
        /// <param name="nTrainingTimeLimitInMinutes">Optionally, specifies a maximum number of minutes to train.  When set to 0, this parameter is ignored and no time limit is imposed.</param>
        /// <param name="step">Optionally, specifies whether or not to single step the training on the forward pass, backward pass or both.  The default is <i>TRAIN_STEP.NONE</i> which runs the training to the maximum number of iterations specified.</param>
        /// <param name="dfLearningRateOverride">Optionally, specifies a learning rate override (default = 0 which ignores this parameter)</param>
        /// <param name="bReset">Optionally, reset the iterations to zero.</param>
        /// <remarks>
        /// Note when single stepping, no testing cycles are performed.  Currently, the single-step parameter is only suppored when running in single GPU mode.
        /// </remarks>
        public void Train(int nIterationOverride = -1, int nTrainingTimeLimitInMinutes = 0, TRAIN_STEP step = TRAIN_STEP.NONE, double dfLearningRateOverride = 0, bool bReset = false)
        {
            m_lastPhaseRun = Phase.TRAIN;

            if (nIterationOverride == -1)
                nIterationOverride = m_settings.MaximumIterationOverride;

            if (bReset)
                m_solver.Reset();

            m_solver.TrainingTimeLimitInMinutes = nTrainingTimeLimitInMinutes;
            m_solver.TrainingIterationOverride = nIterationOverride;
            m_solver.TestingIterationOverride = m_solver.TestingIterationOverride;

            if (dfLearningRateOverride > 0)
                m_solver.LearningRateOverride = dfLearningRateOverride;

            try
            {
                if (m_rgGpu.Count > 1)
                {
                    if (nTrainingTimeLimitInMinutes > 0)
                    {
                        m_log.WriteLine("You have a training time-limit of " + nTrainingTimeLimitInMinutes.ToString("N0") + " minutes.  Multi-GPU training is not supported when a training time-limit is imposed.");
                        return;
                    }

                    m_log.WriteLine("Starting multi-GPU training on GPUs: " + listToString(m_rgGpu));
                    NCCL<T> nccl = new NCCL<T>(m_cuda, m_log, m_solver, m_rgGpu[0], 0, null);
                    nccl.Run(m_rgGpu, m_solver.TrainingIterationOverride);
                }
                else
                {
                    m_solver.Solve(-1, null, null, step);
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_solver.LearningRateOverride = 0;
            }
        }

        private string listToString(List<int> rg)
        {
            string strOut = "";

            for (int i = 0; i < rg.Count; i++)
            {
                strOut += rg[i].ToString();

                if (i < rg.Count - 1)
                    strOut += ", ";
            }

            return strOut;
        }

        /// <summary>
        /// Test the network a given number of iterations.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies number of iterations to run that override the iterations specified in the solver desctiptor.</param>
        /// <returns>The accuracy value from the test is returned.</returns>
        public double Test(int nIterationOverride = -1)
        {
            m_lastPhaseRun = Phase.TEST;

            if (nIterationOverride == -1)
                nIterationOverride = m_settings.TestingIterationOverride;

            m_solver.TestingIterationOverride = nIterationOverride;

            return m_solver.TestAll();
        }

        /// <summary>
        /// Test on custom input data.
        /// </summary>
        /// <param name="customInput">Specifies the custom input data separated by ';' characters.</param>
        /// <returns>A property set containing the results is returned.</returns>
        /// <remarks>Running test many on custom data requires a MODEL dataset, where the data is queried from the model itself.
        /// </remarks>
        public PropertySet TestMany(PropertySet customInput)
        {
            if (!m_project.Dataset.IsModelData)
                throw new Exception("Custom input is only supported by MODEL based datasets!");

            m_lastPhaseRun = Phase.RUN;

            UpdateRunWeights(false, false);

            double dfThreshold = customInput.GetPropertyAsDouble("Threshold", 0.2);
            int nMax = customInput.GetPropertyAsInt("Max", 80);
            int nK = customInput.GetPropertyAsInt("K", 1);
            PropertySet res;

            if (customInput.GetPropertyAsBool("Temporal", false))
            {
                res = RunModel(customInput);
                res.SetProperty("Temporal", "True");
            }
            else
            {
                res = Run(customInput, nK, dfThreshold, nMax);
            }

            return res;
        }

        /// <summary>
        /// Test on custom input data.
        /// </summary>
        /// <param name="customInput">Specifies the custom input data separated by ';' characters.</param>
        /// <returns>A property set containing the results is returned.</returns>
        /// <remarks>Running test many on custom data requires a MODEL dataset, where the data is queried from the model itself.
        /// </remarks>
        public BlobCollection<T> TestManyEx(PropertySet customInput)
        {
            m_lastPhaseRun = Phase.RUN;

            UpdateRunWeights(false, false);

            double dfThreshold = customInput.GetPropertyAsDouble("Threshold", 0.2);
            int nMax = customInput.GetPropertyAsInt("Max", 80);
            int nK = customInput.GetPropertyAsInt("K", 1);
            
            if (customInput.GetProperty("Temporal") == "True")
                return RunModelEx(customInput);

            throw new Exception("TestManyEx currently only supports temporal testing.");
        }

        /// <summary>
        /// Test on a number of images by selecting random images from the database, running them through the Run network, and then comparing the results with the 
        /// expected results.
        /// </summary>
        /// <param name="nCount">Specifies the number of cycles to run.</param>
        /// <param name="bOnTrainingSet">Specifies on whether to select images from the training set, or when <i>false</i> the testing set of data.</param>
        /// <param name="bOnTargetSet">Optionally, specifies to test on the target dataset (if exists) as opposed to the source dataset.  The default is <i>false</i>, which tests on the default (source) dataset.</param>
        /// <param name="imgSelMethod">Optionally, specifies the image selection method (default = RANDOM).</param>
        /// <param name="nImageStartIdx">Optionally, specifies the image start index (default = 0).</param>
        /// <param name="dtImageStartTime">Optionally, specifies the image start time (default = null).  Note either the 'nImageStartIdx' or 'dtImageStartTime' may be used, but not both.</param>
        /// <param name="dfThreshold">Optionally, specifies a threshold where the winning selection must also be above the threshold in score.</param>
        /// <returns>The list of SimpleDatum and their ResultCollections (after running the model on each) is returned.</returns>
        public List<Tuple<SimpleDatum, ResultCollection>> TestMany(int nCount, bool bOnTrainingSet, bool bOnTargetSet = false, DB_ITEM_SELECTION_METHOD imgSelMethod = DB_ITEM_SELECTION_METHOD.RANDOM, int nImageStartIdx = 0, DateTime? dtImageStartTime = null, double? dfThreshold = null)
        {
            Dictionary<int, int> rgMissedThreshold = new Dictionary<int, int>();

            m_lastPhaseRun = Phase.RUN;

            UpdateRunWeights(false);

            m_log.CHECK_GT(nCount, 0, "You must select at least 1 image to train on!");

            Stopwatch sw = new Stopwatch();
            DB_LABEL_SELECTION_METHOD? lblSelMethod = null;

            if (imgSelMethod == DB_ITEM_SELECTION_METHOD.NONE)
                lblSelMethod = DB_LABEL_SELECTION_METHOD.NONE;

            Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> sel = m_db.GetSelectionMethod();
            if ((sel.Item2 & DB_ITEM_SELECTION_METHOD.BOOST) == DB_ITEM_SELECTION_METHOD.BOOST)
                imgSelMethod |= DB_ITEM_SELECTION_METHOD.BOOST;

            int nSrcId = (bOnTrainingSet) ? m_dataSet.TrainingSource.ID : m_dataSet.TestingSource.ID;
            string strSrc = (bOnTrainingSet) ? m_dataSet.TrainingSourceName : m_dataSet.TestingSourceName;
            string strSet = (bOnTrainingSet) ? "training" : "test";
            int nCorrectCount = 0;
            Dictionary<int, int> rgCorrectCounts = new Dictionary<int, int>();
            Dictionary<int, int> rgLabelTotals = new Dictionary<int, int>();
            Dictionary<int, Dictionary<int, int>> rgDetectedCounts = new Dictionary<int, Dictionary<int, int>>();

            if (bOnTargetSet && m_project.DatasetTarget != null)
            {
                nSrcId = (bOnTrainingSet) ? m_project.DatasetTarget.TrainingSource.ID : m_project.DatasetTarget.TestingSource.ID;
                strSrc = (bOnTrainingSet) ? m_project.DatasetTarget.TrainingSourceName : m_project.DatasetTarget.TestingSourceName;
                strSet = (bOnTrainingSet) ? "target training" : "target test";
            }

            sw.Start();

            m_log.WriteHeader("Test Many (" + nCount.ToString() + ") - on " + strSet + " '" + strSrc + "'");

            LabelMappingParameter labelMapping = null;
            Net<T> net = m_solver.TestingNet;
            if (net == null)
                net = m_solver.TrainingNet;

            AccuracyParameter accuracyParam = null;
            foreach (Layer<T> layer in net.layers)
            {
                if (layer.type == LayerParameter.LayerType.LABELMAPPING)
                {
                    labelMapping = layer.layer_param.labelmapping_param;
                    break;
                }
                else if (layer.type == LayerParameter.LayerType.ACCURACY ||
                    layer.type == LayerParameter.LayerType.ACCURACY_DECODE ||
                    layer.type == LayerParameter.LayerType.ACCURACY_ENCODING)
                {
                    accuracyParam = layer.layer_param.accuracy_param;
                }
            }

            if (nImageStartIdx < 0)
                nImageStartIdx = 0;

            List<SimpleDatum> rgImg = null;
            if (dtImageStartTime.HasValue && dtImageStartTime.Value > DateTime.MinValue)
            {
                m_log.WriteLine("INFO: Starting test many at images with time " + dtImageStartTime.Value.ToString() + " or later...");
                rgImg = m_db.GetItemsFromTime(nSrcId, dtImageStartTime.Value, nCount);
                if (nCount > rgImg.Count)
                    nCount = rgImg.Count;

                if (nCount == 0)
                    throw new Exception("No images found after time '" + dtImageStartTime.Value.ToString() + "'.  Make sure to use the LOAD_ALL image loading method when running TestMany after a specified time.");
            }

            List<Tuple<SimpleDatum, ResultCollection>> rgrgResults = new List<Tuple<SimpleDatum, ResultCollection>>();
            int nTotalCount = 0;
            int nMidPoint = 0;
            bool bPad = true;
            int nImgCount = nCount;

            Blob<T> blobData = null;

            try
            {
                SimpleDatum sd = null;
                List<int> rgOriginalRunNetInputShape = null;

                if (m_net.input_blobs != null && m_net.input_blobs.Count > 0)
                    rgOriginalRunNetInputShape = Utility.Clone<int>(m_net.input_blobs[0].shape());

                for (int i = 0; i < nCount; i++)
                {
                    if (m_evtCancel.WaitOne(0))
                    {
                        m_log.WriteLine("Test Many aborted!");
                        return null;
                    }


                    sd = (rgImg != null) ? rgImg[i] : m_db.QueryItem(nSrcId, nImageStartIdx + i, lblSelMethod, imgSelMethod, null, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);

                    if (sd.Height != m_dataSet.TrainingSource.Height || sd.Width != m_dataSet.TestingSource.Width)
                    {
                        m_log.WriteLine("WARNING: Image size mismatch!  Current image size " + sd.Width.ToString() + " x " + sd.Height.ToString() + " does not match the dataset image size " + m_dataSet.TrainingSource.Width.ToString() + " x " + m_dataSet.TrainingSource.Height.ToString() + "!");
                        continue;
                    }

                    m_dataTransformer.TransformLabel(sd);

                    if (!sd.GetDataValid(false))
                    {
                        Trace.WriteLine("You should not be here.");
                        throw new Exception("NO DATA!");
                    }

                    // Create blob masks images (when enabled) during the data transform.
                    blobData = CreateDataBlob(sd, blobData, bPad);
                    
                    List<int> rgIgnoreLabels = null;
                    if (accuracyParam != null && accuracyParam.ignore_labels != null)
                        rgIgnoreLabels = accuracyParam.ignore_labels;
                    
                    List<ResultCollection> rgrgResults1 = Run(blobData, false, false, int.MaxValue, rgIgnoreLabels);
                    ResultCollection rgResults = rgrgResults1[0];

                    // If masked, mask the sd to match the actual input in the blobData.
                    m_dataTransformer.MaskImage(sd);
                    rgrgResults.Add(new Tuple<SimpleDatum, ResultCollection>(sd, rgResults));

                    if (rgResults.ResultType == ResultCollection.RESULT_TYPE.MULTIBOX)
                    {
                        Dictionary<int, List<Result>> rgLabeledResults = new Dictionary<int, List<Result>>();
                        Dictionary<int, int> rgLabeledOrder = new Dictionary<int, int>();

                        int nIdx = 0;
                        foreach (Result result in rgResults.ResultsSorted)
                        {
                            if (!rgLabeledResults.ContainsKey(result.Label))
                            {
                                rgLabeledResults.Add(result.Label, new List<Result>());
                                rgLabeledOrder.Add(result.Label, nIdx);
                                nIdx++;
                            }

                            rgLabeledResults[result.Label].Add(result);
                        }

                        List<Tuple<int, List<Result>>> rgBestResults = new List<Tuple<int, List<Result>>>();
                        List<int> rgDetectedLabels = rgLabeledOrder.OrderBy(p => p.Value).Select(p => p.Key).ToList();

                        if (sd.annotation_group != null)
                        {
                            rgDetectedLabels = rgDetectedLabels.Take(sd.annotation_group.Count).ToList();

                            for (int j = 0; j < sd.annotation_group.Count; j++)
                            {
                                int nExpectedLabel = sd.annotation_group[j].group_label;

                                if (!rgCorrectCounts.ContainsKey(nExpectedLabel))
                                    rgCorrectCounts.Add(nExpectedLabel, 0);

                                if (!rgLabelTotals.ContainsKey(nExpectedLabel))
                                    rgLabelTotals.Add(nExpectedLabel, 1);
                                else
                                    rgLabelTotals[nExpectedLabel]++;

                                if (rgDetectedLabels.Contains(nExpectedLabel))
                                {
                                    rgCorrectCounts[nExpectedLabel]++;
                                    nCorrectCount++;
                                }

                                nTotalCount++;
                            }
                        }
                        else
                        {
                            m_log.WriteLine("WARNING: No annotation data found in image with ID = " + sd.ImageID.ToString());
                        }
                    }
                    else
                    {
                        int nDetectedLabel = rgResults.DetectedLabel;
                        int nExpectedLabel = sd.Label;

                        if (!dfThreshold.HasValue || rgResults.DetectedLabelOutput >= dfThreshold.Value)
                        {
                            if (rgResults.ResultsOriginal.Count % 2 != 0)
                                nMidPoint = (int)Math.Floor(rgResults.ResultsOriginal.Count / 2.0);


                            if (labelMapping != null)
                            {
                                if (m_dataTransformer.param.label_mapping.Active)
                                    m_log.FAIL("You can use either the LabelMappingLayer or the DataTransformer label_mapping, but not both!");

                                nExpectedLabel = labelMapping.MapLabel(nExpectedLabel);
                            }

                            if (!rgCorrectCounts.ContainsKey(nExpectedLabel))
                                rgCorrectCounts.Add(nExpectedLabel, 0);

                            if (!rgLabelTotals.ContainsKey(nExpectedLabel))
                                rgLabelTotals.Add(nExpectedLabel, 1);
                            else
                                rgLabelTotals[nExpectedLabel]++;

                            if (nExpectedLabel == nDetectedLabel)
                            {
                                nCorrectCount++;
                                rgCorrectCounts[nExpectedLabel]++;
                            }

                            if (!rgDetectedCounts.ContainsKey(nExpectedLabel))
                                rgDetectedCounts.Add(nExpectedLabel, new Dictionary<int, int>());

                            if (!rgDetectedCounts[nExpectedLabel].ContainsKey(nDetectedLabel))
                                rgDetectedCounts[nExpectedLabel].Add(nDetectedLabel, 0);

                            rgDetectedCounts[nExpectedLabel][nDetectedLabel]++;

                            nTotalCount++;
                        }
                        else
                        {
                            if (!rgMissedThreshold.ContainsKey(nExpectedLabel))
                                rgMissedThreshold.Add(nExpectedLabel, 0);

                            rgMissedThreshold[nExpectedLabel]++;
                        }
                    }

                    double dfPct = ((double)i / (double)nCount);
                    m_log.Progress = dfPct;

                    if (sw.ElapsedMilliseconds > 1000)
                    {
                        m_log.WriteLine("processing test many at " + dfPct.ToString("P"));
                        sw.Stop();
                        sw.Reset();
                        sw.Start();
                    }
                }

                // Resize inputs back to unpaded.
                if (rgOriginalRunNetInputShape != null)
                {
                    if (m_net.input_blobs != null && m_net.input_blobs.Count > 0)
                        m_net.input_blobs[0].Reshape(rgOriginalRunNetInputShape);
                }
            }
            finally
            {
                if (blobData != null)
                {
                    blobData.Dispose();
                    blobData = null;
                }
            }

            double dfCorrectPct = (nTotalCount == 0) ? 0 : ((double)nCorrectCount / (double)nTotalCount);

            m_log.WriteLine("Test Many Completed.");
            m_log.WriteLine(" " + dfCorrectPct.ToString("P") + " correct detections.");
            m_log.WriteLine(" " + (nTotalCount - nCorrectCount).ToString("N") + " incorrect detections.");

            foreach (KeyValuePair<int, int> kv in rgCorrectCounts.OrderBy(p => p.Key).ToList())
            {
                nCount = 0;

                foreach (KeyValuePair<int, int> kv1 in rgLabelTotals)
                {
                    if (kv1.Key == kv.Key)
                    {
                        nCount = kv1.Value;
                        break;
                    }
                }

                if (nCount > 0)
                {
                    string strSecondDetection = "";
                    if (rgDetectedCounts.ContainsKey(kv.Key))
                    {
                        List<KeyValuePair<int, int>> rgDetectedCountsSorted = rgDetectedCounts[kv.Key].OrderByDescending(p => p.Value).ToList();
                        if (rgDetectedCountsSorted.Count > 1)
                        {
                            strSecondDetection = " (secondary detections: " + rgDetectedCountsSorted[1].Key.ToString();

                            if (rgDetectedCountsSorted.Count > 2)
                                strSecondDetection += " and " + rgDetectedCountsSorted[2].Key.ToString();

                            strSecondDetection += ")";
                        }
                    }
                    
                    dfCorrectPct = ((double)kv.Value / (double)nCount);
                    m_log.WriteLine("Label #" + kv.Key.ToString() + " had " + dfCorrectPct.ToString("P") + " correct detections out of " + nCount.ToString("N0") + " items with this label." + strSecondDetection);
                }
            }

            if (nMidPoint > 0)
            {
                int nTotalBelow = 0;
                int nCorrectBelow = 0;
                int nTotalAbove = 0;
                int nCorrectAbove = 0;
                int nTotalBelowAndAbove = 0;
                int nCorrectBelowAndAbove = 0;

                List<KeyValuePair<int, int>> rgLabelTotalsList = rgLabelTotals.OrderBy(p => p.Key).ToList();
                List<KeyValuePair<int, int>> rgCorrectCountsList = rgCorrectCounts.OrderBy(p => p.Key).ToList();

                for (int i = 0; i < rgLabelTotalsList.Count; i++)
                {
                    if (i < nMidPoint)
                    {
                        nTotalBelow += rgLabelTotalsList[i].Value;
                        nCorrectBelow += rgCorrectCountsList[i].Value;
                        nTotalBelowAndAbove += rgLabelTotalsList[i].Value;
                        nCorrectBelowAndAbove += rgCorrectCountsList[i].Value;
                    }
                    else if (i > nMidPoint)
                    {
                        nTotalAbove += rgLabelTotalsList[i].Value;
                        nCorrectAbove += rgCorrectCountsList[i].Value;
                        nTotalBelowAndAbove += rgLabelTotalsList[i].Value;
                        nCorrectBelowAndAbove += rgCorrectCountsList[i].Value;
                    }
                }

                dfCorrectPct = (nTotalBelow == 0) ? 0 : nCorrectBelow / (double)nTotalBelow;
                m_log.WriteLine("Correct below midpoint of " + nMidPoint.ToString() + " = " + dfCorrectPct.ToString("P"));
                dfCorrectPct = (nTotalAbove == 0) ? 0 : nCorrectAbove / (double)nTotalAbove;
                m_log.WriteLine("Correct above midpoint of " + nMidPoint.ToString() + " = " + dfCorrectPct.ToString("P"));
                dfCorrectPct = (nTotalBelowAndAbove == 0) ? 0 : nCorrectBelowAndAbove / (double)nTotalBelowAndAbove;
                m_log.WriteLine("Correct below and above midpoint of " + nMidPoint.ToString() + " = " + dfCorrectPct.ToString("P"));
            }

            if (rgMissedThreshold.Count > 0)
            {
                m_log.WriteLine("---Missed Threshold Items---");

                int nTotal = 0;
                foreach (KeyValuePair<int, int> kv in rgMissedThreshold)
                {
                    m_log.WriteLine("Expected Label " + kv.Key.ToString() + ": " + kv.Value.ToString() + " items missed threshold (" + ((double)kv.Value/nImgCount).ToString("P") + ").");
                    nTotal += kv.Value;
                }

                m_log.WriteLine("A total of " + nTotal.ToString() + " items did not meet the threshold of " + dfThreshold.Value.ToString() + ", (" + ((double)nTotal / nImgCount).ToString("P") + ")");
            }

            return rgrgResults;
        }

        /// <summary>
        /// Run on a given image in the MyCaffeImageDatabase based on its image index.
        /// </summary>
        /// <param name="nImageIdx">Specifies the image index.</param>
        /// <param name="bPad">PAdd the input with an extra input image.</param>
        /// <returns>The result of the run is returned.</returns>
        public ResultCollection Run(int nImageIdx, bool bPad = true)
        {
            SimpleDatum sd = m_db.QueryItem(m_dataSet.TrainingSource.ID, nImageIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, null, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);
            m_dataTransformer.TransformLabel(sd);
            return Run(sd, true, bPad);
        }

        /// <summary>
        /// Run on a set of images in the MyCaffeImageDatabase based on their image indexes.
        /// </summary>
        /// <param name="rgImageIdx">Specifies a list of image indexes.</param>
        /// <param name="blob">Specifies a work blob.</param>
        /// <returns>A list of results from the run is returned - one result per image.</returns>
        public List<ResultCollection> Run(List<int> rgImageIdx, ref Blob<T> blob)
        {
            List<SimpleDatum> rgSd = new List<SimpleDatum>();

            foreach (int nImageIdx in rgImageIdx)
            {
                SimpleDatum sd = m_db.QueryItem(m_dataSet.TrainingSource.ID, nImageIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, null, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);
                m_dataTransformer.TransformLabel(sd);
                rgSd.Add(sd);
            }

            return Run(rgSd, ref blob, false, int.MaxValue);
        }

        /// <summary>
        /// Run on a set of images in the MyCaffeImageDatabase based on their image indexes.
        /// </summary>
        /// <param name="rgImageIdx">Specifies a list of image indexes.</param>
        /// <returns>A list of results from the run is returned - one result per image.</returns>
        public List<ResultCollection> Run(List<int> rgImageIdx)
        {
            List<SimpleDatum> rgSd = new List<SimpleDatum>();

            if (m_dataSet == null)
                throw new Exception("Running on indexes requires a full Load that includes loading the dataset.");

            foreach (int nImageIdx in rgImageIdx)
            {
                SimpleDatum sd = m_db.QueryItem(m_dataSet.TrainingSource.ID, nImageIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, null, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);
                m_dataTransformer.TransformLabel(sd);
                rgSd.Add(sd);
            }

            Blob<T> blob = null;
            List<ResultCollection> rgRes = Run(rgSd, ref blob);

            if (blob != null)
                blob.Dispose();

            return rgRes;
        }

        private int getCount(List<int> rg)
        {
            int nCount = 1;

            foreach (int nDim in rg)
            {
                nCount *= nDim;
            }

            return nCount;
        }

        /// <summary>
        /// Create a data blob from a SimpleDatum by transforming the data and placing the results in the blob returned.
        /// </summary>
        /// <param name="d">Specifies the datum to load into the blob.</param>
        /// <param name="blob">Optionally, specifies a blob to use instead of creating a new one.</param>
        /// <param name="bPad">Optionally, pad the blob with an extra item (default = false).</param>
        /// <returns>The data blob containing the transformed data is returned.</returns>
        public Blob<T> CreateDataBlob(SimpleDatum d, Blob<T> blob = null, bool bPad = true)
        {
            if (m_dataTransformer == null)
            {
                if (blob != null)
                    blob.SetData(d, true);
                else
                    blob = new Blob<T>(m_cuda, m_log, d, true);
            }
            else
            {
                if (blob == null)
                    blob = new Blob<T>(m_cuda, m_log);

                Datum datum = new Datum(d);

                List<int> rgShape = m_dataTransformer.InferBlobShape(datum);

                if (bPad)
                    rgShape[0] = 2;

                int nCount = getCount(rgShape);
                blob.Reshape(rgShape);
                blob.Padded = bPad;

                if (m_rgRunData == null || m_rgRunData.Length != nCount)
                    m_rgRunData = new T[nCount];

                T[] rgData = m_dataTransformer.Transform(datum);
                Array.Copy(rgData, 0, m_rgRunData, 0, rgData.Length);

                blob.mutable_cpu_data = m_rgRunData;

                m_dataTransformer.SetRange(blob);
            }

            return blob;
        }

        /// <summary>
        /// Run on a given Datum. 
        /// </summary>
        /// <param name="d">Specifies the Datum to run.</param>
        /// <param name="bSort">Specifies whether or not to sor the results.</param>
        /// <param name="bUseSolverNet">Optionally, specifies whether or not to use the training net vs. the run net.</param>
        /// <param name="bPad">Optionally, specifies to pad the data with a dummy trailing item (default = true).</param>
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(SimpleDatum d, bool bSort, bool bUseSolverNet, bool bPad = true)
        {
            if (m_net == null)
                throw new Exception("The Run net has not been created!");

            ResultCollection result = null;
            Blob<T> blob = null;

            try
            {
                blob = CreateDataBlob(d, null, bPad);
                BlobCollection<T> colBottom = new BlobCollection<T>() { blob };
                double dfLoss = 0;

                BlobCollection<T> colResults;
                LayerParameter.LayerType lastLayerType;

                if (bUseSolverNet)
                {
                    lastLayerType = m_solver.TrainingNet.layers[m_solver.TrainingNet.layers.Count - 1].type;
                    colResults = m_solver.TrainingNet.Forward(colBottom, out dfLoss, bPad);
                }
                else
                {
                    lastLayerType = m_net.layers[m_net.layers.Count - 1].type;
                    colResults = m_net.Forward(colBottom, out dfLoss, bPad);
                }

                if (blob.Padded)
                {
                    List<int> rgShape = Utility.Clone<int>(colResults[0].shape());
                    rgShape[0]--;
                    if (rgShape[0] <= 0)
                        rgShape[0] = 1;
                    colResults[0].Reshape(rgShape);
                }

                List<Result> rgResults = new List<Result>();
                float[] rgData = Utility.ConvertVecF<T>(colResults[0].update_cpu_data());

                if (colResults[0].type == BLOB_TYPE.MULTIBBOX)
                {
                    int nNum = rgData.Length / 7;

                    for (int n = 0; n < nNum; n++)
                    {
                        int i = (int)rgData[(n * 7)];
                        int nLabel = (int)rgData[(n * 7) + 1];
                        double dfScore = rgData[(n * 7) + 2];
                        double[] rgExtra = new double[4];
                        rgExtra[0] = rgData[(n * 7) + 3]; // xmin
                        rgExtra[1] = rgData[(n * 7) + 4]; // ymin
                        rgExtra[2] = rgData[(n * 7) + 5]; // xmax
                        rgExtra[3] = rgData[(n * 7) + 6]; // ymax

                        rgResults.Add(new Result(nLabel, dfScore, rgExtra));
                    }
                }
                else
                {
                    for (int i = 0; i < rgData.Length; i++)
                    {
                        double dfProb = rgData[i];
                        rgResults.Add(new Result(i, dfProb));
                    }
                }

                result = new ResultCollection(rgResults, lastLayerType);
                if (m_db != null && m_db.GetVersion() != DB_VERSION.TEMPORAL)
                    result.SetLabels(((IXImageDatabaseBase)m_db).GetLabels(m_dataSet.TrainingSource.ID));
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                if (blob != null)
                    blob.Dispose();
            }

            return result;
        }

        /// <summary>
        /// Run on a given list of Datum. 
        /// </summary>
        /// <param name="rgSd">Specifies the list of Datum to run.</param>
        /// <param name="blob">Specifies a work blob.</param>
        /// <param name="bUseSolverNet">Optionally, specifies whether or not to use the training net vs. the run net.</param>
        /// <param name="nMax">Optionally, specifies a maximum number of SimpleDatums to process (default = int.MaxValue).</param>
        /// <returns>A list of results of the run are returned.</returns>
        public List<ResultCollection> Run(List<SimpleDatum> rgSd, ref Blob<T> blob, bool bUseSolverNet = false, int nMax = int.MaxValue)
        {
            m_log.CHECK(m_dataTransformer != null, "The data transformer is not initialized!");

            if (m_net == null)
                throw new Exception("The Run net has not been created!");

            List<ResultCollection> rgFinalResults = new List<ResultCollection>();
            int nBatchSize = rgSd.Count;
            int nChannels = rgSd[0].Channels;
            int nHeight = rgSd[0].Height;
            int nWidth = rgSd[0].Width;
            List<T> rgDataInput = new List<T>();

            if (blob == null)
                blob = new common.Blob<T>(m_cuda, m_log, nBatchSize, nChannels, nHeight, nWidth, false);

            int nCount = 0;
            for (int i=0; i<rgSd.Count && i < nMax; i++)
            {
                rgDataInput.AddRange(m_dataTransformer.Transform(rgSd[i]));
                nCount++;
            }

            blob.Reshape(nCount, nChannels, nHeight, nWidth);
            blob.mutable_cpu_data = rgDataInput.ToArray();
            m_dataTransformer.SetRange(blob);

            BlobCollection<T> colBottom = new BlobCollection<T>() { blob };
            double dfLoss = 0;

            BlobCollection<T> colResults;
            LayerParameter.LayerType lastLayerType;

            if (bUseSolverNet)
            {
                lastLayerType = m_solver.TrainingNet.layers[m_net.layers.Count - 1].type;
                m_solver.TrainingNet.SetEnablePassthrough(true);
                colResults = m_solver.TrainingNet.Forward(colBottom, out dfLoss);
                m_solver.TrainingNet.SetEnablePassthrough(false);
            }
            else
            {
                lastLayerType = m_net.layers[m_net.layers.Count - 1].type;
                colResults = m_net.Forward(colBottom, out dfLoss, true);
            }

            T[] rgDataOutput = colResults[0].update_cpu_data();
            int nOutputCount = rgDataOutput.Length / rgSd.Count;

            for (int i = 0; i < rgSd.Count && i < nMax; i++)
            {
                List<Result> rgResults = new List<Result>();

                for (int j = 0; j < nOutputCount; j++)
                {
                    int nIdx = i * nOutputCount + j;
                    double dfProb = (double)Convert.ChangeType(rgDataOutput[nIdx], typeof(double));
                    rgResults.Add(new Result(j, dfProb));
                }

                ResultCollection result = new ResultCollection(rgResults, lastLayerType);

                if (m_db != null && m_dataSet != null && m_db.GetVersion() != DB_VERSION.TEMPORAL)
                    result.SetLabels(((IXImageDatabaseBase)m_db).GetLabels(m_dataSet.TrainingSource.ID));

                rgFinalResults.Add(result);
            }

            return rgFinalResults;
        }

        /// <summary>
        /// Run on a Blob of data. 
        /// </summary>
        /// <param name="blob">Specifies the blob of data.</param>
        /// <param name="bSort">Optionally, specifies whether or not to sor the results.</param>
        /// <param name="bUseSolverNet">Optionally, specifies whether or not to use the training net vs. the run net.</param>
        /// <param name="nMax">Optionally, specifies a maximum number of SimpleDatums to process (default = int.MaxValue).</param>
        /// <param name="rgIgnoreLabels">Optionally, specifies a set of labels to ignore.</param>
        /// <returns>A list of results of the run are returned.</returns>
        public List<ResultCollection> Run(Blob<T> blob, bool bSort = true, bool bUseSolverNet = false, int nMax = int.MaxValue, List<int> rgIgnoreLabels = null)
        {
            m_log.CHECK(m_dataTransformer != null, "The data transformer is not initialized!");

            if (m_net == null)
                throw new Exception("The Run net has not been created!");

            if (m_dataSet == null && (m_loadToRunShape == null || m_loadToRunShape.dim.Count < 4))
                throw new Exception("Cannot determine the blob shape, you must either load with a database, or use LoadToRun before calling Run with a Blob.  When using LoadToRun, the shape must have at least 4 dimensions.");

            List<ResultCollection> rgFinalResults = new List<ResultCollection>();
            int nBatchSize = blob.num;
            int nChannels = (m_dataSet != null) ? m_dataSet.TestingSource.Channels : m_loadToRunShape.dim[1];
            if (blob.channels != nChannels)
                throw new Exception("The blob channels must match those of the testing dataset which has channels = " + m_dataSet.TestingSource.Channels.ToString());

            int nHeight = (m_dataSet != null) ? m_dataSet.TestingSource.Height : m_loadToRunShape.dim[2];
            int nWidth = (m_dataSet != null) ? m_dataSet.TestingSource.Width : m_loadToRunShape.dim[3];

            if (m_dataTransformer.param.resize_param != null && m_dataTransformer.param.resize_param.Active)
            {
                List<int> rgShape = m_dataTransformer.InferBlobShape(nChannels, nWidth, nHeight);
                nHeight = rgShape[2];
                nWidth = rgShape[3];
            }

            if (blob.height != nHeight)
                throw new Exception("The blob height must match those of the testing dataset which has height = " + nHeight.ToString());

            if (blob.width != nWidth)
                throw new Exception("The blob width must match those of the testing dataset which as width = " + nWidth.ToString());

            m_dataTransformer.SetRange(blob);

            BlobCollection<T> colBottom = new BlobCollection<T>() { blob };
            double dfLoss = 0;

            BlobCollection<T> colResults;
            LayerParameter.LayerType lastLayerType;

            if (bUseSolverNet)
            {
                lastLayerType = m_solver.TrainingNet.layers[m_net.layers.Count - 1].type;
                m_solver.TrainingNet.SetEnablePassthrough(true);
                colResults = m_solver.TrainingNet.Forward(colBottom, out dfLoss);
                m_solver.TrainingNet.SetEnablePassthrough(false);
            }
            else
            {
                lastLayerType = m_net.layers[m_net.layers.Count - 1].type;
                colResults = m_net.Forward(colBottom, out dfLoss, true);
            }

            float[] rgData = Utility.ConvertVecF<T>(colResults[0].update_cpu_data());
            int nOutputCount = rgData.Length / blob.num;

            int nNum = blob.num;
            if (blob.Padded)
                nNum--;

            ResultCollection.RESULT_TYPE resType = ResultCollection.GetResultType(lastLayerType);

            for (int n = 0; n < nNum && n < nMax; n++)
            {
                List<Result> rgResults = new List<Result>();

                if (colResults[0].type == BLOB_TYPE.MULTIBBOX)
                {
                    int i = (int)rgData[(n * 7)];
                    int nLabel = (int)rgData[(n * 7) + 1];
                    double dfScore = rgData[(n * 7) + 2];
                    double[] rgExtra = new double[4];
                    rgExtra[0] = rgData[(n * 7) + 3]; // xmin
                    rgExtra[1] = rgData[(n * 7) + 4]; // ymin
                    rgExtra[2] = rgData[(n * 7) + 5]; // xmax
                    rgExtra[3] = rgData[(n * 7) + 6]; // ymax

                    rgResults.Add(new Result(nLabel, dfScore, rgExtra));
                }
                else
                {
                    for (int j = 0; j < nOutputCount; j++)
                    {
                        int nIdx = n * nOutputCount + j;
                        double dfProb = rgData[nIdx];

                        if (rgIgnoreLabels != null && rgIgnoreLabels.Contains(j))
                        {
                            if (resType == ResultCollection.RESULT_TYPE.DISTANCES)
                                dfProb = double.MaxValue;
                            else
                                dfProb = 0;
                        }

                        rgResults.Add(new Result(j, dfProb));
                    }
                }

                ResultCollection result = new ResultCollection(rgResults, lastLayerType);
                if (m_db != null && m_db.GetVersion() != DB_VERSION.TEMPORAL)
                    result.SetLabels(((IXImageDatabaseBase)m_db).GetLabels(m_dataSet.TrainingSource.ID));

                rgFinalResults.Add(result);
            }

            return rgFinalResults;
        }

        /// <summary>
        /// Run on a given bitmap image.
        /// </summary>
        /// <remarks>
        /// This method does not use the MyCaffeImageDatabase.
        /// </remarks>
        /// <param name="img">Specifies the input image.</param>
        /// <param name="bSort">Specifies whether or not to sort the results.</param>
        /// <param name="bPad">Optionally, specifies to pad the input by 1.</param>
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(Bitmap img, bool bSort = true, bool bPad = true)
        {
            if (m_net == null)
                throw new Exception("The Run net has not been created!");

            int nChannels = m_inputShape.dim[1];

            if (typeof(T) == typeof(double))
                return Run(ImageData.GetImageDataD(img, nChannels, false, -1), bSort, bPad);
            else
                return Run(ImageData.GetImageDataF(img, nChannels, false, -1), bSort, bPad);
        }

        /// <summary>
        /// Run on a given Datum. 
        /// </summary>
        /// <param name="d">Specifies the Datum to run.</param>
        /// <param name="bSort">Specifies whether or not to sort the results.</param>
        /// <param name="bPad">Optionally, specifies to pad the data with a dummy item and reshape the net (default = false).</param>
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(SimpleDatum d, bool bSort = true, bool bPad = true)
        {
            return Run(d, bSort, false, bPad);
        }

        /// <summary>
        /// Run the model using data from the model itself - requires a Data layer with the RUN phase.
        /// </summary>
        /// <param name="customInput">Specifies custom inputs.  Properties used: 'Phase' specifies the phase for which to run the model (e.g. which net to use), when missing, the default run net is used.</param>
        /// <returns>The results are returned in a property set, where each blob is stored as a byte array packed with the 'float' values from each blob.</returns>
        public PropertySet RunModel(PropertySet customInput)
        {
            Net<T> net = m_net;
            Phase phase = Phase.TRAIN;

            if (customInput != null)
            {
                string strPhase = customInput.GetProperty("Phase", false);
                if (!string.IsNullOrEmpty(strPhase))
                {
                    if (strPhase == Phase.TRAIN.ToString())
                        phase = Phase.TRAIN;
                    else if (strPhase == Phase.TEST.ToString())
                        phase = Phase.TEST;
                    else if (strPhase == Phase.RUN.ToString())
                        phase = Phase.RUN;
                }
            }

            m_log.WriteLine("INFO: Running TestMany with the " + phase.ToString() + " phase.");
            net = GetInternalNet(phase);

            int nItemIndexVerify = customInput.GetPropertyAsInt("ItemIndexVerify", 1);
            int nDataStartIndex = customInput.GetPropertyAsInt("DataStartIndex", -1);
            if (nDataStartIndex >= 0)
            {
                // Set the value index stating point for the forward pass.  A batch of items will
                // be loaded starting at this index.  This setting allows external users to
                // specifiy the exact starting point for the forward pass.
                Layer<T> dataLayer = net.FindLayer(LayerParameter.LayerType.DATA_TEMPORAL, null);
                if (dataLayer != null)
                {
                    IXTemporalDatabaseBase db = m_db as IXTemporalDatabaseBase;
                    if (db != null)
                    {
                        phase = dataLayer.layer_param.data_temporal_param.forced_phase.GetValueOrDefault(phase);
                        string strSrc = dataLayer.layer_param.data_temporal_param.source;
                        if (!strSrc.EndsWith("." + phase.ToString().ToLower()))
                            strSrc += "." + phase.ToString().ToLower();

                        try
                        {
                            if (!db.IsValueIndexValid(strSrc, nItemIndexVerify, nDataStartIndex, (int)(dataLayer.layer_param.data_temporal_param.num_historical_steps + dataLayer.layer_param.data_temporal_param.num_future_steps)))
                            {
                                PropertySet res1 = new PropertySet();
                                res1.SetProperty("Error", "Out of data, reset the 'DataStartIndex' to 0.");
                                return res1;
                            }
                        }
                        catch (Exception)
                        {
                            PropertySet res1 = new PropertySet();
                            res1.SetProperty("Error", "Could not find the source '" + strSrc + "'.");
                            return res1;
                        }

                        dataLayer.layer_param.data_temporal_param.value_start_index_override = nDataStartIndex;
                    }
                }
            }

            if (nDataStartIndex == -1 || nDataStartIndex == 80)
                Trace.Write("Found it.");

            BlobCollection<T> colTop = net.Forward();
            BlobCollection<T> colRes = null;

            string strPredictionBlob = customInput.GetProperty("PredictionBlob");
            string strLabelBlob = customInput.GetProperty("LabelBlob");
            string strTimeBlob = customInput.GetProperty("TimeBlob");
            string strIDBlob = customInput.GetProperty("IDBlob");
            DateTime? dtLast = null;

            if (!string.IsNullOrEmpty(strPredictionBlob) || !string.IsNullOrEmpty(strLabelBlob))
            {
                colRes = new BlobCollection<T>();

                if (!string.IsNullOrEmpty(strPredictionBlob))
                {
                    Blob<T> blob = net.FindBlob(strPredictionBlob);
                    if (blob != null)
                        colRes.Add(blob);
                }

                if (!string.IsNullOrEmpty(strLabelBlob))
                {
                    Blob<T> blob = net.FindBlob(strLabelBlob);
                    if (blob != null)
                        colRes.Add(blob);
                }

                if (!string.IsNullOrEmpty(strTimeBlob))
                {
                    Blob<T> blob = net.FindBlob(strTimeBlob);
                    if (blob != null)
                    {
                        colRes.Add(blob);
                        if (blob.Tag != null && blob.Tag is DateTime)
                            dtLast = (DateTime)blob.Tag;
                    }
                }

                if (!string.IsNullOrEmpty(strIDBlob))
                {
                    Blob<T> blob = net.FindBlob(strIDBlob);
                    if (blob != null)
                        colRes.Add(blob);
                }
            }

            if (colRes == null)
                colRes = colTop;

            PropertySet res = new PropertySet();

            foreach (Blob<T> blob in colRes)
            {
                byte[] rgBytes = blob.ToByteArray();
                res.SetPropertyBlob(blob.Name, rgBytes);
                res.SetPropertyInt(blob.Name, (int)blob.type);
            }

            if (dtLast.HasValue)
                res.SetProperty("Time", dtLast.Value.ToString("yyyy-MM-dd HH:mm:ss"));

            return res;
        }

        /// <summary>
        /// Run the model using data from the model itself - requires a Data layer with the RUN phase.
        /// </summary>
        /// <param name="customInput">Specifies custom inputs.  Properties used: 'Phase' specifies the phase for which to run the model (e.g. which net to use), when missing, the default run net is used.</param>
        /// <returns>The results are returned in a property set, where each blob is stored as a byte array packed with the 'float' values from each blob.</returns>
        public BlobCollection<T> RunModelEx(PropertySet customInput)
        {
            Net<T> net = m_net;
            Phase phase = Phase.TRAIN;

            if (customInput != null)
            {
                string strPhase = customInput.GetProperty("Phase", false);
                if (!string.IsNullOrEmpty(strPhase))
                {
                    if (strPhase == Phase.TRAIN.ToString())
                        phase = Phase.TRAIN;
                    else if (strPhase == Phase.TEST.ToString())
                        phase = Phase.TEST;
                    else if (strPhase == Phase.RUN.ToString())
                        phase = Phase.RUN;
                }
            }

            m_log.WriteLine("INFO: Running TestMany with the " + phase.ToString() + " phase.");
            net = GetInternalNet(phase);

            return net.Forward();
        }

        /// <summary>
        /// Run the model on custom input data.
        /// </summary>
        /// <param name="customInput">Specifies the custom input data.</param>
        /// <param name="nK">Optionally, specifies the number of items to use in the search where nK=1 runs a greedy search and any values > 1 run a beam search with that width.</param>
        /// <param name="dfThreshold">Specifies the threshold where detected items with probabilities less than the threshold are ignored (default = 0.01).</param>
        /// <param name="nMax">Optionally, specifies the maximum number of outputs (default = 80).</param>
        /// <param name="bBeamSearch">Optionally, specifies to run a Beam Search.</param>
        /// <returns>The results are returned as in a property set.</returns>
        public PropertySet Run(PropertySet customInput, int nK = 1, double dfThreshold = 0.01, int nMax = 80, bool bBeamSearch = false)
        {
            m_log.CHECK_GE(nK, 1, "The K must be >= 1!");

            BlobCollection<T> colBottom = null;
            Layer<T> layerInput = null;
            string strInput = customInput.GetProperty("InputData");
            string[] rgstrInput = strInput.Split('|');
            List<string> rgstrOutput = new List<string>();
            int nSeqLen = nMax;

            foreach (string strInput1 in rgstrInput)
            {
                PropertySet input = new PropertySet("InputData=" + strInput1);
                string strOut = "\n";

                if (!bBeamSearch)
                {
                    foreach (Layer<T> layer in m_net.layers)
                    {
                        int nSeqLen1;
                        colBottom = layer.PreProcessInput(input, out nSeqLen1, colBottom);
                        if (colBottom != null)
                        {
                            layerInput = layer;
                            nSeqLen = nSeqLen1;
                            break;
                        }
                    }

                    if (colBottom == null)
                        throw new Exception("At least one layer must support the 'PreprocessInput' method!");

                    double dfLoss;
                    int nAxis = 1;
                    BlobCollection<T> colTop = m_net.Forward(colBottom, out dfLoss, layerInput.SupportsPostProcessingLogits);
                    Blob<T> blobTop = colTop[0];
                    Layer<T> softmax = null;

                    if (m_net.layers[m_net.layers.Count - 1].layer_param.type == LayerParameter.LayerType.SOFTMAX)
                    {
                        softmax = m_net.layers[m_net.layers.Count - 1];
                        nAxis = softmax.layer_param.softmax_param.axis;
                    }
                    else if (m_net.layers[m_net.layers.Count - 2].layer_param.type == LayerParameter.LayerType.SOFTMAX)
                    {
                        softmax = m_net.layers[m_net.layers.Count - 2];
                        nAxis = softmax.layer_param.softmax_param.axis;
                    }

                    List<string> rgOutput = new List<string>();
                    List<Tuple<string, int, double>> res;
                    int nCount = 0;
                    Stopwatch sw = new Stopwatch();
                    sw.Start();

                    for (int i = 0; i < nMax; i++)
                    {
                        if (layerInput.SupportsPostProcessingLogits)
                        {
                            nK = 10;
                            blobTop = m_net.FindBlob("logits");
                            if (blobTop == null)
                                throw new Exception("Could not find the 'logits' blob!");
                            res = layerInput.PostProcessLogitsOutput(i, blobTop, softmax, nAxis, nK);
                        }
                        else
                            res = layerInput.PostProcessOutput(blobTop);

                        if (!layerInput.PreProcessInput(null, res[0].Item2, colBottom))
                            break;

                        rgOutput.Add(res[0].Item1);

                        colTop = m_net.Forward(colBottom, out dfLoss, layerInput.SupportsPostProcessingLogits);
                        blobTop = colTop[0];
                        nCount++;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            double dfPct = (double)nCount / nMax;
                            m_log.WriteLine("Generating response at " + dfPct.ToString("P") + "...");
                        }
                    }

                    strOut = "";
                    foreach (string str in rgOutput)
                    {
                        strOut += str;
                    }
                }
                else
                {
                    BeamSearch<T> search = new BeamSearch<T>(m_net);

                    List<Tuple<double, bool, List<Tuple<string, int, double>>>> res = search.Search(input, nK, dfThreshold, nMax);

                    if (res.Count > 0)
                    {
                        for (int i = 0; i < res[0].Item3.Count; i++)
                        {
                            strOut += res[0].Item3[i].Item1.ToString() + " ";
                        }
                    }

                    strOut = strOut.Trim();
                }

                rgstrOutput.Add(strOut);
            }

            string strFinal = "";
            foreach (string str in rgstrOutput)
            {
                strFinal += str + "|";
            }

            strFinal = clean(strFinal);
            return new PropertySet("Results=" + strFinal);
        }

        /// <summary>
        /// Run the network forward on the bottom blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the input blobs.</param>
        /// <returns>The top output blobs are returned.</returns>
        public BlobCollection<T> Run(BlobCollection<T> colBottom)
        {
            double dfLoss;
            return m_net.Forward(colBottom, out dfLoss, true);
        }

        private string clean(string strFinal)
        {
            string str = "";

            foreach (char ch in strFinal)
            {
                if (ch == ';')
                    str += ' ';
                else
                    str += ch;
            }

            return str;
        }

        /// <summary>
        /// Retrieves a random image from either the training or test set depending on the Phase specified.
        /// </summary>
        /// <param name="phase">Specifies whether to select images from the training set or testing set.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <param name="strLabel">Returns the expected label name for the image.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTestImage(Phase phase, out int nLabel, out string strLabel)
        {
            if (m_db.GetVersion() == DB_VERSION.TEMPORAL)
                throw new Exception("The GetTestImage only works with non-temporal databases.");

            int nSrcId = (phase == Phase.TRAIN) ? m_dataSet.TrainingSource.ID : m_dataSet.TestingSource.ID;
            SimpleDatum sd = m_db.QueryItem(nSrcId, 0, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.RANDOM, null, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);
            m_dataTransformer.TransformLabel(sd);

            nLabel = sd.Label;
            strLabel = ((IXImageDatabaseBase)m_db).GetLabelName(nSrcId, nLabel);

            if (strLabel == null || strLabel.Length == 0)
                strLabel = nLabel.ToString();

            return new Bitmap(ImageData.GetImage(new Datum(sd), null));
        }

        /// <summary>
        /// Retrieves a random image from either the training or test set depending on the Phase specified.
        /// </summary>
        /// <param name="phase">Specifies whether to select images from the training set or testing set.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTestImage(Phase phase, int nLabel)
        {
            if (m_db.GetVersion() == DB_VERSION.TEMPORAL)
                throw new Exception("The GetTestImage only works with non-temporal databases.");

            int nSrcId = (phase == Phase.TRAIN) ? m_dataSet.TrainingSource.ID : m_dataSet.TestingSource.ID;
            SimpleDatum sd = m_db.QueryItem(nSrcId, 0, DB_LABEL_SELECTION_METHOD.RANDOM, DB_ITEM_SELECTION_METHOD.RANDOM, nLabel, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);
            m_dataTransformer.TransformLabel(sd);

            return new Bitmap(ImageData.GetImage(new Datum(sd), null));
        }

        /// <summary>
        /// Retrives the image at a given index within the Testing data set.
        /// </summary>
        /// <param name="nSrcId">Specifies the Source ID.</param>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <param name="strLabel">Returns the expected label name for the image.</param>
        /// <param name="rgCriteria">Returns the data criteria if one exists.</param>
        /// <param name="fmtCriteria">Returns the format of the data criteria, if one exists.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTargetImage(int nSrcId, int nIdx, out int nLabel, out string strLabel, out byte[] rgCriteria, out SimpleDatum.DATA_FORMAT fmtCriteria)
        {
            if (m_db.GetVersion() == DB_VERSION.TEMPORAL)
                throw new Exception("The GetTestImage only works with non-temporal databases.");

            SimpleDatum sd = m_db.QueryItem(nSrcId, nIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, null, m_settings.ItemDbLoadDataCriteria, m_settings.ItemDbLoadDebugData);
            m_dataTransformer.TransformLabel(sd);

            nLabel = sd.Label;
            strLabel = ((IXImageDatabaseBase)m_db).GetLabelName(nSrcId, nLabel);

            if (strLabel == null || strLabel.Length == 0)
                strLabel = nLabel.ToString();

            rgCriteria = sd.DataCriteria;
            fmtCriteria = sd.DataCriteriaFormat;

            return new Bitmap(ImageData.GetImage(new Datum(sd), null));
        }

        /// <summary>
        /// Retrives the image with a given ID.
        /// </summary>
        /// <param name="nImageID">Specifies the Raw Image ID.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <param name="strLabel">Returns the expected label name for the image.</param>
        /// <param name="rgCriteria">Returns the data criteria if one exists.</param>
        /// <param name="fmtCriteria">Returns the format of the data criteria, if one exists.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTargetImage(int nImageID, out int nLabel, out string strLabel, out byte[] rgCriteria, out SimpleDatum.DATA_FORMAT fmtCriteria)
        {
            if (m_db.GetVersion() == DB_VERSION.TEMPORAL)
                throw new Exception("The GetTestImage only works with non-temporal databases.");

            SimpleDatum d = m_db.GetItem(nImageID, m_dataSet.TrainingSource.ID, m_dataSet.TestingSource.ID);

            nLabel = d.Label;
            strLabel = ((IXImageDatabaseBase)m_db).GetLabelName(m_dataSet.TestingSource.ID, nLabel);

            if (strLabel == null || strLabel.Length == 0)
                strLabel = nLabel.ToString();

            rgCriteria = d.DataCriteria;
            fmtCriteria = d.DataCriteriaFormat;

            return new Bitmap(ImageData.GetImage(new Datum(d), null));
        }

        /// <summary>
        /// Returns the item (e.g., image or temporal item) mean used by the solver network used during training.
        /// </summary>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        public SimpleDatum GetItemMean()
        {
            if (m_db == null)
                throw new Exception("The image database is null!");

            if (m_solver == null)
                throw new Exception("The solver is null - make sure that you are loaded for training.");

            if (m_solver.net == null)
                throw new Exception("The solver net is null - make sure that you are loaded for training.");

            string strSrc = m_solver.net.GetDataSource();
            int nSrcId = m_db.GetSourceID(strSrc);

            return m_db.GetItemMean(nSrcId);
        }

        /// <summary>
        /// Returns the current dataset used when training and testing.
        /// </summary>
        /// <returns>The DatasetDescriptor is returned.</returns>
        public DatasetDescriptor GetDataset()
        {
            return m_dataSet;
        }

        /// <summary>
        /// Retrieves the weights of the training network.
        /// </summary>
        /// <returns>The weights are returned.</returns>
        public byte[] GetWeights()
        {
            if (m_net != null)
            {
                m_net.ShareTrainedLayersWith(m_solver.net);
                return m_net.SaveWeights(m_persist);
            }
            else
            {
                return m_solver.net.SaveWeights(m_persist);
            }
        }

        /// <summary>
        /// Loads the weights from the training net into the Net used for running.
        /// </summary>
        /// <param name="bOutputStatus">Optionally, specifies to output status as the weights are updated (default = false).</param>
        /// <param name="bVerifyWeights">Optionally, specifies to verify the run weights copied (default = true).</param>
        public void UpdateRunWeights(bool bOutputStatus = false, bool bVerifyWeights = true)
        {
            bool? bLogEnabled = null;

            try
            {
                if (!bOutputStatus)
                {
                    bLogEnabled = m_log.IsEnabled;
                    m_log.Enable = false;
                }

                if (m_net != null && m_bOwnRunNet)
                {
                    try
                    {
                        int nCopyCount = 0;

                        if (m_solver.net.learnable_parameters.Count == m_net.learnable_parameters.Count)
                        {
                            for (int i = 0; i < m_solver.net.learnable_parameters.Count; i++)
                            {
                                Blob<T> b = m_solver.net.learnable_parameters[i];
                                Blob<T> bRun = m_net.learnable_parameters[i];

                                m_log.CHECK(b.Name == bRun.Name, "The learnable parameter names do not match!");
                                if (bRun.CopyFrom(b, false, true) != 0)
                                    nCopyCount++;
                            }
                        }
                        else
                        {
                            for (int i = 0; i < m_net.learnable_parameters.Count; i++)
                            {
                                Blob<T> bRun = m_net.learnable_parameters[i];
                                Blob<T> b = m_solver.net.FindBlob(bRun.Name);

                                if (b == null)
                                    m_log.FAIL("Could not find the run blob '" + bRun.Name + "' in the solver net!");

                                bRun.CopyFrom(b, false, true);
                            }
                        }

                        if (nCopyCount == 0)
                            loadWeights(m_net, m_solver.net.SaveWeights(m_persist));
                    }
                    catch (Exception excpt)
                    {
                        m_log.WriteLine("WARNING: " + excpt.Message + ", attempting to load with legacy (slower method)...");
                        loadWeights(m_net, m_solver.net.SaveWeights(m_persist));
                    }
                }

                if (bVerifyWeights)
                {
                    if (!CompareWeights(m_net, m_solver.net))
                        m_log.WriteLine("WARNING: The run weights differ from the training weights!");
                }
            }
            finally
            {
                if (bLogEnabled.HasValue)
                    m_log.Enable = bLogEnabled.Value;
            }
        }

        /// <summary>
        /// Loads the training Net with new weights.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights to load.</param>
        public void UpdateWeights(byte[] rgWeights)
        {
            if (m_net != null)
                loadWeights(m_net, rgWeights);

            m_log.WriteLine("Updating weights in solver.");

            List<string> rgExpectedShapes = new List<string>();

            foreach (Blob<T> b in m_solver.TrainingNet.learnable_parameters)
            {
                rgExpectedShapes.Add(b.shape_string);
            }

            bool bLoadDiffs;
            m_persist.LoadWeights(rgWeights, rgExpectedShapes, m_solver.TrainingNet.learnable_parameters, false, out bLoadDiffs);

            m_solver.WeightsUpdated = true;
            m_log.WriteLine("Solver weights updated.");
        }

        /// <summary>
        /// Creates a new Net, loads the weights specified into it and returns it.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights to load.</param>
        /// <param name="cudaOverride">Optionally, specifies a different cuda instance for the Net to use.</param>
        /// <returns>The new Net is returned.</returns>
        public Net<T> CreateNet(byte[] rgWeights, CudaDnn<T> cudaOverride = null)
        {
            if (cudaOverride == null)
                cudaOverride = m_cuda;

            NetParameter p = (m_net != null) ? m_net.ToProto(false) : m_solver.net.ToProto(false);
            Net<T> net = new Net<T>(cudaOverride, m_log, p, m_evtCancel, m_db);
            loadWeights(net, rgWeights);
            return net;
        }

        /// <summary>
        /// Returns the internal net based on the Phase specified: TRAIN, TEST or RUN.
        /// </summary>
        /// <param name="phase">Specifies the Phase used to select the Net.</param>
        /// <returns>The internal Net is returned.</returns>
        /// <remarks>
        /// The following net is returned under the following conditions:
        ///   phase = ALL, return the net from the LastPhase run.  If the LastPhase run = NONE, return the RUN net.
        ///   phase = n/a, return the default RUN net.
        ///   phase = NONE, return the default RUN net.
        ///   phase = TRAIN, return the training net.
        ///   phase = TEST, return the testing net.
        /// </remarks>
        public Net<T> GetInternalNet(Phase phase = Phase.RUN)
        {
            if (phase == Phase.ALL)
                phase = m_lastPhaseRun;

            if (phase == Phase.NONE)
                phase = Phase.RUN;

            if (phase == Phase.TEST)
                return (m_solver != null) ? m_solver.TestingNet : null;

            else if (phase == Phase.TRAIN)
                return (m_solver != null) ? m_solver.TrainingNet : null;

            return m_net;
        }

        /// <summary>
        /// Get the internal solver.
        /// </summary>
        /// <returns></returns>
        public Solver<T> GetInternalSolver()
        {
            return m_solver;
        }

        /// <summary>
        /// The Snapshot function forces a snapshot to occur.
        /// </summary>
        /// <param name="bUpdateDatabase">Optionally, specifies to update the database (default = true).</param>
        public void Snapshot(bool bUpdateDatabase = true)
        {
            m_solver.Snapshot(true, false, bUpdateDatabase);
        }

        /// <summary>
        /// Reset the device at the given device ID.
        /// </summary>
        /// <remarks>
        /// <b>WARNING!</b> It is recommended that this only be used when testing, for calling this will throw off all other users
        /// of the device and may cause unpredictable behavior.
        /// </remarks>
        /// <param name="nDeviceID">Specifies the device ID of the device to reset.</param>
        public static void ResetDevice(int nDeviceID)
        {
        }

        /// <summary>
        /// Returns the license text for MyCaffe.
        /// </summary>
        /// <param name="strOtherLicenses">Specifies other licenses to append to the license text.</param>
        /// <returns></returns>
        public static string GetLicenseTextEx(string strOtherLicenses)
        {
            string str = Properties.Resources.LICENSE;
            int nYear = DateTime.Now.Year;

            if (nYear > 2016)
                str = replaceMacro(str, "$$YEAR$$", "-" + nYear.ToString());
            else
                str = replaceMacro(str, "$$YEAR$$", "");

            if (strOtherLicenses != null && strOtherLicenses.Length > 0)
                str = replaceMacro(str, "$$OTHERLICENSES$$", strOtherLicenses);

            return fixupReturns(str);
        }

        /// <summary>
        /// Returns the license text for MyCaffe.
        /// </summary>
        /// <param name="strOtherLicenses">Specifies other licenses to append to the license text.</param>
        /// <returns></returns>
        public string GetLicenseText(string strOtherLicenses)
        {
            return GetLicenseTextEx(strOtherLicenses);
        }

        /// <summary>
        /// VerifyCompute compares the current compute of the current device (or device specified) against the required compute of the current CudaDnnDLL.dll used.
        /// </summary>
        /// <param name="strExtra">Optionally, specifies extra information for the exception if one is thrown.</param>
        /// <param name="nDeviceID">Optionally, specifies a specific device ID to check, otherwise uses the current device used (default = -1, which uses the current device).</param>
        /// <param name="bThrowException">Optionally, specifies whether or not to throw an exception on a compute mis-match (default = true).</param>
        /// <returns>If the device's compute is >= to the required compute fo the CudaDnnDll.dll used, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool VerifyCompute(string strExtra = null, int nDeviceID = -1, bool bThrowException = true)
        {
            if (m_cuda == null)
                throw new Exception("You must initialize the MyCaffeControl with an instance of CudaDnn<T>, or Load a new project.");

            int nMinMajor;
            int nMinMinor;
            string strDll = m_cuda.GetRequiredCompute(out nMinMajor, out nMinMinor);

            if (nDeviceID == -1)
                nDeviceID = m_cuda.GetDeviceID();

            string strDevName = m_cuda.GetDeviceName(nDeviceID);
            string strCompute = parse(strDevName, "compute ", ")");
            string[] rgstr = strCompute.Split('.');
            string strMajor = rgstr[0];
            string strMinor = rgstr[1];
            if (strMajor == null || strMinor == null)
                throw new Exception("Could not find the current device's major and minor version information!");

            int nMajor = int.Parse(strMajor);
            int nMinor = int.Parse(strMinor);

            if (nMajor < nMinMajor || (nMajor == nMinMajor && nMinor < nMinMinor))
            {
                string strErr = "The device " + nDeviceID.ToString() + " - '" + strDevName + " does not meet the minimum compute of '" + nMinMajor.ToString() + "." + nMinMinor.ToString() + "' required by the CudaDnnDll used ('" + strDll + "')!";
                if (!string.IsNullOrEmpty(strExtra))
                    strErr += strExtra;
                throw new Exception(strErr);
            }

            return true;
        }

        private string parse(string str, string strT1, string strT2)
        {
            int nPos = str.IndexOf(strT1);
            if (nPos < 0)
                return null;

            str = str.Substring(nPos + strT1.Length);
            nPos = str.IndexOf(strT2);
            if (nPos < 0)
                return null;

            return str.Substring(0, nPos).Trim();
        }

        private static string replaceMacro(string str, string strMacro, string strReplacement)
        {
            int nPos = str.IndexOf(strMacro);

            if (nPos < 0)
                return str;

            string strA = str.Substring(0, nPos);

            strA += strReplacement;
            strA += str.Substring(nPos + strMacro.Length);

            return strA;
        }

        private static string fixupReturns(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (ch == '\n')
                    strOut += "\r\n";
                else
                    strOut += ch;
            }

            return strOut;
        }

        /// <summary>
        /// Create an unsized blob and set its name.
        /// </summary>
        /// <param name="strName">Specifies the Blob name.</param>
        /// <returns>The Blob is returned.</returns>
        public Blob<T> CreateBlob(string strName)
        {
            Blob<T> b = new Blob<T>(m_cuda, m_log);
            b.Name = strName;
            return b;
        }

        /// <summary>
        /// Create and load a new extension DLL.
        /// </summary>
        /// <param name="strExtensionDLLPath">Specifies the path to the extension DLL.</param>
        /// <returns>The handle to the extension is returned.</returns>
        public long CreateExtension(string strExtensionDLLPath)
        {
            return m_cuda.CreateExtension(strExtensionDLLPath);
        }

        /// <summary>
        /// Free an existing extension and unload it.
        /// </summary>
        /// <param name="hExtension">Specifies the handle to the extension to free.</param>
        public void FreeExtension(long hExtension)
        {
            m_cuda.FreeExtension(hExtension);
        }
        /// <summary>
        /// Run a function on an existing extension.
        /// </summary>
        /// <param name="hExtension">Specifies the extension.</param>
        /// <param name="lfnIdx">Specifies the function to run on the extension.</param>
        /// <param name="rgParam">Specifies the parameters.</param>
        /// <returns>The return values of the function are returned.</returns>
        public T[] RunExtension(long hExtension, long lfnIdx, T[] rgParam)
        {
            return m_cuda.RunExtension(hExtension, lfnIdx, rgParam);
        }
        /// <summary>
        /// Run a function on an existing extension using the <i>double</i> base type.
        /// </summary>
        /// <param name="hExtension">Specifies the extension.</param>
        /// <param name="lfnIdx">Specifies the function to run on the extension.</param>
        /// <param name="rgParam">Specifies the parameters.</param>
        /// <returns>The return values of the function are returned.</returns>
        public double[] RunExtensionD(long hExtension, long lfnIdx, double[] rgParam)
        {
            T[] rgP = (rgParam == null) ? null : Utility.ConvertVec<T>(rgParam);
            T[] rg = m_cuda.RunExtension(hExtension, lfnIdx, rgP);

            if (rg == null)
                return null;

            return Utility.ConvertVec<T>(rg);
        }
        /// <summary>
        /// Run a function on an existing extension using the <i>float</i> base type.
        /// </summary>
        /// <param name="hExtension">Specifies the extension.</param>
        /// <param name="lfnIdx">Specifies the function to run on the extension.</param>
        /// <param name="rgParam">Specifies the parameters.</param>
        /// <returns>The return values of the function are returned.</returns>
        public float[] RunExtensionF(long hExtension, long lfnIdx, float[] rgParam)
        {
            T[] rgP = (rgParam == null) ? null : Utility.ConvertVec<T>(rgParam);
            T[] rg = m_cuda.RunExtension(hExtension, lfnIdx, rgP);

            if (rg == null)
                return null;

            return Utility.ConvertVecF<T>(rg);
        }
    }
}
