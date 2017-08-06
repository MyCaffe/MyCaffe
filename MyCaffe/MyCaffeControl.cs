using System;
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
using MyCaffe.imagedb;
using MyCaffe.solvers;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.data;
using MyCaffe.layers;

/// <summary>
/// The MyCaffe namespace contains the main body of MyCaffe code that closesly tracks the C++ %Caffe open-source project.  
/// </summary>
namespace MyCaffe
{
    /// <summary>
    /// The MyCaffeControl is the main object used to manage all training, testing and running of the MyCaffe system.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public partial class MyCaffeControl<T> : Component, IXMyCaffeState<T>, IXMyCaffe<T>, IXMyCaffeNoDb<T>, IDisposable
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
        protected IXImageDatabase m_imgDb = null;
        /// <summary>
        /// Whether or not the control owns the image database.
        /// </summary>
        protected bool m_bImgDbOwner = true;
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
        bool m_bDisposing = false;
        MemoryStream m_msWeights = new MemoryStream();
        Guid m_guidUser;
        PersistCaffe<T> m_persist;
        BlobShape m_inputShape = null;


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
        public MyCaffeControl(SettingsCaffe settings, Log log, CancelEvent evtCancel, AutoResetEvent evtSnapshot = null, AutoResetEvent evtForceTest = null, ManualResetEvent evtPause = null, List<int> rgGpuId = null, string strCudaPath = "")
        {
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
        }

        /// <summary>
        /// Releases all GPU and Host resources used by the CaffeControl.
        /// </summary>
        public void dispose()
        {
            if (m_bDisposing)
                return;

            m_bDisposing = true;

            if (m_evtCancel != null)
                m_evtCancel.Set();

            Unload();

            if (m_cuda != null)
            {
                m_cuda.Dispose();
                m_cuda = null;
            }

            if (m_msWeights != null)
            {
                m_msWeights.Dispose();
                m_msWeights = null;
            }
        }

        /// <summary>
        /// Unload the currently loaded project, if any.
        /// </summary>
        public void Unload(bool bUnloadImageDb = true)
        {
            if (m_solver != null)
            {
                m_solver.Dispose();
                m_solver = null;
            }

            if (m_net != null)
            {
                m_net.Dispose();
                m_net = null;
            }

            if (m_imgDb != null && bUnloadImageDb)
            {
                if (m_bImgDbOwner)
                {
                    m_imgDb.CleanUp();

                    IDisposable idisp = m_imgDb as IDisposable;
                    if (idisp != null)
                        idisp.Dispose();
                }

                m_imgDb = null;
            }

            m_project = null;
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
        /// Sets the cance override.
        /// </summary>
        /// <param name="evtCancel">Specifies the new cancel event to use.</param>
        public void SetCancelOverride(ManualResetEvent evtCancel)
        {
            m_evtCancel.SetCancelOverride(evtCancel);
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
        /// Returns the CaffeImageDatabase used.
        /// </summary>
        public IXImageDatabase ImageDatabase
        {
            get { return m_imgDb; }
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
        /// This string can help diagnos label balancing issue.
        /// </remarks>
        public string ActiveLabelCounts
        {
            get { return m_solver.ActiveLabelCounts; }
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
            return createNetParameterForRunning(p.Dataset, p.ModelDescription, out transform_param);
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
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        protected NetParameter createNetParameterForRunning(DatasetDescriptor ds, string strModel, out TransformationParameter transform_param)
        {
            return createNetParameterForRunning(datasetToShape(ds), strModel, out transform_param);
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
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        public NetParameter createNetParameterForRunning(BlobShape shape, string strModel, out TransformationParameter transform_param)
        {
            m_inputShape = shape;
            return CreateNetParameterForRunning(shape, strModel, out transform_param);
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
        /// <returns>The new NetParameter suitable for the RUN phase is returned.</returns>
        public static NetParameter CreateNetParameterForRunning(BlobShape shape, string strModel, out TransformationParameter transform_param)
        {
            int nImageChannels = shape.dim[1];
            int nImageHeight = shape.dim[2];
            int nImageWidth = shape.dim[3];

            RawProto protoTransform = null;
            RawProto protoModel = ProjectEx.CreateModelForRunning(strModel, "data", 1, nImageChannels, nImageHeight, nImageWidth, out protoTransform);

            if (protoTransform != null)
                transform_param = TransformationParameter.FromProto(protoTransform);
            else
                transform_param = new param.TransformationParameter();

            NetParameter np = NetParameter.FromProto(protoModel);

            np.ProjectID = 0;
            np.state.phase = Phase.RUN;

            return np;
        }

        private BlobShape datasetToShape(DatasetDescriptor ds)
        {
            int nH = ds.TestingSource.ImageHeight;
            int nW = ds.TestingSource.ImageWidth;
            int nC = ds.TestingSource.ImageChannels;
            List<int> rgShape = new List<int>() { 1, nC, nH, nW };
            return new BlobShape(rgShape);
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
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection override (overides the image selection in SettingsCaffe).  The image selection dictates how the images are selected from each label set.</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="imgdb">Optionally, specifies the MyCaffeImageDatabase to use.  When <i>null</i>, an instance if the MyCaffeImageDatabase is created internally.</param>
        public void Load(Phase phase, ProjectEx p, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, bool bResetFirst = false, IXImageDatabase imgdb = null)
        {
            m_imgDb = imgdb;
            m_bImgDbOwner = false;

            if (m_imgDb == null)
            {
                m_imgDb = new MyCaffeImageDatabase(m_log);
                m_bImgDbOwner = true;

                m_log.WriteLine("Loading images...");

                m_imgDb.Initialize(m_settings, p.Dataset, m_evtCancel);
                m_imgDb.UpdateLabelBoosts(p.ID, p.Dataset.TrainingSource.ID);

                KeyValuePair<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> kvSelection = MyCaffeImageDatabase.GetSelectionMethod(p);

                if (labelSelectionOverride.HasValue)
                    m_imgDb.LabelSelectionMethod = labelSelectionOverride.Value;
                else
                    m_imgDb.LabelSelectionMethod = kvSelection.Key;

                if (imageSelectionOverride.HasValue)
                    m_imgDb.ImageSelectionMethod = imageSelectionOverride.Value;
                else
                    m_imgDb.ImageSelectionMethod = kvSelection.Value;

                m_imgDb.QueryImageMean(p.Dataset.TrainingSource.ID);
                m_log.WriteLine("Images loaded.");
            }

            m_project = p;

            if (m_project == null)
                throw new Exception("You must specify a project.");

            m_dataSet = m_project.Dataset;

            if (m_cuda != null)
                m_cuda.Dispose();

            m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath, bResetFirst);

            if (phase == Phase.TEST || phase == Phase.TRAIN)
            {
                m_log.WriteLine("Creating solver...");

                m_solver = Solver<T>.Create(m_cuda, m_log, p, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_imgDb, m_persist, m_rgGpu.Count, 0);
                m_solver.SnapshotWeightUpdateMethod = m_settings.SnapshotWeightUpdateMethod;
                if (p.WeightsState != null || p.SolverState != null)
                    m_solver.Restore(p.WeightsState, p.SolverState);

                m_solver.OnSnapshot += new EventHandler<SnapshotArgs>(m_solver_OnSnapshot);
                m_solver.OnTrainingIteration += new EventHandler<TrainingIterationArgs<T>>(m_solver_OnTrainingIteration);
                m_solver.OnTestingIteration += new EventHandler<TestingIterationArgs<T>>(m_solver_OnTestingIteration);
                m_log.WriteLine("Solver created.");
            }

            if (phase == Phase.TRAIN)
                m_imgDb.UpdateLabelBoosts(p.ID, m_dataSet.TrainingSource.ID);

            if (phase == Phase.TEST)
                m_imgDb.UpdateLabelBoosts(p.ID, m_dataSet.TestingSource.ID);

            if (p == null)
                return;

            TransformationParameter tp = null;
            NetParameter netParam = createNetParameterForRunning(p, out tp);

            if (tp != null)
                m_dataTransformer = new DataTransformer<T>(m_log, tp, Phase.RUN, m_imgDb.QueryImageMean(m_dataSet.TrainingSource.ID));
            else
                m_dataTransformer = null;

            if (phase == Phase.RUN)
            {
                m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_imgDb);

                if (p.WeightsState != null)
                    loadWeights(m_net, p.WeightsState);
            }
            else if (phase == Phase.TEST || phase == Phase.TRAIN)
            {
                m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_imgDb, Phase.RUN, null, m_solver.TrainingNet);
            }
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
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection override (overides the image selection in SettingsCaffe).  The image selection dictates how the images are selected from each label set.</param>
        /// <param name="bResetFirst">Optionally, resets the device before loading.  IMPORTANT: this functionality is only recommendned during testing, for resetting the device will throw off all other users of the device.</param>
        /// <param name="imgdb">Optionally, specifies the MyCaffeImageDatabase to use.  When <i>null</i>, an instance if the MyCaffeImageDatabase is created internally.</param>
        public void Load(Phase phase, string strSolver, string strModel, byte[] rgWeights, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, bool bResetFirst = false, IXImageDatabase imgdb = null)
        {
            m_imgDb = imgdb;
            m_bImgDbOwner = false;

            RawProto protoSolver = RawProto.Parse(strSolver);
            SolverParameter solverParam = SolverParameter.FromProto(protoSolver);

            RawProto protoModel = RawProto.Parse(strModel);
            solverParam.net_param = NetParameter.FromProto(protoModel);

            m_dataSet = findDataset(solverParam.net_param);

            if (m_imgDb == null)
            {
                m_imgDb = new MyCaffeImageDatabase(m_log);
                m_bImgDbOwner = true;

                m_log.WriteLine("Loading images...");

                m_imgDb.Initialize(m_settings, m_dataSet, m_evtCancel);
                KeyValuePair<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> kvSelection = MyCaffeImageDatabase.GetSelectionMethod(m_settings);

                if (labelSelectionOverride.HasValue)
                    m_imgDb.LabelSelectionMethod = labelSelectionOverride.Value;
                else
                    m_imgDb.LabelSelectionMethod = kvSelection.Key;

                if (imageSelectionOverride.HasValue)
                    m_imgDb.ImageSelectionMethod = imageSelectionOverride.Value;
                else
                    m_imgDb.ImageSelectionMethod = kvSelection.Value;

                m_imgDb.QueryImageMean(m_dataSet.TrainingSource.ID);
                m_log.WriteLine("Images loaded.");
            }

            m_project = null;

            if (m_cuda != null)
                m_cuda.Dispose();

            m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath, bResetFirst);

            if (phase == Phase.TEST || phase == Phase.TRAIN)
            {
                m_log.WriteLine("Creating solver...");

                m_solver = Solver<T>.Create(m_cuda, m_log, solverParam, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, m_imgDb, m_persist, m_rgGpu.Count, 0);

                if (rgWeights != null)
                    m_solver.Restore(rgWeights, null);

                m_solver.OnSnapshot += new EventHandler<SnapshotArgs>(m_solver_OnSnapshot);
                m_solver.OnTrainingIteration += new EventHandler<TrainingIterationArgs<T>>(m_solver_OnTrainingIteration);
                m_solver.OnTestingIteration += new EventHandler<TestingIterationArgs<T>>(m_solver_OnTestingIteration);
                m_log.WriteLine("Solver created.");
            }

            TransformationParameter tp = null;
            NetParameter netParam = createNetParameterForRunning(m_dataSet, strModel, out tp);

            if (tp != null)
                m_dataTransformer = new DataTransformer<T>(m_log, tp, Phase.RUN, m_imgDb.QueryImageMean(m_dataSet.TrainingSource.ID));
            else
                m_dataTransformer = null;

            if (phase == Phase.RUN)
            {
                m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_imgDb);

                if (rgWeights != null)
                    loadWeights(m_net, rgWeights);
            }
            else if (phase == Phase.TEST || phase == Phase.TRAIN)
            {
                netParam.force_backward = true;
                m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, m_imgDb, Phase.RUN, null, m_solver.TrainingNet);
            }
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
        public void LoadToRun(string strModel, byte[] rgWeights, BlobShape shape, SimpleDatum sdMean = null, TransformationParameter transParam = null, bool bForceBackward = false)
        {
            m_dataSet = null;
            m_project = null;

            if (m_cuda != null)
                m_cuda.Dispose();

            m_cuda = new CudaDnn<T>(m_rgGpu[0], DEVINIT.CUBLAS | DEVINIT.CURAND, null, m_strCudaPath);
           
            TransformationParameter tp = null;
            NetParameter netParam = createNetParameterForRunning(shape, strModel, out tp);

            netParam.force_backward = bForceBackward;

            if (transParam != null)
                tp = transParam;

            if (tp != null)
            {
                if (tp.use_imagedb_mean && sdMean == null)
                    throw new Exception("The transformer expects an image mean, yet the sdMean parameter is null!");

                m_dataTransformer = new DataTransformer<T>(m_log, tp, Phase.RUN, sdMean);
            }
            else
            {
                m_dataTransformer = null;
            }

            m_net = new Net<T>(m_cuda, m_log, netParam, m_evtCancel, null);
            loadWeights(m_net, rgWeights);
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
                else if (lp.type == LayerParameter.LayerType.DATA ||
                         lp.type == LayerParameter.LayerType.TRIPLET_DATA ||
                         lp.type == LayerParameter.LayerType.BATCHDATA)
                {
                    switch (lp.type)
                    {
                        case LayerParameter.LayerType.DATA:
                        case LayerParameter.LayerType.TRIPLET_DATA:
                            strSrc = lp.data_param.source;
                            break;

                        case LayerParameter.LayerType.BATCHDATA:
                            strSrc = lp.batch_data_param.source;
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

            return factory.QueryImageMean(0, sd.ID);
        }

        private DatasetDescriptor findDataset(NetParameter p)
        {
            string strTestSrc = null;
            string strTrainSrc = null;

            foreach (LayerParameter lp in p.layer)
            {
                if (lp.type == LayerParameter.LayerType.DATA ||
                    lp.type == LayerParameter.LayerType.TRIPLET_DATA ||
                    lp.type == LayerParameter.LayerType.BATCHDATA)
                {
                    string strSrc = null;

                    switch (lp.type)
                    {
                        case LayerParameter.LayerType.DATA:
                        case LayerParameter.LayerType.TRIPLET_DATA:
                            strSrc = lp.data_param.source;
                            break;

                        case LayerParameter.LayerType.BATCHDATA:
                            strSrc = lp.batch_data_param.source;
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
                    break;
            }

            if (strTrainSrc == null || strTestSrc == null)
                return null;

            DatasetFactory factory = new DatasetFactory();
            return factory.LoadDataset(strTestSrc, strTrainSrc);
        }

        private void loadWeights(Net<T> net, byte[] rgWeights)
        {
            net.LoadWeights(rgWeights, m_persist);
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
        public void Train(int nIterationOverride = -1)
        {
            if (nIterationOverride == -1)
                nIterationOverride = m_settings.MaximumIterationOverride;

            m_solver.TrainingIterationOverride = nIterationOverride;
            m_solver.TestingIterationOverride = m_solver.TestingIterationOverride;

            if (m_rgGpu.Count > 1)
            {
                NCCL<T> nccl = new NCCL<T>(m_cuda, m_log, m_solver, m_rgGpu[0], 0, null);
                nccl.Run(m_rgGpu, m_solver.TrainingIterationOverride);
            }
            else
            {
                m_solver.Solve();
            }
        }

        /// <summary>
        /// Test the network a given number of iterations.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies number of iterations to run that override the iterations specified in the solver desctiptor.</param>
        /// <returns>The accuracy value from the test is returned.</returns>
        public double Test(int nIterationOverride = -1)
        {
            if (nIterationOverride == -1)
                nIterationOverride = m_settings.TestingIterationOverride;

            m_solver.TestingIterationOverride = nIterationOverride;

            return m_solver.TestAll();
        }

        /// <summary>
        /// Test on a number of images by selecting random images from the database, running them through the Run network, and then comparing the results with the 
        /// expected results.
        /// </summary>
        /// <param name="nCount">Specifies the number of cycles to run.</param>
        /// <param name="bOnTrainingSet">Specifies on whether to select images from the training set, or when <i>false</i> the testing set of data.</param>
        public void TestMany(int nCount, bool bOnTrainingSet)
        {
            Stopwatch sw = new Stopwatch();
            int nSrcId = (bOnTrainingSet) ? m_dataSet.TrainingSource.ID : m_dataSet.TestingSource.ID;
            int nCorrectCount = 0;
            Dictionary<int, int> rgCorrectCounts = new Dictionary<int, int>();
            Dictionary<int, int> rgLabelTotals = new Dictionary<int, int>();

            sw.Start();

            UpdateRunWeights();
            m_log.WriteHeader("Test Many (" + nCount.ToString() + ") - " + ((bOnTrainingSet) ? ("on training set") : ("on testing set")));

            LabelMappingParameter labelMapping = null;

            foreach (Layer<T> layer in m_solver.TestingNet.layers)
            {
                if (layer.type == LayerParameter.LayerType.LABELMAPPING)
                {
                    labelMapping = layer.layer_param.labelmapping_param;
                    break;
                }
            }

            for (int i = 0; i < nCount; i++)
            {
                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Test Many aborted!");
                    return;
                }

                SimpleDatum sd = m_imgDb.QueryImage(nSrcId, i);

                if (sd.ByteData == null && sd.RealData == null)
                {
                    Trace.WriteLine("You should not be here.");
                    throw new Exception("NO DATA!");
                }

                ResultCollection rgResults = Run(sd);
                int nExpectedLabel = sd.Label;

                if (labelMapping != null)
                    nExpectedLabel = labelMapping.MapLabel(nExpectedLabel);

                if (!rgCorrectCounts.ContainsKey(nExpectedLabel))
                    rgCorrectCounts.Add(nExpectedLabel, 0);

                if (!rgLabelTotals.ContainsKey(nExpectedLabel))
                    rgLabelTotals.Add(nExpectedLabel, 1);
                else
                    rgLabelTotals[nExpectedLabel]++;

                if (nExpectedLabel == rgResults.DetectedLabel)
                {
                    nCorrectCount++;
                    rgCorrectCounts[nExpectedLabel]++;
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

            double dfCorrectPct = ((double)nCorrectCount / (double)nCount);

            m_log.WriteLine("Test Many Completed.");
            m_log.WriteLine(" " + dfCorrectPct.ToString("P") + " correct detections.");
            m_log.WriteLine(" " + (nCount - nCorrectCount).ToString("N") + " incorrect detections.");

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
                    dfCorrectPct = ((double)kv.Value / (double)nCount);
                    m_log.WriteLine("Label #" + kv.Key.ToString() + " had " + dfCorrectPct.ToString("P") + " correct detections out of " + nCount.ToString("N0") + " items with this label.");
                }
            }
        }

        /// <summary>
        /// Train a batch of information described by the BatchInformationCollection.
        /// </summary>
        /// <param name="col">Specifies the mini batch information to train on.</param>
        /// <param name="bEnableTesting">When <i>true</i>, testing is performed, otherwise it is not.</param>
        /// <returns></returns>
        public bool TrainBatch(BatchInformationCollection col, bool bEnableTesting)
        {
            try
            {
                m_solver.EnableTesting = bEnableTesting;
                m_solver.Solve(-1, null, null, col);

                if (m_evtCancel.WaitOne(0))
                    return false;

                return true;
            }
            finally
            {
                m_solver.EnableTesting = true;
            }
        }

        /// <summary>
        /// Run on a given mini batch.
        /// </summary>
        /// <param name="col">Specifies the mini batch information to run.</param>
        /// <returns></returns>
        public bool RunBatch(BatchInformationCollection col)
        {
            Stopwatch sw = new Stopwatch();


            m_log.WriteLine("Running " + col.Count.ToString("N0") + " batches...");
            sw.Start();

            int nBatchCount = col.Count;
            int nBatchSize = col[0].Count;
            List<T> rgImageIdx = new List<T>();

            for (int i = 0; i < nBatchCount; i++)
            {
                for (int j = 0; j < nBatchSize; j++)
                {
                    rgImageIdx.Add((T)Convert.ChangeType(col[i][j].ImageIndex1, typeof(T)));
                }
            }

            Blob<T> blobInput = new Blob<T>(m_cuda, m_log, nBatchCount, nBatchSize, 1, 1);
            blobInput.mutable_cpu_data = rgImageIdx.ToArray();

            BlobCollection<T> colBottom = new common.BlobCollection<T>();
            colBottom.Add(blobInput);

            m_solver.CancelEvent.Reset();
            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.AddRange(m_solver.CancelEvent.Handles);
            rgWait.Add(m_solver.CompletedEvent);
            int nWait;
            int nBatchIdx = 0;

            while ((nWait = WaitHandle.WaitAny(rgWait.ToArray(), 0)) == WaitHandle.WaitTimeout)
            {
                double dfLoss;
                BlobCollection<T> colResults = m_solver.TrainingNet.Forward(colBottom, out dfLoss);
                T[] rgData = colResults[0].update_cpu_data();
                int nOutputCount = rgData.Length / nBatchSize;

                for (int i = 0; i < nBatchSize; i++)
                {
                    double dfMax = -double.MaxValue;
                    int nMaxIdx = -1;

                    for (int j = 0; j < nOutputCount; j++)
                    {
                        double dfVal;
                        int nIdx = (i * nOutputCount) + j;

                        if (typeof(T) == typeof(float))
                            dfVal = (float)Convert.ChangeType(rgData[nIdx], typeof(float));
                        else
                            dfVal = (double)Convert.ChangeType(rgData[nIdx], typeof(double));

                        if (dfVal > dfMax)
                        {
                            dfMax = dfVal;
                            nMaxIdx = j;
                        }
                    }

                    col[nBatchIdx][i].QMax1 = dfMax;
                    col[nBatchIdx][i].Label = nMaxIdx;
                }

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = (double)nBatchIdx / (double)nBatchCount;
                    m_log.WriteLine("Batch #" + (nBatchIdx + 1).ToString() + " - calculating q-values (" + dfPct.ToString("P") + ")...");
                    sw.Restart();
                }

                nBatchIdx++;
            }

            blobInput.Dispose();

            if (nWait < rgWait.Count-1)
                return false;

            return true;
        }

        /// <summary>
        /// Run on a given image in the MyCaffeImageDatabase based on its image index.
        /// </summary>
        /// <param name="nImageIdx">Specifies the image index.</param>
        /// <returns>The result of the run is returned.</returns>
        public ResultCollection Run(int nImageIdx)
        {
            SimpleDatum sd = m_imgDb.QueryImage(m_dataSet.TrainingSource.ID, nImageIdx, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
            return Run(sd, true, true);
        }

        /// <summary>
        /// Run on a set of images in the MyCaffeImageDatabase based on their image indexes.
        /// </summary>
        /// <param name="rgImageIdx">Specifies a list of image indexes.</param>
        /// <returns>A list of results from the run is returned - one result per image.</returns>
        public List<ResultCollection> Run(List<int> rgImageIdx)
        {
            List<SimpleDatum> rgSd = new List<SimpleDatum>();

            foreach (int nImageIdx in rgImageIdx)
            {
                SimpleDatum sd = m_imgDb.QueryImage(m_dataSet.TrainingSource.ID, nImageIdx, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                rgSd.Add(sd);
            }

            return Run(rgSd, true, true);
        }

        /// <summary>
        /// Run on a given Datum. 
        /// </summary>
        /// <param name="d">Specifies the Datum to run.</param>
        /// <param name="bSort">Specifies whether or not to sor the results.</param>
        /// <param name="bUseSolverNet">Optionally, specifies whether or not to use the training net vs. the run net.</param>
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(SimpleDatum d, bool bSort, bool bUseSolverNet)
        {
            Blob<T> blob = null;

            if (m_dataTransformer == null)
            {
                blob = new Blob<T>(m_cuda, m_log, d, true);
            }
            else
            {
                blob = new Blob<T>(m_cuda, m_log);
                Datum datum = new Datum(d);
                List<int> rgShape = m_dataTransformer.InferBlobShape(datum);
                blob.Reshape(rgShape);
                blob.SetData(m_dataTransformer.Transform(datum));
                m_dataTransformer.SetRange(blob);
            }

            BlobCollection<T> colBottom = new BlobCollection<T>() { blob };
            double dfLoss = 0;

            BlobCollection<T> colResults;

            if (bUseSolverNet)
                colResults = m_solver.TrainingNet.Forward(colBottom, out dfLoss);
            else
                colResults = m_net.Forward(colBottom, out dfLoss);

            List<KeyValuePair<int, double>> rgResults = new List<KeyValuePair<int, double>>();
            T[] rgData = colResults[0].update_cpu_data();

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfProb = (double)Convert.ChangeType(rgData[i], typeof(double));
                rgResults.Add(new KeyValuePair<int, double>(i, dfProb));
            }

            blob.Dispose();

            ResultCollection result = new ResultCollection(rgResults);

            if (m_imgDb != null)
                result.SetLabels(m_imgDb.GetLabels(m_dataSet.TrainingSource.ID));

            return result;
        }

        /// <summary>
        /// Run on a given list of Datum. 
        /// </summary>
        /// <param name="rgSd">Specifies the list of Datum to run.</param>
        /// <param name="bSort">Optionally, specifies whether or not to sor the results.</param>
        /// <param name="bUseSolverNet">Optionally, specifies whether or not to use the training net vs. the run net.</param>
        /// <returns>A list of results of the run are returned.</returns>
        protected List<ResultCollection> Run(List<SimpleDatum> rgSd, bool bSort = true, bool bUseSolverNet = false)
        {
            m_log.CHECK(m_dataTransformer != null, "The data transformer is not initialized!");

            List<ResultCollection> rgFinalResults = new List<ResultCollection>();
            int nBatchSize = rgSd.Count;
            int nChannels = m_dataSet.TestingSource.ImageChannels;
            int nHeight = m_dataSet.TestingSource.ImageHeight;
            int nWidth = m_dataSet.TestingSource.ImageWidth;
            Blob<T> blob = new common.Blob<T>(m_cuda, m_log, nBatchSize, nChannels, nHeight, nWidth);
            List<T> rgDataInput = new List<T>();

            foreach (SimpleDatum sd in rgSd)
            {
                Datum d = new Datum(sd);
                rgDataInput.AddRange(m_dataTransformer.Transform(d));
            }

            blob.mutable_cpu_data = rgDataInput.ToArray();
            m_dataTransformer.SetRange(blob);

            BlobCollection<T> colBottom = new BlobCollection<T>() { blob };
            double dfLoss = 0;

            BlobCollection<T> colResults;

            if (bUseSolverNet)
            {
                m_solver.TrainingNet.SetEnablePassthrough(true);
                colResults = m_solver.TrainingNet.Forward(colBottom, out dfLoss);
                m_solver.TrainingNet.SetEnablePassthrough(false);
            }
            else
                colResults = m_net.Forward(colBottom, out dfLoss);

            T[] rgDataOutput = colResults[0].update_cpu_data();
            int nOutputCount = rgDataOutput.Length / rgSd.Count;

            for (int i = 0; i < rgSd.Count; i++)
            {
                List<KeyValuePair<int, double>> rgResults = new List<KeyValuePair<int, double>>();

                for (int j = 0; j < nOutputCount; j++)
                {
                    int nIdx = i * nOutputCount + j;
                    double dfProb = (double)Convert.ChangeType(rgDataOutput[nIdx], typeof(double));
                    rgResults.Add(new KeyValuePair<int, double>(j, dfProb));
                }

                ResultCollection result = new ResultCollection(rgResults);
                result.SetLabels(m_imgDb.GetLabels(m_dataSet.TrainingSource.ID));

                rgFinalResults.Add(result);
            }

            blob.Dispose();

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
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(Bitmap img, bool bSort = true)
        {
            int nChannels = m_inputShape.dim[1];
            return Run(ImageData.GetImageData(img, nChannels, false, -1), bSort, false);
        }

        /// <summary>
        /// Run on a given Datum. 
        /// </summary>
        /// <param name="d">Specifies the Datum to run.</param>
        /// <param name="bSort">Specifies whether or not to sort the results.</param>
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(SimpleDatum d, bool bSort = true)
        {
            return Run(d, bSort, false);
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
            int nSrcId = (phase == Phase.TRAIN) ? m_dataSet.TrainingSource.ID : m_dataSet.TestingSource.ID;
            SimpleDatum d = m_imgDb.QueryImage(nSrcId, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);

            nLabel = d.Label;
            strLabel = m_imgDb.GetLabelName(nSrcId, nLabel);

            if (strLabel == null || strLabel.Length == 0)
                strLabel = nLabel.ToString();

            return new Bitmap(ImageData.GetImage(new Datum(d), null));
        }

        /// <summary>
        /// Retrieves a random image from either the training or test set depending on the Phase specified.
        /// </summary>
        /// <param name="phase">Specifies whether to select images from the training set or testing set.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTestImage(Phase phase, int nLabel)
        {
            int nSrcId = (phase == Phase.TRAIN) ? m_dataSet.TrainingSource.ID : m_dataSet.TestingSource.ID;
            SimpleDatum d = m_imgDb.QueryImage(nSrcId, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, nLabel);

            return new Bitmap(ImageData.GetImage(new Datum(d), null));
        }

        /// <summary>
        /// Retrives the image at a given index within the Testing data set.
        /// </summary>
        /// <param name="nSrcId">Specifies the Source ID.</param>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <param name="strLabel">Returns the expected label name for the image.</param>
        /// <param name="rgCriteria">Returns the data criteria if one exists.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTargetImage(int nSrcId, int nIdx, out int nLabel, out string strLabel, out byte[] rgCriteria)
        {
            SimpleDatum d = m_imgDb.QueryImage(nSrcId, nIdx, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);

            nLabel = d.Label;
            strLabel = m_imgDb.GetLabelName(nSrcId, nLabel);

            if (strLabel == null || strLabel.Length == 0)
                strLabel = nLabel.ToString();

            rgCriteria = d.DataCriteria;

            return new Bitmap(ImageData.GetImage(new Datum(d), null));
        }


        /// <summary>
        /// Retrives the image with a given ID.
        /// </summary>
        /// <param name="nImageID">Specifies the Raw Image ID.</param>
        /// <param name="nLabel">Returns the expected label for the image.</param>
        /// <param name="strLabel">Returns the expected label name for the image.</param>
        /// <param name="rgCriteria">Returns the data criteria if one exists.</param>
        /// <returns>The image queried is returned.</returns>
        public Bitmap GetTargetImage(int nImageID, out int nLabel, out string strLabel, out byte[] rgCriteria)
        {
            SimpleDatum d = m_imgDb.GetImage(nImageID, m_dataSet.TrainingSource.ID, m_dataSet.TestingSource.ID);

            nLabel = d.Label;
            strLabel = m_imgDb.GetLabelName(m_dataSet.TestingSource.ID, nLabel);

            if (strLabel == null || strLabel.Length == 0)
                strLabel = nLabel.ToString();

            rgCriteria = d.DataCriteria;

            return new Bitmap(ImageData.GetImage(new Datum(d), null));
        }

        /// <summary>
        /// Returns the image mean used by the solver network used during training.
        /// </summary>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        public SimpleDatum GetImageMean()
        {
            if (m_imgDb == null)
                throw new Exception("The image database is null!");

            if (m_solver == null)
                throw new Exception("The solver is null - make sure that you are loaded for training.");

            if (m_solver.net == null)
                throw new Exception("The solver net is null - make sure that you are loaded for training.");

            string strSrc = m_solver.net.GetDataSource();
            int nSrcId = m_imgDb.GetSourceID(strSrc);

            return m_imgDb.GetImageMean(nSrcId);
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
            return m_net.SaveWeights(false, m_persist);
        }

        /// <summary>
        /// Loads the weights from the training net into the Net used for running.
        /// </summary>
        public void UpdateRunWeights()
        {
            loadWeights(m_net, m_solver.net.SaveWeights(false, m_persist));
        }

        /// <summary>
        /// Loads the training Net with new weights.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights to load.</param>
        public void UpdateWeights(byte[] rgWeights)
        {
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
        /// <returns>The new Net is returned.</returns>
        public Net<T> GetNet(byte[] rgWeights)
        {
            NetParameter p = m_net.ToProto(false);
            Net<T> net = new Net<T>(m_cuda, m_log, p, m_evtCancel, m_imgDb);
            loadWeights(net, rgWeights);
            return net;
        }

        /// <summary>
        /// Returns the internal net based on the Phase specified: TRAIN, TEST or RUN.
        /// </summary>
        /// <param name="phase">Specifies the Phase used to select the Net.</param>
        /// <returns>The internal Net is returned.</returns>
        public Net<T> GetInternalNet(Phase phase = Phase.RUN)
        {
            if (phase == Phase.TEST)
                return m_solver.TestingNet;

            else if (phase == Phase.TRAIN)
                return m_solver.TrainingNet;

            return m_net;
        }

        /// <summary>
        /// The Snapshot function forces a snapshot to occur.
        /// </summary>
        public void Snapshot()
        {
            m_solver.Snapshot(true, false);
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
            string str = Properties.Resources.LICENSETXT;
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
    }
}
