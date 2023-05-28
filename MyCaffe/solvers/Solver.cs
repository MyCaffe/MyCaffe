using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO;
using System.Diagnostics;
using System.Collections;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;

/// <summary>
/// The MyCaffe.solvers namespace contains all solver classes, including the base Solver.
/// </summary>
namespace MyCaffe.solvers
{
    /// <summary>
    /// An interface for classes that perform optimization on Nets - this class serves as the base class for all solvers.
    /// </summary>
    /// <remarks>
    /// Requires implementation of ApplyUpdate to compute a parameter update
    /// given the current state of the Net parameters.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public abstract class Solver<T> : IDisposable
    {
        /// <summary>
        /// Specifies the instance of CudaDnn used by the Solver that provides a connection to Cuda.
        /// </summary>
        protected CudaDnn<T> m_cuda;
        /// <summary>
        /// Specifies the Log for output.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// Specifies the SolverParameter that defines how the Solver operates.
        /// </summary>
        protected SolverParameter m_param;
        /// <summary>
        /// Specifies the training Net.
        /// </summary>
        protected Net<T> m_net;
        /// <summary>
        /// Specifies the testing Nets.
        /// </summary>
        protected List<Net<T>> m_rgTestNets = new List<Net<T>>();
        /// <summary>
        /// Specifies the current iteration.
        /// </summary>
        protected int m_nIter;
        /// <summary>
        /// Specifies the current step.
        /// </summary>
        protected int m_nCurrentStep;
        /// <summary>
        /// Specifies the Losses used to calculate the smoothed Loss.
        /// </summary>
        protected List<double> m_rgLosses = new List<double>();
        AutoResetEvent m_evtCompleted = new AutoResetEvent(false);
        bool m_bEnableTest = true;
        bool m_bEnableBlobDebugging = false;
        bool m_bEnableBreakOnNan = false;
        bool m_bEnableDetailedNanDetection = false;
        bool m_bEnableSingleStep = false;
        /// <summary>
        /// Specifies the smoothed loss protected for derived classes to use.
        /// </summary>
        protected double m_dfSmoothedLoss = 0;
        /// <summary>
        /// Specifies the iteration accuracy calculated when a blob exists with the name 'accuracy'.
        /// </summary>
        protected double? m_dfIterAccuracy = null;
        Blob<T> m_blobAccuracy = null;
        CancelEvent m_evtCancel;
        AutoResetEvent m_evtForceSnapshot;
        AutoResetEvent m_evtForceTest;
        /// <summary>
        /// Specifies the Solver count in a multi-GPU training session.
        /// </summary>
        protected int m_nSolverCount = 1;
        /// <summary>
        /// Specifies the Solver rank of this solver, where rank == 0 is the root Solver.
        /// </summary>
        protected int m_nSolverRank = 0;
        /// <summary>
        /// Specifies the persistance object used to save weight and solver states.
        /// </summary>
        protected IXPersist<T> m_persist;
        /// <summary>
        /// Optionally, specifies a learning rate override (default = 0, which ignores this setting).
        /// </summary>
        protected double m_dfLearningRateOverride = 0;
        double m_dfLastAccuracy = 0;
        double m_dfLastError = double.MaxValue;
        double m_dfBestAccuracy = 0;
        double m_dfBestError = double.MaxValue;
        IXImageDatabaseBase m_db = null;
        int m_nTrainingIterationOverride = -1;
        int m_nTestingIterationOverride = -1;
        object m_tag = null;
        bool m_bWeightsUpdated = false;
        static object m_syncGetRi = new object();
        Blob<T> m_blobBatchInputData = null;
        double m_dfAverageTestTime = 0;
        SNAPSHOT_WEIGHT_UPDATE_METHOD m_snapshotWeightUpdatemMethod = SNAPSHOT_WEIGHT_UPDATE_METHOD.FAVOR_ACCURACY;
        int m_nTrainingTimeLimitInMinutes = 0;
        long m_hWorkspaceData = 0;  // shared among the layers and nets, only grows in size.
        ulong m_lWorkspaceSizeInBytes = 0;
        bool m_bFirstNanError = true;
        List<double> m_rgAverageAccuracyWindow = null;

        /// <summary>
        /// The OnStart event fires at the start of each training iteration.
        /// </summary>
        public event EventHandler OnStart;
        /// <summary>
        /// The OnAborted event fires after aborting a training cycle.
        /// </summary>
        public event EventHandler OnAborted;
        /// <summary>
        /// The OnGradientsReady event fires after the gradients of a Solver are ready for distribution to other Solvers in a multi-GPU training session.
        /// </summary>
        public event EventHandler<GradientsReadyArgs> OnGradientsReady;
        /// <summary>
        /// The OnSnapshot event fires when the Solver detects that a snapshot is needed.
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
        /// When specified, the OnTestResults event fires after each single test run.  The recipient is responsible for setting the accuracy.
        /// </summary>
        public event EventHandler<TestResultArgs<T>> OnTestResults;
        /// <summary>
        /// When specified, the OnTest event fires during a TestAll and overrides the call to Test.
        /// </summary>
        public event EventHandler<TestArgs> OnTest;
        /// <summary>
        /// The OnTestStart event fires at the start of each testing iteration.
        /// </summary>
        public event EventHandler OnTestStart;
        /// <summary>
        /// The OnCustomForwardBack allows for overriding the forward/backward operations within
        /// the solver.
        /// </summary>
        public event EventHandler<CustomForwardBackArgs<T>> OnCustomForwardBack;
        /// <summary>
        /// Specifies the OnGetWorkspace event that fires when the getWorkspace() function is called by a layer to get a shareable workspace to conserve GPU memory.
        /// </summary>
        public event EventHandler<WorkspaceArgs> OnGetWorkspace;
        /// <summary>
        /// Specifies the OnSetWorkspace event that fires when the setWorkspace() function is called by a layer to get a shareable workspace to conserve GPU memory.
        /// </summary>
        public event EventHandler<WorkspaceArgs> OnSetWorkspace;

        /// <summary>
        /// The Solver constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies teh SolverParameter.</param>
        /// <param name="evtCancel">Specifies a CancelEvent used to cancel the current operation (e.g. training, testing) for which the Solver is performing.</param>
        /// <param name="evtForceSnapshot">Specifies an automatic reset event that causes the Solver to perform a Snapshot when set.</param>
        /// <param name="evtForceTest">Specifies an automatic reset event that causes teh Solver to run a testing cycle when set.</param>
        /// <param name="imgDb">Specifies the MyCaffeImageDatabase.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPU session.</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver in a multi-GPU session.</param>
        /// <param name="shareNet">Optionally, specifies the net to share when creating the training network (default = null, meaning no share net is used).</param>
        /// <param name="getws">Optionally, specifies the handler for getting the workspace.</param>
        /// <param name="setws">Optionally, specifies the handler for setting the workspace.</param>
        public Solver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabaseBase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
        {
            m_cuda = cuda;
            m_log = log;
            m_evtCancel = evtCancel;
            m_evtForceSnapshot = evtForceSnapshot;
            m_evtForceTest = evtForceTest;

            if (m_log.IsEnabled)
                m_log.Enable = is_root_solver;

            m_db = imgDb;
            m_persist = persist;
            m_nSolverCount = nSolverCount;
            m_nSolverRank = nSolverRank;

            if (getws != null)
                OnGetWorkspace += new EventHandler<WorkspaceArgs>(getws);

            if (setws != null)
                OnSetWorkspace += new EventHandler<WorkspaceArgs>(setws);

            if (p.accuracy_average_window > 0)
            {
                m_rgAverageAccuracyWindow = new List<double>();
                for (int i = 0; i < p.accuracy_average_window; i++)
                {
                    m_rgAverageAccuracyWindow.Add(0);
                }
            }

            Init(p, shareNet);
        }

        /// <summary>
        /// Discards the resources (GPU and Host) used by this Solver.
        /// </summary>
        public void Dispose()
        {
            dispose();
        }

        /// <summary>
        /// Get/set the learning rate override.  When 0, this setting is ignored.
        /// </summary>
        public double LearningRateOverride
        {
            get { return m_dfLearningRateOverride; }
            set { m_dfLearningRateOverride = value; }
        }

        /// <summary>
        /// Force an OnTrainingIterationEvent to fire.
        /// </summary>
        /// <returns>When fired, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool ForceOnTrainingIterationEvent()
        {
            int nTimingCount = 0;
            double dfTotalTime = 0;
            return fireOnTrainingIterationEvent(false, 0, 0, ref nTimingCount, ref dfTotalTime);
        }

        private bool fireOnTrainingIterationEvent(bool bFwdPassNanFree, double dfLoss, double dfLastLearningRate, ref int nTimingCount, ref double dfTotalTime)
        {
            if (is_root_solver && OnTrainingIteration != null)
            {
                string strFirstNanBlob = null;
                DebugInformation<T> dbgInfo = null;

                if (m_bEnableBlobDebugging)
                {
                    dbgInfo = TrainingNet.GetDebugInformation(m_bEnableDetailedNanDetection);

                    if (m_bEnableBreakOnNan && dbgInfo != null)
                    {
                        string strType;
                        strFirstNanBlob = dbgInfo.DetectFirstNaN(out strType);

                        if (strFirstNanBlob != null)
                        {
                            string strPass = (!bFwdPassNanFree) ? "Forward" : "Backward";
                            m_log.WriteLine("First NaN detected in the '" + strType + "' of blob '" + strFirstNanBlob + "' after " + strPass + " pass.");

                            string strTypeLast;
                            string strLastNanBlob = dbgInfo.DetectLastNaN(out strTypeLast);

                            if (strLastNanBlob != strFirstNanBlob && strType != strTypeLast)
                                m_log.WriteLine("Last NaN detected in the '" + strTypeLast + "' of blob '" + strLastNanBlob + "' after " + strPass + " pass.");
                        }
                    }
                }

                double dfTime = (nTimingCount > 0) ? (dfTotalTime / nTimingCount) : 0;
                OnTrainingIteration(this, new TrainingIterationArgs<T>(m_nIter, m_dfLastAccuracy, dfLoss, m_dfSmoothedLoss, m_dfBestError, m_bWeightsUpdated, m_net.ActiveLabelCounts, m_net.LabelQueryHitPercents, m_net.LabelQueryEpochs, m_net.BoostQueryHitPercents, dfLastLearningRate, dfTime, dbgInfo));
                dfTotalTime = 0;
                nTimingCount = 0;

                if (strFirstNanBlob != null)
                {
                    m_log.WriteLine("Training is now stopping at iteration " + m_nIter.ToString("N0") + " as the first NaN has been detected ('" + strFirstNanBlob + "').");
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Get/set the training time limit in minutes.  When set to 0, no time limit is imposed on training.
        /// </summary>
        public int TrainingTimeLimitInMinutes
        {
            get { return m_nTrainingTimeLimitInMinutes; }
            set { m_nTrainingTimeLimitInMinutes = value; }
        }

        /// <summary>
        /// Get/set the snapshot weight update method.
        /// </summary>
        public SNAPSHOT_WEIGHT_UPDATE_METHOD SnapshotWeightUpdateMethod
        {
            get { return m_snapshotWeightUpdatemMethod; }
            set { m_snapshotWeightUpdatemMethod = value; }
        }

        /// <summary>
        /// Returns the MyCaffeImageDatabase used.
        /// </summary>
        public IXImageDatabaseBase Database
        {
            get { return m_db; }
        }

        /// <summary>
        /// Override that allows discarding of resources (GPU and Host) used by this Solver.
        /// </summary>
        protected virtual void dispose()
        {
            if (m_net != null)
            {
                m_net.Dispose();
                m_net = null;
            }

            foreach (Net<T> net in m_rgTestNets)
            {
                net.Dispose();
            }

            m_rgTestNets.Clear();

            if (m_blobBatchInputData != null)
            {
                m_blobBatchInputData.Dispose();
                m_blobBatchInputData = null;
            }

            if (m_hWorkspaceData != 0)
            {
                m_cuda.DisableGhostMemory();
                m_cuda.FreeMemory(m_hWorkspaceData);
                m_cuda.ResetGhostMemory();
                m_hWorkspaceData = 0;
                m_lWorkspaceSizeInBytes = 0;
            }
        }

        /// <summary>
        /// When enabled, the training cycle calls TestAll periodically based on the SolverParameter.  Otherwise testing is not performed.
        /// </summary>
        public bool EnableTesting
        {
            get { return m_bEnableTest; }
            set { m_bEnableTest = value; }
        }

        /// <summary>
        /// When enabled, the OnTrainingIteration event is set extra debugging information describing the state of each Blob used by the Solver.
        /// </summary>
        public bool EnableBlobDebugging
        {
            get { return m_bEnableBlobDebugging; }
            set { m_bEnableBlobDebugging = value; }
        }

        /// <summary>
        /// Enable/disable layer debugging which causes each layer to check for NAN/INF on each forward/backward pass and throw an exception when found.
        /// </summary>
        /// <remarks>
        /// This option dramatically slows down training and is only recommended during debugging.
        /// </remarks>
        public bool EnableLayerDebugging
        {
            get { return TrainingNet.EnableLayerDebugging; }
            set { TrainingNet.EnableLayerDebugging = value; }
        }

        /// <summary>
        /// When enabled (requires EnableBlobDebugging = <i>true</i>), the Solver immediately stop training upon detecting the first NaN withing the training Net.
        /// </summary>
        public bool EnableBreakOnFirstNaN
        {
            get { return m_bEnableBreakOnNan; }
            set { m_bEnableBreakOnNan = value; }
        }

        /// <summary>
        /// When enabled (requires EnableBlobDebugging = <i>true</i>), the detailed Nan (and Infinity) detection is perofmed on each blob when training Net.
        /// </summary>
        /// <remarks>
        /// IMPORTANT: The use of this setting is only recommended on TCC enabled drivers.
        /// @see [NVIDIA Tesla Compute Cluster (TCC) Help](http://docs.nvidia.com/gameworks/content/developertools/desktop/tesla_compute_cluster.htm)
        /// </remarks>
        public bool EnableDetailedNanDetection
        {
            get { return m_bEnableDetailedNanDetection; }
            set { m_bEnableDetailedNanDetection = value; }
        }

        /// <summary>
        /// When enabled (requires EnableBlobDebugging = <i>true</i>), the Solver only runs one training cycle.
        /// </summary>
        public bool EnableSingleStep
        {
            get { return m_bEnableSingleStep; }
            set { m_bEnableSingleStep = value; }
        }

        /// <summary>
        /// Get/set when the weights have been updated.
        /// </summary>
        public bool WeightsUpdated
        {
            get { return m_bWeightsUpdated; }
            set { m_bWeightsUpdated = value; }
        }

        /// <summary>
        /// Returns a generic tag associated with the Solver.
        /// </summary>
        public object Tag
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

        /// <summary>
        /// Returns the testing Net used by the solver.
        /// </summary>
        public Net<T> TestingNet
        {
            get
            {
                if (m_rgTestNets.Count == 0)
                    return null;

                return m_rgTestNets[0];
            }
        }

        /// <summary>
        /// Returns the training Net used by the solver.
        /// </summary>
        public Net<T> TrainingNet
        {
            get { return m_net; }
        }

        /// <summary>
        /// Initializes the Solver.
        /// </summary>
        /// <param name="p">Specifies the SolverParameters used to initialize the Solver.</param>
        /// <param name="shareNet">Optionally, specifies a network to share (default = null).</param>
        public void Init(SolverParameter p, Net<T> shareNet = null)
        {
            m_log.WriteLine("Initializing solver from parameters: " + p.DebugString());
            m_param = p;
            m_log.CHECK_GE(m_param.average_loss, 1, "Average loss should be non-negative and >= 1.0.");

            if (m_param.random_seed >= 0)
                m_cuda.rng_setseed(m_param.random_seed + m_nSolverRank);

            // Scaffolding code.
            InitTrainNet(shareNet);
            InitTestNets();

            if (is_root_solver)
                m_log.WriteLine("Solver scaffolding done.");

            Reset();
        }

        /// <summary>
        /// Reset the iterations of the net.
        /// </summary>
        public void Reset()
        {
            m_nIter = 0;
            m_nCurrentStep = 0;
        }

        /// <summary>
        /// Initializes the Net used by the solver for training.
        /// </summary>
        /// <param name="shareNet">Optionally, specifies a network to share (default = null).</param>
        protected void InitTrainNet(Net<T> shareNet = null)
        {
            try
            {
                int num_train_nets = ((m_param.net_param != null) ? 1 : 0) + ((m_param.train_net_param != null) ? 1 : 0);
                string field_names = "net_param, train_net_param";
                m_log.CHECK_GE(num_train_nets, 1, "SolverParameter must specify a train net using one of these fields: " + field_names);
                m_log.CHECK_LE(num_train_nets, 1, "SolverParameter must not contain more than one of these fields specifying a train_net: " + field_names);
                NetParameter net_param = null;

                if (m_param.train_net_param != null)
                {
                    m_log.WriteLine("Creating training net specified in train_net_param.");
                    net_param = m_param.train_net_param.Clone(true);
                }

                if (m_param.net_param != null)
                {
                    m_log.WriteLine("Creating training net specified in net_param.");
                    net_param = m_param.net_param.Clone(true);
                }

                // Set the correct NetState.  We start with the solver defaults (lowest
                // precedence); then, merge in any NetState specified by the net_param itself;
                // finally, merge in any NetState specified by the train-state (highest
                // precedence).
                NetState net_state = new NetState();
                net_state.phase = Phase.TRAIN;
                net_state.MergeFrom(net_param.state);
                net_state.MergeFrom(m_param.train_state);
                net_param.state = net_state;
                net_param.solver_count = m_nSolverCount;
                net_param.solver_rank = m_nSolverRank;
                m_net = new Net<T>(m_cuda, m_log, net_param, m_evtCancel, m_db, Phase.NONE, m_evtCompleted, shareNet, net_OnGetWorkspace, net_OnSetWorkspace);
                m_net.OnGetIteration += net_OnGetIteration;

                m_blobAccuracy = m_net.FindBlob("accuracy");
            }
            catch(Exception excpt)
            {
                throw new Exception("Initializing Training Net: " + excpt.Message);
            }
        }

        private void net_OnSetWorkspace(object sender, WorkspaceArgs e)
        {
            if (OnSetWorkspace != null)
            {
                OnSetWorkspace(sender, e);
                return;
            }

            m_cuda.DisableGhostMemory();

            if (e.WorkspaceSizeInBytes > m_lWorkspaceSizeInBytes)
            {
                m_lWorkspaceSizeInBytes = e.WorkspaceSizeInBytes;

                if (m_hWorkspaceData != 0)
                    m_cuda.FreeMemory(m_hWorkspaceData);

                ulong lCount = CudaDnn<T>.ConvertByteSizeToCount(m_lWorkspaceSizeInBytes);
                m_hWorkspaceData = m_cuda.AllocMemory((long)lCount);
            }

            m_cuda.ResetGhostMemory();
        }

        private void net_OnGetWorkspace(object sender, WorkspaceArgs e)
        {
            if (OnGetWorkspace != null)
            {
                OnGetWorkspace(sender, e);
                return;
            }

            e.WorkspaceData = m_hWorkspaceData;
            e.WorkspaceSizeInBytes = m_lWorkspaceSizeInBytes;
        }

        private void net_OnGetIteration(object sender, GetIterationArgs e)
        {
            e.SetIteration(Phase.TRAIN, m_nIter);
        }

        /// <summary>
        /// Initializes the Net used by the Solver for testing.
        /// </summary>
        protected void InitTestNets()
        {
            try
            {
                int num_generic_nets = ((m_param.net_param != null) ? 1 : 0);
                int num_test_net_params = m_param.test_net_param.Count;
                int num_test_nets = num_test_net_params;

                if (num_generic_nets > 0)
                    m_log.CHECK_GE(m_param.test_iter.Count, num_test_nets, "test_iter must be specified fore each test network.");
                else
                    m_log.CHECK_EQ(m_param.test_iter.Count, num_test_nets, "test_iter must be specified fore each test network.");

                // If we have a generic net (specified by net or net_param, rather than
                // test_net or test_net_param), we may have an unlimited number of actual
                // test networks -- the actual number is given by the number of remaining
                // test_iters after any test nets specified by test_net_param and/or test_net
                // are evaluated.
                int num_generic_net_instances = m_param.test_iter.Count - num_test_nets;
                int num_test_net_instances = num_test_nets + num_generic_net_instances;

                if (m_param.test_state.Count > 0)
                    m_log.CHECK_EQ(m_param.test_state.Count, num_test_net_instances, "test_state must be unspecified or specified once per test net.");

                if (num_test_net_instances > 0)
                    m_log.CHECK_GT(m_param.test_interval, 0, "The test interval must be greater than zero.");

                List<string> sources = new List<string>();
                List<NetParameter> net_params = new List<NetParameter>();

                for (int i = 0; i < num_test_net_params; i++)
                {
                    sources.Add("test_net_param");
                    net_params.Add(m_param.test_net_param[i].Clone());
                }

                int remaining_test_nets = m_param.test_iter.Count - num_test_net_params;

                if (m_param.net_param != null)
                {
                    for (int i = 0; i < remaining_test_nets; i++)
                    {
                        sources.Add("net_param");
                        net_params.Add(m_param.net_param.Clone());
                    }
                }

                m_rgTestNets = new List<Net<T>>();

                for (int i = 0; i < num_test_net_instances; i++)
                {
                    // Set the correct NetState. We start with the solver defaults (lowest
                    // precedence); then, merge in any NetState specified by the net_param
                    // itself; finally, merge in any NetState specified by the test_state
                    // (highest precedence).
                    NetState net_state = new NetState();
                    net_state.phase = Phase.TEST;
                    net_state.MergeFrom(net_params[i].state);

                    if (m_param.test_state.Count > 0)
                        net_state.MergeFrom(m_param.test_state[i]);

                    net_params[i].state = net_state;

                    m_log.WriteLine("Creating test net (#" + i.ToString() + ") specified by " + sources[i], true);
                    Net<T> net = new Net<T>(m_cuda, m_log, net_params[i], m_evtCancel, m_db, Phase.NONE, null, TrainingNet, net_OnGetWorkspace, net_OnSetWorkspace);

                    m_rgTestNets.Add(net);
                    m_rgTestNets[i].set_debug_info(m_param.debug_info);
                }
            }
            catch (Exception excpt)
            {
                throw new Exception("Initializing Testing Nets: " + excpt.Message);
            }
        }

        /// <summary>
        /// Returns the CudaDnn instance used by the Solver.
        /// </summary>
        public CudaDnn<T> Cuda
        {
            get { return m_cuda; }
        }

        /// <summary>
        /// Returns a string describing the labels detected in the training along with the % that each label has participated in the training.
        /// </summary>
        public string ActiveLabelCounts
        {
            get { return m_net.ActiveLabelCounts; }
        }

        /// <summary>
        /// Return the label query hit percentages for the active datasource.
        /// </summary>
        public string LabelQueryHitPercents
        {
            get { return m_net.LabelQueryHitPercents; }
        }

        /// <summary>
        /// Return the label query epochs for the active datasource.
        /// </summary>
        public string LabelQueryEpochs
        {
            get { return m_net.LabelQueryEpochs; }
        }

        /// <summary>
        /// Returns the current training iteration.
        /// </summary>
        public int CurrentIteration
        {
            get { return m_nIter; }
        }

        /// <summary>
        /// Returns the maximum training iterations.
        /// </summary>
        public int MaximumIteration
        {
            get { return m_param.max_iter; }
        }

        /// <summary>
        /// Returns the current training iterations remaining.
        /// </summary>
        public int TrainingIterations
        {
            get
            {
                int nIters = m_param.max_iter - m_nIter;

                if (m_nTrainingIterationOverride > 0)
                    nIters = m_nTrainingIterationOverride;

                return nIters;
            }
        }

        /// <summary>
        /// Returns the current testing iterations remaining.
        /// </summary>
        public int TestingIterations
        {
            get
            {
                int nIters = (m_param.test_iter.Count == 0) ? 0 : m_param.test_iter[0];

                if (m_nTestingIterationOverride > 0)
                    nIters = m_nTestingIterationOverride;

                return nIters;
            }
        }

        /// <summary>
        /// The main entry of the solver function.  In default, iter will be zero.  Pass
        /// in a non-zero iter number to resume training for a pre-trained net.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override value to use for the number of iterations run.  The default is -1, which ignores the parameter.</param>
        /// <param name="rgWeights">Optionally, specifies weights to load via the Restore method.  The default is <i>null</i> which ignores the parameter.</param>
        /// <param name="rgState">Optionally, specifies the state to load via the Restore method.  The default is <i>null</i> which ignores the parameter.</param>
        /// <param name="step">Optionally, specifies to single step the training pass - typically this is used during debugging. The default = <i>TRAIN_STEP.NONE</i> which runs the solver in the normal manner.</param>
        public virtual void Solve(int nIterationOverride = -1, byte[] rgWeights = null, byte[] rgState = null, TRAIN_STEP step = TRAIN_STEP.NONE)
        {
            m_log.CHECK(is_root_solver, "Solve is only supported by the root solver.");
            m_log.WriteLine("Solving " + m_net.name);
            m_log.WriteLine("Learing Rate Policy: " + m_param.lr_policy);

            if (rgWeights != null || rgState != null)
                Restore(rgWeights, rgState);

            // For a network that is trained by the solver, no bottom or top vecs
            // should be given, and we will just provide dummy vecs.
            int start_iter = m_nIter;

            if (nIterationOverride <= 0)
                nIterationOverride = TrainingIterations;

            if (!Step(nIterationOverride, step))
                return;

            // If we haven't already, save a snapshot after optimization, unless
            // overriden by setting snapshot_after_train = false.
            if (step == TRAIN_STEP.NONE && (m_param.snapshot_after_train && (m_param.snapshot == 0 || (m_nIter % m_param.snapshot) != 0)))
                Snapshot(false, true);
            else if (m_net.learnable_parameters.SnapshotRequested(true))
                Snapshot(true, false);

            if (m_evtCancel.WaitOne(0))
            {
                m_log.WriteLine("Optimization stopped early.");
                return;
            }

            // After the optimization is done, run an additional train and test pass to
            // display the train and test loss/outputs if appropriate (based on the 
            // display and test_interval settings, respectively).  Unlike in the rest of
            // training, for the train net we only run a forward pass as we've already
            // updated the parameters 'max_iter' times -- this final pass is only done to
            // display the loss, which is computed in the forward pass.
            if (m_param.display > 0 && (m_nIter % m_param.display) == 0)
            {
                double dfLoss;
                m_net.Forward(out dfLoss);

                UpdateSmoothedLoss(dfLoss, start_iter);
                m_log.WriteLine("Iteration " + m_nIter + ", loss = " + m_dfSmoothedLoss.ToString());
            }

            if (m_param.test_interval > 0 && (m_nIter % m_param.test_interval) == 0)
            {
                if (m_bEnableTest)
                    TestAll();
            }

            m_log.WriteLine("Optimization done.");

            if (m_blobBatchInputData != null)
            {
                m_blobBatchInputData.Dispose();
                m_blobBatchInputData = null;
            }
        }

        /// <summary>
        /// Steps a set of iterations through a training cycle.
        /// </summary>
        /// <param name="nIters">Specifies the number of steps to iterate.</param>
        /// <param name="step">Optionally, specifies to single step the training pass - typically this is used during debugging. The default = <i>TRAIN_STEP.NONE</i> for no stepping.</param>
        /// <param name="bZeroDiffs">Optionally, specifies whether or not to zero out the gradient diffs (default = <i>true</i>).</param>
        /// <param name="bApplyUpdates">Optionally, specifies to apply the gradient updates to the weights (default = <i>true</i>).</param>
        /// <param name="bDisableOutput">Optionally, disable the output to the log.</param>
        /// <param name="bDisableProgress">Optionally, disables the progress updating to the log.</param>
        /// <param name="dfLossOverride">Optionally, specifies a loss override which can be useful when using a backward step only.</param>
        /// <param name="bAllowSnapshot">Optionally, specifies whether or not a snapshot is allowed even during TRAIN_STEP.</param>
        /// <returns></returns>
        public bool Step(int nIters, TRAIN_STEP step = TRAIN_STEP.NONE, bool bZeroDiffs = true, bool bApplyUpdates = true, bool bDisableOutput = false, bool bDisableProgress = false, double? dfLossOverride = null, bool? bAllowSnapshot = null)
        {
            Exception err = null;

            try
            {
                BlobCollection<T> colBottom = new BlobCollection<T>();
                int start_iter = m_nIter;
                int stop_iter = m_nIter + nIters;

                m_rgLosses.Clear();
                m_dfSmoothedLoss = 0;

                // Break on first NaN is a debugging tool
                // that causes the network to stop training
                // right after a NaN is discovered either
                // just after the forward pass or just
                // after the backward pass.
                m_net.EnableBreakOnFirstNaN = m_bEnableBreakOnNan && m_bEnableBlobDebugging;
                m_net.EnableDetailedNanDetection = m_bEnableDetailedNanDetection & m_bEnableBlobDebugging;

                Stopwatch sw = new Stopwatch();
                sw.Start();

                Stopwatch swTimeout = new Stopwatch();
                swTimeout.Start();

                while (m_nIter < stop_iter && !m_evtCompleted.WaitOne(0))
                {
                    // zero-init the params.
                    if (bZeroDiffs)
                        m_net.ClearParamDiffs();

                    if (OnStart != null)
                        OnStart(this, new EventArgs());

                    if (step == TRAIN_STEP.NONE && (forceTest ||
                         (m_param.test_interval > 0 &&
                          (m_nIter % m_param.test_interval) == 0 &&
                          (m_nIter > 0 || m_param.test_initialization))))
                    {
                        if (m_bEnableTest && is_root_solver)
                            m_dfLastAccuracy = TestAll();

                        // Break out of the while loop because a stop was requested while testing.
                        if (m_evtCancel.WaitOne(0))
                            break;
                    }

                    // on_start currently not used, so no event added.
                    bool bDisplay1 = (is_root_solver && m_param.display > 0 && (m_nIter % m_param.display) == 0 && !bDisableOutput) ? true : false;
                    m_net.set_debug_info(bDisplay1 && m_param.debug_info);

                    // accumulate the loss and gradient
                    double dfLoss = 0;
                    double dfLossTotal = 0;
                    double? dfAccuracyTotal = null;
                    int nIterCount = 0;

                    Stopwatch swTiming = new Stopwatch();
                    double dfTotalTime = 0;
                    int nTimingCount = 0;
                    bool bFwdPassNanFree = true;

                    for (int i = 0; i < m_param.iter_size; i++)
                    {
                        double dfLocalLoss;
                        double? dfLocalAccuracy = null;

                        swTiming.Restart();

                        if (OnCustomForwardBack != null)
                        {
                            CustomForwardBackArgs<T> args = new CustomForwardBackArgs<T>(m_net, step);
                            OnCustomForwardBack(this, args);
                            bFwdPassNanFree = args.FwdPassNanFree;
                            dfLocalLoss = args.LocalLoss;
                        }
                        else
                        {
                            bFwdPassNanFree = m_net.ForwardBackward(colBottom, out dfLocalLoss, step);

                            if (m_blobAccuracy != null)
                                dfLocalAccuracy = Utility.ConvertVal<T>(m_blobAccuracy.GetData(0));                                
                        }

                        if (double.IsNaN(dfLocalLoss) || double.IsInfinity(dfLocalLoss))
                        {
                            if (m_bFirstNanError)
                            {
                                m_log.WriteError(new Exception("The local loss at iteration " + m_nIter.ToString() + " is invalid (NAN or INFINITY)!"));
                                m_bFirstNanError = false;
                            }
                        }
                        
                        if (dfLocalAccuracy.HasValue)
                        {
                            if (!dfAccuracyTotal.HasValue)
                                dfAccuracyTotal = 0;

                            dfAccuracyTotal = dfAccuracyTotal + dfLocalAccuracy.Value;
                        }

                        dfLossTotal += dfLocalLoss;
                        swTiming.Stop();

                        dfTotalTime += swTiming.Elapsed.TotalMilliseconds;
                        nTimingCount++;
                        nIterCount++;

                        if (!bFwdPassNanFree)
                            break;
                    }

                    dfLoss = dfLossTotal / nIterCount;
                    dfLoss = dfLossOverride.GetValueOrDefault(dfLoss);

                    if (dfAccuracyTotal.HasValue)
                        m_dfIterAccuracy = dfAccuracyTotal.Value / nIterCount;

                    // average the loss across iterations for smoothed reporting
                    UpdateSmoothedLoss(dfLoss, start_iter);

                    bool bDisplay = false;
                    if (!bDisplay1 && sw.ElapsedMilliseconds > 2000 && !bDisableOutput)
                    {
                        bDisplay = true;
                        m_bFirstNanError = true;
                        sw.Restart();
                    }

                    if (bDisplay && bDisplay1)
                    {
                        m_log.WriteLine("Iteration " + m_nIter.ToString() + ", loss = " + m_dfSmoothedLoss.ToString());

                        BlobCollection<T> colResult = m_net.output_blobs;
                        int score_index = 0;

                        if (is_root_solver)
                        {
                            for (int j = 0; j < colResult.Count; j++)
                            {
                                double[] result_vec = Utility.ConvertVec<T>(colResult[j].update_cpu_data());
                                int nIdx = m_net.output_blob_indices[j];
                                string output_name = m_net.blob_names[nIdx];
                                double loss_weight = m_net.blob_loss_weights[nIdx];
                                double dfTotalLossWeight = 0;
                                int nResultCount = colResult[j].count();

                                for (int k = 0; k < nResultCount; k++)
                                {
                                    if (!m_param.output_average_results)
                                    {
                                        string strOut = "";

                                        if (loss_weight != 0)
                                            strOut += " (* " + loss_weight.ToString() + " = " + (loss_weight * result_vec[k]).ToString() + " loss)";

                                        m_log.WriteLine("    Train net output #" + score_index.ToString() + ": " + output_name + " = " + result_vec[k].ToString() + strOut);
                                        score_index++;
                                    }
                                    else
                                    {
                                        dfTotalLossWeight += loss_weight * result_vec[k];
                                    }
                                }

                                if (m_param.output_average_results)
                                {
                                    double dfAverage = dfTotalLossWeight / nResultCount;
                                    m_log.WriteLine("  Average weighted score = " + dfAverage.ToString() + " for '" + output_name + "' - averaged over " + nResultCount.ToString("N0") + " results.");
                                }
                            }
                        }
                    }

                    if (OnGradientsReady != null && bFwdPassNanFree)
                        OnGradientsReady(this, new GradientsReadyArgs());

                    double dfLastLearningRate = 0;

                    if (step != TRAIN_STEP.FORWARD && bApplyUpdates)
                        dfLastLearningRate = ApplyUpdate(m_nIter);

                    if (m_evtCancel.WaitOne(0))
                        break;

                    if (!bDisableProgress)
                        m_log.Progress = (double)m_nIter / (double)stop_iter;

                    bool bSnapshotTaken = false;
                    bool bForceSnapshot = forceSnapshot;

                    if ((step == TRAIN_STEP.NONE || bAllowSnapshot.GetValueOrDefault(false)) && (is_root_solver && bFwdPassNanFree &&
                        (bForceSnapshot ||
                         (m_param.snapshot > 0 && (m_nIter % m_param.snapshot) == 0) ||
                         (m_dfLastAccuracy > m_dfBestAccuracy))))
                    {
                        bSnapshotTaken = true;
                        Snapshot(bForceSnapshot, ((m_param.snapshot > 0 && (m_nIter % m_param.snapshot) == 0)) ? true : false);

                        if (m_dfLastAccuracy > m_dfBestAccuracy)
                            m_dfBestAccuracy = m_dfLastAccuracy;
                    }

                    //-------------------------------------
                    //  Call the training iteration event
                    //  on the root solver.  
                    //-------------------------------------
                    fireOnTrainingIterationEvent(bFwdPassNanFree, dfLoss, dfLastLearningRate, ref nTimingCount, ref dfTotalTime);

                    //-------------------------------------
                    //  If single stepping, stop the solver.
                    //-------------------------------------
                    if (step != TRAIN_STEP.NONE || m_bEnableSingleStep)
                    {
                        if (step == TRAIN_STEP.BOTH)
                        {
                            if (!bDisableOutput)
                                m_log.WriteLine("Single step (both) triggered - solving stopped after a single forward/backward pass.");
                        }
                        else if (step == TRAIN_STEP.FORWARD)
                        {
                            if (!bDisableOutput)
                                m_log.WriteLine("Single step (forward) triggered - solving stopped after a single forward pass.");
                        }
                        else if (step == TRAIN_STEP.BACKWARD)
                        {
                            if (!bDisableOutput)
                                m_log.WriteLine("Single step (backward) triggered - solving stopped after a single backward pass.");
                        }
                        else
                        {
                            // When single stepping, force the snapshot so as to allow
                            //  debugging the net visually.
                            if (!bSnapshotTaken)
                                Snapshot(true, false);
                        }
                        break;
                    }

                    //-------------------------------------
                    //  If a time-limit has been imposed
                    //  and we have exceeded it, stop
                    //  training.
                    //-------------------------------------
                    if (m_nTrainingTimeLimitInMinutes > 0 && swTimeout.Elapsed.TotalMinutes > m_nTrainingTimeLimitInMinutes)
                    {
                        m_log.WriteLine("A training time-limit of " + m_nTrainingTimeLimitInMinutes.ToString("N0") + " minutes has been exceeded - training will now stop.");
                        return true;
                    }

                    if (!bApplyUpdates)
                        break;
                }

                return true;
            }
            catch (Exception excpt)
            {
                err = excpt;
                throw excpt;
            }
            finally
            {
                if (err != null || m_evtCancel.WaitOne(0))
                {
                    if (OnAborted != null)
                        OnAborted(this, new EventArgs());
                }
            }
        }

        /// <summary>
        /// The restore method simply calls the RestoreSolverState method of the inherited class.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights to load, or <i>null</i> to ignore.</param>
        /// <param name="rgState">Specifies the state to load, or <i>null</i> to ignore.</param>
        /// <param name="strSkipBlobTypes">Specifies the blob types to ignore and not load, or <i>null</i> to ignore.</param>
        public void Restore(byte[] rgWeights, byte[] rgState, string strSkipBlobTypes = null)
        {
            m_net.LoadWeights(rgWeights, m_persist, null, null, strSkipBlobTypes);

            if (rgState != null)
            {
                m_log.WriteLine("Restoring previous solver state from restore state...");
                RestoreSolverState(rgState);
            }
        }

        /// <summary>
        /// The snapshot function implements the basic snapshotting utility that stores the
        /// learned net.  This method calls the SnapshotSolverState method of the inherited class.
        /// </summary>
        /// <param name="bForced">Specifies whehter or not to force the snapshot.</param>
        /// <param name="bScheduled">Specifies whether or not the snapshot is a scheduled snapshot that occurs at regular intervals, or a snapshot based on an improving accuracy.</param>
        /// <param name="bUpdateDatabase">Optionally, specifies to update the database (default = true).</param>
        public void Snapshot(bool bForced, bool bScheduled, bool bUpdateDatabase = true)
        {
            m_log.WriteLine("Starting snap shot...");
            m_log.CHECK(is_root_solver, "Snapshot only supported on the root solver.");

            if (OnSnapshot == null)
                return;

            if (m_snapshotWeightUpdatemMethod == SNAPSHOT_WEIGHT_UPDATE_METHOD.DISABLED && !bForced)
            {
                m_log.WriteLine("WARNING: Snapshot UPDATE_METHOD = DISABLED.");
                return;
            }

            SnapshotArgs args = GetSnapshotArgs(null, null, m_dfLastAccuracy, m_dfLastError, m_nIter, m_snapshotWeightUpdatemMethod);
            args.Forced = bForced;
            args.Scheduled = bScheduled;
            args.UpdateDatabase = bUpdateDatabase;

            OnSnapshot(this, args);
            m_log.WriteLine("Snapshot completed.");
        }

        private void args_OnGetWeights(object sender, GetBytesArgs e)
        {
            if (m_net != null)
                e.Data = m_net.SaveWeights(m_persist, m_param.snapshot_diff);
        }

        private void args_OnGetState(object sender, GetBytesArgs e)
        {
            e.Data = SnapshotSolverState();
        }

        /// <summary>
        /// The GetSnapshotArgs method fills out a snapshot args structure.
        /// </summary>
        /// <param name="rgState">Specifies the state bytes or null.</param>
        /// <param name="rgWeights">Specifies the weight bytes or null.</param>
        /// <param name="dfAccuracy">Specifies the accuracy.</param>
        /// <param name="dfError">Specifies the error.</param>
        /// <param name="nIteration">Specifies the interation.</param>
        /// <param name="wtUpdt">Specifies the weight update method.</param>
        /// <returns>The args are returned.</returns>
        public SnapshotArgs GetSnapshotArgs(byte[] rgState, byte[] rgWeights, double dfAccuracy, double dfError, int nIteration, SNAPSHOT_WEIGHT_UPDATE_METHOD wtUpdt)
        {
            if (dfAccuracy == 0)
                dfAccuracy = 0.0001;

            SnapshotArgs args = new SnapshotArgs(rgState, rgWeights, dfAccuracy, dfError, nIteration, wtUpdt);

            args.IncludeState = m_param.snapshot_include_state;
            args.IncludeWeights = m_param.snapshot_include_weights;
            args.SingleStep = m_bEnableSingleStep;
            args.OnGetState += args_OnGetState;
            args.OnGetWeights += args_OnGetWeights;

            return args;
        }

        /// <summary>
        /// Get/set the training iteration override.
        /// </summary>
        public int TrainingIterationOverride
        {
            get { return m_nTrainingIterationOverride; }
            set { m_nTrainingIterationOverride = value; }
        }

        /// <summary>
        /// Get/set the testing iteration override.
        /// </summary>
        public int TestingIterationOverride
        {
            get { return m_nTestingIterationOverride; }
            set { m_nTestingIterationOverride = value; }
        }

        /// <summary>
        /// Returns an auto reset event that is set upon training completion.
        /// </summary>
        public AutoResetEvent CompletedEvent
        {
            get { return m_evtCompleted; }
        }

        /// <summary>
        /// Returns the cancel event which when set cancels the current operation run by the Solver.
        /// </summary>
        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        /// <summary>
        /// Returns the smoothed loss.
        /// </summary>
        public double smoothed_loss
        {
            get { return m_dfSmoothedLoss; }
        }

        /// <summary>
        /// Returns the SolverParameter used.
        /// </summary>
        public SolverParameter parameter
        {
            get { return m_param; }
        }

        /// <summary>
        /// Returns the main training Net.
        /// </summary>
        public Net<T> net
        {
            get { return m_net; }
        }

        /// <summary>
        /// Returns the testing Nets.
        /// </summary>
        public List<Net<T>> test_nets
        {
            get { return m_rgTestNets; }
        }

        /// <summary>
        /// Returns the current training iteration.
        /// </summary>
        public int iter
        {
            get { return m_nIter; }
        }

        /// <summary>
        /// Returns the type of solver.
        /// </summary>
        public SolverParameter.SolverType type
        {
            get { return m_param.type; }
        }

        /// <summary>
        /// Returns whether or not a snapshot has been forced.
        /// </summary>
        protected bool forceSnapshot
        {
            get
            {
                if (m_evtForceSnapshot == null)
                    return false;

                return m_evtForceSnapshot.WaitOne(0);
            }
        }

        /// <summary>
        /// Returns whether or not a test has been forced.
        /// </summary>
        public bool forceTest
        {
            get
            {
                if (m_evtForceTest == null)
                    return false;

                return m_evtForceTest.WaitOne(0);
            }
        }

        /// <summary>
        /// Returns the solver count in a multi-GPU session.
        /// </summary>
        public int solver_count
        {
            get { return m_nSolverCount; }
        }

        /// <summary>
        /// Returns this Solver's rank in a multi-GPU session.
        /// </summary>
        public int solver_rank
        {
            get { return m_nSolverRank;  }
        }

        /// <summary>
        /// Returns whether or not this is the root solver. 
        /// </summary>
        /// <remarks>
        /// The root solver has rank = 0.
        /// </remarks>
        public bool is_root_solver
        {
            get { return (m_nSolverRank == 0) ? true : false; }
        }

        /// <summary>
        /// Run a TestAll by running all test Nets.
        /// </summary>
        /// <remarks>
        /// Depending on the eval_type tests are run as Test (default), TestClassification (for SSD Classification),
        /// or TestDetection (for SSD Detection).
        /// </remarks>
        /// <param name="nIterationOverride">Specifies an override to the iterations to run.</param>
        /// <returns>The accuracy of the testing run is returned as a percentage value in the range [0, 1].</returns>
        public double TestAll(int nIterationOverride = -1)
        {
            double dfTotalAccuracy = 0;
            double dfTotalTime = 0;
            int nTotalCount = 0;

            for (int test_net_id = 0; test_net_id < m_rgTestNets.Count; test_net_id++)
            {
                if (m_evtCancel.WaitOne(0))
                    return 0;

                if (OnTest != null)
                {
                    TestArgs args = new TestArgs(nIterationOverride, test_net_id);
                    OnTest(this, args);
                    dfTotalAccuracy += args.Accuracy;
                }
                else
                    dfTotalAccuracy += testOne(nIterationOverride, test_net_id);

                dfTotalTime += m_dfAverageTestTime;
                nTotalCount++;
            }

            if (m_rgTestNets.Count == 0)
            {
                if (OnTest != null)
                {
                    TestArgs args = new TestArgs(nIterationOverride, 0);
                    OnTest(this, args);
                    dfTotalAccuracy += args.Accuracy;
                }
                else
                    dfTotalAccuracy += testOne(nIterationOverride, 0);
            }

            double dfAccuracy = (m_rgTestNets.Count > 0) ? dfTotalAccuracy / m_rgTestNets.Count : 0;

            if (m_rgAverageAccuracyWindow != null)
            {
                m_rgAverageAccuracyWindow.Add(dfAccuracy);
                m_rgAverageAccuracyWindow.RemoveAt(0);
                dfAccuracy = m_rgAverageAccuracyWindow.Average();
            }

            if (OnTestingIteration != null)
            {
                double dfTime = (nTotalCount > 0) ? dfTotalTime / nTotalCount : 0;
                OnTestingIteration(this, new TestingIterationArgs<T>(m_nIter, dfAccuracy, dfTime));
            }

            return dfAccuracy;
        }

        private double testOne(int nIterationOverride = -1, int nTestNetId = 0)
        {
            switch (m_param.eval_type)
            {
                // Test SSD Detection
                case SolverParameter.EvaluationType.DETECTION:
                    return TestDetection(nIterationOverride, nTestNetId);

                // Perform regular classification Test.
                default:
                    return TestClassification(nIterationOverride, nTestNetId);
            }
        }

        /// <summary>
        /// Run an SSD detection test on a given test Net by running it through its iterations.
        /// </summary>
        /// <param name="nIterationOverride">Specifies an override the the number of iterations to run.</param>
        /// <param name="nTestNetId">Specifies the ID of the test Net to run.</param>
        /// <returns>The accuracy of the test run is returned as a percentage in the range [0, 1].</returns>
        public double TestDetection(int nIterationOverride = -1, int nTestNetId = 0)
        {
            Stopwatch sw = new Stopwatch();
            BBoxUtility<T> bboxUtil = new BBoxUtility<T>(m_cuda, m_log);

            try
            {
                if (is_root_solver)
                    m_log.WriteLine("Iteration " + m_nIter.ToString() + ", Testing net (#" + nTestNetId.ToString() + ")");

                Net<T> test_net = m_net;

                if (m_rgTestNets.Count > nTestNetId)
                {
                    m_log.CHECK(m_rgTestNets[nTestNetId] != null, "The test net at " + nTestNetId.ToString() + " is null!");
                    m_rgTestNets[nTestNetId].ShareTrainedLayersWith(m_net);
                    test_net = m_rgTestNets[nTestNetId];
                }

                Dictionary<int, Dictionary<int, List<Tuple<float, int>>>> rgAllTruePos = new Dictionary<int, Dictionary<int, List<Tuple<float, int>>>>();
                Dictionary<int, Dictionary<int, List<Tuple<float, int>>>> rgAllFalsePos = new Dictionary<int, Dictionary<int, List<Tuple<float, int>>>>();
                Dictionary<int, Dictionary<int, int>> rgAllNumPos = new Dictionary<int, Dictionary<int, int>>();

                double dfLoss = 0;

                if (nIterationOverride <= 0)
                    nIterationOverride = TestingIterations;

                int nIter = nIterationOverride;
                sw.Start();

                for (int i = 0; i < nIter; i++)
                {
                    // Check to see if stoppage of testing/training has been requested.
                    if (m_evtCancel.WaitOne(0))
                        break;

                    if (OnTestStart != null)
                        OnTestStart(this, new EventArgs());

                    double iter_loss;
                    BlobCollection<T> colResult = test_net.Forward(out iter_loss);

                    if (m_param.test_compute_loss)
                        dfLoss += iter_loss;

                    for (int j = 0; j < colResult.Count; j++)
                    {
                        m_log.CHECK_EQ(colResult[j].width, 5, "The width must be = 5 for SSD.");
                        double[] result_vec = Utility.ConvertVec<T>(colResult[j].update_cpu_data());
                        int num_det = colResult[j].height;

                        for (int k = 0; k < num_det; k++)
                        {
                            int item_id = (int)result_vec[k * 5];
                            int nLabel = (int)result_vec[k * 5 + 1];

                            // Special row for storing number of positives for a label.
                            if (item_id == -1)
                            {
                                if (!rgAllNumPos.ContainsKey(j))
                                    rgAllNumPos.Add(j, new Dictionary<int, int>());

                                if (!rgAllNumPos[j].ContainsKey(nLabel))
                                    rgAllNumPos[j].Add(nLabel, (int)result_vec[k * 5 + 2]);
                                else
                                    rgAllNumPos[j][nLabel] += (int)result_vec[k * 5 + 2];
                            }
                            // Normal row storing detection status.
                            else
                            {
                                float fScore = (float)result_vec[k * 5 + 2];
                                int tp = (int)result_vec[k * 5 + 3];
                                int fp = (int)result_vec[k * 5 + 4];

                                // Ignore such case, which happens when a detection bbox is matched to
                                // a difficult gt bbox and we don't evaluate on difficult gt bbox.
                                if (tp == 0 && fp == 0)
                                    continue;

                                if (!rgAllTruePos.ContainsKey(j))
                                    rgAllTruePos.Add(j, new Dictionary<int, List<Tuple<float, int>>>());

                                if (!rgAllTruePos[j].ContainsKey(nLabel))
                                    rgAllTruePos[j].Add(nLabel, new List<Tuple<float, int>>());

                                if (!rgAllFalsePos.ContainsKey(j))
                                    rgAllFalsePos.Add(j, new Dictionary<int, List<Tuple<float, int>>>());

                                if (!rgAllFalsePos[j].ContainsKey(nLabel))
                                    rgAllFalsePos[j].Add(nLabel, new List<Tuple<float, int>>());

                                rgAllTruePos[j][nLabel].Add(new Tuple<float, int>(fScore, tp));
                                rgAllFalsePos[j][nLabel].Add(new Tuple<float, int>(fScore, fp));
                            }
                        }
                    }

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        m_log.Progress = (double)i / (double)nIter;
                        m_log.WriteLine("Testing at " + m_log.Progress.ToString("P") + " " + i.ToString() + " of " + nIter.ToString() + "...");
                        sw.Restart();
                    }
                }

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Test interrupted.");
                    return 0;
                }

                if (m_param.test_compute_loss)
                {
                    dfLoss /= m_param.test_iter[nTestNetId];
                    m_log.WriteLine("Test loss: " + dfLoss.ToString());
                }

                float fTotalmAP = 0;
                for (int i = 0; i < rgAllTruePos.Count; i++)
                {
                    if (!rgAllTruePos.ContainsKey(i))
                        m_log.FAIL("Missing output_blob true_pos: " + i.ToString());

                    Dictionary<int, List<Tuple<float, int>>> rgTruePos = rgAllTruePos[i];

                    if (!rgAllFalsePos.ContainsKey(i))
                        m_log.FAIL("Missing output_blob false_pos: " + i.ToString());

                    Dictionary<int, List<Tuple<float, int>>> rgFalsePos = rgAllFalsePos[i];

                    if (!rgAllNumPos.ContainsKey(i))
                        m_log.FAIL("Missing output_blob num_pos: " + i.ToString());

                    Dictionary<int, int> rgNumPos = rgAllNumPos[i];

                    Dictionary<int, float> rgAPs = new Dictionary<int, float>();
                    float fmAP = 0.0f;

                    // Sort true_pos and false_pos with descending scores.
                    foreach (KeyValuePair<int, int> kv in rgNumPos)
                    {
                        int nLabel = kv.Key;
                        int nLabelNumPos = kv.Value;

                        if (!rgTruePos.ContainsKey(nLabel))
                        {
                            m_log.WriteLine("WARNING: Missing true_pos for label: " + nLabel.ToString() + "!");
                            continue;
                        }
                        List<Tuple<float, int>> rgLabelTruePos = rgTruePos[nLabel];

                        if (!rgFalsePos.ContainsKey(nLabel))
                        {
                            m_log.WriteLine("WARNING: Missing false_pos for label: " + nLabel.ToString() + "!");
                            continue;
                        }
                        List<Tuple<float, int>> rgLabelFalsePos = rgFalsePos[nLabel];

                        List<float> rgPrec;
                        List<float> rgRec;
                        float fAp = bboxUtil.ComputeAP(rgLabelTruePos, nLabelNumPos, rgLabelFalsePos, m_param.ap_version, out rgPrec, out rgRec);

                        if (!rgAPs.ContainsKey(nLabel))
                            rgAPs.Add(nLabel, fAp);
                        else
                            rgAPs[nLabel] = fAp;

                        fmAP += fAp;

                        if (m_param.show_per_class_result)
                            m_log.WriteLine("class " + nLabel.ToString() + ": " + fAp.ToString());
                    }

                    fmAP /= rgNumPos.Count;

                    int nOutputBlobIdx = test_net.output_blob_indices[i];
                    string strOutputName = test_net.blob_names[nOutputBlobIdx];

                    m_log.WriteLine("    Test net output #" + i.ToString() + ": " + strOutputName + " = " + fmAP.ToString());
                    fTotalmAP += fmAP;
                }

                return fTotalmAP / rgAllTruePos.Count;
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                bboxUtil.Dispose();
            }
        }

        /// <summary>
        /// Run a test on a given test Net by running it through its iterations.
        /// </summary>
        /// <param name="nIterationOverride">Specifies an override the the number of iterations to run.</param>
        /// <param name="nTestNetId">Specifies the ID of the test Net to run.</param>
        /// <returns>The accuracy of the test run is returned as a percentage in the range [0, 1].</returns>
        public double TestClassification(int nIterationOverride = -1, int nTestNetId = 0)
        {
            bool bDisplay = (is_root_solver && m_param.display > 0 && (m_nIter % m_param.display) == 0) ? true : false;

            if (bDisplay)
                m_log.WriteLine("Iteration " + m_nIter.ToString() + ", Testing net (#" + nTestNetId.ToString() + ")");

            Net<T> test_net = m_net;

            if (m_rgTestNets.Count > nTestNetId)
            {
                m_log.CHECK(m_rgTestNets[nTestNetId] != null, "The test net at " + nTestNetId.ToString() + " is null!");
                m_rgTestNets[nTestNetId].ShareTrainedLayersWith(m_net);
                test_net = m_rgTestNets[nTestNetId];
            }

            List<double> test_score = new List<double>();
            List<int> test_score_output_id = new List<int>();
            double dfLoss = 0;

            if (nIterationOverride <= 0)
                nIterationOverride = TestingIterations;

            int nIter = nIterationOverride;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            double dfTotalTiming = 0;
            int nTestCount = 0;
            int nAccuracyIdx = 0;
            int nMinRank = int.MaxValue;
            bool bAccuracyValid = false;
            Stopwatch swTiming = new Stopwatch();

            for (int i = 0; i < nIter; i++)
            {
                // Check to see if stoppage of testing/training has been requested.
                if (m_evtCancel.WaitOne(0))
                    break;

                if (OnTestStart != null)
                    OnTestStart(this, new EventArgs());

                swTiming.Restart();

                double iter_loss;
                BlobCollection<T> colResult = test_net.Forward(out iter_loss);

                if (m_param.test_compute_loss)
                    dfLoss += iter_loss;

                TestResultArgs<T> args = new TestResultArgs<T>(colResult);
                if (OnTestResults != null)
                {
                    OnTestResults(this, args);
                    if (args.AccuracyValid)
                    {
                        test_score.Add(args.Accuracy);
                        test_score_output_id.Add(1);
                        bAccuracyValid = true;
                    }
                }

                if (!args.AccuracyValid)
                {
                    if (i == 0)
                    {
                        for (int j = 0; j < colResult.Count; j++)
                        {
                            double[] result_vec = Utility.ConvertVec<T>(colResult[j].update_cpu_data());

                            for (int k = 0; k < colResult[j].count(); k++)
                            {
                                test_score.Add(result_vec[k]);
                                test_score_output_id.Add(j);
                            }

                            if (colResult[j].type == BLOB_TYPE.ACCURACY)
                            {
                                int nRank = (int)getNumber(colResult[j].Tag, 0);
                                if (nRank < nMinRank)
                                {
                                    nMinRank = nRank;
                                    nAccuracyIdx = j;
                                }
                            }
                        }
                    }
                    else
                    {
                        int idx = 0;

                        for (int j = 0; j < colResult.Count; j++)
                        {
                            double[] result_vec = Utility.ConvertVec<T>(colResult[j].update_cpu_data());

                            for (int k = 0; k < colResult[j].count(); k++)
                            {
                                test_score[idx] += result_vec[k];
                                idx++;
                            }
                        }
                    }
                }

                swTiming.Stop();
                dfTotalTiming += swTiming.Elapsed.TotalMilliseconds;
                nTestCount++;

                if (sw.ElapsedMilliseconds > 2000)
                {
                    double dfPct = (double)i / (double)nIter;

                    if (bDisplay)
                    {
                        m_log.Progress = dfPct;
                        m_log.WriteLine("Testing '" + test_net.name + "' at " + dfPct.ToString("P"));
                    }

                    sw.Restart();
                }
            }

            m_dfAverageTestTime = (nTestCount > 0) ? dfTotalTiming / nTestCount : 0;

            if (m_evtCancel.WaitOne(0))
            {
                m_log.WriteLine("Test interrupted.");
                return 0;
            }

            if (m_param.test_compute_loss)
            {
                dfLoss /= m_param.test_iter[nTestNetId];
                m_log.WriteLine("Test loss: " + dfLoss.ToString());
            }

            double dfFinalScore = 0;

            if (bAccuracyValid)
            {
                dfFinalScore = test_score.Sum();
                int nTotal = test_score_output_id.Sum();
                dfFinalScore /= nTotal;
            }
            else
            {
                for (int i = 0; i < test_score.Count; i++)
                {
                    int nIdxTestScore = test_score_output_id[i];
                    int output_blob_index = test_net.output_blob_indices[nIdxTestScore];
                    string output_name = test_net.blob_names[output_blob_index];
                    double loss_weight = test_net.blob_loss_weights[output_blob_index];
                    double dfMeanScore = test_score[i] / nIter;
                    string strOut = "";

                    if (bDisplay)
                    {
                        if (loss_weight != 0)
                            strOut += " (* " + loss_weight.ToString() + " = " + (loss_weight * dfMeanScore).ToString() + " loss)";

                        m_log.WriteLine("   Test net output #" + i.ToString() + ": " + output_name + " = " + dfMeanScore.ToString() + strOut);
                    }

                    if (i == nAccuracyIdx)
                        dfFinalScore = dfMeanScore;
                }
            }

            if (test_score.Count == 0)
                return 0;

            return dfFinalScore;
        }

        private double getNumber(object value, double dfDefault)
        {
            if (value == null)
                return dfDefault;

            if (value is sbyte)
                return (double)(sbyte)value;

            if (value is byte)
                return (double)(byte)value;

            if (value is short)
                return (double)(short)value;

            if (value is ushort)
                return (double)(ushort)value;

            if (value is int)
                return (double)(int)value;

            if (value is uint)
                return (double)(uint)value;

            if (value is long)
                return (double)(long)value;

            if (value is ulong)
                return (double)(ulong)value;

            if (value is float)
                return (double)(float)value;

            if (value is double)
                return (double)value;

            if (value is decimal)
                return (double)(decimal)value;

            return dfDefault;
        }

        /// <summary>
        /// Update the avaraged loss value.
        /// </summary>
        /// <param name="dfLoss">Specifies the new loss value to add into the average.</param>
        /// <param name="nStartIter">Specifies the starting iteration.</param>
        /// <param name="nAverageLoss">Optionally, specifies the number of iterations to average over (default = param.average_loss).</param>
        public void UpdateSmoothedLoss(double dfLoss, int nStartIter, int nAverageLoss = 0)
        {
            if (nAverageLoss == 0)
                nAverageLoss = m_param.average_loss;

            if (m_rgLosses.Count < nAverageLoss)
            {
                m_rgLosses.Add(dfLoss);
                int nCount = m_rgLosses.Count;
                m_dfSmoothedLoss = (m_dfSmoothedLoss * (nCount - 1) + dfLoss) / nCount;
            }
            else
            {
                int nIdx = (m_nIter - nStartIter) % nAverageLoss;
                m_dfSmoothedLoss += (dfLoss - m_rgLosses[nIdx]) / nAverageLoss;
                m_rgLosses[nIdx] = dfLoss;
            }

            if (m_bWeightsUpdated)
            {
                m_dfSmoothedLoss = dfLoss;
                m_bWeightsUpdated = false;
            }

            m_dfLastError = m_dfSmoothedLoss;

            if (m_dfLastError < m_dfBestError)
                m_dfBestError = m_dfLastError;
        }

        /// <summary>
        /// Make and apply the update value for the current iteration.
        /// </summary>
        /// <param name="nIterationOverride">Optionally, specifies an iteration override, or -1 which is ignored.</param>
        public abstract double ApplyUpdate(int nIterationOverride = -1);

        /// <summary>
        /// Save the current solver state.
        /// </summary>
        protected abstract byte[] SnapshotSolverState();

        /// <summary>
        /// Restore a solver state.
        /// </summary>
        protected abstract void RestoreSolverState(byte[] rgState);

        /// <summary>
        /// Create a new Solver based on the project containing the SolverParameter.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance that the new Solver will use.</param>
        /// <param name="log">Specifies the Log for output that the new Solver will use.</param>
        /// <param name="p">Specifies the project to use in initialzing the new Solver.</param>
        /// <param name="evtCancel">Specifies the CancelEvent that the new Solver will use.</param>
        /// <param name="evtForceSnapshot">Specifies the force snapshot event that the new Solver will use.</param>
        /// <param name="evtForceTest">Specifies the force test event that the new Solver will use.</param>
        /// <param name="imgDb">Specifies the MyCaffeImageDatabase that the new Solver will use.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPu session.</param>
        /// <param name="nSolverRank">Specifies the rank of the new Solver.</param>
        /// <param name="shareNet">Optionally, specifies the net to share when creating the training network (default = null, meaning no share net is used).</param>
        /// <param name="getws">Optionally, specifies the handler for getting the workspace.</param>
        /// <param name="setws">Optionally, specifies the handler for setting the workspace.</param>
        /// <returns>A new Solver instance is returned.</returns>
        public static SGDSolver<T> Create(CudaDnn<T> cuda, Log log, ProjectEx p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabaseBase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
        {
            SolverParameter solverParam = null;

            if (p.SolverDescription != null)
            {
                RawProto protoSolver = RawProto.Parse(p.SolverDescription);
                solverParam = SolverParameter.FromProto(protoSolver);
            }
            else
            {
                solverParam = new param.SolverParameter();
            }

            if (solverParam.net_param == null)
            {
                RawProto protoModel = RawProto.Parse(p.ModelDescription);
                solverParam.net_param = NetParameter.FromProto(protoModel);
                solverParam.net_param.ProjectID = p.ID;
            }

            return Create(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
        }

        /// <summary>
        /// Create a new Solver based on the project containing the SolverParameter.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance that the new Solver will use.</param>
        /// <param name="log">Specifies the Log for output that the new Solver will use.</param>
        /// <param name="solverParam">Specifies the SolverParameter used to create the Solver.</param>
        /// <param name="evtCancel">Specifies the CancelEvent that the new Solver will use.</param>
        /// <param name="evtForceSnapshot">Specifies the force snapshot event that the new Solver will use.</param>
        /// <param name="evtForceTest">Specifies the force test event that the new Solver will use.</param>
        /// <param name="imgDb">Specifies the MyCaffeImageDatabase that the new Solver will use.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPu session.</param>
        /// <param name="nSolverRank">Specifies the rank of the new Solver.</param>
        /// <param name="shareNet">Optionally, specifies the net to share when creating the training network (default = null, meaning no share net is used).</param>
        /// <param name="getws">Optionally, specifies the handler for getting the workspace.</param>
        /// <param name="setws">Optionally, specifies the handler for setting the workspace.</param>
        /// <returns></returns>
        public static SGDSolver<T> Create(CudaDnn<T> cuda, Log log, SolverParameter solverParam, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabaseBase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0, Net<T> shareNet = null, onGetWorkspace getws = null, onSetWorkspace setws = null)
        {
            SGDSolver<T> solver = null;

            switch (solverParam.type)
            {
                case SolverParameter.SolverType.SGD:
                    solver = new SGDSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                case SolverParameter.SolverType.NESTEROV:
                    solver = new NesterovSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                case SolverParameter.SolverType.ADAGRAD:
                    solver = new AdaGradSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                case SolverParameter.SolverType.ADADELTA:
                    solver = new AdaDeltaSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                case SolverParameter.SolverType.ADAM:
                    solver = new AdamSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                case SolverParameter.SolverType.ADAMW:
                    solver = new AdamWSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                case SolverParameter.SolverType.RMSPROP:
                    solver = new RmsPropSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank, shareNet, getws, setws);
                    break;

                default:
                    throw new NotImplementedException("The solver " + solverParam.type.ToString() + " is not implemented yet!");
            }

            return solver;
        }
    }

#pragma warning disable 1591 

    public class OutputCollection /** @private */
    {
        OutputDataCollection m_rgError = new OutputDataCollection();
        OutputDataCollection m_rgAccuracy = new OutputDataCollection();

        public OutputCollection()
        {
        }

        public OutputDataCollection Errors
        {
            get { return m_rgError; }
        }

        public OutputDataCollection Accuracies
        {
            get { return m_rgAccuracy; }
        }
    }

    public class OutputDataCollection : IEnumerable<OutputData> /** @private */
    {
        List<OutputData> m_rgData = new List<OutputData>();

        public OutputDataCollection()
        {
        }

        public List<OutputData> Data
        {
            get { return m_rgData; }
        }

        public int Count
        {
            get { return m_rgData.Count; }
        }

        public OutputData this[int nIdx]
        {
            get { return m_rgData[nIdx]; }
            set { m_rgData[nIdx] = value; }
        }

        public void Add(int nTotal, string strName, int nIdx, double dfVal)
        {
            OutputData data = Find(strName);

            if (data == null)
            {
                data = new OutputData(strName, nIdx);
                m_rgData.Add(data);
            }

            data.Add(nTotal, dfVal);
        }

        public OutputData Find(string strName)
        {
            foreach (OutputData data in m_rgData)
            {
                if (data.Name == strName)
                    return data;
            }

            return null;
        }

        public IEnumerator<OutputData> GetEnumerator()
        {
            return m_rgData.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgData.GetEnumerator();
        }
    }

    public class OutputData /** @private */
    {
        string m_strName;
        double m_dfValue = 0;
        int m_nIdx;

        public OutputData(string strName, int nIdx)
        {
            m_strName = strName;
            m_nIdx = nIdx;
        }

        public int Index
        {
            get { return m_nIdx; }
        }

        public string Name
        {
            get { return m_strName; }
        }

        public double Value
        {
            get { return m_dfValue; }
            set { m_dfValue = value; }
        }

        public void Add(int nTotal, double dfVal)
        {
            double dfRatio = 1.0 / (double)nTotal;
            m_dfValue = (m_dfValue * (1.0 - dfRatio)) + (dfRatio * dfVal);
        }
    }

#pragma warning restore 1591
}
