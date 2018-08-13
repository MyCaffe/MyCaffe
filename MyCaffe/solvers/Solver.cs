using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO;
using System.Diagnostics;
using System.Collections;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using MyCaffe.common;
using MyCaffe.param;

/// <summary>
/// The MyCaffe.solvers namespace contains all solver classes, including the base Solver.
/// </summary>
namespace MyCaffe.solvers
{
    /// <summary>
    /// An interface for classes that perform optimization on Nets
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

        double m_dfSmoothedLoss = 0;
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
        IXImageDatabase m_db = null;
        int m_nTrainingIterationOverride = -1;
        int m_nTestingIterationOverride = -1;
        object m_tag = null;
        bool m_bWeightsUpdated = false;
        static object m_syncGetRi = new object();
        Blob<T> m_blobBatchInputData = null;
        double m_dfAverageTestTime = 0;
        SNAPSHOT_WEIGHT_UPDATE_METHOD m_snapshotWeightUpdatemMethod = SNAPSHOT_WEIGHT_UPDATE_METHOD.FAVOR_ACCURACY;
        int m_nTrainingTimeLimitInMinutes = 0;

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
        /// When specifies, the OnTest event fires during a TestAll and overrides the call to Test.
        /// </summary>
        public event EventHandler<TestArgs> OnTest;

        /// <summary>
        /// The Solver constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies teh SolverParameter.</param>
        /// <param name="evtCancel">Specifies a CancelEvent used to cancel the current operation (e.g. training, testing) for which the Solver is performing.</param>
        /// <param name="evtForceSnapshot">Specifies an automatic reset event that causes the Solver to perform a Snapshot when set.</param>
        /// <param name="evtForceTest">Specifies an automatic reset event that causes teh Solver to run a testing cycle when set.</param>
        /// <param name="imgDb">Specifies the CaffeImageDatabase.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPU session.</param>
        /// <param name="nSolverRank">Specifies the rank of this Solver in a multi-GPU session.</param>
        public Solver(CudaDnn<T> cuda, Log log, SolverParameter p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0)
        {
            m_cuda = cuda;
            m_log = log;
            m_evtCancel = evtCancel;
            m_evtForceSnapshot = evtForceSnapshot;
            m_evtForceTest = evtForceTest;
            m_log.Enable = is_root_solver;
            m_db = imgDb;
            m_persist = persist;
            m_nSolverCount = nSolverCount;
            m_nSolverRank = nSolverRank;

            Init(p);
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
                OnTrainingIteration(this, new TrainingIterationArgs<T>(m_nIter, m_dfLastAccuracy, dfLoss, m_dfSmoothedLoss, m_dfBestError, m_bWeightsUpdated, m_net.ActiveLabelCounts, dfLastLearningRate, dfTime, dbgInfo));
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
        /// Returns the CaffeImageDatabase used.
        /// </summary>
        public IXImageDatabase Database
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

        private Blob<T> getBatchInputData(BatchInformationCollection col)
        {
            if (col == null || col.Count == 0)
                return null;

            if (m_blobBatchInputData != null)
                return m_blobBatchInputData;

            Blob<T> blob = new Blob<T>(m_cuda, m_log);
            blob.Reshape(col.Count, col[0].Count, 1, 1);

            List<T> rgInput = new List<T>();

            for (int i = 0; i < col.Count; i++)
            {
                for (int j = 0; j < col[i].Count; j++)
                {
                    rgInput.Add((T)Convert.ChangeType(col[i][j].ImageIndex0, typeof(float)));
                }
            }

            blob.mutable_cpu_data = rgInput.ToArray();

            m_blobBatchInputData = blob;

            return blob;
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
        public void Init(SolverParameter p)
        {
            m_log.WriteLine("Initializing solver from parameters: " + p.DebugString());
            m_param = p;
            m_log.CHECK_GE(m_param.average_loss, 1, "Average loss should be non-negative and >= 1.0.");

            if (m_param.random_seed >= 0)
                m_cuda.rng_setseed(m_param.random_seed + m_nSolverRank);

            // Scaffolding code.
            InitTrainNet();
            InitTestNets();

            if (is_root_solver)
                m_log.WriteLine("Solver scaffolding done.");

            m_nIter = 0;
            m_nCurrentStep = 0;
        }

        /// <summary>
        /// Initializes the Net used by the solver for training.
        /// </summary>
        protected void InitTrainNet()
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
                m_net = new Net<T>(m_cuda, m_log, net_param, m_evtCancel, m_db, Phase.NONE, m_evtCompleted);
                m_net.OnGetIteration += net_OnGetIteration;
            }
            catch(Exception excpt)
            {
                throw new Exception("Initializing Training Net: " + excpt.Message);
            }
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
                    m_rgTestNets.Add(new Net<T>(m_cuda, m_log, net_params[i], m_evtCancel, m_db, Phase.NONE, null, TrainingNet));

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
        /// <param name="col">Optionally, specifies batch information used when using reinforcement learning.  The default is <i>null</i> which ignores the parameter.</param>
        public virtual void Solve(int nIterationOverride = -1, byte[] rgWeights = null, byte[] rgState = null, TRAIN_STEP step = TRAIN_STEP.NONE, BatchInformationCollection col = null)
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

            if (!Step(nIterationOverride, step, col))
                return;

            // If we haven't already, save a snapshot after optimization, unless
            // overriden by setting snapshot_after_train = false.
            if (step == TRAIN_STEP.NONE && (m_param.snapshot_after_train && (m_param.snapshot == 0 || (m_nIter % m_param.snapshot) != 0)))
                Snapshot(false, true);

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
                int average_loss = m_param.average_loss;
                double dfLoss;
                m_net.Forward(out dfLoss);

                UpdateSmoothedLoss(dfLoss, start_iter, average_loss);
                m_log.WriteLine("Iteration " + m_nIter + ", loss = " + m_dfSmoothedLoss.ToString(), true);
            }

            if (m_param.test_interval > 0 && (m_nIter % m_param.test_interval) == 0)
            {
                if (m_bEnableTest)
                    TestAll();
            }

            m_log.WriteLine("Optimization done.", true);

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
        /// <param name="col">Optionally, specifies a collection of BatchInformation used when performing custom training. The default = <i>null</i>.</param>
        /// <returns></returns>
        public bool Step(int nIters, TRAIN_STEP step = TRAIN_STEP.NONE, BatchInformationCollection col = null)
        {
            Exception err = null;

            try
            {
                Blob<T> blobBatchInput = getBatchInputData(col);
                BlobCollection<T> colBottom = new BlobCollection<T>();
                int start_iter = m_nIter;
                int stop_iter = m_nIter + nIters;
                int average_loss = m_param.average_loss;

                if (m_net.layers[0].type == LayerParameter.LayerType.BATCHDATA)
                {
                    m_log.CHECK(blobBatchInput != null, "There is no batch input data!");
                    colBottom.Add(blobBatchInput);
                }
                else
                {
                    m_log.CHECK(blobBatchInput == null, "The blob batch input data is only supported with networks using the BATCHDATA layer.");
                }

                m_net.SetReinforcementInformation(col);

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
                    bool bDisplay = (is_root_solver && m_param.display > 0 && (m_nIter % m_param.display) == 0) ? true : false;
                    m_net.set_debug_info(bDisplay && m_param.debug_info);

                    // accumulate the loss and gradient
                    double dfLoss = 0;
                    double dfLossTotal = 0;
                    int nIterCount = 0;

                    Stopwatch swTiming = new Stopwatch();
                    double dfTotalTime = 0;
                    int nTimingCount = 0;
                    bool bFwdPassNanFree = true;

                    for (int i = 0; i < m_param.iter_size; i++)
                    {
                        double dfLocalLoss;

                        swTiming.Restart();

                        bFwdPassNanFree = m_net.ForwardBackward(colBottom, out dfLocalLoss, step);

                        dfLossTotal += dfLocalLoss;
                        swTiming.Stop();

                        dfTotalTime += swTiming.Elapsed.TotalMilliseconds;
                        nTimingCount++;
                        nIterCount++;

                        if (!bFwdPassNanFree)
                            break;
                    }

                    dfLoss = dfLossTotal / nIterCount;

                    // average the loss across iterations for smoothed reporting
                    UpdateSmoothedLoss(dfLoss, start_iter, average_loss);

                    if (!bDisplay && sw.ElapsedMilliseconds > 2000)
                    {
                        bDisplay = true;
                        sw.Restart();
                    }

                    if (bDisplay)
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

                    if (step != TRAIN_STEP.FORWARD)
                        dfLastLearningRate = ApplyUpdate();

                    if (m_evtCancel.WaitOne(0))
                        break;

                    // Increment the internal iter_ counter -- its value should always indicate
                    // the number of times the weights have been updated.
                    m_nIter++;
                    m_log.Progress = (double)m_nIter / (double)stop_iter;

                    bool bSnapshotTaken = false;
                    bool bForceSnapshot = forceSnapshot;

                    if (step == TRAIN_STEP.NONE && (is_root_solver && bFwdPassNanFree &&
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
                            m_log.WriteLine("Single step (both) triggered - solving stopped after a single forward/backward pass.");
                        else if (step == TRAIN_STEP.FORWARD)
                            m_log.WriteLine("Single step (forward) triggered - solving stopped after a single forward pass.");
                        else if (step == TRAIN_STEP.BACKWARD)
                            m_log.WriteLine("Single step (backward) triggered - solving stopped after a single backward pass.");
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
        public void Snapshot(bool bForced, bool bScheduled)
        {
            m_log.WriteLine("Starting snap shot...");
            m_log.CHECK(is_root_solver, "Snapshot only supported on the root solver.");

            if (OnSnapshot == null)
                return;

            SnapshotArgs args = new common.SnapshotArgs(null, null, m_dfLastAccuracy, m_dfLastError, m_nIter, m_snapshotWeightUpdatemMethod);
            args.IncludeState = m_param.snapshot_include_state;
            args.IncludeWeights = m_param.snapshot_include_weights;
            args.SingleStep = m_bEnableSingleStep;
            args.Forced = bForced;
            args.Scheduled = bScheduled;
            args.OnGetState += args_OnGetState;
            args.OnGetWeights += args_OnGetWeights;

            OnSnapshot(this, args);
            m_log.WriteLine("Snapshot completed.");
        }

        private void args_OnGetWeights(object sender, GetBytesArgs e)
        {
            e.Data = m_net.SaveWeights(m_persist, m_param.snapshot_diff);
        }

        private void args_OnGetState(object sender, GetBytesArgs e)
        {
            e.Data = SnapshotSolverState();
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
                    dfTotalAccuracy += Test(nIterationOverride, test_net_id);

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
                    dfTotalAccuracy += Test(nIterationOverride, 0);
            }

            double dfAccuracy = (m_rgTestNets.Count > 0) ? dfTotalAccuracy / m_rgTestNets.Count : 0;

            if (OnTestingIteration != null)
            {
                double dfTime = (nTotalCount > 0) ? dfTotalTime / nTotalCount : 0;
                OnTestingIteration(this, new TestingIterationArgs<T>(m_nIter, dfAccuracy, dfTime));
            }

            return dfAccuracy;
        }

        /// <summary>
        /// Run a test on a given test Net by running it through its iterations.
        /// </summary>
        /// <param name="nIterationOverride">Specifies an override the the number of iterations to run.</param>
        /// <param name="nTestNetId">Specifies the ID of the test Net to run.</param>
        /// <returns>The accuracy of the test run is returned as a percentage in the range [0, 1].</returns>
        public double Test(int nIterationOverride = -1, int nTestNetId = 0)
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
            Stopwatch swTiming = new Stopwatch();

            for (int i = 0; i < nIter; i++)
            {
                // Check to see if stoppage of testing/training has been requested.
                if (m_evtCancel.WaitOne(0))
                    break;

                swTiming.Restart();

                double iter_loss;
                BlobCollection<T> colResult = test_net.Forward(out iter_loss);

                if (m_param.test_compute_loss)
                    dfLoss += iter_loss;

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

                        if (colResult[j].type == Blob<T>.BLOB_TYPE.ACCURACY)
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

                swTiming.Stop();
                dfTotalTiming += swTiming.Elapsed.TotalMilliseconds;
                nTestCount++;

                if (sw.ElapsedMilliseconds > 2000)
                {
                    double dfPct = (double)i / (double)nIter;

                    if (is_root_solver)
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

            for (int i = 0; i < test_score.Count; i++)
            {
                int nIdxTestScore = test_score_output_id[i];
                int output_blob_index = test_net.output_blob_indices[nIdxTestScore];
                string output_name = test_net.blob_names[output_blob_index];
                double loss_weight = test_net.blob_loss_weights[output_blob_index];
                double dfMeanScore = test_score[i] / nIter;
                string strOut = "";

                if (loss_weight != 0)
                    strOut += " (* " + loss_weight.ToString() + " = " + (loss_weight * dfMeanScore).ToString() + " loss)";

                m_log.WriteLine("   Test net output #" + i.ToString() + ": " + output_name + " = " + dfMeanScore.ToString() + strOut);

                if (i == nAccuracyIdx)
                    dfFinalScore = dfMeanScore;
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
        /// <param name="nAverageLoss">Specifies the number of iterations to average over.</param>
        protected void UpdateSmoothedLoss(double dfLoss, int nStartIter, int nAverageLoss)
        {
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
        /// <param name="imgDb">Specifies the CaffeImageDatabase that the new Solver will use.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPu session.</param>
        /// <param name="nSolverRank">Specifies the rank of the new Solver.</param>
        /// <returns>A new Solver instance is returned.</returns>
        public static SGDSolver<T> Create(CudaDnn<T> cuda, Log log, ProjectEx p, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0)
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

            return Create(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
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
        /// <param name="imgDb">Specifies the CaffeImageDatabase that the new Solver will use.</param>
        /// <param name="persist">Specifies the peristence used for loading and saving weights.</param>
        /// <param name="nSolverCount">Specifies the number of Solvers participating in a multi-GPu session.</param>
        /// <param name="nSolverRank">Specifies the rank of the new Solver.</param>
        /// <returns></returns>
        public static SGDSolver<T> Create(CudaDnn<T> cuda, Log log, SolverParameter solverParam, CancelEvent evtCancel, AutoResetEvent evtForceSnapshot, AutoResetEvent evtForceTest, IXImageDatabase imgDb, IXPersist<T> persist, int nSolverCount = 1, int nSolverRank = 0)
        {
            SGDSolver<T> solver = null;

            switch (solverParam.type)
            {
                case SolverParameter.SolverType.SGD:
                    solver = new SGDSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
                    break;

                case SolverParameter.SolverType.NESTEROV:
                    solver = new NesterovSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
                    break;

                case SolverParameter.SolverType.ADAGRAD:
                    solver = new AdaGradSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
                    break;

                case SolverParameter.SolverType.ADADELTA:
                    solver = new AdaDeltaSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
                    break;

                case SolverParameter.SolverType.ADAM:
                    solver = new AdamSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
                    break;

                case SolverParameter.SolverType.RMSPROP:
                    solver = new RmsPropSolver<T>(cuda, log, solverParam, evtCancel, evtForceSnapshot, evtForceTest, imgDb, persist, nSolverCount, nSolverRank);
                    break;

                default:
                    throw new NotImplementedException("The solver " + solverParam.type.ToString() + " is not implemented yet!");
            }

            return solver;
        }
    }

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
}
