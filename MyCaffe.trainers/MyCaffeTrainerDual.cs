using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.gym;
using MyCaffe.param;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The MyCaffeTraininerDual is used to perform both reinforcement and recurrent learning training tasks on an instance of the MyCaffeControl.
    /// </summary>
    /// <remarks>
    /// --RL--
    /// Currently, the MyCaffeTrainerDual supports the following reinforcement trainers, each of which are selected with the 'TrainerType=type' property
    /// value within the property set specified when calling the Initialize method.
    /// 
    /// TrainerType=PG.SIMPLE - creates the initial simple policy gradient trainer that only supports single-threaded Sigmoid based models.
    /// TrainerType=PG.ST - creates a single-threaded policy gradient trainer that supports both Sigmoid and Softmax based models.
    /// TrainerType=PG.MT - creates a multi-threaded policy gradient trainer that supports both Sigmoid and Softmax based models and can train across GPU's.
    /// TrainerType=C51.ST = creates a single-threaded network that implements the C51 DDQN algorithm.
    /// 
    /// Other intitialization properties include:
    /// 
    /// RewardType=VAL - reports the actual reward values.
    /// RewardType=MAX - reports the maximum reward value observed (recommended setting)
    /// 
    /// Threads=# - specifies the number of threads.
    /// 
    /// GPUs=#,#,... - specifies the GPU's on which to run each thread.  The GPU IDs may be the same as the open project or other GPU's in the system.  GPU
    /// selection starts with the first GPUID in the list, continues to the end, and then wraps back around to the start of the list.  For example if you
    /// specifiy to use 3 thread with GPUIDs=0,1 the GPUs will be assigned to each thread as follows: Thread0 => GPUID0, Thread1 => GPUID1, Thread2 => GPUID0
    /// 
    /// Gamma - specifies the discount rate (default = 0.99)
    /// UseRawInput - when <i>true</i> the actual input is used directly, otherwise a difference between the current and previous input is used (default = <i>false</i>).
    /// 
    /// 
    /// --RNN--
    /// Currently, the MyCaffeTrainerDual also supports the following recurrent trainers, each of which are selected with the 'TrainerType=type' property
    /// value within the property set specified when calling the Initialize method.
    /// 
    /// TrainerType=RNN.SIMPLE - creates the initial simple policy gradient trainer that only supports single-threaded Sigmoid based models.
    /// 
    /// 
    /// The following settings are used from the Model and Solver descriptions:
    /// 
    /// Solver: base_lr - specifies the learning rate used.
    /// Model: batch_size - specifies how often accumulated gradients are applied.
    /// </remarks>
    public partial class MyCaffeTrainerDual : Component, IXMyCaffeCustomTrainerRL, IXMyCaffeCustomTrainerRNN, IxTrainerCallbackRNN
    {
        /// <summary>
        /// Random number generator used to get initial actions, etc.
        /// </summary>
        protected CryptoRandom m_random = new CryptoRandom(true);
        /// <summary>
        /// Specifies the properties parsed from the key-value pair passed to the Initialize method.
        /// </summary>
        protected PropertySet m_properties = null;
        /// <summary>
        /// Specifies the project ID of the project held by the instance of MyCaffe.
        /// </summary>
        protected int m_nProjectID = 0;
        IxTrainer m_itrainer = null;
        double m_dfExplorationRate = 0;
        double m_dfOptimalSelectionRate = 0;
        double m_dfImmediateRewards = 0;
        double m_dfGlobalRewards = 0;
        double m_dfGlobalRewardsAve = 0;
        double m_dfGlobalRewardsMax = -double.MaxValue;
        int m_nGlobalEpisodeCount = 0;
        int m_nGlobalEpisodeMax = 0;
        double m_dfLoss = 0;
        int m_nThreads = 1;
        REWARD_TYPE m_rewardType = REWARD_TYPE.MAXIMUM;
        TRAINER_TYPE m_trainerType = TRAINER_TYPE.PG_ST;
        double m_dfAccuracy = 0;
        int m_nIteration = 0;
        int m_nIterations = -1;
        IXMyCaffeCustomTrainerCallback m_icallback = null;
        int m_nSnapshot = 0;
        bool m_bSnapshot = false;
        BucketCollection m_rgVocabulary = null;
        Stage m_stage = Stage.RL;
        bool m_bUsePreloadData = false;

        enum TRAINER_TYPE
        {
            PG_MT,
            PG_ST,
            PG_SIMPLE,
            C51_ST,
            C51b_ST,
            RNN_SIMPLE
        }

        enum REWARD_TYPE
        {
            VALUE,
            AVERAGE,
            MAXIMUM
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeTrainerDual()
        {
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">The container of the component.</param>
        public MyCaffeTrainerDual(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        #region Overrides

        /// <summary>
        /// Overriden to give the actual name of the custom trainer.
        /// </summary>
        protected virtual string name
        {
            get { return "MyCaffe RL/RNN Dual Trainer"; }
        }

        /// <summary>
        /// Override when using a training method other than the REINFORCEMENT method (the default).
        /// </summary>
        protected virtual TRAINING_CATEGORY category
        {
            get { return TRAINING_CATEGORY.DUAL; }
        }

        /// <summary>
        /// Returns a dataset override to use (if any) instead of the project's dataset.  If there is no dataset override
        /// <i>null</i> is returned and the project's dataset is used.
        /// </summary>
        /// <param name="nProjectID">Specifies the project ID associated with the trainer (if any)</param>
        protected virtual DatasetDescriptor get_dataset_override(int nProjectID)
        {
            return null;
        }

        /// <summary>
        /// Returns information describing the specific trainer, such as the gym used, if any.
        /// </summary>
        /// <returns>The string describing the trainer is returned.</returns>
        protected virtual string get_information()
        {
            return "";
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="caffe">Specifies the MyCaffeControl used.</param>
        /// <param name="stage">Specifies the stage under which the trainer is created.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerD(Component caffe, Stage stage)
        {
            MyCaffeControl<double> mycaffe = caffe as MyCaffeControl<double>;
            m_nProjectID = mycaffe.CurrentProject.ID;
            int.TryParse(mycaffe.CurrentProject.GetSolverSetting("max_iter"), out m_nIterations);
            int.TryParse(mycaffe.CurrentProject.GetSolverSetting("snapshot"), out m_nSnapshot);

            if (stage == Stage.RNN)
            {
                switch (m_trainerType)
                {
                    case TRAINER_TYPE.RNN_SIMPLE:
                        return new rnn.simple.TrainerRNN<double>(mycaffe, m_properties, m_random, this, m_rgVocabulary, m_bUsePreloadData);

                    default:
                        throw new Exception("The trainer type '" + m_trainerType.ToString() + "' is not supported in the RNN stage!");
                }
            }
            else
            {
                switch (m_trainerType)
                {
                    case TRAINER_TYPE.PG_SIMPLE:
                        return new pg.simple.TrainerPG<double>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.PG_ST:
                        return new pg.st.TrainerPG<double>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.PG_MT:
                        return new pg.mt.TrainerPG<double>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.C51_ST:
                        return new c51.ddqn.TrainerC51<double>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.C51b_ST:
                        return new c51b.ddqn.TrainerC51<double>(mycaffe, m_properties, m_random, this);

                    default:
                        throw new Exception("The trainer type '" + m_trainerType.ToString() + "' is not supported in the RL stage!");
                }
            }
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="caffe">Specifies the MyCaffeControl used.</param>
        /// <param name="stage">Specifies the stage under which the trainer is created.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerF(Component caffe, Stage stage)
        {
            MyCaffeControl<float> mycaffe = caffe as MyCaffeControl<float>;
            m_nProjectID = mycaffe.CurrentProject.ID;
            int.TryParse(mycaffe.CurrentProject.GetSolverSetting("max_iter"), out m_nIterations);
            int.TryParse(mycaffe.CurrentProject.GetSolverSetting("snapshot"), out m_nSnapshot);

            if (stage == Stage.RNN)
            {
                switch (m_trainerType)
                {
                    case TRAINER_TYPE.RNN_SIMPLE:
                        return new rnn.simple.TrainerRNN<float>(mycaffe, m_properties, m_random, this, m_rgVocabulary, m_bUsePreloadData);

                    default:
                        throw new Exception("The trainer type '" + m_trainerType.ToString() + "' is not supported in the RNN stage!");
                }
            }
            else
            {
                switch (m_trainerType)
                {
                    case TRAINER_TYPE.PG_SIMPLE:
                        return new pg.simple.TrainerPG<float>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.PG_ST:
                        return new pg.st.TrainerPG<float>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.PG_MT:
                        return new pg.mt.TrainerPG<float>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.C51_ST:
                        return new c51.ddqn.TrainerC51<float>(mycaffe, m_properties, m_random, this);

                    case TRAINER_TYPE.C51b_ST:
                        return new c51b.ddqn.TrainerC51<float>(mycaffe, m_properties, m_random, this);

                    default:
                        throw new Exception("The trainer type '" + m_trainerType.ToString() + "' is not supported in the RL stage!");
                }
            }
        }

        /// <summary>
        /// Override to dispose of resources used.
        /// </summary>
        protected virtual void dispose()
        {
        }

        /// <summary>
        /// Override called by the Initialize method of the trainer.
        /// </summary>
        /// <remarks>
        /// When providing a new trainer, this method is not used.
        /// </remarks>
        /// <param name="e">Specifies the initialization arguments.</param>
        protected virtual void initialize(InitializeArgs e)
        {
        }

        /// <summary>
        /// Override called from within the CleanUp method.
        /// </summary>
        protected virtual void shutdown()
        {
        }

        /// <summary>
        /// Override called by the OnGetData event fired by the Trainer to retrieve a new set of observation collections making up a set of experiences.
        /// </summary>
        /// <param name="e">Specifies the getData argments used to return the new observations.</param>
        /// <returns>A value of <i>true</i> is returned when data is retrieved.</returns>
        protected virtual bool getData(GetDataArgs e)
        {
            return false;
        }

        /// <summary>
        /// Override called by the OnConvertOutput event fired by the Trainer to convert the network output into its native format.
        /// </summary>
        /// <param name="e">Specifies the event arguments.</param>
        /// <returns>When handled this function retunrs <i>true</i>, otherwise it returns <i>false</i>.</returns>
        protected virtual bool convertOutput(ConvertOutputArgs e)
        {
            return false;
        }

        /// <summary>
        /// Returns <i>true</i> when the training is ready for a snap-shot, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="dfAccuracy">Specifies the current rewards.</param>
        protected virtual bool get_update_snapshot(out int nIteration, out double dfAccuracy)
        {
            nIteration = GlobalEpisodeCount;
            dfAccuracy = GlobalRewards;

            if (m_bSnapshot)
            {
                m_bSnapshot = false;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Called by OpenUi, override this when a UI (via WCF) should be displayed.
        /// </summary>
        protected virtual void openUi()
        {
        }

        /// <summary>
        /// The preloaddata method gives the custom trainer an opportunity to pre-load any data.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nProjectID">Specifies the project ID if any.</param>
        /// <param name="bUsePreloadData">Returns whether or not to use pre-load data, or dynamic data.</param>
        /// <returns>When data is pre-loaded the discovered vocabulary is returned as a bucket collection.</returns>
        protected virtual BucketCollection preloaddata(Log log, CancelEvent evtCancel, int nProjectID, out bool bUsePreloadData)
        {
            bUsePreloadData = false;
            return null;
        }

        #endregion

        #region IXMyCaffeCustomTrainer Interface

        /// <summary>
        /// Returns the stage under which the trainer is running based on the trainer type.
        /// </summary>
        public Stage Stage
        {
            get { return m_stage; }
        }

        /// <summary>
        /// Returns the name of the custom trainer.  This method calls the 'name' override.
        /// </summary>
        public string Name
        {
            get { return name; }
        }

        /// <summary>
        /// Returns the training category of the custom trainer (default = REINFORCEMENT).
        /// </summary>
        public TRAINING_CATEGORY TrainingCategory
        {
            get { return category; }
        }

        /// <summary>
        /// Returns <i>true</i> when the training is ready for a snap-shot, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="dfAccuracy">Specifies the current rewards.</param>
        public bool GetUpdateSnapshot(out int nIteration, out double dfAccuracy)
        {
            return get_update_snapshot(out nIteration, out dfAccuracy);
        }

        /// <summary>
        /// Returns a dataset override to use (if any) instead of the project's dataset.  If there is no dataset override
        /// <i>null</i> is returned and the project's dataset is used.
        /// </summary>
        /// <param name="nProjectID">Specifies the project ID associated with the trainer (if any)</param>
        public DatasetDescriptor GetDatasetOverride(int nProjectID)
        {
            return get_dataset_override(nProjectID);
        }

        /// <summary>
        /// Returns whether or not Training is supported.
        /// </summary>
        public bool IsTrainingSupported
        {
            get { return true; }
        }

        /// <summary>
        /// Returns whether or not Testing is supported.
        /// </summary>
        public bool IsTestingSupported
        {
            get { return true; }
        }

        /// <summary>
        /// Returns whether or not Running is supported.
        /// </summary>
        public bool IsRunningSupported
        {
            get { return true; }
        }

        /// <summary>
        /// Releases any resources used by the component.
        /// </summary>
        public void CleanUp()
        {
            if (m_itrainer != null)
            {
                m_itrainer.Shutdown(3000);
                m_itrainer = null;
            }

            shutdown();
        }

        /// <summary>
        /// Initializes a new custom trainer by loading the key-value pair of properties into the property set.
        /// </summary>
        /// <param name="strProperties">Specifies the key-value pair of properties each separated by ';'.  For example the expected
        /// format is 'key1'='value1';'key2'='value2';...</param>
        /// <param name="icallback">Specifies the parent callback.</param>
        public void Initialize(string strProperties, IXMyCaffeCustomTrainerCallback icallback)
        {
            m_icallback = icallback;
            m_properties = new PropertySet(strProperties);
            m_nThreads = m_properties.GetPropertyAsInt("Threads", 1);

            string strRewardType = m_properties.GetProperty("RewardType", false);
            if (strRewardType == null)
                strRewardType = "VAL";
            else
                strRewardType = strRewardType.ToUpper();

            if (strRewardType == "VAL" || strRewardType == "VALUE")
                m_rewardType = REWARD_TYPE.VALUE;
            else if (strRewardType == "AVE" || strRewardType == "AVERAGE")
                m_rewardType = REWARD_TYPE.AVERAGE;

            string strTrainerType = m_properties.GetProperty("TrainerType");

            switch (strTrainerType)
            {
                case "PG.SIMPLE":   // bare bones model (Sigmoid only)
                    m_trainerType = TRAINER_TYPE.PG_SIMPLE;
                    m_stage = Stage.RL;
                    break;

                case "PG.ST":       // single thread (Sigmoid and Softmax)
                    m_trainerType = TRAINER_TYPE.PG_ST;
                    m_stage = Stage.RL;
                    break;

                case "PG":
                case "PG.MT":       // multi-thread (Sigmoid and Softmax)
                    m_trainerType = TRAINER_TYPE.PG_MT;
                    m_stage = Stage.RL;
                    break;

                case "C51.ST":      // single threaded C51
                    m_trainerType = TRAINER_TYPE.C51_ST;
                    m_stage = Stage.RL;
                    break;

                case "C51b.ST":      // single threaded C51b
                    m_trainerType = TRAINER_TYPE.C51b_ST;
                    m_stage = Stage.RL;
                    break;

                case "RNN.SIMPLE":
                    m_trainerType = TRAINER_TYPE.RNN_SIMPLE;
                    m_stage = Stage.RNN;
                    break;

                default:
                    throw new Exception("Unknown trainer type '" + strTrainerType + "'!");
            }
        }

        private Stage getStage()
        {
            if (m_trainerType == TRAINER_TYPE.RNN_SIMPLE)
                return Stage.RNN;
            else
                return Stage.RL;
        }

        private IxTrainer createTrainer(Component mycaffe, Stage stage)
        {
            IxTrainer itrainer = null;

            if (mycaffe is MyCaffeControl<double>)
                itrainer = create_trainerD(mycaffe, stage);
            else
                itrainer = create_trainerF(mycaffe, stage);

            itrainer.Initialize();

            return itrainer;
        }

        /// <summary>
        /// Create a new trainer and use it to run a test cycle using the current 'stage' = RNN or RL.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        /// <param name="type">Specifies the type of iterator to use.</param>
        public void Test(Component mycaffe, int nIterationOverride, ITERATOR_TYPE type = ITERATOR_TYPE.ITERATION)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe, getStage());

            if (nIterationOverride == -1)
                nIterationOverride = m_nIterations;

            m_itrainer.Test(nIterationOverride, type);
            m_itrainer.Shutdown(500);
            m_itrainer = null;
        }

        /// <summary>
        /// Create a new trainer and use it to run a training cycle using the current 'stage' = RNN or RL.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        /// <param name="type">Specifies the type of iterator to use.</param>
        /// <param name="step">Optionally, specifies whether or not to step the training for debugging (default = NONE).</param>
        public void Train(Component mycaffe, int nIterationOverride, ITERATOR_TYPE type = ITERATOR_TYPE.ITERATION, TRAIN_STEP step = TRAIN_STEP.NONE)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe, getStage());

            if (nIterationOverride == -1)
                nIterationOverride = m_nIterations;

            m_itrainer.Train(nIterationOverride, type, step);
            m_itrainer.Shutdown(1000);
            m_itrainer = null;
        }

        #endregion

        /// <summary>
        /// The OnIntialize callback fires when initializing the trainer.
        /// </summary>
        public void OnInitialize(InitializeArgs e)
        {
            initialize(e);
        }

        /// <summary>
        /// The OnShutdown callback fires when shutting down the trainer.
        /// </summary>
        public void OnShutdown()
        {
            shutdown();
        }

        /// <summary>
        /// The OnGetData callback fires from within the Train method and is used to get a new observation data.
        /// </summary>
        public void OnGetData(GetDataArgs e)
        {
            getData(e);
        }

        /// <summary>
        /// The OnConvertOutput callback fires from within the Run method and is used to convert the network output into its native format.
        /// </summary>
        /// <param name="e">Specifies the event arguments.</param>
        public void OnConvertOutput(ConvertOutputArgs e)
        {
            convertOutput(e);
        }

        /// <summary>
        /// The OnGetStatus callback fires on each iteration within the Train method.
        /// </summary>
        public void OnUpdateStatus(GetStatusArgs e)
        {
            m_nIteration = e.Iteration;
            m_dfAccuracy = e.TotalReward;
            m_nIterations = e.MaxFrames;
            m_dfImmediateRewards = e.Reward;
            m_dfGlobalRewards = e.TotalReward;
            m_dfGlobalRewardsMax = Math.Max(m_dfGlobalRewardsMax, e.TotalReward);
            m_dfGlobalRewardsAve = (1.0 / (double)m_nThreads) * e.TotalReward + ((m_nThreads - 1) / (double)m_nThreads) * m_dfGlobalRewardsAve;
            m_dfExplorationRate = e.ExplorationRate;
            m_dfOptimalSelectionRate = e.OptimalSelectionCoefficient;

            if (m_nThreads > 1)
                m_nGlobalEpisodeCount++;
            else
                m_nGlobalEpisodeCount = e.Frames;

            m_nGlobalEpisodeMax = e.MaxFrames;
            m_dfLoss = e.Loss;

            if (m_icallback != null)
            {
                Dictionary<string, double> rgValues = new Dictionary<string, double>();
                rgValues.Add("GlobalIteration", GlobalEpisodeCount);
                rgValues.Add("GlobalLoss", GlobalLoss);
                rgValues.Add("LearningRate", e.LearningRate);
                rgValues.Add("GlobalAccuracy", GlobalRewards);
                rgValues.Add("Threads", m_nThreads);

                m_icallback.Update(TrainingCategory, rgValues);
            }

            e.NewFrameCount = m_nGlobalEpisodeCount;

            if (e.Index == 0 && m_nSnapshot > 0 && m_nGlobalEpisodeCount > 0 && (m_nGlobalEpisodeCount % m_nSnapshot) == 0)
                m_bSnapshot = true;
        }

        /// <summary>
        /// The OnWait callback fires when waiting for a shutdown.
        /// </summary>
        public void OnWait(WaitArgs e)
        {
            Thread.Sleep(e.Wait);
        }

        public double GetProperty(string strProp)
        {
            switch (strProp)
            {
                case "GlobalLoss":
                    return GlobalLoss;

                case "GlobalAccuracy": // RNN
                    return m_dfAccuracy;

                case "GlobalIteration": // RNN
                    return m_nIteration;

                case "GlobalMaxIterations":
                    return m_nIterations;

                case "GlobalRewards":
                    return GlobalRewards;

                case "GlobalEpisodeCount":
                    return GlobalEpisodeCount;

                case "ExplorationRate":
                    return ExplorationRate;

                default:
                    throw new Exception("The property '" + strProp + "' is not supported by the MyCaffeTrainerRNN.");
            }
        }

        /// <summary>
        /// Returns the global rewards based on the reward type specified by the 'RewardType' property.
        /// </summary>
        /// <remarks>
        /// The reward type can be one of the following:
        ///    'VAL' - report the global reward value.
        ///    'AVE' - report the global reward averaged over all threads.
        ///    'MAX' - report maximum global rewards (default)
        /// </remarks>
        public double GlobalRewards
        {
            get
            {
                switch (m_rewardType)
                {
                    case REWARD_TYPE.VALUE:
                        return m_dfGlobalRewards;

                    case REWARD_TYPE.AVERAGE:
                        return m_dfGlobalRewardsAve;

                    default:
                        return (m_dfGlobalRewardsMax == -double.MaxValue) ? 0 : m_dfGlobalRewardsMax;
                }
            }
        }

        public double ImmediateRewards
        {
            get { return m_dfImmediateRewards; }
        }

        /// <summary>
        /// Return the global loss.
        /// </summary>
        public double GlobalLoss
        {
            get { return m_dfLoss; }
        }

        /// <summary>
        /// Returns the global iteration.
        /// </summary>
        public int GlobalIteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Returns the global episode count.
        /// </summary>
        public int GlobalEpisodeCount
        {
            get { return m_nGlobalEpisodeCount; }
        }

        /// <summary>
        /// Returns the maximum global episode count.
        /// </summary>
        public int GlobalEpisodeMax
        {
            get { return m_nGlobalEpisodeMax; }
        }

        /// <summary>
        /// Returns the current exploration rate.
        /// </summary>
        public double ExplorationRate
        {
            get { return m_dfExplorationRate; }
        }

        /// <summary>
        /// Returns the rate of selection from the optimal set with the highest reward (this setting is optional, default = 0).
        /// </summary>
        public double OptimalSelectionRate
        {
            get { return m_dfOptimalSelectionRate; }
        }

        /// <summary>
        /// Returns information describing the trainer.
        /// </summary>
        public string Information
        {
            get { return get_information(); }
        }

        /// <summary>
        /// Open the user interface for the trainer, of one exists.
        /// </summary>
        public void OpenUi()
        {
            openUi();
        }

        #region IXMyCaffeCustomTrainerRL Methods

        /// <summary>
        /// Create a new trainer and use it to run a single run cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nDelay">Specifies a delay to wait before getting the action.</param>
        /// <returns>The results of the run are returned.</returns>
        ResultCollection IXMyCaffeCustomTrainerRL.RunOne(Component mycaffe, int nDelay)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe, Stage.RL);

            IxTrainerRL itrainer = m_itrainer as IxTrainerRL;
            if (itrainer == null)
                throw new Exception("The trainer must be set to to 'C51.ST', PG.SIMPLE', 'PG.ST' or 'PG.MT' to run in reinforcement learning mode.");

            ResultCollection res = itrainer.RunOne(nDelay);
            m_itrainer.Shutdown(50);
            m_itrainer = null;

            return res;
        }

        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="type">Specifies the output data type returned as a raw byte stream.</param>
        /// <returns>The run results are returned in the same native type as that of the CustomQuery used.</returns>
        public byte[] Run(Component mycaffe, int nN, out string type)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe, Stage.RL);

            string strRunProperties = null;
            IXMyCaffeCustomTrainerCallbackRNN icallback = m_icallback as IXMyCaffeCustomTrainerCallbackRNN;
            if (icallback != null)
                strRunProperties = icallback.GetRunProperties();

            IxTrainerRL itrainer = m_itrainer as IxTrainerRL;
            if (itrainer == null)
                throw new Exception("The IxTrainerRL interface must be implemented.");

            byte[] rgResults = itrainer.Run(nN, strRunProperties, out type);
            m_itrainer.Shutdown(0);
            m_itrainer = null;

            return rgResults;
        }

        #endregion

        #region IXMyCaffeCustomTrainerRNN

        /// <summary>
        /// Create a new trainer and use it to run a single run cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <returns>The results of the run are returned.</returns>
        float[] IXMyCaffeCustomTrainerRNN.Run(Component mycaffe, int nN)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe, Stage.RNN);

            IxTrainerRNN itrainer = m_itrainer as IxTrainerRNN;
            if (itrainer == null)
                throw new Exception("The trainer must be set to to 'RNN.SIMPLE' to run in recurrent learning mode.");

            string strRunProperties = null;
            IXMyCaffeCustomTrainerCallbackRNN icallback = m_icallback as IXMyCaffeCustomTrainerCallbackRNN;
            if (icallback != null)
                strRunProperties = icallback.GetRunProperties();

            float[] rgResults = itrainer.Run(nN, strRunProperties);
            m_itrainer.Shutdown(0);
            m_itrainer = null;

            return rgResults;
        }

        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="type">Specifies the output data type returned as a raw byte stream.</param>
        /// <returns>The run results are returned in the same native type as that of the CustomQuery used.</returns>
        byte[] IXMyCaffeCustomTrainerRNN.Run(Component mycaffe, int nN, out string type)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe, Stage.RNN);

            IxTrainerRNN itrainer = m_itrainer as IxTrainerRNN;
            if (itrainer == null)
                throw new Exception("The trainer must be set to to 'RNN.SIMPLE' to run in recurrent learning mode.");

            string strRunProperties = null;
            IXMyCaffeCustomTrainerCallbackRNN icallback = m_icallback as IXMyCaffeCustomTrainerCallbackRNN;
            if (icallback != null)
                strRunProperties = icallback.GetRunProperties();

            byte[] rgResults = itrainer.Run(nN, strRunProperties, out type);
            m_itrainer.Shutdown(0);
            m_itrainer = null;

            return rgResults;
        }

        /// <summary>
        /// The PreloadData method gives the custom trainer an opportunity to pre-load any data.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nProjectID">Specifies the project ID used, if any.</param>
        /// <returns>When data is pre-loaded the discovered vocabulary is returned as a BucketCollection.</returns>
        BucketCollection IXMyCaffeCustomTrainerRNN.PreloadData(Log log, CancelEvent evtCancel, int nProjectID)
        {
            return preloaddata(log, evtCancel, nProjectID, out m_bUsePreloadData);
        }

        /// <summary>
        /// The ResizeModel method gives the custom trainer the opportunity to resize the model if needed.
        /// </summary>
        /// <param name="strModel">Specifies the model descriptor.</param>
        /// <param name="rgVocabulary">Specifies the vocabulary.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <returns>A new model discriptor is returned (or the same 'strModel' if no changes were made).</returns>
        /// <remarks>Note, this method is called after PreloadData.</remarks>
        string IXMyCaffeCustomTrainerRNN.ResizeModel(Log log, string strModel, BucketCollection rgVocabulary)
        {
            if (rgVocabulary == null || rgVocabulary.Count == 0)
                return strModel;

            int nVocabCount = rgVocabulary.Count;
            NetParameter p = NetParameter.FromProto(RawProto.Parse(strModel));
            string strEmbedName = "";
            EmbedParameter embed = null;
            string strIpName = "";
            InnerProductParameter ip = null;

            foreach (LayerParameter layer in p.layer)
            {
                if (layer.type == LayerParameter.LayerType.EMBED)
                {
                    strEmbedName = layer.name;
                    embed = layer.embed_param;
                }
                else if (layer.type == LayerParameter.LayerType.INNERPRODUCT)
                {
                    strIpName = layer.name;
                    ip = layer.inner_product_param;
                }
            }

            if (embed != null)
            {
                if (embed.input_dim != (uint)nVocabCount)
                {
                    log.WriteLine("WARNING: Embed layer '" + strEmbedName + "' input dim changed from " + embed.input_dim.ToString() + " to " + nVocabCount.ToString() + " to accomodate for the vocabulary count.");
                    embed.input_dim = (uint)nVocabCount;
                }
            }

            if (ip.num_output != (uint)nVocabCount)
            {
                log.WriteLine("WARNING: InnerProduct layer '" + strIpName + "' num_output changed from " + ip.num_output.ToString() + " to " + nVocabCount.ToString() + " to accomodate for the vocabulary count.");
                ip.num_output = (uint)nVocabCount;
            }

            m_rgVocabulary = rgVocabulary;

            RawProto proto = p.ToProto("root");
            return proto.ToString();
        }

        #endregion
    }
}
