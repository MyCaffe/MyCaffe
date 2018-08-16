using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The MyCaffeCustomTraininer is used to perform custom training tasks like those use when performing reinforcement learning.
    /// </summary>
    public partial class MyCaffeCustomTrainer : Component, IXMyCaffeCustomTrainer
    {
        /// <summary>
        /// Specifies the training mode to use (A2C = single trainer), (A3C = multi trainer).
        /// </summary>
        protected TRAINING_MODE m_trainingMode = TRAINING_MODE.SINGLE_INSTANCE;
        /// <summary>
        /// Random number generator used to get initial actions, etc.
        /// </summary>
        protected Random m_random = new Random();
        /// <summary>
        /// Specifies the properties parsed from the key-value pair passed to the Initialize method.
        /// </summary>
        protected PropertySet m_properties = null;
        /// <summary>
        /// Specifies the global rewards.
        /// </summary>
        protected double m_dfGlobalRewards = 0;
        /// <summary>
        /// Specifies the global episode count.
        /// </summary>
        protected int m_nGlobalEpisodeCount = 0;
        /// <summary>
        /// Specifies the maximum number of global episodes.
        /// </summary>
        protected int m_nMaxGlobalEpisodes = 0;
        object m_syncGlobalEpisodeCount = new object();
        object m_syncGlobalRewards = new object();
        IxTrainer m_itrainer = null;
 
        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeCustomTrainer()
        {
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">The container of the component.</param>
        public MyCaffeCustomTrainer(IContainer container)
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
            get { return "MyCaffe Custom Trainer"; }
        }

        /// <summary>
        /// Override when using a training method other than the REINFORCEMENT method (the default).
        /// </summary>
        protected virtual TRAINING_CATEGORY category
        {
            get { return TRAINING_CATEGORY.REINFORCEMENT; }
        }

        /// <summary>
        /// Returns a dataset override to use (if any) instead of the project's dataset.  If there is no dataset override
        /// <i>null</i> is returned and the project's dataset is used.
        /// </summary>
        protected virtual DatasetDescriptor dataset_override
        {
            get { return null; }
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="caffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event to use.</param>
        /// <param name="nGpuID">Optionally, specifies the GPUID to run the trainer on.</param>
        /// <param name="nIndex">Optionally, specifies teh index of the trainer.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerD(Component caffe, Log log, CancelEvent evtCancel, int nGpuID = 0, int nIndex = 0)
        {
            MyCaffeControl<double> mycaffe = caffe as MyCaffeControl<double>;

            if (m_trainingMode == TRAINING_MODE.SINGLE_INSTANCE)
            {
                m_nMaxGlobalEpisodes = mycaffe.CurrentProject.GetSolverSettingAsInt("max_iter").GetValueOrDefault(0);
                Trainer<double> trainer = new Trainer<double>(mycaffe, log, evtCancel, m_properties, m_trainingMode, nGpuID, nIndex);
                trainer.OnInitialize += Trainer_OnInitialize;
                trainer.OnGetData += Trainer_OnGetData;
                trainer.OnGetGlobalEpisodeCount += Trainer_OnGetGlobalEpisodeCount;
                trainer.OnUpdateGlobalRewards += Trainer_OnUpdateGlobalRewards;
                return trainer;
            }
            else
            {
                return null;
            }
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="caffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event to use.</param>
        /// <param name="nGpuID">Optionally, specifies the GPUID to run the trainer on.</param>
        /// <param name="nIndex">Optionally, specifies teh index of the trainer.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerF(Component caffe, Log log, CancelEvent evtCancel, int nGpuID = 0, int nIndex = 0)
        {
            MyCaffeControl<float> mycaffe = caffe as MyCaffeControl<float>;

            if (m_trainingMode == TRAINING_MODE.SINGLE_INSTANCE)
            {
                m_nMaxGlobalEpisodes = mycaffe.CurrentProject.GetSolverSettingAsInt("max_iter").GetValueOrDefault(0);
                Trainer<float> trainer = new Trainer<float>(mycaffe, log, evtCancel, m_properties, m_trainingMode, nGpuID, nIndex);
                trainer.OnInitialize += Trainer_OnInitialize;
                trainer.OnGetData += Trainer_OnGetData;
                trainer.OnGetGlobalEpisodeCount += Trainer_OnGetGlobalEpisodeCount;
                trainer.OnUpdateGlobalRewards += Trainer_OnUpdateGlobalRewards;
                return trainer;
            }
            else
            {
                return null;
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
        /// Override called by the OnGetData event fired by the Trainer to retrieve a new set of observation collections making up a set of experiences.
        /// </summary>
        /// <param name="e">Specifies the getData argments used to return the new observations.</param>
        protected virtual void getData(GetDataArgs e)
        {
        }

        /// <summary>
        /// Override called to get the current explorationr rate.
        /// </summary>
        protected virtual double exploration_rate
        {
            get { return 0; }
        }

        #endregion

        #region IXMyCaffeCustomTrainer Interface

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
        /// Returns a dataset override to use (if any) instead of the project's dataset.  If there is no dataset override
        /// <i>null</i> is returned and the project's dataset is used.
        /// </summary>
        public DatasetDescriptor DatasetOverride
        {
            get { return dataset_override; }
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
        /// Releases any resources used by the component.
        /// </summary>
        public void CleanUp()
        {
        }

        /// <summary>
        /// Initializes a new custom trainer by loading the key-value pair of properties into the property set.
        /// </summary>
        /// <param name="strProperties">Specifies the key-value pair of properties each separated by ';'.  For example the expected
        /// format is 'key1'='value1';'key2'='value2';...</param>
        /// <param name="mode">Specifies the training mode to use A2C (single mode) or A3C (multi mode).</param>
        public void Initialize(string strProperties, TRAINING_MODE mode)
        {
            m_properties = new PropertySet(strProperties);

            if (mode == TRAINING_MODE.USE_PROPERTIES)
            {
                mode = TRAINING_MODE.SINGLE_INSTANCE;
                string strGpu = m_properties.GetProperty("GPUID", false);
                if (strGpu != null)
                {
                    string[] rgID = strGpu.Split(',');
                    if (rgID.Length > 1)
                        mode = TRAINING_MODE.MULTI_INSTANCE;
                }
            }

            m_trainingMode = mode;
        }

        /// <summary>
        /// Create a new trainer and use it to run a test cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        public void Test(Component mycaffe, Log log, CancelEvent evtCancel, int nIterationOverride)
        {
            IxTrainer itrainer;

            if (mycaffe is MyCaffeControl<double>)
                itrainer = create_trainerD(mycaffe, log, evtCancel);
            else
                itrainer = create_trainerF(mycaffe, log, evtCancel);

            itrainer.Initialize();
            itrainer.Test(nIterationOverride);
            ((IDisposable)itrainer).Dispose();
        }

        /// <summary>
        /// Create a new trainer and use it to run a training cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        /// <param name="step">Optionally, specifies whether or not to step the training for debugging (default = NONE).</param>
        public void Train(Component mycaffe, Log log, CancelEvent evtCancel, int nIterationOverride, TRAIN_STEP step = TRAIN_STEP.NONE)
        {
            m_nGlobalEpisodeCount = 0;
            m_dfGlobalRewards = 0;

            if (step == TRAIN_STEP.NONE)
            {
                if (m_itrainer != null)
                {
                    ((IDisposable)m_itrainer).Dispose();
                    m_itrainer = null;
                }
            }

            if (m_itrainer == null)
            {
                if (mycaffe is MyCaffeControl<double>)
                    m_itrainer = create_trainerD(mycaffe, log, evtCancel);
                else
                    m_itrainer = create_trainerF(mycaffe, log, evtCancel);

                m_itrainer.Initialize();
            }

            if (nIterationOverride > 0)
                m_nMaxGlobalEpisodes = nIterationOverride;

            m_itrainer.Train(nIterationOverride, step);

            if (step == TRAIN_STEP.NONE)
            {
                ((IDisposable)m_itrainer).Dispose();
                m_itrainer = null;
            }
        }

        #endregion

        private void Trainer_OnInitialize(object sender, InitializeArgs e)
        {
            initialize(e);
        }

        private void Trainer_OnGetData(object sender, GetDataArgs e)
        {
            getData(e);
        }

        private void Trainer_OnGetGlobalEpisodeCount(object sender, GlobalEpisodeCountArgs e)
        {
            lock (m_syncGlobalEpisodeCount)
            {
                e.GlobalEpisodeCount = m_nGlobalEpisodeCount;
                e.MaximumGlobalEpisodeCount = m_nMaxGlobalEpisodes;
                m_nGlobalEpisodeCount++;
            }
        }

        private void Trainer_OnUpdateGlobalRewards(object sender, UpdateGlobalRewardArgs e)
        {
            lock (m_syncGlobalRewards)
            {
                if (m_dfGlobalRewards == 0)
                    m_dfGlobalRewards = e.Reward;
                else
                    m_dfGlobalRewards = m_dfGlobalRewards * 0.99 + e.Reward * 0.01;
            }
        }

        /// <summary>
        /// Returns the global rewards for either the single MyCaffeCustomTrainer (when A2C) or the set of MyCaffeCustomTrainers (when A3C)
        /// </summary>
        public double GlobalRewards
        {
            get { return m_dfGlobalRewards; }
        }

        /// <summary>
        /// Returns the global episode count for either the single MyCaffeCustomTrainer (when A2C) or the set of MyCaffeCustomTrainers (when A3C)
        /// </summary>
        public int GlobalEpisodeCount
        {
            get { return m_nGlobalEpisodeCount; }
        }

        /// <summary>
        /// Returns the current exploration rate.
        /// </summary>
        public double ExplorationRate
        {
            get { return exploration_rate; }
        }
    }
}
