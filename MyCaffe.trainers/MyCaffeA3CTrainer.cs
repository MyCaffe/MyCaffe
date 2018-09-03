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

namespace MyCaffe.trainers
{
    /// <summary>
    /// The MyCaffeA3CTraininer is used to perform A3C training tasks which are used for reinforcement learning.
    /// </summary>
    public partial class MyCaffeA3CTrainer : Component, IXMyCaffeCustomTrainer
    {
        /// <summary>
        /// Random number generator used to get initial actions, etc.
        /// </summary>
        protected Random m_random = new Random();
        /// <summary>
        /// Specifies the properties parsed from the key-value pair passed to the Initialize method.
        /// </summary>
        protected PropertySet m_properties = null;
        /// <summary>
        /// Specifies the project ID of the project held by the instance of MyCaffe.
        /// </summary>
        protected int m_nProjectID = 0;
        /// <summary>
        /// Specifies the observation collection filled by your class that overrides the trainer.
        /// </summary>
        protected ObservationReadyCollection m_rgObservations = new ObservationReadyCollection();
        IxTrainer m_itrainer = null;
        double m_dfExplorationRate = 0;
        double m_dfGlobalRewards = 0;
        int m_nGlobalEpisodeCount = 0;
        int m_nGlobalEpisodeMax = 0;
 
        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeA3CTrainer()
        {
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">The container of the component.</param>
        public MyCaffeA3CTrainer(IContainer container)
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
            get { return "MyCaffe A3C Trainer"; }
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
        /// <param name="nProjectID">Specifies the project ID associated with the trainer (if any)</param>
        protected virtual DatasetDescriptor get_dataset_override(int nProjectID)
        {
            return null;
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="caffe">Specifies the MyCaffeControl used.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerD(Component caffe)
        {
            MyCaffeControl<double> mycaffe = caffe as MyCaffeControl<double>;
            m_nProjectID = mycaffe.CurrentProject.ID;
            m_rgObservations.CancelEvent = mycaffe.CancelEvent;

            Trainer<double> trainer = new Trainer<double>(mycaffe, m_properties, m_random);
            trainer.OnInitialize += Trainer_OnInitialize;
            trainer.OnShutdown += Trainer_OnShutdown;
            trainer.OnGetData += Trainer_OnGetData;
            trainer.OnGetStatus += Trainer_OnGetStatus;
            return trainer;
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="caffe">Specifies the MyCaffeControl used.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerF(Component caffe)
        {
            MyCaffeControl<float> mycaffe = caffe as MyCaffeControl<float>;
            m_nProjectID = mycaffe.CurrentProject.ID;
            m_rgObservations.CancelEvent = mycaffe.CancelEvent;

            Trainer<float> trainer = new Trainer<float>(mycaffe, m_properties, m_random);
            trainer.OnInitialize += Trainer_OnInitialize;
            trainer.OnShutdown += Trainer_OnShutdown;
            trainer.OnGetData += Trainer_OnGetData;
            trainer.OnGetStatus += Trainer_OnGetStatus;
            return trainer;
        }

        private void Trainer_OnShutdown(object sender, EventArgs e)
        {
            shutdown();
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
        /// Open the Gym UI if one exists.
        /// </summary>
        protected virtual void open()
        {
        }

        /// <summary>
        /// Returns <i>true</i> when the training is ready for a snap-shot, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="dfRewards">Specifies the best rewards to this point.</param>
        protected virtual bool get_update_snapshot(out int nIteration, out double dfRewards)
        {
            nIteration = 0;
            dfRewards = 0;
            return false;
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
        /// Returns <i>true</i> when the training is ready for a snap-shot, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="dfRewards">Specifies the best rewards to this point.</param>
        public bool GetUpdateSnapshot(out int nIteration, out double dfRewards)
        {
            return get_update_snapshot(out nIteration, out dfRewards);
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
        public void Initialize(string strProperties)
        {
            m_properties = new PropertySet(strProperties);
        }

        private IxTrainer createTrainer(Component mycaffe)
        {
            IxTrainer itrainer = null;

            if (mycaffe is MyCaffeControl<double>)
                itrainer = create_trainerD(mycaffe);
            else
                itrainer = create_trainerF(mycaffe);

            itrainer.Initialize();

            return itrainer;
        }

        /// <summary>
        /// Create a new trainer and use it to run a single run cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nDelay">Specifies a delay to wait before getting the action.</param>
        /// <returns>The results of the run are returned.</returns>
        public ResultCollection Run(Component mycaffe, int nDelay = 1000)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe);

            ResultCollection res = m_itrainer.Run(nDelay);
            m_itrainer = null;

            return res;
        }

        /// <summary>
        /// Create a new trainer and use it to run a test cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        public void Test(Component mycaffe, int nIterationOverride)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe);

            m_itrainer.Test(nIterationOverride);
            m_itrainer = null;
        }

        /// <summary>
        /// Create a new trainer and use it to run a training cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        /// <param name="step">Optionally, specifies whether or not to step the training for debugging (default = NONE).</param>
        public void Train(Component mycaffe, int nIterationOverride, TRAIN_STEP step = TRAIN_STEP.NONE)
        {
            if (m_itrainer == null)
                m_itrainer = createTrainer(mycaffe);

            m_itrainer.Train(nIterationOverride, step);
            m_itrainer = null;
        }

        #endregion

        private void Trainer_OnInitialize(object sender, InitializeArgs e)
        {
            initialize(e);
        }

        private void Trainer_OnGetData(object sender, GetDataArgs e)
        {
            e.Success = getData(e);
        }

        private void Trainer_OnGetStatus(object sender, GetStatusArgs e)
        {
            m_dfGlobalRewards = Math.Max(m_dfGlobalRewards, e.Reward);
            m_dfExplorationRate = e.ExplorationRate;
            m_nGlobalEpisodeCount = Math.Max(m_nGlobalEpisodeCount, e.Frames);
            m_nGlobalEpisodeMax = e.MaxFrames;
        }

        /// <summary>
        /// Returns the global rewards.
        /// </summary>
        public double GlobalRewards
        {
            get { return m_dfGlobalRewards; }
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
        /// Open the GYM UI if one exists.
        /// </summary>
        public void Open()
        {
            open();   
        }
    }
}
