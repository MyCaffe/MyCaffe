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
        /// Random number generator used to get initial actions, etc.
        /// </summary>
        protected Random m_random = new Random();
        /// <summary>
        /// Specifies the properties parsed from the key-value pair passed to the Initialize method.
        /// </summary>
        protected PropertySet m_properties = null;
        IxTrainer m_itrainer = null;
        double m_dfExplorationRate = 0;
        double m_dfGlobalRewards = 0;
        int m_nGlobalEpisodeCount = 0;
        int m_nGlobalEpisodeMax = 0;
 
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
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerD(Component caffe)
        {
            MyCaffeControl<double> mycaffe = caffe as MyCaffeControl<double>;

            Trainer<double> trainer = new Trainer<double>(mycaffe, m_properties, m_random);
            trainer.OnInitialize += Trainer_OnInitialize;
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

            Trainer<float> trainer = new Trainer<float>(mycaffe, m_properties, m_random);
            trainer.OnInitialize += Trainer_OnInitialize;
            trainer.OnGetData += Trainer_OnGetData;
            trainer.OnGetStatus += Trainer_OnGetStatus;
            return trainer;
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
        public void Initialize(string strProperties)
        {
            m_properties = new PropertySet(strProperties);
        }

        /// <summary>
        /// Create a new trainer and use it to run a test cycle.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl to use.</param>
        /// <param name="nIterationOverride">Specifies the iterations to run if greater than zero.</param>
        public void Test(Component mycaffe, int nIterationOverride)
        {
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
            {
                if (mycaffe is MyCaffeControl<double>)
                    m_itrainer = create_trainerD(mycaffe);
                else
                    m_itrainer = create_trainerF(mycaffe);

                m_itrainer.Initialize();
            }

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
            getData(e);
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
    }
}
