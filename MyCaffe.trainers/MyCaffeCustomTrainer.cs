using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The MyCaffeCustomTraininer is used to perform custom training tasks like those use when performing reinforcement learning.
    /// </summary>
    public partial class MyCaffeCustomTrainer : Component, IXMyCaffeCustomTrainer
    {
        /// <summary>
        /// Specifies the properties parsed from the key-value pair passed to the Initialize method.
        /// </summary>
        protected PropertySet m_properties = null;

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
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>double</i> base type.
        /// </remarks>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event to use.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerD(Component mycaffe, Log log, CancelEvent evtCancel)
        {
            SimpleTrainer<double> trainer = new SimpleTrainer<double>(mycaffe as MyCaffeControl<double>, log, evtCancel, m_properties);
            trainer.OnInitialize += Trainer_OnInitialize;
            trainer.OnGetObservations += Trainer_OnGetObservations;
            trainer.OnProcessObservations += Trainer_OnProcessObservations;
            return trainer;
        }

        /// <summary>
        /// Optionally overridden to return a new type of trainer.
        /// </summary>
        /// <remarks>
        /// Override this method when using the MyCaffeControl that uses the <i>float</i> base type.
        /// </remarks>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event to use.</param>
        /// <returns>The IxTraininer interface implemented by the new trainer is returned.</returns>
        protected virtual IxTrainer create_trainerF(Component mycaffe, Log log, CancelEvent evtCancel)
        {
            SimpleTrainer<float> trainer = new SimpleTrainer<float>(mycaffe as MyCaffeControl<float>, log, evtCancel, m_properties);
            trainer.OnInitialize += Trainer_OnInitialize;
            trainer.OnGetObservations += Trainer_OnGetObservations;
            trainer.OnProcessObservations += Trainer_OnProcessObservations;
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
        /// Override called by the OnGetObservations event fired by the Trainer to retrieve a new set of observation collections making up a set of experiences.
        /// </summary>
        /// <param name="e">Specifies the getObservations argments used to return the new observations.</param>
        protected virtual void getObservations(GetObservationArgs e)
        {
        }

        /// <summary>
        /// Override called by the OnProcessObservations event fired by the Trainer when training the network.  Processing the observations should run the
        /// network on the data of the observation and determine a new set of actions to take, take those actions and determine the reward.
        /// </summary>
        /// <param name="e">Specifies the processObservations arguments that contain the observations and returns the new actions and reward.</param>
        protected virtual void processObservations(ProcessObservationArgs e)
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
        public void Train(Component mycaffe, Log log, CancelEvent evtCancel, int nIterationOverride)
        {
            IxTrainer itrainer;

            if (mycaffe is MyCaffeControl<double>)
                itrainer = create_trainerD(mycaffe, log, evtCancel);
            else
                itrainer = create_trainerF(mycaffe, log, evtCancel);

            itrainer.Initialize();
            itrainer.Train(nIterationOverride);
            ((IDisposable)itrainer).Dispose();
        }

        /// <summary>
        /// Defines the event handler used to handle the trainers OnInitialize event.
        /// </summary>
        /// <remarks>
        /// This event fires when the trainer is initialized.
        /// </remarks>
        /// <param name="sender">Specifies the sender, which is the trainer.</param>
        /// <param name="e">Specifies the event arguments.</param>
        private void Trainer_OnInitialize(object sender, InitializeArgs e)
        {
            initialize(e);
        }

        /// <summary>
        /// Defines the event handler used to handle the trainers OnProcessObservations event.
        /// </summary>
        /// <remarks>
        /// This event fires when the trainer needs to process the observations by running the
        /// network on each observation, run each new action and determine the new reward for each
        /// observation.
        /// </remarks>
        /// <param name="sender">Specifies the sender, which is the trainer.</param>
        /// <param name="e">Specifies the event arguments.</param>
        private void Trainer_OnProcessObservations(object sender, ProcessObservationArgs e)
        {
            processObservations(e);
        }

        /// <summary>
        /// Defines the event handler used to handle the trainers OnGetObservations event.
        /// </summary>
        /// <remarks>
        /// This event fires when the trainer needs to collect a new set of observations.
        /// </remarks>
        /// <param name="sender">Specifies the sender, which is the trainer.</param>
        /// <param name="e">Specifies the event arguments.</param>
        private void Trainer_OnGetObservations(object sender, GetObservationArgs e)
        {
            getObservations(e);
        }

        #endregion
    }
}
