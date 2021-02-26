using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.gym;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// Specifies the iterator type to use.
    /// </summary>
    public enum ITERATOR_TYPE
    {
        /// <summary>
        /// Use the iteration type.
        /// </summary>
        ITERATION = 0,
        /// <summary>
        /// Use the episode type.
        /// </summary>
        EPISODE = 1
    }

    /// <summary>
    /// The IXMyCaffeCustomTrainer interface is used by the MyCaffeCustomTraininer components that
    /// provide various training techniques such as Reinforcement Training.
    /// </summary>
    public interface IXMyCaffeCustomTrainer
    {
        /// <summary>
        /// Initialize the trainer passing in a set of key-value pairs as properties.
        /// </summary>
        /// <remarks>Use the ProeprtySet object to easily parse the key-value pair properties.</remarks>
        /// <param name="strProperties">Specifies the properties.</param>
        /// <param name="icallback">Specifies the parent callback for updates.</param>
        void Initialize(string strProperties, IXMyCaffeCustomTrainerCallback icallback);
        /// <summary>
        /// Clean-up the trainer by releasing all resources used.
        /// </summary>
        void CleanUp();
        /// <summary>
        /// Returns the stage that the trainer is running under based on the trainer type.
        /// </summary>
        Stage Stage { get; }
        /// <summary>
        /// Returns the name of the custom trainer.
        /// </summary>
        string Name { get; }
        /// <summary>
        /// Returns the training category supported by the implementer of the interface.
        /// </summary>
        TRAINING_CATEGORY TrainingCategory { get; }
        /// <summary>
        /// Returns <i>true</i> when the training is ready for a snap-shot, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="dfAccuracy">Specifies the current accuracy or rewards for Reinforcement trainers.</param>
        bool GetUpdateSnapshot(out int nIteration, out double dfAccuracy);
        /// <summary>
        /// Returns a dataset override to use (if any) instead of the project's dataset.  If there is no dataset override
        /// <i>null</i> is returned and the project's dataset is used.
        /// </summary>
        /// <param name="nProjectID">Specifies the project ID associated with the trainer (if any)</param>
        /// <param name="ci">Optionally, specifies the database connection information (default = null).</param>
        DatasetDescriptor GetDatasetOverride(int nProjectID, ConnectInfo ci = null);
        /// <summary>
        /// Returns <i>true</i> when the 'Train' method is supported - this should almost always be <i>true</i>. 
        /// </summary>
        bool IsTrainingSupported { get; }
        /// <summary>
        /// Returns <i>true</i> when the 'Test' method is supported.
        /// </summary>
        bool IsTestingSupported { get; }
        /// <summary>
        /// Returns <i>true</i> when the 'Run' method is supported.
        /// </summary>
        bool IsRunningSupported { get; }
        /// <summary>
        /// Train the network using the training technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nIterationOverride">Specifies the iteration override if any.</param>
        /// <param name="type">Specifies the type of iterator to use.</param>
        /// <param name="step">Specifies whether or not to step the training for debugging.</param>
        void Train(Component mycaffe, int nIterationOverride, ITERATOR_TYPE type = ITERATOR_TYPE.ITERATION, TRAIN_STEP step = TRAIN_STEP.NONE);
        /// <summary>
        /// Test the network using the testing technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nIterationOverride">Specifies the iteration override if any.</param>
        /// <param name="type">Specifies the type of iterator to use.</param>
        void Test(Component mycaffe, int nIterationOverride, ITERATOR_TYPE type = ITERATOR_TYPE.ITERATION);
        /// <summary>
        /// Returns a specific property value.
        /// </summary>
        /// <param name="strName">Specifies the property to get.</param>
        /// <returns>The property value is returned.</returns>
        /// <remarks>
        /// The following properties are supported by all trainers:
        ///     'GlobalLoss'
        ///     
        /// The following properties are supported by the RL trainers:
        ///     'GlobalRewards'
        ///     'GlobalEpisodeCount'
        ///     'ExplorationRate'
        ///    
        /// The following properties are supported by the RNN trainers:
        ///     'GlobalIteration'
        /// </remarks>
        double GetProperty(string strName);
        /// <summary>
        /// Returns general information about the custom trainer.
        /// </summary>
        string Information { get; }
        /// <summary>
        /// Open the user interface if one exists for the trainer.
        /// </summary>
        void OpenUi();
    }

    /// <summary>
    /// The IXMyCaffeCustomTrainer interface is used by the MyCaffeCustomTraininer components that
    /// provide various training techniques such as Reinforcement Training.
    /// </summary>
    public interface IXMyCaffeCustomTrainerRL : IXMyCaffeCustomTrainer
    {
        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nDelay">Specifies a delay to wait before getting the action.</param>
        /// <returns>The run results are returned.</returns>
        ResultCollection RunOne(Component mycaffe, int nDelay);
        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="type">Specifies the output data type returned as a raw byte stream.</param>
        /// <returns>The run results are returned in the same native type as that of the CustomQuery used.</returns>
        byte[] Run(Component mycaffe, int nN, out string type);
    }

    /// <summary>
    /// The IXMyCaffeCustomTrainer interface is used by the MyCaffeCustomTraininer components that
    /// provide various training techniques such as Reinforcement Training.
    /// </summary>
    public interface IXMyCaffeCustomTrainerRNN : IXMyCaffeCustomTrainer
    {
        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <returns>The run results are returned.</returns>
        float[] Run(Component mycaffe, int nN);
        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="type">Specifies the output data type returned as a raw byte stream.</param>
        /// <returns>The run results are returned in the same native type as that of the CustomQuery used.</returns>
        byte[] Run(Component mycaffe, int nN, out string type);
        /// <summary>
        /// The PreloadData method gives the custom trainer an opportunity to pre-load any data.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the event used to cancel the pre-load.</param>
        /// <param name="nProjectID">Specifies the ProjectID if any.</param>
        /// <param name="propertyOverride">Optionally, specifies the properites to override those already specified during initialization (default = null).</param>
        /// <param name="ci">Optionally, specifies the database connection information (default = null).</param>
        /// <returns>When data is pre-loaded the vocabulary discovered is returned as a Bucket Collection.</returns>
        BucketCollection PreloadData(Log log, CancelEvent evtCancel, int nProjectID, PropertySet propertyOverride = null, ConnectInfo ci = null);
        /// <summary>
        /// The ResizeModel method gives the custom trainer the opportunity to resize the model if needed.
        /// </summary>
        /// <param name="strModel">Specifies the model descriptor.</param>
        /// <param name="rgVocabulary">Specifies the vocabulary, if any.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <returns>A new model discriptor is returned (or the same 'strModel' if no changes were made).</returns>
        /// <remarks>Note, this method is called after PreloadData.</remarks>
        string ResizeModel(Log log, string strModel, BucketCollection rgVocabulary);
    }

    /// <summary>
    /// The IXMyCaffeCustomTrainerCallback interface is used to call back to the parent running the custom trainer.
    /// </summary>
    public interface IXMyCaffeCustomTrainerCallback
    {
        /// <summary>
        /// The Update method updates the parent with the global iteration, reward and loss.
        /// </summary>
        /// <param name="cat">Specifies the category of the trainer used.</param>
        /// <param name="rgValues">Specifies a dictionary of values that contains 'GlobalIteration', 'GlobalLoss', "LearningRate' and 'GlobalReward' (PG trainers only) values.</param>
        void Update(TRAINING_CATEGORY cat, Dictionary<string, double> rgValues);
    }

    /// <summary>
    /// The IXMyCaffeCustomTrainerCallbackRNN interface is used to call back to the parent running the custom RNN trainer.
    /// </summary>
    public interface IXMyCaffeCustomTrainerCallbackRNN : IXMyCaffeCustomTrainerCallback
    {
        /// <summary>
        /// The GetRunProperties method is used to qeury the properties used when Running, if any.
        /// </summary>
        /// <returns>The property set returned.</returns>
        PropertySet GetRunProperties();
    }

    /// <summary>
    /// The IxTrainer interface is implemented by each Trainer.
    /// </summary>
    public interface IxTrainer
    {
        /// <summary>
        /// Initialize the trainer.
        /// </summary>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Initialize();
        /// <summary>
        /// Shutdown the trainer.
        /// </summary>
        /// <param name="nWait">Specifies a wait for the shtudown.</param>
        /// <returns>Returns <i>true</i>.</returns>
        bool Shutdown(int nWait);
        /// <summary>
        /// Train the network.
        /// </summary>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        /// <param name="step">Specifies whether or not to step the training for debugging.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Train(int nN, ITERATOR_TYPE type, TRAIN_STEP step);
        /// <summary>
        /// Test the newtork.
        /// </summary>
        /// <param name="nN">Specifies the number of iterations (based on the ITERATION_TYPE) to run, or -1 to ignore.</param>
        /// <param name="type">Specifies the iteration type (default = ITERATION).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Test(int nN, ITERATOR_TYPE type);
    }

    /// <summary>
    /// The IxTrainerRL interface is implemented by each RL Trainer.
    /// </summary>
    public interface IxTrainerRL : IxTrainer
    {
        /// <summary>
        /// Run a single cycle on the trainer.
        /// </summary>
        /// <param name="nDelay">Specifies a delay to wait before getting the action.</param>
        /// <returns>The result collection containing the action is returned.</returns>
        ResultCollection RunOne(int nDelay = 1000);

        /// <summary>
        /// Run a number of 'nN' samples on the trainer.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="runProp">Optionally specifies properties to use when running.</param>
        /// <param name="type">Specifies the output data type returned as a raw byte stream.</param>
        /// <returns>The run results are returned in the same native type as that of the CustomQuery used.</returns>
        byte[] Run(int nN, PropertySet runProp, out string type);
    }

    /// <summary>
    /// The IxTrainerRL interface is implemented by each RL Trainer.
    /// </summary>
    public interface IxTrainerRNN : IxTrainer
    {
        /// <summary>
        /// Run a number of 'nN' samples on the trainer.
        /// </summary>
        /// <param name="nN">specifies the number of samples to run.</param>
        /// <param name="runProp">Optionally specifies properties to use when running.</param>
        /// <returns>The result collection containing the action is returned.</returns>
        float[] Run(int nN, PropertySet runProp);

        /// <summary>
        /// Run a number of 'nN' samples on the trainer.
        /// </summary>
        /// <param name="nN">Specifies the number of samples to run.</param>
        /// <param name="runProp">Optionally specifies properties to use when running.</param>
        /// <param name="type">Specifies the output data type returned as a raw byte stream.</param>
        /// <returns>The run results are returned in the same native type as that of the CustomQuery used.</returns>
        byte[] Run(int nN, PropertySet runProp, out string type);
    }

    /// <summary>
    /// The IxTrainerCallback provides functions used by each trainer to 'call-back' to the parent for information and updates.
    /// </summary>
    /// <remarks>The IxTrainerCallback is passed to each trainer.</remarks>
    public interface IxTrainerCallback
    {
        /// <summary>
        /// The OnIntialize callback fires when initializing the trainer.
        /// </summary>
        void OnInitialize(InitializeArgs e);
        /// <summary>
        /// The OnShutdown callback fires when shutting down the trainer.
        /// </summary>
        void OnShutdown();
        /// <summary>
        /// The OnGetData callback fires from within the Train method and is used to get a new observation data.
        /// </summary>
        void OnGetData(GetDataArgs e);
        /// <summary>
        /// The OnGetStatus callback fires on each iteration within the Train method.
        /// </summary>
        void OnUpdateStatus(GetStatusArgs e);
        /// <summary>
        /// The OnWait callback fires when waiting for a shutdown.
        /// </summary>
        void OnWait(WaitArgs e);
    }

    /// <summary>
    /// The IxTrainerGetDataCallback interface is called right after rendering the output image and just before
    /// sending it to the display, thus giving the implementor a chance to 'overlay' information onto the image.
    /// </summary>
    public interface IxTrainerGetDataCallback
    {
        /// <summary>
        /// The OnOverlay method is optionally called just before displaying a gym image thus allowing for an overlay to be applied.
        /// </summary>
        /// <param name="e"></param>
        void OnOverlay(OverlayArgs e);
    }

    /// <summary>
    /// The IxTrainerCallbackRNN provides functions used by each trainer to 'call-back' to the parent for information and updates.
    /// </summary>
    /// <remarks>The IxTrainerCallbackRNN is passed to each RNN trainer.</remarks>
    public interface IxTrainerCallbackRNN : IxTrainerCallback
    {
        /// <summary>
        /// The OnConvertOutput callback fires from within the Run method and is used to convert the network's output into the native format used by the CustomQuery.
        /// </summary>
        /// <param name="e"></param>
        void OnConvertOutput(ConvertOutputArgs e);
    }
}
