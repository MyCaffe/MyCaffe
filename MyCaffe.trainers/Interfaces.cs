using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// Defines the category of training.
    /// </summary>
    public enum TRAINING_CATEGORY
    {
        /// <summary>
        /// Defines a purely custom training method.
        /// </summary>
        CUSTOM,
        /// <summary>
        /// Defines the reinforcement training method such as A2C or A3C.
        /// </summary>
        REINFORCEMENT
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
        /// <param name="strProperties"></param>
        void Initialize(string strProperties);
        /// <summary>
        /// Clean-up the trainer by releasing all resources used.
        /// </summary>
        void CleanUp();
        /// <summary>
        /// Returns the name of the custom trainer.
        /// </summary>
        string Name { get; }
        /// <summary>
        /// Returns the training category supported by the implementer of the interface.
        /// </summary>
        TRAINING_CATEGORY TrainingCategory { get; }
        /// <summary>
        /// Returns a dataset override to use (if any) instead of the project's dataset.  If there is no dataset override
        /// <i>null</i> is returned and the project's dataset is used.
        /// </summary>
        DatasetDescriptor DatasetOverride { get; }
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
        /// <param name="step">Specifies whether or not to step the training for debugging.</param>
        void Train(Component mycaffe, int nIterationOverride, TRAIN_STEP step);
        /// <summary>
        /// Test the network using the testing technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nIterationOverride">Specifies the iteration override if any.</param>
        void Test(Component mycaffe, int nIterationOverride);
        /// <summary>
        /// Run the network using the run technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="nDelay">Specifies a delay to wait before getting the action.</param>
        /// <returns>The run results are returned.</returns>
        ResultCollection Run(Component mycaffe, int nDelay);
        /// <summary>
        /// Returns the global rewards.
        /// </summary>
        double GlobalRewards { get; }
        /// <summary>
        /// Returns the global episode count.
        /// </summary>
        int GlobalEpisodeCount { get; }
        /// <summary>
        /// Returns the current exploration rate.
        /// </summary>
        double ExplorationRate { get; }
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
        /// <returns></returns>
        bool Shutdown();
        /// <summary>
        /// Train the network.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <param name="step">Specifies whether or not to step the training for debugging.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Train(int nIterations, TRAIN_STEP step);
        /// <summary>
        /// Test the newtork.
        /// </summary>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        bool Test(int nIterations);
        /// <summary>
        /// Run a single cycle on the trainer.
        /// </summary>
        /// <param name="nDelay">Specifies a delay to wait before getting the action.</param>
        /// <returns>The result collection containing the action is returned.</returns>
        ResultCollection Run(int nDelay = 1000);
    }
}
