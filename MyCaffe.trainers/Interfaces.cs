using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{

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
        /// Returns <i>true</i> when the 'Train' method is supported - this should almost always be <i>true</i>. 
        /// </summary>
        bool IsTrainingSupported { get; }
        /// <summary>
        /// Returns <i>true</i> when the 'Test' method is supported.
        /// </summary>
        bool IsTestingSupported { get; }
        /// <summary>
        /// Train the network using the training technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="log">Specifies the output Log object.</param>
        /// <param name="evtCancel">Specifies the cancel event used to halt training.</param>
        /// <param name="nIterationOverride">Specifies the iteration override if any.</param>
        void Train(Component mycaffe, Log log, CancelEvent evtCancel, int nIterationOverride);
        /// <summary>
        /// Test the network using the testing technique implemented by this trainer.
        /// </summary>
        /// <param name="mycaffe">Specifies an instance to the MyCaffeControl component.</param>
        /// <param name="log">Specifies the output Log object.</param>
        /// <param name="evtCancel">Specifies the cancel event used to halt training.</param>
        /// <param name="nIterationOverride">Specifies the iteration override if any.</param>
        void Test(Component mycaffe, Log log, CancelEvent evtCancel, int nIterationOverride);
    }
}
