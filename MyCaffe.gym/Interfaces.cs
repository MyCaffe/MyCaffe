using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// Defines the Standard GYM Dataset ID's.
    /// </summary>
    public enum GYM_DS_ID
    {
        /// <summary>
        /// Specifies the Standard CARTPOLE GYM Dataset ID.
        /// </summary>
        CARTPOLE = 99990001,
        /// <summary>
        /// Specifies the Standard ATARI GYM Dataset ID.
        /// </summary>
        ATARI = 99990002,
        /// <summary>
        /// Specifies the Standard DATAGENERAL GYM Dataset ID.
        /// </summary>
        DATAGENERAL = 99991001,
    }

    /// <summary>
    /// Defines the Standard GYM Training Data Source ID's.
    /// </summary>
    public enum GYM_SRC_TRAIN_ID
    {
        /// <summary>
        /// Specifies the Standard CARTPOLE GYM Training Dataset ID.
        /// </summary>
        CARTPOLE = 99995001,
        /// <summary>
        /// Specifies the Standard ATARI GYM Training Dataset ID.
        /// </summary>
        ATARI = 99995002,
        /// <summary>
        /// Specifies the Standard DATAGENERAL GYM Training Dataset ID.
        /// </summary>
        DATAGENERAL = 99996001
    }

    /// <summary>
    /// Defines the Standard GYM Testing Data Source ID's.
    /// </summary>
    public enum GYM_SRC_TEST_ID
    {
        /// <summary>
        /// Specifies the Standard CARTPOLE GYM Testing Dataset ID.
        /// </summary>
        CARTPOLE = 99997001,
        /// <summary>
        /// Specifies the Standard ATARI GYM Testing Dataset ID.
        /// </summary>
        ATARI = 99997002,
        /// <summary>
        /// Specifies the Standard DATAGENERAL GYM Testing Dataset ID.
        /// </summary>
        DATAGENERAL = 99998001
    }

    /// <summary>
    /// Defines the gym data type.
    /// </summary>
    public enum DATA_TYPE
    {
        /// <summary>
        /// Specifies to use the default data type of the gym used.
        /// </summary>
        DEFAULT,
        /// <summary>
        /// Specifies to use the raw state values of the gym (if supported).
        /// </summary>
        VALUES,
        /// <summary>
        /// Specifies to use a SimpleDatum blob of data of the gym (if supported).
        /// </summary>
        BLOB
    }

    /// <summary>
    /// The IXMyCaffeGym interface is used to interact with each Gym.
    /// </summary>
    public interface IXMyCaffeGym
    {
        /// <summary>
        /// Initialize the gym using the properties in the PropertySet.
        /// </summary>
        /// <param name="log">Specifies the output Log for the gym to use.</param>
        /// <param name="properties">Specifies the properties used to initialize the gym.</param>
        void Initialize(Log log, PropertySet properties);
        /// <summary>
        /// Close a previously initialized gym.
        /// </summary>
        void Close();
        /// <summary>
        /// Copy a gym creating a new one.
        /// </summary>
        /// <param name="properties">Optionally, specifies the properties used to initialize the gym (default = <i>null</i> which skips calling Initialize).</param>
        /// <returns>The new gym copy is returned.</returns>
        IXMyCaffeGym Clone(PropertySet properties = null);
        /// <summary>
        /// Returns the name of the gym.
        /// </summary>
        string Name { get; }
        /// <summary>
        /// Resets the state of they gym.
        /// </summary>
        /// <param name="bGetLabel">Optionally, specifies to query the label (default = false).</param>
        /// <returns>A tuple containing state information and gym data, the reward and whether or not the gym is done is returned.</returns>
        Tuple<State, double, bool> Reset(bool bGetLabel = false);
        /// <summary>
        /// Run an action on the gym.
        /// </summary>
        /// <param name="nAction">Specifies the action to run, which is an index into the action space.</param>
        /// <param name="bGetLabel">Optionally, specifies to query the label (default = false).</param>
        /// <returns>A tuple containing state information and gym data, the reward and whether or not the gym is done is returned.</returns>
        Tuple<State, double, bool> Step(int nAction, bool bGetLabel = false);
        /// <summary>
        /// Render the gym on a bitmap.
        /// </summary>
        /// <param name="bShowUi">Specifies whether or not the gym rendering is intended for the user interface.</param>
        /// <param name="nWidth">Specifies the width of the user interface.</param>
        /// <param name="nHeight">Specifies the height of the user interface.</param>
        /// <param name="bGetAction">Specifies to get the action data.</param>
        /// <returns>A tuple containing image showing the gym and the action data is returned.</returns>
        Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, bool bGetAction);
        /// <summary>
        /// Render the gym on a bitmap.
        /// </summary>
        /// <param name="bShowUi">Specifies whether or not the gym rendering is intended for the user interface.</param>
        /// <param name="nWidth">Specifies the width of the user interface.</param>
        /// <param name="nHeight">Specifies the height of the user interface.</param>
        /// <param name="rgData">Specifies the state information of the gym.</param>
        /// <param name="bGetAction">Specifies to collect the action data.</param>
        /// <returns>A tuple containing image showing the gym and the action data is returned.</returns>
        Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, double[] rgData, bool bGetAction);
        /// <summary>
        /// Returns a dictionary containing the action space where each entry contains the action name and action value.
        /// </summary>
        /// <returns>The action space dictionary is returned.</returns>
        Dictionary<string, int> GetActionSpace();
        /// <summary>
        /// Returns the dataset of the gym.
        /// </summary>
        /// <param name="dt">Specifies the datatype to use.</param>
        /// <param name="log">Optionally, specifies a Log override to use (default = <i>null</i>).</param>
        /// <returns>The dataset descriptor is returned.</returns>
        DatasetDescriptor GetDataset(DATA_TYPE dt, Log log = null);
        /// <summary>
        /// Returns the user-interface delay to use (if any).
        /// </summary>
        int UiDelay { get; }
        /// <summary>
        /// Returns the selected data-type.
        /// </summary>
        DATA_TYPE SelectedDataType { get; }
        /// <summary>
        /// Returns an array of data types supported by the gym.
        /// </summary>
        DATA_TYPE[] SupportedDataType { get; }
        /// <summary>
        /// Returns whether or not the gym requires the display image.
        /// </summary>
        bool RequiresDisplayImage { get; }
    }

    /// <summary>
    /// The IXMyCaffeGym interface is used to interact with each Gym.
    /// </summary>
    public interface IXMyCaffeGymData : IXMyCaffeGym
    {
        /// <summary>
        /// Converts the output values into the native type used by the Gym during queries.
        /// </summary>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="type">Returns the output type.</param>
        /// <returns>The converted output data is returned.</returns>
        byte[] ConvertOutput(float[] rg, out string type);
    }

    /// <summary>
    /// The IXMyCaffeGymRange interface is used to query the data range for the vocabulary.
    /// </summary>
    public interface IXMyCaffeGymRange : IXMyCaffeGym
    {
        /// <summary>
        /// Returns the data range of the gym which is used to build the vocabulary.
        /// </summary>
        /// <returns></returns>
        Tuple<double, double> GetDataRange();
        /// <summary>
        /// Returns true to use the fixed bucket collection based on the GetDataRange values, otherwise the bucket collection is created from the data.
        /// </summary>
        bool UseFixedVocabulary { get; }
        /// <summary>
        /// Returns the vocabulary size to use (e.g. the number of buckets).
        /// </summary>
        int VocabularySize { get; }
        /// <summary>
        /// Specifies whether or not to use the pre-load data.
        /// </summary>
        bool UsePreLoadData { get; }
    }

    /// <summary>
    /// The State class defines an abstract base class for the state information and gym data.
    /// </summary>
    public abstract class State
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public State()
        {
        }
        /// <summary>
        /// Copies the state to another new state.
        /// </summary>
        /// <returns>The new state information and gym data is returned.</returns>
        public abstract State Clone();
        /// <summary>
        /// Get the data.
        /// </summary>
        /// <param name="bNormalize">Specifies to normalize the data.</param>
        /// <param name="nDataLen">Returns the non-render data length (the actual data used in training)</param>
        /// <returns>A tuple with the SimpleDatum containing the data and the non rendering data length is returned.</returns>
        public abstract SimpleDatum GetData(bool bNormalize, out int nDataLen);
        /// <summary>
        /// Returns 'true' if this State supports clip data.
        /// </summary>
        public virtual bool HasClip { get { return false; } }
        /// <summary>
        /// Get the clip data (used with recurrent models such as LSTM).
        /// </summary>
        /// <returns>The clip data corresponding to the GetData is returned.</returns>
        public virtual SimpleDatum GetClip() { throw new NotImplementedException(); }
        /// <summary>
        /// Returns 'true' if this state supports label data.
        /// </summary>
        public virtual bool HasLabel {  get { return false; } }
        /// <summary>
        /// Get the label data (used with recurrent models such as LSTM with dynamic data).
        /// </summary>
        /// <returns>The label data is returned.</returns>
        public virtual SimpleDatum GetLabel() { throw new NotImplementedException(); }
    }
}
