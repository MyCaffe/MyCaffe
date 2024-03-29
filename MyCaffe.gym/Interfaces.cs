﻿using MyCaffe.basecode;
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
        /// Specifies the Curve GYM Dataset ID.
        /// </summary>
        CURVE = 99990003,
        /// <summary>
        /// Specifies the Standard DATAGENERAL GYM Dataset ID.
        /// </summary>
        DATAGENERAL = 99991001,
        /// <summary>
        /// Specifies the Standard MODEL GYM Dataset ID.
        /// </summary>
        MODEL = 99991002
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
        /// Specifies the Curve GYM Dataset ID.
        /// </summary>
        CURVE = 99995003,
        /// <summary>
        /// Specifies the Standard DATAGENERAL GYM Training Dataset ID.
        /// </summary>
        DATAGENERAL = 99996001,
        /// <summary>
        /// Specifies the Standard MODEL Training GYM Dataset ID.
        /// </summary>
        MODEL = 99996002
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
        /// Specifies the Curve GYM Dataset ID.
        /// </summary>
        CURVE = 99997003,
        /// <summary>
        /// Specifies the Standard DATAGENERAL GYM Testing Dataset ID.
        /// </summary>
        DATAGENERAL = 99998001,
        /// <summary>
        /// Specifies the Standard MODEL GYM Testing Dataset ID.
        /// </summary>
        MODEL = 99998002
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
        /// Reset the base value (if any).
        /// </summary>
        void ResetValue();
        /// <summary>
        /// Resets the state of they gym.
        /// </summary>
        /// <param name="bGetLabel">Optionally, specifies to query the label (default = false).</param>
        /// <param name="props">Optionally, specifies the properties used when resetting.</param>
        /// <returns>A tuple containing state information and gym data, the reward and whether or not the gym is done is returned.</returns>
        Tuple<State, double, bool> Reset(bool bGetLabel = false, PropertySet props = null);
        /// <summary>
        /// Run an action on the gym.
        /// </summary>
        /// <param name="nAction">Specifies the action to run, which is an index into the action space.</param>
        /// <param name="bGetLabel">Optionally, specifies to query the label (default = false).</param>
        /// <param name="extraProp">Optionally, specifies extra properties.</param>
        /// <returns>A tuple containing state information and gym data, the reward and whether or not the gym is done is returned.</returns>
        Tuple<State, double, bool> Step(int nAction, bool bGetLabel = false, PropertySet extraProp = null);
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
        /// <param name="predictions">Optionally, specifies the future predictions.</param>
        /// <returns>A tuple containing image showing the gym and the action data is returned.</returns>
        Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, double[] rgData, bool bGetAction, FuturePredictions predictions = null);
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
        /// <summary>
        /// Returns the percentage of the data to use for testing, or -1 which then uses the default of 0.2.
        /// </summary>
        double TestingPercent { get; }
    }

    /// <summary>
    /// The IXMyCaffeGym interface is used to interact with each Gym.
    /// </summary>
    public interface IXMyCaffeGymData : IXMyCaffeGym
    {
        /// <summary>
        /// Converts the output values into the native type used by the Gym during queries.
        /// </summary>
        /// <param name="stage">Specifies the stage under which the conversion is run.</param>
        /// <param name="nN">Specifies the number of outputs.</param>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="type">Returns the output type.</param>
        /// <returns>The converted output data is returned.</returns>
        byte[] ConvertOutput(Stage stage, int nN, float[] rg, out string type);
        /// <summary>
        /// Specifies the active phase under which to get the data reset and next.
        /// </summary>
        Phase ActivePhase { get; set; }
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
    /// The FuturePredictions manage the extended future prediction data.
    /// </summary>
    public class FuturePredictions
    {
        int m_nStartOffset;
        List<float> m_rgfPredictions = new List<float>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nStartOffset">specifies the offset of the future prediction (offset from the last input data)</param>
        public FuturePredictions(int nStartOffset)
        {
            m_nStartOffset = nStartOffset;
        }

        /// <summary>
        /// Return a copy of the FuturePredictions.
        /// </summary>
        /// <returns>A copy of the future predictions is returned.</returns>
        public FuturePredictions Clone()
        {
            FuturePredictions fp = new FuturePredictions(m_nStartOffset);

            foreach (float f in m_rgfPredictions)
            {
                fp.Add(f);
            }

            return fp;
        }

        /// <summary>
        /// Add a prediction.
        /// </summary>
        /// <param name="fVal">Specifies the future prediction to add.</param>
        public void Add(float fVal)
        {
            m_rgfPredictions.Add(fVal);
        }

        /// <summary>
        /// Returns the start offset of the future prediction (offset from the last input data).
        /// </summary>
        public int StartOffset
        {
            get { return m_nStartOffset; }
        }

        /// <summary>
        /// Returns the future predictions.
        /// </summary>
        public List<float> Predictions
        {
            get { return m_rgfPredictions; }
        }
    }

    /// <summary>
    /// The FuturePredictionsCollection manages a collection of FuturePredictions.
    /// </summary>
    public class FuturePredictionsCollection : IEnumerable<FuturePredictions> 
    {
        int m_nMaxCount;
        List<FuturePredictions> m_rgItems = new List<FuturePredictions>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public FuturePredictionsCollection(int nMaxCount = int.MaxValue)
        {
            m_nMaxCount = nMaxCount;
        }

        /// <summary>
        /// The copy constructor.
        /// </summary>
        /// <param name="col">Specifies the FuturePredictionCollection to copy.</param>
        public FuturePredictionsCollection(FuturePredictionsCollection col)
        {
            foreach (FuturePredictions fp in col)
            {
                Add(fp.Clone());
            }
        }

        /// <summary>
        /// Clear the collection.
        /// </summary>
        public void Clear()
        {
            m_rgItems.Clear();
        }

        /// <summary>
        /// Add a new set of future predictions.
        /// </summary>
        /// <param name="fp">Specifies the future predictions to add.</param>
        public void Add(FuturePredictions fp)
        {
            m_rgItems.Add(fp);

            if (m_rgItems.Count > m_nMaxCount)
                m_rgItems.RemoveAt(0);
        }

        /// <summary>
        /// Remove the future predictions at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to remove.</param>
        /// <returns>If the item is removed, True is returned, otherwise False is returned.</returns>
        public bool RemoveAt(int nIdx)
        {
            if (nIdx < 0 || nIdx >= m_rgItems.Count)
                return false;

            m_rgItems.RemoveAt(nIdx);
            return true;
        }

        /// <summary>
        /// Returns the future predictions at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The future predictions at the index is returned.</returns>
        public FuturePredictions this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
        }

        /// <summary>
        /// Returns the number of future predictions.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Returns the maximum number of future predictions.
        /// </summary>
        public int MaxCount
        {
            get { return m_nMaxCount; }
        }

        /// <summary>
        /// Gets the enumerator.
        /// </summary>
        /// <returns>Returns the enumerator for the colelction.</returns>
        public IEnumerator<FuturePredictions> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        /// <summary>
        /// Gets the enumerator.
        /// </summary>
        /// <returns>Returns the enumerator for the colelction.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }
    }


    /// <summary>
    /// The DataPoint contains the data used when training.
    /// </summary>
    public class DataPoint
    {
        float[] m_rgfInputs;
        float[] m_rgfMask;
        float m_fTarget;
        float m_fTime;
        List<double> m_rgdfPredicted = null;
        List<string> m_rgstrPredicted = null;
        List<bool> m_rgbEmphasize = null;
        FuturePredictions m_predictions = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rgfInputs">Specifies the inputs.</param>
        /// <param name="rgfMask">Specifies the mask, where 1 indicates that the input at that same location is valid.</param>
        /// <param name="fTarget">Specifies the output target.</param>
        /// <param name="rgdfPredicted">Specifies the predicted values.</param>
        /// <param name="rgstrPredicted">Specifies the predicted labels.</param>
        /// <param name="rgbEmphasize">Specifies which predicted labels to emphasize.</param>
        /// <param name="fTime">Specifies the time of the data point.</param>
        /// <param name="predictions">Specifies the future predictions.</param>
        public DataPoint(float[] rgfInputs, float[] rgfMask, float fTarget, List<double> rgdfPredicted, List<string> rgstrPredicted, List<bool> rgbEmphasize, float fTime, FuturePredictions predictions)
        {
            m_rgfInputs = rgfInputs;
            m_rgfMask = rgfMask;
            m_fTarget = fTarget;
            m_fTime = fTime;

            m_rgdfPredicted = rgdfPredicted;
            m_rgstrPredicted = rgstrPredicted;
            m_rgbEmphasize = rgbEmphasize;
            m_predictions = predictions;
        }

        /// <summary>
        /// Returns the inputs.
        /// </summary>
        public float[] Inputs
        {
            get { return m_rgfInputs; }
        }

        /// <summary>
        /// Returns the mask where a value of 1 indicates that the input at that same location is valid.
        /// </summary>
        public float[] Mask
        {
            get { return m_rgfMask; }
        }

        /// <summary>
        /// Returns the target value.
        /// </summary>
        public float Target
        {
            get { return m_fTarget; }
        }

        /// <summary>
        /// Returns the predicted values.
        /// </summary>
        public List<double> Predicted
        {
            get { return m_rgdfPredicted; }
        }

        /// <summary>
        /// Returns the predicted labels.   
        /// </summary>
        public List<string> PredictedLabels
        {
            get { return m_rgstrPredicted; }
        }

        /// <summary>
        /// Returns the predicted labels to emphasize.
        /// </summary>
        public List<bool> PredictedEmphasize
        {
            get { return m_rgbEmphasize; }
        }

        /// <summary>
        /// Returns the time for the data point.
        /// </summary>
        public float Time
        {
            get { return m_fTime; }
        }

        /// <summary>
        /// Returns the future predictions.
        /// </summary>
        public FuturePredictions Predictions
        {
            get { return m_predictions; }
        }

        /// <summary>
        /// Copies the data point to a new data point.
        /// </summary>
        /// <returns>The new copy is returned.</returns>
        public DataPoint Clone()
        {
            return new DataPoint(m_rgfInputs, m_rgfMask, m_fTarget, Utility.Clone<double>(m_rgdfPredicted), Utility.Clone<string>(m_rgstrPredicted), Utility.Clone<bool>(m_rgbEmphasize), m_fTime, (m_predictions == null) ? null : m_predictions.Clone());
        }
    }

    /// <summary>
    /// The State class defines an abstract base class for the state information and gym data.
    /// </summary>
    public abstract class State
    {
        /// <summary>
        /// Contains the history of the previous target points.
        /// </summary>
        protected List<DataPoint> m_rgPrevPoints = new List<DataPoint>();

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

        /// <summary>
        /// Returns the history of the previous points if provided by the gym.
        /// </summary>
        public List<DataPoint> History
        {
            get { return m_rgPrevPoints; }
        }
    }
}
