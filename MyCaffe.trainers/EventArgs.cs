using MyCaffe.basecode;
using MyCaffe.gym;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The ApplyUpdateArgs is passed to the OnApplyUpdates event.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ApplyUpdateArgs<T> : EventArgs
    {
        MyCaffeControl<T> m_mycaffeWorker;
        int m_nIteration;
        double m_dfLearningRate;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIteration">Specifies the iteration from which the gradients are to be applied.</param>
        /// <param name="mycaffeWorker">Specifies the MyCaffe worker instance whos gradients are to be applied.</param>
        public ApplyUpdateArgs(int nIteration, MyCaffeControl<T> mycaffeWorker)
        {
            m_nIteration = nIteration;
            m_mycaffeWorker = mycaffeWorker;
        }

        /// <summary>
        /// Returns the MyCaffe worker instance whos gradients are to be applied.
        /// </summary>
        public MyCaffeControl<T> MyCaffeWorker
        {
            get { return m_mycaffeWorker; }
        }

        /// <summary>
        /// Returns the iteration from which the gradients are to be applied.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Returns the learning rate at the time the gradients were applied.
        /// </summary>
        public double LearningRate
        {
            get { return m_dfLearningRate; }
            set { m_dfLearningRate = value; }
        }
    }

    /// <summary>
    /// The WaitArgs is passed to the OnWait event.
    /// </summary>
    public class WaitArgs : EventArgs
    {
        int m_nWait;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nWait">The amount of time to wait in milliseconds.</param>
        public WaitArgs(int nWait)
        {
            m_nWait = nWait;
        }

        /// <summary>
        /// Returns the amount of time to wait in milliseconds.
        /// </summary>
        public int Wait
        {
            get { return m_nWait; }
        }
    }

    /// <summary>
    /// The InitializeArgs is passed to the OnInitialize event.
    /// </summary>
    public class InitializeArgs : EventArgs
    {
        int m_nOriginalDsId = 0;
        int m_nDsID = 0;
        Component m_caffe;
        Log m_log;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        public InitializeArgs(Component mycaffe)
        {
            m_caffe = mycaffe;

            if (mycaffe is MyCaffeControl<double>)
            {
                MyCaffeControl<double> mycaffe1 = mycaffe as MyCaffeControl<double>;
                m_log = mycaffe1.Log;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
            else
            {
                MyCaffeControl<float> mycaffe1 = mycaffe as MyCaffeControl<float>;
                m_log = mycaffe1.Log;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
        }

        /// <summary>
        /// Returns the output log.
        /// </summary>
        public Log OutputLog
        {
            get { return m_log; }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }

        /// <summary>
        /// Returns the original Dataset ID of the open project held by the MyCaffeControl.
        /// </summary>
        public int OriginalDatasetID
        {
            get { return m_nOriginalDsId; }
        }

        /// <summary>
        /// Get/set a new Dataset ID which is actually used. 
        /// </summary>
        public int DatasetID
        {
            get { return m_nDsID; }
            set { m_nDsID = value; }
        }
    }

    /// <summary>
    /// The GetStatusArgs is passed to the OnGetStatus event.
    /// </summary>
    public class GetStatusArgs : EventArgs
    {
        int m_nIndex = 0;
        int m_nNewFrameCount = 0;
        int m_nTotalFrames = 0;
        int m_nMaxFrames = 0;
        int m_nIteration = 0;
        double m_dfTotalReward = 0;
        double m_dfReward = 0;
        double m_dfExplorationRate = 0;
        double m_dfOptimalCoeff = 0;
        double m_dfLoss = 0;
        double m_dfLearningRate = 0;
        bool m_bModelUpdated = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIndex">Specifies the index of the caller.</param>
        /// <param name="nFrames">Specifies the total number of frames across all agents.</param>
        /// <param name="nMaxFrames">Specifies the maximum number of frames across all agents.</param>
        /// <param name="nIteration">Specifies the number of iterations run.</param>
        /// <param name="dfTotalReward">Specifies the total reward.</param>
        /// <param name="dfReward">Specifies the immediate reward for the current episode.</param>
        /// <param name="dfExplorationRate">Specifies the current exploration rate.</param>
        /// <param name="dfOptimalCoeff">Specifies the current optimal selection coefficient.</param>
        /// <param name="dfLoss">Specifies the loss.</param>
        /// <param name="dfLearningRate">Specifies the learning rate.</param>
        /// <param name="bModelUpdated">Specifies whether or not the model has been updated.</param>
        public GetStatusArgs(int nIndex, int nIteration, int nFrames, int nMaxFrames, double dfTotalReward, double dfReward, double dfExplorationRate, double dfOptimalCoeff, double dfLoss, double dfLearningRate, bool bModelUpdated = false)
        {
            m_nIndex = nIndex;
            m_nIteration = nIteration;
            m_nTotalFrames = nFrames;
            m_nMaxFrames = nMaxFrames;
            m_dfTotalReward = dfTotalReward;
            m_dfReward = dfReward;
            m_dfExplorationRate = dfExplorationRate;
            m_dfOptimalCoeff = dfOptimalCoeff;
            m_dfLoss = dfLoss;
            m_dfLearningRate = dfLearningRate;
            m_bModelUpdated = bModelUpdated;
        }

        /// <summary>
        /// Returns the index of the caller.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
        }

        /// <summary>
        /// Returns the number of iterations (steps) run.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Get/set the new frame count.
        /// </summary>
        public int NewFrameCount
        {
            get { return m_nNewFrameCount; }
            set { m_nNewFrameCount = value; }
        }

        /// <summary>
        /// Returns the total frame count across all agents.
        /// </summary>
        public int Frames
        {
            get { return m_nTotalFrames; }
        }

        /// <summary>
        /// Returns the maximum frame count.
        /// </summary>
        public int MaxFrames
        {
            get { return m_nMaxFrames; }
        }

        /// <summary>
        /// Returns whether or not the model has been updated or not.
        /// </summary>
        public bool ModelUpdated
        {
            get { return m_bModelUpdated; }
        }

        /// <summary>
        /// Returns the loss value.
        /// </summary>
        public double Loss
        {
            get { return m_dfLoss; }
        }

        /// <summary>
        /// Returns the current learning rate.
        /// </summary>
        public double LearningRate
        {
            get { return m_dfLearningRate; }
        }

        /// <summary>
        /// Returns the total rewards.
        /// </summary>
        public double TotalReward
        {
            get { return m_dfTotalReward; }
        }

        /// <summary>
        /// Returns the immediate reward for the current episode.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
        }

        /// <summary>
        /// Returns the current exploration rate.
        /// </summary>
        public double ExplorationRate
        {
            get { return m_dfExplorationRate; }
        }

        /// <summary>
        /// Returns the optimal selection coefficient.
        /// </summary>
        public double OptimalSelectionCoefficient
        {
            get { return m_dfOptimalCoeff; }
        }
    }

    /// <summary>
    /// The ConvertOutputArgs is passed to the OnConvertOutput event.
    /// </summary>
    public class ConvertOutputArgs : EventArgs
    {
        float[] m_rgOutput;
        string m_type;
        int m_nN;
        byte[] m_rgRawOutput;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nN">Specifies the number of outputs.</param>
        /// <param name="rgOutput">Specifies the output to convert.</param>
        public ConvertOutputArgs(int nN, float[] rgOutput)
        {
            m_nN = nN;
            m_rgOutput = rgOutput;
        }

        /// <summary>
        /// Returns the number of results.
        /// </summary>
        public int ResultCount
        {
            get { return m_nN; }
        }

        /// <summary>
        /// Specifies the output to convert.
        /// </summary>
        public float[] Output
        {
            get { return m_rgOutput; }
        }

        /// <summary>
        /// Specifies the type of the raw output byte stream.
        /// </summary>
        public string RawType
        {
            get { return m_type; }
        }

        /// <summary>
        /// Specifies the raw output byte stream.
        /// </summary>
        public byte[] RawOutput
        {
            get { return m_rgRawOutput; }
        }

        /// <summary>
        /// Sets the raw output byte stream and type.
        /// </summary>
        /// <param name="rgData">Specifies the raw output byte stream.</param>
        /// <param name="type">Specifies the raw output type.</param>
        public void SetRawOutput(byte[] rgData, string type)
        {
            m_rgRawOutput = rgData;
            m_type = type;
        }
    }

    /// <summary>
    /// The OverlayArgs is passed ot the OnOverlay event, optionally fired just before displaying a gym image.
    /// </summary>
    public class OverlayArgs : EventArgs
    {
        Bitmap m_bmp;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bmp">Specifies the display image.</param>
        public OverlayArgs(Bitmap bmp)
        {
            m_bmp = bmp;
        }

        /// <summary>
        /// Get/set the display image.
        /// </summary>
        public Bitmap DisplayImage
        {
            get { return m_bmp; }
            set { m_bmp = value; }
        }
    }

    /// <summary>
    /// The GetDataArgs is passed to the OnGetData event to retrieve data.
    /// </summary>
    public class GetDataArgs : EventArgs
    {
        int m_nAction;
        bool m_bReset;
        Component m_caffe;
        Log m_log;
        ManualResetEvent m_evtDataReady = null;
        CancelEvent m_evtCancel;
        StateBase m_state = null;
        int m_nIndex = 0;
        bool m_bGetLabel = false;
        Phase m_phase = Phase.NONE;
        IxTrainerGetDataCallback m_iOnGetData = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="phase">Specifies the phase under which to get the data.</param>
        /// <param name="nIdx">Specifies the index of the thread.</param>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="bReset">Specifies to reset the environment.</param>
        /// <param name="nAction">Specifies the action to run.  If less than zero this parameter is ignored.</param>
        /// <param name="bAllowUi">Optionally, specifies whether or not to allow the user interface.</param>
        /// <param name="bGetLabel">Optionally, specifies to get the label in addition to the data.</param>
        /// <param name="bBatchMode">Optionally, specifies to get the data in batch mode (default = false).</param>
        /// <param name="iOnGetData">Optionally, specifies the callback called after rendering the gym output, yet just before displaying it.</param>
        public GetDataArgs(Phase phase, int nIdx, Component mycaffe, Log log, CancelEvent evtCancel, bool bReset, int nAction = -1, bool bAllowUi = true, bool bGetLabel = false, bool bBatchMode = false, IxTrainerGetDataCallback iOnGetData = null)
        {
            if (bBatchMode)
                m_evtDataReady = new ManualResetEvent(false);

            m_phase = phase;
            m_nIndex = nIdx;
            m_nAction = nAction;
            m_caffe = mycaffe;
            m_log = log;
            m_evtCancel = evtCancel;
            m_bReset = bReset;
            m_bGetLabel = bGetLabel;
            m_iOnGetData = iOnGetData;
        }

        /// <summary>
        /// Returns the OnGetData Callback called just after rendering yet before displaying the gym image.
        /// </summary>
        public IxTrainerGetDataCallback GetDataCallback
        {
            get { return m_iOnGetData; }
        }

        /// <summary>
        /// Returns the active phase under which to get the data.
        /// </summary>
        public Phase ActivePhase
        {
            get { return m_phase; }
        }

        /// <summary>
        /// Returns the data ready event that is set once the data has been retrieved.  This field is only
        /// used when using the OnGetDataAsync event.
        /// </summary>
        public ManualResetEvent DataReady
        {
            get { return m_evtDataReady; }
        }

        /// <summary>
        /// Returns the index of the thread asking for the gym.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
        }

        /// <summary>
        /// Returns whether or not to retrieve the label in addition to the data.
        /// </summary>
        public bool GetLabel
        {
            get { return m_bGetLabel; }
        }

        /// <summary>
        /// Returns the output log for general output.
        /// </summary>
        public Log OutputLog
        {
            get { return m_log; }
        }

        /// <summary>
        /// Returns the cancel event.
        /// </summary>
        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        /// <summary>
        /// Specifies the state data of the observations.
        /// </summary>
        public StateBase State
        {
            get { return m_state; }
            set { m_state = value; }
        }

        /// <summary>
        /// Returns the action to run.  If less than zero, this parameter is ignored.
        /// </summary>
        public int Action
        {
            get { return m_nAction; }
        }

        /// <summary>
        /// Returns the MyCaffeControl used.
        /// </summary>
        public Component MyCaffe
        {
            get { return m_caffe; }
        }

        /// <summary>
        /// Returns whether or not to reset the observation environment or not.
        /// </summary>
        public bool Reset
        {
            get { return m_bReset; }
        }
    }
}
