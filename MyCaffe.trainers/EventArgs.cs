using MyCaffe.basecode;
using MyCaffe.gym;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
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
        double m_dfTotalReward = 0;
        double m_dfExplorationRate = 0;
        double m_dfOptimalCoeff = 0;
        double m_dfLoss = 0;
        double m_dfLearningRate = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIndex">Specifies the index of the caller.</param>
        /// <param name="nFrames">Specifies the total number of frames across all agents.</param>
        /// <param name="nMaxFrames">Specifies the maximum number of frames across all agents.</param>
        /// <param name="dfR">Specifies the total reward.</param>
        /// <param name="dfExplorationRate">Specifies the current exploration rate.</param>
        /// <param name="dfOptimalCoeff">Specifies the current optimal selection coefficient.</param>
        /// <param name="dfLoss">Specifies the loss.</param>
        /// <param name="dfLearningRate">Specifies the learning rate.</param>
        public GetStatusArgs(int nIndex, int nFrames, int nMaxFrames, double dfR, double dfExplorationRate, double dfOptimalCoeff, double dfLoss, double dfLearningRate)
        {
            m_nIndex = nIndex;
            m_nTotalFrames = nFrames;
            m_nMaxFrames = nMaxFrames;
            m_dfTotalReward = dfR;
            m_dfExplorationRate = dfExplorationRate;
            m_dfOptimalCoeff = dfOptimalCoeff;
            m_dfLoss = dfLoss;
            m_dfLearningRate = dfLearningRate;
        }

        /// <summary>
        /// Returns the index of the caller.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
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
        public double Reward
        {
            get { return m_dfTotalReward; }
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
    /// The GetDataArgs is passed to the OnGetData event.
    /// </summary>
    public class GetDataArgs : EventArgs
    {
        int m_nAction;
        bool m_bReset;
        Component m_caffe;
        Log m_log;
        CancelEvent m_evtCancel;
        StateBase m_state = null;
        int m_nIndex = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the thread.</param>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="bReset">Specifies to reset the environment.</param>
        /// <param name="nAction">Specifies the action to run.  If less than zero this parameter is ignored.</param>
        /// <param name="bAllowUi">Optionally, specifies whether or not to allow the user interface.</param>
        public GetDataArgs(int nIdx, Component mycaffe, Log log, CancelEvent evtCancel, bool bReset, int nAction = -1, bool bAllowUi = true)
        {
            m_nIndex = nIdx;
            m_nAction = nAction;
            m_caffe = mycaffe;
            m_log = log;
            m_evtCancel = evtCancel;
            m_bReset = bReset;
        }


        /// <summary>
        /// Returns the index of the thread asking for the gym.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
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
