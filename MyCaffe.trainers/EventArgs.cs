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
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
            else
            {
                MyCaffeControl<float> mycaffe1 = mycaffe as MyCaffeControl<float>;
                m_nOriginalDsId = mycaffe1.CurrentProject.Dataset.ID;
            }
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
        int m_nTotalFrames = 0;
        int m_nMaxFrames = 0;
        double m_dfTotalReward = 0;
        double m_dfExplorationRate = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nFrames">Specifies the total number of frames across all agents.</param>
        /// <param name="nMaxFrames">Specifies the maximum number of frames across all agents.</param>
        /// <param name="dfR">Specifies the total reward.</param>
        /// <param name="dfExplorationRate">Specifies the current exploration rate.</param>
        public GetStatusArgs(int nFrames, int nMaxFrames, double dfR, double dfExplorationRate)
        {
            m_nTotalFrames = nFrames;
            m_nMaxFrames = nMaxFrames;
            m_dfTotalReward = dfR;
            m_dfExplorationRate = dfExplorationRate;
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
    }

    /// <summary>
    /// The GetDataArgs is passed to the OnGetData event.
    /// </summary>
    public class GetDataArgs : EventArgs
    {
        int m_nIndex = -1;
        int m_nAction;
        bool m_bReset;
        bool m_bAllowUI = true;
        Component m_caffe;
        Log m_log;
        CancelEvent m_evtCancel;
        StateBase m_state = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="mycaffe">Specifies the MyCaffeControl used.</param>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="bReset">Specifies to reset the environment.</param>
        /// <param name="nIndex">Specifies the instance index.</param>
        /// <param name="nAction">Specifies the action to run.  If less than zero this parameter is ignored.</param>
        /// <param name="bAllowUi">Optionally, specifies whether or not to allow the user interface.</param>
        public GetDataArgs(Component mycaffe, Log log, CancelEvent evtCancel, bool bReset, int nIndex, int nAction = -1, bool bAllowUi = true)
        {
            m_nIndex = nIndex;
            m_nAction = nAction;
            m_caffe = mycaffe;
            m_log = log;
            m_evtCancel = evtCancel;
            m_bReset = bReset;
            m_bAllowUI = bAllowUi;
        }

        /// <summary>
        /// Returns whether or not to allow the user interface.
        /// </summary>
        public bool AllowUi
        {
            get { return m_bAllowUI; }
        }

        /// <summary>
        /// Returns the index of the trainer that fires the event.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
            set { m_nIndex = value; }
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
