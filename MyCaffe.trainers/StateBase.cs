using MyCaffe.basecode;
using MyCaffe.gym;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The StateBase is the base class for the state of each observation - this is defined by actual trainer that overrides the MyCaffeCustomTrainer.
    /// </summary>
    public class StateBase
    {
        bool m_bDone = false;
        bool m_bValid = true;
        double m_dfReward = 0;
        int m_nActionCount = 0;
        SimpleDatum m_data = null;
        SimpleDatum m_clip = null;
        SimpleDatum m_label = null;
        double[] m_rgState = null;
        double[] m_rgClip = null;
        Image m_img = null;
        double m_dfTestingPercent = 0.2; // default to 20% of the tail end of the data for testing.
        List<DataPoint> m_rgHistory = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public StateBase(int nActionCount)
        {
            m_nActionCount = nActionCount;
        }

        /// <summary>
        /// Get/set the percentage of the data to reserve for testing.
        /// </summary>
        /// <remarks>
        /// When set to 0, the same data is used for both testing and training (not recommended).
        /// </remarks>
        public double TestingPercent
        {
            get { return m_dfTestingPercent; }
            set { m_dfTestingPercent = value; }
        }

        /// <summary>
        /// Get/set whether or not the state is valid.
        /// </summary>
        public bool IsValid
        {
            get { return m_bValid; }
            set { m_bValid = value; }
        }

        /// <summary>
        /// Get/set the reward of the state.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }

        /// <summary>
        /// Get/set whether the state is done or not.
        /// </summary>
        public bool Done
        {
            get { return m_bDone; }
            set { m_bDone = value; }
        }

        /// <summary>
        /// Get/set the data history (if any exists).
        /// </summary>
        public List<DataPoint> History
        {
            get { return m_rgHistory; }
            set { m_rgHistory = value; }
        }

        /// <summary>
        /// Returns the number of actions.
        /// </summary>
        public int ActionCount
        {
            get { return m_nActionCount; }           
        }

        /// <summary>
        /// Returns other data associated with the state.
        /// </summary>
        public SimpleDatum Data
        {
            get { return m_data; }
            set { m_data = value; }
        }

        /// <summary>
        /// Get/set the raw state data.
        /// </summary>
        public double[] RawState
        {
            get { return m_rgState; }
            set { m_rgState = value; }
        }

        /// <summary>
        /// Returns the clip data assoicated with the state. 
        /// </summary>
        public SimpleDatum Clip
        {
            get { return m_clip; }
            set { m_clip = value; }
        }

        /// <summary>
        /// Get/set the raw clip data.
        /// </summary>
        public double[] RawClip
        {
            get { return m_rgClip; }
            set { m_rgClip = value; }
        }

        /// <summary>
        /// Get/set the label data associated with the state.  This field is optional.
        /// </summary>
        public SimpleDatum Label
        {
            get { return m_label; }
            set { m_label = value; }
        }

        /// <summary>
        /// Get/set the image (if any exists)
        /// </summary>
        public Image RawImage
        {
            get { return m_img; }
            set { m_img = value; }
        }

        /// <summary>
        /// Copy the state base information.
        /// </summary>
        /// <returns>A new state base is returned.</returns>
        public virtual StateBase Clone()
        {
            StateBase s = new StateBase(m_nActionCount);

            s.m_bDone = m_bDone;
            s.m_bValid = m_bValid;
            s.m_dfReward = m_dfReward;
            s.m_data = new SimpleDatum(m_data, true);
            s.m_rgState = Utility.Clone<double>(m_rgState);

            if (m_clip != null)
                s.m_clip = new SimpleDatum(m_clip, true);

            s.m_img = (m_img == null) ? null : new Bitmap(m_img);

            return s;
        }

        /// <summary>
        /// Return the string representation of the state.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            double[] rgData = m_data.GetData<double>();

            string str = "{";
            for (int i = 0; i < rgData.Length && i < 5; i++)
            {
                str += rgData[i].ToString("N4") + ",";
            }
            str = str.TrimEnd(',');
            str += "}";

            return "State = " + str + " Done = " + m_bDone.ToString();
        }
    }
}
