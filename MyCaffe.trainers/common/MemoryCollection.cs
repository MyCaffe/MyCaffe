using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// The memory collection stores a set of memory items.
    /// </summary>
    public class MemoryCollection
    {
        int m_nCount = 0;
        int[] m_rgIdx = null;
        double[] m_rgfPriorities = null;
        /// <summary>
        /// Specifies the memory item list.
        /// </summary>
        protected MemoryItem[] m_rgItems;
        /// <summary>
        /// Specifies the next available index in the rolling list.
        /// </summary>
        protected int m_nNextIdx = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items to store.</param>
        public MemoryCollection(int nMax)
        {
            m_rgItems = new MemoryItem[nMax];
        }

        /// <summary>
        /// Get/set the indexes associated with the collection (if any).
        /// </summary>
        public int[] Indexes
        {
            get { return m_rgIdx; }
            set { m_rgIdx = value; }
        }

        /// <summary>
        /// Get/set the priorities associated with the collection (if any).
        /// </summary>
        public double[] Priorities
        {
            get { return m_rgfPriorities; }
            set { m_rgfPriorities = value; }
        }

        /// <summary>
        /// Returns the next index.
        /// </summary>
        public int NextIndex
        {
            get { return m_nNextIdx; }
        }

        /// <summary>
        /// Returns the current count of items.
        /// </summary>
        public int Count
        {
            get { return m_nCount; }
        }

        /// <summary>
        /// Get/set the memory item at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to access.</param>
        /// <returns>Returns the memory item at an index.</returns>
        public MemoryItem this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
        }

        /// <summary>
        /// Adds a new memory item to the array of items and if at capacity, removes an item.
        /// </summary>
        /// <param name="item">Specifies the memory item to add.</param>
        public virtual void Add(MemoryItem item)
        {
            m_rgItems[m_nNextIdx] = item;
            m_nNextIdx++;

            if (m_nNextIdx == m_rgItems.Length)
                m_nNextIdx = 0;

            if (m_nCount < m_rgItems.Length)
                m_nCount++;
        }

        /// <summary>
        /// Retrieves a random sample of items from the list.
        /// </summary>
        /// <param name="random">Specifies the random number generator to use.</param>
        /// <param name="nCount">Specifies the number of items to retrieve.</param>
        /// <returns>The sampled items are returned in a new MemoryCollection.</returns>
        public MemoryCollection GetRandomSamples(CryptoRandom random, int nCount)
        {
            if (nCount >= Count)
                return this;

            MemoryCollection col = new MemoryCollection(nCount);

            while (col.Count < nCount)
            {
                int nIdx = random.Next(Count);
                col.Add(m_rgItems[nIdx]);
            }

            return col;
        }

        /// <summary>
        /// Returns the list of Next State items.
        /// </summary>
        /// <returns>The state items are returned.</returns>
        public List<StateBase> GetNextState()
        {
            return m_rgItems.Select(p => p.NextState).ToList();
        }

        /// <summary>
        /// Returns the list of data items associated with the next state.
        /// </summary>
        /// <returns>The data items are returned.</returns>
        public List<SimpleDatum> GetNextStateData()
        {
            return m_rgItems.Select(p => p.NextData).ToList();
        }

        /// <summary>
        /// Returns the list of clip items associated with the next state.
        /// </summary>
        /// <returns>The data items are returned.</returns>
        public List<SimpleDatum> GetNextStateClip()
        {
            if (m_rgItems[0].NextState != null && m_rgItems[0].NextState.Clip != null)
                return m_rgItems.Select(p => p.NextState.Clip).ToList();

            return null;
        }

        /// <summary>
        /// Returns the list of data items associated with the current state.
        /// </summary>
        /// <returns>The data items are returned.</returns>
        public List<SimpleDatum> GetCurrentStateData()
        {
            return m_rgItems.Select(p => p.CurrentData).ToList();
        }

        /// <summary>
        /// Returns the list of clip items associated with the current state.
        /// </summary>
        /// <returns>The data items are returned.</returns>
        public List<SimpleDatum> GetCurrentStateClip()
        {
            if (m_rgItems[0].CurrentState != null && m_rgItems[0].CurrentState.Clip != null)
                return m_rgItems.Select(p => p.CurrentState.Clip).ToList();

            return null;
        }

        /// <summary>
        /// Returns the action items as a set of one-hot vectors.
        /// </summary>
        /// <param name="nActionCount">Specifies the action count.</param>
        /// <returns>The one-hot vectors are returned as an array.</returns>
        public float[] GetActionsAsOneHotVector(int nActionCount)
        {
            float[] rg = new float[m_rgItems.Length * nActionCount];

            for (int i = 0; i < m_rgItems.Length; i++)
            {
                int nAction = m_rgItems[i].Action;

                for (int j = 0; j < nActionCount; j++)
                {
                    rg[(i * nActionCount) + j] = (j == nAction) ? 1 : 0;
                }
            }

            return rg;
        }

        /// <summary>
        /// Returns the inverted done (1 - done) values as a one-hot vector.
        /// </summary>
        /// <returns>The one-hot vectors are returned as an array.</returns>
        public float[] GetInvertedDoneAsOneHotVector()
        {
            float[] rgDoneInv = new float[m_rgItems.Length];

            for (int i = 0; i < m_rgItems.Length; i++)
            {
                if (m_rgItems[i].IsTerminated)
                    rgDoneInv[i] = 0;
                else
                    rgDoneInv[i] = 1;
            }

            return rgDoneInv;
        }

        /// <summary>
        /// Returns the rewards as a vector.
        /// </summary>
        /// <returns>The rewards are returned as an array.</returns>
        public float[] GetRewards()
        {
            return m_rgItems.Select(p => (float)p.Reward).ToArray();
        }

        /// <summary>
        /// Save the memory items to file.
        /// </summary>
        /// <param name="strFile">Specifies the file name.</param>
        public void Save(string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (StreamWriter sw = new StreamWriter(strFile))
            {
                for (int i = 0; i < Count; i++)
                {
                    MemoryItem mi = m_rgItems[i];
                    string strLine = mi.CurrentData.ToArrayAsString(4) + "," + mi.Action.ToString() + "," + mi.Reward.ToString() + "," + (mi.IsTerminated ? 1 : 0).ToString() + "," + mi.NextData.ToArrayAsString(4);
                    sw.WriteLine(strLine);
                }
            }
        }

        /// <summary>
        /// Load all memory items from file.
        /// </summary>
        /// <param name="strFile">Specifies the file containing the memory items.</param>
        public void Load(string strFile)
        {
            m_nNextIdx = 0;
            m_nCount = 0;

            List<MemoryItem> rg = new List<MemoryItem>();

            using (StreamReader sr = new StreamReader(strFile))
            {
                string strLine = sr.ReadLine();

                while (strLine != null)
                {
                    string[] rgstr = strLine.Split(',');
                    int nIdx = 0;

                    List<double> rgdfData = new List<double>();
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    SimpleDatum sdCurrent = new SimpleDatum(true, 4, 1, 1, -1, DateTime.MinValue, null, rgdfData, 0, false, -1);

                    int nAction = int.Parse(rgstr[nIdx]); nIdx++;
                    double dfReward = double.Parse(rgstr[nIdx]); nIdx++;
                    bool bTerminated = (int.Parse(rgstr[nIdx]) == 1) ? true : false; nIdx++;

                    rgdfData = new List<double>();
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    rgdfData.Add(double.Parse(rgstr[nIdx])); nIdx++;
                    SimpleDatum sdNext = new SimpleDatum(true, 4, 1, 1, -1, DateTime.MinValue, null, rgdfData, 0, false, -1);

                    rg.Add(new MemoryItem(null, sdCurrent, nAction, null, sdNext, dfReward, bTerminated, 0, 0));
                    strLine = sr.ReadLine();
                }
            }

            foreach (MemoryItem m in rg)
            {
                Add(m);
            }
        }
    }

    /// <summary>
    /// The MemoryItem stores the information about a given cycle.
    /// </summary>
    public class MemoryItem 
    {
        StateBase m_state0;
        StateBase m_state1;
        SimpleDatum m_x0;
        SimpleDatum m_x1;
        int m_nAction;
        int m_nIteration;
        int m_nEpisode;
        bool m_bTerminated;
        double m_dfReward;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="currentState">Specifies the current state.</param>
        /// <param name="currentData">Specifies the current data.</param>
        /// <param name="nAction">Specifies the action.</param>
        /// <param name="nextState">Specifies the next state.</param>
        /// <param name="nextData">Specifies the next data.</param>
        /// <param name="dfReward">Specifies the reward.</param>
        /// <param name="bTerminated">Specifies whether or not this is a termination state or not.</param>
        /// <param name="nIteration">Specifies the iteration.</param>
        /// <param name="nEpisode">Specifies the episode.</param>
        public MemoryItem(StateBase currentState, SimpleDatum currentData, int nAction, StateBase nextState, SimpleDatum nextData, double dfReward, bool bTerminated, int nIteration, int nEpisode)
        {
            m_state0 = currentState;
            m_state1 = nextState;
            m_x0 = currentData;
            m_x1 = nextData;
            m_nAction = nAction;
            m_bTerminated = bTerminated;
            m_dfReward = dfReward;
            m_nIteration = nIteration;
            m_nEpisode = nEpisode;
        }

        /// <summary>
        /// Returns the termination status of the next state.
        /// </summary>
        public bool IsTerminated
        {
            get { return m_bTerminated; }
        }

        /// <summary>
        /// Returns the reward of the state transition.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }

        /// <summary>
        /// Returns the current state.
        /// </summary>
        public StateBase CurrentState
        {
            get { return m_state0; }
        }

        /// <summary>
        /// Returns the next state.
        /// </summary>
        public StateBase NextState
        {
            get { return m_state1; }
        }

        /// <summary>
        /// Returns the data associated with the current state.
        /// </summary>
        public SimpleDatum CurrentData
        {
            get { return m_x0; }
        }

        /// <summary>
        /// Returns the data associated with the next state.
        /// </summary>
        public SimpleDatum NextData
        {
            get { return m_x1; }
        }

        /// <summary>
        /// Returns the action.
        /// </summary>
        public int Action
        {
            get { return m_nAction; }
        }

        /// <summary>
        /// Returns the iteration of the state transition.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Returns the episode of the state transition.
        /// </summary>
        public int Episode
        {
            get { return m_nEpisode; }
        }

        /// <summary>
        /// Returns a string representation of the state transition.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "episode = " + m_nEpisode.ToString() + " action = " + m_nAction.ToString() + " reward = " + m_dfReward.ToString("N2");
        }

        private string tostring(float[] rg)
        {
            string str = "{";

            for (int i = 0; i < rg.Length; i++)
            {
                str += rg[i].ToString("N5");
                str += ",";
            }

            str = str.TrimEnd(',');
            str += "}";

            return str;
        }
    }
}
