using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The DataSet class loads the training and testing data.
    /// </summary>
    public class DataSet
    {
        Log m_log;
        CryptoRandom m_random = new CryptoRandom();
        DatabaseTemporal m_db = new DatabaseTemporal();
        DatasetDescriptor m_ds;
        Dictionary<int, TemporalSet> m_rgTemporalSets = new Dictionary<int, TemporalSet>();
        Dictionary<Phase, TemporalSet> m_rgTemporalSetsEx = new Dictionary<Phase, TemporalSet>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="ds">Specifies the dataset descriptor.</param>
        /// <param name="log">Specifies the output log.</param>
        public DataSet(DatasetDescriptor ds, Log log)
        {
            m_ds = ds;
            m_log = log;
        }

        /// <summary>
        /// Remove all data for the dataset from memory.
        /// </summary>
        public void CleanUp()
        {
            foreach (KeyValuePair<int, TemporalSet> kv in m_rgTemporalSets)
            {
                if (kv.Value != null)
                    kv.Value.CleanUp();
            }

            m_rgTemporalSets.Clear();
            m_rgTemporalSetsEx.Clear();
        }

        /// <summary>
        /// Return the dataset descriptor.
        /// </summary>
        public DatasetDescriptor Dataset
        {
            get { return m_ds; }
        }

        /// <summary>
        /// Returns the temporal set for the specified phase.
        /// </summary>
        /// <param name="phase">Specifies the phase.</param>
        /// <returns>The temporal set associated with the phase is returned.</returns>
        public TemporalSet GetTemporalSetByPhase(Phase phase)
        {
            if (m_rgTemporalSetsEx.ContainsKey(phase))
                return m_rgTemporalSetsEx[phase];

            return null;
        }

        /// <summary>
        /// Returns the temporal set for the specified source ID.
        /// </summary>
        /// <param name="nSourceID">Specifies the source ID associated with the temporal set.</param>
        /// <returns>The temporal set associated with the source ID is returned.</returns>
        public TemporalSet GetTemporalSetBySourceID(int nSourceID)
        {
            if (m_rgTemporalSets.ContainsKey(nSourceID))
                return m_rgTemporalSets[nSourceID];

            return null;
        }

        /// <summary>
        /// Return the load percentage for the dataset.
        /// </summary>
        /// <param name="dfTraining">Specifies the training percent loaded.</param>
        /// <param name="dfTesting">Specifies the testing percent loaded.</param>
        /// <returns>Returns the total load percent.</returns>
        public double GetLoadPercent(out double dfTraining, out double dfTesting)
        {            
            dfTraining = m_rgTemporalSetsEx[Phase.TRAIN].LoadPercent;
            dfTesting = m_rgTemporalSetsEx[Phase.TEST].LoadPercent;

            return (dfTraining + dfTesting) / 2.0;
        }

        /// <summary>
        /// Load the training and testing data.
        /// </summary>
        /// <param name="loadMethod">Specifies the loading method.</param>
        /// <param name="nLoadLimit">Specifies the load limit (or 0 to ignore)</param>
        /// <param name="bNormalizedData">Specifies to load the normalized data.</param>
        /// <param name="nHistoricalSteps">Specifies the number of sequential historical steps in each block.</param>
        /// <param name="nFutureSteps">Specifies the number of sequential future steps in each block.</param>
        /// <param name="nChunks">Specifies the number of blocks to load at a time.</param>
        /// <param name="evtCancel">Specifies the event used to cancel the initialization process.</param>
        public bool Load(DB_LOAD_METHOD loadMethod, int nLoadLimit, bool bNormalizedData, int nHistoricalSteps, int nFutureSteps, int nChunks, EventWaitHandle evtCancel)
        {
            TemporalSet ts = new TemporalSet(m_log, m_db, m_ds.TrainingSource, loadMethod, nLoadLimit, m_random, nHistoricalSteps, nFutureSteps, nChunks);
            m_rgTemporalSets.Add(m_ds.TrainingSource.ID, ts);
            m_rgTemporalSetsEx.Add(Phase.TRAIN, ts);

            ts = new TemporalSet(m_log, m_db, m_ds.TestingSource, loadMethod, nLoadLimit, m_random, nHistoricalSteps, nFutureSteps, nChunks);
            m_rgTemporalSets.Add(m_ds.TestingSource.ID, ts);
            m_rgTemporalSetsEx.Add(Phase.TEST, ts);

            foreach (KeyValuePair<int, TemporalSet> kv in m_rgTemporalSets)
            {
                if (!kv.Value.Initialize(bNormalizedData, evtCancel))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Wait for the loading to complete.
        /// </summary>
        /// <param name="evtCancel">Specifies an auto reset event used to abort waiting.</param>
        /// <returns>True is returned once loaded, otherwise false is returned when aborting.</returns>
        public bool WaitForLoadingToComplete(AutoResetEvent evtCancel)
        {
            foreach (KeyValuePair<int, TemporalSet> kv in m_rgTemporalSets)
            {
                if (!kv.Value.WaitForLoadingToComplete(evtCancel))
                    return false;
            }

            return true;
        }
    }
}
