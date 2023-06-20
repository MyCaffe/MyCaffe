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
        TemporalSet m_tsTrain;
        TemporalSet m_tsTest;

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
            if (m_tsTrain != null)
            {
                m_tsTrain.CleanUp();
                m_tsTrain = null;
            }

            if (m_tsTest != null)
            {
                m_tsTest.CleanUp();
                m_tsTest = null;
            }
        }

        /// <summary>
        /// Return the dataset descriptor.
        /// </summary>
        public DatasetDescriptor Dataset
        {
            get { return m_ds; }
        }

        /// <summary>
        /// Load the training and testing data.
        /// </summary>
        /// <param name="loadMethod">Specifies the loading method.</param>
        /// <param name="nLoadLimit">Specifies the load limit (or 0 to ignore)</param>
        /// <param name="bNormalizeData">Specifies to load the normalized data.</param>
        /// <param name="nHistoricalSteps">Specifies the number of sequential historical steps in each block.</param>
        /// <param name="nFutureSteps">Specifies the number of sequential future steps in each block.</param>
        /// <param name="nChunks">Specifies the number of blocks to load at a time.</param>
        /// <param name="evtCancel">Specifies the event used to cancel the initialization process.</param>
        public bool Load(DB_LOAD_METHOD loadMethod, int nLoadLimit, bool bNormalizeData, int nHistoricalSteps, int nFutureSteps, int nChunks, AutoResetEvent evtCancel)
        {
            m_tsTrain = new TemporalSet(m_log, m_db, m_ds.TrainingSource, loadMethod, nLoadLimit, m_random, nHistoricalSteps, nFutureSteps, nChunks);
            m_tsTest = new TemporalSet(m_log, m_db, m_ds.TestingSource, loadMethod, nLoadLimit, m_random, nHistoricalSteps, nFutureSteps, nChunks);

            if (!m_tsTrain.Initialize(bNormalizeData, evtCancel))
                return false;

            if (!m_tsTest.Initialize(bNormalizeData, evtCancel))
                return false;

            return true;
        }

        /// <summary>
        /// Wait for the loading to complete.
        /// </summary>
        /// <param name="evtCancel">Specifies an auto reset event used to abort waiting.</param>
        /// <returns>True is returned once loaded, otherwise false is returned when aborting.</returns>
        public bool WaitForLoadingToComplete(AutoResetEvent evtCancel)
        {
            if (!m_tsTrain.WaitForLoadingToComplete(evtCancel))
                return false;

            if (!m_tsTest.WaitForLoadingToComplete(evtCancel))
                return false;

            return true;
        }
    }
}
