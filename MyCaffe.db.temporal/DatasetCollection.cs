using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The DatasetCollection manages a set of datasets.
    /// </summary>
    public class DatasetCollection : IDisposable
    {
        Dictionary<int, DataSet> m_rgDatasets = new Dictionary<int, DataSet>();
        Dictionary<string, int> m_rgDatasetIDs = new Dictionary<string, int>();
        Dictionary<int, int> m_rgSourceIDtoDatasetID = new Dictionary<int, int>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public DatasetCollection()
        {
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            CleanUp(0);
        }

        /// <summary>
        /// Release all resources used by the dataset with the specified ID.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset ID to cleanup.</param>
        public void CleanUp(int nDsID)
        {
            if (nDsID == 0)
            {
                foreach (KeyValuePair<int, DataSet> kv in m_rgDatasets)
                {
                    kv.Value.CleanUp();
                }
            }
            else
            {
                if (m_rgDatasets.ContainsKey(nDsID))
                {
                    m_rgDatasets[nDsID].CleanUp();
                    m_rgDatasets.Remove(nDsID);
                }
            }
        }

        /// <summary>
        /// Add a new dataset to the collection if it does not already exist.
        /// </summary>
        /// <param name="dsd">Specifies the dataset descriptor.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <returns>The dataset is returned.</returns>
        public DataSet Add(DatasetDescriptor dsd, Log log)
        {
            DataSet ds = Find(dsd.ID);
            if (ds != null)
                return ds;

            ds = new DataSet(dsd, log);
            Add(ds);

            if (!m_rgSourceIDtoDatasetID.ContainsKey(ds.Dataset.TrainingSource.ID))
                m_rgSourceIDtoDatasetID.Add(ds.Dataset.TrainingSource.ID, ds.Dataset.ID);
            if (!m_rgSourceIDtoDatasetID.ContainsKey(ds.Dataset.TestingSource.ID))
                m_rgSourceIDtoDatasetID.Add(ds.Dataset.TestingSource.ID, ds.Dataset.ID);

            return ds;
        }

        /// <summary>
        /// Add a new dataset to the collection.
        /// </summary>
        /// <param name="ds">Specifies the dataset to add.</param>
        public void Add(DataSet ds)
        {
            m_rgDatasets.Add(ds.Dataset.ID, ds);

            if (!m_rgDatasetIDs.ContainsKey(ds.Dataset.Name))
                m_rgDatasetIDs.Add(ds.Dataset.Name, ds.Dataset.ID);

            if (!m_rgSourceIDtoDatasetID.ContainsKey(ds.Dataset.TrainingSource.ID))
                m_rgSourceIDtoDatasetID.Add(ds.Dataset.TrainingSource.ID, ds.Dataset.ID);
            if (!m_rgSourceIDtoDatasetID.ContainsKey(ds.Dataset.TestingSource.ID))
                m_rgSourceIDtoDatasetID.Add(ds.Dataset.TestingSource.ID, ds.Dataset.ID);
        }

        /// <summary>
        /// Find and return the dataset associated with the dataset ID.
        /// </summary>
        /// <param name="nDatasetID">Specifies the dataset ID of the dataset to find.</param>
        /// <returns>The DataSet is returned if found, or null otherwise.</returns>
        public DataSet Find(int nDatasetID)
        {
            if (!m_rgDatasets.ContainsKey(nDatasetID))
                return null;

            return m_rgDatasets[nDatasetID];
        }

        /// <summary>
        /// Find and return the dataset associated with the dataset name.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <returns>The dataset is returned when found, otherwise null is returned.</returns>
        public DataSet Find(string strDs)
        {
            int nDsID = GetDatasetID(strDs);
            if (nDsID == 0)
                return null;

            return Find(nDsID);
        }

        /// <summary>
        /// Find the temporal set associated with the source ID.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID associated with the temporal set.</param>
        /// <returns>The temporal set is returned.</returns>
        public TemporalSet FindTemporalSetBySourceID(int nSrcID)
        {
            if (!m_rgSourceIDtoDatasetID.ContainsKey(nSrcID))
                return null;

            int nDsID = m_rgSourceIDtoDatasetID[nSrcID];

            DataSet ds = Find(nDsID);
            if (ds == null)
                return null;

            return ds.GetTemporalSetBySourceID(nSrcID);
        }

        /// <summary>
        /// Returns the dataset ID associated with the dataset name.
        /// </summary>
        /// <param name="strDs">Specifies the name of the dataset.</param>
        /// <returns>The dataset ID is returned when found, or 0.</returns>
        public int GetDatasetID(string strDs)
        {
            if (!m_rgDatasetIDs.ContainsKey(strDs))
                return 0;

            return m_rgDatasetIDs[strDs];
        }
    }
}
