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
        /// Add a new dataset to the collection.
        /// </summary>
        /// <param name="ds">Specifies the dataset to add.</param>
        public void Add(DataSet ds)
        {
            m_rgDatasets.Add(ds.Dataset.ID, ds);
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
    }
}
