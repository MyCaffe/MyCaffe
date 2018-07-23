using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.imagedb
{
    /// <summary>
    /// The DatasetExCollection contains a list of DatasetEx objects.
    /// </summary>
    public class DatasetExCollection : IEnumerable<DatasetEx>, IDisposable
    {
        List<DatasetEx> m_rgDatasets = new List<DatasetEx>();
        bool m_bUseTrainingSourcesForTesting = false;
        ImageSet m_lastImgSet = null;
        object m_syncObj = new object();

        /// <summary>
        /// The DatasetExCollection constructor.
        /// </summary>
        public DatasetExCollection()
        {
        }

        /// <summary>
        /// Remove the dataset specified.
        /// </summary>
        /// <param name="ds">Specifies the dataset to remove.</param>
        /// <returns>If the dataset is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool RemoveDataset(DatasetEx ds)
        {
            return m_rgDatasets.Remove(ds);
        }

        /// <summary>
        /// Remove all dynamically created datasets.
        /// </summary>
        public void RemoveCreatedDatasets()
        {
            List<int> rgIdx = new List<int>();

            for (int i = 0; i < m_rgDatasets.Count; i++)
            {
                if (m_rgDatasets[i].Descriptor.ID < 0)
                    rgIdx.Add(i);
            }

            for (int i = rgIdx.Count - 1; i >= 0; i--)
            {
                m_rgDatasets.RemoveAt(rgIdx[i]);
            }
        }

        /// <summary>
        /// Removes a user from the list of users using the DatasetExCollection.
        /// </summary>
        /// <param name="user">Specifies the unique user ID.</param>
        /// <returns>Returns <i>true</i> after all users are released from all datasets, <i>false</i> otherwise.</returns>
        public bool RemoveUser(Guid user)
        {
            bool bReleased = true;

            foreach (DatasetEx ds in m_rgDatasets)
            {
                if (ds.RemoveUser(user) > 0)
                    bReleased = false;
            }

            return bReleased;
        }

        /// <summary>
        /// Saves the image mean in a SimpleDatum to the database for a data source.
        /// </summary>
        /// <param name="nSrcID">Specifies the ID of the data source.</param>
        /// <param name="sd">Specifies the image mean data.</param>
        /// <param name="bUpdate">Specifies whether or not to update the mean image.</param>
        /// <returns>Returns <i>true</i> after a successful save, <i>false</i> otherwise.</returns>
        public bool SaveImageMean(int nSrcID, SimpleDatum sd, bool bUpdate)
        {
            foreach (DatasetEx ds in m_rgDatasets)
            {
                if (ds.SaveImageMean(nSrcID, sd, bUpdate))
                    return true;
            }

            return false;
        }

        /// <summary>
        /// Returns the image mean for a data source.
        /// </summary>
        /// <param name="nSrcID">Specifies the ID of the data source.</param>
        /// <returns>The image mean queried is returned as a SimpleDatum.</returns>
        public SimpleDatum QueryImageMean(int nSrcID)
        {
            foreach (DatasetEx ds in m_rgDatasets)
            {
                SimpleDatum sd = ds.QueryImageMean(nSrcID);
                if (sd != null)
                    return sd;
            }

            return null;
        }

        /// <summary>
        /// Creates a copy of the entire DatasetExCollection.
        /// </summary>
        /// <returns>The new DatasetExCollection is returned.</returns>
        public DatasetExCollection Clone()
        {
            DatasetExCollection col = new DatasetExCollection();

            col.m_bUseTrainingSourcesForTesting = m_bUseTrainingSourcesForTesting;

            foreach (DatasetEx ds in m_rgDatasets)
            {
                col.Add(ds.Clone());
            }

            return col;
        }

        /// <summary>
        /// Resets the last image set used to <i>null</i>, thus clearing it.
        /// </summary>
        public void Reset()
        {
            m_lastImgSet = null;
        }

        /// <summary>
        /// Relabels all datasets using a label mapping collection.
        /// </summary>
        /// <param name="col">Specifies the label mapping collection.</param>
        public void Relabel(LabelMappingCollection col)
        {
            foreach (DatasetEx ds in m_rgDatasets)
            {
                ds.Relabel(col);
            }
        }

        /// <summary>
        /// Returns the number of datasets in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgDatasets.Count; }
        }

        /// <summary>
        /// Enable/disable the using of the training sources for testing on all datasets.
        /// </summary>
        /// <param name="bEnable">Enable/disable the training sources for testing.</param>
        public void EnableUsingTrainingSourcesForTesting(bool bEnable)
        {
            m_bUseTrainingSourcesForTesting = bEnable;

            foreach (DatasetEx ds in m_rgDatasets)
            {
                ds.UseTrainingImagesForTesting = bEnable;
            }
        }

        /// <summary>
        /// Returns whether or not the training sources are set to be used for testing.
        /// </summary>
        public bool UseTrainingSourcesForTesting
        {
            get { return m_bUseTrainingSourcesForTesting; }
        }

        /// <summary>
        /// Returns the dataset at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The dataset at a the index is returned.</returns>
        public DatasetEx this[int nIdx]
        {
            get { return m_rgDatasets[nIdx]; }
        }

        /// <summary>
        /// Searches for the dataset with a dataset ID.
        /// </summary>
        /// <param name="nDatasetID">Specifies the dataset ID.</param>
        /// <returns>If found, the DatasetEx is returned, otherwise <i>null</i> is returned.</returns>
        public DatasetEx FindDataset(int nDatasetID)
        {
            foreach (DatasetEx ds in m_rgDatasets)
            {
                if (ds.DatasetID == nDatasetID)
                    return ds;
            }

            return null;
        }

        /// <summary>
        /// Searches for the dataset with the dataset name.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <returns>If found, the DatasetEx is returned, otherwise <i>null</i> is returned.</returns>
        public DatasetEx FindDataset(string strDs)
        {
            foreach (DatasetEx ds in m_rgDatasets)
            {
                if (ds.DatasetName == strDs)
                    return ds;
            }

            return null;
        }

        /// <summary>
        /// Searches for the ImageSet with a given data source ID.
        /// </summary>
        /// <param name="nSourceID">Specifies the ID of the data source.</param>
        /// <returns>If found, the ImageSet is returned, otherwise an Exception is thrown.</returns>
        public ImageSet FindImageset(int nSourceID)
        {
            lock (m_syncObj)
            {
                if (m_lastImgSet != null && m_lastImgSet.SourceID == nSourceID)
                    return m_lastImgSet;

                foreach (DatasetEx ds in m_rgDatasets)
                {
                    ImageSet imgSet = ds.Find(nSourceID);

                    if (imgSet != null)
                    {
                        m_lastImgSet = imgSet;
                        return imgSet;
                    }
                }

                throw new Exception("Could not find source with ID = " + nSourceID.ToString() + "!");
            }
        }

        /// <summary>
        /// Searches for the ImageSet with a given data source name.
        /// </summary>
        /// <param name="strSource">Specifies the name of the data source.</param>
        /// <returns>If found, the ImageSet is returned, otherwise an Exception is thrown.</returns>
        public ImageSet FindImageset(string strSource)
        {
            lock (m_syncObj)
            {
                if (m_lastImgSet != null && m_lastImgSet.SourceName == strSource)
                    return m_lastImgSet;

                foreach (DatasetEx ds in m_rgDatasets)
                {
                    ImageSet imgSet = ds.Find(strSource);

                    if (imgSet != null)
                    {
                        m_lastImgSet = imgSet;
                        return imgSet;
                    }
                }

                throw new Exception("Could not find source with Name = " + strSource + "!");
            }
        }

        /// <summary>
        /// Adds a DatasetEx to the collection.
        /// </summary>
        /// <param name="ds">Specifies the DatasetEx.</param>
        public void Add(DatasetEx ds)
        {
            m_rgDatasets.Add(ds);
        }

        /// <summary>
        /// Releases all resources used by the collection.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called from Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            foreach (DatasetEx ds in m_rgDatasets)
            {
                ds.Dispose();
            }

            m_rgDatasets.Clear();
        }

        /// <summary>
        /// Releases all resources used by the collection.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The collection's enumerator is returned.</returns>
        public IEnumerator<DatasetEx> GetEnumerator()
        {
            return m_rgDatasets.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The collection's enumerator is returned.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgDatasets.GetEnumerator();
        }
    }
}
