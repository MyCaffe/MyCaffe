using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.db.image
{
    /// <summary>
    /// [V2 Image Database]
    /// The DatasetExCollection2 contains a list of DatasetEx2 objects.
    /// </summary>
    public class DatasetExCollection2 : IEnumerable<DatasetEx2>, IDisposable
    {
        List<DatasetEx2> m_rgDatasets = new List<DatasetEx2>();
        bool m_bUseTrainingSourcesForTesting = false;
        ImageSet2 m_lastImgSet = null;
        object m_syncObj = new object();

        /// <summary>
        /// The DatasetExCollection2 constructor.
        /// </summary>
        public DatasetExCollection2()
        {
        }

        /// <summary>
        /// Remove the dataset specified.
        /// </summary>
        /// <param name="ds">Specifies the dataset to remove.</param>
        /// <returns>If the dataset is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool RemoveDataset(DatasetEx2 ds)
        {
            return m_rgDatasets.Remove(ds);
        }

        /// <summary>
        /// Removes a user from the list of users using the DatasetExCollection.
        /// </summary>
        /// <param name="user">Specifies the unique user ID.</param>
        /// <returns>Returns <i>true</i> after all users are released from all datasets, <i>false</i> otherwise.</returns>
        public bool RemoveUser(Guid user)
        {
            bool bReleased = true;

            foreach (DatasetEx2 ds in m_rgDatasets)
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
            foreach (DatasetEx2 ds in m_rgDatasets)
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
            foreach (DatasetEx2 ds in m_rgDatasets)
            {
                SimpleDatum sd = ds.QueryImageMean(nSrcID);
                if (sd != null)
                    return sd;
            }

            return null;
        }

        /// <summary>
        /// Resets the last image set used to <i>null</i>, thus clearing it.
        /// </summary>
        public void Reset()
        {
            m_lastImgSet = null;
        }

        /// <summary>
        /// Reload the dataset's indexing.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        public void ReloadIndexing(int nDsId)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetID == nDsId)
                    {
                        ds.ReloadIndexing();
                        return;
                    }
                }

                throw new Exception("Failed to create a new query state for dataset ID = " + nDsId.ToString() + ".");
            }
        }

        /// <summary>
        /// Reload the dataset's indexing.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        public void ReloadIndexing(string strDs)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetName == strDs)
                    {
                        ds.ReloadIndexing();
                        return;
                    }
                }

                throw new Exception("Failed to create a new query state for dataset = '" + strDs.ToString() + "'.");
            }
        }

        /// <summary>
        /// Relabels all datasets using a label mapping collection.
        /// </summary>
        /// <param name="col">Specifies the label mapping collection.</param>
        public void Relabel(LabelMappingCollection col)
        {
            foreach (DatasetEx2 ds in m_rgDatasets)
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

            foreach (DatasetEx2 ds in m_rgDatasets)
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
        public DatasetEx2 this[int nIdx]
        {
            get { return m_rgDatasets[nIdx]; }
        }

        /// <summary>
        /// Searches for the dataset with a dataset ID.
        /// </summary>
        /// <param name="nDatasetID">Specifies the dataset ID.</param>
        /// <returns>If found, the DatasetEx is returned, otherwise <i>null</i> is returned.</returns>
        public DatasetEx2 FindDataset(int nDatasetID)
        {
            foreach (DatasetEx2 ds in m_rgDatasets)
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
        public DatasetEx2 FindDataset(string strDs)
        {
            foreach (DatasetEx2 ds in m_rgDatasets)
            {
                if (ds.DatasetName == strDs)
                    return ds;
            }

            return null;
        }

        /// <summary>
        /// Searches for the dataset containing the given Source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the source ID.</param>
        /// <returns>If found, the dataset is returned.</returns>
        public DatasetEx2 FindDatasetFromSource(int nSrcId)
        {
            foreach (DatasetEx2 ds in m_rgDatasets)
            {
                if (ds.Find(nSrcId) != null)
                    return ds;
            }

            return null;
        }

        /// <summary>
        /// Searches for the ImageSet with a given data source ID.
        /// </summary>
        /// <param name="nSourceID">Specifies the ID of the data source.</param>
        /// <returns>If found, the ImageSet is returned, otherwise an Exception is thrown.</returns>
        public ImageSet2 FindImageset(int nSourceID)
        {
            lock (m_syncObj)
            {
                if (m_lastImgSet != null && m_lastImgSet.Source.ID == nSourceID)
                    return m_lastImgSet;

                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    ImageSet2 imgSet = ds.Find(nSourceID);

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
        public ImageSet2 FindImageset(string strSource)
        {
            lock (m_syncObj)
            {
                if (m_lastImgSet != null && m_lastImgSet.Source.Name == strSource)
                    return m_lastImgSet;

                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    ImageSet2 imgSet = ds.Find(strSource);

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
        /// Wait for the dataset loading to complete.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to load.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to load.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete loading, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForDatasetToLoad(int nDsId, bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetID == nDsId)
                        return ds.WaitForLoadingToComplete(bTraining, bTesting, nWait);
                }

                throw new Exception("Failed to create a new query state for dataset ID = " + nDsId.ToString() + ".");
            }
        }

        /// <summary>
        /// Wait for the dataset loading to complete.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to load.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to load.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete loading, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForDatasetToLoad(string strDs, bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetName == strDs)
                        return ds.WaitForLoadingToComplete(bTraining, bTesting, nWait);
                }

                throw new Exception("Failed to create a new query state for dataset = '" + strDs + "'.");
            }
        }

        /// <summary>
        /// Create a new query state, optionally with a certain sorting.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset on which to create the query state.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch eually (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Specifies the sorting method, if any.</param>
        /// <returns>The query state is returned.</returns>
        public long CreateQueryState(int nDsId, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetID == nDsId)
                        return ds.CreateQueryState(bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
                }

                throw new Exception("Failed to create a new query state for dataset ID = " + nDsId.ToString() + ".");
            }
        }

        /// <summary>
        /// Create a new query state, optionally with a certain sorting.
        /// </summary>
        /// <param name="strDs">Specifies the dataset on which to create the query state.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Specifies the sorting method, if any.</param>
        /// <returns>The query state is returned.</returns>
        public long CreateQueryState(string strDs, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetName == strDs)
                        return ds.CreateQueryState(bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
                }

                throw new Exception("Failed to create a new query state for dataset = '" + strDs + "'.");
            }
        }

        /// <summary>
        /// Set the default query state to the query state specified for the dataset specified.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="lQueryState">Specifies the query state to set.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        public bool SetDefaultQueryState(int nDsId, long lQueryState)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetID == nDsId)
                        return ds.SetDefaultQueryState(lQueryState);
                }

                throw new Exception("Failed to create a new query state for dataset ID = '" + nDsId.ToString() + "'.");
            }
        }

        /// <summary>
        /// Set the default query state to the query state specified for the dataset specified.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="lQueryState">Specifies the query state to set.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        public bool SetDefaultQueryState(string strDs, long lQueryState)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetName == strDs)
                        return ds.SetDefaultQueryState(lQueryState);
                }

                throw new Exception("Failed to create a new query state for dataset = '" + strDs + "'.");
            }
        }

        /// <summary>
        /// Frees a query state from a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset on which to free the query state.</param>
        /// <param name="lHandle">Specifies the handle to the query state to free.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool FreeQueryState(int nDsId, long lHandle)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetID == nDsId)
                        return ds.FreeQueryState(lHandle);
                }

                return false;
            }
        }

        /// <summary>
        /// Frees a query state from a given dataset.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name on which to free the query state.</param>
        /// <param name="lHandle">Specifies the handle to the query state to free.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool FreeQueryState(string strDs, long lHandle)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    if (ds.DatasetName == strDs)
                        return ds.FreeQueryState(lHandle);
                }

                return false;
            }
        }

        /// <summary>
        /// Returns the query state based on the handle and data source where the dataset that owns the data source is first located and the query handle is then used to lookup
        /// the QueryState for that dataset.
        /// </summary>
        /// <param name="lQueryState">Specifies the query state handle.</param>
        /// <param name="strSource">Specifies the dataset source who's dataset is used.</param>
        /// <returns>The query state is returned.</returns>
        public QueryState FindQueryState(long lQueryState, string strSource)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    ImageSet2 imgSet = ds.Find(strSource);

                    if (imgSet != null)
                        return ds.FindQueryState(lQueryState, imgSet.ImageSetType);
                }

                throw new Exception("Could not find query state for data source with Name = " + strSource + "!");
            }
        }

        /// <summary>
        /// Returns the query state based on the handle and data source where the dataset that owns the data source is first located and the query handle is then used to lookup
        /// the QueryState for that dataset.
        /// </summary>
        /// <param name="lQueryState">Specifies the query state handle.</param>
        /// <param name="nSrcId">Specifies the dataset source who's dataset is used.</param>
        /// <returns>The query state is returned.</returns>
        public QueryState FindQueryState(long lQueryState, int nSrcId)
        {
            lock (m_syncObj)
            {
                foreach (DatasetEx2 ds in m_rgDatasets)
                {
                    ImageSet2 imgSet = ds.Find(nSrcId);

                    if (imgSet != null)
                        return ds.FindQueryState(lQueryState, imgSet.ImageSetType);
                }

                throw new Exception("Could not find query state for data source with ID = " + nSrcId.ToString() + "!");
            }
        }

        /// <summary>
        /// Adds a DatasetEx to the collection.
        /// </summary>
        /// <param name="ds">Specifies the DatasetEx.</param>
        public void Add(DatasetEx2 ds)
        {
            m_rgDatasets.Add(ds);
        }

        /// <summary>
        /// Releases all resources used by the collection.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called from Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            foreach (DatasetEx2 ds in m_rgDatasets)
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
        public IEnumerator<DatasetEx2> GetEnumerator()
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
