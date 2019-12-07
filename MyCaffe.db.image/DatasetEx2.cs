using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using MyCaffe.basecode;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The DatasetEx class provides the in-memory dataset functionality that is used by the image database to manage data sets.  
    /// Both the Testing and Training ImageSet objects are managed by the DatasetEx, which in turn coordinates the loading and usage of each.
    /// </summary>
    public class DatasetEx2 : IDisposable
    {
        CryptoRandom m_random = null;
        object m_syncObj = new object();
        DatasetFactory m_factory = null;
        DatasetDescriptor m_ds = null;
        ImageSet2 m_TestingImages = null;
        ImageSet2 m_TrainingImages = null;
        bool m_bUseTrainingImagesForTesting = false;
        List<Guid> m_rgUsers = new List<Guid>();
        int m_nOriginalDsId = 0;
        QueryStateCollection m_queryStates = new QueryStateCollection();


        /// <summary>
        /// The OnCalculateImageMean event is passed to each image set and fires each time the Image set need to calcualte its image mean.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// The DatasetEx constructor.
        /// </summary>
        /// <param name="user">Specifies the unique ID of the dataset user.</param>
        /// <param name="factory">Specifies the DatasetFactory used to manage the database datasets.</param>
        /// <param name="random">Specifies the random number generator.</param>
        public DatasetEx2(Guid user, DatasetFactory factory, CryptoRandom random)
        {
            m_random = random;

            if (user != Guid.Empty)
                m_rgUsers.Add(user);

            m_factory = new DatasetFactory(factory);
        }


        /// <summary>
        /// Adds a user of the dataset.
        /// </summary>
        /// <param name="user">Specifies the unique ID of the dataset user.</param>
        /// <returns>The number of users is returned.</returns>
        public int AddUser(Guid user)
        {
            m_rgUsers.Add(user);
            return m_rgUsers.Count;
        }

        /// <summary>
        /// Remove a user of the dataset.
        /// </summary>
        /// <param name="user">Specifies the unique ID of the dataset user.</param>
        /// <returns>The number of users is returned.</returns>
        public int RemoveUser(Guid user)
        {
            m_rgUsers.Remove(user);
            return m_rgUsers.Count;
        }

        /// <summary>
        /// Initialize the DatasetEx by loading the training and testing data sources into memory.
        /// </summary>
        /// <param name="ds">Specifies the dataset to load.</param>
        /// <param name="rgAbort">Specifies a set of wait handles used to cancel the load.</param>
        /// <param name="nPadW">Optionally, specifies a pad to apply to the width of each item (default = 0).</param>
        /// <param name="nPadH">Optionally, specifies a pad to apply to the height of each item (default = 0).</param>
        /// <param name="log">Optionally, specifies an external Log to output status (default = null).</param>
        /// <param name="loadMethod">Optionally, specifies the load method to use (default = LOAD_ALL).</param>
        /// <param name="nImageDbLoadLimit">Optionally, specifies the load limit (default = 0).</param>
        /// <param name="bSkipMeanCheck">Optionally, specifies to skip the mean check (default = false).</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned, or 0 on cancel.</returns>
        public long Initialize(DatasetDescriptor ds, WaitHandle[] rgAbort, int nPadW = 0, int nPadH = 0, Log log = null, IMAGEDB_LOAD_METHOD loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL, int nImageDbLoadLimit = 0, bool bSkipMeanCheck = false)
        {
            lock (m_syncObj)
            {
                if (nImageDbLoadLimit > 0)
                    log.WriteLine("WARNING: ImageDbLoadLimit is not used currently and is ignored.");

                if (ds != null)
                    m_ds = ds;

                if (m_ds.TrainingSource.ImageWidth == -1 || m_ds.TrainingSource.ImageHeight == -1)
                {
                    log.WriteLine("WARNING: Cannot create a mean image for data sources that contain variable sized images.  The mean check will be skipped.");
                    bSkipMeanCheck = true;
                }

                m_TrainingImages = new ImageSet2(ImageSet2.TYPE.TRAIN, log, m_factory, m_ds.TrainingSource, loadMethod, m_random, rgAbort);
                QueryState qsTraining = m_TrainingImages.Initialize();

                if (!bSkipMeanCheck)
                    m_TrainingImages.GetImageMean(log, rgAbort);

                if (EventWaitHandle.WaitAny(rgAbort, 0) != EventWaitHandle.WaitTimeout)
                    return 0;

                m_TestingImages = new ImageSet2(ImageSet2.TYPE.TEST, log, m_factory, m_ds.TestingSource, loadMethod, m_random, rgAbort);
                QueryState qsTesting = m_TestingImages.Initialize();

                if (!bSkipMeanCheck)
                    m_TestingImages.GetImageMean(log, rgAbort);

                if (EventWaitHandle.WaitAny(rgAbort, 0) != EventWaitHandle.WaitTimeout)
                    return 0;

                return m_queryStates.CreateNewState(qsTraining, qsTesting);
            }
        }

        /// <summary>
        /// Wait for either the training, testing or both data sources to complete loading.
        /// </summary>
        /// <param name="bTraining">Specifies to wait for the training data source.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source.</param>
        /// <param name="nWait"></param>
        /// <returns></returns>
        public bool WaitForLoadingToComplete(bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            if (bTraining)
            {
                if (!m_TrainingImages.WaitForLoadingToComplete(nWait))
                    return false;
            }

            if (bTesting)
            {
                if (!m_TestingImages.WaitForLoadingToComplete(nWait))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called by Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            m_ds = null;

            if (m_TestingImages != null)
            {
                m_TestingImages.Dispose();
                m_TestingImages = null;
            }

            if (m_TrainingImages != null)
            {
                m_TrainingImages.Dispose();
                m_TrainingImages = null;
            }

            if (m_factory != null)
            {
                m_factory.Dispose();
                m_factory = null;
            }
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Create a new QueryState on the dataset.
        /// </summary>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Optionally, specifies an ordering for the query state (default = NONE).</param>
        /// <returns>The new query state is returned.</returns>
        public long CreateQueryState(bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            QueryState qsTraining = m_TrainingImages.CreateQueryState(bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
            QueryState qsTesting = m_TestingImages.CreateQueryState(bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
            return m_queryStates.CreateNewState(qsTraining, qsTesting);
        }

        /// <summary>
        /// Free an existing query state.
        /// </summary>
        /// <param name="lHandle">Specifies the handle to the query state to be freed.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool FreeQueryState(long lHandle)
        {
            return m_queryStates.FreeQueryState(lHandle);
        }

        /// <summary>
        /// Returns the query state for a given query state handle and type.
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="type">Specifies the query state type to retrieve.</param>
        /// <returns>The QueryState is returned.</returns>
        public QueryState FindQueryState(long lQueryState, ImageSet2.TYPE type)
        {
            if (type == ImageSet2.TYPE.TEST)
                return m_queryStates.GetTestingState(lQueryState);
            else
                return m_queryStates.GetTrainingState(lQueryState);
        }

        /// <summary>
        /// Reload the indexing for both the training and testing data sources.
        /// </summary>
        public void ReloadIndexing()
        {
            List<DbItem> rgItems = m_TrainingImages.ReloadIndexing();
            m_queryStates.ReIndexTraining(rgItems);

            rgItems = m_TrainingImages.ReloadIndexing();
            m_queryStates.ReIndexTesting(rgItems);
        }

        /// <summary>
        /// Relabels both the testing and training image sets using the label mapping collection.
        /// </summary>
        /// <param name="col">Specifies the label mapping collection.</param>
        public void Relabel(LabelMappingCollection col)
        {
            List<DbItem> rgItems = m_TrainingImages.Relabel(col);
            m_queryStates.ReIndexTraining(rgItems);

            rgItems = m_TrainingImages.Relabel(col);
            m_queryStates.ReIndexTesting(rgItems);
        }

        /// <summary>
        /// Resets the labels to their original labels.
        /// </summary>
        public void ResetLabels()
        {
            List<DbItem> rgItems = m_TrainingImages.ResetLabels();
            m_queryStates.ReIndexTraining(rgItems);

            rgItems = m_TrainingImages.ResetLabels();
            m_queryStates.ReIndexTesting(rgItems);
        }

        /// <summary>
        /// Reset all boosts for both the testing and training image sets.
        /// </summary>
        public void ResetAllBoosts()
        {
            List<DbItem> rgItems = m_TrainingImages.ResetAllBoosts();
            m_queryStates.ReIndexTraining(rgItems);

            rgItems = m_TrainingImages.ResetAllBoosts();
            m_queryStates.ReIndexTesting(rgItems);
        }

        /// <summary>
        /// Get/set whether or not to use the training images when testing.
        /// </summary>
        public bool UseTrainingImagesForTesting
        {
            get { return m_bUseTrainingImagesForTesting; }
            set { m_bUseTrainingImagesForTesting = value; }
        }

        /// <summary>
        /// Saves the image mean in a SimpleDatum to the database.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source to use.</param>
        /// <param name="sd">Specifies the image mean data.</param>
        /// <param name="bUpdate">Specifies whether or not to update the mean image.</param>
        /// <returns>If saved successfully, this method returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        public bool SaveImageMean(int nSrcId, SimpleDatum sd, bool bUpdate)
        {
            if (m_TestingImages.Source.ID != nSrcId &&
                m_TrainingImages.Source.ID != nSrcId)
                return false;

            return m_factory.SaveImageMean(sd, bUpdate, nSrcId);
        }

        /// <summary>
        /// Query the image mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source to use.</param>
        /// <returns>The SimpleDatum containing the image mean is returned.</returns>
        public SimpleDatum QueryImageMean(int nSrcId)
        {
            if (m_TestingImages.Source.ID != nSrcId &&
                m_TrainingImages.Source.ID != nSrcId)
                return null;

            return m_factory.QueryImageMean(nSrcId);
        }

        /// <summary>
        /// Unload the images of the training and testing image sets.
        /// </summary>
        public void Unload(bool bReload)
        {
            lock (m_syncObj)
            {
                m_TestingImages.Unload(bReload);
                m_TrainingImages.Unload(bReload);
            }
        }

        /// <summary>
        /// Returns the total percentage of images loaded for testing, training and combined.
        /// </summary>
        /// <param name="dfTraining">Returns the total percentage of training images loaded.</param>
        /// <param name="dfTesting">Returns the total percentage of testing images loaded.</param>
        /// <returns>Returns the combined total percentage of images loaded for both testing and training.</returns>
        public double GetPercentageLoaded(out double dfTraining, out double dfTesting)
        {
            int nTrainingTotal = m_TrainingImages.GetTotalCount();
            int nTrainingLoaded = m_TrainingImages.GetLoadedCount();
            int nTestingTotal = m_TestingImages.GetTotalCount();
            int nTestingLoaded = m_TestingImages.GetLoadedCount();

            dfTraining = (double)nTrainingLoaded / (double)nTrainingTotal;
            dfTesting = (double)nTestingLoaded / (double)nTestingTotal;

            int nTotalLoaded = nTrainingLoaded + nTestingLoaded;
            int nTotalImages = nTrainingTotal + nTestingTotal;

            return (double)nTotalLoaded / (double)nTotalImages;
        }

        /// <summary>
        /// Returns the ImageSet corresponding to a data source ID.
        /// </summary>
        /// <param name="nSourceID">Specifies the ID of the data source to use.</param>
        /// <returns>The ImageSet of images is returned.</returns>
        public ImageSet2 Find(int nSourceID)
        {
            if (m_TestingImages.Source.ID == nSourceID)
            {
                if (m_bUseTrainingImagesForTesting)
                    return m_TrainingImages;

                return m_TestingImages;
            }

            if (m_TrainingImages.Source.ID == nSourceID)
            {
                return m_TrainingImages;
            }

            return null;
        }

        /// <summary>
        /// Returns the ImageSet corresponding to a data source name.
        /// </summary>
        /// <param name="strSource">Specifies the name of the data source to use.</param>
        /// <returns>The ImageSet of images is returned.</returns>
        public ImageSet2 Find(string strSource)
        {
            if (m_TestingImages.Source.Name == strSource)
            {
                if (m_bUseTrainingImagesForTesting)
                    return m_TrainingImages;

                return m_TestingImages;
            }

            if (m_TrainingImages.Source.Name == strSource)
            {
                return m_TrainingImages;
            }

            return null;
        }

        /// <summary>
        /// Returns the dataset descriptor of the dataset managesd by the DatasetEx object.
        /// </summary>
        public DatasetDescriptor Descriptor
        {
            get { return m_ds; }
        }

        /// <summary>
        /// Returns the dataset ID of the dataset managesd by the DatasetEx object.
        /// </summary>
        public int DatasetID
        {
            get { return m_ds.ID; }
            set { m_ds.ID = value; }
        }

        /// <summary>
        /// Returns the original DatsetID if this is a cloned re-organized dataset, otherwise 0 is returned.
        /// </summary>
        public int OriginalDatasetID
        {
            get { return m_nOriginalDsId; }
        }

        /// <summary>
        /// Returns the dataset name of the dataset managesd by the DatasetEx object.
        /// </summary>
        public string DatasetName
        {
            get { return m_ds.Name; }
        }
    }
}
