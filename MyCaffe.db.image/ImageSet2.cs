using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.image
{
    /// <summary>
    /// [V2 Image Database]
    /// The ImageSet2 manages the data source data including the master list of images, and the master indexes that describe the data source layout (e.g. labels, boosts, etc).
    /// </summary>
    public class ImageSet2 : ImageSetBase, IDisposable
    {
        CryptoRandom m_random;
        IMAGEDB_LOAD_METHOD m_loadMethod;
        MasterList m_masterList = null;
        MasterIndexes m_masterIdx = null;
        List<WaitHandle> m_rgAbort = new List<WaitHandle>();
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        Log m_log;
        TYPE m_type;

        /// <summary>
        /// The OnCalculateImageMean event fires when the ImageSet needs to calculate the image mean for the image set.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// Defines the type of image set.
        /// </summary>
        public enum TYPE
        {
            /// <summary>
            /// Specifies an image set containing the trianing data source.
            /// </summary>
            TRAIN,
            /// <summary>
            /// Specifies an image set containing the testing data source.
            /// </summary>
            TEST
        }

        /// <summary>
        /// The ImageSet2 constructor.
        /// </summary>
        /// <param name="type">Specifies the type of data source managed.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="factory">Specifies the data factory used to access the database data.</param>
        /// <param name="src">Specifies the data source descriptor.</param>
        /// <param name="loadMethod">Specifies the load method used to load the data.</param>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="rgAbort">Specifies the cancellation handles.</param>
        public ImageSet2(TYPE type, Log log, DatasetFactory factory, SourceDescriptor src, IMAGEDB_LOAD_METHOD loadMethod, CryptoRandom random, WaitHandle[] rgAbort)
            : base(factory, src)
        {
            m_type = type;
            m_log = log;
            m_loadMethod = loadMethod;
            m_random = random;

            m_rgAbort.Add(m_evtCancel);
            if (rgAbort.Length > 0)
                m_rgAbort.AddRange(rgAbort);
        }

        /// <summary>
        /// Releases the resouces used.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called by Dispose()</param>
        protected override void Dispose(bool bDisposing)
        {
            m_evtCancel.Set();

            if (m_masterIdx != null)
            {
                m_masterIdx.Dispose();
                m_masterIdx = null;
            }

            if (m_masterList != null)
            {
                m_masterList.Dispose();
                m_masterList = null;
            }

            base.Dispose(bDisposing);
        }

        /// <summary>
        /// Initialize the ImageSet by creating the master list of images, starting its background image loading thread, and then creating the master index that maps the organization of the dataset.
        /// </summary>
        /// <param name="bSilentLoad">Specifies to load the data silently without status output.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="nMaxLoadCount">Optionally, specifies to automaticall start the image refresh which only applies when the number of images loaded into memory is less than the actual number of images (default = false).</param>
        /// <returns>Once initialized, the default query state for the image set is returned.  This method may be called multiple times and each time returns a new QueryState.</returns>
        public QueryState Initialize(bool bSilentLoad, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, int nMaxLoadCount = 0)
        {
            if (m_masterList == null)
            {
                m_masterList = new MasterList(m_random, m_log, m_src, m_factory, m_rgAbort, nMaxLoadCount);

                if (OnCalculateImageMean != null)
                    m_masterList.OnCalculateImageMean += OnCalculateImageMean;

                if (m_loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ALL || m_loadMethod == IMAGEDB_LOAD_METHOD.LOAD_EXTERNAL || m_loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND_BACKGROUND)
                    m_masterList.Load(bSilentLoad);
            }

            if (m_masterIdx == null || m_masterIdx.LoadLimit != nMaxLoadCount)
                m_masterIdx = new MasterIndexes(m_random, m_src, nMaxLoadCount);

            QueryState state = new QueryState(m_masterIdx, bUseUniqueLabelIndexes, bUseUniqueImageIndexes);

            if (m_loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ALL)
                m_masterList.WaitForLoadingToComplete(m_rgAbort);

            return state;
        }

        /// <summary>
        /// Wait for the image set to complete loading.
        /// </summary>
        /// <param name="nWait">Specifies the maximum number of ms to wait (default = int.MaxValue).</param>
        /// <returns>If the load has completed <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForLoadingToComplete(int nWait = int.MaxValue)
        {
            return m_masterList.WaitForLoadingToComplete(m_rgAbort, nWait);
        }

        /// <summary>
        /// Returns whether or not the refresh is running.
        /// </summary>
        public bool IsRefreshRunning
        {
            get { return m_masterList.IsRefreshRunning; }
        }

        /// <summary>
        /// Start the refresh process which only valid when initialized with LoadLimit > 0.
        /// </summary>
        /// <param name="dfReplacementPct">Optionally, specifies the replacement percentage (default = 0.25 or 25%).</param>
        /// <returns>false is returned if the refresh thread is already running, or if the number of images in memory equal the number of images in the data source.</returns>
        public bool StartRefresh(double dfReplacementPct = 0.25)
        {
            return m_masterList.StartRefresh(dfReplacementPct);
        }

        /// <summary>
        /// Wait for the image refresh to complete loading.
        /// </summary>
        /// <param name="nWait">Specifies the maximum number of ms to wait (default = int.MaxValue).</param>
        /// <returns>If the refresh has completed <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForRefreshToComplete(int nWait = int.MaxValue)
        {
            return m_masterList.WaitForRefreshToComplete(m_rgAbort, nWait);
        }

        /// <summary>
        /// Abort any refresh currently running.
        /// </summary>
        public void StopRefresh()
        {
            m_masterList.StopRefresh();
        }

        /// <summary>
        /// Returns the type of use for the ImageSet.
        /// </summary>
        public TYPE ImageSetType
        {
            get { return m_type; }
        }

        /// <summary>
        /// Starts loading the image set on the background thread if it is not already loading.
        /// </summary>
        /// <returns>If the load is already running <i>false</i> is returned, otherwise <i>true</i> is returned.</returns>
        public bool Load()
        {
            return m_masterList.Load();
        }

        /// <summary>
        /// Unload all images from the master list (freeing memory) and optionally reload the dataset.
        /// </summary>
        /// <param name="bReload">When <i>true</i>, the image set starts loading right after it is unloaded causing a refresh.</param>
        public void Unload(bool bReload)
        {
            m_masterList.Unload(bReload);
        }

        /// <summary>
        /// Create a new QueryState and optionally sort the results.
        /// </summary>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Optionally, specifies a sorting method for the query set.</param>
        /// <returns>The new QueryState is returned.</returns>
        public QueryState CreateQueryState(bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            return new QueryState(m_masterIdx, bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
        }

        /// <summary>
        /// Get the total number of images in the image set whether loaded or not.
        /// </summary>
        /// <returns>The total number of images is returned.</returns>
        public int GetTotalCount()
        {
            return m_masterList.GetTotalCount();
        }

        /// <summary>
        /// Get the total number of images already loaded in the image set.
        /// </summary>
        /// <returns>The total number of images loaded is returned.</returns>
        public int GetLoadedCount()
        {
            return m_masterList.GetLoadedCount();
        }

        /// <summary>
        /// Get the image mean for the iamge set, or create one if it does not exist.
        /// </summary>
        /// <param name="log">Specifies the output log used when creating the image mean.</param>
        /// <param name="rgAbort">Specifies the cancellation handles used to cancel the creation of the image mean.</param>
        /// <returns>The image mean is returned.</returns>
        public SimpleDatum GetImageMean(Log log, WaitHandle[] rgAbort)
        {
            return m_masterList.GetImageMean(log, rgAbort);
        }

        /// <summary>
        /// Reload the indexing for the image set.
        /// </summary>
        /// <returns>The new indexes are returned.</returns>
        public List<DbItem> ReloadIndexing()
        {
            List<DbItem> rgItems = m_masterList.ReloadIndexing();
            m_masterIdx.Reload(rgItems);
            return rgItems;
        }

        /// <summary>
        /// Reset all labels of the image set to the original labels.
        /// </summary>
        /// <returns>The new list of DbItem's is returned based on the newly reset labels.</returns>
        public List<DbItem> ResetLabels()
        {
            List<DbItem> rgItems = m_masterList.ResetLabels();
            m_masterIdx.Reload(rgItems);
            return rgItems;
        }

        /// <summary>
        /// Reset the image set based on the LabelMappingCollection.
        /// </summary>
        /// <param name="col">Specifies the label mapping that defines how to relabel the image set.</param>
        /// <returns>The new list of DbItem's is returned based on the newly updated labels.</returns>
        public List<DbItem> Relabel(LabelMappingCollection col)
        {
            List<DbItem> rgItems = m_masterList.Relabel(col);
            m_masterIdx.Reload(rgItems);
            return rgItems;
        }

        /// <summary>
        /// Reset all boosts to their original settings.
        /// </summary>
        /// <returns>The new list of DbItem's is returned based on the newly reset boosts.</returns>
        public List<DbItem> ResetAllBoosts()
        {
            List<DbItem> rgItems = m_masterList.ResetAllBoosts();
            m_masterIdx.Reload(rgItems);
            return rgItems;
        }

        /// <summary>
        /// Returns a list of all labels used by the data source.
        /// </summary>
        /// <returns>A list of LabelDescriptors is returned.</returns>
        public List<LabelDescriptor> GetLabels()
        {
            return m_src.Labels;
        }

        /// <summary>
        /// Return the label name of a given label.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>The name of the label is returned.</returns>
        public string GetLabelName(int nLabel)
        {
            foreach (LabelDescriptor label in m_src.Labels)
            {
                if (label.ActiveLabel == nLabel)
                    return label.Name;
            }

            return nLabel.ToString();
        }

        /// <summary>
        /// Sets the data source image mean.
        /// </summary>
        /// <param name="sd">Specifies the iamge mean to set.</param>
        /// <param name="bSave">Optionally, specifies whether or not to save the image mean in the database (default = false).</param>
        public void SetImageMean(SimpleDatum sd, bool bSave = false)
        {
            m_masterList.SetImageMean(sd, bSave);
        }

        /// <summary>
        /// Find the index of an image with the tiven date and (optionally) description.
        /// </summary>
        /// <param name="dt">Specifies the date to look for.</param>
        /// <param name="strDesc">Specifies the description to look for.</param>
        /// <returns>If found, the image index is returned, otherwise -1 is returned.</returns>
        public int FindImageIndex(DateTime dt, string strDesc)
        {
            List<DbItem> rgItems = m_masterIdx.FindImageIndexes(dt);
            if (rgItems.Count == 0)
                return -1;

            return m_masterList.FindImageIndex(rgItems, strDesc);
        }

        /// <summary>
        /// Get the image at a given image ID.
        /// </summary>
        /// <param name="nImageId">Specifies the image ID (within the database) of the image to retrieve.</param>
        /// <returns>The image is returned.</returns>
        public SimpleDatum GetImage(int nImageId)
        {
            return m_masterList.FindImage(nImageId);
        }

        /// <summary>
        /// Returns the image based on its label and image selection method.
        /// </summary>
        /// <param name="state">Specifies the query state.</param>
        /// <param name="labelSelectionMethod">Specifies the label selection method.</param>
        /// <param name="imageSelectionMethod">Specifies the image selection method.</param>
        /// <param name="log">Specifies the Log for status output.</param>
        /// <param name="nLabel">Optionally, specifies the label (default = null).</param>
        /// <param name="nDirectIdx">Optionally, specifies the image index to use when loading a specific index (default = -1).</param>
        /// <param name="bLoadDataCriteria">Optionally, specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Optionally, specifies to load the debug data (default = false).</param>
        /// <returns>The SimpleDatum containing the image is returned.</returns>
        public SimpleDatum GetImage(QueryState state, IMGDB_LABEL_SELECTION_METHOD labelSelectionMethod, IMGDB_IMAGE_SELECTION_METHOD imageSelectionMethod, Log log, int? nLabel = null, int nDirectIdx = -1, bool bLoadDataCriteria = false, bool bLoadDebugData = false)
        {
            if ((imageSelectionMethod & IMGDB_IMAGE_SELECTION_METHOD.BOOST) == IMGDB_IMAGE_SELECTION_METHOD.BOOST &&
                (labelSelectionMethod & IMGDB_LABEL_SELECTION_METHOD.RANDOM) == IMGDB_LABEL_SELECTION_METHOD.RANDOM)
                labelSelectionMethod |= IMGDB_LABEL_SELECTION_METHOD.BOOST;

            if (!nLabel.HasValue && (labelSelectionMethod & IMGDB_LABEL_SELECTION_METHOD.RANDOM) == IMGDB_LABEL_SELECTION_METHOD.RANDOM)
                nLabel = state.GetNextLabel(labelSelectionMethod);

            int? nIdx = state.GetNextImage(imageSelectionMethod, nLabel, nDirectIdx);

            if (!nIdx.HasValue || nIdx.Value < 0)
            {
                nIdx = state.GetNextImage(imageSelectionMethod, nLabel, nDirectIdx);
                if (!nIdx.HasValue || nIdx.Value < 0)
                {
                    string strBoosted = ((imageSelectionMethod & IMGDB_IMAGE_SELECTION_METHOD.BOOST) == IMGDB_IMAGE_SELECTION_METHOD.BOOST) ? "Boosted" : "";
                    string strLabel = (nLabel.HasValue) ? " for label '" + nLabel.Value.ToString() + "'." : ".";
                    throw new Exception("Failed to find the image index! The data source '" + m_src.Name + "' has no " + strBoosted + " images" + strLabel + ". You may need to re-index the dataset.");
                }
            }

            SimpleDatum sd = m_masterList.GetImage(nIdx.Value, bLoadDataCriteria, bLoadDebugData, m_loadMethod);

            state.UpdateStats(sd);

            return sd;
        }

        /// <summary>
        /// Returns the number of images in the image set, optionally with super-boost only.
        /// </summary>
        /// <param name="state">Specifies the query state to use.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The number of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public int GetCount(QueryState state, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            return m_masterList.GetCount(state, strFilterVal, nBoostVal, bBoostValIsExact);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="state">Specifies the query state to use.</param>
        /// <param name="nStartIdx">Specifies a starting index from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <param name="bAttemptDirectLoad">Optionaly, specifies to directly load all images not already loaded.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetImages(QueryState state, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false)
        {
            return m_masterList.GetImages(state, nStartIdx, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact, bAttemptDirectLoad);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="state">Specifies the query state to use.</param>
        /// <param name="dtStart">Specifies a starting time from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetImages(QueryState state, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            return m_masterList.GetImages(state, dtStart, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="bSuperboostOnly">Specifies whether or not to return images with super-boost.</param>
        /// <param name="strFilterVal">specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="rgIdx">Specifies a set of indexes to search for where the images returned must have an index greater than or equal to the individual index.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetImages(bool bSuperboostOnly, string strFilterVal, int? nBoostVal, int[] rgIdx)
        {
            return m_masterList.GetImages(bSuperboostOnly, strFilterVal, nBoostVal, rgIdx);
        }
    }
}
