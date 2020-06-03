using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The MasterList is responsible for loading and managing access to the master list of images for a data source.
    /// </summary>
    public class MasterList : IDisposable
    {
        CryptoRandom m_random;
        DatasetFactory m_factory;
        SourceDescriptor m_src;
        SimpleDatum[] m_rgImages = null;
        RefreshManager m_refreshManager = null;
        LoadSequence m_loadSequence = null;
        List<WaitHandle> m_rgAbort = new List<WaitHandle>();
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        ManualResetEvent m_evtDone = new ManualResetEvent(false);
        ManualResetEvent m_evtRunning = new ManualResetEvent(false);
        ManualResetEvent m_evtRefreshCancel = new ManualResetEvent(false);
        ManualResetEvent m_evtRefreshRunning = new ManualResetEvent(false);
        ManualResetEvent m_evtRefreshDone = new ManualResetEvent(false);
        Thread m_dataLoadThread;
        Thread m_dataRefreshThread;
        SimpleDatum m_imgMean = null;
        Log m_log = null;
        int m_nLoadedCount = 0;
        bool m_bSilent = false;
        int m_nLoadCount = 0;
        int m_nReplacementBatch = 100;
        object m_syncObj = new object();


        /// <summary>
        /// The OnCalculateImageMean event fires when the ImageSet needs to calculate the image mean for the image set.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="random">Specifies the CryptoRandom to use for random selection.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="src">Specifies the data source that holds the data on the database.</param>
        /// <param name="factory">Specifies the data factory used to access the database data.</param>
        /// <param name="rgAbort">Specifies the cancel handles.</param>
        /// <param name="nMaxLoadCount">Optionally, specifies to automaticall start the image refresh which only applies when the number of images loaded into memory is less than the actual number of images (default = false).</param>
        public MasterList(CryptoRandom random, Log log, SourceDescriptor src, DatasetFactory factory, List<WaitHandle> rgAbort, int nMaxLoadCount = 0)
        {
            m_random = random;
            m_log = log;
            m_src = src;
            m_factory = factory;

            m_rgAbort.Add(m_evtCancel);
            if (rgAbort.Count > 0)
                m_rgAbort.AddRange(rgAbort);

            m_imgMean = m_factory.LoadImageMean(m_src.ID);

            m_nLoadCount = nMaxLoadCount;
            if (m_nLoadCount == 0)
                m_nLoadCount = m_src.ImageCount;

            m_rgImages = new SimpleDatum[m_nLoadCount];
            m_nLoadedCount = 0;

            if (m_nLoadCount < m_src.ImageCount)
                m_refreshManager = new RefreshManager(random, m_src, m_factory);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            m_evtCancel.Set();
            if (m_evtRunning.WaitOne(0))
                m_evtDone.WaitOne();

            m_evtRefreshCancel.Set();
            if (m_evtRefreshRunning.WaitOne(0))
                m_evtRefreshDone.WaitOne();
        }

        /// <summary>
        /// Returns true when the database is loaded with LoadLimit > 0, false otherwise.
        /// </summary>
        public bool IsLoadLimitEnabled
        {
            get
            {
                if (m_refreshManager != null)
                    return true;

                return false;
            }
        }

        /// <summary>
        /// Returns <i>true</i> when the master list is fully loaded, <i>false</i> otherwise.
        /// </summary>
        public bool IsFull
        {
            get
            {
                return (m_nLoadedCount == GetTotalCount()) ? true : false;
            }
        }

        /// <summary>
        /// Start loading the dataset.
        /// </summary>
        /// <param name="bSilent">Specifies whether or not to output the loading status.</param>
        /// <returns>If the dataset is already loading <i>false</i> is returned.</returns>
        public bool Load(bool bSilent = false)
        {
            lock (m_syncObj)
            {
                if (m_evtRunning.WaitOne(0))
                    return false;

                m_bSilent = bSilent;
                Unload(false);

                m_dataLoadThread = new Thread(new ThreadStart(dataLoadThread));
                m_dataLoadThread.Priority = ThreadPriority.AboveNormal;
                m_dataLoadThread.Start();
                m_evtRunning.WaitOne(1000);

                return true;
            }
        }

        /// <summary>
        /// Unload the data source images.
        /// </summary>
        /// <param name="bReLoad">Re-load the data source images right after the unload completes.</param>
        public void Unload(bool bReLoad)
        {
            lock (m_syncObj)
            {
                if (m_evtRunning.WaitOne(0))
                {
                    m_evtCancel.Set();
                    m_evtDone.WaitOne();
                }

                m_evtCancel.Reset();
                m_evtDone.Reset();
                m_evtRunning.Reset();
                m_rgImages = new SimpleDatum[m_nLoadCount];
                m_nLoadedCount = 0;
                GC.Collect();

                StopRefresh();

                m_loadSequence = new LoadSequence(m_random, m_rgImages.Length, m_src.ImageCount, m_refreshManager);
            }

            if (bReLoad)
                Load();
        }

        /// <summary>
        /// Start the refresh thread which will run if the number of images stored in memory is less than the total number of images in the data source, otherwise this function returns false.
        /// </summary>
        /// <param name="dfReplacementPct">Optionally, specifies the replacement percentage (default = 0.25 or 25%).</param>
        /// <returns>false is returned if the refresh thread is already running, or if the number of images in memory equal the number of images in the data source.</returns>
        public bool StartRefresh(double dfReplacementPct = 0.25)
        {
            lock (m_syncObj)
            {
                if (m_evtRefreshRunning.WaitOne(0))
                    return false;

                if (m_rgImages.Length >= m_src.ImageCount)
                    return false;

                int nMaxRefresh = (int)(m_nLoadCount * dfReplacementPct);
                if (nMaxRefresh == 0)
                    nMaxRefresh = 1;

                m_nReplacementBatch = nMaxRefresh;
                m_evtRefreshCancel.Reset();
                m_evtRefreshDone.Reset();
                m_dataRefreshThread = new Thread(new ThreadStart(dataRefreshThread));
                m_dataRefreshThread.Start();
                m_evtRefreshRunning.WaitOne();

                return true;
            }
        }

        /// <summary>
        /// Wait for the refres to complete.
        /// </summary>
        /// <param name="rgAbort">Specifies one or more cancellation handles.</param>
        /// <param name="nWait">Specifies an amount of time to wait in milliseconds.</param>
        /// <returns>If the refresh is done running, true is returned, otherwise false.</returns>
        public bool WaitForRefreshToComplete(List<WaitHandle> rgAbort, int nWait)
        {
            if (!m_evtRefreshRunning.WaitOne(0))
                return true;

            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.Add(m_evtRefreshDone);
            rgWait.AddRange(rgAbort);

            int nRes = WaitHandle.WaitAny(rgWait.ToArray(), nWait);
            if (nRes == 0)
                return true;

            return false;
        }

        /// <summary>
        /// Returns true after the refresh completes.
        /// </summary>
        public bool IsRefreshDone
        {
            get
            {
                if (m_evtRefreshDone.WaitOne(0) || !m_evtRefreshRunning.WaitOne(0))
                    return true;

                return false;
            }
        }

        /// <summary>
        /// Returns true if the refresh is running, false otherwise.
        /// </summary>
        public bool IsRefreshRunning
        {
            get
            {
                return m_evtRefreshRunning.WaitOne(0);
            }
        }

        /// <summary>
        /// Stop the refresh thread if running.
        /// </summary>
        public void StopRefresh()
        {
            lock (m_syncObj)
            {
                if (!m_evtRefreshRunning.WaitOne(0))
                    return;

                m_evtRefreshCancel.Set();
                m_evtRefreshDone.WaitOne();
            }
        }

        /// <summary>
        /// Set the image mean.
        /// </summary>
        /// <param name="d">Specifies the image mean.</param>
        /// <param name="bSave">Optionally, specifies whether or not to save the image mean in the database (default = false).</param>
        public void SetImageMean(SimpleDatum d, bool bSave = false)
        {
            m_imgMean = d;

            if (bSave)
                m_factory.SaveImageMean(d, true);
        }

        /// <summary>
        /// Return the total number of images whether loaded or not, in the data source.
        /// </summary>
        /// <returns></returns>
        public int GetTotalCount()
        {
            lock (m_syncObj)
            {
                return m_rgImages.Length;
            }
        }

        /// <summary>
        /// Return the currently loaded images in the data source.
        /// </summary>
        /// <returns></returns>
        public int GetLoadedCount()
        {
            return m_nLoadedCount;
        }

        /// <summary>
        /// Returns the image mean for the ImageSet.
        /// </summary>
        /// <param name="log">Specifies the Log used to output status.</param>
        /// <param name="rgAbort">Specifies a set of wait handles for aborting the operation.</param>
        /// <returns>The SimpleDatum with the image mean is returned.</returns>
        public SimpleDatum GetImageMean(Log log, WaitHandle[] rgAbort)
        {
            if (m_imgMean != null)
                return m_imgMean;

            int nLoadedCount = GetLoadedCount();
            int nTotalCount = GetTotalCount();

            if (nLoadedCount < nTotalCount)
            {
                double dfPct = (double)nLoadedCount / (double)nTotalCount;

                if (log != null)    
                    log.WriteLine("WARNING: Cannot create the image mean until all images have loaded - the data is currently " + dfPct.ToString("P") + " loaded.");

                return null;
            }

            if (OnCalculateImageMean != null)
            {
                CalculateImageMeanArgs args = new CalculateImageMeanArgs(m_rgImages);
                OnCalculateImageMean(this, args);

                if (args.Cancelled)
                    return null;

                m_imgMean = args.ImageMean;
                return m_imgMean;
            }

            RawImageMean imgMean = m_factory.GetRawImageMean();
            if (m_imgMean != null)
            {
                m_imgMean = m_factory.LoadDatum(imgMean);
            }
            else
            {
                log.WriteLine("Calculating mean...");
                m_imgMean = SimpleDatum.CalculateMean(log, m_rgImages, rgAbort);
                m_factory.PutRawImageMean(m_imgMean, true);
            }

            m_imgMean.SetLabel(0);

            return m_imgMean;
        }

        /// <summary>
        /// Reload the image indexing.
        /// </summary>
        /// <returns>The indexes are returned as a list.</returns>
        public List<DbItem> ReloadIndexing()
        {
            return m_rgImages.Select(p => new DbItem { id = p.ImageID, index = p.Index, label = p.Label, boost = p.Boost, time = p.TimeStamp, desc = p.Description }).ToList();
        }

        /// <summary>
        /// Reset the labels of all images to the original labels.
        /// </summary>
        /// <returns>The new set of DBItems is returned for the images.</returns>
        public List<DbItem> ResetLabels()
        {
            if (!IsFull)
                throw new Exception("Relabeling only supported on fully loaded data sets.");

            foreach (SimpleDatum sd in m_rgImages)
            {
                if (sd != null) 
                    sd.ResetLabel();
            }

            return ReloadIndexing();
        }

        /// <summary>
        /// Relabel the images based on the LabelMappingCollection.
        /// </summary>
        /// <param name="col">Specifies the label mapping collection.</param>
        /// <returns>The new set of DBItems is returned for the images.</returns>
        public List<DbItem> Relabel(LabelMappingCollection col)
        {
            if (!IsFull)
                throw new Exception("Relabeling only supported on fully loaded data sets.");

            foreach (SimpleDatum sd in m_rgImages)
            {
                if (sd != null)
                    sd.SetLabel(col.MapLabelWithoutBoost(sd.OriginalLabel));
            }

            return ReloadIndexing();
        }

        /// <summary>
        /// Reset all image boosts.
        /// </summary>
        /// <returns>The new set of DBItems is returned for the images.</returns>
        public List<DbItem> ResetAllBoosts()
        {
            if (!IsFull)
                throw new Exception("Relabeling only supported on fully loaded data sets.");

            foreach (SimpleDatum sd in m_rgImages)
            {
                if (sd != null)
                    sd.ResetBoost();
            }

            return ReloadIndexing();
        }

        /// <summary>
        /// Find the image index based by searching the rgItems for an image that contains the description specified.
        /// </summary>
        /// <param name="rgItems">Specifies the image items to use to search.</param>
        /// <param name="strDesc">Specifies the image description to look for.</param>
        /// <returns>If found the image index is returned, otherwise -1 is returned.</returns>
        public int FindImageIndex(List<DbItem> rgItems, string strDesc)
        {
            List<int?> rgIdx = rgItems.Select(p => p.index).ToList();
            List<SimpleDatum> rgSd;

            lock (m_syncObj)
            {
                if (m_rgImages == null)
                    return -1;

                rgSd = m_rgImages.Where(p => p != null && rgIdx.Contains(p.Index)).ToList();
            }

            if (rgSd.Count == 0)
                return -1;

            rgSd = rgSd.Where(p => p.Description == strDesc).ToList();
            if (rgSd.Count == 0)
                return -1;

            return rgSd[0].Index;
        }

        /// <summary>
        /// Find an image based on its image ID (e.g. the image ID in the database).
        /// </summary>
        /// <param name="nImageId">Specifies the image ID in the database.</param>
        /// <returns>If found the image is returned, otherwise <i>null</i> is returned.</returns>
        public SimpleDatum FindImage(int nImageId)
        {
            List<SimpleDatum> rgSd;

            lock (m_syncObj)
            {
                rgSd = m_rgImages.Where(p => p != null && p.ImageID == nImageId).ToList();
            }

            if (rgSd.Count == 0)
                return null;

            return rgSd[0];
        }

        private IEnumerable<SimpleDatum> getQuery(bool bSuperboostOnly, string strFilterVal = null, int? nBoostVal = null)
        {
            IEnumerable<SimpleDatum> iQuery = m_rgImages.Where(p => p != null);

            if (bSuperboostOnly)
            {
                if (nBoostVal.HasValue)
                {
                    int nVal = nBoostVal.Value;

                    if (nVal < 0)
                    {
                        nVal = Math.Abs(nVal);
                        iQuery = iQuery.Where(p => p.Boost == nVal);
                    }
                    else
                    {
                        iQuery = iQuery.Where(p => p.Boost >= nVal);
                    }
                }
                else
                {
                    iQuery = iQuery.Where(p => p.Boost > 0);
                }
            }

            if (!string.IsNullOrEmpty(strFilterVal))
                iQuery = iQuery.Where(p => p.Description == strFilterVal);

            return iQuery;
        }

        /// <summary>
        /// Get the image with a specific image index.
        /// </summary>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="bLoadDataCriteria">Specifies whether or not to load the data criteria along with the image.</param>
        /// <param name="bLoadDebugData">Specifies whether or not to load the debug data with the image.</param>
        /// <param name="loadMethod">Specifies the image loading method used.</param>
        /// <returns>If found, the image is returned.</returns>
        public SimpleDatum GetImage(int nIdx, bool bLoadDataCriteria, bool bLoadDebugData, IMAGEDB_LOAD_METHOD loadMethod)
        {
            SimpleDatum sd = m_rgImages[nIdx];

            if (sd == null)
            {
                if (m_refreshManager != null)
                {
                    while (nIdx > 0 && m_rgImages[nIdx] == null)
                    {
                        nIdx--;
                    }

                    sd = m_rgImages[nIdx];

                    if (sd == null)
                        throw new Exception("No images should be null when using LoadLimit loading!");
                }
                else
                {
                    if (!m_evtRunning.WaitOne(0) && (loadMethod != IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND))
                        Load((loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND_BACKGROUND) ? true : false);

                    sd = directLoadImage(nIdx);
                    if (sd == null)
                        throw new Exception("The image is still null yet should have loaded!");

                    if (loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND)
                        m_rgImages[nIdx] = sd;
                }
            }

            // Double check that the conditional data has loaded (if needed).
            if (bLoadDataCriteria || bLoadDebugData)
                m_factory.LoadRawData(sd, bLoadDataCriteria, bLoadDebugData);

            return sd;
        }

        /// <summary>
        /// Returns the number of images in the image set, optionally with super-boosted values only.
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
            List<int> rgIdx = state.GetIndexes(0, int.MaxValue, strFilterVal, nBoostVal, bBoostValIsExact);
            return rgIdx.Count();
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
            List<int> rgIdx = state.GetIndexes(nStartIdx, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);
            List<SimpleDatum> rgSd;

            lock (m_syncObj)
            {
                rgSd = m_rgImages.Where(p => p != null && rgIdx.Contains(p.Index)).ToList();
            }

            if (bAttemptDirectLoad)
            {
                foreach (SimpleDatum sd in rgSd)
                {
                    if (sd != null)
                        rgIdx.Remove(sd.Index);
                }

                for (int i = 0; i < rgIdx.Count; i++)
                {
                    rgSd.Add(directLoadImage(rgIdx[i]));
                }

                rgSd = rgSd.OrderBy(p => p.Index).ToList();
            }

            return rgSd;
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
            List<int> rgIdx = state.GetIndexes(dtStart, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);

            lock (m_syncObj)
            {
                return m_rgImages.Where(p => p != null && rgIdx.Contains(p.Index)).ToList();
            }
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
            lock (m_syncObj)
            {
                IEnumerable<SimpleDatum> iQuery = getQuery(bSuperboostOnly, strFilterVal, nBoostVal);

                iQuery = iQuery.Where(p => p != null && rgIdx.Contains(p.Index));

                return iQuery.ToList();
            }
        }

        /// <summary>
        /// Wait for the image loading to complete - this is used when performing LOAD_ALL.
        /// </summary>
        /// <param name="rgAbort">Specifies one or more cancellation handles.</param>
        /// <param name="nWait">Optionally, specifies an amount to wait (default = int.MaxValue).</param>
        /// <returns>If the load is completed <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForLoadingToComplete(List<WaitHandle> rgAbort, int nWait = int.MaxValue)
        {
            int nLoadCount = GetLoadedCount();

            lock (m_syncObj)
            {
                if (nLoadCount == 0 && !m_evtRunning.WaitOne(0))
                    return false;
            }

            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.Add(m_evtDone);
            rgWait.AddRange(rgAbort);

            int nRes = WaitHandle.WaitAny(rgWait.ToArray(), nWait);
            if (nRes == 0)
                return true;

            return false;
        }

        private int getBatchSize(SourceDescriptor src)
        {
            int nBatchSize = 20000;

            int nImageSize = src.ImageHeight * src.ImageWidth * src.ImageChannels;
            if (nImageSize > 60000)
                nBatchSize = 5000;
            else if (nImageSize > 20000)
                nBatchSize = 7500;
            else if (nImageSize > 3000)
                nBatchSize = 10000;

            if (nBatchSize > m_nLoadCount)
                nBatchSize = m_nLoadCount;

            return nBatchSize;
        }

        /// <summary>
        /// Directly load an image, preempting the backgroud load - used by LOAD_ON_DEMAND for images not already loaded.
        /// </summary>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <returns>The image at the image index is returned.</returns>
        private SimpleDatum directLoadImage(int nIdx)
        {
            return m_factory.LoadImageAt(nIdx);
        }

        /// <summary>
        /// The dataLoadThread is responsible for loading the data source images in the background.
        /// </summary>
        private void dataLoadThread()
        {
            m_evtRunning.Set();
            DatasetFactory factory = new DatasetFactory(m_factory);
            int? nNextIdx = m_loadSequence.GetNext();
            Stopwatch sw = new Stopwatch();

            if (m_refreshManager != null)
                m_refreshManager.Reset();

            try
            {
                sw.Start();

                List<int> rgIdxBatch = new List<int>();
                int nBatchSize = getBatchSize(m_src);

                if (m_nLoadedCount > 0)
                    throw new Exception("The loaded count is > 0!");

                factory.Open(m_src);

                m_log.WriteLine(m_src.Name + " loading " + m_loadSequence.Count.ToString("N0") + " items...");

                while (nNextIdx.HasValue || rgIdxBatch.Count > 0)
                {
                    if (nNextIdx.HasValue)
                        rgIdxBatch.Add(nNextIdx.Value);

                    if (rgIdxBatch.Count >= nBatchSize || !nNextIdx.HasValue)
                    {
                        List<RawImage> rgImg;

                        if (m_refreshManager == null)
                            rgImg = factory.GetRawImagesAt(rgIdxBatch[0], rgIdxBatch.Count);
                        else                        
                            rgImg = factory.GetRawImagesAt(rgIdxBatch, m_evtCancel);

                        if (rgImg == null)
                            break;

                        for (int j = 0; j < rgImg.Count; j++)
                        {
                            SimpleDatum sd = factory.LoadDatum(rgImg[j]);

                            if (m_refreshManager != null)
                                m_refreshManager.AddLoaded(sd);

                            m_rgImages[m_nLoadedCount] = sd;
                            m_nLoadedCount++;

                            if (sw.Elapsed.TotalMilliseconds > 1000)
                            {
                                if (m_log != null && !m_bSilent)
                                {
                                    double dfPct = m_nLoadedCount / (double)m_rgImages.Length;
                                    m_log.Progress = dfPct;
                                    m_log.WriteLine("Loading '" + m_src.Name + "' at " + dfPct.ToString("P") + " (" + m_nLoadedCount.ToString("N0") + " of " + m_rgImages.Length.ToString("N0") + ")...");
                                }

                                int nWait = WaitHandle.WaitAny(m_rgAbort.ToArray(), 0);
                                if (nWait != WaitHandle.WaitTimeout)
                                    return;

                                sw.Restart();
                            }
                        }

                        rgIdxBatch = new List<int>();
                    }

                    nNextIdx = m_loadSequence.GetNext();
                }

                if (rgIdxBatch.Count > 0)
                    m_log.FAIL("Not all images were loaded!");
            }
            finally
            {
                factory.Close();
                factory.Dispose();
                m_evtRunning.Reset();
                m_evtDone.Set();
            }
        }

        private void dataRefreshThread()
        {
            m_evtRefreshRunning.Set();
            DatasetFactory factory = new DatasetFactory(m_factory);
            Stopwatch sw = new Stopwatch();

            try
            {
                sw.Start();

                m_log.WriteLine("Starting refresh of " + m_nReplacementBatch.ToString("N0") + " items...");

                List<Tuple<int, SimpleDatum>> rgReplace = new List<Tuple<int, SimpleDatum>>();
                List<Tuple<int, DbItem>> rgItems = new List<Tuple<int, DbItem>>();
                List<DbItem> rgDbItems = new List<DbItem>();

                // Load the replacement set.
                for (int i = 0; i < m_nReplacementBatch; i++)
                {
                    int nIdx = m_random.Next(m_rgImages.Length);
                    int? nLabel = null;

                    if (m_rgImages[nIdx] != null)
                        nLabel = m_rgImages[nIdx].Label;

                    DbItem img = m_refreshManager.GetNextImageId(nLabel);
                    rgItems.Add(new Tuple<int, DbItem>(nIdx, img));
                    rgDbItems.Add(img);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        if (m_evtRefreshCancel.WaitOne(0))
                            return;

                        sw.Restart();
                    }
                }

                // Get the Datums, ordered by ID.
                List<SimpleDatum> rgImg = m_factory.GetImagesAt(rgDbItems, m_evtCancel);
                if (rgImg == null)
                    return;

                rgImg = rgImg.OrderBy(p => p.ImageID).ToList();
                rgItems = rgItems.OrderBy(p => p.Item2.ID).ToList();

                if (rgImg.Count != rgItems.Count)
                {
                    List<Tuple<int, DbItem>> rgItems1 = new List<Tuple<int, DbItem>>();
                    int nIdx = 0;
                    for (int i = 0; i < rgImg.Count; i++)
                    {
                        while (nIdx < rgItems.Count && rgItems[nIdx].Item2.ID < rgImg[i].ImageID)
                        {
                            nIdx++;
                        }

                        if (rgItems[nIdx].Item2.ID == rgImg[i].ImageID)
                        {
                            rgItems1.Add(rgItems[nIdx]);
                            nIdx++;
                        }
                    }

                    rgItems = rgItems1;
                }

                for (int i = 0; i < rgItems.Count; i++)
                {
                    rgReplace.Add(new Tuple<int, SimpleDatum>(rgItems[i].Item1, rgImg[i]));
                }

                lock (m_syncObj)
                {
                    int nMismatchCount = 0;

                    for (int i = 0; i < rgReplace.Count; i++)
                    {
                        int nIdx = rgReplace[i].Item1;
                        if (m_rgImages[nIdx] != null && m_rgImages[nIdx].Label != rgReplace[i].Item2.Label)
                            nMismatchCount++;
                        else
                            m_rgImages[nIdx] = rgReplace[i].Item2;
                    }

                    if (nMismatchCount > 0)
                        m_log.WriteLine("WARNING: " + nMismatchCount.ToString("N0") + " label mismatches found!");
                }
            }
            finally
            {
                m_evtRefreshRunning.Reset();
                m_evtRefreshDone.Set();
            }
        }
    }

#pragma warning disable 1591

    public class LoadSequence /** @private */
    {
        List<int> m_rgLoadSequence = new List<int>();
        Dictionary<int, Tuple<bool, bool>> m_rgLoadConditions = new Dictionary<int, Tuple<bool, bool>>();
        Dictionary<int, AutoResetEvent> m_rgPending = new Dictionary<int, AutoResetEvent>();
        object m_syncObj = new object();

        public LoadSequence(CryptoRandom random, int nCount, int nImageCount, RefreshManager refresh)
        {
            // When using load limit (count < image count), load item indexes in such
            // a way that balances the number of items per label.
            if (nCount < nImageCount)
            {
                Dictionary<int, List<DbItem>> rgItemsByLabel = refresh.GetItemsByLabel();
                List<int> rgLabel = rgItemsByLabel.Where(p => p.Value.Count > 0).Select(p => p.Key).ToList();

                for (int i = 0; i < nCount; i++)
                {
                    int nLabelIdx = random.Next(rgLabel.Count);
                    int nLabel = rgLabel[nLabelIdx];
                    List<DbItem> rgItems = rgItemsByLabel[nLabel];
                    int nItemIdx = random.Next(rgItems.Count);
                    DbItem item = rgItems[nItemIdx];

                    m_rgLoadSequence.Add(item.Index);

                    rgLabel.Remove(nLabel);
                    if (rgLabel.Count == 0)
                        rgLabel = rgItemsByLabel.Where(p => p.Value.Count > 0).Select(p => p.Key).ToList();
                }

                refresh.Reset();
            }
            // Otherwise just load all item indexes.
            else
            {
                for (int i = 0; i < nCount; i++)
                {
                    m_rgLoadSequence.Add(i);
                }
            }
        }

        public int Count
        {
            get { return m_rgLoadSequence.Count; }
        }

        public bool IsEmpty
        {
            get
            {
                lock (m_syncObj)
                {
                    return (m_rgLoadSequence.Count == 0) ? true : false;
                }
            }
        }

        public void PreEmpt(int nIdx, bool bLoadDataCriteria, bool bLoadDebugData)
        {
            lock (m_syncObj)
            {
                m_rgLoadConditions.Add(nIdx, new Tuple<bool, bool>(bLoadDataCriteria, bLoadDebugData));
                m_rgPending.Add(nIdx, new AutoResetEvent(false));
                m_rgLoadSequence.Remove(nIdx);
                m_rgLoadSequence.Insert(0, nIdx);
            }
        }

        public void SetLoaded(int nIdx)
        {
            lock (m_syncObj)
            {
                m_rgLoadConditions.Remove(nIdx);

                if (!m_rgPending.ContainsKey(nIdx))
                    return;

                m_rgPending[nIdx].Set();
            }
        }

        public bool WaitForLoad(int nIdx, int nWaitMs = 5000)
        {
            if (!m_rgPending.ContainsKey(nIdx))
                return false;

            bool bRes = m_rgPending[nIdx].WaitOne(nWaitMs);

            lock (m_syncObj)
            {
                m_rgPending.Remove(nIdx);
            }

            return bRes;
        }

        public int? GetNext()
        {
            lock (m_syncObj)
            {
                if (m_rgLoadSequence.Count == 0)
                    return null;

                int nIdx = m_rgLoadSequence[0];
                m_rgLoadSequence.RemoveAt(0);

                return nIdx;
            }
        }
    }

    public class RefreshManager
    {
        CryptoRandom m_random;
        List<DbItem> m_rgItems;
        Dictionary<int, List<DbItem>> m_rgItemsByLabel = null;
        Dictionary<int, List<int>> m_rgLoadedIdx = new Dictionary<int, List<int>>();
        Dictionary<int, List<int>> m_rgNotLoadedIdx = new Dictionary<int, List<int>>();

        public RefreshManager(CryptoRandom random, SourceDescriptor src, DatasetFactory factory)
        {
            m_random = random;
            m_rgItems = factory.LoadImageIndexes(false, true);
        }

        public Dictionary<int, List<DbItem>> GetItemsByLabel()
        {
            if (m_rgItemsByLabel == null)
            {
                m_rgItemsByLabel = new Dictionary<int, List<DbItem>>();

                for (int i=0; i<m_rgItems.Count; i++)
                {
                    DbItem item = m_rgItems[i];
                    item.Tag = i;

                    if (!m_rgItemsByLabel.ContainsKey(item.Label))
                        m_rgItemsByLabel.Add(item.Label, new List<DbItem>());

                    m_rgItemsByLabel[item.Label].Add(item);
                }

                Reset();
            }

            return m_rgItemsByLabel;
        }

        public void Reset()
        {
            m_rgLoadedIdx = new Dictionary<int, List<int>>();
            m_rgNotLoadedIdx = new Dictionary<int, List<int>>();

            foreach (KeyValuePair<int, List<DbItem>> kv in m_rgItemsByLabel)
            {
                m_rgNotLoadedIdx.Add(kv.Key, kv.Value.Select(p => (int)p.Tag).ToList());
            }
        }

        public void AddLoaded(SimpleDatum sd)
        {
            if (!m_rgLoadedIdx.ContainsKey(sd.Label))
                m_rgLoadedIdx.Add(sd.Label, new List<int>());

            DbItem item = m_rgItemsByLabel[sd.Label].Where(p => p.ID == sd.ImageID).First();
            m_rgLoadedIdx[sd.Label].Add((int)item.Tag);

            m_rgNotLoadedIdx[sd.Label].Remove((int)item.Tag);
        }

        public DbItem GetNextImageId(int? nLabel)
        {
            if (!nLabel.HasValue)
            {
                int nLabelIdx = m_random.Next(m_rgItemsByLabel.Count);
                nLabel = m_rgItemsByLabel.ElementAt(nLabelIdx).Key;
            }

            if (!m_rgLoadedIdx.ContainsKey(nLabel.Value))
                m_rgLoadedIdx.Add(nLabel.Value, new List<int>());

            List<int> rgLoadedIdx = m_rgLoadedIdx[nLabel.Value];
            List<int> rgNotLoadedIdx = m_rgNotLoadedIdx[nLabel.Value];

            if (rgNotLoadedIdx.Count == 0)
            {
                rgLoadedIdx.Clear();
                rgNotLoadedIdx = m_rgItemsByLabel[nLabel.Value].Select(p => (int)p.Tag).ToList();
            }

            int nIdx = m_random.Next(rgNotLoadedIdx.Count);
            int nMainIdx = rgNotLoadedIdx[nIdx];
            DbItem item = m_rgItems[nMainIdx];

            rgNotLoadedIdx.RemoveAt(nIdx);
            rgLoadedIdx.Add(nMainIdx);

            return item;
        }
    }

#pragma warning restore 1591
}
