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
    public class DatasetEx : IDisposable
    {
        object m_syncObj = new object();
        DatasetFactory m_factory = null;
        DatasetDescriptor m_ds = null;
        ImageSet m_TestingImages = null;
        ImageSet m_TrainingImages = null;
        bool m_bUseTrainingImagesForTesting = false;
        CryptoRandom m_random = new CryptoRandom();
        int m_nLastTestingImageIdx = 0;
        int m_nLastTrainingImageIdx = 0;
        List<Guid> m_rgUsers = new List<Guid>();
        int m_nOriginalDsId = 0;

        /// <summary>
        /// The OnCalculateImageMean event is passed to each image set and fires each time the Image set need to calcualte its image mean.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// The DatasetEx constructor.
        /// </summary>
        /// <param name="user">Specifies the unique ID of the dataset user.</param>
        /// <param name="factory">Specifies the DatasetFactory used to manage the database datasets.</param>
        public DatasetEx(Guid user, DatasetFactory factory)
        {
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
        /// <returns></returns>
        public bool Initialize(DatasetDescriptor ds, WaitHandle[] rgAbort, int nPadW = 0, int nPadH = 0, Log log = null, IMAGEDB_LOAD_METHOD loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL, int nImageDbLoadLimit = 0)
        {
            lock (m_syncObj)
            {
                if (loadMethod != IMAGEDB_LOAD_METHOD.LOAD_ALL && nImageDbLoadLimit > 0)
                    throw new Exception("Currently the load-limit only works with the LOAD_ALLL image loading method.");

                SimpleDatum imgMean = null;

                if (ds != null)
                    m_ds = ds;

                m_TrainingImages = loadImageset("Training", m_ds.TrainingSource, rgAbort, ref imgMean, out m_nLastTrainingImageIdx, nPadW, nPadH, log, loadMethod, nImageDbLoadLimit, m_nLastTrainingImageIdx, (ds == null) ? true : false);
                if (m_nLastTrainingImageIdx >= m_ds.TrainingSource.ImageCount)
                    m_nLastTrainingImageIdx = 0;

                if (EventWaitHandle.WaitAny(rgAbort, 0) != EventWaitHandle.WaitTimeout)
                    return false;

                m_TestingImages = loadImageset("Testing", m_ds.TestingSource, rgAbort, ref imgMean, out m_nLastTestingImageIdx, nPadW, nPadH, log, loadMethod, nImageDbLoadLimit, m_nLastTestingImageIdx, (ds == null) ? true : false);
                if (m_nLastTestingImageIdx >= m_ds.TestingSource.ImageCount)
                    m_nLastTestingImageIdx = 0;

                if (EventWaitHandle.WaitAny(rgAbort, 0) != EventWaitHandle.WaitTimeout)
                    return false;

                return true;
            }
        }

        /// <summary>
        /// Copy the DatasetEx and its contents.
        /// </summary>
        /// <param name="bReOrganizeByTime">Optionally, specifies to re-organize the training and testing sources so that they are both organized chronologically.</param>
        /// <returns>The new DatasetEx is returned.</returns>
        public DatasetEx Clone(bool bReOrganizeByTime = false)
        {
            DatasetEx ds = new DatasetEx(Guid.Empty, m_factory);

            foreach (Guid g in m_rgUsers)
            {
                ds.m_rgUsers.Add(g);
            }

            ds.m_ds = new DatasetDescriptor(m_ds);
            ds.m_TestingImages = m_TestingImages.Clone();
            ds.m_TrainingImages = m_TrainingImages.Clone();
            ds.m_bUseTrainingImagesForTesting = m_bUseTrainingImagesForTesting;

            if (bReOrganizeByTime)
            {
                ds.m_nOriginalDsId = ds.DatasetID;
                ds.DatasetID *= -1;

                int nTestingCount = ds.m_TestingImages.Count;
                int nTrainingCount = ds.m_TrainingImages.Count;

                List<SimpleDatum> rgSd = new List<SimpleDatum>();

                rgSd.AddRange(ds.m_TestingImages.Images);
                rgSd.AddRange(ds.m_TrainingImages.Images);

                int nCount = rgSd.Where(p => p != null).Count();
                if (nCount == 0)
                    throw new Exception("You must first load 'all' images with LOAD_ALL or LOAD_FROM_SERVICE!");

                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();

                for (int i = 0; i < nTrainingCount; i++)
                {
                    ds.m_TrainingImages.Images[i] = rgSd[i];
                    ds.m_TrainingImages.Images[i].Index = i;
                }

                for (int i = 0; i < nTestingCount; i++)
                {
                    ds.m_TestingImages.Images[i] = rgSd[i + nTrainingCount];
                    ds.m_TestingImages.Images[i].Index = i;
                }

                ds.m_TrainingImages.Source.ID *= -1;                
                ds.m_TestingImages.Source.ID *= -1;
                ds.Descriptor.TrainingSource.ID *= -1;
                ds.Descriptor.TestingSource.ID *= -1;

                ds.m_TrainingImages.ReloadLabelSets();
                ds.m_TestingImages.ReloadLabelSets();
            }

            return ds;
        }

        /// <summary>
        /// Relabels both the testing and training image sets using the label mapping collection.
        /// </summary>
        /// <param name="col">Specifies the label mapping collection.</param>
        public void Relabel(LabelMappingCollection col)
        {
            m_TestingImages.Relabel(col);
            m_TrainingImages.Relabel(col);
        }

        /// <summary>
        /// Reloads bot the training and testing label sets.
        /// </summary>
        public void ReloadLabelSets()
        {
            m_TrainingImages.ReloadLabelSets();
            m_TestingImages.ReloadLabelSets();
        }

        /// <summary>
        /// Get/set whether or not to use the training images when testing.
        /// </summary>
        public bool UseTrainingImagesForTesting
        {
            get { return m_bUseTrainingImagesForTesting; }
            set { m_bUseTrainingImagesForTesting = value; }
        }

        private ImageSet loadImageset(string strType, SourceDescriptor src, WaitHandle[] rgAbort, ref SimpleDatum imgMean, out int nLastImageIdx, int nPadW = 0, int nPadH = 0, Log log = null, IMAGEDB_LOAD_METHOD loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL, int nImageDbLoadLimit = 0, int nImageDbLoadLimitStartIdx = 0, bool bLoadNext = false)
        {
            try
            {
                RawImageMean imgMeanRaw = null;

                m_factory.Open(src);
                nLastImageIdx = nImageDbLoadLimitStartIdx;

                if (loadMethod != IMAGEDB_LOAD_METHOD.LOAD_ALL)
                {
                    if (imgMean == null)
                    {
                        imgMeanRaw = m_factory.GetRawImageMean();
                        if (imgMeanRaw == null)
                        {
                            if (log != null)
                                log.WriteLine("WARNING: No image mean exists in the database, changing image database load from " + loadMethod.ToString() + " to " + IMAGEDB_LOAD_METHOD.LOAD_ALL.ToString());

                            loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
                        }
                    }
                }

                int nCount = src.ImageCount;
                if (nCount == 0)
                {
                    if (log != null)
                        log.WriteLine("WARNING: Could not find any images with " + strType + " Source = '" + src.Name + "'.  If this is a training dataset, you will need to enable the 'UseTrainingSrcForTesting' setting.");

                    if (loadMethod != IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND)
                    {
                        if (log != null)
                            log.WriteLine("Because there are no images in this set, the image loading method has been changed to LOAD_ON_DEMAND for this dataset.");

                        loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND;
                    }
                }

                ImageSet imgset = new ImageSet(m_factory, src, loadMethod, nImageDbLoadLimit);

                if (log != null && nCount > 0)
                    log.WriteLine("Loading '" + src.Name + "' - " + nCount.ToString("N0") + " images.");

                if (OnCalculateImageMean != null)
                    imgset.OnCalculateImageMean += OnCalculateImageMean;

                if (loadMethod != IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND)
                {
                    bool bDataIsReal = src.IsRealData;
                    int nBatchSize = 20000;
                    Stopwatch sw = new Stopwatch();

                    int nImageSize = src.ImageHeight * src.ImageWidth;
                    if (nImageSize > 60000)
                        nBatchSize = 5000;
                    else if (nBatchSize > 20000)
                        nBatchSize = 7500;
                    else if (nImageSize > 3000)
                        nBatchSize = 10000;

                    if (nImageDbLoadLimit <= 0)
                        nImageDbLoadLimit = nCount;

                    List<int> rgIdx = (nCount == 0) ? new List<int>() : getIndexList(nImageDbLoadLimitStartIdx, nImageDbLoadLimit);
                    int nIdx = 0;

                    sw.Start();

                    while (nIdx < rgIdx.Count)
                    {
                        int nImageIdx = rgIdx[nIdx];
                        int nImageCount = Math.Min(rgIdx.Count - nIdx, nBatchSize);

                        List<RawImage> rgImg = m_factory.GetRawImagesAt(nImageIdx, nImageCount);

                        for (int j = 0; j < rgImg.Count; j++)
                        {
                            SimpleDatum sd1 = m_factory.LoadDatum(rgImg[j], nPadW, nPadH);
                            imgset.Add(nIdx + j, sd1);

                            if (sw.Elapsed.TotalMilliseconds > 1000)
                            {
                                if (log != null)
                                {
                                    double dfPct = (double)(nIdx + j) / (double)nCount;
                                    log.Progress = dfPct;
                                    log.WriteLine("image loading at " + dfPct.ToString("P") + "...");
                                }

                                sw.Restart();

                                if (EventWaitHandle.WaitAny(rgAbort, 0) != EventWaitHandle.WaitTimeout)
                                    return null;
                            }
                        }

                        nIdx += rgImg.Count;

                        if (loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ALL && rgImg.Count == 0 && nIdx < nCount)
                        {
                            log.WriteLine("WARNING: Loaded " + nIdx.ToString("N0") + " images, yet " + (nCount - nIdx).ToString("N0") + " images are unaccounted for.  You may need to reindex the dataset.");
                            break;
                        }
                    }

                    if (log != null)
                        log.Progress = 0;

                    if (rgIdx.Count > 0)
                        nLastImageIdx = rgIdx[rgIdx.Count - 1] + 1;
                }
                else if (bLoadNext)
                {
                    nLastImageIdx += nImageDbLoadLimit;
                }

                if (imgMean == null)
                {
                    if (imgMeanRaw == null)
                        imgMeanRaw = m_factory.GetRawImageMean();

                    if (imgMeanRaw != null)
                        imgMean = m_factory.LoadDatum(imgMeanRaw, nPadW, nPadH);
                    else
                    {
                        if (log != null)
                            log.WriteLine("Calculating mean...");

                        imgMean = imgset.GetImageMean(log, rgAbort);
                        m_factory.PutRawImageMean(imgMean, true);
                    }
                }

                if (imgMean != null)
                    imgset.SetImageMean(imgMean);

                imgset.CompleteLoad(nLastImageIdx);

                return imgset;
            }
            finally
            {
                m_factory.Close();
            }
        }

        private List<int> getIndexList(int nStartIdx, int nCount)
        {
            List<int> rgIdx = new List<int>();

            for (int i = 0; i < nCount; i++)
            {
                rgIdx.Add(nStartIdx + i);
            }

            return rgIdx;
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
            if (m_TestingImages.SourceID != nSrcId &&
                m_TrainingImages.SourceID != nSrcId)
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
            if (m_TestingImages.SourceID != nSrcId &&
                m_TrainingImages.SourceID != nSrcId)
                return null;

            return m_factory.QueryImageMean(nSrcId);
        }

        /// <summary>
        /// Unload the images of the training and testing image sets.
        /// </summary>
        public void Unload()
        {
            lock (m_syncObj)
            {
                m_TestingImages.Unload();
                m_TrainingImages.Unload();
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
        /// Returns the ImageSet corresponding to a data source ID.
        /// </summary>
        /// <param name="nSourceID">Specifies the ID of the data source to use.</param>
        /// <returns>The ImageSet of images is returned.</returns>
        public ImageSet Find(int nSourceID)
        {
            if (m_TestingImages.SourceID == nSourceID)
            {
                if (m_bUseTrainingImagesForTesting)
                    return m_TrainingImages;

                return m_TestingImages;
            }

            if (m_TrainingImages.SourceID == nSourceID)
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
        public ImageSet Find(string strSource)
        {
            if (m_TestingImages.SourceName == strSource)
            {
                if (m_bUseTrainingImagesForTesting)
                    return m_TrainingImages;

                return m_TestingImages;
            }

            if (m_TrainingImages.SourceName == strSource)
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
