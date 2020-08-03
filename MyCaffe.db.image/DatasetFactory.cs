using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The DatasetFactory manages the connection to the Database object.
    /// </summary>
    public class DatasetFactory : IDisposable
    {
        /// <summary>
        /// Specifies the Database managed.
        /// </summary>
        protected Database m_db = new Database();
        /// <summary>
        /// Specifies the open source descriptor (if any).
        /// </summary>
        protected SourceDescriptor m_openSource = null;
        /// <summary>
        /// Specifies the original source ID (if any).
        /// </summary>
        protected int? m_nOriginalSourceID = null;
        /// <summary>
        /// Specifies whether or not to load the data criteria data if any exists.  When false, the data criteria data is not loaded from file (default = false).
        /// </summary>
        protected bool m_bLoadDataCriteria = false;
        /// <summary>
        /// Specifies whether or not to load the debug data if any exists.  When false, the debug data is not loaded from file (default = false).
        /// </summary>
        protected bool m_bLoadDebugData = false;

        ImageCache m_imageCache = null;
        ParamCache m_paramCache = null;


        /// <summary>
        /// The DatasetFactory constructor.
        /// </summary>
        public DatasetFactory()
        {
            m_db = new Database();
        }

        /// <summary>
        /// The DatasetFactory constructor.
        /// </summary>
        /// <param name="bLoadDataCriteria">Optionally, specifies to load the data criteria when loading images.</param>
        /// <param name="bLoadDebugData">Optionally, specifies to load the debug data when loading images.</param>
        public DatasetFactory(bool bLoadDataCriteria, bool bLoadDebugData)
        {
            m_bLoadDataCriteria = bLoadDataCriteria;
            m_bLoadDebugData = bLoadDebugData;
            m_db = new Database();
        }

        /// <summary>
        /// The DatasetFactory constructor.
        /// </summary>
        /// <param name="factory">Specifies the DatasetFactory to create this one from.</param>
        public DatasetFactory(DatasetFactory factory)
        {
            m_bLoadDebugData = factory.m_bLoadDebugData;
            m_bLoadDataCriteria = factory.m_bLoadDataCriteria;
        }

        /// <summary>
        /// The DatasetFactory constructor.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID of data source to open within the constructor.</param>
        public DatasetFactory(int nSrcId)
        {
            Open(nSrcId);
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        public void Dispose()
        {
        }

        /// <summary>
        /// Returns the index of the last image added to the database.
        /// </summary>
        public int LastIndex
        {
            get { return m_db.LastIndex; }
        }

        /// <summary>
        /// Sets the loading parameters that are used to determine which data to load with each image.
        /// </summary>
        /// <param name="bLoadDataCriteria">Specifies whether or not to load the data criteria data if any exists.  When false, the data criteria data is not loaded from file. (default = true).</param>
        /// <param name="bLoadDebugData">Specifies whether or not to load the debug data if any exists.  When false, the debug data is not loaded from file. (default = true).</param>
        public void SetLoadingParameters(bool bLoadDataCriteria, bool bLoadDebugData)
        {
            m_bLoadDataCriteria = bLoadDataCriteria;
            m_bLoadDebugData = bLoadDebugData;
        }

        /// <summary>
        /// Returns whether or not the image data criteria is to be loaded when loading each image.
        /// </summary>
        public bool LoadDataCriteria
        {
            get { return m_bLoadDataCriteria; }
        }

        /// <summary>
        /// Returns whether or not the image debug data is to be loaded when loading each image.
        /// </summary>
        public bool LoadDebugData
        {
            get { return m_bLoadDebugData; }
        }

        /// <summary>
        /// Change the data source ID on a raw image - currently only allowed on virtual raw images.
        /// </summary>
        /// <param name="nID">Specifies the raw image ID.</param>
        /// <param name="nNewSrcID">Specifies the ID of the new source.</param>
        /// <param name="bSave">Optionally, specifies whether or not to save the changes (default = true).</param>
        /// <returns>If the source ID is replaced, true is returned, otherwise false.</returns>
        public bool ChangeRawImageSourceID(int nID, int nNewSrcID, bool bSave = true)
        {
            return m_db.ChangeRawImageSourceID(nID, nNewSrcID, bSave);
        }

        /// <summary>
        /// Save the changes on the open data source.
        /// </summary>
        public void SaveChanges()
        {
            m_db.SaveChanges();
        }

        /// <summary>
        /// Open a given data source.
        /// </summary>
        /// <param name="src">Specifies the data source.</param>
        /// <param name="nCacheMax">Specifies the maximum cache count to use when adding RawImages (default = 500).</param>
        public void Open(SourceDescriptor src, int nCacheMax = 500)
        {
            m_openSource = src;
            Open(src.ID, nCacheMax);
        }

        /// <summary>
        /// Open a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source to use.</param>
        /// <param name="nCacheMax">Optionally, specifies the maximum cache count to use when adding RawImages (default = 500).</param>
        /// <param name="bForceLoadImageFilePath">Optionally, specfies to force load the image file path (default = <i>false</i>) and use file based data.</param>
        /// <param name="log">Optionally, specifies the output log (default = null).</param>
        public void Open(int nSrcId, int nCacheMax = 500, bool bForceLoadImageFilePath = false, Log log = null)
        {
            if (m_openSource != null)
            {
                if (m_openSource.ID != nSrcId)
                {
                    if (log != null)
                        log.WriteLine("WARNING: Closing currently open source ID of " + m_openSource.ID.ToString() + " and opening new Source ID of " + nSrcId.ToString() + ".");
                    Close();
                }
            }

            m_openSource = LoadSource(nSrcId);
            m_db.Open(nSrcId, bForceLoadImageFilePath);

            m_imageCache = new ImageCache(nCacheMax);
            m_paramCache = new ParamCache(nCacheMax);
        }

        /// <summary>
        /// Close the current data source used.
        /// </summary>
        public void Close()
        {
            m_openSource = null;
            m_db.Close();
        }

        /// <summary>
        /// Close and re-open the current data source used.
        /// </summary>
        public void Refresh()
        {
            m_db.Refresh();
        }

        /// <summary>
        /// Returns the currently open data source.
        /// </summary>
        public SourceDescriptor OpenSource
        {
            get { return m_openSource; }
        }


        //---------------------------------------------------------------------
        //  Raw Images
        //---------------------------------------------------------------------
        #region RawImages

        /// <summary>
        /// Add a new parameter to the parameter cache making sure to save once a maximum count is reached.
        /// </summary>
        /// <param name="nImageID">Specifies the image ID associated with the parameter.</param>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="strVal">Specifies the parameter value.</param>
        /// <param name="dfVal">Specifies the parameter numeric value.</param>
        /// <param name="rgData">Specifies the parameter data.</param>
        /// <param name="bOnlyAddNew">Specifies to only add the parameter if it does not already exist.</param>
        public void PutRawImageParameterCache(int nImageID, string strParam, string strVal, double? dfVal, byte[] rgData, bool bOnlyAddNew)
        {
            if (m_paramCache.Add(new ParameterData(strParam, strVal, dfVal, rgData, nImageID, bOnlyAddNew, m_openSource.ID)))
                ClearParamCache(true);
        }

        /// <summary>
        /// Clear the param cache and save when specified.
        /// </summary>
        /// <param name="bSave">Specifies to save the parameter values in the cache before clearing.</param>
        public void ClearParamCache(bool bSave)
        {
            if (m_paramCache.Count == 0)
                return;

            if (bSave)
                m_db.PutRawImageParameters(m_paramCache.Parameters);

            m_paramCache.Clear();
        }

        /// <summary>
        /// Add a SimpleDatum to the RawImage cache.
        /// </summary>
        /// <param name="nIdx">Specifies the RawImage index.</param>
        /// <param name="sd">Specifies the data.</param>
        /// <param name="strDescription">Optionally, specifies the description (default = null).</param>
        /// <param name="rgParams">Optionally, specifies a variable number of parameters to add to the RawImage.</param>
        public void PutRawImageCache(int nIdx, SimpleDatum sd, string strDescription = null, params ParameterData[] rgParams)
        {
            RawImage img = m_db.CreateRawImage(nIdx, sd, strDescription, m_nOriginalSourceID);

            if (m_imageCache.Add(img, sd, rgParams))
                ClearImageCashe(true);
        }

        /// <summary>
        /// Clear the RawImage cache and optionally save the images.
        /// </summary>
        /// <param name="bSave">When <i>true</i> the images in the cache are saved to the database in a bulk save, otherwise they are just flushed from the cache.</param>
        public void ClearImageCashe(bool bSave)
        {
            if (m_imageCache == null || m_imageCache.Count == 0)
                return;

            if (bSave)
                m_db.PutRawImages(m_imageCache.Images, m_imageCache.Parameters);

            m_imageCache.Clear();
        }

        /// <summary>
        /// Save a SimpleDatum to the database.
        /// </summary>
        /// <param name="nIdx">Specifies the RawImage index.</param>
        /// <param name="sd">Specifies the data.</param>
        /// <param name="strDescription">Optionally, specifies the description (default = null).</param>
        /// <returns>The ID of the saved RawImage is returned.</returns>
        public int PutRawImage(int nIdx, SimpleDatum sd, string strDescription = null)
        {
            return m_db.PutRawImage(nIdx, sd, strDescription);
        }

        /// <summary>
        /// Add a new or Set an existing RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter.</param>
        /// <param name="dfVal">Specifies the numeric value of the parameter (default = null).</param>
        /// <param name="rgData">Optionally, specifies raw data to associate with the RawImage (default = null).</param>
        /// <param name="bOnlyAddNew">Optionally, specifies to only add the parameter if it doesnt exist (default = false).</param>
        /// <returns>The ID of the parameter is returned.</returns>
        public int SetRawImageParameter(int nRawImageID, string strName, string strValue, double? dfVal = null, byte[] rgData = null, bool bOnlyAddNew = false)
        {
            return m_db.SetRawImageParameter(nRawImageID, strName, strValue, dfVal, rgData, true, bOnlyAddNew);
        }

        /// <summary>
        /// Set the RawImage parameter for all RawImages with the given time-stamp in the data source.
        /// </summary>
        /// <param name="dt">Specifies the time-stamp.</param>
        /// <param name="strName">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter as a string.</param>
        /// <param name="dfVal">Specifies the numeric value of the parameter (default = null).</param>
        /// <param name="rgData">Optionally, specifies the <i>byte</i> data associated with the parameter (default = null).</param>
        /// <returns>The ID of the RawImageParameter is returned.</returns>
        public int SetRawImageParameterAt(DateTime dt, string strName, string strValue, double? dfVal = null, byte[] rgData = null)
        {
            return m_db.SetRawImageParameterAt(dt, strName, strValue, dfVal, rgData);
        }

        /// <summary>
        /// Query a list of all raw image parameters of a give name stored with a given source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the source ID.</param>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <returns>The list of RawImageParameter values is returned.</returns>
        public List<RawImageParameter> QueryRawImageParameters(int nSrcId, string strName)
        {
            return m_db.QueryRawImageParameters(nSrcId, strName);
        }

        /// <summary>
        /// Query all image parameters for a given image.
        /// </summary>
        /// <param name="nImageID">Specifies the image ID who's image parameters are to be queried.</param>
        /// <returns>The list of any image parameters forund for the image are returned</returns>
        public List<RawImageParameter> QueryRawImageParameters(int nImageID)
        {
            return m_db.QueryRawImageParameters(nImageID);
        }

        /// <summary>
        /// Return the RawImageMean for the open data source.
        /// </summary>
        /// <returns>The RawImageMean is returned if found, otherwise <i>null</i> is returned.</returns>
        public RawImageMean GetRawImageMean()
        {
            return m_db.GetRawImageMean();
        }

        /// <summary>
        /// Returns a list of RawImages from the database for a data source.
        /// </summary>
        /// <param name="nImageIdx">Specifies the starting image index.</param>
        /// <param name="nImageCount">Specifies the number of images to retrieve from the starting index <i>nIdx</i>.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDescription">Optionally, specifies a description to filter the images retrieved (when specified, only images matching the filter are returned) (default = null).</param>
        /// <returns>The list of RawImage items is returned.</returns>
        public List<RawImage> GetRawImagesAt(int nImageIdx, int nImageCount, int nSrcId = 0, string strDescription = null)
        {
            return m_db.GetRawImagesAt(nImageIdx, nImageCount, nSrcId, strDescription);
        }

        /// <summary>
        /// Returns a list of RawImages from the database for a data source.
        /// </summary>
        /// <param name="rgImageIdx">Specifies the list of image indexes (no maximum).</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDescription">Optionally, specifies a description to filter the images retrieved (when specified, only images matching the filter are returned) (default = null).</param>
        /// <returns>The list of RawImage items is returned.</returns>
        public List<RawImage> GetRawImagesAt(List<int> rgImageIdx, ManualResetEvent evtCancel, int nSrcId = 0, string strDescription = null)
        {
            List<RawImage> rgImg = new List<RawImage>();

            while (rgImageIdx.Count > 0)
            {
                List<int> rgIdx = new List<int>();

                for (int i=0; i<rgImageIdx.Count && i < 100; i++)
                {
                    rgIdx.Add(rgImageIdx[i]);
                }

                List<RawImage> rgImg1 = m_db.GetRawImagesAt(rgIdx, nSrcId, strDescription);
                rgImg.AddRange(rgImg1);

                for (int i = 0; i < rgIdx.Count; i++)
                {
                    rgImageIdx.RemoveAt(0);
                }

                if (evtCancel.WaitOne(0))
                    return null;
            }

            return rgImg;
        }

        /// <summary>
        /// Returns a list of RawImages from the database for a data source.
        /// </summary>
        /// <param name="rgImgItems">Specifies the list of image DbItems.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDescription">Optionally, specifies a description to filter the images retrieved (when specified, only images matching the filter are returned) (default = null).</param>
        /// <returns>The list of RawImage items is returned.</returns>
        public List<RawImage> GetRawImagesAt(List<DbItem> rgImgItems, ManualResetEvent evtCancel, int nSrcId = 0, string strDescription = null)
        {
            List<RawImage> rgImg = new List<RawImage>();

            while (rgImgItems.Count > 0)
            {
                List<int> rgID = new List<int>();

                for (int i = 0; i < rgImgItems.Count && i < 100; i++)
                {
                    rgID.Add(rgImgItems[i].ID);
                }

                List<RawImage> rgImg1 = m_db.GetRawImagesAtID(rgID, nSrcId, strDescription);
                rgImg.AddRange(rgImg1);

                for (int i = 0; i < rgID.Count; i++)
                {
                    rgImgItems.RemoveAt(0);
                }

                if (evtCancel.WaitOne(0))
                    return null;
            }

            return rgImg;
        }

        /// <summary>
        /// Returns a list of SimpleDatum from the database for a data source.
        /// </summary>
        /// <param name="rgImageIdx">Specifies the list of image indexes (no maximum).</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDescription">Optionally, specifies a description to filter the images retrieved (when specified, only images matching the filter are returned) (default = null).</param>
        /// <returns>The list of SimpleDatum items is returned.</returns>
        public List<SimpleDatum> GetImagesAt(List<int> rgImageIdx, ManualResetEvent evtCancel, int nSrcId = 0, string strDescription = null)
        {
            List<RawImage> rgImg = GetRawImagesAt(rgImageIdx, evtCancel, nSrcId, strDescription);
            if (rgImg == null)
                return null;

            List<SimpleDatum> rgSd = new List<SimpleDatum>();

            foreach (RawImage img in rgImg)
            {
                rgSd.Add(LoadDatum(img));
            }

            return rgSd;
        }

        /// <summary>
        /// Returns a list of SimpleDatum from the database for a data source.
        /// </summary>
        /// <param name="rgImageItems">Specifies the list of image DbItems.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDescription">Optionally, specifies a description to filter the images retrieved (when specified, only images matching the filter are returned) (default = null).</param>
        /// <returns>The list of SimpleDatum items is returned.</returns>
        public List<SimpleDatum> GetImagesAt(List<DbItem> rgImageItems, ManualResetEvent evtCancel, int nSrcId = 0, string strDescription = null)
        {
            List<RawImage> rgImg = GetRawImagesAt(rgImageItems, evtCancel, nSrcId, strDescription);
            if (rgImg == null)
                return null;

            List<SimpleDatum> rgSd = new List<SimpleDatum>();

            foreach (RawImage img in rgImg)
            {
                rgSd.Add(LoadDatum(img));
            }

            return rgSd;
        }

        /// <summary>
        /// Returns the list of raw images that have a source ID from a selected list.
        /// </summary>
        /// <param name="nSrcId">Specifies the source ID.</param>
        /// <param name="bActive">Optionally, specifies to query active (or non active) images (default = <i>null</i>, which queries all images).</param>
        /// <param name="bLoadCriteria">Optionally, specifies to load the data criteria which can take longer (default = <i>false</i>).</param>
        /// <param name="bLoadDebug">Optionally, specifies to load the debug data which can take longer (default = <i>false</i>).</param>
        /// <param name="log">Optionally, specifies the output log (default = <i>null</i>).</param>
        /// <param name="evtCancel">Optionally, specifies the cancel event to abort loading (default = <i>null</i>).</param>
        /// <param name="nBoostVal">Optionally, specifies a boost value to query (default = 0, which ignores this filter).</param>
        /// <param name="bExactBoostVal">Optionally, specifies whether or not the boost value is an exact value or to be treated as a value greater than or equal to (default = false).</param>
        /// <returns>The list of RawImage's is returned.</returns>
        public List<RawImage> QueryRawImages(int nSrcId, bool? bActive = null, bool bLoadCriteria = false, bool bLoadDebug = false, Log log = null, CancelEvent evtCancel = null, int nBoostVal = 0, bool bExactBoostVal = false)
        {
            List<RawImage> rgImg = m_db.QueryRawImages(nSrcId, bActive, nBoostVal, bExactBoostVal);

            if (!bLoadCriteria && !bLoadDebug)
                return rgImg;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < rgImg.Count; i++)
            {
                int? nFmt;

                if (bLoadCriteria)
                    rgImg[i].DataCriteria = m_db.GetRawImageDataCriteria(rgImg[i], out nFmt);

                if (bLoadDebug)
                    rgImg[i].DebugData = m_db.GetRawImageDebugData(rgImg[i], out nFmt);

                if (evtCancel != null && evtCancel.WaitOne(0))
                    return null;

                if (log != null)
                {
                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        log.Progress = (double)i / (double)rgImg.Count;
                        log.WriteLine("loading " + i.ToString("N0") + " of " + rgImg.Count.ToString("N0") + "...");
                        sw.Restart();
                    }
                }
            }

            return rgImg;
        }

        /// <summary>
        /// Returns the list of raw images that have a source ID from a selected list.
        /// </summary>
        /// <param name="img">Specifies the Raw Image.</param>
        /// <param name="bLoadCriteria">Optionally, specifies to load the data criteria.</param>
        /// <param name="bLoadDebug">Optionally, specifies to load the debug data.</param>
        public void LoadRawImageData(RawImage img, bool bLoadCriteria, bool bLoadDebug)
        {
            int? nFmt;

            if (bLoadCriteria)
                img.DataCriteria = m_db.GetRawImageDataCriteria(img, out nFmt);

            if (bLoadDebug)
                img.DebugData = m_db.GetRawImageDebugData(img, out nFmt);
        }

        /// <summary>
        /// Save the SimpleDatum as a RawImageMean in the database for the open data source.
        /// </summary>
        /// <param name="sd">Specifies the data.</param>
        /// <param name="bUpdate">Specifies whether or not to update the mean image.</param>
        /// <returns>The ID of the RawImageMean is returned.</returns>
        public int PutRawImageMean(SimpleDatum sd, bool bUpdate)
        {
            return m_db.PutRawImageMean(sd, bUpdate);
        }

        /// <summary>
        /// Save the SimpleDatum as a RawImageMean in the database.
        /// </summary>
        /// <param name="sd">Specifies the data.</param>
        /// <param name="bUpdate">Specifies whether or not to update the mean image.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The ID of the RawImageMean is returned.</returns>
        public bool SaveImageMean(SimpleDatum sd, bool bUpdate, int nSrcId = 0)
        {
            if (m_db.PutRawImageMean(sd, bUpdate, nSrcId) != 0)
                return true;

            return false;
        }

        /// <summary>
        /// Return the SimpleDatum for the image mean from the open data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The SimpleDatum is returned if found, otherwise <i>null</i> is returned.</returns>
        public SimpleDatum QueryImageMean(int nSrcId = 0)
        {
            RawImageMean img = m_db.GetRawImageMean(nSrcId);
            return LoadDatum(img);
        }

        /// <summary>
        /// Copy the raw image mean from one source to another.
        /// </summary>
        /// <param name="strSrcSrc">Specifies the Data Source with the source image mean to copy.</param>
        /// <param name="strDstSrc">Specifies the Data Source with the destination image mean where the source is copied to.</param>
        /// <returns>On success, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool CopyImageMean(string strSrcSrc, string strDstSrc)
        {
            int nSrcIdSrc = GetSourceID(strSrcSrc);
            int nDstIdSrc = GetSourceID(strDstSrc);
            return m_db.CopyImageMean(nSrcIdSrc, nDstIdSrc);
        }

        /// <summary>
        /// Returns the number of images in the database for the open data source.
        /// </summary>
        /// <returns></returns>
        public int GetImageCount()
        {
            return m_db.GetImageCount();
        }

        /// <summary>
        /// Returns all raw image IDs for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nMax">Optionally, specifies the maximum number of ID's to query (default = int.MaxValue).</param>
        /// <param name="nLabel">Optionally, specifies a label from which images are to be queried (default = -1, which ignores this parameter).</param>
        /// <param name="nBoost">Optionally, specifies a boost from which images are to be queried (default = -1, which ignores this parameter).</param>
        /// <param name="bBoostIsExact">Optionally, specifies whether the boost value is exact (<i>true</i>) or the minimum boost where all values equal are greater are retrieved (<i>false</i>).  Default = false.</param>
        /// <returns>The list of raw image ID's is returned.</returns>
        public List<int> QueryRawImageIDs(int nSrcId = 0, int nMax = int.MaxValue, int nLabel = -1, int nBoost = -1, bool bBoostIsExact = false)
        {
            return m_db.QueryAllRawImageIDs(nSrcId, nMax, nLabel, nBoost, bBoostIsExact);
        }

        /// <summary>
        /// Returns the raw image ID for the image mean associated with a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The raw image ID is returned.</returns>
        public int GetRawImageMeanID(int nSrcId = 0)
        {
            RawImageMean img = m_db.GetRawImageMean(nSrcId);
            return (img == null) ? 0 : img.ID;
        }

        /// <summary>
        /// Returns the RawImage ID for the image with the given time-stamp. 
        /// </summary>
        /// <param name="dt">Specifies the image time-stamp.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The ID of the RawImage is returned.</returns>
        public int GetRawImageID(DateTime dt, int nSrcId = 0)
        {
            return m_db.GetRawImageID(dt, nSrcId);
        }

        /// <summary>
        /// Returns the raw image with a specified image ID.
        /// </summary>
        /// <param name="nImageID">Specifies the image ID of the image to retrieve.</param>
        /// <returns>The raw image is returned.</returns>
        public RawImage GetRawImageFromID(int nImageID)
        {
            return m_db.GetRawImage(nImageID);
        }

        /// <summary>
        /// Delete all RawImageResults for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteRawImageResults(int nSrcId = 0)
        {
            m_db.DeleteRawImageResults(nSrcId);
        }

        /// <summary>
        /// Save the results of a Run as a RawImageResult.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nIdx">Specifies the index of the result.</param>
        /// <param name="nLabel">Specifies the expected label of the result.</param>
        /// <param name="dt">Specifies the time-stamp of the result.</param>
        /// <param name="rgResults">Specifies the results of the run as a list of (int nLabel, double dfReult) values.</param>
        /// <param name="bInvert">Specifies whether or not the results are inverted.</param>
        /// <returns></returns>
        public int PutRawImageResults(int nSrcId, int nIdx, int nLabel, DateTime dt, List<Result> rgResults, bool bInvert)
        {
            return m_db.PutRawImageResults(nSrcId, nIdx, nLabel, dt, rgResults, bInvert);
        }

        /// <summary>
        /// Adds a new RawImage group to the database.
        /// </summary>
        /// <param name="img">Specifies an image associated with the group.</param>
        /// <param name="nIdx">Specifies an index associated with the group.</param>
        /// <param name="dtStart">Specifies the start time stamp for the group.</param>
        /// <param name="dtEnd">Specifies the end time stamp for the group.</param>
        /// <param name="rgProperties">Specifies the properties of the group.</param>
        /// <returns>The ID of the RawImageGroup is returned.</returns>
        public int AddRawImageGroup(Image img, int nIdx, DateTime dtStart, DateTime dtEnd, List<double> rgProperties)
        {
            return m_db.AddRawImageGroup(img, nIdx, dtStart, dtEnd, rgProperties);
        }

        /// <summary>
        /// Searches fro the RawImageGroup ID.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the raw image group.</param>
        /// <param name="dtStart">Specifies the start time-stamp of the image group.</param>
        /// <param name="dtEnd">Specifies the end time-stamp of the image group.</param>
        /// <returns>If found, the RawImageGroup ID is returned, otherwise 0 is returned.</returns>
        public int FindRawImageGroupID(int nIdx, DateTime dtStart, DateTime dtEnd)
        {
            return m_db.FindRawImageGroupID(nIdx, dtStart, dtEnd);
        }

        /// <summary>
        /// Returns a list of distinct RawImage parameter descriptions for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The list of distinct descriptions is returned.</returns>
        public List<string> GetRawImageDistinctParameterDescriptions(int nSrcId)
        {
            return m_db.GetRawImageDistinctParameterDescriptions(nSrcId);
        }

        /// <summary>
        /// Return the <i>byte</i> array data of a RawImage parameter.
        /// </summary>
        /// <param name="nRawImageID">Specifies the ID of the RawImage.</param>
        /// <param name="strParam">Specifies the name of the parameter.</param>
        /// <returns>The parameter <i>byte</i> array data is returned.</returns>
        public byte[] GetRawImageParameterData(int nRawImageID, string strParam)
        {
            return m_db.GetRawImageParameterData(nRawImageID, strParam);
        }

        /// <summary>
        /// Returns the RawImage parameter count for a data source.
        /// </summary>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strType">Optionally, specifies the parameter type (default = "TEXT").</param>
        /// <returns>The number of RawImage parameters is returned.</returns>
        public int GetRawImageParameterCount(string strParam, int nSrcId = 0, string strType = "TEXT")
        {
            return m_db.GetRawImageParameterCount(strParam, nSrcId, strType);
        }

        /// <summary>
        /// Returns whether or not a given RawImage parameter exists.
        /// </summary>
        /// <param name="strName">Specifies the parameter name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strType">Optionally, specifies the parameter type (default = "TEXT").</param>
        /// <returns>Returns <i>true</i> if the parameter exists, <i>false</i> otherwise.</returns>
        public bool GetRawImageParameterExist(string strName, int nSrcId = 0, string strType = "TEXT")
        {
            return m_db.GetRawImageParameterExist(strName, nSrcId, strType);
        }

        /// <summary>
        /// Update the active label on a given raw image.
        /// </summary>
        /// <param name="nImageID">Specifies the raw image ID.</param>
        /// <param name="nNewActiveLabel">Specifies the new active label.</param>
        /// <param name="bActivate">Optionally, specifies whether or not to activate/deactivate the image.</param>
        /// <param name="bSaveChanges">Optionally, save the changes if any.</param>
        /// <returns>If the image is updated this function returns <i>true</i>, otherwise it returns <i>false</i>.</returns>
        public bool UpdateActiveLabel(int nImageID, int nNewActiveLabel, bool bActivate = true, bool bSaveChanges = true)
        {
            return m_db.UpdateActiveLabel(nImageID, nNewActiveLabel, bActivate, bSaveChanges);
        }

        /// <summary>
        /// Directly update the active label and activate the image with the specified ID.
        /// </summary>
        /// <param name="nID">Specifies the image ID.</param>
        /// <param name="nLabel">Specifies the new active label.</param>
        public void UpdateActiveLabelDirect(int nID, int nLabel)
        {
            m_db.UpdateActiveLabelDirect(nID, nLabel);
        }

        /// <summary>
        /// Directly update all active labels and activate all of the images for the open Source ID.
        /// </summary>
        /// <param name="nLabel">Specifies the new active label.</param>
        public void UpdateAllActiveLabelsDirect(int nLabel)
        {
            m_db.UpdateAllActiveLabels(m_openSource.ID, nLabel);
        }

        /// <summary>
        /// Update the active label on a given raw image by its index.
        /// </summary>
        /// <param name="nIdx">Specifies the raw image index.</param>
        /// <param name="nNewActiveLabel">Specifies the new active label.</param>
        public void UpdateActiveLabelByIndex(int nIdx, int nNewActiveLabel)
        {
            m_db.UpdateActiveLabelByIndex(m_openSource.ID, nIdx, nNewActiveLabel);
        }

        /// <summary>
        /// Activate/deactivate a raw image based on its index.
        /// </summary>
        /// <param name="nIdx">Specifies the raw image index.</param>
        /// <param name="bActive">Specifies the new active state of the image to set.</param>
        public void ActivateRawImageByIndex(int nIdx, bool bActive)
        {
            m_db.ActivateRawImageByIndex(m_openSource.ID, nIdx, bActive);
        }

        /// <summary>
        /// Change the source ID on an image to another source ID.
        /// </summary>
        /// <param name="nImageID">Specifies the ID of the image to update.</param>
        /// <param name="nSrcID">Specifies the new source ID.</param>
        public void UpdateRawImageSourceID(int nImageID, int nSrcID)
        {
            m_db.UpdateRawImageSourceID(nImageID, nSrcID);
        }

        /// <summary>
        /// Activate/Deactivate a given image.
        /// </summary>
        /// <param name="nImageID">Specifies the ID of the image to activate/deactivate.</param>
        /// <param name="bActive">Specifies whether to activate (<i>true</i>) or deactivate (<i>false</i>) the image.</param>
        /// <param name="bSave">Specifies whether or not to save the changes (when false, calling SaveChanges() is needed).</param>
        /// <returns>If the active state is changed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool ActivateRawImage(int nImageID, bool bActive, bool bSave = true)
        {
            return m_db.ActivateRawImage(nImageID, bActive, bSave);
        }

        /// <summary>
        /// Activates all images with the given source ID's.
        /// </summary>
        /// <param name="bActive">Specifies whether or not to activate the images.</param>
        /// <param name="rgSrcId">Specifies the source ID's who's images are to be activated.</param>
        public void ActivateAllRawImages(bool bActive, params int[] rgSrcId)
        {
            m_db.ActivateAllRawImages(bActive, rgSrcId);
        }

        /// <summary>
        /// The FixupRawImageCopy function is used to fix errors in the copy source ID of a copied
        /// raw image.  For original images, this function does nothing.
        /// </summary>
        /// <remarks>
        /// When creating a copy of a Data Source that uses both training and testing Data Sources (e.g., 
        /// re-arranging the time period used for training vs testing), it is important that the 
        /// OriginalSourceID be set with the Data Source ID that holds the data file.
        /// </remarks>
        /// <param name="nImageID">Specifies the ID of the raw image to fixup.</param>
        /// <param name="nSecondarySrcId">Specifies the secondary Source ID to use if the data file is not found.</param>
        public void FixupRawImageCopy(int nImageID, int nSecondarySrcId)
        {
            m_db.FixupRawImageCopy(nImageID, nSecondarySrcId);
        }

        /// <summary>
        /// Converts the raw image data criteria data which may be stored as a path to the underlying data file, to the actual data.
        /// </summary>
        /// <param name="rgData">Specifies the raw data, which may contain an image path.</param>
        /// <param name="nOriginalSourceID">Specifies the original source ID that stores the image.</param>
        /// <returns>The actual raw data is returned.</returns>
        public byte[] GetRawImageDataCriteria(byte[] rgData, int nOriginalSourceID)
        {
            return m_db.GetRawImageDataCriteria(rgData, nOriginalSourceID);
        }

        /// <summary>
        /// Converts the raw image debug data which may be stored as a path to the underlying data file, to the actual data.
        /// </summary>
        /// <param name="rgData">Specifies the raw data, which may contain an image path.</param>
        /// <param name="nOriginalSourceID">Specifies the original source ID that stores the image.</param>
        /// <returns>The actual raw data is returned.</returns>
        public byte[] GetRawImageDebugData(byte[] rgData, int nOriginalSourceID)
        {
            return m_db.GetRawImageDebugData(rgData, nOriginalSourceID);
        }

        #endregion


        //---------------------------------------------------------------------
        //  Labels
        //---------------------------------------------------------------------
        #region Labels

        /// <summary>
        /// Returns a list of all label boosts set on a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of a project.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>A list of LabelBoostDescriptors is returned.</returns>
        public List<LabelBoostDescriptor> GetLabelBoosts(int nProjectId, int nSrcId = 0)
        {
            List<LabelBoost> rgBoosts = m_db.GetLabelBoosts(nProjectId, true, nSrcId);
            List<LabelBoostDescriptor> rgDesc = new List<LabelBoostDescriptor>();

            foreach (LabelBoost b in rgBoosts)
            {
                rgDesc.Add(new LabelBoostDescriptor(b.ActiveLabel.GetValueOrDefault(), (double)b.Boost.GetValueOrDefault(1)));
            }

            return rgDesc;
        }

        /// <summary>
        /// Saves a label mapping in the database for a data source.
        /// </summary>
        /// <param name="map">Specifies the label mapping.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void SetLabelMapping(LabelMapping map, int nSrcId = 0)
        {
            m_db.SetLabelMapping(map, nSrcId);
        }

        /// <summary>
        /// Update a label mapping in the database for a data source.
        /// </summary>
        /// <param name="nNewLabel">Specifies the new label.</param>
        /// <param name="rgOriginalLabels">Specifies the original labels that are to be mapped to the new label.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void UpdateLabelMapping(int nNewLabel, List<int> rgOriginalLabels, int nSrcId = 0)
        {
            m_db.UpdateLabelMapping(nNewLabel, rgOriginalLabels, nSrcId);
        }

        /// <summary>
        /// Resets all labels back to their original labels for a project.
        /// </summary>
        /// <param name="nProjectId">Optionally, specifies the ID of a project (default = 0).</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void ResetLabels(int nProjectId = 0, int nSrcId = 0)
        {
            m_db.ResetLabels(nProjectId, nSrcId);
        }

        /// <summary>
        /// Delete all label boosts for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of a project.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteLabelBoosts(int nProjectId, int nSrcId = 0)
        {
            m_db.DeleteLabelBoosts(nProjectId, nSrcId);
        }

        /// <summary>
        /// Add a label boost to the database for a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project for which the label boost is to be added.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the boost.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void AddLabelBoost(int nProjectId, int nLabel, double dfBoost, int nSrcId = 0)
        {
            m_db.AddLabelBoost(nProjectId, nLabel, dfBoost, nSrcId);
        }

        /// <summary>
        /// Returns the Label boosts as a string.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of a project.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="bSort">Optionally, specifies whether or not to sort the labels by active label (default = true).</param>
        /// <returns></returns>
        public string GetLabelBoostsAsText(int nProjectId, int nSrcId = 0, bool bSort = true)
        {
            return m_db.GetLabelBoostsAsText(nProjectId, nSrcId, bSort);
        }

        /// <summary>
        /// Load the label counts from the database for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>A dictionary containing (int nLabel, int nCount) pairs is returned.</returns>
        public Dictionary<int, int> LoadLabelCounts(int nSrcId = 0)
        {
            Dictionary<int, int> rgCounts = new Dictionary<int, int>();
            m_db.LoadLabelCounts(rgCounts, nSrcId);
            return rgCounts;
        }

        /// <summary>
        /// Updates the label counts in the database for the open data source.
        /// </summary>
        /// <param name="rgCounts">Specifies a dictionary containing (int nLabel, int nCount) pairs.</param>
        public void UpdateLabelCounts(Dictionary<int, int> rgCounts)
        {
            m_db.UpdateLabelCounts(rgCounts);
        }

        /// <summary>
        /// Update the label counts for a given data source and project (optionally) by querying the database for the actual counts.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="nProjectId">Optionally, specifies the ID of a project to use (default = 0).</param>
        public void UpdateLabelCounts(int nSrcId = 0, int nProjectId = 0)
        {
            m_db.UpdateLabelCounts(nSrcId, nProjectId);
        }

        /// <summary>
        /// Returns the label counts for a given data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>A string containing the label counts is returned.</returns>
        public string GetLabelCountsAsText(int nSrcId)
        {
            return m_db.GetLabelCountsAsText(nSrcId);
        }

        /// <summary>
        /// Update the name of a label.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="strName">Specifies the new name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void UpdateLabelName(int nLabel, string strName, int nSrcId = 0)
        {
            m_db.UpdateLabelName(nLabel, strName, nSrcId);
        }

        /// <summary>
        /// Get the Label name of a label within a data source.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>When found, the Label is returned, otherwise <i>null</i> is returned.</returns>
        public string GetLabelName(int nLabel, int nSrcId = 0)
        {
            return m_db.GetLabelName(nLabel, nSrcId);
        }

        /// <summary>
        /// Add a label to the database for a data source.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="strName">Optionally, specifies a label name (default = "").</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The ID of the added label is returned.</returns>
        public int AddLabel(int nLabel, string strName, int nSrcId = 0)
        {
            return m_db.AddLabel(nLabel, strName, nSrcId);
        }

        /// <summary>
        /// Delete the labels of a data source from the database.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteLabels(int nSrcId = 0)
        {
            m_db.DeleteLabels(nSrcId);
        }

        /// <summary>
        /// Activate (or deactivate) the labels specified for each of the source ID's specified.
        /// </summary>
        /// <param name="rgLabels">Specifies the labels.</param>
        /// <param name="bActive">Specifies whether to activate (<i>true</i>) or deactivate (<i>false</i>) the labels.</param>
        /// <param name="rgSrcId">Specifies the source ID's who's labels are to be activated.</param>
        public void ActivateLabels(List<int> rgLabels, bool bActive, params int[] rgSrcId)
        {
            m_db.ActivateLabels(rgLabels, bActive, rgSrcId);
        }

        /// <summary>
        /// Update the label and boost for a given search target criteria.
        /// </summary>
        /// <param name="nTgtLbl">Specifies the target label to replace, or null to ignore.</param>
        /// <param name="bTgtLblExact">When a target label is specified, this parameter specifies whether to treat the target label as an exact value (true) for a minimum value (false).</param>
        /// <param name="nTgtBst">Specifies the target boost to replace, or null to ignore.</param>
        /// <param name="bTgtBstExact">When a target boost is specified, this parameter specifies whether to treat the target boost as an exact value (true) for a minimum value (false).</param>
        /// <param name="nNewLbl">Specifies the new label, or null to ignore.</param>
        /// <param name="nNewBst">Specifies the new boost, or null to ignore.</param>
        /// <param name="rgSrcId">Specifies the SourceID's on which to alter the label and/or boost.</param>
        public void UpdateLabelBoost(int? nTgtLbl, bool bTgtLblExact, int? nTgtBst, bool bTgtBstExact, int? nNewLbl, int? nNewBst, params int[] rgSrcId)
        {
            m_db.UpdateLabelBoost(nTgtLbl, bTgtLblExact, nTgtBst, bTgtBstExact, nNewLbl, nNewBst, rgSrcId);
        }

        #endregion


        //---------------------------------------------------------------------
        //  Sources
        //---------------------------------------------------------------------
        #region Sources

        /// <summary>
        /// Returns the ID of a data source given its name.
        /// </summary>
        /// <param name="strName">Specifies the data source name.</param>
        /// <returns>The ID of the data source is returned.</returns>
        public int GetSourceID(string strName)
        {
            return m_db.GetSourceID(strName);
        }

        /// <summary>
        /// Returns the name of a data source given its ID.
        /// </summary>
        /// <param name="nId">Specifies the ID of the data source.</param>
        /// <returns>The data source name is returned.</returns>
        public string GetSourceName(int nId)
        {
            return m_db.GetSourceName(nId);
        }

        /// <summary>
        /// Adds a new data source to the database.
        /// </summary>
        /// <param name="src">Specifies source desciptor to add.</param>
        /// <returns>The ID of the data source added is returned.</returns>
        public int AddSource(SourceDescriptor src)
        {
            src.ID = m_db.AddSource(src.Name, src.ImageChannels, src.ImageWidth, src.ImageHeight, src.IsRealData, src.CopyOfSourceID, src.SaveImagesToFile);
            return src.ID;
        }

        /// <summary>
        /// Adds a new data source to the database.
        /// </summary>
        /// <param name="strName">Specifies the data source name.</param>
        /// <param name="nChannels">Specifies the number of channels per item.</param>
        /// <param name="nWidth">Specifies the width of each item.</param>
        /// <param name="nHeight">Specifies the height of each item.</param>
        /// <param name="bDataIsReal">Specifies whether or not the item uses real or <i>byte</i> data.</param>
        /// <param name="nCopyOfSourceID">Optionally, specifies the ID of the source from which this source was copied (and has virtual raw image references).  The default 
        /// of 0 specifies that this is an original source.</param>
        /// <param name="bSaveImagesToFile">Optionally, specifies whether or not to save the images to the file system (<i>true</i>) or directly into the database (<i>false</i>).  The default is <i>true</i>.</param>
        /// <returns>The ID of the data source added is returned.</returns>
        public int AddSource(string strName, int nChannels, int nWidth, int nHeight, bool bDataIsReal, int nCopyOfSourceID = 0, bool bSaveImagesToFile = true)
        {
            return m_db.AddSource(strName, nChannels, nWidth, nHeight, bDataIsReal, nCopyOfSourceID, bSaveImagesToFile);
        }

        /// <summary>
        /// Delete the list of data sources, listed by name, from the database.
        /// </summary>
        /// <param name="rgstrSrc">Specifies the list of data sources.</param>
        public void DeleteSources(params string[] rgstrSrc)
        {
            m_db.DeleteSources(rgstrSrc);
        }

        /// <summary>
        /// Delete the data source data (images, means, results and parameters) from the database.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteSourceData(int nSrcId = 0)
        {
            m_db.DeleteSourceData(nSrcId);
        }

        /// <summary>
        /// Return the number of boosted images for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strFilterVal">Optionally, specifies a parameter filtering value (default = <i>null</i>).</param>
        /// <param name="nBoostVal">Optionally, specifies a boost filtering value (default = <i>null</i>).</param>
        /// <returns>The number of boosted images is returned.</returns>
        public int GetBoostCount(int nSrcId = 0, string strFilterVal = null, int? nBoostVal = null)
        {
            return m_db.GetBoostCount(nSrcId, strFilterVal, nBoostVal);
        }

        /// <summary>
        /// Activate the images that meet the filtering criteria in the Data Source.  If no filtering criteria is set, all images are activated.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strFilterVal">Optionally, specifies a parameter filtering value (default = <i>null</i>).</param>
        /// <param name="nBoostVal">Optionally, specifies a boost filtering value (default = <i>null</i>).</param>
        /// <returns>The number of activated images is returned.</returns>
        public int ActivateFiltered(int nSrcId = 0, string strFilterVal = null, int? nBoostVal = null)
        {
            return m_db.ActivateFiltered(nSrcId, strFilterVal, nBoostVal);
        }

        /// <summary>
        /// Reindex the RawImages of a data source.
        /// </summary>
        /// <param name="log">Specifies the Log to use for status output.</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the operation.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="bCreateImageMean">Optionally, specifies whether or not to create (or recreate) the image mean (default = false).</param>
        /// <returns>Upon completion <i>true</i> is returned, otherwise <i>false</i> is returned when cancelled.</returns>
        public bool ReindexRawImages(Log log, CancelEvent evtCancel, int nSrcId = 0, bool bCreateImageMean = false)
        {
            List<RawImage> rgImg = m_db.ReindexRawImages(log, evtCancel, nSrcId);
            if (rgImg == null)
                return false;

            if (bCreateImageMean)
            {
                bool bOpened = false;
                if (m_openSource == null)
                {
                    Open(nSrcId);
                    bOpened = true;
                }

                List<SimpleDatum> rgSd = new List<SimpleDatum>();
                foreach (RawImage img in rgImg)
                {
                    rgSd.Add(LoadDatum(img));
                }

                SimpleDatum sdMean = SimpleDatum.CalculateMean(log, rgSd.ToArray(), evtCancel.Handles);
                if (sdMean == null)
                    return false;

                bool bRes = SaveImageMean(sdMean, true, nSrcId);
                if (bOpened)
                    Close();

                return bRes;
            }

            return true;
        }

        /// <summary>
        /// Updates a data source.
        /// </summary>
        /// <param name="nChannels">Specifies the number of channels per item.</param>
        /// <param name="nWidth">Specifies the width of each item.</param>
        /// <param name="nHeight">Specifies the height of each item.</param>
        /// <param name="bDataIsReal">Specifies whether or not the item uses real or <i>byte</i> data.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void UpdateSource(int nChannels, int nWidth, int nHeight, bool bDataIsReal, int nSrcId = 0)
        {
            m_db.UpdateSource(nChannels, nWidth, nHeight, bDataIsReal, nSrcId);
        }

        /// <summary>
        /// Saves the label cache, updates the label counts from the database and then updates the source counts from the database.
        /// </summary>
        public void UpdateSourceCounts()
        {
            m_db.SaveLabelCache();
            m_db.UpdateLabelCounts();
            m_db.UpdateSourceCounts();
        }

        /// <summary>
        /// Set the value of a data source parameter.
        /// </summary>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="strValue">Specifies the value of the parameter.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void SetSourceParameter(string strParam, string strValue, int nSrcId = 0)
        {
            m_db.SetSourceParameter(strParam, strValue, nSrcId);
        }

        /// <summary>
        /// Return the data source parameter as a string.
        /// </summary>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as a string.</returns>
        public string GetSourceParameter(string strParam, int nSrcId = 0)
        {
            return m_db.GetSourceParameter(strParam, nSrcId);
        }

        /// <summary>
        /// Return the data source parameter as an <i>int</i>.
        /// </summary>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="nDefault">Specifies the default value returned if the parameter is not found.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as an <i>int</i>.</returns>
        public int GetSourceParameter(string strParam, int nDefault, int nSrcId = 0)
        {
            return m_db.GetSourceParameter(strParam, nDefault, nSrcId);
        }

        /// <summary>
        /// Return the data source parameter as a <i>bool</i>.
        /// </summary>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="bDefault">Specifies the default value returned if the parameter is not found.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as a <i>bool</i>.</returns>
        public bool GetSourceParameter(string strParam, bool bDefault, int nSrcId = 0)
        {
            return m_db.GetSourceParameter(strParam, bDefault, nSrcId);
        }

        /// <summary>
        /// Return the data source parameter as a <i>double</i>.
        /// </summary>
        /// <param name="strParam">Specifies the parameter name.</param>
        /// <param name="dfDefault">Specifies the default value returned if the parameter is not found.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>The parameter value is returned as a <i>double</i>.</returns>
        public double GetSourceParameter(string strParam, double dfDefault, int nSrcId = 0)
        {
            return m_db.GetSourceParameter(strParam, dfDefault, nSrcId);
        }

        /// <summary>
        /// Returns the first time-stamp in the data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetFirstTimeStamp(int nSrcId = 0, string strDesc = null)
        {
            return m_db.GetFirstTimeStamp(nSrcId, strDesc);
        }

        /// <summary>
        /// Returns the last time-stamp in the data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetLastTimeStamp(int nSrcId = 0, string strDesc = null)
        {
            return m_db.GetLastTimeStamp(nSrcId, strDesc);
        }

        /// <summary>
        /// Returns the last time-stamp in the data source from within a time period.
        /// </summary>
        /// <param name="dtStart">Specifies the start of the time range.</param>
        /// <param name="dtEnd">Specifies the end of the time range.</param>
        /// <param name="bEndInclusive">Specifies whether or not to include the end time in the range.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetLastTimeStamp(DateTime dtStart, DateTime dtEnd, bool bEndInclusive, int nSrcId = 0, string strDesc = null)
        {
            return m_db.GetLastTimeStamp(dtStart, dtEnd, bEndInclusive, nSrcId, strDesc);
        }


        /// <summary>
        /// Returns the last time-stamp and index in the data source.
        /// </summary>
        /// <param name="nIndex">Specifies the index of the last item.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetLastTimeStamp(out int nIndex, int nSrcId = 0, string strDesc = null)
        {
            return m_db.GetLastTimeStamp(out nIndex, nSrcId, strDesc);
        }

        /// <summary>
        /// Returns the last time stamp within a given time range.
        /// </summary>
        /// <param name="dtStart">Specifies the start of the time range.</param>
        /// <param name="dtEnd">Specifies the end of the time range.</param>
        /// <param name="bEndInclusive">Specifies whether or not to include the end time in the range.</param>
        /// <param name="nIndex">Returns the index of the last item.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="strDesc">Optionally, specifies a description to filter the values with (default = null, no filter).</param>
        /// <returns>If found, the time-stamp is returned, otherwise, DateTime.MinValue is returned.</returns>
        public DateTime GetLastTimeStamp(DateTime dtStart, DateTime dtEnd, bool bEndInclusive, out int nIndex, int nSrcId = 0, string strDesc = null)
        {
            return m_db.GetLastTimeStamp(dtStart, dtEnd, bEndInclusive, out nIndex);
        }

#pragma warning disable 1591

        public List<int> GetAllDataSourceIDs() /** @private */
        {
            return m_db.GetAllDataSourcesIDs();
        }

        public bool ConvertRawImagesSaveToFile(int nIdx, int nCount) /** @private */
        {
            return m_db.ConvertRawImagesSaveToFile(nIdx, nCount);
        }

        public bool ConvertRawImagesSaveToDatabase(int nIdx, int nCount) /** @private */
        {
            return m_db.ConvertRawImagesSaveToDatabase(nIdx, nCount);
        }

        public void UpdateSaveImagesToFile(bool bSaveToFile, int nSrcId = 0) /** @private */
        {
            m_db.UpdateSaveImagesToFile(bSaveToFile, nSrcId);
        }

#pragma warning restore 1591

        /// <summary>
        /// Get/set the original source ID (if any).  This field is used when copying a source and using the virutal image reference,
        /// but retaining the original source ID for the internal image lookup.
        /// </summary>
        public int? OriginalSourceID
        {
            get { return m_nOriginalSourceID; }
            set { m_nOriginalSourceID = value; }
        }

        #endregion


        //---------------------------------------------------------------------
        //  Datasets
        //---------------------------------------------------------------------
        #region Datasets

        /// <summary>
        /// Returns a datasets ID given its name.
        /// </summary>
        /// <param name="strDsName">Specifies the dataset name.</param>
        /// <returns>The ID of the dataset is returned.</returns>
        public int GetDatasetID(string strDsName)
        {
            return m_db.GetDatasetID(strDsName);
        }

        /// <summary>
        /// Returns the name of a dataset given its ID.
        /// </summary>
        /// <param name="nId">Specifies the dataset ID.</param>
        /// <returns>The dataset name is returned.</returns>
        public string GetDatasetName(int nId)
        {
            return m_db.GetDatasetName(nId);
        }

        /// <summary>
        /// Adds or updates the training source, testing source, dataset creator and dataset to the database.
        /// </summary>
        /// <param name="ds"></param>
        /// <returns></returns>
        public int AddDataset(DatasetDescriptor ds)
        {
            ds.TestingSource.ID = AddSource(ds.TestingSource);
            ds.TrainingSource.ID = AddSource(ds.TrainingSource);

            int nDsCreatorID = 0;
            if (ds.CreatorName != null)
                nDsCreatorID = m_db.GetDatasetCreatorID(ds.CreatorName);

            ds.ID = m_db.AddDataset(nDsCreatorID, ds.Name, ds.TestingSource.ID, ds.TrainingSource.ID, ds.DatasetGroup.ID, ds.ModelGroup.ID);
            return ds.ID;
        }

        /// <summary>
        /// Add a new (or update an existing if exists) dataset to the database.
        /// </summary>
        /// <param name="nDsCreatorID">Specifies the ID of the creator.</param>
        /// <param name="strName">Specifies the name of the dataset.</param>
        /// <param name="nTestSrcId">Specifies the ID of the testing data source.</param>
        /// <param name="nTrainSrcId">Specifies the ID of the training data source.</param>
        /// <param name="nDsGroupID">Optionally, specifies the ID of the dataset group (default = 0).</param>
        /// <param name="nModelGroupID">Optionally, specifies the ID of the model group (default = 0).</param>
        /// <returns></returns>
        public int AddDataset(int nDsCreatorID, string strName, int nTestSrcId, int nTrainSrcId, int nDsGroupID = 0, int nModelGroupID = 0)
        {
            return m_db.AddDataset(nDsCreatorID, strName, nTestSrcId, nTrainSrcId, nDsGroupID, nModelGroupID);
        }

        /// <summary>
        /// Update the description of a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset to update.</param>
        /// <param name="strDesc">Specifies the new description.</param>
        public void UpdateDatasetDescription(int nDsId, string strDesc)
        {
            m_db.UpdateDatasetDescription(nDsId, strDesc);
        }

        /// <summary>
        /// Updates the dataset counts, and training/testing source counts.
        /// </summary>
        /// <param name="nDsId"></param>
        public void UpdateDatasetCounts(int nDsId)
        {
            Dataset ds = m_db.GetDataset(nDsId);
            Database db = new Database();

            db.Open(ds.TestingSourceID.GetValueOrDefault());
            db.UpdateSourceCounts();
            db.UpdateLabelCounts();
            db.Close();

            db.Open(ds.TrainingSourceID.GetValueOrDefault());
            db.UpdateSourceCounts();
            db.UpdateLabelCounts();
            db.Close();

            m_db.UpdateDatasetCounts(nDsId);
        }

        /// <summary>
        /// Updates the dataset counts for a set of datasets.
        /// </summary>
        /// <param name="evtCancel">Specifies a cancel event used to abort the process.</param>
        /// <param name="log">Specifies the Log used for status output.</param>
        /// <param name="nDatasetCreatorID">Specifies the ID of the dataset creator.</param>
        /// <param name="rgstrDs">Specifies a list of the dataset names to update.</param>
        /// <param name="strParamNameForDescription">Specifies the parameter name used for descriptions.</param>
        public void UpdateDatasetCounts(CancelEvent evtCancel, Log log, int nDatasetCreatorID, List<string> rgstrDs, string strParamNameForDescription)
        {
            m_db.UpdateDatasetCounts(evtCancel, log, nDatasetCreatorID, rgstrDs, strParamNameForDescription);
        }

        /// <summary>
        /// Searches for the data set name based on the training and testing source names.
        /// </summary>
        /// <param name="strTrainSrc">Specifies the data source name for training.</param>
        /// <param name="strTestSrc">Specifies the data source name for testing.</param>
        /// <returns>If found, the dataset name is returned, otherwise <i>null</i> is returned.</returns>
        public string FindDatasetNameFromSourceName(string strTrainSrc, string strTestSrc)
        {
            return m_db.FindDatasetNameFromSourceName(strTrainSrc, strTestSrc);
        }

        /// <summary>
        /// Reset all dataset relabel flags with a given creator.
        /// </summary>
        /// <param name="nDsCreatorId">Specifies the ID of the dataset creator.</param>
        public void ResetAllDatasetRelabelWithCreator(int nDsCreatorId)
        {
            m_db.ResetAllDatasetRelabelWithCreator(nDsCreatorId);
        }

        /// <summary>
        /// Update the dataset relabel flag for a dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="bRelabel">Specifies the re-label flag.</param>
        public void UpdateDatasetRelabel(int nDsId, bool bRelabel)
        {
            m_db.UpdateDatasetRelabel(nDsId, bRelabel);
        }

        /// <summary>
        /// Adds a new parameter or Sets the value of an existing dataset parameter.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strParam">Specifies the name of the parameter.</param>
        /// <param name="strValue">Specifies the value of the parameter.</param>
        public void SetDatasetParameter(int nDsId, string strParam, string strValue)
        {
            m_db.SetDatasetParameter(nDsId, strParam, strValue);
        }

        /// <summary>
        /// Returns the value of a dataset parameter as a string.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strParam">Specifies the name of the parameter.</param>
        /// <returns>If the parameter is found it is returned as a string, otherwise <i>null</i> is returned.</returns>
        public string GetDatasetParameter(int nDsId, string strParam)
        {
            return m_db.GetDatasetParameter(nDsId, strParam);
        }

        /// <summary>
        /// Returns the value of a dataset parameter as an <i>int</i>.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strParam">Specifies the name of the parameter.</param>
        /// <param name="nDefault">Specifies the default value to return if not found.</param>
        /// <returns>If the parameter is found it is returned as an <i>int</i>, otherwise the default value is returned.</returns>
        public int GetDatasetParameter(int nDsId, string strParam, int nDefault)
        {
            return m_db.GetDatasetParameter(nDsId, strParam, nDefault);
        }

        /// <summary>
        /// Returns the value of a dataset parameter as a <i>bool</i>.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strParam">Specifies the name of the parameter.</param>
        /// <param name="bDefault">Specifies the default value to return if not found.</param>
        /// <returns>If the parameter is found it is returned as a <i>bool</i>, otherwise the default value is returned.</returns>
        public bool GetDatasetParameter(int nDsId, string strParam, bool bDefault)
        {
            return m_db.GetDatasetParameter(nDsId, strParam, bDefault);
        }

        /// <summary>
        /// Returns the value of a dataset parameter as a <i>double</i>.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <param name="strParam">Specifies the name of the parameter.</param>
        /// <param name="dfDefault">Specifies the default value to return if not found.</param>
        /// <returns>If the parameter is found it is returned as a <i>double</i>, otherwise the default value is returned.</returns>
        public double GetDatasetParameter(int nDsId, string strParam, double dfDefault)
        {
            return m_db.GetDatasetParameter(nDsId, strParam, dfDefault);
        }

        /// <summary>
        /// Returns the ID of a dataset group given its name.
        /// </summary>
        /// <param name="strName">Specifies the name of the group.</param>
        /// <returns>Returns the ID of the dataset group, or 0 if not found.</returns>
        public int GetDatasetGroupID(string strName)
        {
            return m_db.GetDatasetGroupID(strName);
        }

        #endregion


        //---------------------------------------------------------------------
        //  Loading Descriptors
        //---------------------------------------------------------------------
        #region Loading Descriptors

        /// <summary>
        /// Returns a list of the image indexes of all boosted images in the Data Source.
        /// </summary>
        /// <param name="bBoostedOnly">Specifies to only return the indexes of boosted images.</param>
        /// <param name="bActiveOnly">Optionally, specifies to query active images only (default = true).</param>
        /// <returns>The list of DbItem's is returned where each DbItem contains the image index, label, and boost.</returns>
        public List<DbItem> LoadImageIndexes(bool bBoostedOnly, bool bActiveOnly = true)
        {
            return m_db.GetAllRawImageIndexes(bBoostedOnly, bActiveOnly);
        }

        /// <summary>
        /// Load the image descriptors for a set of given source ID's.
        /// </summary>
        /// <param name="evtCancel">Optionally specifies to cancel the load, when <i>null</i> this parameter is ignored.</param>
        /// <param name="rgSrcId">Specifies the source ID's to load.</param>
        /// <returns>A list of image descriptors for each image is returned.</returns>
        public List<ImageDescriptor> LoadImages(CancelEvent evtCancel, params int[] rgSrcId)
        {
            if (rgSrcId.Length == 0)
                throw new Exception("You must specify at least one source ID.");

            List<ImageDescriptor> rgImgDesc = new List<ImageDescriptor>();
            List<RawImage> rgImg = m_db.QueryRawImages(rgSrcId);

            foreach (RawImage img in rgImg)
            {
                if (evtCancel != null && evtCancel.WaitOne(0))
                    return null;

                rgImgDesc.Add(new ImageDescriptor(img.ID,
                                                  img.Height.GetValueOrDefault(0),
                                                  img.Width.GetValueOrDefault(0),
                                                  img.Channels.GetValueOrDefault(0),
                                                  img.Encoded.GetValueOrDefault(false),
                                                  img.SourceID.GetValueOrDefault(0),
                                                  img.Idx.GetValueOrDefault(0),
                                                  img.ActiveLabel.GetValueOrDefault(0),
                                                  img.Active.GetValueOrDefault(false),
                                                  img.Description,
                                                  img.TimeStamp.GetValueOrDefault(DateTime.MinValue)));
            }

            return rgImgDesc;
        }

        /// <summary>
        /// Loads a new SimpleDataum from a RawImage ID.
        /// </summary>
        /// <param name="nImageId">Specifies the RawImage ID.</param>
        /// <param name="nChannels">Specifies the number of channels.</param>
        /// <param name="bDataIsReal">Specifies whether or not the data contains real or <i>byte</i> data.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>A new SimpleDatum is returned containing the image.</returns>
        public SimpleDatum LoadDatum(int nImageId, int nChannels, bool bDataIsReal, int nLabel, int nSrcId = 0)
        {
            RawImage img = m_db.GetRawImage(nImageId);

            if (img == null)
                return null;

            if (nSrcId == 0)
                nSrcId = m_openSource.ID;

            if (img.SourceID != nSrcId)
                throw new Exception("The source of the raw image with ID = " + nImageId.ToString() + " (SourceID = " + img.SourceID.ToString() + ") does not match the expected source id of " + nSrcId.ToString());

            if (img.Channels != nChannels)
                throw new Exception("The raw image with ID = " + nImageId.ToString() + " has a different channel count than the expected count of " + nChannels.ToString());

            if (img.Encoded.GetValueOrDefault() != bDataIsReal)
                throw new Exception("The raw image with ID = " + nImageId.ToString() + " does not have a matching encoding scheme as that of the expected encoded = " + bDataIsReal.ToString());

            if (img.ActiveLabel != (int)nLabel)
                throw new Exception("The raw image with ID = " + nImageId.ToString() + " does not have a label equal to the expected label of " + nLabel.ToString());

            return new Datum(LoadDatum(img));
        }

        /// <summary>
        /// Loads a new SimpleDatum from a RawImage.
        /// </summary>
        /// <param name="img">Specifies the RawImage.</param>
        /// <param name="nPadW">Optionally, specifies a pad to apply to the width (default = 0).</param>
        /// <param name="nPadH">Optionally, specifies a pad to apply to the height (default = 0).</param>
        /// <returns>A new SimpleDatum is returned containing the image.</returns>
        public SimpleDatum LoadDatum(RawImage img, int nPadW = 0, int nPadH = 0)
        {
            if (img == null)
                return null;

            byte[] rgDataCriteria = null;
            int? nDataCriteriaFormatId = null;
            byte[] rgDebugData = null;
            int? nDebugDataFormatId = null;
            byte[] rgData = m_db.GetRawImageData(img, m_bLoadDataCriteria, m_bLoadDebugData, out rgDataCriteria, out nDataCriteriaFormatId, out rgDebugData, out nDebugDataFormatId);

            // Annotation Data is used as the label, so it must be loaded.
            if (!m_bLoadDataCriteria &&
                (nDataCriteriaFormatId.GetValueOrDefault(0) == (int)SimpleDatum.DATA_FORMAT.SEGMENTATION ||
                 nDataCriteriaFormatId.GetValueOrDefault(0) == (int)SimpleDatum.DATA_FORMAT.ANNOTATION_DATA))
            {
                LoadRawImageData(img, true, false);
                rgDataCriteria = img.DataCriteria;
            }

            int nHeight = img.Height.GetValueOrDefault();
            int nWidth = img.Width.GetValueOrDefault();
            int nChannels = img.Channels.GetValueOrDefault();
            SimpleDatum sd = null;

            if (img.Encoded.GetValueOrDefault())
            {
                Tuple<double[], float[]> rgRealData = SimpleDatum.GetRealData(rgData, nPadW, nPadH, nHeight, nWidth, nChannels);
                double[] rgDataFloat = rgRealData.Item1;
                float[] rgDataDouble = rgRealData.Item2;

                if (rgDataFloat != null)
                {
                    sd = new SimpleDatum(img.Encoded.GetValueOrDefault(),
                                           nChannels,
                                           nWidth + nPadW,
                                           nHeight + nPadH,
                                           img.ActiveLabel.GetValueOrDefault(),
                                           img.TimeStamp.GetValueOrDefault(),
                                           rgDataFloat,
                                           img.ActiveBoost.GetValueOrDefault(),
                                           img.AutoLabel.GetValueOrDefault(),
                                           img.Idx.GetValueOrDefault(),
                                           img.VirtualID.GetValueOrDefault(),
                                           img.ID,
                                           img.SourceID.GetValueOrDefault(),
                                           img.OriginalSourceID.GetValueOrDefault());
                }
                else
                {
                    sd = new SimpleDatum(img.Encoded.GetValueOrDefault(),
                                           nChannels,
                                           nWidth + nPadW,
                                           nHeight + nPadH,
                                           img.ActiveLabel.GetValueOrDefault(),
                                           img.TimeStamp.GetValueOrDefault(),
                                           rgDataDouble,
                                           img.ActiveBoost.GetValueOrDefault(),
                                           img.AutoLabel.GetValueOrDefault(),
                                           img.Idx.GetValueOrDefault(),
                                           img.VirtualID.GetValueOrDefault(),
                                           img.ID,
                                           img.SourceID.GetValueOrDefault(),
                                           img.OriginalSourceID.GetValueOrDefault());
                }
            }
            else
            {
                List<byte> rgDataBytes = new List<byte>(SimpleDatum.GetByteData(rgData, nPadW, nPadH, nHeight, nWidth, nChannels));
                sd = new SimpleDatum(img.Encoded.GetValueOrDefault(),
                                   nChannels,
                                   nWidth + nPadW,
                                   nHeight + nPadH,
                                   img.ActiveLabel.GetValueOrDefault(),
                                   img.TimeStamp.GetValueOrDefault(),
                                   rgDataBytes,                                   
                                   img.ActiveBoost.GetValueOrDefault(),
                                   img.AutoLabel.GetValueOrDefault(),
                                   img.Idx.GetValueOrDefault(),
                                   img.VirtualID.GetValueOrDefault(),
                                   img.ID,
                                   img.SourceID.GetValueOrDefault(),
                                   img.OriginalSourceID.GetValueOrDefault());
            }

            sd.OriginalLabel = img.OriginalLabel.GetValueOrDefault();
            sd.Description = img.Description;
            sd.GroupID = img.GroupID.GetValueOrDefault();
            sd.DataCriteria = rgDataCriteria;
            sd.DataCriteriaFormat = (SimpleDatum.DATA_FORMAT)nDataCriteriaFormatId.GetValueOrDefault(0);
            sd.DebugData = rgDebugData;
            sd.DebugDataFormat = (SimpleDatum.DATA_FORMAT)nDebugDataFormatId.GetValueOrDefault(0);

            sd.LoadAnnotationDataFromDataCriteria();

            return sd;
        }

        /// <summary>
        /// Loads a new image mean from a RawImageMean.
        /// </summary>
        /// <param name="img">Specifies the RawImageMean.</param>
        /// <param name="nPadW">Optionally, specifies a pad to apply to the width (default = 0).</param>
        /// <param name="nPadH">Optionally, specifies a pad to apply to the height (default = 0).</param>
        /// <returns>A new SimpleDatum is returned containing the image mean.</returns>
        public SimpleDatum LoadDatum(RawImageMean img, int nPadW = 0, int nPadH = 0)
        {
            if (img == null)
                return null;

            int nHeight = img.Height.GetValueOrDefault();
            int nWidth = img.Width.GetValueOrDefault();
            int nChannels = img.Channels.GetValueOrDefault();
            SimpleDatum sd;

            if (img.Encoded.GetValueOrDefault())
            {
                Tuple<double[], float[]> rgRealData = SimpleDatum.GetRealData(img.Data, nPadW, nPadH, nHeight, nWidth, nChannels);
                double[] rgDataFloat = rgRealData.Item1;
                float[] rgDataDouble = rgRealData.Item2;

                if (rgDataFloat != null)
                {
                    sd = new SimpleDatum(img.Encoded.GetValueOrDefault(),
                                                  nChannels,
                                                  nWidth + nPadW,
                                                  nHeight + nPadH,
                                                  0,
                                                  DateTime.MinValue,
                                                  rgDataFloat,
                                                  0,
                                                  false,
                                                  0,
                                                  0,
                                                  img.ID,
                                                  img.SourceID.GetValueOrDefault());
                }
                else
                {
                    sd = new SimpleDatum(img.Encoded.GetValueOrDefault(),
                                                  nChannels,
                                                  nWidth + nPadW,
                                                  nHeight + nPadH,
                                                  0,
                                                  DateTime.MinValue,
                                                  rgDataDouble,
                                                  0,
                                                  false,
                                                  0,
                                                  0,
                                                  img.ID,
                                                  img.SourceID.GetValueOrDefault());
                }
            }
            else
            {
                List<byte> rgDataBytes = new List<byte>(SimpleDatum.GetByteData(img.Data, nPadW, nPadH, nHeight, nWidth, nChannels));
                sd = new SimpleDatum(img.Encoded.GetValueOrDefault(),
                                              nChannels,
                                              nWidth + nPadW,
                                              nHeight + nPadH,
                                              0,
                                              DateTime.MinValue,
                                              rgDataBytes,
                                              0,
                                              false,
                                              0,
                                              0,
                                              img.ID,
                                              img.SourceID.GetValueOrDefault());
            }

            return sd;
        }

        /// <summary>
        /// Returns the image at a given image ID.
        /// </summary>
        /// <param name="nImageId">Specifies the image ID within the database.</param>
        /// <param name="nSrcId">Optionally, specifies the expected Source ID.  The default is 0, which specifies to use the open Source ID.</param>
        /// <returns>The SimpleDatum containing the image is returned.</returns>
        public SimpleDatum LoadImage(int nImageId, int nSrcId = 0)
        {
            if (m_db.CurrentSource == null)
                throw new Exception("You must open a data source first!");

            if (nSrcId == 0)
                nSrcId = m_db.CurrentSource.ID;

            RawImage img = m_db.GetRawImage(nImageId);
            if (img == null || img.SourceID != nSrcId)
                return null;

            return LoadDatum(img);
        }

        /// <summary>
        /// Returns the image mean for a give data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the source data ID.</param>
        /// <returns>The image mean is returned in a SimpleDatum.</returns>
        public SimpleDatum LoadImageMean(int nSrcId)
        {
            RawImageMean imgMean = m_db.GetRawImageMean(nSrcId);
            if (imgMean == null)
                return null;

            return LoadDatum(imgMean);
        }

        /// <summary>
        /// Load an image at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="bLoadDataCriteria">Optionally, specifies to load the data criteria data (default = null, which uses the default of false).</param>
        /// <param name="bLoadDebugData">Optionally, specifies to load the debug data (default = null, which uses the default of false).</param>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="nPadW">Optionally, specifies a pad to apply to the width (default = 0).</param>
        /// <param name="nPadH">Optionally, specifies a pad to apply to the height (default = 0).</param>
        /// <returns>A new SimpleDatum is returned containing the image.</returns>
        public SimpleDatum LoadImageAt(int nIdx, bool? bLoadDataCriteria = null, bool? bLoadDebugData = null, int nSrcId = 0, int nPadW = 0, int nPadH = 0)
        {            
            RawImage img = m_db.GetRawImageAt(nIdx, nSrcId);
            if (img == null)
                return null;

            if (!bLoadDataCriteria.HasValue)
                bLoadDataCriteria = m_bLoadDataCriteria;

            if (bLoadDataCriteria.Value && img.DataCriteria != null)
                img.DataCriteria = m_db.GetRawImageDataCriteria(img.DataCriteria);

            if (!bLoadDebugData.HasValue)
                bLoadDebugData = m_bLoadDebugData;

            if (bLoadDebugData.Value && img.DebugData != null)
                img.DebugData = m_db.GetRawImageDebugData(img.DebugData);

            return LoadDatum(img, nPadW, nPadH);
        }

        /// <summary>
        /// Load the data criteria and/or debug data.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum to load.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        public void LoadRawData(SimpleDatum sd, bool bLoadDataCriteria, bool bLoadDebugData)
        {
            if (bLoadDataCriteria && sd.DataCriteria != null)
                sd.DataCriteria = m_db.GetRawImageDataCriteria(sd.DataCriteria);

            if (bLoadDebugData && sd.DebugData != null)
                sd.DebugData = m_db.GetRawImageDebugData(sd.DebugData);
        }

        /// <summary>
        /// Load a list of LabelDescriptors for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <param name="bSort">Specifies whether or not to sort the labels (default = false).</param>
        /// <param name="bWithImagesOnly">Specifies whether or not to only load labels with images associated with them (default = false).</param>
        /// <returns></returns>
        public List<LabelDescriptor> LoadLabels(int nSrcId = 0, bool bSort = true, bool bWithImagesOnly = false)
        {
            List<Label> rgLabels = m_db.GetLabels(bSort, bWithImagesOnly, nSrcId);
            List<LabelDescriptor> rgDesc = new List<LabelDescriptor>();

            foreach (Label l in rgLabels)
            {
                LabelDescriptor label = new LabelDescriptor(l.Label1.GetValueOrDefault(),
                                                            l.ActiveLabel.GetValueOrDefault(),
                                                            l.Name,
                                                            l.ImageCount.GetValueOrDefault());
                rgDesc.Add(label);
            }

            return rgDesc;
        }

        /// <summary>
        /// Load the source descriptor from a data source name.
        /// </summary>
        /// <param name="strSource">Specifies the Id of the data source.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        public SourceDescriptor LoadSource(string strSource)
        {
            return LoadSource(m_db.GetSourceID(strSource));
        }

        /// <summary>
        /// Load the source descriptor from a data source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the name of the data source.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        public SourceDescriptor LoadSource(int nSrcId)
        {
            Source src = m_db.GetSource(nSrcId);

            if (src == null)
                return null;

            SourceDescriptor srcDesc = new SourceDescriptor(src.ID,
                                                            src.Name,
                                                            src.ImageHeight.GetValueOrDefault(),
                                                            src.ImageWidth.GetValueOrDefault(),
                                                            src.ImageChannels.GetValueOrDefault(),
                                                            src.ImageEncoded.GetValueOrDefault(),
                                                            src.SaveImagesToFile.GetValueOrDefault(),
                                                            src.CopyOfSourceID.GetValueOrDefault(0),
                                                            src.OwnerID,
                                                            src.ImageCount.GetValueOrDefault());
            srcDesc.Labels = LoadLabels(nSrcId);
            srcDesc.LabelCountsAsText = m_db.GetLabelCountsAsText(nSrcId);
            srcDesc.SetInactiveImageCount(m_db.GetImageCount(nSrcId, false, true));
            srcDesc.Parameters = LoadSourceParameters(srcDesc.ID);

            return srcDesc;
        }

        /// <summary>
        /// Loads the data source parameters for a given source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The collection of data source parameters is returned.</returns>
        public ParameterDescriptorCollection LoadSourceParameters(int nSrcId)
        {
            ParameterDescriptorCollection col = new ParameterDescriptorCollection();

            Dictionary<string, string> rgParam = m_db.GetSourceParameters(nSrcId);
            foreach (KeyValuePair<string, string> kv in rgParam)
            {
                col.Add(new ParameterDescriptor(0, kv.Key, kv.Value));
            }

            return col;
        }

        /// <summary>
        /// Load a dataset group descriptor from a group ID.
        /// </summary>
        /// <param name="nGroupId">Specifies the ID of the group.</param>
        /// <returns>The GroupDescriptor is returned.</returns>
        public GroupDescriptor LoadDatasetGroup(int nGroupId)
        {
            DatasetGroup grp = m_db.GetDatasetGroup(nGroupId);

            if (grp == null)
                return null;

            return new GroupDescriptor(nGroupId, grp.Name, grp.OwnerID);
        }

        /// <summary>
        /// Load a model group descriptor from a group ID.
        /// </summary>
        /// <param name="nGroupId">Specifies the ID of the group.</param>
        /// <returns>The GroupDescriptor is returned.</returns>
        public GroupDescriptor LoadModelGroup(int nGroupId)
        {
            ModelGroup grp = m_db.GetModelGroup(nGroupId);

            if (grp == null)
                return null;

            return new GroupDescriptor(nGroupId, grp.Name, grp.OwnerID);
        }

        /// <summary>
        /// Load a dataset descriptor from a dataset name.
        /// </summary>
        /// <param name="strDataset">Specifies the dataset name.</param>
        /// <returns>The DatasetDescriptor is returned.</returns>
        public DatasetDescriptor LoadDataset(string strDataset)
        {
            return LoadDataset(m_db.GetDatasetID(strDataset));
        }

        /// <summary>
        /// Load a dataset descriptor from a dataset ID.
        /// </summary>
        /// <param name="nDatasetID">Specifies the dataset ID.</param>
        /// <returns>The DatasetDescriptor is returned.</returns>
        public DatasetDescriptor LoadDataset(int nDatasetID)
        {
            Dataset ds = m_db.GetDataset(nDatasetID);
            if (ds == null)
                return null;

            return loadDataset(ds);
        }

        /// <summary>
        /// Load a dataset descriptor from a dataset ID or name, where the ID is tried first and name second.
        /// </summary>
        /// <param name="nDatasetID">Specifies the dataset ID.</param>
        /// <param name="strDataset">Specifies the dataset name.</param>
        /// <returns>The DatasetDescriptor is returned.</returns>
        public DatasetDescriptor LoadDataset(int nDatasetID, string strDataset)
        {
            DatasetDescriptor ds = null;

            if (nDatasetID != 0)
                ds = LoadDataset(nDatasetID);

            if (ds == null && !String.IsNullOrEmpty(strDataset))
                ds = LoadDataset(strDataset);

            return ds;
        }

        /// <summary>
        /// Load the dataset descriptor that contains the testing and training data source names.
        /// </summary>
        /// <param name="strTestingSrc">Specifies the testing data source name.</param>
        /// <param name="strTrainingSrc">Specifies the training data source name.</param>
        /// <returns>The DatasetDescriptor is returned.</returns>
        public DatasetDescriptor LoadDataset(string strTestingSrc, string strTrainingSrc)
        {
            Dataset ds = m_db.GetDataset(strTestingSrc, strTrainingSrc);
            if (ds == null)
                return null;

            return loadDataset(ds);
        }

        /// <summary>
        /// Loads all dataset descriptors within a group that have a creator.
        /// </summary>
        /// <param name="nGroupId">Specifies the ID of the dataset group.</param>
        /// <returns>A list of DatasetDescriptors is returned.</returns>
        public List<DatasetDescriptor> LoadAllDatasetsWithCreators(int nGroupId)
        {
            List<Dataset> rgDs = m_db.GetAllDatasetsWithCreators(nGroupId);
            List<DatasetDescriptor> rgDesc = new List<DatasetDescriptor>();

            foreach (Dataset ds in rgDs)
            {
                rgDesc.Add(loadDataset(ds));
            }

            return rgDesc;
        }

        /// <summary>
        /// Loads all dataset descriptors with a given dataset creator.
        /// </summary>
        /// <param name="nCreatorID">Specifies the ID of the dataset creator.</param>
        /// <param name="bRelabeled">Optionally, specifies whether or not only re-labeled datasets should be returned.</param>
        /// <returns>A list of DatasetDescriptors is returned.</returns>
        public List<DatasetDescriptor> LoadAllDatasetsWithCreator(int nCreatorID, bool? bRelabeled = null)
        {
            List<Dataset> rgDs = m_db.GetAllDatasetsWithCreator(nCreatorID, bRelabeled);
            List<DatasetDescriptor> rgDesc = new List<DatasetDescriptor>();

            foreach (Dataset ds in rgDs)
            {
                rgDesc.Add(LoadDataset(ds.ID));
            }

            return rgDesc;
        }

        /// <summary>
        /// Loads a dataset creator from a Dataset entity.
        /// </summary>
        /// <param name="ds">Specifies the Dataset entity.</param>
        /// <returns>The DatasetDescriptor is returned.</returns>
        protected virtual DatasetDescriptor loadDataset(Dataset ds)
        {
            SourceDescriptor srcTrain = LoadSource(ds.TrainingSourceID.GetValueOrDefault());
            SourceDescriptor srcTest = LoadSource(ds.TestingSourceID.GetValueOrDefault());
            GroupDescriptor dsGroup = LoadDatasetGroup(ds.DatasetGroupID.GetValueOrDefault());
            GroupDescriptor mdlGroup = LoadModelGroup(ds.ModelGroupID.GetValueOrDefault());
            DatasetDescriptor dsDesc = new DatasetDescriptor(ds.ID, ds.Name, mdlGroup, dsGroup, srcTrain, srcTest, m_db.GetDatasetCreatorName(ds.DatasetCreatorID.GetValueOrDefault()), ds.OwnerID, ds.Description);

            dsDesc.Parameters = LoadDatasetParameters(ds.ID);

            return dsDesc;
        }

        /// <summary>
        /// Loads the dataset parameters for a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset.</param>
        /// <returns>The collection of dataset parameters is returned.</returns>
        public ParameterDescriptorCollection LoadDatasetParameters(int nDsId)
        {
            ParameterDescriptorCollection col = new ParameterDescriptorCollection();

            Dictionary<string, string> rgParam = m_db.GetDatasetParameters(nDsId);
            foreach (KeyValuePair<string, string> kv in rgParam)
            {
                col.Add(new ParameterDescriptor(0, kv.Key, kv.Value));
            }

            return col;
        }

        /// <summary>
        /// Loads a list of RawImage results for a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>A list of the ResultDescriptors is returned.</returns>
        public List<ResultDescriptor> LoadRawImageResults(int nSrcId = 0)
        {
            List<RawImageResult> rgResults = m_db.GetRawImageResults(nSrcId);
            List<ResultDescriptor> rgDesc = new List<ResultDescriptor>();

            foreach (RawImageResult res in rgResults)
            {
                rgDesc.Add(new ResultDescriptor(res.ID, null, null, res.Idx.GetValueOrDefault(), res.Label.GetValueOrDefault(), res.ResultCount.GetValueOrDefault(), res.Results, res.SourceID.GetValueOrDefault(), res.TimeStamp.GetValueOrDefault()));
            }

            return rgDesc;
        }

        #endregion
    }

    class ParamCache /** @private */
    {
        List<ParameterData> m_rgParam = new List<ParameterData>();
        int m_nMax;

        public ParamCache(int nMax)
        {
            m_nMax = nMax;
        }

        public int Count
        {
            get { return m_rgParam.Count; }
        }

        public void Clear()
        {
            m_rgParam.Clear();
        }

        public bool Add(ParameterData p)
        {
            m_rgParam.Add(p);

            if (m_rgParam.Count == m_nMax)
                return true;

            return false;
        }

        public List<ParameterData> Parameters
        {
            get { return m_rgParam; }
        }
    }

    class ImageCache /** @private */
    {
        List<RawImage> m_rgImages = new List<RawImage>();
        List<List<ParameterData>> m_rgrgParams = new List<List<ParameterData>>();
        int m_nMax;

        public ImageCache(int nMax)
        {
            m_nMax = nMax;
        }

        public int Count
        {
            get { return m_rgImages.Count; }
        }

        public void Clear()
        {
            m_rgImages.Clear();
            m_rgrgParams.Clear();
        }

        public bool Add(RawImage img, SimpleDatum sd, params ParameterData[] rgParams)
        {
            m_rgImages.Add(img);

            List<ParameterData> rgParam = new List<db.image.ParameterData>();

            if (sd is Datum)
            {
                Datum d = sd as Datum;
                string strVal = d.Tag as string;
                string strTag = d.TagName as string;

                if (!String.IsNullOrEmpty(strTag) && !String.IsNullOrEmpty(strVal))
                    rgParam.Add(new db.image.ParameterData(strTag, strVal, null, null, img.ID, false, img.SourceID.GetValueOrDefault()));
            }

            foreach (ParameterData param in rgParams)
            {
                param.ImageID = img.ID;
                param.SourceID = img.SourceID.GetValueOrDefault();
                rgParam.Add(param);
            }

            m_rgrgParams.Add(rgParam);

            if (m_rgImages.Count == m_nMax)
                return true;

            return false;
        }

        public List<RawImage> Images
        {
            get { return m_rgImages; }
        }

        public List<List<ParameterData>> Parameters
        {
            get { return m_rgrgParams; }
        }
    }

    /// <summary>
    /// The LabelBoostDescriptor class describes a label boost.
    /// </summary>
    public class LabelBoostDescriptor
    {
        int m_nLabel;
        double m_dfBoost;

        /// <summary>
        /// The LabelBoostDescriptor constructor.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the boost.</param>
        public LabelBoostDescriptor(int nLabel, double dfBoost)
        {
            m_nLabel = nLabel;
            m_dfBoost = dfBoost;
        }

        /// <summary>
        /// Returns the label.
        /// </summary>
        public int Label
        {
            get { return m_nLabel; }
        }

        /// <summary>
        /// Returns the boost.
        /// </summary>
        public double Boost
        {
            get { return m_dfBoost; }
        }
    }
}
