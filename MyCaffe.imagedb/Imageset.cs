using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using MyCaffe.basecode;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.imagedb
{
    /// <summary>
    /// The ImageSet class contains the list of image for a data source as well as a list of LabelSets that map into it.
    /// </summary>
    public class ImageSet : IDisposable
    {
        SourceDescriptor m_src;
        List<LabelSet> m_rgLabelSet = new List<LabelSet>();
        List<LabelSet> m_rgLabelSetWithData = new List<LabelSet>();
        SimpleDatum[] m_rgImages;
        List<SimpleDatum> m_rgImagesLimitLoaded = new List<SimpleDatum>();
        SimpleDatum m_imgMean = null;
        CryptoRandom m_random = new CryptoRandom();
        int m_nLastFindIdx = 0;
        int m_nLastIdx = -1;
        int m_nFixedIndex = -1;
        object m_syncObj = new object();
        int m_nLoadLimit = 0;
        IMAGEDB_LOAD_METHOD m_loadMethod;
        Dictionary<int, double> m_rgLabelBoosts = new Dictionary<int, double>();
        double m_dfLabelBoostTotal = 0;
        DatasetFactory m_factory;
        int m_nLastImageIdxFromLoad = 0;

        /// <summary>
        /// The OnCalculateImageMean event fires when the ImageSet needs to calculate the image mean for the image set.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// The ImageSet constructor.
        /// </summary>
        /// <param name="factory">Specifies the DatasetFactory.</param>
        /// <param name="src">Specifies the data source.</param>
        /// <param name="loadMethod">Specifies the method to use when loading the images.</param>
        /// <param name="nLoadLimit">Specifies the image load limit.</param>
        public ImageSet(DatasetFactory factory, SourceDescriptor src, IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            m_factory = new DatasetFactory(factory);
            m_factory.Open(src.ID);
            m_loadMethod = loadMethod;
            m_nLoadLimit = nLoadLimit;
            m_src = new SourceDescriptor(src);            
            m_imgMean = null;

            m_rgImages = new SimpleDatum[m_src.ImageCount];

            foreach (LabelDescriptor label in src.Labels)
            {
                if (label.ImageCount > 0)
                {
                    m_rgLabelSet.Add(new LabelSet(label));
                }
            }
        }

        /// <summary>
        /// The Imageset constructor.
        /// </summary>
        protected ImageSet()
        {
        }


        /// <summary>
        /// Releases the resouces used.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called by Dispose()</param>
        protected virtual void Dispose(bool bDisposing)
        {
            if (m_random != null)
            {
                m_random.Dispose();
                m_random = null;
            }

            if (m_factory != null)
            {
                m_factory.Close();
                m_factory.Dispose();
                m_factory = null;
            }
        }

        /// <summary>
        /// Releases the resouces used.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }


        /// <summary>
        /// Returns a list of label descriptors used by the image set.
        /// </summary>
        /// <returns></returns>
        public List<LabelDescriptor> GetLabels()
        {
            List<LabelDescriptor> rgLabels = new List<LabelDescriptor>();

            foreach (LabelSet ls in m_rgLabelSet)
            {
                rgLabels.Add(new LabelDescriptor(ls.Label));
            }

            return rgLabels;
        } 

        /// <summary>
        /// Returns the label name of a label.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>Returns the string name of the label.</returns>
        public string GetLabelName(int nLabel)
        {
            foreach (LabelSet ls in m_rgLabelSet)
            {
                if (ls.Label.Label == nLabel)
                    return ls.Label.Name;
            }

            return nLabel.ToString();
        }

        /// <summary>
        /// Applies the label mapping to the image set.
        /// </summary>
        /// <param name="col"></param>
        public void Relabel(LabelMappingCollection col)
        {
            foreach (SimpleDatum sd in m_rgImages)
            {
                sd.SetLabel(col.MapLabel(sd.OriginalLabel));
            }

            ReloadLabelSets();
        }

        /// <summary>
        /// Returns a copy of the ImageSet.
        /// </summary>
        /// <returns>The ImageSet copy is returned.</returns>
        public ImageSet Clone()
        {
            ImageSet imgSet = new imagedb.ImageSet();

            imgSet.m_src = m_src;
            imgSet.m_factory = new imagedb.DatasetFactory(m_factory);

            foreach (LabelSet ls in m_rgLabelSet)
            {
                imgSet.m_rgLabelSet.Add(new imagedb.LabelSet(ls.Label));
            }

            List<SimpleDatum> rgSd = new List<basecode.SimpleDatum>();

            foreach (SimpleDatum sd in m_rgImages)
            {
                rgSd.Add(new SimpleDatum(sd));
            }

            foreach (SimpleDatum sd in m_rgImagesLimitLoaded)
            {
                imgSet.m_rgImagesLimitLoaded.Add(sd);
            }

            imgSet.m_rgImages = rgSd.ToArray();
            imgSet.m_imgMean = new SimpleDatum(m_imgMean);
            imgSet.ReloadLabelSets();

            return imgSet;
        }

        /// <summary>
        /// Searches for an image index based on its time-stamp and description.
        /// </summary>
        /// <param name="dt">Specifies the time-stamp.</param>
        /// <param name="strDesc">Specifies the description.</param>
        /// <returns>If found the image index is returned, otherwise -1 is returned.</returns>
        public int FindImageIndex(DateTime dt, string strDesc)
        {
            SimpleDatum[] rgImages = m_rgImages;

            if (m_rgImagesLimitLoaded.Count > 0)
                rgImages = m_rgImagesLimitLoaded.ToArray();

            for (int i = m_nLastFindIdx; i < rgImages.Length; i++)
            {
                if (rgImages[i].TimeStamp == dt && rgImages[i].Description == strDesc)
                {
                    m_nLastFindIdx = i;
                    return i;
                }
            }

            for (int i = 0; i < m_nLastFindIdx; i++)
            {
                if (rgImages[i].TimeStamp == dt && rgImages[i].Description == strDesc)
                {
                    m_nLastFindIdx = i;
                    return i;
                }
            }

            return -1;
        }

        /// <summary>
        /// Resets the limited loaded images.
        /// </summary>
        public void Reset()
        {
            m_rgImagesLimitLoaded = new List<SimpleDatum>();
        }

        /// <summary>
        /// Adds a new image to the image set.
        /// </summary>
        /// <param name="nIdx">Specifies the index on where to add the image.</param>
        /// <param name="d">Specifies the image data.</param>
        /// <returns>If added successfully within the load limit, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Add(int nIdx, SimpleDatum d)
        {
            if (m_nLoadLimit > 0 && m_rgImagesLimitLoaded.Count == m_nLoadLimit)
                return false;

            m_rgImages[nIdx] = d;

            if (m_nLoadLimit > 0)
                m_rgImagesLimitLoaded.Add(d);

            bool bAdded = false;

            foreach (LabelSet ls in m_rgLabelSet)
            {
                if (ls.Label.ActiveLabel == d.Label)
                {
                    ls.Add(d);
                    bAdded = true;
                    break;
                }
            }

            if (!bAdded)
            {
                LabelSet ls = new LabelSet(new LabelDescriptor(d.Label, d.Label, "label #" + d.Label.ToString(), 0));
                ls.Add(d);
                m_rgLabelSet.Add(ls);
            }

            return true;
        }

        /// <summary>
        /// Reload the label sets.
        /// </summary>
        public void ReloadLabelSets()
        {
            lock (m_syncObj)
            {
                SimpleDatum[] rgImages = m_rgImages;

                if (m_rgImagesLimitLoaded.Count > 0)
                    rgImages = m_rgImagesLimitLoaded.ToArray();

                foreach (LabelSet ls in m_rgLabelSet)
                {
                    ls.Clear();
                }

                foreach (SimpleDatum d in rgImages)
                {
                    foreach (LabelSet ls in m_rgLabelSet)
                    {
                        if (ls.Label.ActiveLabel == d.Label)
                        {
                            ls.Add(d);
                            break;
                        }
                    }
                }

                CompleteLoad(0);
            }
        }

        /// <summary>
        /// Complete the image loading process.
        /// </summary>
        /// <param name="nLastImageIdx">Specifies the last image index loaded.</param>
        public void CompleteLoad(int nLastImageIdx)
        {
            m_rgLabelSetWithData = new List<LabelSet>();

            foreach (LabelSet ls in m_rgLabelSet)
            {
                if (ls.Count > 0)
                    m_rgLabelSetWithData.Add(ls);
            }

            m_nLastImageIdxFromLoad = nLastImageIdx;
        }

        /// <summary>
        /// Returns the data source of the image set.
        /// </summary>
        public SourceDescriptor Source
        {
            get { return m_src; }
        }

        /// <summary>
        /// Returns the data source ID of the image set.
        /// </summary>
        public int SourceID
        {
            get { return m_src.ID; }
        }

        /// <summary>
        /// Returns the data source name of the image set.
        /// </summary>
        public string SourceName
        {
            get { return m_src.Name.Trim(); }
        }

        /// <summary>
        /// Returns whether or not the image set contains real or <i>byte</i> based data.
        /// </summary>
        public bool IsRealData
        {
            get { return m_src.IsRealData; }
        }

        /// <summary>
        /// Returns the number of images in the image set.
        /// </summary>
        public int Count
        {
            get { return m_rgImages.Length; }
        }

        /// <summary>
        /// Returns the number of images in the image set, optionally with super-boost only.
        /// </summary>
        /// <param name="bSuperboostOnly">Specifies whether or not to only count images with super-boost.</param>
        /// <returns></returns>
        public int GetCount(bool bSuperboostOnly)
        {
            if (!bSuperboostOnly)
                return Count;

            return m_rgImages.Where(p => p != null && p.Boost > 0).Count();
        }

        /// <summary>
        /// Returns the image based on its label and image selection method.
        /// </summary>
        /// <param name="nIdx">Specifies the image index to use when loading sequentially.</param>
        /// <param name="labelSelectionMethod">Specifies the label selection method.</param>
        /// <param name="imageSelectionMethod">Specifies the image selection method.</param>
        /// <param name="log">Specifies the Log for status output.</param>
        /// <returns>The SimpleDatum containing the image is returned.</returns>
        public SimpleDatum GetImage(int nIdx, IMGDB_LABEL_SELECTION_METHOD labelSelectionMethod, IMGDB_IMAGE_SELECTION_METHOD imageSelectionMethod, Log log)
        {
            lock (m_syncObj)
            {
                SimpleDatum[] rgImages = m_rgImages;

                if (m_nLoadLimit > 0 && m_rgImagesLimitLoaded.Count == m_nLoadLimit)
                    rgImages = m_rgImagesLimitLoaded.ToArray();

                if (rgImages.Length == 0)
                    throw new Exception("There are no images in the dataset '" + m_src.Name + "' to select from!");

                SimpleDatum sd = null;

                if ((labelSelectionMethod & IMGDB_LABEL_SELECTION_METHOD.RANDOM) == IMGDB_LABEL_SELECTION_METHOD.RANDOM)
                {
                    if (m_rgLabelSet.Count == 0)
                        throw new Exception("There are no label specified in the Labels table for the dataset '" + m_src.Name + "'!");

                    LabelSet labelSet = getLabelSet(labelSelectionMethod);
                    sd = labelSet.GetImage(nIdx, imageSelectionMethod);
                }

                int nImageIdx = 0;

                if (sd == null)
                    sd = LabelSet.GetImage(rgImages, rgImages.Length, nIdx, m_random, imageSelectionMethod, ref m_nLastIdx, ref m_nFixedIndex, out nImageIdx);


                //-----------------------------------------
                //  Handle dynamic loading of the image.
                //-----------------------------------------

                if (sd == null)
                {
                    int nRetries = 1;

                    if ((imageSelectionMethod & IMGDB_IMAGE_SELECTION_METHOD.RANDOM) == IMGDB_IMAGE_SELECTION_METHOD.RANDOM)
                        nRetries = 5;

                    for (int i = 0; i < nRetries; i++)
                    {
                        sd = m_factory.LoadImageAt(nImageIdx);
                        if (sd != null)
                        {
                            Add(nIdx, sd);
                            break;
                        }

                        if (i < nRetries - 1)
                            nImageIdx = m_random.Next(rgImages.Length);
                    }

                    if (sd == null)
                        log.WriteLine("WARNING! The dataset needs to be re-indexed. Could not find the image at index " + nImageIdx.ToString() + " - attempting several random queries to get an image.");
                }

                return sd;
            }
        }

        /// <summary>
        /// Retuns the LabelSet corresponding to a label.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>The LabelSet is returned.</returns>
        public LabelSet GetLabelSet(int nLabel)
        {
            foreach (LabelSet ls in m_rgLabelSet)
            {
                if (ls.Label.ActiveLabel == nLabel)
                    return ls;
            }

            return null;
        }

        private LabelSet getLabelSet(IMGDB_LABEL_SELECTION_METHOD labelSelectionMethod)
        {
            double dfBoostTotal = m_dfLabelBoostTotal;
            Dictionary<int, double> rgBoosts = m_rgLabelBoosts;
            int nIdx;

            if ((labelSelectionMethod & IMGDB_LABEL_SELECTION_METHOD.BOOST) != IMGDB_LABEL_SELECTION_METHOD.BOOST)
            {
                nIdx = m_random.Next(m_rgLabelSetWithData.Count);
                return m_rgLabelSetWithData[nIdx];
            }


            //---------------------------------------------
            //  Handle Label Sets with label boost.
            //---------------------------------------------
            else
            {
                double dfVal = m_random.NextDouble() * dfBoostTotal;
                double dfTotal = 0;

                nIdx = m_rgLabelSet.Count - 1;

                for (int i = 0; i < m_rgLabelSet.Count; i++)
                {
                    int nLabel = m_rgLabelSet[i].Label.ActiveLabel;

                    if (rgBoosts != null && rgBoosts.ContainsKey(nLabel))
                        dfTotal += (double)rgBoosts[nLabel];
                    else
                        dfTotal += 1;

                    if (dfTotal >= dfVal)
                    {
                        nIdx = i;
                        break;
                    }
                }

                return m_rgLabelSet[nIdx];
            }
        }

        /// <summary>
        /// Set the image mean on for the ImageSet.
        /// </summary>
        /// <param name="d">Specifies the image mean.</param>
        public void SetImageMean(SimpleDatum d)
        {
            m_imgMean = d;
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

            if (m_rgImages.Length == 0)
                throw new Exception("Cannot create image mean with no images!");

            if (m_loadMethod != IMAGEDB_LOAD_METHOD.LOAD_ALL)
                throw new Exception("Can only create image mean when using LOAD_ALL.");

            if (m_nLoadLimit != 0)
                throw new Exception("Can only create image mean when LoadLimit = 0.");

            if (OnCalculateImageMean != null)
            {
                CalculateImageMeanArgs args = new CalculateImageMeanArgs(m_rgImages);
                OnCalculateImageMean(this, args);

                if (args.Cancelled)
                    return null;

                m_imgMean = args.ImageMean;
                return m_imgMean;
            }

            m_imgMean = SimpleDatum.CalculateMean(log, m_rgImages, rgAbort);
            m_imgMean.SetLabel(0);

            return m_imgMean;
        }

        /// <summary>
        /// Update the label boosts for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void UpdateLabelBoosts(int nProjectId)
        {
            List<LabelBoostDescriptor> rgBoosts = m_factory.GetLabelBoosts(nProjectId, m_src.ID);
            m_rgLabelBoosts = new Dictionary<int, double>();

            m_dfLabelBoostTotal = 0;

            foreach (LabelBoostDescriptor boost in rgBoosts)
            {
                m_dfLabelBoostTotal += (double)boost.Boost;

                int nLabel = boost.Label;
                double dfBoost = (double)boost.Boost;

                if (!m_rgLabelBoosts.ContainsKey(nLabel))
                    m_rgLabelBoosts.Add(nLabel, dfBoost);
                else
                    m_rgLabelBoosts[nLabel] = dfBoost;
            }
        }

        /// <summary>
        /// Set the label mapping of the ImageSet.
        /// </summary>
        /// <param name="map">Specifies the label map.</param>
        public void SetLabelMapping(LabelMapping map)
        {
            m_factory.SetLabelMapping(map, m_src.ID);
        }

        /// <summary>
        /// Update the label mapping on the ImageSet.
        /// </summary>
        /// <param name="nNewLabel">Specifies the new label.</param>
        /// <param name="rgOriginalLabels">Specifies the labels to be mapped to the new label.</param>
        public void UpdateLabelMapping(int nNewLabel, List<int> rgOriginalLabels)
        {
            m_factory.UpdateLabelMapping(nNewLabel, rgOriginalLabels, m_src.ID);
        }

        /// <summary>
        /// Resets the labels for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void ResetLabels(int nProjectId)
        {
            m_factory.ResetLabels(nProjectId, m_src.ID);
        }

        /// <summary>
        /// Deletes the label boosts for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void DeleteLabelBoosts(int nProjectId)
        {
            m_factory.DeleteLabelBoosts(nProjectId, m_src.ID);
        }

        /// <summary>
        /// Adds a label boost for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the label boost.</param>
        public void AddLabelBoost(int nProjectId, int nLabel, double dfBoost)
        {
            m_factory.AddLabelBoost(nProjectId, nLabel, dfBoost, m_src.ID);
        }

        /// <summary>
        /// Returns the label boosts as text.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <returns>The label boosts are returned.</returns>
        public string GetLabelBoostsAsText(int nProjectId)
        {
            return m_factory.GetLabelBoostsAsText(nProjectId, m_src.ID);
        }

        /// <summary>
        /// Returns the label counts as a dictionary of item pairs (int nLabel, int nCount).
        /// </summary>
        /// <returns>The label counts are returned as item pairs (int nLabel, int nCount).</returns>
        public Dictionary<int, int> LoadLabelCounts()
        {
            return m_factory.LoadLabelCounts(m_src.ID);
        }

        /// <summary>
        /// Updates the label counts for a project.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        public void UpdateLabelCounts(int nProjectId)
        {
            m_factory.UpdateLabelCounts(m_src.ID, nProjectId);
        }

        /// <summary>
        /// Returns the label counts for the ImageList as text.
        /// </summary>
        /// <returns>The label counts are retuned.</returns>
        public string GetLabelCountsAsText()
        {
            return m_factory.GetLabelCountsAsText(m_src.ID);
        }
    }
}
