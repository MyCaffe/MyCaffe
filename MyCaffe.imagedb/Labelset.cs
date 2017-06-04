using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.imagedb
{
    /// <summary>
    /// The LabelSet 'points' into the main image list via references objects that are already created in the main image list of the ImageSet.
    /// </summary>
    public class LabelSet : IDisposable
    {
        LabelDescriptor m_label;
        SimpleDatum[] m_rgImages;
        CryptoRandom m_random = new CryptoRandom();
        int m_nCurrentIdx = 0;

        /// <summary>
        /// The LabelSet constructor.
        /// </summary>
        /// <param name="lbl">Specifies the label.</param>
        public LabelSet(LabelDescriptor lbl)
        {
            m_label = lbl;
            m_rgImages = new SimpleDatum[lbl.ImageCount];
        }

        /// <summary>
        /// Get/set the label of the LabelSet.
        /// </summary>
        public LabelDescriptor Label
        {
            get { return m_label; }
            set { m_label = value; }
        }

        /// <summary>
        /// Returns the number of images in the label set.
        /// </summary>
        public int Count
        {
            get { return m_rgImages.Length; }
        }

        /// <summary>
        /// Clears the list of images.
        /// </summary>
        public void Clear()
        {
            m_rgImages = new SimpleDatum[m_rgImages.Length];
            m_nCurrentIdx = 0;
        }

        /// <summary>
        /// Get/set an image at an index within the LabelSet.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The image at the index is returned.</returns>
        public SimpleDatum this[int nIdx]
        {
            get { return m_rgImages[nIdx]; }
            set { m_rgImages[nIdx] = value; }
        }

        /// <summary>
        /// Adds an image to the current index and then advances the internal index.
        /// </summary>
        /// <param name="s">Specifies the image.</param>
        public void Add(SimpleDatum s)
        {
            if (m_nCurrentIdx < m_rgImages.Length)
            {
                m_rgImages[m_nCurrentIdx] = s;
                m_nCurrentIdx++;
            }
        }

        /// <summary>
        /// Returns an image from the LabelSet using the image selection method.
        /// </summary>
        /// <param name="nIdx">Specifies the index to use when performing sequential selection.</param>
        /// <param name="selectionMethod">Specifies the image selection method.</param>
        /// <returns>The image is returned.</returns>
        public SimpleDatum GetImage(int nIdx, IMGDB_IMAGE_SELECTION_METHOD selectionMethod)
        {
            if ((selectionMethod & IMGDB_IMAGE_SELECTION_METHOD.RANDOM) != IMGDB_IMAGE_SELECTION_METHOD.RANDOM)
                throw new ArgumentException("Label balancing image selection only supports the RANDOM and RANDOM+BOOST selection methods");

            if (m_nCurrentIdx == 0)
                return null;

            int nLastIdx = 0;
            int nFixedIdx = -1;
            int nImageIdx = 0;

            return GetImage(m_rgImages, m_nCurrentIdx, nIdx, m_random, selectionMethod, ref nLastIdx, ref nFixedIdx, out nImageIdx);
        }

        /// <summary>
        /// Returns an image from a list of images.
        /// </summary>
        /// <param name="rgImages">Specifies the image list to select from.</param>
        /// <param name="nCount">Specifies the maximum count to use.</param>
        /// <param name="nIdx">Specifies the index to use when selecting sequentially or in pair selection.</param>
        /// <param name="random">Specifies the random number generator to use.</param>
        /// <param name="selectionMethod">Specifies the image selection method.</param>
        /// <param name="nLastIndex">Specifies the last index used.</param>
        /// <param name="nFixedIndex">Specifies the fixed index to use.</param>
        /// <param name="nImageIdx">Returns the image index used.</param>
        /// <returns></returns>
        public static SimpleDatum GetImage(SimpleDatum[] rgImages, int nCount, int nIdx, CryptoRandom random, IMGDB_IMAGE_SELECTION_METHOD selectionMethod, ref int nLastIndex, ref int nFixedIndex, out int nImageIdx)
        {
            if ((selectionMethod & IMGDB_IMAGE_SELECTION_METHOD.BOOST) == IMGDB_IMAGE_SELECTION_METHOD.BOOST)
            {
                IEnumerable<SimpleDatum> iQuery = rgImages.Where(p => p != null && p.Boost > 0);
                List<SimpleDatum> rgItems = new List<SimpleDatum>(iQuery);

                if (rgItems.Count > 0)
                {
                    if ((selectionMethod & IMGDB_IMAGE_SELECTION_METHOD.RANDOM) == IMGDB_IMAGE_SELECTION_METHOD.RANDOM)
                        nIdx = random.Next(rgItems.Count);

                    SimpleDatum sd = rgItems[nIdx];
                    nImageIdx = nIdx;

                    return sd;
                }
            }

            if ((selectionMethod & IMGDB_IMAGE_SELECTION_METHOD.PAIR) == IMGDB_IMAGE_SELECTION_METHOD.PAIR)
            {
                nIdx = nLastIndex + 1;

                if (nIdx == nCount)
                    nIdx = 0;
            }
            else if ((selectionMethod & IMGDB_IMAGE_SELECTION_METHOD.RANDOM) == IMGDB_IMAGE_SELECTION_METHOD.RANDOM)
            {
                nIdx = random.Next(nCount);
            }
            else if (selectionMethod == IMGDB_IMAGE_SELECTION_METHOD.FIXEDINDEX)
            {
                nFixedIndex = nIdx;
            }
            else if ((selectionMethod & IMGDB_IMAGE_SELECTION_METHOD.CLEARFIXEDINDEX) == IMGDB_IMAGE_SELECTION_METHOD.CLEARFIXEDINDEX)
            {
                nFixedIndex = -1;
            }

            if (nFixedIndex >= 0)
                nIdx = nFixedIndex;

            if (nIdx >= rgImages.Length)
                nIdx = nIdx % rgImages.Length;

            SimpleDatum d = rgImages[nIdx];
            nLastIndex = nIdx;
            nImageIdx = nIdx;

            return d;
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called from Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            if (m_random != null)
            {
                m_random.Dispose();
                m_random = null;
            }
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }
    }
}
