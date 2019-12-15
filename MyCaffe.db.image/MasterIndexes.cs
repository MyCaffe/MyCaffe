using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The MasterIndexes stores the indexes that define the index structure of the data source data.
    /// </summary>
    public class MasterIndexes : IDisposable
    {
        /// <summary>
        /// Specifies the data source descriptor.
        /// </summary>
        protected SourceDescriptor m_src;
        /// <summary>
        /// Specifies the index into all of the data source images.
        /// </summary>
        protected Index m_index;
        /// <summary>
        /// Specifies the list of images listed by label where each label contains an index into all images with that label.
        /// </summary>
        protected LabelIndex m_rgLabels;
        /// <summary>
        /// Specifies the list of all boosted images.
        /// </summary>
        protected Index m_boosted;
        /// <summary>
        /// Specifies the list of all boosted images listed by label where each label contains an index into all boosted images with that label.
        /// </summary>
        protected LabelIndex m_rgLabelsBoosted;
        CryptoRandom m_random;
        DatasetFactory m_factory = new DatasetFactory();
        List<DbItem> m_rgImageIdx = null;
        IMGDB_SORT m_sort = IMGDB_SORT.BYIDX;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="src">Specifies the data source.</param>
        public MasterIndexes(CryptoRandom random, SourceDescriptor src)
        {
            m_random = random;
            m_src = src;
            m_factory.Open(src);
            m_rgImageIdx = m_factory.LoadImageIndexes(false);

            load(m_rgImageIdx);
        }

        private void load(List<DbItem> rgItems)
        {
            m_index = new Index("ALL", m_random, rgItems);
            List<DbItem> rgBoosted = rgItems.Where(p => p.Boost > 0).ToList();
            m_boosted = new Index("BOOSTED", m_random, rgBoosted, -1, true);

            List<LabelDescriptor> rgLabels = m_src.Labels.OrderBy(p => p.ActiveLabel).ToList();

            m_rgLabels = new LabelIndex("LABELED", m_random, m_src, false, rgItems);
            m_rgLabelsBoosted = new LabelIndex("LABELED BOOSTED", m_random, m_src, true, rgBoosted);
        }

        /// <summary>
        /// The constructor used to copy another MasterIndexes and optionally specify a sorting for the indexes.
        /// </summary>
        /// <param name="idx">Specifies the MasterIndexes to copy.</param>
        /// <param name="sort">Optionally, specifies a sorting to use on the indexes.</param>
        public MasterIndexes(MasterIndexes idx, IMGDB_SORT sort)
        {
            m_sort = sort;
            m_src = idx.m_src;
            m_random = idx.m_random;
            m_factory = new DatasetFactory(idx.m_factory);

            m_index = idx.m_index.Clone(sort);
            m_rgLabels = new LabelIndex(idx.m_rgLabels);

            m_boosted = idx.m_boosted.Clone(sort);
            m_rgLabelsBoosted = new LabelIndex(idx.m_rgLabelsBoosted);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_factory != null)
            {
                m_factory.Close();
                m_factory.Dispose();
                m_factory = null;
            }
        }

        /// <summary>
        /// Returns the number of labels.
        /// </summary>
        public int LabelCount
        {
            get { return m_rgLabels.Count; }
        }

        /// <summary>
        /// Returns all DbItems that point to images iwth a given date.
        /// </summary>
        /// <param name="dt">Specifies the date to look for.</param>
        /// <returns>The list of DbItems matching the images is returned.</returns>
        public List<DbItem> FindImageIndexes(DateTime dt)
        {
            return m_index.FindImageIndexes(dt);
        }

        /// <summary>
        /// Reload all images by re-loading the master index list.
        /// </summary>
        /// <param name="rgItems">Specifies the list of DbItem's used to re-load the indexes.</param>
        public void Reload(List<DbItem> rgItems)
        {
            load(rgItems);
        }

        /// <summary>
        /// Returns the Index matching the criteria.
        /// </summary>
        /// <param name="nLabel">Optionally, specifies a label to use (default = null).</param>
        /// <param name="bBoosted">Optionally, specifies to use boosted images (default = false).</param>
        /// <returns>The Index matching the criteria is returned.</returns>
        public Index GetIndex(int? nLabel = null, bool bBoosted = false)
        {
            if (!nLabel.HasValue)
            {
                if (!bBoosted)
                    return m_index;
                else
                    return m_boosted;
            }
            else
            {
                if (!bBoosted)
                    return m_rgLabels.GetNextIndex(Index.SELECTION_TYPE.DIRECT, nLabel);
                else
                    return m_rgLabelsBoosted.GetNextIndex(Index.SELECTION_TYPE.DIRECT, nLabel);
            }
        }

        /// <summary>
        /// Set a given index based on the criteria.
        /// </summary>
        /// <param name="idx">Specifies the Index source.</param>
        /// <param name="nLabel">Optionally, specifies a label to use (default = null).</param>
        /// <param name="bBoosted">Optionally, specifies to use boosted images (default = false).</param>
        public void SetIndex(Index idx, int? nLabel = null, bool bBoosted = false)
        {
            if (!nLabel.HasValue)
            {
                if (!bBoosted)
                    m_index = idx;
                else
                    m_boosted = idx;
            }
            else
            {
                if (!bBoosted)
                    m_rgLabels.SetIndex(nLabel.Value, idx);
                else
                    m_rgLabelsBoosted.SetIndex(nLabel.Value, idx);
            }
        }

        /// <summary>
        /// Returns the indexes fitting the criteria.
        /// </summary>
        /// <param name="nStartIdx">Specifies a starting index from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>A list with the image indexes is returned.</returns>
        public List<int> GetIndexes(int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            Index idx = GetIndex(null, nBoostVal.HasValue);
            List<DbItem> rgIdx = idx.FindImageIndexes(nStartIdx, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);
            return rgIdx.Select(p => p.Index).ToList();
        }

        /// <summary>
        /// Returns the indexes fitting the criteria.
        /// </summary>
        /// <param name="dtStart">Specifies a starting time from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>A list with the image indexes is returned.</returns>
        public List<int> GetIndexes(DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            Index idx = GetIndex(null, nBoostVal.HasValue);
            List<DbItem> rgIdx = idx.FindImageIndexes(dtStart, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);
            return rgIdx.Select(p => p.Index).ToList();
        }

        /// <summary>
        /// Returns the next label in the Index set selected based on the selection criteria.
        /// </summary>
        /// <param name="type">Specifies the selection type (e.g. RANDOM, SEQUENTIAL).</param>
        /// <param name="bBoosted">Optionally, specifies to use label sets of boosted images (default = false).</param>
        /// <returns>The next label index is returned.</returns>
        public virtual int? GetNextLabel(Index.SELECTION_TYPE type, bool bBoosted = false)
        {
            LabelIndex rgIdx = (bBoosted) ? m_rgLabelsBoosted : m_rgLabels;
            if (rgIdx.Count == 0)
                return null;

            return rgIdx.GetNextLabel(type, null, false);
        }

        /// <summary>
        /// Returns the next image in the Index set based on the selection criteria.
        /// </summary>
        /// <param name="type">Specifies the selection type (e.g. RANDOM, SEQUENTIAL).</param>
        /// <param name="nLabel">Optionally, specifies a label (default = null).</param>
        /// <param name="bBoosted">Optionally, specifies to query boosted images (default = false).</param>
        /// <param name="nDirectIdx">Optionally, specifies to query the image at this index (only applies when type = DIRECT).</param>
        /// <returns>The next image index is returned.</returns>
        public virtual int? GetNextImage(Index.SELECTION_TYPE type, int? nLabel = null, bool bBoosted = false, int nDirectIdx = -1)
        {
            Index idx = GetIndex(nLabel, bBoosted);
            return idx.GetNext(type);
        }

        /// <summary>
        /// Returns a string representation of the master indexes.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "";

            if (m_sort == IMGDB_SORT.NONE)
            {
                strOut = "NONE";
            }
            else
            {
                if ((m_sort & IMGDB_SORT.BYID) == IMGDB_SORT.BYID)
                    strOut += "ID";

                if ((m_sort & IMGDB_SORT.BYDESC) == IMGDB_SORT.BYDESC)
                {
                    if (strOut.Length > 0)
                        strOut += " | ";
                    strOut += "DESC";
                }

                if ((m_sort & IMGDB_SORT.BYIDX) == IMGDB_SORT.BYIDX)
                {
                    if (strOut.Length > 0)
                        strOut += " | ";
                    strOut += "IDX";
                }

                if ((m_sort & IMGDB_SORT.BYID_DESC) == IMGDB_SORT.BYID_DESC)
                {
                    if (strOut.Length > 0)
                        strOut += " | ";
                    strOut += "ID(desc)";
                }


                if ((m_sort & IMGDB_SORT.BYTIME) == IMGDB_SORT.BYTIME)
                {
                    if (strOut.Length > 0)
                        strOut += " | ";
                    strOut += "TIME";
                }
            }

            return strOut;
        }
    }

    public class LabelIndex /** @private */
    {
        string m_strName;
        CryptoRandom m_random;
        SourceDescriptor m_src;
        Dictionary<int, int> m_rgLabelToIdxMap = new Dictionary<int, int>();
        Dictionary<int, int> m_rgIdxToLabelMap = new Dictionary<int, int>();
        Index[] m_rgLabels = null;
        bool m_bBoosted = false;
        List<int> m_rgIdx = new List<int>();
        int m_nIdx = 0;

        public LabelIndex(string strName, CryptoRandom random, SourceDescriptor src, bool bBoosted, List<DbItem> rgItems)
        {
            m_strName = strName;
            m_random = random;
            m_src = src;
            m_bBoosted = bBoosted;

            m_rgIdx = new List<int>();
            m_rgLabelToIdxMap = new Dictionary<int, int>();
            m_rgIdxToLabelMap = new Dictionary<int, int>();

            List<LabelDescriptor> rgLabels = src.Labels.Where(p => p.ImageCount > 0).OrderBy(p => p.ActiveLabel).ToList();
            if (rgLabels.Count > 0)
            {
                m_rgLabels = new Index[rgLabels.Count];

                for (int i = 0; i < rgLabels.Count; i++)
                {
                    int nLabel = rgLabels[i].ActiveLabel;
                    List<DbItem> rgLabelList = rgItems.Where(p => p.Label == nLabel).ToList();

                    if (i < rgLabels.Count - 1)
                        rgItems = rgItems.Where(p => p.Label != nLabel).ToList();

                    m_rgLabels[i] = new Index(strName + " label " + nLabel.ToString(), random, rgLabelList, nLabel, false);
                    if (rgLabelList.Count > 0)
                        m_rgIdx.Add(i);

                    m_rgLabelToIdxMap[nLabel] = i;
                    m_rgIdxToLabelMap[i] = nLabel;
                }
            }
        }

        public LabelIndex(LabelIndex idx)
        {
            m_strName = idx.m_strName + " copy";
            m_random = idx.m_random;
            m_src = idx.m_src;
            m_bBoosted = idx.m_bBoosted;

            m_rgIdx = new List<int>();

            if (idx.m_rgLabels != null && idx.m_rgLabels.Length > 0)
            {
                m_rgLabels = new Index[idx.m_rgLabels.Length];

                bool bFillLabelMap = false;
                if (m_rgLabelToIdxMap == null || m_rgLabelToIdxMap.Count == 0 || m_rgIdxToLabelMap == null || m_rgIdxToLabelMap.Count == 0)
                {
                    m_rgLabelToIdxMap = new Dictionary<int, int>();
                    m_rgIdxToLabelMap = new Dictionary<int, int>();
                    bFillLabelMap = true;
                }

                for (int i = 0; i < idx.m_rgLabels.Length; i++)
                {
                    m_rgLabels[i] = idx.m_rgLabels[i].Clone();
                    if (m_rgLabels[i].Count > 0)
                        m_rgIdx.Add(i);

                    if (bFillLabelMap)
                    {
                        int nLabel = m_rgLabels[i].Label;
                        m_rgLabelToIdxMap[nLabel] = i;
                        m_rgIdxToLabelMap[i] = nLabel;
                    }
                }
            }
        }

        public void ReLoad()
        {
            if (m_rgIdx.Count == m_rgLabels.Length)
                return;

            for (int i = 0; i < m_rgLabels.Length; i++)
            {
                if (m_rgLabels[i].Count > 0)
                   m_rgIdx.Add(i);
            }
        }

        public void SetIndex(int nLabel, Index idx)
        {
            int nIdx = m_rgLabelToIdxMap[nLabel];
            m_rgLabels[nIdx] = idx;
        }

        public int Count
        {
            get { return (m_rgLabels == null) ? 0 : m_rgLabels.Length; }
        }

        public bool Boosted
        {
            get { return m_bBoosted; }
        }

        public bool IsEmpty
        {
            get { return (m_rgIdx.Count == 0) ? true : false; }
        }

        public LabelIndex Clone()
        {
            return new LabelIndex(this);
        }

        public int? GetNextLabel(Index.SELECTION_TYPE type, int? nLabel, bool bRemove = false)
        {
            if (m_rgIdx.Count == 0)
                return null;

            if (nLabel.HasValue)
            {
                return nLabel.Value;
            }
            else if (type == Index.SELECTION_TYPE.SEQUENTIAL)
            {
                int nIdx = m_rgIdx[m_nIdx];

                m_nIdx++;
                if (m_nIdx >= m_rgIdx.Count)
                    m_nIdx = 0;

                return m_rgIdxToLabelMap[nIdx];
            }
            else
            {
                int nIdx = m_rgIdx[0];

                if (m_rgIdx.Count > 1)
                {
                    nIdx = m_random.Next(m_rgIdx.Count);
                    nIdx = m_rgIdx[nIdx];
                }

                if (bRemove)
                    m_rgIdx.Remove(nIdx);

                return m_rgIdxToLabelMap[nIdx];
            }
        }

        public Index GetNextIndex(Index.SELECTION_TYPE type, int? nLabel, bool bRemove = false)
        {
            nLabel = GetNextLabel(type, nLabel, bRemove);
            if (!nLabel.HasValue)
                return null;

            int nIdx = m_rgLabelToIdxMap[nLabel.Value];

            return m_rgLabels[nIdx];
        }

        public override string ToString()
        {
            return m_strName;
        }
    }

    public class Index /** @private */
    {
        string m_strName;
        CryptoRandom m_random;
        int m_nLabel = -1;
        bool m_bBoosted = false;
        int m_nIdx = 0;
        List<DbItem> m_rgItems;
        double m_dfProbability = 0;

        public enum SELECTION_TYPE
        {
            DIRECT,
            SEQUENTIAL,
            RANDOM
        }

        public Index(string strName, CryptoRandom random, List<DbItem> rgItems, int nLabel = -1, bool bBoosted = false, double dfProbability = 0)
        {
            m_strName = strName;
            m_random = random;
            m_rgItems = rgItems;
            m_nLabel = nLabel;
            m_bBoosted = bBoosted;
            m_dfProbability = dfProbability;
        }

        public int Count
        {
            get { return m_rgItems.Count; }
        }

        public double Probability
        {
            get { return m_dfProbability; }
            set { m_dfProbability = value; }
        }

        public int Label
        {
            get { return m_nLabel; }
        }

        public bool Boosted
        {
            get { return m_bBoosted; }
        }

        public bool IsEmpty
        {
            get { return (m_rgItems.Count == 0) ? true : false; }
        }

        public Index Clone(IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            List<DbItem> rgItems = new List<DbItem>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rgItems.Add(m_rgItems[i].Clone());
            }

            switch (sort)
            {
                case IMGDB_SORT.BYID:
                    rgItems = rgItems.OrderBy(p => p.ID).ToList();
                    break;

                case IMGDB_SORT.BYID_DESC:
                    rgItems = rgItems.OrderByDescending(p => p.ID).ToList();
                    break;

                case IMGDB_SORT.BYIDX:
                    rgItems = rgItems.OrderBy(p => p.Index).ToList();
                    break;

                case IMGDB_SORT.BYDESC:
                    rgItems = rgItems.OrderBy(p => p.Desc).ThenBy(p => p.Index).ToList();
                    break;

                case IMGDB_SORT.BYTIME:
                    rgItems = rgItems.OrderBy(p => p.Time).ToList();
                    break;

                default:
                    if ((sort & (IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME)) == (IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME))
                        rgItems = rgItems.OrderBy(p => p.Desc).ThenBy(p => p.Time).ToList();
                    break;
            }

            return new Index(m_strName + " copy", m_random, rgItems, m_nLabel, m_bBoosted, m_dfProbability);
        }

        public List<DbItem> FindImageIndexes(int nStartIdx, int nQueryCount = int.MaxValue, string strFilter = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            IEnumerable<DbItem> iQuery = m_rgItems.Where(p => p.Index >= nStartIdx);

            if (strFilter != null)
                iQuery = iQuery.Where(p => p.Desc == strFilter);

            if (nBoostVal.HasValue)
            {
                if (bBoostValIsExact)
                    iQuery = iQuery.Where(p => p.Boost == nBoostVal.Value);
                else
                    iQuery = iQuery.Where(p => p.Boost >= nBoostVal.Value);
            }

            return iQuery.Take(nQueryCount).ToList();
        }

        public List<DbItem> FindImageIndexes(DateTime dtStart, int nQueryCount = int.MaxValue, string strFilter = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            IEnumerable<DbItem> iQuery = m_rgItems.Where(p => p.Time >= dtStart);

            if (strFilter != null)
                iQuery = iQuery.Where(p => p.Desc == strFilter);

            if (nBoostVal.HasValue)
            {
                if (bBoostValIsExact)
                    iQuery = iQuery.Where(p => p.Boost == nBoostVal.Value);
                else
                    iQuery = iQuery.Where(p => p.Boost >= nBoostVal.Value);
            }

            return iQuery.Take(nQueryCount).ToList();
        }

        public List<DbItem> FindImageIndexes(DateTime dt)
        {
            return m_rgItems.Where(p => p.Time == dt).ToList();
        }

        public int? GetNext(SELECTION_TYPE type, bool bRemove = false)
        {
            if (m_rgItems.Count == 0)
                return null;

            if (type == SELECTION_TYPE.SEQUENTIAL)
            {
                int nIdx = m_rgItems[m_nIdx].Index;

                m_nIdx++;
                if (m_nIdx >= m_rgItems.Count)
                    m_nIdx = 0;

                return nIdx;
            }
            else
            {
                int nIdx = m_random.Next(m_rgItems.Count);
                int nFinalIdx = m_rgItems[nIdx].Index;

                if (bRemove)
                    m_rgItems.RemoveAt(nIdx);

                m_nIdx = nIdx + 1;
                if (m_nIdx == m_rgItems.Count)
                    m_nIdx = 0;

                return nFinalIdx;
            }
        }

        public int? GetIndex(int nDirectIdx)
        {
            if (nDirectIdx < 0 || nDirectIdx >= m_rgItems.Count)
                return null;

            m_nIdx = nDirectIdx;
            if (m_nIdx == m_rgItems.Count)
                m_nIdx = 0;

            return m_rgItems[nDirectIdx].Index;
        }

        public override string ToString()
        {
            return m_strName + ": Count = " + m_rgItems.Count().ToString() + " CurIdx = " + m_nIdx.ToString() + "; Label = " + m_nLabel.ToString() + "; Boosted = " + m_bBoosted.ToString() + " => (" + m_rgItems.Count.ToString() + ") p = " + m_dfProbability.ToString("P");
        }
    }
}
