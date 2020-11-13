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
    /// Initially the QueryState is copied from the MasterIndexes and during each query is altered by removing items already observed.  Once empty, each Index within
    /// the QueryState is then refreshed with the corresponding MasterIndexes ensuring that all images are hit over time.
    /// </summary>
    /// <remarks>
    /// QueryStates may also be ordered which is usedful in SEQUENTIAL querries.
    /// </remarks>
    public class QueryState : MasterIndexes
    {
        MasterIndexes m_master;
        LabelStats m_stat;
        bool m_bUseUniqueImageIndexes = true;
        bool m_bUseUniqueLabelIndexes = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="master">Specifies the MasterIndexes to copy.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Optionally, specifies the ordering to use on the indexes (default = BYIDX).</param>
        public QueryState(MasterIndexes master, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.BYIDX) : base(master, sort)
        {
            m_master = master;
            m_stat = new LabelStats(master.LabelCount);
            m_bUseUniqueLabelIndexes = bUseUniqueLabelIndexes;
            m_bUseUniqueImageIndexes = bUseUniqueImageIndexes;

            foreach (LabelDescriptor label in m_src.Labels)
            {
                m_stat.Add(label);
            }
        }

        /// <summary>
        /// Update the label stats.
        /// </summary>
        /// <param name="sd">Specifies the recently queried simple datum.</param>
        public void UpdateStats(SimpleDatum sd)
        {
            m_stat.UpdateLabel(sd.Label);
            m_stat.UpdateBoost(sd.Boost);
        }

        /// <summary>
        /// Returns the next label in the Index set selected based on the selection criteria.
        /// </summary>
        /// <param name="lblSel">Specifies the label selection method used.</param>
        /// <returns>The next label index is returned.</returns>
        public int? GetNextLabel(IMGDB_LABEL_SELECTION_METHOD lblSel)
        {
            Index.SELECTION_TYPE selType = Index.SELECTION_TYPE.RANDOM;
            bool bBoost = false;

            if ((lblSel & IMGDB_LABEL_SELECTION_METHOD.RANDOM) != IMGDB_LABEL_SELECTION_METHOD.RANDOM)
                selType = Index.SELECTION_TYPE.SEQUENTIAL;

            if ((lblSel & IMGDB_LABEL_SELECTION_METHOD.BOOST) == IMGDB_LABEL_SELECTION_METHOD.BOOST)
                bBoost = true;

            return GetNextLabel(selType, bBoost);
        }

        /// <summary>
        /// Returns the next label in the Index set selected based on the selection criteria.
        /// </summary>
        /// <param name="type">Specifies the selection type (e.g. RANDOM, SEQUENTIAL).</param>
        /// <param name="bBoosted">Optionally, specifies to use label sets of boosted images (default = false).</param>
        /// <returns>The next label index is returned.</returns>
        public override int? GetNextLabel(Index.SELECTION_TYPE type, bool bBoosted = false)
        {
            LabelIndex rgIdx = (bBoosted) ? m_rgLabelsBoosted : m_rgLabels;
            if (rgIdx.Count == 0)
                return null;

            int? nIdx = rgIdx.GetNextLabel(type, null, m_bUseUniqueLabelIndexes);
            if (rgIdx.IsEmpty)
                rgIdx.ReLoad();

            return nIdx;
        }

        /// <summary>
        /// Returns the next image in the Index set based on the selection criteria.
        /// </summary>
        /// <param name="imgSel">Specifies the image selection method used.</param>
        /// <param name="nLabel">Optionally, specifies a label (default = null).</param>
        /// <param name="nDirectIdx">Optionally, specifies to query the image at this index (only applies when type = DIRECT).</param>
        /// <returns></returns>
        public int? GetNextImage(IMGDB_IMAGE_SELECTION_METHOD imgSel, int? nLabel, int nDirectIdx)
        {
            if (!nLabel.HasValue && imgSel == IMGDB_IMAGE_SELECTION_METHOD.NONE && nDirectIdx >= 0)
                return GetNextImage(Index.SELECTION_TYPE.DIRECT, null, false, nDirectIdx);

            Index.SELECTION_TYPE selType = Index.SELECTION_TYPE.RANDOM;
            bool bBoosted = false;

            if ((imgSel & IMGDB_IMAGE_SELECTION_METHOD.RANDOM) != IMGDB_IMAGE_SELECTION_METHOD.RANDOM)
            {
                selType = Index.SELECTION_TYPE.SEQUENTIAL;

                // When using PAIR, advance to the next item.
                if ((imgSel & IMGDB_IMAGE_SELECTION_METHOD.PAIR) == IMGDB_IMAGE_SELECTION_METHOD.PAIR)
                    GetNextImage(selType, nLabel, bBoosted, -1);
            }

            if ((imgSel & IMGDB_IMAGE_SELECTION_METHOD.BOOST) == IMGDB_IMAGE_SELECTION_METHOD.BOOST)
                bBoosted = true;

            return GetNextImage(selType, nLabel, bBoosted, -1);
        }

        /// <summary>
        /// Returns the next image in the Index set based on the selection criteria.
        /// </summary>
        /// <param name="type">Specifies the selection type (e.g. RANDOM, SEQUENTIAL).</param>
        /// <param name="nLabel">Optionally, specifies a label (default = null).</param>
        /// <param name="bBoosted">Optionally, specifies to query boosted images (default = false).</param>
        /// <param name="nDirectIdx">Optionally, specifies to query the image at this index (only applies when type = DIRECT).</param>
        /// <returns>The next image index is returned.</returns>
        public override int? GetNextImage(Index.SELECTION_TYPE type, int? nLabel = null, bool bBoosted = false, int nDirectIdx = -1)
        {
            int? nIdx = null;

            if (m_master.LoadLimit > 0)
            {
                nIdx = base.GetNextImage(type, nLabel, bBoosted, nDirectIdx);
            }
            else
            {
                Index idx;
                if (type == Index.SELECTION_TYPE.DIRECT)
                {
                    if (nDirectIdx < 0)
                        throw new Exception("Invalid direct index, must be >= 0.");

                    idx = GetIndex(null, false);
                    nIdx = idx.GetIndex(nDirectIdx);
                }
                else
                {
                    idx = GetIndex(nLabel, bBoosted);
                    if (idx == null)
                    {
                        SetIndex(m_master.GetIndex(nLabel, bBoosted).Clone(), nLabel, bBoosted);
                        idx = GetIndex(nLabel, bBoosted);
                        nIdx = idx.GetNext(type, m_bUseUniqueImageIndexes);
                    }
                    else
                    {
                        nIdx = idx.GetNext(type, m_bUseUniqueImageIndexes);
                        if (idx.IsEmpty)
                            SetIndex(m_master.GetIndex(nLabel, bBoosted).Clone(), nLabel, bBoosted);
                    }
                }
            }

            return nIdx;
        }

        /// <summary>
        /// Returns the query label counts.
        /// </summary>
        /// <returns>The query label counts are returned.</returns>
        public Dictionary<int, ulong> GetQueryLabelCounts()
        {
            return m_stat.GetCounts();
        }

        /// <summary>
        /// Returns the number of times each boosted image vs. non boosted images are hit.
        /// </summary>
        /// <returns>The percentage of non-boosted vs. boosted images is returned as {non-boosted%, boosted%}.</returns>
        public string GetQueryBoostHitPercentsAsText()
        {
            return m_stat.GetQueryBoostHitPercentsAsText();
        }

        /// <summary>
        /// Returns the number of times each label is hit.
        /// </summary>
        /// <returns>The percentage of times each label hit occurs is returned in label order (e.g. label 0%, label 1%,... label n%).</returns>
        public string GetQueryLabelHitPercentsAsText()
        {
            return m_stat.GetQueryLabelHitPercentsAsText();
        }

        /// <summary>
        /// Returns the number of epochs each label has experienced.
        /// </summary>
        /// <returns></returns>
        public string GetQueryLabelEpochsAsText()
        {
            return m_stat.GetQueryLabelEpochAsText();
        }
    }

    /// <summary>
    /// The QueryStateCollection manages all query states used by matching the QueryState handles with the QueryStates where each handle points to both the training set query state and testing set query state.
    /// </summary>
    public class QueryStateCollection : IDisposable
    {
        Dictionary<long, Tuple<QueryState, QueryState>> m_rgQueryStates = new Dictionary<long, Tuple<QueryState, QueryState>>();
        object m_objSync = new object();

        /// <summary>
        /// The constructor.
        /// </summary>
        public QueryStateCollection()
        {
            // Fill with blank item for first handle, for handle = 0 is not used.
            m_rgQueryStates.Add(0, null); 
        }

        /// <summary>
        /// Releases all resources.
        /// </summary>
        public void Dispose()
        {
            foreach (KeyValuePair<long, Tuple<QueryState, QueryState>> kv in m_rgQueryStates)
            {
                kv.Value.Item1.Dispose();
                kv.Value.Item2.Dispose();
            }

            m_rgQueryStates.Clear();
        }

        /// <summary>
        /// Create a new QueryState handle based on the QueryStates passed to this function.
        /// </summary>
        /// <param name="training">Specifies the training QueryState used on the training data source.</param>
        /// <param name="testing">Specifies the testing QueryState used on the testing data source.</param>
        /// <returns>The QueryState handle is returned.</returns>
        public long CreateNewState(QueryState training, QueryState testing)
        {
            long lHandle = 0;

            lock (m_objSync)
            {
                if (m_rgQueryStates.Count > 0)
                    lHandle = m_rgQueryStates.Max(p => p.Key) + 1;

                m_rgQueryStates.Add(lHandle, new Tuple<QueryState, QueryState>(training, testing));
            }

            return lHandle;
        }

        /// <summary>
        /// Free the QueryStates associated with the handle and remove it from the handle list.
        /// </summary>
        /// <param name="lHandle">Specifies the QueryState handle.</param>
        /// <returns>If found and freed, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public bool FreeQueryState(long lHandle)
        {
            if (lHandle == 0)
                return false;

            lock (m_objSync)
            {
                if (m_rgQueryStates.ContainsKey(lHandle))
                {
                    m_rgQueryStates[lHandle].Item1.Dispose();
                    m_rgQueryStates[lHandle].Item2.Dispose();
                    m_rgQueryStates.Remove(lHandle);
                    return true;
                }

                return false;
            }
        }

        /// <summary>
        /// Returns the QueryState used with thet Training data source.
        /// </summary>
        /// <param name="lHandle">Specifies the QueryState handle.</param>
        /// <returns>The training set QueryState is returned.</returns>
        public QueryState GetTrainingState(long lHandle)
        {
            if (lHandle == 0)
                throw new Exception("Invalid training QueryState handle, cannot = 0.");

            lock (m_objSync)
            {
                return m_rgQueryStates[lHandle].Item1;
            }
        }

        /// <summary>
        /// Returns the QueryState used with thet Testing data source.
        /// </summary>
        /// <param name="lHandle">Specifies the QueryState handle.</param>
        /// <returns>The testing set QueryState is returned.</returns>
        public QueryState GetTestingState(long lHandle)
        {
            if (lHandle == 0)
                throw new Exception("Invalid testing QueryState handle, cannot = 0.");

            lock (m_objSync)
            {
                return m_rgQueryStates[lHandle].Item2;
            }
        }

        /// <summary>
        /// Relabels the training QueryState based on the DbItems.
        /// </summary>
        /// <param name="rgItems">Specifies the DbItems to use to relabel the QueryState.</param>
        public void ReIndexTraining(List<DbItem> rgItems)
        {
            lock (m_objSync)
            {
                foreach (KeyValuePair<long, Tuple<QueryState, QueryState>> kv in m_rgQueryStates)
                {
                    if (kv.Value != null)
                        kv.Value.Item1.Reload(rgItems);
                }
            }
        }

        /// <summary>
        /// Relabels the testing QueryState based on the DbItems.
        /// </summary>
        /// <param name="rgItems">Specifies the DbItems to use to relabel the QueryState.</param>
        public void ReIndexTesting(List<DbItem> rgItems)
        {
            lock (m_objSync)
            {
                foreach (KeyValuePair<long, Tuple<QueryState, QueryState>> kv in m_rgQueryStates)
                {
                    if (kv.Value != null)
                        kv.Value.Item2.Reload(rgItems);
                }
            }
        }
    }
}
