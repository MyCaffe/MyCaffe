using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// /b DEPRECIATED (use DataLayer DataLabelMappingParameter instead) The LabelMappingLayer converts original labels to new labels specified by the label mapping.
    /// This layer is initialized with the MyCaffe.param.LabelMappingParameter.
    /// </summary>
    /// <remarks>
    /// The LabelMappingLayer is a neuron layer attached to the data layer.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LabelMappingLayer<T> : NeuronLayer<T>
    {
        IXImageDatabaseBase m_db;
        string m_strSource = null;
        int m_nSourceId = 0;
        int m_nProjectID = 0;
        Dictionary<int, int> m_rgActualMappedLabelCounts = new Dictionary<int, int>();
        object m_syncActualMappedLabels = new object();
        int m_nLabelCount = 0;

        /// <summary>
        /// The InnerProductLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter labelmapping_param, with options:
        ///   - mapping. Defines the mappings from original label to new label.
        ///   
        ///   - update_database (/b optional, default = false). Whether or not to update the database with the new mapping, otherwise mappings are used online.
        ///   
        ///   - reset_database_labels (/b optional, default = false). Whether or not to reset the database to its original labels or not. This setting requires 'update_database' = true.
        ///   
        ///   - label_boosts (/b optional, default = ""). A string that defines which labels to boost, giving them a higher probability of being selected.
        /// </param>
        /// <param name="db">Specifies the MyCaffeImageDatabase.</param>
        public LabelMappingLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXDatabaseBase db)
            : base(cuda, log, p)
        {
            if (db.GetVersion() != DB_VERSION.IMG_V1 && db.GetVersion() != DB_VERSION.IMG_V2)
                throw new Exception("Currently only image databases are supported by the LabelMappingLayer.");

            m_db = (IXImageDatabaseBase)db;
            m_type = LayerParameter.LayerType.LABELMAPPING;
        }

        /// <summary>
        /// Set the parameters needed from the Net, namely the data source used.
        /// </summary>
        /// <param name="np">Specifies the NetParameter used.</param>
        public override void SetNetParameterUsed(NetParameter np)
        {
            base.SetNetParameterUsed(np);

            m_strSource = null;
            m_nSourceId = 0;
            m_nProjectID = np.ProjectID;

            foreach (LayerParameter p in np.layer)
            {
                if (p.type == LayerParameter.LayerType.DATA)
                {
                    m_strSource = p.data_param.source;
                    break;
                }
            }

            if (m_strSource != null)
                m_nSourceId = m_db.GetSourceID(m_strSource);
        }

        /// <summary>
        /// Returns a string describing the actual label counts observed during training.
        /// </summary>
        /// <param name="strSrc">Specifies the data source to query.</param>
        /// <returns>The string describing the actual label counts observed is returned.</returns>
        public string GetActualLabelCounts(string strSrc)
        {
            if (m_nLabelCount == 0)
                m_nLabelCount = m_db.GetLabels(m_nSourceId).Count;

            List<KeyValuePair<int, int>> rgKv;

            lock (m_syncActualMappedLabels)
            {
                rgKv = m_rgActualMappedLabelCounts.OrderBy(p => p.Key).ToList();
            }

            string str = "";
            int nIdx = 0;

            for (int i = 0; i < m_nLabelCount; i++)
            {
                if (nIdx < rgKv.Count)
                {
                    str += rgKv[nIdx].Key.ToString() + "->" + rgKv[nIdx].Value.ToString();
                    nIdx++;
                }
                else
                {
                    str += "0";
                }

                str += ", ";
            }

            return str.TrimEnd(',', ' ');
        }


        /// <summary>
        /// The LayerSetUp method adjusts the label boost values according to the number
        /// of mappings made to each label.
        /// </summary>
        /// <remarks>
        /// The basic idea is to even out the image selection between all mapped and all
        /// non mapped categories.
        /// <br/>
        /// So for example, given the following mappings:
        /// <code>
        /// mapping                boost
        /// %-----------------------------------------------
        /// 1 -> 3                 7/3 = 2.33
        /// 2 -> 3                 7/3 = 2.33
        /// 3 (no mapping)         7/3 = 2.33
        /// 4 (no mapping)         7   = 7.00
        /// 5 (no mapping)         7/3 = 2.33
        /// 6 -> 5                 7/3 = 2.33
        /// 7 -> 5                 7/3 = 2.33
        /// </code>
        /// <br/>
        /// This evens out and makes it equally probable to get an image
        /// from 1,2,3 vs 4 vs 5,6,7 thus treating the images as though
        /// they actually only had the labels 3, 4 and 5.
        /// <br/>
        /// Note, mappings now support mapping conditionally only when
        /// an image has a specific boost value.  To use conditional
        /// mapping use the following format:
        /// <br/>
        /// "FromLabel->ToLabel?boost=1"        // i.e. only map to the 'ToLabel' if the boost = 1.
        /// <br/>
        /// For example:
        /// <br/>
        /// "1->2?boost=1"                      // only map from 1 to 2 if boost = 1 on the image.     
        /// <br/>
        /// NOTE: spaces are not allowed between the 'ToLabel', the '?', and the 'boost=', etc.
        /// </remarks>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            if (m_strSource != null && m_nProjectID > 0)
            {
                int nSrcId = m_nSourceId;

                if (nSrcId > 0)
                {
                    Dictionary<int, int> rgMapCounts = new Dictionary<int, int>();
                    Dictionary<int, List<int>> rgMaps = new Dictionary<int, List<int>>();
                    bool bHasConditions = false;

                    foreach (LabelMapping map in m_param.labelmapping_param.mapping)
                    {
                        if (!rgMapCounts.Keys.Contains(map.NewLabel))
                            rgMapCounts.Add(map.NewLabel, 2);
                        else
                            rgMapCounts[map.NewLabel]++;

                        if (!rgMaps.Keys.Contains(map.NewLabel))
                            rgMaps.Add(map.NewLabel, new List<int>());

                        rgMaps[map.NewLabel].Add(map.OriginalLabel);

                        if (map.ConditionBoostEquals.HasValue)
                            bHasConditions = true;
                    }

                    if (m_param.labelmapping_param.update_database)
                    {
                        string strLabelCounts = m_db.GetLabelCountsAsTextFromSourceId(nSrcId);

                        if (m_param.labelmapping_param.reset_database_labels)
                        {
                            m_log.WriteLine("Resetting global relabeling to original labels for source '" + m_strSource + "'...");
                            m_db.ResetLabels(m_nProjectID, nSrcId);
                        }

                        if (rgMaps.Count > 0)
                            m_log.WriteLine("Starting global relabeling for source '" + m_strSource + "'...");

                        if (bHasConditions)
                        {
                            foreach (LabelMapping map in m_param.labelmapping_param.mapping)
                            {
                                m_db.SetLabelMapping(nSrcId, map);
                            }
                        }
                        else
                        {
                            foreach (KeyValuePair<int, List<int>> kv in rgMaps)
                            {
                                m_db.UpdateLabelMapping(nSrcId, kv.Key, kv.Value);
                            }
                        }

                        m_db.UpdateLabelCounts(m_nProjectID, nSrcId);

                        if (rgMaps.Count > 0)
                            m_log.WriteLine("Global relabeling completed for source '" + m_strSource + "'...");

                        if (m_param.labelmapping_param.label_boosts != null && m_param.labelmapping_param.label_boosts.Length > 0)
                        {
                            IXImageDatabase1 db = m_db as IXImageDatabase1;
                            bool bReloadImageSet = false;

                            if (db != null)
                            {
#warning ImageDatabase version 1 Only
                                string strLabelBoosts = db.GetLabelBoostsAsTextFromProject(m_nProjectID, nSrcId);
                                Dictionary<int, int> rgCounts = db.LoadLabelCounts(nSrcId);
                                string[] rgstrBoosts = m_param.labelmapping_param.label_boosts.Split(',');
                                Dictionary<int, int> rgBoostedLabelCounts = new Dictionary<int, int>();
                                double dfTotal = 0;

                                foreach (string strLabel in rgstrBoosts)
                                {
                                    int nLabel = int.Parse(strLabel);

                                    if (rgCounts.ContainsKey(nLabel))
                                    {
                                        int nCount = rgCounts[nLabel];
                                        rgBoostedLabelCounts.Add(nLabel, nCount);
                                        dfTotal += nCount;
                                    }
                                    else
                                    {
                                        rgBoostedLabelCounts.Add(nLabel, 0);
                                    }
                                }

                                db.DeleteLabelBoosts(m_nProjectID, nSrcId);

                                foreach (KeyValuePair<int, int> kv in rgCounts)
                                {
                                    double dfBoost = 0;

                                    if (rgBoostedLabelCounts.ContainsKey(kv.Key))
                                    {
                                        int nCount = rgBoostedLabelCounts.Count;
                                        dfBoost = (nCount == 0) ? 0 : (1.0 / (double)nCount);
                                    }

                                    db.AddLabelBoost(m_nProjectID, nSrcId, kv.Key, dfBoost);
                                }

                                string strNewLabelBoosts = db.GetLabelBoostsAsTextFromProject(m_nProjectID, nSrcId);
                                if (strNewLabelBoosts != strLabelBoosts)
                                    bReloadImageSet = true;
                            }

                            if (db != null)
                            {
                                string strNewLabelCounts = db.GetLabelCountsAsTextFromSourceId(nSrcId);

                                if (strNewLabelCounts != strLabelCounts || bReloadImageSet)
                                    db.ReloadImageSet(nSrcId);
                            }

                            m_log.WriteLine("WARNING: Label boosts are depreciated and soon to be removed.");
                        }
                        else
                        {
                            m_log.WriteLine("WARNING: ImageDatabase Version 2 currently does not support label mapping.");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Converts the input label to the new label specified by the label mapping.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the inputs x</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the newly mapped labels.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();

            m_log.CHECK_EQ(nCount, colBottom[0].num, "The count should equal the number of items for the label blobs.");
            m_log.CHECK_EQ(nCount, colTop[0].count(), "The top and bottom should have the same number of items.");

            double[] rgBottom = convertD(colBottom[0].update_cpu_data());

            for (int i = 0; i < rgBottom.Length; i++)
            {
                int nLabel = m_param.labelmapping_param.MapLabel((int)rgBottom[i]);
                rgBottom[i] = nLabel;

                lock (m_syncActualMappedLabels)
                {
                    if (!m_rgActualMappedLabelCounts.ContainsKey(nLabel))
                        m_rgActualMappedLabelCounts.Add(nLabel, 1);
                    else
                        m_rgActualMappedLabelCounts[nLabel]++;
                }
            }

            colTop[0].mutable_cpu_data = convert(rgBottom);
        }

        /// @brief Not implemented - The LabelMappingLayer does not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
