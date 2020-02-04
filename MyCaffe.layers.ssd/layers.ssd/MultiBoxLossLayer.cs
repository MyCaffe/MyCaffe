using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.ssd;

namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// The MultiBoxLossLayer performs multibox operations including the following:
    /// 
    /// - decode the predictions.
    /// - perform matching between priors/predictions and ground truth.
    /// - use matched boxes and confidences to compute loss.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MultiBoxLossLayer<T> : LossLayer<T>
    {
        // The internal localization loss layer.
        Layer<T> m_locLossLayer;
        MultiBoxLossParameter.LocLossType m_locLossType;
        float m_fLocWeight;
        // Bottom vector holder used in forward function.
        BlobCollection<T> m_colLocBottom = new BlobCollection<T>();
        // Top vector holder used in forward function.
        BlobCollection<T> m_colLocTop = new BlobCollection<T>();
        // Blob which stores the matched location prediction.
        Blob<T> m_blobLocPred;
        // Blob which stores the corresponding matched ground truth.
        Blob<T> m_blobLocGt;
        // Localization loss.
        Blob<T> m_blobLocLoss;

        // The internal confidence loss layer.
        Layer<T> m_confLossLayer;
        MultiBoxLossParameter.ConfLossType m_confLossType;
        // Bottom vector holder used in the forward function.
        BlobCollection<T> m_colConfBottom = new BlobCollection<T>();
        // Top vector holder used in the forward function.
        BlobCollection<T> m_colConfTop = new BlobCollection<T>();
        // Blob which stores the confidence prediction.
        Blob<T> m_blobConfPred;
        // Blob which stores the corresponding ground truth.
        Blob<T> m_blobConfGt;
        // Confidence loss.
        Blob<T> m_blobConfLoss;

        HostBuffer<T> m_hostGt;
        HostBuffer<T> m_hostLoc;
        HostBuffer<T> m_hostConf;
        HostBuffer<T> m_hostPrio;

        int m_nNumClasses;
        bool m_bShareLocation;
        int m_nBackgroundLabelId;
        bool m_bUseDifficultGt;
        bool m_bDoNegMining;
        MultiBoxLossParameter.MiningType m_miningType;

        int m_nLocClasses;
        int m_nNumGt;
        int m_nNum;
        int m_nNumPriors;

        int m_nNumMatches;
        int m_nNumConf;

        List<DictionaryMap<List<int>>> m_rgAllMatchIndices = new List<DictionaryMap<List<int>>>();
        List<List<int>> m_rgrgAllNegIndices = new List<List<int>>();
        BBoxUtility<T> m_bboxUtil;
        long m_hSsd = 0;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides MultiBoxLossParameter multiboxloss_param
        /// with MultiBoxLossLayer.
        /// </param>
        public MultiBoxLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MULTIBOX_LOSS;

            m_blobLocPred = new Blob<T>(cuda, log);
            m_blobLocPred.Name = "loc_pred";
            m_blobLocGt = new Blob<T>(cuda, log);
            m_blobLocGt.Name = "loc_gt";
            m_blobLocLoss = new Blob<T>(cuda, log);
            m_blobLocLoss.Name = "loc_loss";

            m_blobConfPred = new Blob<T>(cuda, log);
            m_blobConfPred.Name = "conf_pred";
            m_blobConfGt = new Blob<T>(cuda, log);
            m_blobConfGt.Name = "conf_gt";
            m_blobConfLoss = new Blob<T>(cuda, log);
            m_blobConfLoss.Name = "conf_loss";

            m_bboxUtil = new BBoxUtility<T>(cuda, log);

            m_hostConf = new HostBuffer<T>(cuda);
            m_hostLoc = new HostBuffer<T>(cuda);
            m_hostGt = new HostBuffer<T>(cuda);
            m_hostPrio = new HostBuffer<T>(cuda);
        }

        private void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        /// <summary>
        /// Release any resources used.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobLocPred);
            dispose(ref m_blobLocGt);
            dispose(ref m_blobLocLoss);
            dispose(ref m_blobConfPred);
            dispose(ref m_blobConfGt);
            dispose(ref m_blobConfLoss);

            if (m_hSsd != 0)
            {
                m_cuda.FreeSSD(m_hSsd);
                m_hSsd = 0;
            }

            if (m_locLossLayer != null)
            {
                m_locLossLayer.Dispose();
                m_locLossLayer = null;
            }

            if (m_confLossLayer != null)
            {
                m_confLossLayer.Dispose();
                m_confLossLayer = null;
            }

            if (m_bboxUtil != null)
            {
                m_bboxUtil.Dispose();
                m_bboxUtil = null;
            }

            if (m_hostConf != null)
            {
                m_hostConf.Dispose();
                m_hostConf = null;
            }

            if (m_hostLoc != null)
            {
                m_hostLoc.Dispose();
                m_hostLoc = null;
            }

            if (m_hostGt != null)
            {
                m_hostGt.Dispose();
                m_hostGt = null;
            }

            if (m_hostPrio != null)
            {
                m_hostPrio.Dispose();
                m_hostPrio = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Returns the internal blobs of this layer.
        /// </summary>
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                {
                    BlobCollection<T> col = new BlobCollection<T>();

                    col.Add(m_blobConfPred);
                    col.Add(m_blobLocGt);
                    col.Add(m_blobLocLoss);
                    col.Add(m_blobConfPred);
                    col.Add(m_blobConfGt);
                    col.Add(m_blobConfLoss);

                    return col;
                }
            }
        }

        /// <summary>
        /// Returns the exact number of bottom blobs required: input
        /// </summary>
        /// <remarks>
        /// bottom[0] stores the location predictions.
        /// bottom[1] stores the confidence predictions
        /// bottom[2] stores the prior bounding boxes.
        /// bottom[3] stores the ground truth bounding boxes.
        /// </remarks>
        public override int ExactNumBottomBlobs
        {
            get { return 4; }
        }

        /// <summary>
        /// Returns the exact number of top blobs required: argmax
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            if (m_param.propagate_down.Count == 0)
            {
                m_param.propagate_down.Add(true);
                m_param.propagate_down.Add(true);
                m_param.propagate_down.Add(false);
                m_param.propagate_down.Add(false);
            }

            m_nNum = colBottom[0].num;
            m_nNumPriors = colBottom[2].height / 4;

            // Get other parameters.
            m_nNumClasses = (int)m_param.multiboxloss_param.num_classes;
            m_log.CHECK_GE(m_nNumClasses, 1, "The num_classes should not be less than 1.");

            m_bShareLocation = m_param.multiboxloss_param.share_location;
            m_nLocClasses = (m_bShareLocation) ? 1 : m_nNumClasses;

            m_nBackgroundLabelId = (int)m_param.multiboxloss_param.background_label_id;
            m_bUseDifficultGt = m_param.multiboxloss_param.use_difficult_gt;
            m_miningType = m_param.multiboxloss_param.mining_type;

            if (m_param.multiboxloss_param.do_neg_mining.HasValue)
            {
                m_log.WriteLine("WARNING: do_neg_mining is depreciated, use mining_type instead.");
                m_bDoNegMining = m_param.multiboxloss_param.do_neg_mining.Value;
                m_log.CHECK(m_bDoNegMining == (m_miningType != MultiBoxLossParameter.MiningType.NONE), "The mining type specified is inconsistent with do_neg_mining.");
            }

            m_bDoNegMining = (m_miningType != MultiBoxLossParameter.MiningType.NONE);

            if (m_bDoNegMining)
                m_log.CHECK(m_bShareLocation, "Currently only support negative mining if share_location is true.");

            // Setup localization loss layer.
            m_fLocWeight = m_param.multiboxloss_param.loc_weight;
            m_locLossType = m_param.multiboxloss_param.loc_loss_type;
            // fake shape
            List<int> rgLocShape = Utility.Create<int>(1, 1);
            rgLocShape.Add(4);
            m_blobLocPred.Reshape(rgLocShape);
            m_blobLocGt.Reshape(rgLocShape);
            m_colLocBottom.Add(m_blobLocPred);
            m_colLocBottom.Add(m_blobLocGt);

            List<int> rgLossShape = Utility.Create<int>(1, 1);
            m_blobLocLoss.Reshape(rgLossShape);
            m_colLocTop.Add(m_blobLocLoss);

            if (m_locLossType == MultiBoxLossParameter.LocLossType.L2)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                p.name += "_l2_loc";
                p.loss_weight.Add(m_fLocWeight);
                m_locLossLayer = Layer<T>.Create(m_cuda, m_log, p, null);
                m_locLossLayer.Setup(m_colLocBottom, m_colLocTop);
            }
            else if (m_locLossType == MultiBoxLossParameter.LocLossType.SMOOTH_L1)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SMOOTHL1_LOSS);
                p.name += "_smooth_l1_loc";
                p.loss_weight.Add(m_fLocWeight);
                m_locLossLayer = Layer<T>.Create(m_cuda, m_log, p, null);
                m_locLossLayer.Setup(m_colLocBottom, m_colLocTop);
            }
            else
            {
                m_log.FAIL("Unknown localization loss type.");
            }

            // Setup confidence loss layer.
            m_confLossType = m_param.multiboxloss_param.conf_loss_type;
            m_colConfBottom.Add(m_blobConfPred);
            m_colConfBottom.Add(m_blobConfGt);
            m_blobConfLoss.Reshape(rgLossShape);
            m_colConfTop.Add(m_blobConfLoss);

            List<int> rgConfShape = Utility.Create<int>(1, 1);

            if (m_confLossType == MultiBoxLossParameter.ConfLossType.SOFTMAX)
            {
                m_log.CHECK_GE(m_nBackgroundLabelId, 0, "The background_label_id should be within [0, num_classes) for Softmax.");
                m_log.CHECK_LT(m_nBackgroundLabelId, m_nNumClasses, "The background_label_id should be within [0, num_classes) for Softmax.");
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
                p.name += "_softmax_conf";
                p.loss_weight.Add(1);
                p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
                p.softmax_param.axis = 1;

                // Fake shape.
                m_blobConfGt.Reshape(rgConfShape);
                rgConfShape.Add(m_nNumClasses);
                m_blobConfPred.Reshape(rgConfShape);
                m_confLossLayer = Layer<T>.Create(m_cuda, m_log, p, null);
                m_confLossLayer.Setup(m_colConfBottom, m_colConfTop);
            }
            else if (m_confLossType == MultiBoxLossParameter.ConfLossType.LOGISTIC)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOIDCROSSENTROPY_LOSS);
                p.name += "_logistic_conf";
                p.loss_weight.Add(1);

                // Fake shape
                rgConfShape.Add(m_nNumClasses);
                m_blobConfGt.Reshape(rgConfShape);
                m_blobConfPred.Reshape(rgConfShape);
                m_confLossLayer = Layer<T>.Create(m_cuda, m_log, p, null);
                m_confLossLayer.Setup(m_colConfBottom, m_colConfTop);
            }
            else
            {
                m_log.FAIL("Unknown confidence loss type.");
            }

            // Create the low-level SSD support.
            if (m_param.multiboxloss_param.use_gpu)
            {
                float? fNmsThreshold = null;
                int? nNmsTopK = null;
                float? fNmsEta = null;
                bool bNmsParam = false;

                if (m_param.multiboxloss_param.nms_param != null && m_param.multiboxloss_param.nms_param.Active)
                {
                    bNmsParam = true;
                    fNmsThreshold = m_param.multiboxloss_param.nms_param.nms_threshold;
                    nNmsTopK = m_param.multiboxloss_param.nms_param.top_k;
                    fNmsEta = m_param.multiboxloss_param.nms_param.eta;
                }

                m_hSsd = m_cuda.CreateSSD(m_nNumClasses,
                                          m_bShareLocation,
                                          m_nLocClasses,
                                          m_nBackgroundLabelId,
                                          m_bUseDifficultGt,
                                          (SSD_MINING_TYPE)(int)m_miningType,
                                          (SSD_MATCH_TYPE)(int)m_param.multiboxloss_param.match_type,
                                          m_param.multiboxloss_param.overlap_threshold,
                                          m_param.multiboxloss_param.use_prior_for_matching,
                                          (SSD_CODE_TYPE)(int)m_param.multiboxloss_param.code_type,
                                          m_param.multiboxloss_param.encode_variance_in_target,
                                          m_param.multiboxloss_param.bp_inside,
                                          m_param.multiboxloss_param.ignore_cross_boundary_bbox,
                                          m_param.multiboxloss_param.use_prior_for_nms,
                                          (SSD_CONF_LOSS_TYPE)(int)m_param.multiboxloss_param.conf_loss_type,
                                          (SSD_LOC_LOSS_TYPE)(int)m_param.multiboxloss_param.loc_loss_type,
                                          m_param.multiboxloss_param.neg_pos_ratio,
                                          m_param.multiboxloss_param.neg_overlap,
                                          m_param.multiboxloss_param.sample_size,
                                          m_param.multiboxloss_param.map_object_to_agnostic,
                                          bNmsParam,
                                          fNmsThreshold,
                                          nNmsTopK,
                                          fNmsEta);
                if (m_hSsd == 0)
                    throw new Exception("Could not create the SSD!");
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_nNum = colBottom[0].num;
            m_nNumPriors = colBottom[2].height / 4;
            m_nNumGt = colBottom[3].height;

            if (m_param.multiboxloss_param.use_gpu)
                m_cuda.SetupSSD(m_hSsd, m_nNum, m_nNumPriors, m_nNumGt);

            m_log.CHECK_EQ(colBottom[0].num, colBottom[1].num, "The bottom[0] and bottom[1] num must be equal.");
            m_log.CHECK_EQ(m_nNumPriors * m_nLocClasses * 4, colBottom[0].channels, "The number of priors must match the number of location predictions.");
            m_log.CHECK_EQ(m_nNumPriors * m_nNumClasses, colBottom[1].channels, "The number of priors must match the number of confidence predictions.");
        }

        /// <summary>
        /// Forward GPU computation.
        /// </summary>
        /// <param name="colBottom">input blob vector.</param>
        /// <param name="colTop">output blob vector.</param>
        /// <remarks>
        /// Work in progress - NOT COMPLETE YET.
        /// </remarks>
        protected void forwardGpu(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Calculate the Loc and Conf predictions
            int nLocDataCount = colBottom[0].count();
            long hLocGpuData = colBottom[0].gpu_data;
            int nConfDataCount = colBottom[1].count();
            long hConfGpuData = colBottom[1].gpu_data;
            int nPriorDataCount = colBottom[2].count();
            long hPriorGpuData = colBottom[2].gpu_data;
            int nGtDataCount = colBottom[3].count();
            long hGtGpuData = colBottom[3].gpu_data;
            int nNumNegs;

            m_nNumMatches = m_cuda.SsdMultiBoxLossForward(m_hSsd, nLocDataCount, hLocGpuData, nConfDataCount, hConfGpuData, nPriorDataCount, hPriorGpuData, nGtDataCount, hGtGpuData, out m_rgAllMatchIndices, out m_rgrgAllNegIndices, out nNumNegs);

            // Retrieve all ground truth.
            // m_hostGt.CopyFrom(colBottom[3]);
            // float[] rgfGtData = m_hostGt.GetHostDataAsFloat();
            // DictionaryMap<List<NormalizedBBox>> rgAllGtBboxes = m_bboxUtil.GetGroundTruth(rgfGtData, m_nNumGt, m_nBackgroundLabelId, m_bUseDifficultGt);

            // Retrieve all prior bboxes. It is the same within a batch since we assume all
            // images in a batch are of the same dimension.
            // List<List<float>> rgrgPriorVariances;
            // m_hostPrio.CopyFrom(colBottom[2]);
            // float[] rgfPriorData = m_hostPrio.GetHostDataAsFloat();
            // List<NormalizedBBox> rgPriorBboxes = m_bboxUtil.GetPrior(rgfPriorData, m_nNumPriors, out rgrgPriorVariances);

            // Retrieve all predictions.
            // m_hostLoc.CopyFrom(colBottom[0]);
            // float[] rgfLocData = m_hostLoc.GetHostDataAsFloat();
            // List<LabelBBox> rgAllLocPreds = m_bboxUtil.GetLocPredictions(rgfLocData, m_nNum, m_nNumPriors, m_nLocClasses, m_bShareLocation);

            // Find matches between source bboxes and ground truth bboxes.
            // List<DictionaryMap<List<float>>> rgAllMatchOverlaps;
            // m_bboxUtil.FindMatches(rgAllLocPreds, rgAllGtBboxes, rgPriorBboxes, rgrgPriorVariances, m_param.multiboxloss_param, out rgAllMatchOverlaps, out m_rgAllMatchIndices);

            // Sample hard negative (and positive) examples based on mining type.
            // int nNumNegs;
            // m_nNumMatches = m_bboxUtil.MineHardExamples(colBottom[1], rgAllLocPreds, rgAllGtBboxes, rgPriorBboxes, rgrgPriorVariances, rgAllMatchOverlaps, m_param.multiboxloss_param, m_rgAllMatchIndices, m_rgrgAllNegIndices, out nNumNegs);

            if (m_nNumMatches >= 1)
            {
                // Form data to pass on to loc_loss_layer.
                List<int> rgLocShape = new List<int>() { 1, m_nNumMatches * 4 };
                m_blobLocPred.Reshape(rgLocShape);
                m_blobLocGt.Reshape(rgLocShape);

                m_cuda.SsdEncodeLocPrediction(m_hSsd, m_blobLocPred.count(), m_blobLocPred.mutable_gpu_data, m_blobLocGt.count(), m_blobLocGt.mutable_gpu_data);

            //  m_bboxUtil.EncodeLocPrediction(rgAllLocPreds, rgAllGtBboxes, m_rgAllMatchIndices, rgPriorBboxes, rgrgPriorVariances, m_param.multiboxloss_param, m_blobLocPred, m_blobLocGt);

                m_locLossLayer.Reshape(m_colLocBottom, m_colLocTop);
                m_locLossLayer.Forward(m_colLocBottom, m_colLocTop);
            }
            else
            {
                m_blobLocLoss.SetData(0, 0);
            }

            // Form data to pass on to conf_loss_layer
            if (m_bDoNegMining)
                m_nNumConf = m_nNumMatches + nNumNegs;
            else
                m_nNumConf = m_nNum * m_nNumPriors;

            if (m_nNumConf >= 1)
            {
                // Reshape the confidence data.
                List<int> rgConfShape = new List<int>();

                if (m_confLossType == MultiBoxLossParameter.ConfLossType.SOFTMAX)
                {
                    rgConfShape.Add(m_nNumConf);
                    m_blobConfGt.Reshape(rgConfShape);
                    rgConfShape.Add(m_nNumClasses);
                    m_blobConfPred.Reshape(rgConfShape);
                }
                else if (m_confLossType == MultiBoxLossParameter.ConfLossType.LOGISTIC)
                {
                    rgConfShape.Add(1);
                    rgConfShape.Add(m_nNumConf);
                    rgConfShape.Add(m_nNumClasses);
                    m_blobConfGt.Reshape(rgConfShape);
                    m_blobConfPred.Reshape(rgConfShape);
                }
                else
                {
                    m_log.FAIL("Unknown confidence loss type.");
                }

                if (!m_bDoNegMining)
                {
                    // Consider all scores.
                    // Share data and diff with bottom[1].
                    m_log.CHECK_EQ(m_blobConfPred.count(), colBottom[1].count(), "The conf pred and bottom[1] should have the same count.");
                    m_blobConfPred.ShareData(colBottom[1]);
                }

                m_blobConfGt.SetData(m_nBackgroundLabelId);

                m_cuda.SsdEncodeConfPrediction(m_hSsd, m_blobConfPred.count(), m_blobConfPred.mutable_gpu_data, m_blobConfGt.count(), m_blobConfGt.mutable_gpu_data);

                // m_hostConf.CopyFrom(colBottom[1]);
                // float[] rgfConfData = m_hostConf.GetHostDataAsFloat();
                // m_bboxUtil.EncodeConfPrediction(rgfConfData, m_nNum, m_nNumPriors, m_param.multiboxloss_param, m_rgAllMatchIndices, m_rgrgAllNegIndices, rgAllGtBboxes, m_blobConfPred, m_blobConfGt);
                m_confLossLayer.Reshape(m_colConfBottom, m_colConfTop);
                m_confLossLayer.Forward(m_colConfBottom, m_colConfTop);
            }
            else
            {
                m_blobConfLoss.SetData(0, 0);
            }

            colTop[0].SetData(0, 0);

            if (m_param.propagate_down[0])
            {
                double dfNormalizer = GetNormalizer(m_param.loss_param.normalization.Value, m_nNum, m_nNumPriors, m_nNumMatches);
                double dfLocLoss = Utility.ConvertVal<T>(m_blobLocLoss.GetData(0));
                double dfLoss = Utility.ConvertVal<T>(colTop[0].GetData(0));

                dfLoss += m_fLocWeight * dfLocLoss / dfNormalizer;
                colTop[0].SetData(dfLoss, 0);
            }

            if (m_param.propagate_down[1])
            {
                double dfNormalizer = GetNormalizer(m_param.loss_param.normalization.Value, m_nNum, m_nNumPriors, m_nNumMatches);
                double dfConfLoss = Utility.ConvertVal<T>(m_blobConfLoss.GetData(0));
                double dfLoss = Utility.ConvertVal<T>(colTop[0].GetData(0));

                dfLoss += dfConfLoss / dfNormalizer;
                colTop[0].SetData(dfLoss, 0);
            }
        }



        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector.
        /// </param>
        /// <param name="colTop">output blob vector.
        /// </param>
        protected void forwardCpu(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Retrieve all ground truth.
            m_hostGt.CopyFrom(colBottom[3]);
            float[] rgfGtData = m_hostGt.GetHostDataAsFloat();
            DictionaryMap<List<NormalizedBBox>> rgAllGtBboxes = m_bboxUtil.GetGroundTruth(rgfGtData, m_nNumGt, m_nBackgroundLabelId, m_bUseDifficultGt);

            // Retrieve all prior bboxes. It is the same within a batch since we assume all
            // images in a batch are of the same dimension.
            List<List<float>> rgrgPriorVariances;
            m_hostPrio.CopyFrom(colBottom[2]);
            float[] rgfPriorData = m_hostPrio.GetHostDataAsFloat();
            List<NormalizedBBox> rgPriorBboxes = m_bboxUtil.GetPrior(rgfPriorData, m_nNumPriors, out rgrgPriorVariances);

            // Retrieve all predictions.
            m_hostLoc.CopyFrom(colBottom[0]);
            float[] rgfLocData = m_hostLoc.GetHostDataAsFloat();
            List<LabelBBox> rgAllLocPreds = m_bboxUtil.GetLocPredictions(rgfLocData, m_nNum, m_nNumPriors, m_nLocClasses, m_bShareLocation);

            // Find matches between source bboxes and ground truth bboxes.
            List<DictionaryMap<List<float>>> rgAllMatchOverlaps;
            m_bboxUtil.FindMatches(rgAllLocPreds, rgAllGtBboxes, rgPriorBboxes, rgrgPriorVariances, m_param.multiboxloss_param, out rgAllMatchOverlaps, out m_rgAllMatchIndices);

            // Sample hard negative (and positive) examples based on mining type.
            int nNumNegs;
            m_nNumMatches = m_bboxUtil.MineHardExamples(colBottom[1], rgAllLocPreds, rgAllGtBboxes, rgPriorBboxes, rgrgPriorVariances, rgAllMatchOverlaps, m_param.multiboxloss_param, m_rgAllMatchIndices, m_rgrgAllNegIndices, out nNumNegs);

            if (m_nNumMatches >= 1)
            {
                // Form data to pass on to loc_loss_layer.
                List<int> rgLocShape = new List<int>() { 1, m_nNumMatches * 4 };
                m_blobLocPred.Reshape(rgLocShape);
                m_blobLocGt.Reshape(rgLocShape);

                m_bboxUtil.EncodeLocPrediction(rgAllLocPreds, rgAllGtBboxes, m_rgAllMatchIndices, rgPriorBboxes, rgrgPriorVariances, m_param.multiboxloss_param, m_blobLocPred, m_blobLocGt);

                m_locLossLayer.Reshape(m_colLocBottom, m_colLocTop);
                m_locLossLayer.Forward(m_colLocBottom, m_colLocTop);
            }
            else
            {
                m_blobLocLoss.SetData(0, 0);
            }

            // Form data to pass on to conf_loss_layer
            if (m_bDoNegMining)
                m_nNumConf = m_nNumMatches + nNumNegs;
            else
                m_nNumConf = m_nNum * m_nNumPriors;

            if (m_nNumConf >= 1)
            {
                // Reshape the confidence data.
                List<int> rgConfShape = new List<int>();

                if (m_confLossType == MultiBoxLossParameter.ConfLossType.SOFTMAX)
                {
                    rgConfShape.Add(m_nNumConf);
                    m_blobConfGt.Reshape(rgConfShape);
                    rgConfShape.Add(m_nNumClasses);
                    m_blobConfPred.Reshape(rgConfShape);
                }
                else if (m_confLossType == MultiBoxLossParameter.ConfLossType.LOGISTIC)
                {
                    rgConfShape.Add(1);
                    rgConfShape.Add(m_nNumConf);
                    rgConfShape.Add(m_nNumClasses);
                    m_blobConfGt.Reshape(rgConfShape);
                    m_blobConfPred.Reshape(rgConfShape);
                }
                else
                {
                    m_log.FAIL("Unknown confidence loss type.");
                }

                if (!m_bDoNegMining)
                {
                    // Consider all scores.
                    // Share data and diff with bottom[1].
                    m_log.CHECK_EQ(m_blobConfPred.count(), colBottom[1].count(), "The conf pred and bottom[1] should have the same count.");
                    m_blobConfPred.ShareData(colBottom[1]);
                }

                m_blobConfGt.SetData(m_nBackgroundLabelId);

                m_hostConf.CopyFrom(colBottom[1]);
                float[] rgfConfData = m_hostConf.GetHostDataAsFloat();
                m_bboxUtil.EncodeConfPrediction(rgfConfData, m_nNum, m_nNumPriors, m_param.multiboxloss_param, m_rgAllMatchIndices, m_rgrgAllNegIndices, rgAllGtBboxes, m_blobConfPred, m_blobConfGt);
                m_confLossLayer.Reshape(m_colConfBottom, m_colConfTop);
                m_confLossLayer.Forward(m_colConfBottom, m_colConfTop);
            }
            else
            {
                m_blobConfLoss.SetData(0, 0);
            }

            colTop[0].SetData(0, 0);

            if (m_param.propagate_down[0])
            {
                double dfNormalizer = GetNormalizer(m_param.loss_param.normalization.Value, m_nNum, m_nNumPriors, m_nNumMatches);
                double dfLocLoss = Utility.ConvertVal<T>(m_blobLocLoss.GetData(0));
                double dfLoss = Utility.ConvertVal<T>(colTop[0].GetData(0));

                dfLoss += m_fLocWeight * dfLocLoss / dfNormalizer;
                colTop[0].SetData(dfLoss, 0);
            }

            if (m_param.propagate_down[1])
            {
                double dfNormalizer = GetNormalizer(m_param.loss_param.normalization.Value, m_nNum, m_nNumPriors, m_nNumMatches);
                double dfConfLoss = Utility.ConvertVal<T>(m_blobConfLoss.GetData(0));
                double dfLoss = Utility.ConvertVal<T>(colTop[0].GetData(0));

                dfLoss += dfConfLoss / dfNormalizer;
                colTop[0].SetData(dfLoss, 0);
            }
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector.
        /// </param>
        /// <param name="colTop">output blob vector.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.multiboxloss_param.use_gpu)
                forwardGpu(colBottom, colTop);
            else
                forwardCpu(colBottom, colTop);
        }

        /// <summary>
        /// Computes the multibox loss error gradient w.r.t the predictions.
        /// </summary>
        /// <remarks>
        /// Gradients cannot be computed with respect to the label inputs (bottom[1]),
        /// so this method ignores bottom[1] and requires !propagate_down[1], crashing
        /// if propagate_down[1] == true.
        /// </remarks>
        /// <param name="colTop">top output blob vector, providing the error gradient with
        /// respect to the outputs.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels.</param>
        /// <param name="colBottom">bottom input blob vector
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[2])
                m_log.FAIL(m_type.ToString() + " Layer cannot backpropagate to prior inputs.");

            if (rgbPropagateDown[3])
                m_log.FAIL(m_type.ToString() + " Layer cannot backpropagate to label inputs.");

            // Back propagate on location prediction.
            if (rgbPropagateDown[0])
            {
                colBottom[0].SetDiff(0);

                if (m_nNumMatches >= 1)
                {
                    int nLocBottomDiffOffset = 0;
                    float[] rgfLocBottomDiff = Utility.ConvertVecF<T>(colBottom[0].mutable_cpu_diff);
                    List<bool> rgLocPropagateDown = new List<bool>();

                    // Only back propagate on prediction, not ground truth.
                    rgLocPropagateDown.Add(true);
                    rgLocPropagateDown.Add(false);
                    m_locLossLayer.Backward(m_colLocTop, rgLocPropagateDown, m_colLocBottom);

                    // Scale gradient.
                    double dfNormalizer = GetNormalizer(m_param.loss_param.normalization.Value, m_nNum, m_nNumPriors, m_nNumMatches);
                    double dfLossWeight = Utility.ConvertVal<T>(colTop[0].GetDiff(0)) / dfNormalizer;
                    m_cuda.scal(m_blobLocPred.count(), dfLossWeight, m_blobLocPred.mutable_gpu_diff);

                    // Copy gradient back to bottom[0];
                    float[] rgfLocPredDiff = Utility.ConvertVecF<T>(m_blobLocPred.mutable_cpu_diff);
                    int nCount = 0;

                    for (int i = 0; i < m_nNum; i++)
                    {
                        DictionaryMap<List<int>> rgMap = m_rgAllMatchIndices[i];

                        foreach (KeyValuePair<int, List<int>> kv in rgMap.Map)
                        {
                            int nLabel = (m_bShareLocation) ? 0 : kv.Key;
                            List<int> rgMatchIndex = kv.Value;

                            for (int j = 0; j < rgMatchIndex.Count; j++)
                            {
                                if (rgMatchIndex[j] <= -1)
                                    continue;

                                // Copy the diff to the right place.
                                int nStartIdx = m_nLocClasses * 4 * j + nLabel * 4;
                                Array.Copy(rgfLocPredDiff, nCount * 4, rgfLocBottomDiff, nLocBottomDiffOffset + nStartIdx, 4);
                                nCount++;
                            }
                        }

                        nLocBottomDiffOffset += colBottom[0].offset(1);
                    }

                    colBottom[0].mutable_cpu_diff = Utility.ConvertVec<T>(rgfLocBottomDiff);
                }
            }

            // Back propagate on confidence prediction
            if (rgbPropagateDown[1])
            {
                colBottom[1].SetDiff(0);

                if (m_nNumConf >= 1)
                {
                    int nConfBottomDiffOffset = 0;
                    float[] rgfConfBottomDiff = Utility.ConvertVecF<T>(colBottom[1].mutable_cpu_diff);
                    List<bool> rgConfPropagateDown = new List<bool>();

                    // Only back propagate on prediction, not ground truth.
                    rgConfPropagateDown.Add(true);
                    rgConfPropagateDown.Add(false);
                    m_locLossLayer.Backward(m_colConfTop, rgConfPropagateDown, m_colConfBottom);

                    // Scale gradient.
                    double dfNormalizer = GetNormalizer(m_param.loss_param.normalization.Value, m_nNum, m_nNumPriors, m_nNumMatches);
                    double dfLossWeight = Utility.ConvertVal<T>(colTop[0].GetDiff(0)) / dfNormalizer;
                    m_cuda.scal(m_blobConfPred.count(), dfLossWeight, m_blobConfPred.mutable_gpu_diff);

                    // Copy gradient back to bottom[1];
                    float[] rgfConfPredDiff = Utility.ConvertVecF<T>(m_blobConfPred.mutable_cpu_diff);
                    if (m_bDoNegMining)
                    {
                        int nCount = 0;

                        for (int i = 0; i < m_nNum; i++)
                        {
                            // Copy matched (positive) bboxes scores' diff.
                            Dictionary<int, List<int>> rgMap = m_rgAllMatchIndices[i].Map;

                            foreach (KeyValuePair<int, List<int>> kv in rgMap)
                            {
                                List<int> rgMatchIndex = kv.Value;
                                m_log.CHECK_EQ(rgMatchIndex.Count, m_nNumPriors, "The match index count should equal the num priors!");

                                for (int j = 0; j < m_nNumPriors; j++)
                                {
                                    if (rgMatchIndex[j] <= -1)
                                        continue;

                                    // Copy the diff to the right place.
                                    Array.Copy(rgfConfPredDiff, nCount * m_nNumClasses, rgfConfBottomDiff, nConfBottomDiffOffset + j * m_nNumClasses, m_nNumClasses);
                                    nCount++;
                                }
                            }

                            // Copy negative bboxes scores' diff
                            for (int n = 0; n < m_rgrgAllNegIndices[i].Count; n++)
                            {
                                int j = m_rgrgAllNegIndices[i][n];
                                m_log.CHECK_LT(j, m_nNumPriors, "The index must be less than the num priors!");

                                Array.Copy(rgfConfPredDiff, nCount * m_nNumClasses, rgfConfBottomDiff, nConfBottomDiffOffset + j * m_nNumClasses, m_nNumClasses);
                                nCount++;
                            }

                            nConfBottomDiffOffset += colBottom[1].offset(1);
                        }

                        colBottom[1].mutable_cpu_diff = Utility.ConvertVec<T>(rgfConfBottomDiff);
                    }
                    else
                    {
                        // The diff is already computed and stored.
                        m_cuda.copy(colBottom[1].count(), m_blobConfPred.gpu_diff, colBottom[1].mutable_gpu_diff);
                    }
                }
            }

            // After backward, remove match statistics.
            m_rgAllMatchIndices.Clear();
            m_rgrgAllNegIndices.Clear();
        }
    }
}
