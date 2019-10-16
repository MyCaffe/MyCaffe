using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.param.ssd;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.model
{
    /// <summary>
    /// The SsdPascalModelBuilder adds the extra layers to a 'base' model for the Pascal model used with SSD.
    /// </summary>
    public class SsdPascalModelBuilder : ModelBuilder
    {
        int m_nGpuID = 0;
        List<int> m_rgGpuID = new List<int>();
        int m_nBatchSize;
        int m_nAccumBatchSize;
        int m_nBatchSizePerDevice;
        int m_nIterSize;
        double m_dfBaseLr;
        double m_dfLrMult = 1;
        bool m_bUseBatchNorm = false;
        int m_nNumTestImage = 4952; // Evaluate on the whole test set.
        int m_nTestBatchSize = 8;
        int m_nTestIter;
        string m_strNameSizeFile = "data\\VOC0712\\test_name_size.txt";
        string m_strLabelMapFile = "data\\VOC0712\\labelmap_voc.prototxt";
        string m_strPreTrainModel = "models\\VGGNet\\VGG_ILSVRC_16_layers_fc_reduced.caffemodel";
        TransformationParameter m_transformTrain = null;
        TransformationParameter m_transformTest = null;
        DetectionEvaluateParameter m_detectionEval = null;
        DetectionOutputParameter m_detectionOut = null;

        // Batch sampler.
        int m_nResizeWidth = 300;
        int m_nResizeHeight = 300;
        List<BatchSampler> m_rgBatchSampler = new List<BatchSampler>();

        // MultiBoxLoss Parameters
        LossParameter.NormalizationMode m_normalizationMode = LossParameter.NormalizationMode.VALID;
        LayerParameter m_multiBoxLossLayer;
        List<float> m_rgPriorVariance;
        bool m_bFlip = true;
        bool m_bClip = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strBaseDirectory">Specifies the base directory that contains the data and models.</param>
        /// <param name="nBatchSize">Optionally, specifies the batch size (default = 32).</param>
        /// <param name="nAccumBatchSize">Optionally, specifies the accumulation batch size (default = 32).</param>
        /// <param name="rgGpuId">Optionally, specifies a set of GPU ID's to use (when null, GPU=0 is used).</param>
        /// <param name="bUseBatchNorm">Optionally, specifies to use batch normalization (default = false).</param>
        /// <param name="normMode">Optionally, specifies the normalization mode (default = VALID).</param>
        /// <param name="net">Specifies the 'base' net parameter that is to be altered.</param>
        public SsdPascalModelBuilder(string strBaseDirectory, int nBatchSize = 32, int nAccumBatchSize = 32, List<int> rgGpuId = null, bool bUseBatchNorm = false, LossParameter.NormalizationMode normMode = LossParameter.NormalizationMode.VALID, NetParameter net = null) 
            : base(strBaseDirectory, net)
        {
            if (rgGpuId == null)
                m_rgGpuID.Add(0);
            else
                m_rgGpuID = new List<int>(rgGpuId);

            m_nBatchSize = nBatchSize;
            m_nAccumBatchSize = nAccumBatchSize;
            m_nIterSize = m_nAccumBatchSize / m_nBatchSize;

            m_bUseBatchNorm = bUseBatchNorm;
            m_normalizationMode = normMode;

            // Set the MultiBoxLayer parameters.
            m_multiBoxLossLayer = new LayerParameter(LayerParameter.LayerType.MULTIBOX_LOSS);
            m_multiBoxLossLayer.multiboxloss_param.loc_loss_type = MultiBoxLossParameter.LocLossType.SMOOTH_L1;
            m_multiBoxLossLayer.multiboxloss_param.conf_loss_type = MultiBoxLossParameter.ConfLossType.SOFTMAX;
            m_multiBoxLossLayer.multiboxloss_param.neg_pos_ratio = 3.0f;
            m_multiBoxLossLayer.multiboxloss_param.num_classes = 21;
            m_multiBoxLossLayer.multiboxloss_param.loc_weight = (m_multiBoxLossLayer.multiboxloss_param.neg_pos_ratio + 1.0f) / 4.0f;
            m_multiBoxLossLayer.multiboxloss_param.share_location = true;
            m_multiBoxLossLayer.multiboxloss_param.match_type = MultiBoxLossParameter.MatchType.PER_PREDICTION;
            m_multiBoxLossLayer.multiboxloss_param.overlap_threshold = 0.5f;
            m_multiBoxLossLayer.multiboxloss_param.use_prior_for_matching = true;
            m_multiBoxLossLayer.multiboxloss_param.background_label_id = 0;
            m_multiBoxLossLayer.multiboxloss_param.use_difficult_gt = true;
            m_multiBoxLossLayer.multiboxloss_param.mining_type = MultiBoxLossParameter.MiningType.MAX_NEGATIVE;
            m_multiBoxLossLayer.multiboxloss_param.neg_overlap = 0.5f;
            m_multiBoxLossLayer.multiboxloss_param.code_type = PriorBoxParameter.CodeType.CENTER_SIZE;
            m_multiBoxLossLayer.multiboxloss_param.ignore_cross_boundary_bbox = false;
            m_multiBoxLossLayer.loss_param.normalization = m_normalizationMode;

            if (m_multiBoxLossLayer.multiboxloss_param.code_type == PriorBoxParameter.CodeType.CENTER_SIZE)
                m_rgPriorVariance = new List<float>() { 0.1f, 0.1f, 0.2f, 0.2f };
            else
                m_rgPriorVariance = new List<float>() { 0.1f };

            // Set the base learning rate.
            m_dfBaseLr = (m_bUseBatchNorm) ? 0.0004 : 0.00004;

            m_nBatchSizePerDevice = (m_rgGpuID.Count == 1) ? m_nBatchSize : m_nBatchSize / m_rgGpuID.Count;
            m_nIterSize = (int)Math.Ceiling((float)m_nAccumBatchSize / (m_nBatchSizePerDevice * m_rgGpuID.Count));
            m_nGpuID = m_rgGpuID[0];

            switch (m_normalizationMode)
            {
                case LossParameter.NormalizationMode.NONE:
                    m_dfBaseLr /= m_nBatchSizePerDevice;
                    break;

                case LossParameter.NormalizationMode.VALID:
                    m_dfBaseLr *= 25.0 / m_multiBoxLossLayer.multiboxloss_param.loc_weight;
                    break;

                case LossParameter.NormalizationMode.FULL:
                    // Roughly there are 2000 prior bboxes per images (TODO: calculate and use exact number).
                    m_dfBaseLr *= 2000;
                    break;
            }

            // Ideally the test_batch_size should be divisible by the num_test_image,
            // otherwise mAP will be slightly off the true value.
            m_nTestIter = (int)Math.Ceiling((float)m_nNumTestImage / (float)m_nTestBatchSize);

            // Create the transformer for TRAINing
            m_transformTrain = new TransformationParameter();
            m_transformTrain.mirror = true;
            m_transformTrain.mean_value.Add(104);
            m_transformTrain.mean_value.Add(117);
            m_transformTrain.mean_value.Add(123);
            m_transformTrain.resize_param = new ResizeParameter();
            m_transformTrain.resize_param.prob = 1;
            m_transformTrain.resize_param.resize_mode = ResizeParameter.ResizeMode.WARP;
            m_transformTrain.resize_param.height = (uint)m_nResizeHeight;
            m_transformTrain.resize_param.width = (uint)m_nResizeWidth;
            m_transformTrain.resize_param.interp_mode.Add(ResizeParameter.InterpMode.LINEAR);
            m_transformTrain.resize_param.interp_mode.Add(ResizeParameter.InterpMode.AREA);
            m_transformTrain.resize_param.interp_mode.Add(ResizeParameter.InterpMode.NEAREST);
            m_transformTrain.resize_param.interp_mode.Add(ResizeParameter.InterpMode.CUBIC);
            m_transformTrain.resize_param.interp_mode.Add(ResizeParameter.InterpMode.LANCZOS4);
            m_transformTrain.distortion_param = new DistortionParameter();
            m_transformTrain.distortion_param.brightness_prob = 0.5f;
            m_transformTrain.distortion_param.brightness_delta = 32;
            m_transformTrain.distortion_param.contrast_prob = 0.5f;
            m_transformTrain.distortion_param.contrast_lower = 0.5f;
            m_transformTrain.distortion_param.contrast_upper = 1.5f;
            m_transformTrain.distortion_param.saturation_prob = 0.5f;
            m_transformTrain.distortion_param.saturation_lower = 0.5f;
            m_transformTrain.distortion_param.saturation_upper = 1.5f;
            m_transformTrain.distortion_param.random_order_prob = 0.0f;
            m_transformTrain.expansion_param = new ExpansionParameter();
            m_transformTrain.expansion_param.prob = 0.5f;
            m_transformTrain.expansion_param.max_expand_ratio = 4.0f;
            m_transformTrain.emit_constraint = new EmitConstraint();
            m_transformTrain.emit_constraint.emit_type = EmitConstraint.EmitType.CENTER;

            // Create the transformer for TESTing
            m_transformTrain = new TransformationParameter();
            m_transformTrain.mean_value.Add(104);
            m_transformTrain.mean_value.Add(117);
            m_transformTrain.mean_value.Add(123);
            m_transformTrain.resize_param = new ResizeParameter();
            m_transformTrain.resize_param.prob = 1;
            m_transformTrain.resize_param.resize_mode = ResizeParameter.ResizeMode.WARP;
            m_transformTrain.resize_param.height = (uint)m_nResizeHeight;
            m_transformTrain.resize_param.width = (uint)m_nResizeWidth;
            m_transformTrain.resize_param.interp_mode.Add(ResizeParameter.InterpMode.LINEAR);

            // Create the batch samplers.
            BatchSampler sampler = createSampler(1, 1);
            m_rgBatchSampler.Add(sampler);

            sampler = createSampler(50, 1, 0.3f, 1.0f, 0.5f, 2.0f, 0.1f);
            m_rgBatchSampler.Add(sampler);

            sampler = createSampler(50, 1, 0.3f, 1.0f, 0.5f, 2.0f, 0.3f);
            m_rgBatchSampler.Add(sampler);

            sampler = createSampler(50, 1, 0.3f, 1.0f, 0.5f, 2.0f, 0.5f);
            m_rgBatchSampler.Add(sampler);

            sampler = createSampler(50, 1, 0.3f, 1.0f, 0.5f, 2.0f, 0.7f);
            m_rgBatchSampler.Add(sampler);

            sampler = createSampler(50, 1, 0.3f, 1.0f, 0.5f, 2.0f, 0.9f);
            m_rgBatchSampler.Add(sampler);

            sampler = createSampler(50, 1, 0.3f, 1.0f, 0.5f, 2.0f, null, 1.0f);
            m_rgBatchSampler.Add(sampler);

            // Create Detection Output param.
            m_detectionOut = new DetectionOutputParameter();
            m_detectionOut.num_classes = m_multiBoxLossLayer.multiboxloss_param.num_classes;
            m_detectionOut.share_location = m_multiBoxLossLayer.multiboxloss_param.share_location;
            m_detectionOut.background_label_id = (int)m_multiBoxLossLayer.multiboxloss_param.background_label_id;
            m_detectionOut.nms_param = new NonMaximumSuppressionParameter();
            m_detectionOut.nms_param.nms_threshold = 0.45f;
            m_detectionOut.nms_param.top_k = 400;
            m_detectionOut.save_output_param = new SaveOutputParameter();
            m_detectionOut.save_output_param.output_directory = m_strBaseDir + "\\results";
            m_detectionOut.save_output_param.output_name_prefix = "comp4_det_test_";
            m_detectionOut.save_output_param.label_map_file = m_strLabelMapFile;
            m_detectionOut.save_output_param.name_size_file = m_strNameSizeFile;
            m_detectionOut.save_output_param.num_test_image = (uint)m_nNumTestImage;
            m_detectionOut.keep_top_k = 200;
            m_detectionOut.confidence_threshold = 0.01f;
            m_detectionOut.code_type = m_multiBoxLossLayer.multiboxloss_param.code_type;

            // Create Detection Evaluation param.
            m_detectionEval = new DetectionEvaluateParameter();
            m_detectionEval.num_classes = m_multiBoxLossLayer.multiboxloss_param.num_classes;
            m_detectionEval.background_label_id = m_multiBoxLossLayer.multiboxloss_param.background_label_id;
            m_detectionEval.overlap_threshold = 0.5f;
            m_detectionEval.evaulte_difficult_gt = false;
            m_detectionEval.name_size_file = m_strNameSizeFile;
        }

        private BatchSampler createSampler(int nMaxTrials, int nMaxSample, float fMinScale = 1.0f, float fMaxScale = 1.0f, float fMinAspectRatio = 1.0f, float fMaxAspectRatio = 1.0f, float? fMinJaccardOverlap = null, float? fMaxJaccardOverlap = null)
        {
            BatchSampler sampler = new BatchSampler();
            sampler.max_trials = (uint)nMaxTrials;
            sampler.max_sample = (uint)nMaxSample;
            sampler.sampler.min_scale = fMinScale;
            sampler.sampler.max_scale = fMaxScale;
            sampler.sampler.min_aspect_ratio = fMinAspectRatio;
            sampler.sampler.max_aspect_ratio = fMaxAspectRatio;

            if (fMinJaccardOverlap.HasValue)
                sampler.sample_constraint.min_jaccard_overlap = fMinJaccardOverlap.Value;

            if (fMaxJaccardOverlap.HasValue)
                sampler.sample_constraint.max_jaccard_overlap = fMaxJaccardOverlap.Value;

            return sampler;
        }

        /// <summary>
        /// Create the base solver to use.
        /// </summary>
        public override void CreateSolver()
        {
            m_solver = new SolverParameter();
            m_solver.type = SolverParameter.SolverType.SGD;
            m_solver.base_lr = m_dfBaseLr;
            m_solver.weight_decay = 0.0005;
            m_solver.LearningRatePolicy = SolverParameter.LearningRatePolicyType.MULTISTEP;
            m_solver.stepvalue = new List<int>() { 80000, 100000, 120000 };
            m_solver.gamma = 0.1;
            m_solver.momentum = 0.9;
            m_solver.iter_size = m_nIterSize;
            m_solver.max_iter = 120000;
            m_solver.snapshot = 80000;
            m_solver.display = 10;
            m_solver.average_loss = 10;
            m_solver.device_id = m_nGpuID;
            m_solver.debug_info = false;
            m_solver.snapshot_after_train = true;

            // Test parameters.
            m_solver.test_iter.Add(m_nTestIter);
            m_solver.test_interval = 10000;
            m_solver.test_initialization = false;
            m_solver.eval_type = SolverParameter.EvaluationType.DETECTION;
            m_solver.ap_version = ApVersion.ELEVENPOINT;
        }


        /// <summary>
        /// Create the training SSD model for the pascal dataset.
        /// </summary>
        public override NetParameter CreateTrainingModel()
        {
            string strTrainSource = "VOC0712.training";
            string strLabelMapFile = getFileName(m_strLabelMapFile);


            LayerParameter data = createAnnotatedDataLayer(strTrainSource, m_nBatchSize, true, true, strLabelMapFile, SimpleDatum.ANNOTATION_TYPE.NONE, m_transformTrain, m_rgBatchSampler);
            LayerParameter lastLayer = addVGGNetBody(data, 0, true, true, true, true, false, false);
            lastLayer = addExtraLayers(m_bUseBatchNorm, m_dfLrMult);

            // conv4_3 ==> 38 x 38
            // fc7     ==> 19 x 19
            // conv6_2 ==> 10 x 10
            // conv7_2 ==>  5 x 5
            // conv8_2 ==>  3 x 3
            // conv9_2 ==>  1 x 1
            List<string> rgstrMboxSourceLayers = new List<string>() { "conv4_3", "fc7", "conv6_2", "conv7_2", "conv8_2", "conv9_2" };
            List<double> rgAspectWid = new List<double>() { 2, 2, 2, 2, 2, 2 };
            List<double> rgAspectHt = new List<double>() { 2, 3, 3, 3, 2, 2 };
            // L2 normalize conv4_3
            List<double> rgNormalization = new List<double>() { 20, -1, -1, -1, -1, -1 };
            List<double> rgStepsW = new List<double>() { 8, 16, 32, 64, 100, 300 };
            List<double> rgStepsH = new List<double>() { 8, 16, 32, 64, 100, 300 };
            List<MultiBoxHeadInfo> rgInfo = new List<MultiBoxHeadInfo>();
            int nMinDim = 300;
            // in percent %
            double dfMinRatio = 20;
            double dfMaxRatio = 90;
            double dfRatio = dfMinRatio;
            double dfRatioStep = (int)Math.Floor((dfMaxRatio - dfMinRatio) / (rgstrMboxSourceLayers.Count - 2));

            for (int i = 0; i < rgstrMboxSourceLayers.Count; i++)
            {
                string strSrc = rgstrMboxSourceLayers[i];
                double dfMinSize = nMinDim * dfRatio / 100.0;
                double dfMaxSize = nMinDim * (dfRatio + dfRatioStep) / 100.0;
                double dfStepW = rgStepsW[i];
                double dfStepH = rgStepsH[i];
                double dfAspectW = rgAspectWid[i];
                double dfAspectH = rgAspectHt[i];
                double dfNorm = rgNormalization[i];

                rgInfo.Add(new MultiBoxHeadInfo(strSrc, dfMinSize, dfMaxSize, dfStepW, dfStepH, dfAspectW, dfAspectH, dfNorm, null));
            }

            List<LayerParameter> rgMboxLayers = createMultiBoxHead("data", (int)m_multiBoxLossLayer.multiboxloss_param.num_classes, rgInfo, m_rgPriorVariance, false, m_bUseBatchNorm, m_dfLrMult, true, 0, 0, m_multiBoxLossLayer.multiboxloss_param.share_location, m_bFlip, m_bClip, 0.5, 3, 1);

            // Create the MultiboxLossLayer.
            string strName = "mbox_loss";
            LayerParameter mbox_loss = new LayerParameter(LayerParameter.LayerType.MULTIBOX_LOSS);
            mbox_loss.name = strName;
            mbox_loss.multiboxloss_param = m_multiBoxLossLayer.multiboxloss_param;
            mbox_loss.loss_param = m_multiBoxLossLayer.loss_param;
            mbox_loss.include.Add(new NetStateRule(Phase.TRAIN));
            mbox_loss.propagate_down = new List<bool>() { true, true, true, true };
            mbox_loss.bottom.Clear();

            foreach (LayerParameter p in rgMboxLayers)
            {
                mbox_loss.bottom.Add(p.name);
            }

            m_net.layer.Add(mbox_loss);

            return m_net;
        }

        /// <summary>
        /// Create the testing SSD model for the pascal dataset.
        /// </summary>
        public override NetParameter CreateTestingModel()
        {
            string strTestSource = "VOC0712.testing";
            string strLabelMapFile = getFileName(m_strLabelMapFile);


            LayerParameter data = createAnnotatedDataLayer(strTestSource, m_nBatchSize, false, true, strLabelMapFile, SimpleDatum.ANNOTATION_TYPE.NONE, m_transformTest);
            LayerParameter lastLayer = addVGGNetBody(data, 0, true, true, true, true, false, false);
            lastLayer = addExtraLayers(m_bUseBatchNorm, m_dfLrMult);

            // conv4_3 ==> 38 x 38
            // fc7     ==> 19 x 19
            // conv6_2 ==> 10 x 10
            // conv7_2 ==>  5 x 5
            // conv8_2 ==>  3 x 3
            // conv9_2 ==>  1 x 1
            List<string> rgstrMboxSourceLayers = new List<string>() { "conv4_3", "fc7", "conv6_2", "conv7_2", "conv8_2", "conv9_2" };
            List<double> rgAspectWid = new List<double>() { 2, 2, 2, 2, 2, 2 };
            List<double> rgAspectHt = new List<double>() { 2, 3, 3, 3, 2, 2 };
            // L2 normalize conv4_3
            List<double> rgNormalization = new List<double>() { 20, -1, -1, -1, -1, -1 };
            List<double> rgStepsW = new List<double>() { 8, 16, 32, 64, 100, 300 };
            List<double> rgStepsH = new List<double>() { 8, 16, 32, 64, 100, 300 };
            List<MultiBoxHeadInfo> rgInfo = new List<MultiBoxHeadInfo>();
            int nMinDim = 300;
            // in percent %
            double dfMinRatio = 20;
            double dfMaxRatio = 90;
            double dfRatio = dfMinRatio;
            double dfRatioStep = (int)Math.Floor((dfMaxRatio - dfMinRatio) / (rgstrMboxSourceLayers.Count - 2));

            for (int i = 0; i < rgstrMboxSourceLayers.Count; i++)
            {
                string strSrc = rgstrMboxSourceLayers[i];
                double dfMinSize = nMinDim * dfRatio / 100.0;
                double dfMaxSize = nMinDim * (dfRatio + dfRatioStep) / 100.0;
                double dfStepW = rgStepsW[i];
                double dfStepH = rgStepsH[i];
                double dfAspectW = rgAspectWid[i];
                double dfAspectH = rgAspectHt[i];
                double dfNorm = rgNormalization[i];

                rgInfo.Add(new MultiBoxHeadInfo(strSrc, dfMinSize, dfMaxSize, dfStepW, dfStepH, dfAspectW, dfAspectH, dfNorm, null));
            }

            List<LayerParameter> rgMboxLayers = createMultiBoxHead("data", (int)m_multiBoxLossLayer.multiboxloss_param.num_classes, rgInfo, m_rgPriorVariance, false, m_bUseBatchNorm, m_dfLrMult, true, 0, 0, m_multiBoxLossLayer.multiboxloss_param.share_location, m_bFlip, m_bClip, 0.5, 3, 1);

            string strConfName = "mbox_conf";
            lastLayer = findLayer(strConfName);

            if (m_multiBoxLossLayer.multiboxloss_param.conf_loss_type == MultiBoxLossParameter.ConfLossType.SOFTMAX)
            {
                string strReshapeName = strConfName + "_reshape";
                LayerParameter reshape = new LayerParameter(LayerParameter.LayerType.RESHAPE);
                reshape.name = strReshapeName;
                reshape.reshape_param.shape = new BlobShape(new List<int>() { 0, -1, (int)m_multiBoxLossLayer.multiboxloss_param.num_classes });
                lastLayer = connectAndAddLayer(lastLayer, reshape);

                string strSoftmaxName = strConfName + "_softmax";
                LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                softmax.name = strSoftmaxName;
                softmax.softmax_param.axis = 2;
                lastLayer = connectAndAddLayer(lastLayer, softmax);

                string strFlattentName = strConfName + "_flatten";
                LayerParameter flatten = new LayerParameter(LayerParameter.LayerType.FLATTEN);
                flatten.name = strFlattentName;
                flatten.flatten_param.axis = 1;
                lastLayer = connectAndAddLayer(lastLayer, flatten);

                rgMboxLayers[1] = lastLayer;
            }
            else
            {
                string strSigmoidName = strConfName + "_sigmoid";
                LayerParameter sigmoid = new LayerParameter(LayerParameter.LayerType.SIGMOID);
                sigmoid.name = strSigmoidName;

                lastLayer = connectAndAddLayer(lastLayer, sigmoid);
                rgMboxLayers[1] = lastLayer;
            }

            LayerParameter detectionOut = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            detectionOut.name = "detection_output";
            detectionOut.detection_output_param = m_detectionOut;
            detectionOut.include.Add(new NetStateRule(Phase.TEST));
            lastLayer = connectAndAddLayer(rgMboxLayers[0], detectionOut);

            LayerParameter detectionEval = new LayerParameter(LayerParameter.LayerType.DETECTION_EVALUATE);
            detectionEval.name = "detection_eval";
            detectionEval.include.Add(new NetStateRule(Phase.TEST));
            detectionEval.detection_evaluate_param = m_detectionEval;
            lastLayer = connectAndAddLayer(data, detectionEval, false, true, 1);

            return m_net;
        }

        /// <summary>
        /// Add extra layers (for SSD with the Pascal dataset) on top of a 'base' network (e.g. VGGNet or Inception)
        /// </summary>
        /// <param name="bUseBatchNorm">Optionally, specifies whether or not to use batch normalization layers (default = <i>true</i>).</param>
        /// <param name="dfLrMult">Optionally, specifies the learning rate multiplier (default = 1.0).</param>
        protected override LayerParameter addExtraLayers(bool bUseBatchNorm = true, double dfLrMult = 1)
        {
            bool bUseRelU = true;
            string strOutLayer;
            string strFromLayer = m_net.layer[m_net.layer.Count - 1].name;
            LayerParameter lastLayer;

            // 10 x 10
            strOutLayer = "conv6_1";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 256, 1, 0, 1, dfLrMult);
            strFromLayer = strOutLayer;

            strOutLayer = "conv6_2";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 512, 3, 1, 2, dfLrMult);
            strFromLayer = strOutLayer;

            // 5 x 5
            strOutLayer = "conv7_1";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 128, 1, 0, 1, dfLrMult);
            strFromLayer = strOutLayer;

            strOutLayer = "conv7_2";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 256, 3, 1, 2, dfLrMult);
            strFromLayer = strOutLayer;

            // 3 x 3
            strOutLayer = "conv8_1";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 128, 1, 0, 1, dfLrMult);
            strFromLayer = strOutLayer;

            strOutLayer = "conv8_2";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 256, 3, 0, 1, dfLrMult);
            strFromLayer = strOutLayer;

            // 1 x 1
            strOutLayer = "conv9_1";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 128, 1, 0, 1, dfLrMult);
            strFromLayer = strOutLayer;

            strOutLayer = "conv9_2";
            lastLayer = addConvBNLayer(strFromLayer, strOutLayer, bUseBatchNorm, bUseRelU, 256, 3, 0, 1, dfLrMult);
            strFromLayer = strOutLayer;

            return lastLayer;
        }
    }
}
