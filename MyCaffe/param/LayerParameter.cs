using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param.nt;
using MyCaffe.param.ssd;
using MyCaffe.param.beta;
using MyCaffe.param.gpt;
using MyCaffe.param.tft;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the base parameter for all layers.
    /// </summary>
    public class LayerParameter : BaseParameter, ICloneable, IComparable, IBinaryPersist  
    {
        /// <summary>
        /// Specifies the level of conversion support for the layer.
        /// </summary>
        protected ONNX_CONVERSION_SUPPORT m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.NONE;
        // The layer name.
        string m_strName;
        // The layer type.
        LayerType m_type;
        // The name of each bottom blob.
        List<string> m_rgstrBottom = new List<string>();
        // The name of each top blob.
        List<string> m_rgstrTop = new List<string>();
        // Used for rendering models only.
        bool m_bGroupStart = false;
        // Use half sized memory
        bool m_bUseHalfSize = false;

        // The train/test phase for computation.
        Phase m_phase;

        // The amout of weight to assign each top blob in the objective.
        // Each layer assigns a default value, usually of either 0 or 1,
        // to each top blob.
        List<double> m_rgLossWeight = new List<double>();

        // Specifies training parameters (multipliers on global learning constants,
        // and the name and other settings used for weight sharing).
        List<ParamSpec> m_rgParams = new List<ParamSpec>();

        /// <summary>
        /// The blobs containing the numeric parameters of the layer.
        /// </summary>
        List<BlobProto> m_rgBlobs = new List<BlobProto>();

        /// <summary>
        /// Specifies whether to backpropagate to each bottom.  If specified,
        /// Caffe will automatically infer whether each input needs backpropagation
        /// to compute parameter gradients.  If set to true for some inputs,
        /// backpropagation to those inputs is forced; if set to false for some inputs,
        /// backpropagation to those inputs is skipped.
        /// </summary>
        List<bool> m_rgbPropagateDown = new List<bool>();

        /// <summary>
        /// Rules controlling whether and when a layer is included in the network,
        /// based on the current NetState.  You may specify a non-zero number of rules
        /// to include OR exclude, but not both.  If no include or exclude rules are
        /// specified, the layer is always included.  If the current NetState meets
        /// ANY (i.e,, one or more) of the specified rules, the layer is
        /// included/excluded.
        /// </summary>
        List<NetStateRule> m_rgInclude = new List<NetStateRule>();
        List<NetStateRule> m_rgExclude = new List<NetStateRule>();
        Dictionary<Phase, int> m_rgMaxBottomCount = new Dictionary<Phase, int>();

        int m_nSolverCount = 1;
        int m_nSolverRank = 0;
        List<string> m_rgstrExpectedTop = new List<string>();
        List<string> m_rgstrExpectedBottom = new List<string>();
        bool m_bFreezeLearning = false;

        /// <summary>
        /// Defines whether a layer node has ONNX conversion support or not.
        /// </summary>
        public enum ONNX_CONVERSION_SUPPORT
        {
            /// <summary>
            /// Specifies that there is no ONNX conversion support.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies that there is ONNX inference conversion support.
            /// </summary>
            INFERENCE,
            /// <summary>
            /// Specifies that there is both ONNX inference and training conversion support.
            /// </summary>
            INFERENCE_AND_TRAINING
        }

        /// <summary>
        /// Specifies the layer type.
        /// </summary>
        public enum LayerType
        {
            /// <summary>
            /// Initializes a parameter for the AbsValLayer.
            /// </summary>
            ABSVAL,
            /// <summary>
            /// Initializes a parameter for the AccuracyLayer.
            /// </summary>
            ACCURACY,
            /// <summary>
            /// Initializes a parameter for the AccuracyDecodeLayer.
            /// </summary>
            ACCURACY_DECODE,
            /// <summary>
            /// Initializes a parameter for the AccuracyEncodingLayer.
            /// </summary>
            ACCURACY_ENCODING,
            /// <summary>
            /// Initializes a parameter for the AnnotatedDataLayer.
            /// </summary>
            ANNOTATED_DATA,
            /// <summary>
            /// Initializes a parameter for the ArgMaxLayer.
            /// </summary>
            ARGMAX,
            /// <summary>
            /// Initializes a parameter for the AttentionLayer.
            /// </summary>
            ATTENTION,
            /// <summary>
            /// Initializes a parameter for the BiasLayer.
            /// </summary>
            BIAS,
            /// <summary>
            /// Initializes a parameter for the BatchNormLayer.
            /// </summary>
            BATCHNORM,
            /// <summary>
            /// Initializes a parameter for the BatchReindexLayer.
            /// </summary>
            BATCHREINDEX,
            /// <summary>
            /// Initializes a parameter for the BNLLLayer.
            /// </summary>
            BNLL,
            /// <summary>
            /// Initializes a parameter for the CategoricalTransformationLayer
            /// </summary>
            CATEGORICAL_TRANS,
            /// <summary>
            /// Initializes a parameter for the CausalSelfAttentionLayer.
            /// </summary>
            CAUSAL_SELF_ATTENTION,
            /// <summary>
            /// Initializes a parameter for the ChannelEmbeddingLayer.
            /// </summary>
            CHANNEL_EMBEDDING,
            /// <summary>
            /// Initializes a parameter for the ClipLayer.
            /// </summary>
            CLIP,
            /// <summary>
            /// Initializes a parameter for the ConcatLayer.
            /// </summary>
            CONCAT,
            /// <summary>
            /// Initializes a parameter for the ConstantLayer.
            /// </summary>
            CONSTANT,
            /// <summary>
            /// Initializes a parameter for the ContrastiveLossLayer.
            /// </summary>
            CONTRASTIVE_LOSS,
            /// <summary>
            /// Initializes a parameter for the ConvolutionLayer.
            /// </summary>
            CONVOLUTION,
            /// <summary>
            /// Initializes a parameter for the ConvolutionOctaveLayer.
            /// </summary>
            CONVOLUTION_OCTAVE,
            /// <summary>
            /// Initializes a parameter for the CopyLayer.
            /// </summary>
            COPY,
            /// <summary>
            /// Initializes a parameter for the CropLayer.
            /// </summary>
            CROP,
            /// <summary>
            /// Initializes a parameter for the DecodeLayer.
            /// </summary>
            DECODE,
            /// <summary>
            /// Initializes a parameter for the DeconvolutionLayer.
            /// </summary>
            DECONVOLUTION,
            /// <summary>
            /// Initializes a parameter for the DetectionEvaluateLayer.
            /// </summary>
            DETECTION_EVALUATE,
            /// <summary>
            /// Initializes a parameter for the DetectionOutputLayer.
            /// </summary>
            DETECTION_OUTPUT,
            /// <summary>
            /// Initializes a parameter for the DataLayer.
            /// </summary>
            DATA,
            /// <summary>
            /// Initializes a parameter for the DataNormalizerLayer.
            /// </summary>
            DATA_NORMALIZER,
            /// <summary>
            /// Initializes a parameter for the DataSequenceLayer.
            /// </summary>
            DATA_SEQUENCE,
            /// <summary>
            /// Initializes a parameter for the DataTemporalLayer used with TFT models.
            /// </summary>
            DATA_TEMPORAL,
            /// <summary>
            /// Initializes a parameter for the DropoutLayer.
            /// </summary>            
            DROPOUT,
            /// <summary>
            /// Initializes a parameter for the DummyDataLayer.
            /// </summary>
            DUMMYDATA,
            /// <summary>
            /// Initializes a parameter for the EltwiseLayer.
            /// </summary>
            ELTWISE,
            /// <summary>
            /// Initializes a parameter for the ELULayer.
            /// </summary>
            ELU,
            /// <summary>
            /// Initializes a parameter for the EmbedLayer.
            /// </summary>
            EMBED,
            /// <summary>
            /// Initializes a parameter for the EuclideanLossLayer.
            /// </summary>
            EUCLIDEAN_LOSS,
            /// <summary>
            /// Initializes a parameter for the EventLayer.
            /// </summary>
            EVENT,
            /// <summary>
            /// Initializes a parameter for the ExpLayer.
            /// </summary>
            EXP,
            /// <summary>
            /// Initializes a parameter for the FilterLayer.
            /// </summary>
            FILTER,
            /// <summary>
            /// Initializes a parameter for the FlattenLayer.
            /// </summary>
            FLATTEN,
            /// <summary>
            /// Initializes a parameter for the GatherLayer.
            /// </summary>
            GATHER,
            /// <summary>
            /// Initializes a parameter for the GateAddNormLayer.
            /// </summary>
            GATEADDNORM,
            /// <summary>
            /// Initializes a parameter for the GeluLayer.
            /// </summary>
            GELU,
            /// <summary>
            /// Initializes a parameter for the GluLayer (Gated Linear Unit)
            /// </summary>
            GLU,
            /// <summary>
            /// Initializes a parameter for the GrnLayer (Gated Response Network)
            /// </summary>
            GRN,
            /// <summary>
            /// Initializes a parameter for the GradScaleLayer (used for gradient reversal)
            /// </summary>
            GRADIENTSCALER,
            /// <summary>
            /// Initializes a parameter for the GramLayer (used with Neural Style)
            /// </summary>
            GRAM,
            /// <summary>
            /// Initializes a parameter for the GRNLayer (global response normalization L2)
            /// </summary>
            GLOBRES_NORM,
            /// <summary>
            /// Initializes a parameter for the HDF5DataLayer.
            /// </summary>
            HDF5_DATA,
            /// <summary>
            /// Initializes a parameter for the HingeLossLayer.
            /// </summary>
            HINGE_LOSS,
            /// <summary>
            /// Initializes a parameter for the ImageDataLayer.
            /// </summary>
            IMAGE_DATA,
            /// <summary>
            /// Initializes a parameter for the Im2ColLayer.
            /// </summary>
            IM2COL,
            /// <summary>
            /// Initializes a parameter for the InfogainLossLayer.
            /// </summary>
            INFOGAIN_LOSS,
            /// <summary>
            /// Initializes a parameter for the InnerProductLayer.
            /// </summary>
            INNERPRODUCT,
            /// <summary>
            /// Initializes a parameter for the InputLayer.
            /// </summary>
            INPUT,
            /// <summary>
            /// Initializes a parameter for the InterpLayer.
            /// </summary>
            INTERP,
            /// <summary>
            /// Initializes a parameter for the LabelMappingLayer.
            /// </summary>
            LABELMAPPING,
            /// <summary>
            /// Initializes a parameter for the LayerNormalizationLayer.
            /// </summary>
            LAYERNORM,
            /// <summary>
            /// Initializes a parameter for the LogLayer.
            /// </summary>
            LOG,
            /// <summary>
            /// Initializes a parameter for the LossLayer.
            /// </summary>
            LOSS,
            /// <summary>
            /// Initializes a parameter for the LRNLayer.
            /// </summary>
            LRN,
            /// <summary>
            /// Initializes a parameter for the MeanErrorLossLayer, used with time series and other regression problems.
            /// </summary>
            MEAN_ERROR_LOSS,
            /// <summary>
            /// Initializes a parameter for the MathLayer.
            /// </summary>
            MATH,
            /// <summary>
            /// Initializes a parameter for the MemoryDataLayer.
            /// </summary>
            MEMORYDATA,
            /// <summary>
            /// Initializes a parameter for the MemoryLossLayer.
            /// </summary>
            MEMORY_LOSS,
            /// <summary>
            /// Initializes a parameter for the MergeLayer.
            /// </summary>            
            MERGE,
            /// <summary>
            /// Initializes a parameter for the MishLayer.
            /// </summary>            
            MISH,
            /// <summary>
            /// Initialize a parameter for the MultiBoxLossLayer.
            /// </summary>
            MULTIBOX_LOSS,
            /// <summary>
            /// Initializes a parameter for the MultiheadAttentionLayer.
            /// </summary>
            MULTIHEAD_ATTENTION,
            /// <summary>
            /// Initializes a parameter for the MultiheadAttentionInterpretableLayer
            /// </summary>
            MULTIHEAD_ATTENTION_INTERP,
            /// <summary>
            /// Initializes a parameter for the MultinomialLogisticLossLayer.
            /// </summary>
            MULTINOMIALLOGISTIC_LOSS,
            /// <summary>
            /// Initializes a parameter for the MVNLayer.
            /// </summary>
            MVN,
            /// <summary>
            /// Initializes a parameter for the NLLLossLayer
            /// </summary>
            NLL_LOSS,
            /// <summary>
            /// Initializes a parameter for the NumericTransformationLayer
            /// </summary>
            NUMERIC_TRANS,
            /// <summary>
            /// Initializes a parameter for the OneHotLayer.
            /// </summary>
            ONEHOT,
            /// <summary>
            /// Initializes a parameter for the ParameterLayer.
            /// </summary>
            PARAMETER,
            /// <summary>
            /// Initializes a parameter for the PermuteLayer used with SSD.
            /// </summary>
            PERMUTE,
            /// <summary>
            /// Initializes a parameter for the PoolingLayer.
            /// </summary>
            POOLING,
            /// <summary>
            /// Initializes a parameter for the PositionalEncoderLayer.
            /// </summary>
            POSITIONAL_ENCODER,
            /// <summary>
            /// Initializes a parameter for the PowerLayer.
            /// </summary>
            POWER,
            /// <summary>
            /// Initializes a parameter for the PReLULayer.
            /// </summary>
            PRELU,
            /// <summary>
            /// Initializes a parameter for the PriorBoxLayer.
            /// </summary>
            PRIORBOX,
            /// <summary>
            /// Initializes a parameter for the QuantileAccuracyLayer used in TFT models.
            /// </summary>
            QUANTILE_ACCURACY,
            /// <summary>
            /// Initializes a parameter for the QuantileLossLayer used in TFT models.
            /// </summary>
            QUANTILE_LOSS,
            /// <summary>
            /// Initializes a parameter for the ReductionLayer.
            /// </summary>
            REDUCTION,
            /// <summary>
            /// Initializes a parameter for the ReLULayer.
            /// </summary>
            RELU,
            /// <summary>
            /// Initializes a parameter for the ReshapeLayer.
            /// </summary>
            RESHAPE,
            /// <summary>
            /// Initializes a parameter for the ReshapeTemporalLayer.
            /// </summary>
            RESHAPE_TEMPORAL,
            /// <summary>
            /// Initializes a parameter for the ScalarLayer.
            /// </summary>
            SCALAR,
            /// <summary>
            /// Initializes a parameter for the ScaleLayer.
            /// </summary>
            SCALE,
            /// <summary>
            /// Initializes a parameter for the SerfLayer.
            /// </summary>
            SERF,
            /// <summary>
            /// Initializes a parameter for the SigmoidLayer.
            /// </summary>
            SIGMOID,
            /// <summary>
            /// Initializes a parameter for the SigmoidCrossEntropyLossLayer.
            /// </summary>
            SIGMOIDCROSSENTROPY_LOSS,
            /// <summary>
            /// Initializes a parameter for the SoftmaxCrossEntropyLossLayer.
            /// </summary>
            SOFTMAXCROSSENTROPY_LOSS,
            /// <summary>
            /// Initializes a parameter for the SoftmaxCrossEntropy2LossLayer.
            /// </summary>
            SOFTMAXCROSSENTROPY2_LOSS,
            /// <summary>
            /// Initializes a parameter for the SoftmaxLayer.
            /// </summary>
            SOFTMAX,
            /// <summary>
            /// Initializes a parameter for the SoftmaxLossLayer.
            /// </summary>
            SOFTMAXWITH_LOSS,
            /// <summary>
            /// Intiializes a parameter for the SmoothL1LossLayer.
            /// </summary>
            SMOOTHL1_LOSS,
            /// <summary>
            /// Initializes a parameter for the SPPLayer.
            /// </summary>
            SPP,
            /// <summary>
            /// Initializes a parameter for the SilenceLayer.
            /// </summary>
            SILENCE,
            /// <summary>
            /// Initializes a parameter for the SliceLayer.
            /// </summary>
            SLICE,
            /// <summary>
            /// Initializes a parameter for the SplitLayer.
            /// </summary>
            SPLIT,
            /// <summary>
            /// Initializes a parameter for the SqueezeLayer.
            /// </summary>
            SQUEEZE,
            /// <summary>
            /// Initializes a parameter for the SqueezeLayer.
            /// </summary>
            UNSQUEEZE,
            /// <summary>
            /// Initializes a parameter for the SwishLayer
            /// </summary>
            SWISH,
            /// <summary>
            /// Initializes a parameter for the ModelDataLayer.
            /// </summary>
            MODEL_DATA,
            /// <summary>
            /// Initializes a parameter for the TextDataLayer.
            /// </summary>
            TEXT_DATA,
            /// <summary>
            /// Initializes a parameter for the TVLossLayer (used with Neural Style).
            /// </summary>
            TV_LOSS,
            /// <summary>
            /// Initializes a parameter for the TanhLayer.
            /// </summary>
            TANH,
            /// <summary>
            /// Initializes a parameter for the ThresholdLayer.
            /// </summary>
            THRESHOLD,
            /// <summary>
            /// Initializes a parameter for the TileLayer.
            /// </summary>
            TILE,
            /// <summary>
            /// Initializes a parameter for the DataTransformer.
            /// </summary>
            TRANSFORM,
            /// <summary>
            /// Initializes a parameter for the TransformerBlockLayer.
            /// </summary>
            TRANSFORMER_BLOCK,
            /// <summary>
            /// Initializes a parameter for the TransformerDataLayer.
            /// </summary>
            TOKENIZED_DATA,
            /// <summary>
            /// Initializes a parameter for the TransformerDataPairsLayer.
            /// </summary>
            TOKENIZED_DATA_PAIRS,
            /// <summary>
            /// Initializes a parameter for the PythonTransformerDataPairsLayer.
            /// </summary>
            TOKENIZED_DATA_PAIRS_PY,
            /// <summary>
            /// Initializes a parameter for the TransposeLayer.
            /// </summary>
            TRANSPOSE,
            /// <summary>
            /// Initializes a parameter for the LSTMSimpleLayer [DEPRECIATED].
            /// </summary>
            LSTM_SIMPLE,
            /// <summary>
            /// Initializes a parameter for the LSTMAttentionLayer.
            /// </summary>
            LSTM_ATTENTION,
            /// <summary>
            /// Initializes a parameter for the RecurrentLayer.
            /// </summary>
            RECURRENT,
            /// <summary>
            /// Initializes a parameter for the RNNLayer.
            /// </summary>
            RNN,
            /// <summary>
            /// Initializes a parameter for the LSTMLayer.
            /// </summary>
            LSTM,
            /// <summary>
            /// Initializes a parameter for the LSTMUnitLayer.
            /// </summary>
            LSTM_UNIT,
            /// <summary>
            /// DEPRECIATED - Initializes a parameter for the UnpoolingLayer1 which uses a CPU based implementation (slower).
            /// </summary>
            UNPOOLING1,
            /// <summary>
            /// Initializes a parameter for the UnpoolingLayer which uses a GPU based implementation (faster).
            /// </summary>
            UNPOOLING,
            /// <summary>
            /// Initializes a parameter for the Normalization1Layer.
            /// </summary>
            NORMALIZATION1,
            /// <summary>
            /// Initializes a parameter for the Normalization2Layer used with SSD.
            /// </summary>
            NORMALIZATION2,
            /// <summary>
            /// Initializes a parameter for the TripletLossSimpleLayer.
            /// </summary>
            TRIPLET_LOSS_SIMPLE,
            /// <summary>
            /// Initializes a parameter for the TripletLossLayer.
            /// </summary>
            TRIPLET_LOSS,
            /// <summary>
            /// Initializes a parameter for the KNNLayer.
            /// </summary>
            KNN,
            /// <summary>
            /// Initializes a parameter for the DebugLayer.
            /// </summary>
            DEBUG,
            /// <summary>
            /// Initializes a parameter for the VideoDataLayer.
            /// </summary>
            VIDEO_DATA,
            /// <summary>
            /// Initializes a parameter for the VariableSelectionNetworkLayer
            /// </summary>
            VARSELNET,
#pragma warning disable 1591
            _MAX
#pragma warning restore 1591
        }

        // Layer type-specific parameters
        //
        // Note: certain layers may have more than one computation engine
        // for their implementation. These layers include an Engine type and
        // engine parameter for selecting the implementation.
        // The default for the engine is set by the ENGINE switch at compile-time.
        Dictionary<LayerType, LayerParameterBase> m_rgLayerParameters = new Dictionary<LayerType, LayerParameterBase>();

        /** @copydoc BaseParameter */
        public LayerParameter() : base()
        {
            for (int i = 0; i < (int)LayerType._MAX; i++)
            {
                m_rgLayerParameters.Add((LayerType)i, null);
            }
        }

        /// <summary>
        /// The LayerParameter constructor.
        /// </summary>
        /// <param name="lt">Assignes this LayerType to the layer.</param>
        /// <param name="strName">Assigns this name to the layer.</param>
        public LayerParameter(LayerType lt, string strName = null)
            : base()
        {
            m_type = lt;
            m_strName = strName;

            if (m_strName == null)
                m_strName = lt.ToString();

            for (int i = 0; i < (int)LayerType._MAX; i++)
            {
                m_rgLayerParameters.Add((LayerType)i, null);
            }

            setupParams(lt);
        }

        /// <summary>
        /// The LayerParameter constructor.
        /// </summary>
        /// <param name="p">Used to initialize the new LayerParameter.</param>
        public LayerParameter(LayerParameter p)
            : base()
        {
            m_type = p.m_type;
            m_strName = p.m_strName;
            m_rgstrBottom = p.m_rgstrBottom;
            m_rgstrTop = p.m_rgstrTop;
            m_phase = p.m_phase;
            m_rgLossWeight = p.m_rgLossWeight;
            m_rgParams = p.m_rgParams;
            m_rgBlobs = p.m_rgBlobs;
            m_rgbPropagateDown = p.m_rgbPropagateDown;
            m_rgInclude = p.m_rgInclude;
            m_rgExclude = p.m_rgExclude;
            m_rgLayerParameters = p.m_rgLayerParameters;
            m_nSolverCount = p.m_nSolverCount;
            m_nSolverRank = p.m_nSolverRank;
            m_bGroupStart = p.m_bGroupStart;
        }

        /// <summary>
        /// Prepare model inputs for the run-net (if any are needed for the layer).
        /// </summary>
        /// <returns>The model input description or null is returned.</returns>
        public string PrepareRunModelInputs()
        {
            if (m_rgLayerParameters[m_type] == null)
                return null;

            return m_rgLayerParameters[m_type].PrepareRunModelInputs();
        }

        /// <summary>
        /// Prepare the layer settings for a run model.
        /// </summary>
        public void PrepareRunModel()
        {
            if (m_rgLayerParameters[m_type] == null)
                return;

            m_rgLayerParameters[m_type].PrepareRunModel(this);
        }

        /// <summary>
        /// Returns the number of ParamSpec parameters used by the layer.
        /// </summary>
        /// <returns>The ParamSpec count is returned.</returns>
        public int GetParameterCount()
        {
            int nOffset = 0;

            switch (m_type)
            {
                case LayerType.CONVOLUTION:
                case LayerType.DECONVOLUTION:
                    if (convolution_param != null && !convolution_param.bias_term && m_rgParams.Count > 1)
                        nOffset = -1;
                    break;

                case LayerType.INNERPRODUCT:
                    if (inner_product_param != null && !inner_product_param.bias_term && m_rgParams.Count > 1)
                        nOffset = -1;
                    break;
            }

            return m_rgParams.Count + nOffset;
        }

        /// <summary>
        /// Copies the defaults from another LayerParameter.
        /// </summary>
        /// <param name="p">Specifies the LayerParameter to copy.</param>
        public void CopyDefaults(LayerParameter p)
        {
            if (p == null)
                return;

            if (p.type != m_type)
                throw new ArgumentOutOfRangeException();

            m_rgInclude = p.include;
            m_rgExclude = p.exclude;
            m_rgParams = p.parameters;

            switch (m_type)
            {
                case LayerType.ANNOTATED_DATA:
                    annotated_data_param = (AnnotatedDataParameter)p.annotated_data_param.Clone();
                    data_param = (DataParameter)p.data_param.Clone();
                    transform_param = (TransformationParameter)p.transform_param.Clone();
                    break;

                case LayerType.DATA:
                    data_param = (DataParameter)p.data_param.Clone();
                    transform_param = (TransformationParameter)p.transform_param.Clone();
                    break;

                case LayerType.IMAGE_DATA:
                    data_param = (DataParameter)p.data_param.Clone();
                    image_data_param = (ImageDataParameter)p.image_data_param.Clone();
                    transform_param = (TransformationParameter)p.transform_param.Clone();
                    break;

                case LayerType.MEMORYDATA:
                    memory_data_param = (MemoryDataParameter)p.memory_data_param.Clone();
                    transform_param = (TransformationParameter)p.transform_param.Clone();
                    break;
            }
        }

        /// <summary>
        /// Determines whether or not this LayerParameter meets a given Phase.
        /// </summary>
        /// <param name="phase">Specifies the Phase.</param>
        /// <returns>Returns <i>true</i> if this LayerParameter meets the Phase, <i>false</i> otherwise.</returns>
        public bool MeetsPhase(Phase phase)
        {
            if (phase == Phase.NONE)
                return true;

            foreach (NetStateRule r in m_rgExclude)
            {
                if (r.phase == phase)
                    return false;
            }

            foreach (NetStateRule r in m_rgInclude)
            {
                if (r.phase == phase)
                    return true;
            }

            if (m_rgInclude.Count == 0)
                return true;

            if (m_rgExclude.Count > 0)
                return true;

            return false;
        }

        /// <summary>
        /// Save this parameter to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write((int)m_type);
            bw.Write(m_strName);
            Utility.Save<string>(bw, m_rgstrBottom);
            Utility.Save<string>(bw, m_rgstrTop);
            Utility.Save<double>(bw, m_rgLossWeight);
            Utility.Save<ParamSpec>(bw, m_rgParams);
            Utility.Save<BlobProto>(bw, m_rgBlobs);
            Utility.Save<bool>(bw, m_rgbPropagateDown);
            Utility.Save<NetStateRule>(bw, m_rgInclude);
            Utility.Save<NetStateRule>(bw, m_rgExclude);

            int nCount = 0;

            foreach (LayerParameterBase p in m_rgLayerParameters.Values)
            {
                if (p != null)
                    nCount++;
            }

            bw.Write(nCount);

            foreach (KeyValuePair<LayerType, LayerParameterBase> kv in m_rgLayerParameters)
            {
                if (kv.Value != null)
                {
                    bw.Write((int)kv.Key);

                    IBinaryPersist bp = kv.Value as IBinaryPersist;
                    bp.Save(bw);
                }
            }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            LayerType lt = (LayerType)br.ReadInt32();
            string strName = br.ReadString();

            LayerParameter p = this;
            
            if (bNewInstance)
                p = new LayerParameter(lt, strName);

            p.m_rgstrBottom = Utility.Load<string>(br);
            p.m_rgstrTop = Utility.Load<string>(br);
            p.m_rgLossWeight = Utility.Load<double>(br);
            p.m_rgParams = Utility.Load<ParamSpec>(br);
            p.m_rgBlobs = Utility.Load<BlobProto>(br);
            p.m_rgbPropagateDown = Utility.Load<bool>(br);
            p.m_rgInclude = Utility.Load<NetStateRule>(br);
            p.m_rgExclude = Utility.Load<NetStateRule>(br);

            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                lt = (LayerType)br.ReadInt32();
                IBinaryPersist bp = m_rgLayerParameters[lt] as IBinaryPersist;
                bp.Load(br, false);
            }

            return p;
        }

        private void setupParams(LayerType lt, bool bNewParams = true)
        {
            switch (lt)
            {
                case LayerType.ABSVAL:
                    expected_bottom.Add("input");
                    expected_top.Add("abs");
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.ACCURACY:
                    expected_bottom.Add("input");
                    expected_bottom.Add("label");
                    expected_top.Add("accuracy");
                    m_rgLayerParameters[LayerType.ACCURACY] = new AccuracyParameter();
                    break;

                case LayerType.ACCURACY_DECODE:
                    expected_bottom.Add("decode");
                    expected_top.Add("accuracy");
                    m_rgLayerParameters[LayerType.ACCURACY] = new AccuracyParameter();
                    break;

                case LayerType.ACCURACY_ENCODING:
                    expected_bottom.Add("input");
                    expected_bottom.Add("label");
                    expected_top.Add("accuracy");
                    m_rgLayerParameters[LayerType.ACCURACY] = new AccuracyParameter();
                    m_rgLayerParameters[LayerType.DECODE] = new DecodeParameter();
                    break;

                case LayerType.ANNOTATED_DATA:
                    expected_top.Add("data");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.TRANSFORM] = new TransformationParameter();
                    m_rgLayerParameters[LayerType.ANNOTATED_DATA] = new AnnotatedDataParameter();
                    m_rgLayerParameters[LayerType.DATA] = new DataParameter();
                    break;

                case LayerType.ARGMAX:
                    expected_bottom.Add("input");
                    expected_top.Add("max");
                    m_rgLayerParameters[lt] = new ArgMaxParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.CAUSAL_SELF_ATTENTION:
                    expected_bottom.Add("input");
                    expected_top.Add("atten");
                    m_rgLayerParameters[lt] = new CausalSelfAttentionParameter();
                    break;

                case LayerType.ATTENTION:
                    expected_bottom.Add("input");
                    expected_top.Add("atten");
                    m_rgLayerParameters[lt] = new AttentionParameter();
                    break;

                case LayerType.BATCHNORM:
                    expected_bottom.Add("input");
                    expected_top.Add("norm");
                    m_rgLayerParameters[lt] = new BatchNormParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.BATCHREINDEX:
                    expected_bottom.Add("input");
                    expected_bottom.Add("idx");
                    expected_top.Add("data");
                    break;

                case LayerType.BIAS:
                    expected_bottom.Add("input");
                    expected_bottom.Add("bias");
                    expected_top.Add("bias");
                    m_rgLayerParameters[lt] = new BiasParameter();
                    break;

                case LayerType.BNLL:
                    expected_bottom.Add("input");
                    expected_top.Add("bnll");
                    break;

                case LayerType.CATEGORICAL_TRANS:
                    expected_bottom.Add("x");
                    expected_top.Add("proj");
                    m_rgLayerParameters[lt] = new CategoricalTransformationParameter();
                    break;

                case LayerType.CHANNEL_EMBEDDING:
                    expected_bottom.Add("x_num");
                    expected_bottom.Add("x_cat");
                    expected_top.Add("emb");
                    m_rgLayerParameters[LayerType.CATEGORICAL_TRANS] = new CategoricalTransformationParameter();
                    m_rgLayerParameters[LayerType.NUMERIC_TRANS] = new NumericTransformationParameter();
                    break;

                case LayerType.CLIP:
                    expected_bottom.Add("input");
                    expected_top.Add("clip");
                    m_rgLayerParameters[lt] = new ClipParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.CONCAT:
                    expected_bottom.Add("x_1");
                    expected_bottom.Add("x_2");
                    expected_top.Add("concat");
                    m_rgLayerParameters[lt] = new ConcatParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.CONSTANT:
                    expected_top.Add("const");
                    m_rgLayerParameters[lt] = new ConstantParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.CONTRASTIVE_LOSS:
                    expected_bottom.Add("f1");
                    expected_bottom.Add("f2");
                    expected_bottom.Add("lbl");
                    expected_top.Add("loss");
                    expected_top.Add("match");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new ContrastiveLossParameter();
                    break;

                case LayerType.CONVOLUTION:
                case LayerType.IM2COL:
                    expected_bottom.Add("enc");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.CONVOLUTION] = new ConvolutionParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.CONVOLUTION_OCTAVE:
                    expected_bottom.Add("in_h");
                    expected_bottom.Add("in_l");
                    expected_top.Add("x_h");
                    expected_top.Add("x_l");
                    m_rgLayerParameters[LayerType.CONVOLUTION] = new ConvolutionParameter();
                    m_rgLayerParameters[LayerType.CONVOLUTION_OCTAVE] = new ConvolutionOctaveParameter();
                    break;

                case LayerType.CROP:
                    expected_bottom.Add("upscore");
                    expected_bottom.Add("data");
                    expected_top.Add("score");
                    m_rgLayerParameters[lt] = new CropParameter();
                    break;

                case LayerType.COPY:
                    expected_bottom.Add("src");
                    expected_bottom.Add("dst");
                    break;

                case LayerType.DECODE:
                    expected_bottom.Add("enc");
                    expected_top.Add("dist1");
                    m_rgLayerParameters[lt] = new DecodeParameter();
                    break;

                case LayerType.DECONVOLUTION:
                    expected_bottom.Add("score");
                    expected_top.Add("upscore");
                    if (bNewParams || m_rgLayerParameters[LayerType.CONVOLUTION] == null)
                        m_rgLayerParameters[LayerType.CONVOLUTION] = new ConvolutionParameter();
                    break;

                case LayerType.DETECTION_EVALUATE:
                    expected_bottom.Add("det");
                    expected_bottom.Add("gt");
                    expected_top.Add("output");
                    m_rgLayerParameters[lt] = new DetectionEvaluateParameter();
                    break;

                case LayerType.DETECTION_OUTPUT:
                    expected_bottom.Add("loc");
                    expected_bottom.Add("conf");
                    expected_bottom.Add("prior");
                    expected_top.Add("output");
                    m_rgLayerParameters[lt] = new DetectionOutputParameter();
                    break;

                case LayerType.DATA:
                    expected_top.Add("data");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.TRANSFORM] = new TransformationParameter();
                    m_rgLayerParameters[LayerType.DATA] = new DataParameter();
                    break;

                case LayerType.DATA_NORMALIZER:
                    expected_bottom.Add("data");
                    expected_bottom.Add("label");
                    expected_top.Add("ndata");
                    expected_bottom.Add("nlabel");
                    m_rgLayerParameters[lt] = new DataNormalizerParameter();
                    break;

                case LayerType.DATA_SEQUENCE:
                    expected_bottom.Add("data");
                    expected_bottom.Add("label");
                    expected_top.Add("anchor");
                    expected_top.Add("datax");
                    m_rgLayerParameters[lt] = new DataSequenceParameter();
                    break;

                case LayerType.DATA_TEMPORAL:
                    expected_top.Add("sn");
                    expected_top.Add("sc");
                    expected_top.Add("hn");
                    expected_top.Add("hc");
                    expected_top.Add("fn");
                    expected_top.Add("fc");
                    expected_top.Add("t");
                    m_rgLayerParameters[lt] = new DataTemporalParameter();
                    break;

                case LayerType.DEBUG:
                    expected_bottom.Add("input");
                    expected_bottom.Add("label");
                    expected_top.Add("output");
                    m_rgLayerParameters[lt] = new DebugParameter();
                    break;

                case LayerType.DROPOUT:
                    expected_bottom.Add("input");
                    expected_top.Add("dropout");
                    m_rgLayerParameters[lt] = new DropoutParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.DUMMYDATA:
                    expected_top.Add("data");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.TRANSFORM] = new TransformationParameter();
                    m_rgLayerParameters[lt] = new DummyDataParameter();
                    break;

                case LayerType.ELTWISE:
                    expected_bottom.Add("x_1");
                    expected_bottom.Add("x_2");
                    expected_top.Add("eltwise");
                    m_rgLayerParameters[lt] = new EltwiseParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.ELU:
                    expected_bottom.Add("input");
                    expected_top.Add("elu");
                    m_rgLayerParameters[lt] = new EluParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.EMBED:
                    expected_bottom.Add("input");
                    expected_top.Add("embed");
                    m_rgLayerParameters[lt] = new EmbedParameter();
                    break;

                case LayerType.EUCLIDEAN_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("trgt");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    break;

                case LayerType.EVENT:
                    expected_bottom.Add("input");
                    expected_top.Add("output");
                    break;

                case LayerType.EXP:
                    expected_bottom.Add("input");
                    expected_top.Add("exp");
                    m_rgLayerParameters[lt] = new ExpParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.FILTER:
                    expected_bottom.Add("x_1");
                    expected_bottom.Add("x_2");
                    expected_top.Add("y_1");
                    expected_top.Add("y_2");
                    break;

                case LayerType.FLATTEN:
                    expected_bottom.Add("x_1");
                    expected_top.Add("flatten");
                    m_rgLayerParameters[lt] = new FlattenParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.GATHER:
                    expected_bottom.Add("input");
                    expected_bottom.Add("idx");
                    expected_top.Add("gthr");
                    m_rgLayerParameters[lt] = new GatherParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.GATEADDNORM:
                    expected_bottom.Add("input");
                    expected_top.Add("gan");
                    m_rgLayerParameters[lt] = new GateAddNormParameter();
                    m_rgLayerParameters[LayerType.GLU] = new GluParameter();
                    m_rgLayerParameters[LayerType.DROPOUT] = new DropoutParameter();
                    m_rgLayerParameters[LayerType.LAYERNORM] = new LayerNormParameter();
                    break;

                case LayerType.GELU:
                    expected_bottom.Add("input");
                    expected_top.Add("gelu");
                    m_rgLayerParameters[lt] = new GeluParameter();
                    break;

                case LayerType.GLU:
                    expected_bottom.Add("input");
                    expected_top.Add("glu");
                    m_rgLayerParameters[lt] = new GluParameter();
                    break;

                case LayerType.GRN:
                    expected_bottom.Add("input");
                    expected_top.Add("grn");
                    m_rgLayerParameters[lt] = new GrnParameter();
                    break;

                case LayerType.GRADIENTSCALER:
                    expected_bottom.Add("input");
                    expected_top.Add("identity");
                    m_rgLayerParameters[lt] = new GradientScaleParameter();
                    break;

                case LayerType.GLOBRES_NORM:
                    expected_bottom.Add("input");
                    expected_top.Add("gresnet");
                    m_rgLayerParameters[lt] = new FlattenParameter();
                    break;

                case LayerType.GRAM:
                    expected_bottom.Add("input");
                    expected_top.Add("gram");
                    m_rgLayerParameters[lt] = new GramParameter();
                    break;

                case LayerType.HDF5_DATA:
                    expected_top.Add("data");
                    m_rgLayerParameters[LayerType.HDF5_DATA] = new HDF5DataParameter();
                    break;

                case LayerType.HINGE_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new HingeLossParameter();
                    break;

                case LayerType.IMAGE_DATA:
                    expected_top.Add("data");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.TRANSFORM] = new TransformationParameter();
                    m_rgLayerParameters[LayerType.IMAGE_DATA] = new ImageDataParameter();
                    DataParameter imgdp = new DataParameter();
                    imgdp.backend = DataParameter.DB.NONE;
                    imgdp.enable_random_selection = false;
                    m_rgLayerParameters[LayerType.DATA] = imgdp;
                    break;

                case LayerType.INFOGAIN_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_bottom.Add("H");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new InfogainLossParameter();
                    break;

                case LayerType.INNERPRODUCT:
                    expected_bottom.Add("input");
                    expected_top.Add("ip");
                    m_rgLayerParameters[lt] = new InnerProductParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.INPUT:
                    expected_top.Add("data");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.INPUT] = new InputParameter();
                    break;

                case LayerType.INTERP:
                    expected_top.Add("input");
                    expected_top.Add("interp");
                    m_rgLayerParameters[lt] = new InterpParameter();
                    break;

                case LayerType.LABELMAPPING:
                    expected_bottom.Add("input");
                    expected_top.Add("output");
                    m_rgLayerParameters[lt] = new LabelMappingParameter();
                    break;

                case LayerType.KNN:
                    expected_bottom.Add("input");
                    expected_bottom.Add("label");
                    expected_top.Add("classes");
                    m_rgMaxBottomCount.Add(Phase.RUN, 1);
                    m_rgLayerParameters[lt] = new KnnParameter();
                    break;

                case LayerType.LAYERNORM:
                    expected_bottom.Add("input");
                    expected_top.Add("norm");
                    m_rgLayerParameters[lt] = new LayerNormParameter();
                    break;

                case LayerType.LOG:
                    expected_bottom.Add("input");
                    expected_top.Add("log");
                    m_rgLayerParameters[lt] = new LogParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.LRN:
                    expected_bottom.Add("input");
                    expected_top.Add("lrn");
                    m_rgLayerParameters[lt] = new LRNParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.MEAN_ERROR_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("target");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new MeanErrorLossParameter();
                    break;

                case LayerType.MATH:
                    expected_bottom.Add("input");
                    expected_top.Add("math");
                    m_rgLayerParameters[lt] = new MathParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.MERGE:
                    expected_bottom.Add("input1");
                    expected_bottom.Add("input2");
                    expected_top.Add("merge");
                    m_rgLayerParameters[lt] = new MergeParameter();
                    break;

                case LayerType.MEMORYDATA:
                    expected_top.Add("data");
                    m_rgLayerParameters[LayerType.TRANSFORM] = new TransformationParameter();
                    m_rgLayerParameters[LayerType.MEMORYDATA] = new MemoryDataParameter();
                    break;

                case LayerType.MEMORY_LOSS:
                    expected_bottom.Add("input");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    break;

                case LayerType.MISH:
                    expected_bottom.Add("input");
                    expected_top.Add("mish");
                    m_rgLayerParameters[lt] = new MishParameter();
                    break;

                case LayerType.MULTIBOX_LOSS:
                    expected_bottom.Add("loc");
                    expected_bottom.Add("conf");
                    expected_bottom.Add("prior");
                    expected_bottom.Add("gt");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new MultiBoxLossParameter();
                    break;

                case LayerType.MULTIHEAD_ATTENTION:
                    expected_bottom.Add("q");
                    expected_bottom.Add("k");
                    expected_bottom.Add("v");
                    expected_top.Add("attn");
                    m_rgLayerParameters[lt] = new MultiheadAttentionParameter();
                    break;

                case LayerType.MULTIHEAD_ATTENTION_INTERP:
                    expected_bottom.Add("q");
                    expected_bottom.Add("k");
                    expected_bottom.Add("v");
                    expected_top.Add("attn");
                    expected_top.Add("y");
                    expected_top.Add("out");
                    expected_top.Add("scr");
                    m_rgLayerParameters[lt] = new MultiHeadAttentionInterpParameter();
                    break;

                case LayerType.MULTINOMIALLOGISTIC_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    break;

                case LayerType.MVN:
                    expected_bottom.Add("input");
                    expected_top.Add("mvn");
                    m_rgLayerParameters[lt] = new MVNParameter();
                    break;

                case LayerType.NLL_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new NLLLossParameter();
                    break;

                case LayerType.NUMERIC_TRANS:
                    expected_bottom.Add("x");
                    expected_top.Add("proj");
                    m_rgLayerParameters[lt] = new NumericTransformationParameter();
                    break;

                case LayerType.ONEHOT:
                    expected_bottom.Add("input");
                    expected_top.Add("onehot");
                    m_rgLayerParameters[lt] = new OneHotParameter();
                    break;

                case LayerType.NORMALIZATION1:
                    expected_bottom.Add("input");
                    expected_top.Add("norm");
                    m_rgLayerParameters[lt] = new Normalization1Parameter();
                    break;

                case LayerType.NORMALIZATION2:
                    expected_bottom.Add("input");
                    expected_top.Add("norm");
                    m_rgLayerParameters[lt] = new Normalization2Parameter();
                    break;

                case LayerType.PARAMETER:
                    expected_bottom.Add("input");
                    expected_top.Add("param");
                    m_rgLayerParameters[lt] = new ParameterParameter();
                    break;

                case LayerType.PERMUTE:
                    expected_bottom.Add("input");
                    expected_top.Add("permute");
                    m_rgLayerParameters[lt] = new PermuteParameter();
                    break;

                case LayerType.POSITIONAL_ENCODER:
                    expected_bottom.Add("input");
                    expected_top.Add("pos");
                    m_rgLayerParameters[lt] = new PositionalEncoderParameter();
                    break;

                case LayerType.POOLING:
                    expected_bottom.Add("input");
                    expected_top.Add("pool");
                    expected_top.Add("mask");
                    m_rgLayerParameters[lt] = new PoolingParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.UNPOOLING1:
                    expected_bottom.Add("pool");
                    expected_top.Add("unpool");
                    m_rgLayerParameters[LayerType.UNPOOLING] = new UnPoolingParameter();
                    break;

                case LayerType.UNPOOLING:
                    expected_bottom.Add("pool");
                    expected_top.Add("unpool");
                    m_rgLayerParameters[lt] = new UnPoolingParameter();
                    break;

                case LayerType.POWER:
                    expected_bottom.Add("input");
                    expected_top.Add("power");
                    m_rgLayerParameters[lt] = new PowerParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.PRELU:
                    expected_bottom.Add("input");
                    expected_top.Add("prelu");
                    m_rgLayerParameters[lt] = new PReLUParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.PRIORBOX:
                    expected_bottom.Add("input");
                    expected_top.Add("priorbox");
                    m_rgLayerParameters[lt] = new PriorBoxParameter();
                    break;

                case LayerType.QUANTILE_ACCURACY:
                    expected_bottom.Add("x");
                    expected_bottom.Add("trgt");
                    expected_top.Add("accuracy");
                    m_rgLayerParameters[lt] = new QuantileAccuracyParameter();
                    break;

                case LayerType.QUANTILE_LOSS:
                    expected_bottom.Add("x");
                    expected_bottom.Add("trgt");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter(LossParameter.NormalizationMode.BATCH_SIZE);
                    m_rgLayerParameters[lt] = new QuantileLossParameter();
                    break;

                case LayerType.REDUCTION:
                    expected_bottom.Add("input");
                    expected_top.Add("reduction");
                    m_rgLayerParameters[lt] = new ReductionParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.RELU:
                    expected_bottom.Add("input");
                    expected_top.Add("relu");
                    m_rgLayerParameters[lt] = new ReLUParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.RESHAPE:
                    expected_bottom.Add("input");
                    expected_top.Add("reshape");
                    m_rgLayerParameters[lt] = new ReshapeParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.RESHAPE_TEMPORAL:
                    expected_bottom.Add("input");
                    expected_top.Add("reshape_t");
                    m_rgLayerParameters[lt] = new ReshapeTemporalParameter();
                    break;

                case LayerType.SQUEEZE:
                    expected_bottom.Add("input");
                    expected_top.Add("squeeze");
                    m_rgLayerParameters[LayerType.SQUEEZE] = new SqueezeParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.UNSQUEEZE:
                    expected_bottom.Add("input");
                    expected_top.Add("unsqueeze");
                    m_rgLayerParameters[LayerType.SQUEEZE] = new SqueezeParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.SCALAR:
                    expected_bottom.Add("input");
                    expected_top.Add("sca");
                    m_rgLayerParameters[lt] = new ScalarParameter();
                    break;

                case LayerType.SCALE:
                    expected_bottom.Add("input");
                    expected_top.Add("scale");
                    m_rgLayerParameters[lt] = new ScaleParameter();
                    break;

                case LayerType.SERF:
                    expected_bottom.Add("input");
                    expected_top.Add("serf");
                    m_rgLayerParameters[lt] = new SerfParameter();
                    break;

                case LayerType.SIGMOID:
                    expected_bottom.Add("input");
                    expected_top.Add("sigmoid");
                    m_rgLayerParameters[lt] = new SigmoidParameter();
                    break;

                case LayerType.SIGMOIDCROSSENTROPY_LOSS:
                    expected_bottom.Add("scores");
                    expected_bottom.Add("trgt");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter(LossParameter.NormalizationMode.BATCH_SIZE);
                    m_rgLayerParameters[LayerType.SIGMOID] = new SigmoidParameter();
                    break;

                case LayerType.SOFTMAXCROSSENTROPY_LOSS:
                    expected_bottom.Add("scores");
                    expected_bottom.Add("trgt");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter(LossParameter.NormalizationMode.BATCH_SIZE);
                    m_rgLayerParameters[LayerType.SOFTMAX] = new SoftmaxParameter();
                    break;

                case LayerType.SOFTMAXCROSSENTROPY2_LOSS:
                    expected_bottom.Add("scores");
                    expected_bottom.Add("trgt");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter(LossParameter.NormalizationMode.BATCH_SIZE);
                    m_rgLayerParameters[LayerType.SOFTMAX] = new SoftmaxParameter();
                    break;

                case LayerType.SILENCE:
                    expected_bottom.Add("input");
                    break;

                case LayerType.SLICE:
                    expected_bottom.Add("input");
                    expected_top.Add("sl1");
                    expected_top.Add("sl2");
                    m_rgLayerParameters[lt] = new SliceParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.SPLIT:
                    expected_bottom.Add("input");
                    expected_top.Add("sp1");
                    expected_top.Add("sp2");
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE_AND_TRAINING;
                    break;

                case LayerType.SOFTMAX:
                    expected_bottom.Add("input");
                    expected_top.Add("softmax");
                    m_rgLayerParameters[lt] = new SoftmaxParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.SOFTMAXWITH_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.SOFTMAX] = new SoftmaxParameter();
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    break;

                case LayerType.SMOOTHL1_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    break;

                case LayerType.SPP:
                    expected_bottom.Add("input");
                    expected_top.Add("spp");
                    m_rgLayerParameters[lt] = new SPPParameter();
                    break;

                case LayerType.SWISH:
                    expected_bottom.Add("input");
                    expected_top.Add("swish");
                    m_rgLayerParameters[lt] = new SwishParameter();
                    break;

                case LayerType.TANH:
                    expected_bottom.Add("input");
                    expected_top.Add("tanh");
                    m_rgLayerParameters[lt] = new TanhParameter();
                    break;

                case LayerType.MODEL_DATA:
                    expected_top.Add("data");
                    expected_top.Add("decinput");
                    m_rgLayerParameters[LayerType.MODEL_DATA] = new ModelDataParameter();
                    break;

                case LayerType.TEXT_DATA:
                    expected_top.Add("data");
                    expected_top.Add("datar");
                    expected_top.Add("decinput");
                    m_rgLayerParameters[LayerType.TEXT_DATA] = new TextDataParameter();
                    break;

                case LayerType.THRESHOLD:
                    expected_bottom.Add("input");
                    expected_top.Add("thresh");
                    m_rgLayerParameters[lt] = new ThresholdParameter();
                    break;

                case LayerType.TILE:
                    expected_bottom.Add("input");
                    expected_top.Add("tile");
                    m_rgLayerParameters[lt] = new TileParameter();
                    break;

                case LayerType.TRANSFORMER_BLOCK:
                    expected_bottom.Add("input");
                    expected_top.Add("tfb");
                    m_rgLayerParameters[lt] = new TransformerBlockParameter();
                    break;

                case LayerType.TOKENIZED_DATA:
                    expected_top.Add("data");
                    expected_top.Add("pos");
                    expected_top.Add("tgt");
                    m_rgLayerParameters[lt] = new TokenizedDataParameter();
                    break;

                case LayerType.TOKENIZED_DATA_PAIRS:
                case LayerType.TOKENIZED_DATA_PAIRS_PY:
                    expected_top.Add("enc");
                    expected_top.Add("dec");
                    expected_top.Add("tgt");
                    expected_top.Add("emsk");
                    expected_top.Add("dmsk");
                    m_rgLayerParameters[LayerType.TOKENIZED_DATA_PAIRS] = new TokenizedDataPairsParameter();
                    break;

                case LayerType.TRANSPOSE:
                    expected_bottom.Add("input");
                    expected_top.Add("output");
                    m_rgLayerParameters[lt] = new TransposeParameter();
                    m_onnxConversionSupport = ONNX_CONVERSION_SUPPORT.INFERENCE;
                    break;

                case LayerType.TRIPLET_LOSS:
                    expected_bottom.Add("anchor");
                    expected_bottom.Add("pos");
                    expected_bottom.Add("neg");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[lt] = new TripletLossParameter();
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    break;

                case LayerType.TV_LOSS:
                    expected_bottom.Add("pred");
                    expected_bottom.Add("label");
                    expected_top.Add("loss");
                    m_rgLayerParameters[LayerType.LOSS] = new LossParameter();
                    m_rgLayerParameters[lt] = new TVLossParameter();
                    break;

                // DEPRECIATED
                case LayerType.LSTM_SIMPLE:
                    expected_bottom.Add("time_seq");
                    expected_bottom.Add("clip");
                    expected_top.Add("lstm");
                    m_rgLayerParameters[LayerType.LSTM_SIMPLE] = new LSTMSimpleParameter();
                    break;

                case LayerType.LSTM_ATTENTION:
                    expected_bottom.Add("input");
                    expected_bottom.Add("clip");
                    expected_top.Add("lstm");
                    m_rgLayerParameters[lt] = new LSTMAttentionParameter();
                    break;

                case LayerType.RNN:
                    expected_bottom.Add("time_seq");
                    expected_bottom.Add("clip");
                    expected_top.Add("rnn");
                    m_rgLayerParameters[LayerType.RECURRENT] = new RecurrentParameter();
                    break;

                case LayerType.LSTM:
                    expected_bottom.Add("time_seq");
                    expected_bottom.Add("clip");
                    expected_top.Add("lstm");
                    m_rgLayerParameters[LayerType.RECURRENT] = new RecurrentParameter();
                    break;

                case LayerType.VIDEO_DATA:
                    expected_top.Add("data");
                    expected_top.Add("label");
                    m_rgLayerParameters[LayerType.VIDEO_DATA] = new VideoDataParameter();
                    m_rgLayerParameters[LayerType.DATA] = new DataParameter();
                    m_rgLayerParameters[LayerType.TRANSFORM] = new TransformationParameter();
                    break;

                case LayerType.VARSELNET:
                    expected_bottom.Add("flatemb");
                    expected_bottom.Add("ctx");
                    expected_top.Add("outsum");
                    expected_top.Add("sprcwts");
                    m_rgLayerParameters[lt] = new VarSelNetParameter();
                    break;
            }
        } 

        /// <summary>
        /// Specifies the name of this LayerParameter.
        /// </summary>
        public string name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        /// <summary>
        /// Specifies the type of this LayerParameter.
        /// </summary>
        public LayerType type
        {
            get { return m_type; }
        }

        /// <summary>
        /// Specifies whether or not to use half sized memory or not.
        /// </summary>
        public bool use_halfsize
        {
            get { return m_bUseHalfSize; }
            set { m_bUseHalfSize = value; }
        }

        /// <summary>
        /// Returns the level of Onnx conversion support.
        /// </summary>
        public ONNX_CONVERSION_SUPPORT onnx_conversion_support
        {
            get { return m_onnxConversionSupport; }
        }

        /// <summary>
        /// Set the layer type.
        /// </summary>
        /// <param name="type">Specifies the new layer type.</param>
        /// <param name="bNewParam">Optionally, specifies to create new params (default = true).</param>
        public void SetType(LayerType type, bool bNewParam = true)
        {
            m_type = type;
            setupParams(type, bNewParam);
        }

        /// <summary>
        /// Specifies the active bottom connections (in the bottom, out the top).
        /// </summary>
        public List<string> bottom
        {
            get { return m_rgstrBottom; }
            set { m_rgstrBottom = value; }
        }

        /// <summary>
        /// Specifies the active top connections (in the bottom, out the top)
        /// </summary>
        public List<string> top
        {
            get { return m_rgstrTop; }
            set { m_rgstrTop = value; }
        }

        /// <summary>
        /// Specifies the Phase for which this LayerParameter is run.
        /// </summary>
        public Phase phase
        {
            get { return m_phase; }
            set { m_phase = value; }
        }

        /// <summary>
        /// Get/set whether or not to freeze the learning for this layer globally.
        /// </summary>
        public bool freeze_learning
        {
            get { return m_bFreezeLearning; }
            set { m_bFreezeLearning = value; }
        }

        /// <summary>
        /// Specifies the loss weight.
        /// </summary>
        public List<double> loss_weight
        {
            get { return m_rgLossWeight; }
            set { m_rgLossWeight = value; }
        }

        /// <summary>
        /// Specifies the ParamSpec parameters of the LayerParameter.
        /// </summary>
        public List<ParamSpec> parameters
        {
            get { return m_rgParams; }
            set { m_rgParams = value; }
        }

        /// <summary>
        /// Specifies the blobs of the LayerParameter.
        /// </summary>
        public List<BlobProto> blobs
        {
            get { return m_rgBlobs; }
            set { m_rgBlobs = value; }
        }

        /// <summary>
        /// Specifies whether or not the LayerParameter (or protions of) should be backpropagated.
        /// </summary>
        public List<bool> propagate_down
        {
            get { return m_rgbPropagateDown; }
            set { m_rgbPropagateDown = value; }
        }

        /// <summary>
        /// Specifies the NetStateRule's for which this LayerParameter should be included.
        /// </summary>
        public List<NetStateRule> include
        {
            get { return m_rgInclude; }
            set { m_rgInclude = value; }
        }

        /// <summary>
        /// Specifies the NetStateRule's for which this LayerParameter should be excluded.
        /// </summary>
        public List<NetStateRule> exclude
        {
            get { return m_rgExclude; }
            set { m_rgExclude = value; }
        }

        /// <summary>
        /// Specifies whether or not this node is the start of a new group - this is only used when rendering models.
        /// </summary>
        public bool group_start
        {
            get { return m_bGroupStart; }
            set { m_bGroupStart = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TRANSFORM
        /// </summary>
        public TransformationParameter transform_param
        {
            get { return (TransformationParameter)m_rgLayerParameters[LayerType.TRANSFORM]; }
            set { m_rgLayerParameters[LayerType.TRANSFORM] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.LOSS
        /// </summary>
        public LossParameter loss_param
        {
            get { return (LossParameter)m_rgLayerParameters[LayerType.LOSS]; }
            set { m_rgLayerParameters[LayerType.LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ACCURACY
        /// </summary>
        public AccuracyParameter accuracy_param
        {
            get { return (AccuracyParameter)m_rgLayerParameters[LayerType.ACCURACY]; }
            set { m_rgLayerParameters[LayerType.ACCURACY] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ARGMAX
        /// </summary>
        public ArgMaxParameter argmax_param
        {
            get { return (ArgMaxParameter)m_rgLayerParameters[LayerType.ARGMAX]; }
            set { m_rgLayerParameters[LayerType.ARGMAX] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.BATCHNORM
        /// </summary>
        public BatchNormParameter batch_norm_param
        {
            get { return (BatchNormParameter)m_rgLayerParameters[LayerType.BATCHNORM]; }
            set { m_rgLayerParameters[LayerType.BATCHNORM] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.BIAS
        /// </summary>
        public BiasParameter bias_param
        {
            get { return (BiasParameter)m_rgLayerParameters[LayerType.BIAS]; }
            set { m_rgLayerParameters[LayerType.BIAS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CLIP
        /// </summary>
        public ClipParameter clip_param
        {
            get { return (ClipParameter)m_rgLayerParameters[LayerType.CLIP]; }
            set { m_rgLayerParameters[LayerType.CLIP] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CONCAT
        /// </summary>
        public ConcatParameter concat_param
        {
            get { return (ConcatParameter)m_rgLayerParameters[LayerType.CONCAT]; }
            set { m_rgLayerParameters[LayerType.CONCAT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CONSTANT
        /// </summary>
        public ConstantParameter constant_param
        {
            get { return (ConstantParameter)m_rgLayerParameters[LayerType.CONSTANT]; }
            set { m_rgLayerParameters[LayerType.CONSTANT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CONTRASTIVE_LOSS
        /// </summary>
        public ContrastiveLossParameter contrastive_loss_param
        {
            get { return (ContrastiveLossParameter)m_rgLayerParameters[LayerType.CONTRASTIVE_LOSS]; }
            set { m_rgLayerParameters[LayerType.CONTRASTIVE_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CONVOLUTION
        /// </summary>
        public ConvolutionParameter convolution_param
        {
            get { return (ConvolutionParameter)m_rgLayerParameters[LayerType.CONVOLUTION]; }
            set { m_rgLayerParameters[LayerType.CONVOLUTION] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CONVOLUTION_OCTAVE
        /// </summary>
        public ConvolutionOctaveParameter convolution_octave_param
        {
            get { return (ConvolutionOctaveParameter)m_rgLayerParameters[LayerType.CONVOLUTION_OCTAVE]; }
            set { m_rgLayerParameters[LayerType.CONVOLUTION_OCTAVE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CROP
        /// </summary>
        public CropParameter crop_param
        {
            get { return (CropParameter)m_rgLayerParameters[LayerType.CROP]; }
            set { m_rgLayerParameters[LayerType.CROP] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initializing with LayerType.DECODE or LayerType.ACCURACY_ENCODING;
        /// </summary>
        public DecodeParameter decode_param
        {
            get { return (DecodeParameter)m_rgLayerParameters[LayerType.DECODE]; }
            set { m_rgLayerParameters[LayerType.DECODE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ANNOTATED_DATA
        /// </summary>
        public AnnotatedDataParameter annotated_data_param
        {
            get { return (AnnotatedDataParameter)m_rgLayerParameters[LayerType.ANNOTATED_DATA]; }
            set { m_rgLayerParameters[LayerType.ANNOTATED_DATA] = value; }
        }


        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ATTENTION
        /// </summary>
        public AttentionParameter attention_param
        {
            get { return (AttentionParameter)m_rgLayerParameters[LayerType.ATTENTION]; }
            set { m_rgLayerParameters[LayerType.ATTENTION] = value; }
        }


        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CATEGORICAL_TRANS
        /// </summary>
        public CategoricalTransformationParameter categorical_trans_param
        {
            get { return (CategoricalTransformationParameter)m_rgLayerParameters[LayerType.CATEGORICAL_TRANS]; }
            set { m_rgLayerParameters[LayerType.CATEGORICAL_TRANS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.CAUSAL_SELF_ATTENTION
        /// </summary>
        public CausalSelfAttentionParameter causal_self_attention_param
        {
            get { return (CausalSelfAttentionParameter)m_rgLayerParameters[LayerType.CAUSAL_SELF_ATTENTION]; }
            set { m_rgLayerParameters[LayerType.CAUSAL_SELF_ATTENTION] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MULTIHEAD_ATTENTION
        /// </summary>
        public MultiheadAttentionParameter multihead_attention_param
        {
            get { return (MultiheadAttentionParameter)m_rgLayerParameters[LayerType.MULTIHEAD_ATTENTION]; }
            set { m_rgLayerParameters[LayerType.MULTIHEAD_ATTENTION] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MULTIHEAD_ATTENTION_INTERP
        /// </summary>
        public MultiHeadAttentionInterpParameter multihead_attention_interp_param
        {
            get { return (MultiHeadAttentionInterpParameter)m_rgLayerParameters[LayerType.MULTIHEAD_ATTENTION_INTERP]; }
            set { m_rgLayerParameters[LayerType.MULTIHEAD_ATTENTION_INTERP] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.POSITIONAL_ENCODER
        /// </summary>
        public PositionalEncoderParameter positional_encoder_param
        {
            get { return (PositionalEncoderParameter)m_rgLayerParameters[LayerType.POSITIONAL_ENCODER]; }
            set { m_rgLayerParameters[LayerType.POSITIONAL_ENCODER] = value; }
        }

        /// <summary>
        /// Returns the parmeter set when initialized with LayerType.DETECTION_EVALUATE
        /// </summary>
        public DetectionEvaluateParameter detection_evaluate_param
        {
            get { return (DetectionEvaluateParameter)m_rgLayerParameters[LayerType.DETECTION_EVALUATE]; }
            set { m_rgLayerParameters[LayerType.DETECTION_EVALUATE] = value; }
        }

        /// <summary>
        /// Returns the parmeter set when initialized with LayerType.DETECTION_OUTPUT
        /// </summary>
        public DetectionOutputParameter detection_output_param
        {
            get { return (DetectionOutputParameter)m_rgLayerParameters[LayerType.DETECTION_OUTPUT]; }
            set { m_rgLayerParameters[LayerType.DETECTION_OUTPUT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DATA
        /// </summary>
        public DataParameter data_param
        {
            get { return (DataParameter)m_rgLayerParameters[LayerType.DATA]; }
            set { m_rgLayerParameters[LayerType.DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DATA_NORMALIZER
        /// </summary>
        public DataNormalizerParameter data_normalizer_param
        {
            get { return (DataNormalizerParameter)m_rgLayerParameters[LayerType.DATA_NORMALIZER]; }
            set { m_rgLayerParameters[LayerType.DATA_NORMALIZER] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DATA_SEQUENCE
        /// </summary>
        public DataSequenceParameter data_sequence_param
        {
            get { return (DataSequenceParameter)m_rgLayerParameters[LayerType.DATA_SEQUENCE]; }
            set { m_rgLayerParameters[LayerType.DATA_SEQUENCE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DATA_TEMPORAL
        /// </summary>
        public DataTemporalParameter data_temporal_param
        {
            get { return (DataTemporalParameter)m_rgLayerParameters[LayerType.DATA_TEMPORAL]; }
            set { m_rgLayerParameters[LayerType.DATA_TEMPORAL] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DEBUG
        /// </summary>
        public DebugParameter debug_param
        {
            get { return (DebugParameter)m_rgLayerParameters[LayerType.DEBUG]; }
            set { m_rgLayerParameters[LayerType.DEBUG] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DROPOUT
        /// </summary>
        public DropoutParameter dropout_param
        {
            get { return (DropoutParameter)m_rgLayerParameters[LayerType.DROPOUT]; }
            set { m_rgLayerParameters[LayerType.DROPOUT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.DUMMYDATA
        /// </summary>
        public DummyDataParameter dummy_data_param
        {
            get { return (DummyDataParameter)m_rgLayerParameters[LayerType.DUMMYDATA]; }
            set { m_rgLayerParameters[LayerType.DUMMYDATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ELTWISE
        /// </summary>
        public EltwiseParameter eltwise_param
        {
            get { return (EltwiseParameter)m_rgLayerParameters[LayerType.ELTWISE]; }
            set { m_rgLayerParameters[LayerType.ELTWISE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ELU
        /// </summary>
        public EluParameter elu_param
        {
            get { return (EluParameter)m_rgLayerParameters[LayerType.ELU]; }
            set { m_rgLayerParameters[LayerType.ELU] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.EMBED
        /// </summary>
        public EmbedParameter embed_param
        {
            get { return (EmbedParameter)m_rgLayerParameters[LayerType.EMBED]; }
            set { m_rgLayerParameters[LayerType.EMBED] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.EXP
        /// </summary>
        public ExpParameter exp_param
        {
            get { return (ExpParameter)m_rgLayerParameters[LayerType.EXP]; }
            set { m_rgLayerParameters[LayerType.EXP] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.FLATTEN
        /// </summary>
        public FlattenParameter flatten_param
        {
            get { return (FlattenParameter)m_rgLayerParameters[LayerType.FLATTEN]; }
            set { m_rgLayerParameters[LayerType.FLATTEN] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GATHER
        /// </summary>
        public GatherParameter gather_param
        {
            get { return (GatherParameter)m_rgLayerParameters[LayerType.GATHER]; }
            set { m_rgLayerParameters[LayerType.GATHER] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GLU
        /// </summary>
        public GateAddNormParameter gateaddnorm_param
        {
            get { return (GateAddNormParameter)m_rgLayerParameters[LayerType.GATEADDNORM]; }
            set { m_rgLayerParameters[LayerType.GATEADDNORM] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GELU
        /// </summary>
        public GeluParameter gelu_param
        {
            get { return (GeluParameter)m_rgLayerParameters[LayerType.GELU]; }
            set { m_rgLayerParameters[LayerType.GELU] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GLU
        /// </summary>
        public GluParameter glu_param
        {
            get { return (GluParameter)m_rgLayerParameters[LayerType.GLU]; }
            set { m_rgLayerParameters[LayerType.GLU] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GLU
        /// </summary>
        public GrnParameter grn_param
        {
            get { return (GrnParameter)m_rgLayerParameters[LayerType.GRN]; }
            set { m_rgLayerParameters[LayerType.GRN] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GSL
        /// </summary>
        public GradientScaleParameter gradient_scale_param
        {
            get { return (GradientScaleParameter)m_rgLayerParameters[LayerType.GRADIENTSCALER]; }
            set { m_rgLayerParameters[LayerType.GRADIENTSCALER] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.GRAM
        /// </summary>
        public GramParameter gram_param
        {
            get { return (GramParameter)m_rgLayerParameters[LayerType.GRAM]; }
            set { m_rgLayerParameters[LayerType.GRAM] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.HDF5_DATA
        /// </summary>
        public HDF5DataParameter hdf5_data_param
        {
            get { return (HDF5DataParameter)m_rgLayerParameters[LayerType.HDF5_DATA]; }
            set { m_rgLayerParameters[LayerType.HDF5_DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.HINGE_LOSS
        /// </summary>
        public HingeLossParameter hinge_loss_param
        {
            get { return (HingeLossParameter)m_rgLayerParameters[LayerType.HINGE_LOSS]; }
            set { m_rgLayerParameters[LayerType.HINGE_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.IMAGE_DATA
        /// </summary>
        public ImageDataParameter image_data_param
        {
            get { return (ImageDataParameter)m_rgLayerParameters[LayerType.IMAGE_DATA]; }
            set { m_rgLayerParameters[LayerType.IMAGE_DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.INFOGAIN_LOSS
        /// </summary>
        public InfogainLossParameter infogain_loss_param
        {
            get { return (InfogainLossParameter)m_rgLayerParameters[LayerType.INFOGAIN_LOSS]; }
            set { m_rgLayerParameters[LayerType.INFOGAIN_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.INNERPRODUCT
        /// </summary>
        public InnerProductParameter inner_product_param
        {
            get { return (InnerProductParameter)m_rgLayerParameters[LayerType.INNERPRODUCT]; }
            set { m_rgLayerParameters[LayerType.INNERPRODUCT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initializing the LayerType.INTERP
        /// </summary>
        public InterpParameter interp_param
        {
            get { return (InterpParameter)m_rgLayerParameters[LayerType.INTERP]; }
            set { m_rgLayerParameters[LayerType.INTERP] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.KNN
        /// </summary>
        public KnnParameter knn_param
        {
            get { return (KnnParameter)m_rgLayerParameters[LayerType.KNN]; }
            set { m_rgLayerParameters[LayerType.KNN] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.LABELMAPPING
        /// </summary>
        public LabelMappingParameter labelmapping_param
        {
            get { return (LabelMappingParameter)m_rgLayerParameters[LayerType.LABELMAPPING]; }
            set { m_rgLayerParameters[LayerType.LABELMAPPING] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.LAYERNORM
        /// </summary>
        public LayerNormParameter layer_norm_param
        {
            get { return (LayerNormParameter)m_rgLayerParameters[LayerType.LAYERNORM]; }
            set { m_rgLayerParameters[LayerType.LAYERNORM] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.LOG
        /// </summary>
        public LogParameter log_param
        {
            get { return (LogParameter)m_rgLayerParameters[LayerType.LOG]; }
            set { m_rgLayerParameters[LayerType.LOG] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.LRN
        /// </summary>
        public LRNParameter lrn_param
        {
            get { return (LRNParameter)m_rgLayerParameters[LayerType.LRN]; }
            set { m_rgLayerParameters[LayerType.LRN] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MEAN_ERROR_LOSS
        /// </summary>
        public MeanErrorLossParameter mean_error_loss_param
        {
            get { return (MeanErrorLossParameter)m_rgLayerParameters[LayerType.MEAN_ERROR_LOSS]; }
            set { m_rgLayerParameters[LayerType.MEAN_ERROR_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MATH
        /// </summary>
        public MathParameter math_param
        {
            get { return (MathParameter)m_rgLayerParameters[LayerType.MATH]; }
            set { m_rgLayerParameters[LayerType.MATH] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MERGE
        /// </summary>
        public MergeParameter merge_param
        {
            get { return (MergeParameter)m_rgLayerParameters[LayerType.MERGE]; }
            set { m_rgLayerParameters[LayerType.MERGE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MEMORY_DATA
        /// </summary>
        public MemoryDataParameter memory_data_param
        {
            get { return (MemoryDataParameter)m_rgLayerParameters[LayerType.MEMORYDATA]; }
            set { m_rgLayerParameters[LayerType.MEMORYDATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MISH
        /// </summary>
        public MishParameter mish_param
        {
            get { return (MishParameter)m_rgLayerParameters[LayerType.MISH]; }
            set { m_rgLayerParameters[LayerType.MISH] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initializing with LayerType.MULTIBOX_LOSS
        /// </summary>
        public MultiBoxLossParameter multiboxloss_param
        {
            get { return (MultiBoxLossParameter)m_rgLayerParameters[LayerType.MULTIBOX_LOSS]; }
            set { m_rgLayerParameters[LayerType.MULTIBOX_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MVN
        /// </summary>
        public MVNParameter mvn_param
        {
            get { return (MVNParameter)m_rgLayerParameters[LayerType.MVN]; }
            set { m_rgLayerParameters[LayerType.MVN] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.NLL_LOSS
        /// </summary>
        public NLLLossParameter nll_loss_param
        {
            get { return (NLLLossParameter)m_rgLayerParameters[LayerType.NLL_LOSS]; }
            set { m_rgLayerParameters[LayerType.NLL_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.NUMERIC_TRANS
        /// </summary>
        public NumericTransformationParameter numeric_trans_param
        {
            get { return (NumericTransformationParameter)m_rgLayerParameters[LayerType.NUMERIC_TRANS]; }
            set { m_rgLayerParameters[LayerType.NUMERIC_TRANS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.ONEHOT
        /// </summary>
        public OneHotParameter onehot_param
        {
            get { return (OneHotParameter)m_rgLayerParameters[LayerType.ONEHOT]; }
            set { m_rgLayerParameters[LayerType.ONEHOT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.NORMALIZATION1
        /// </summary>
        public Normalization1Parameter normalization1_param
        {
            get { return (Normalization1Parameter)m_rgLayerParameters[LayerType.NORMALIZATION1]; }
            set { m_rgLayerParameters[LayerType.NORMALIZATION1] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.NORMALIZATION2
        /// </summary>
        public Normalization2Parameter normalization2_param
        {
            get { return (Normalization2Parameter)m_rgLayerParameters[LayerType.NORMALIZATION2]; }
            set { m_rgLayerParameters[LayerType.NORMALIZATION2] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.POOLING
        /// </summary>
        public PoolingParameter pooling_param
        {
            get { return (PoolingParameter)m_rgLayerParameters[LayerType.POOLING]; }
            set { m_rgLayerParameters[LayerType.POOLING] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.UNPOOLING
        /// </summary>
        public UnPoolingParameter unpooling_param
        {
            get { return (UnPoolingParameter)m_rgLayerParameters[LayerType.UNPOOLING]; }
            set { m_rgLayerParameters[LayerType.UNPOOLING] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.PARAMETER
        /// </summary>
        public ParameterParameter parameter_param
        {
            get { return (ParameterParameter)m_rgLayerParameters[LayerType.PARAMETER]; }
            set { m_rgLayerParameters[LayerType.PARAMETER] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.PERMUTE
        /// </summary>
        public PermuteParameter permute_param
        {
            get { return (PermuteParameter)m_rgLayerParameters[LayerType.PERMUTE]; }
            set { m_rgLayerParameters[LayerType.PERMUTE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.POWER
        /// </summary>
        public PowerParameter power_param
        {
            get { return (PowerParameter)m_rgLayerParameters[LayerType.POWER]; }
            set { m_rgLayerParameters[LayerType.POWER] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.PRELU
        /// </summary>
        public PReLUParameter prelu_param
        {
            get { return (PReLUParameter)m_rgLayerParameters[LayerType.PRELU]; }
            set { m_rgLayerParameters[LayerType.PRELU] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.PRIORBOX
        /// </summary>
        public PriorBoxParameter prior_box_param
        {
            get { return (PriorBoxParameter)m_rgLayerParameters[LayerType.PRIORBOX]; }
            set { m_rgLayerParameters[LayerType.PRIORBOX] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.QUANTILE_ACCURACY
        /// </summary>
        public QuantileAccuracyParameter quantile_accuracy_param
        {
            get { return (QuantileAccuracyParameter)m_rgLayerParameters[LayerType.QUANTILE_ACCURACY]; }
            set { m_rgLayerParameters[LayerType.QUANTILE_ACCURACY] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.QUANTILE_LOSS
        /// </summary>
        public QuantileLossParameter quantile_loss_param
        {
            get { return (QuantileLossParameter)m_rgLayerParameters[LayerType.QUANTILE_LOSS]; }
            set { m_rgLayerParameters[LayerType.QUANTILE_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.REDUCTION
        /// </summary>
        public ReductionParameter reduction_param
        {
            get { return (ReductionParameter)m_rgLayerParameters[LayerType.REDUCTION]; }
            set { m_rgLayerParameters[LayerType.REDUCTION] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.RELU
        /// </summary>
        public ReLUParameter relu_param
        {
            get { return (ReLUParameter)m_rgLayerParameters[LayerType.RELU]; }
            set { m_rgLayerParameters[LayerType.RELU] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.RESHAPE
        /// </summary>
        public ReshapeParameter reshape_param
        {
            get { return (ReshapeParameter)m_rgLayerParameters[LayerType.RESHAPE]; }
            set { m_rgLayerParameters[LayerType.RESHAPE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.RESHAPE_TEMPORAL
        /// </summary>
        public ReshapeTemporalParameter reshape_temporal_param
        {
            get { return (ReshapeTemporalParameter)m_rgLayerParameters[LayerType.RESHAPE_TEMPORAL]; }
            set { m_rgLayerParameters[LayerType.RESHAPE_TEMPORAL] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.RESHAPE
        /// </summary>
        public SqueezeParameter squeeze_param
        {
            get { return (SqueezeParameter)m_rgLayerParameters[LayerType.SQUEEZE]; }
            set { m_rgLayerParameters[LayerType.SQUEEZE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SCALAR
        /// </summary>
        public ScalarParameter scalar_param
        {
            get { return (ScalarParameter)m_rgLayerParameters[LayerType.SCALAR]; }
            set { m_rgLayerParameters[LayerType.SCALAR] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SCALE
        /// </summary>
        public ScaleParameter scale_param
        {
            get { return (ScaleParameter)m_rgLayerParameters[LayerType.SCALE]; }
            set { m_rgLayerParameters[LayerType.SCALE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SERF
        /// </summary>
        public SerfParameter serf_param
        {
            get { return (SerfParameter)m_rgLayerParameters[LayerType.SERF]; }
            set { m_rgLayerParameters[LayerType.SERF] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SIGMOID
        /// </summary>
        public SigmoidParameter sigmoid_param
        {
            get { return (SigmoidParameter)m_rgLayerParameters[LayerType.SIGMOID]; }
            set { m_rgLayerParameters[LayerType.SIGMOID] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SOFTMAX
        /// </summary>
        public SoftmaxParameter softmax_param
        {
            get { return (SoftmaxParameter)m_rgLayerParameters[LayerType.SOFTMAX]; }
            set { m_rgLayerParameters[LayerType.SOFTMAX] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SPP
        /// </summary>
        public SPPParameter spp_param
        {
            get { return (SPPParameter)m_rgLayerParameters[LayerType.SPP]; }
            set { m_rgLayerParameters[LayerType.SPP] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SLICE
        /// </summary>
        public SliceParameter slice_param
        {
            get { return (SliceParameter)m_rgLayerParameters[LayerType.SLICE]; }
            set { m_rgLayerParameters[LayerType.SLICE] = value; }
        }


        /// <summary>
        /// Returns the parameter set when initialized with LayerType.SWISH
        /// </summary>
        public SwishParameter swish_param
        {
            get { return (SwishParameter)m_rgLayerParameters[LayerType.SWISH]; }
            set { m_rgLayerParameters[LayerType.SWISH] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TANH
        /// </summary>
        public TanhParameter tanh_param
        {
            get { return (TanhParameter)m_rgLayerParameters[LayerType.TANH]; }
            set { m_rgLayerParameters[LayerType.TANH] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.MODEL_DATA
        /// </summary>
        public ModelDataParameter model_data_param
        {
            get { return (ModelDataParameter)m_rgLayerParameters[LayerType.MODEL_DATA]; }
            set { m_rgLayerParameters[LayerType.MODEL_DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TEXT_DATA
        /// </summary>
        public TextDataParameter text_data_param
        {
            get { return (TextDataParameter)m_rgLayerParameters[LayerType.TEXT_DATA]; }
            set { m_rgLayerParameters[LayerType.TEXT_DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.THRESHOLD
        /// </summary>
        public ThresholdParameter threshold_param
        {
            get { return (ThresholdParameter)m_rgLayerParameters[LayerType.THRESHOLD]; }
            set { m_rgLayerParameters[LayerType.THRESHOLD] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TILE
        /// </summary>
        public TileParameter tile_param
        {
            get { return (TileParameter)m_rgLayerParameters[LayerType.TILE]; }
            set { m_rgLayerParameters[LayerType.TILE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TRANSPOSE
        /// </summary>
        public TransposeParameter transpose_param
        {
            get { return (TransposeParameter)m_rgLayerParameters[LayerType.TRANSPOSE]; }
            set { m_rgLayerParameters[LayerType.TRANSPOSE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TRANSFORMER_BLOCK
        /// </summary>
        public TransformerBlockParameter transformer_block_param
        {
            get { return (TransformerBlockParameter)m_rgLayerParameters[LayerType.TRANSFORMER_BLOCK]; }
            set { m_rgLayerParameters[LayerType.TRANSFORMER_BLOCK] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TOKENIZED_DATA
        /// </summary>
        public TokenizedDataParameter tokenized_data_param
        {
            get { return (TokenizedDataParameter)m_rgLayerParameters[LayerType.TOKENIZED_DATA]; }
            set { m_rgLayerParameters[LayerType.TOKENIZED_DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TOKENIZED_DATA_PAIRS
        /// </summary>
        public TokenizedDataPairsParameter tokenized_data_pairs_param
        {
            get { return (TokenizedDataPairsParameter)m_rgLayerParameters[LayerType.TOKENIZED_DATA_PAIRS]; }
            set { m_rgLayerParameters[LayerType.TOKENIZED_DATA_PAIRS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TRIPLET_LOSS
        /// </summary>
        public TripletLossParameter triplet_loss_param
        {
            get { return (TripletLossParameter)m_rgLayerParameters[LayerType.TRIPLET_LOSS]; }
            set { m_rgLayerParameters[LayerType.TRIPLET_LOSS] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.TV_LOSS
        /// </summary>
        public TVLossParameter tv_loss_param
        {
            get { return (TVLossParameter)m_rgLayerParameters[LayerType.TV_LOSS]; }
            set { m_rgLayerParameters[LayerType.TV_LOSS] = value; }
        }

        /// <summary>
        /// [DEPRECIATED] Returns the parameter set when initialized with LayerType.LSTM_SIMPLE
        /// </summary>
        public LSTMSimpleParameter lstm_simple_param
        {
            get { return (LSTMSimpleParameter)m_rgLayerParameters[LayerType.LSTM_SIMPLE]; }
            set { m_rgLayerParameters[LayerType.LSTM_SIMPLE] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.LSTM_ATTENTION
        /// </summary>
        public LSTMAttentionParameter lstm_attention_param
        {
            get { return (LSTMAttentionParameter)m_rgLayerParameters[LayerType.LSTM_ATTENTION]; }
            set { m_rgLayerParameters[LayerType.LSTM_ATTENTION] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.RECURRENT
        /// </summary>
        public RecurrentParameter recurrent_param
        {
            get { return (RecurrentParameter)m_rgLayerParameters[LayerType.RECURRENT]; }
            set { m_rgLayerParameters[LayerType.RECURRENT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.INPUT
        /// </summary>
        public InputParameter input_param
        {
            get { return (InputParameter)m_rgLayerParameters[LayerType.INPUT]; }
            set { m_rgLayerParameters[LayerType.INPUT] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.VIDEO_DATA
        /// </summary>
        public VideoDataParameter video_data_param
        {
            get { return (VideoDataParameter)m_rgLayerParameters[LayerType.VIDEO_DATA]; }
            set { m_rgLayerParameters[LayerType.VIDEO_DATA] = value; }
        }

        /// <summary>
        /// Returns the parameter set when initialized with LayerType.VARSELNET
        /// </summary>
        public VarSelNetParameter varselnet_param
        {
            get { return (VarSelNetParameter)m_rgLayerParameters[LayerType.VARSELNET]; }
            set { m_rgLayerParameters[LayerType.VARSELNET] = value; }
        }

        /// <summary>
        /// Clears the collection of Blobs used by this layer.
        /// </summary>
        public void clear_blobs()
        {
            m_rgBlobs.Clear();
        }

        /// <summary>
        /// Returns the number of Solvers participating in a multi-GPU session for which the Solver using this LayerParameter is associated.
        /// </summary>
        public int solver_count
        {
            get { return m_nSolverCount; }
            set { m_nSolverCount = value; }
        }

        /// <summary>
        /// Returns the SolverRank of the Solver using this LayerParameter (if any).
        /// </summary>
        public int solver_rank
        {
            get { return m_nSolverRank; }
            set { m_nSolverRank = value; }
        }

        /// <summary>
        /// Returns a list of <i>expected</i> top connections (in the bottom, out the top).
        /// </summary>
        public List<string> expected_top
        {
            get { return m_rgstrExpectedTop; }
        }

        /// <summary>
        /// Returns a list of <i>expected</i> bottom connections (in the bottom, out the top).
        /// </summary>
        public List<string> expected_bottom
        {
            get { return m_rgstrExpectedBottom; }
        }

        /// <summary>
        /// Copy just the layer specific parameters to this layer parameter.
        /// </summary>
        /// <param name="src">Specifies the source who's specific layer parameters are to be compied.</param>
        public void CopyParameters(LayerParameter src)
        {
            m_rgLayerParameters = new Dictionary<LayerType, LayerParameterBase>();

            foreach (KeyValuePair<LayerType, LayerParameterBase> kv in src.m_rgLayerParameters)
            {
                if (kv.Value != null)
                    m_rgLayerParameters.Add(kv.Key, kv.Value.Clone());
                else
                    m_rgLayerParameters.Add(kv.Key, null);
            }
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public virtual LayerParameter Clone(bool bCloneBlobs)
        {
            LayerParameter p = new LayerParameter(m_type, m_strName);

            p.m_rgstrBottom = Utility.Clone<string>(m_rgstrBottom);
            p.m_rgstrTop = Utility.Clone<string>(m_rgstrTop);
            p.m_phase = m_phase;
            p.m_rgLossWeight = Utility.Clone<double>(m_rgLossWeight);
            p.m_rgParams = Utility.Clone<ParamSpec>(m_rgParams);

            if (bCloneBlobs)
                p.m_rgBlobs = Utility.Clone<BlobProto>(m_rgBlobs);

            p.m_rgbPropagateDown = Utility.Clone<bool>(m_rgbPropagateDown);
            p.m_rgInclude = Utility.Clone<NetStateRule>(m_rgInclude);
            p.m_rgExclude = Utility.Clone<NetStateRule>(m_rgExclude);
            p.m_bFreezeLearning = m_bFreezeLearning;

            p.m_rgLayerParameters = new Dictionary<LayerType, LayerParameterBase>();

            foreach (KeyValuePair<LayerType, LayerParameterBase> kv in m_rgLayerParameters)
            {
                if (kv.Value != null)
                    p.m_rgLayerParameters.Add(kv.Key, kv.Value.Clone());
                else
                    p.m_rgLayerParameters.Add(kv.Key, null);
            }

            p.m_nSolverCount = m_nSolverCount;
            p.m_nSolverRank = m_nSolverRank;
            p.m_bGroupStart = m_bGroupStart;
            p.m_bUseHalfSize = m_bUseHalfSize;

            return p;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        object ICloneable.Clone()
        {
            return Clone(true);
        }

        /** @copydoc BaseParameter */
        public int CompareTo(object obj)
        {
            LayerParameter p = obj as LayerParameter;

            if (p == null)
                return 1;

            if (!Compare(p))
                return 1;

            return 0;
        }

        private string getTypeString(LayerType type)
        {
            switch (type)
            {
                case LayerType.ABSVAL:
                    return "AbsVal";

                case LayerType.ACCURACY:
                    return "Accuracy";

                case LayerType.ACCURACY_DECODE:
                    return "AccuracyDecode";

                case LayerType.ACCURACY_ENCODING:
                    return "AccuracyEncoding";

                case LayerType.ARGMAX:
                    return "ArgMax";

                case LayerType.ANNOTATED_DATA:
                    return "AnnotatedData";

                case LayerType.ATTENTION:
                    return "Attention";

                case LayerType.BATCHNORM:
                    return "BatchNorm";

                case LayerType.BATCHREINDEX:
                    return "BatchReIndex";

                case LayerType.BIAS:
                    return "Bias";

                case LayerType.BNLL:
                    return "BNLL";

                case LayerType.CATEGORICAL_TRANS:
                    return "CategoricalTrans";

                case LayerType.CAUSAL_SELF_ATTENTION:
                    return "CausalSelfAttention";

                case LayerType.CHANNEL_EMBEDDING:
                    return "ChannelEmbedding";

                case LayerType.CLIP:
                    return "Clip";

                case LayerType.CONCAT:
                    return "Concat";

                case LayerType.CONSTANT:
                    return "Constant";

                case LayerType.CONTRASTIVE_LOSS:
                    return "ContrastiveLoss";

                case LayerType.CONVOLUTION:
                    return "Convolution";

                case LayerType.CONVOLUTION_OCTAVE:
                    return "ConvolutionOctave";

                case LayerType.CROP:
                    return "Crop";

                case LayerType.COPY:
                    return "Copy";

                case LayerType.DECODE:
                    return "Decode";

                case LayerType.DATA:
                    return "Data";

                case LayerType.DATA_NORMALIZER:
                    return "DataNormalizer";

                case LayerType.DATA_SEQUENCE:
                    return "DataSequence";

                case LayerType.DATA_TEMPORAL:
                    return "DataTemporal";

                case LayerType.DEBUG:
                    return "Debug";

                case LayerType.DECONVOLUTION:
                    return "Deconvolution";

                case LayerType.DETECTION_EVALUATE:
                    return "DetectionEvaluate";

                case LayerType.DETECTION_OUTPUT:
                    return "DetectionOutput";

                case LayerType.DROPOUT:
                    return "Dropout";

                case LayerType.DUMMYDATA:
                    return "DummyData";

                case LayerType.ELTWISE:
                    return "Eltwise";

                case LayerType.ELU:
                    return "ELU";

                case LayerType.EMBED:
                    return "Embed";

                case LayerType.EUCLIDEAN_LOSS:
                    return "EuclideanLoss";

                case LayerType.EVENT:
                    return "Event";

                case LayerType.EXP:
                    return "EXP";

                case LayerType.FILTER:
                    return "Filter";

                case LayerType.FLATTEN:
                    return "Flatten";

                case LayerType.GATHER:
                    return "Gather";

                case LayerType.GATEADDNORM:
                    return "GateAddNorm";

                case LayerType.GELU:
                    return "GELU";

                case LayerType.GLU:
                    return "GLU";

                case LayerType.GRN:
                    return "GRN";

                case LayerType.GLOBRES_NORM:
                    return "GlobResNorm";

                case LayerType.GRADIENTSCALER:
                    return "GSL";

                case LayerType.GRAM:
                    return "Gram";

                case LayerType.HDF5_DATA:
                    return "HDF5Data";

                case LayerType.HINGE_LOSS:
                    return "HingeLoss";

                case LayerType.IMAGE_DATA:
                    return "ImageData";

                case LayerType.IM2COL:
                    return "Im2Col";

                case LayerType.INFOGAIN_LOSS:
                    return "InfogainLoss";

                case LayerType.INNERPRODUCT:
                    return "InnerProduct";

                case LayerType.INPUT:
                    return "Input";

                case LayerType.INTERP:
                    return "Interp";

                case LayerType.KNN:
                    return "Knn";

                case LayerType.LABELMAPPING:
                    return "LabelMapping";

                case LayerType.LAYERNORM:
                    return "LayerNorm";

                case LayerType.LOG:
                    return "Log";

                case LayerType.LOSS:
                    return "Loss";

                case LayerType.LRN:
                    return "LRN";

                case LayerType.MEAN_ERROR_LOSS:
                    return "MeanErrorLoss";

                case LayerType.MATH:
                    return "MATH";

                case LayerType.MERGE:
                    return "Merge";
               
                case LayerType.MEMORYDATA:
                    return "MemoryData";

                case LayerType.MULTIBOX_LOSS:
                    return "MultiBoxLoss";

                case LayerType.MULTIHEAD_ATTENTION:
                    return "MultiheadAttention";

                case LayerType.MULTIHEAD_ATTENTION_INTERP:
                    return "MultiheadAttentionInterp";

                case LayerType.MEMORY_LOSS:
                    return "MemoryLoss";

                case LayerType.MISH:
                    return "Mish";

                case LayerType.MULTINOMIALLOGISTIC_LOSS:
                    return "MultinomialLogisticLoss";

                case LayerType.MVN:
                    return "MVN";

                case LayerType.NLL_LOSS:
                    return "NLLLoss";

                case LayerType.NUMERIC_TRANS:
                    return "NumericTrans";

                case LayerType.ONEHOT:
                    return "OneHot";

                case LayerType.NORMALIZATION1:
                    return "Normalization1";

                case LayerType.NORMALIZATION2:
                    return "Normalization2";

                case LayerType.PARAMETER:
                    return "Parameter";

                case LayerType.PERMUTE:
                    return "Permute";

                case LayerType.POSITIONAL_ENCODER:
                    return "PositionalEncoder";

                case LayerType.POOLING:
                    return "Pooling";

                case LayerType.UNPOOLING1:
                    return "UnPooling1";

                case LayerType.UNPOOLING:
                    return "UnPooling";

                case LayerType.POWER:
                    return "Power";

                case LayerType.PRELU:
                    return "PReLU";

                case LayerType.PRIORBOX:
                    return "PriorBox";

                case LayerType.QUANTILE_ACCURACY:
                    return "QuantileAccuracy";

                case LayerType.QUANTILE_LOSS:
                    return "QuantileLoss";

                case LayerType.REDUCTION:
                    return "Reduction";

                case LayerType.RELU:
                    return "ReLU";

                case LayerType.RESHAPE:
                    return "Reshape";

                case LayerType.RESHAPE_TEMPORAL:
                    return "ReshapeTemporal";

                case LayerType.SQUEEZE:
                    return "Squeeze";

                case LayerType.UNSQUEEZE:
                    return "Unsqueeze";

                case LayerType.SCALAR:
                    return "Scalar";

                case LayerType.SCALE:
                    return "Scale";

                case LayerType.SERF:
                    return "Serf";

                case LayerType.SIGMOID:
                    return "Sigmoid";

                case LayerType.SIGMOIDCROSSENTROPY_LOSS:
                    return "SigmoidCrossEntropyLoss";

                case LayerType.SOFTMAXCROSSENTROPY_LOSS:
                    return "SoftmaxCrossEntropyLoss";

                case LayerType.SOFTMAXCROSSENTROPY2_LOSS:
                    return "SoftmaxCrossEntropy2Loss";

                case LayerType.SILENCE:
                    return "Silence";

                case LayerType.SLICE:
                    return "Slice";

                case LayerType.SOFTMAX:
                    return "Softmax";

                case LayerType.SOFTMAXWITH_LOSS:
                    return "SoftmaxWithLoss";

                case LayerType.SMOOTHL1_LOSS:
                    return "SmoothL1Loss";

                case LayerType.SPLIT:
                    return "Split";

                case LayerType.SPP:
                    return "SPP";

                case LayerType.SWISH:
                    return "Swish";

                case LayerType.TANH:
                    return "TanH";

                case LayerType.MODEL_DATA:
                    return "ModelData";

                case LayerType.TEXT_DATA:
                    return "TextData";

                case LayerType.THRESHOLD:
                    return "Threshold";

                case LayerType.TILE:
                    return "Tile";

                case LayerType.TRANSPOSE:
                    return "Transpose";

                case LayerType.TRANSFORMER_BLOCK:
                    return "TransformerBlock";

                case LayerType.TOKENIZED_DATA:
                    return "TokenizedData";

                case LayerType.TOKENIZED_DATA_PAIRS:
                    return "TokenizedDataPairs";

                case LayerType.TOKENIZED_DATA_PAIRS_PY:
                    return "TokenizedDataPairsPy";

                case LayerType.TRIPLET_LOSS:
                    return "TripletLoss";

                case LayerType.TV_LOSS:
                    return "TVLoss";

                // DEPRECIATED
                case LayerType.LSTM_SIMPLE:
                    return "LstmSimple";

                case LayerType.LSTM_ATTENTION:
                    return "LstmAttention";

                case LayerType.RNN:
                    return "Rnn";

                case LayerType.LSTM:
                    return "Lstm";

                case LayerType.LSTM_UNIT:
                    return "Lstm_Unit";

                case LayerType.VIDEO_DATA:
                    return "VideoData";

                case LayerType.VARSELNET:
                    return "VarSelNet";

                default:
                    return "Unknown";
            }
        }

        /** @copydoc BaseParameter */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("name", name, RawProto.TYPE.STRING);
            rgChildren.Add("type", getTypeString(type), RawProto.TYPE.STRING);
            rgChildren.Add<string>("bottom", bottom);
            rgChildren.Add<string>("top", top);
            rgChildren.Add<double>("loss_weight", loss_weight);

            if (group_start)
                rgChildren.Add("group_start", group_start.ToString());

            if (freeze_learning)
                rgChildren.Add("freeze_learning", freeze_learning.ToString());

            if (use_halfsize)
                rgChildren.Add("use_halfsize", use_halfsize.ToString());

            foreach (ParamSpec ps in parameters)
            {
                rgChildren.Add(ps.ToProto("param"));
            }

            foreach (BlobProto bp in blobs)
            {
                rgChildren.Add(bp.ToProto("blobs"));
            }

            rgChildren.Add<bool>("propagate_down", propagate_down);

            foreach (NetStateRule nsr in include)
            {
                rgChildren.Add(nsr.ToProto("include"));
            }

            foreach (NetStateRule nsr in exclude)
            {
                rgChildren.Add(nsr.ToProto("exclude"));
            }

            foreach (KeyValuePair<Phase, int> kv in m_rgMaxBottomCount)
            {
                RawProtoCollection prChildren = new RawProtoCollection();
                prChildren.Add("phase", kv.Key.ToString());
                prChildren.Add("count", kv.Value.ToString());
                RawProto prMaxBottomCount = new RawProto("max_bottom_count", "", prChildren);
                rgChildren.Add(prMaxBottomCount);
            }

            List<KeyValuePair<BaseParameter, string>> rgParam = new List<KeyValuePair<BaseParameter,string>>();

            // Standard layers.
            rgParam.Add(new KeyValuePair<BaseParameter,string>(transform_param, "transform_param"));
            rgParam.Add(new KeyValuePair<BaseParameter,string>(loss_param, "loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter,string>(accuracy_param, "accuracy_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(argmax_param, "argmax_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(batch_norm_param, "batch_norm_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(bias_param, "bias_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(clip_param, "clip_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(concat_param, "concat_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(constant_param, "constant_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(contrastive_loss_param, "contrastive_loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(convolution_param, "convolution_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(crop_param, "crop_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(data_param, "data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(debug_param, "debug_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(dropout_param, "dropout_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(dummy_data_param, "dummy_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(eltwise_param, "eltwise_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(elu_param, "elu_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(embed_param, "embed_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(exp_param, "exp_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(flatten_param, "flatten_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(gradient_scale_param, "gradient_scale_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(hinge_loss_param, "hinge_loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(image_data_param, "image_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(infogain_loss_param, "infogain_loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(inner_product_param, "inner_product_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(input_param, "input_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(labelmapping_param, "labelmapping_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(log_param, "log_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(lrn_param, "lrn_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(memory_data_param, "memory_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(mvn_param, "mvn_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(pooling_param, "pooling_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(parameter_param, "parameter_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(power_param, "power_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(prelu_param, "prelu_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(reduction_param, "reduction_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(relu_param, "relu_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(reshape_param, "reshape_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(scale_param, "scale_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(sigmoid_param, "sigmoid_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(softmax_param, "softmax_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(spp_param, "spp_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(slice_param, "slice_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(swish_param, "swish_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(tanh_param, "tanh_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(threshold_param, "threshold_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(tile_param, "tile_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(lstm_simple_param, "lstm_simple_param")); // DEPRECIATED
            rgParam.Add(new KeyValuePair<BaseParameter, string>(recurrent_param, "recurrent_param"));

            // Alpha layers.

            // Beta layers.
            rgParam.Add(new KeyValuePair<BaseParameter, string>(attention_param, "attention_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(convolution_octave_param, "convolution_octave_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(data_sequence_param, "data_sequence_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(decode_param, "decode_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(gather_param, "gather_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(interp_param, "interp_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(knn_param, "knn_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(lstm_attention_param, "lstm_attention_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(mean_error_loss_param, "mean_error_loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(merge_param, "merge_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(mish_param, "mish_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(normalization1_param, "normalization_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(serf_param, "serf_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(squeeze_param, "squeeze_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(model_data_param, "model_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(text_data_param, "text_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(triplet_loss_param, "triplet_loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(unpooling_param, "unpooling_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(transpose_param, "transpose_param"));

            // HDF5 layes.
            rgParam.Add(new KeyValuePair<BaseParameter, string>(hdf5_data_param, "hdf5_data_param"));

            // GPT Layers.
            rgParam.Add(new KeyValuePair<BaseParameter, string>(causal_self_attention_param, "causal_self_attention_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(multihead_attention_param, "multihead_attention_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(positional_encoder_param, "positional_encoder_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(gelu_param, "gelu_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(layer_norm_param, "layer_norm_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(transformer_block_param, "transformer_block_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(tokenized_data_param, "tokenized_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(tokenized_data_pairs_param, "tokenized_data_pairs_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(nll_loss_param, "nll_loss_param"));

            // TFT Layers
            rgParam.Add(new KeyValuePair<BaseParameter, string>(data_temporal_param, "data_temporal_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(categorical_trans_param, "categorical_trans_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(numeric_trans_param, "numeric_trans_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(gateaddnorm_param, "gateaddnorm_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(glu_param, "glu_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(grn_param, "grn_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(varselnet_param, "varselnet_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(multihead_attention_interp_param, "multihead_attention_interp_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(reshape_temporal_param, "reshape_temporal_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(quantile_loss_param, "quantile_loss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(quantile_accuracy_param, "quantile_accuracy_param"));

            // Nt layers.
            rgParam.Add(new KeyValuePair<BaseParameter, string>(gram_param, "gram_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(onehot_param, "onehot_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(scalar_param, "scalar_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(tv_loss_param, "tv_loss_param"));

            // Ssd layers.
            rgParam.Add(new KeyValuePair<BaseParameter, string>(annotated_data_param, "annotated_data_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(detection_evaluate_param, "detection_evaluate_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(detection_output_param, "detection_output_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(multiboxloss_param, "multiboxloss_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(normalization2_param, "normalization2_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(permute_param, "permute_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(prior_box_param, "prior_box_param"));
            rgParam.Add(new KeyValuePair<BaseParameter, string>(video_data_param, "video_data_param"));

            foreach (KeyValuePair<BaseParameter, string> kv in rgParam)
            {
                if (kv.Key != null)
                    rgChildren.Add(kv.Key.ToProto(kv.Value));
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LayerParameter FromProto(RawProto rp)
        {
            string strVal;
            string strName = null;
            LayerType layerType;

            if ((strVal = rp.FindValue("name")) != null)
                strName = strVal;

            if ((strVal = rp.FindValue("type")) == null)
                throw new Exception("No layer type specified!");

            layerType = parseLayerType(strVal);

            LayerParameter p = new LayerParameter(layerType, strName);

            p.bottom = rp.FindArray<string>("bottom");
            for (int i = 0; i < p.bottom.Count; i++)
            {
                p.bottom[i] = p.bottom[i].Trim('\"', ' ');
            }
            p.top = rp.FindArray<string>("top");
            for (int i = 0; i < p.top.Count; i++)
            {
                p.top[i] = p.top[i].Trim('\"', ' ');
            }

            if ((strVal = rp.FindValue("phase")) != null)
                p.phase = parsePhase(strVal);

            p.loss_weight = rp.FindArray<double>("loss_weight");

            if ((strVal = rp.FindValue("group_start")) != null)
                p.group_start = bool.Parse(strVal);

            if ((strVal = rp.FindValue("freeze_learning")) != null)
                p.freeze_learning = bool.Parse(strVal);

            if ((strVal = rp.FindValue("use_halfsize")) != null)
                p.use_halfsize = bool.Parse(strVal);

            RawProtoCollection rgrp;

            rgrp = rp.FindChildren("param");
            foreach (RawProto rpChild in rgrp)
            {
                p.parameters.Add(ParamSpec.FromProto(rpChild));
            }

            rgrp = rp.FindChildren("blobs");
            foreach (RawProto rpChild in rgrp)
            {
                p.blobs.Add(BlobProto.FromProto(rpChild));
            }

            p.propagate_down = rp.FindArray<bool>("propagate_down");

            rgrp = rp.FindChildren("include");
            foreach (RawProto rpChild in rgrp)
            {
                p.include.Add(NetStateRule.FromProto(rpChild));
            }

            rgrp = rp.FindChildren("exclude");
            foreach (RawProto rpChild in rgrp)
            {
                p.exclude.Add(NetStateRule.FromProto(rpChild));
            }

            rgrp = rp.FindChildren("max_bottom_count");
            foreach (RawProto rpChild in rgrp)
            {
                RawProto prPhase = rpChild.FindChild("phase");
                if (prPhase != null)
                {
                    Phase phase = parsePhase(prPhase.Value);
                    if (!p.m_rgMaxBottomCount.ContainsKey(phase))
                    {
                        RawProto prCount = rpChild.FindChild("count");
                        if (prCount != null)
                            p.m_rgMaxBottomCount.Add(phase, int.Parse(prCount.Value));
                    }
                }
            }

            RawProto rpp;

            // Standard layers
            if ((rpp = rp.FindChild("transform_param")) != null)
                p.transform_param = TransformationParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("loss_param")) != null)
                p.loss_param = LossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("accuracy_param")) != null)
                p.accuracy_param = AccuracyParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("argmax_param")) != null)
                p.argmax_param = ArgMaxParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("batch_norm_param")) != null)
                p.batch_norm_param = BatchNormParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("bias_param")) != null)
                p.bias_param = BiasParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("clip_param")) != null)
                p.clip_param = ClipParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("concat_param")) != null)
                p.concat_param = ConcatParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("constant_param")) != null)
                p.constant_param = ConstantParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("contrastive_loss_param")) != null)
                p.contrastive_loss_param = ContrastiveLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("convolution_param")) != null)
                p.convolution_param = ConvolutionParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("convolution_octave_param")) != null)
                p.convolution_octave_param = ConvolutionOctaveParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("crop_param")) != null)
                p.crop_param = CropParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("data_param")) != null)
                p.data_param = DataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("debug_param")) != null)
                p.debug_param = DebugParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("dropout_param")) != null)
                p.dropout_param = DropoutParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("dummy_data_param")) != null)
                p.dummy_data_param = DummyDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("eltwise_param")) != null)
                p.eltwise_param = EltwiseParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("elu_param")) != null)
                p.elu_param = EluParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("embed_param")) != null)
                p.embed_param = EmbedParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("exp_param")) != null)
                p.exp_param = ExpParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("flatten_param")) != null)
                p.flatten_param = FlattenParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("gradient_scale_param")) != null)
                p.gradient_scale_param = GradientScaleParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("hinge_loss_param")) != null)
                p.hinge_loss_param = HingeLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("image_data_param")) != null)
                p.image_data_param = ImageDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("infogain_loss_param")) != null)
                p.infogain_loss_param = InfogainLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("inner_product_param")) != null)
                p.inner_product_param = InnerProductParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("input_param")) != null)
                p.input_param = InputParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("labelmapping_param")) != null)
                p.labelmapping_param = LabelMappingParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("log_param")) != null)
                p.log_param = LogParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("lrn_param")) != null)
                p.lrn_param = LRNParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("mean_error_loss_param")) != null)
                p.mean_error_loss_param = MeanErrorLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("memory_data_param")) != null)
                p.memory_data_param = MemoryDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("mvn_param")) != null)
                p.mvn_param = MVNParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("pooling_param")) != null)
                p.pooling_param = PoolingParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("parameter_param")) != null)
                p.parameter_param = ParameterParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("power_param")) != null)
                p.power_param = PowerParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("prelu_param")) != null)
                p.prelu_param = PReLUParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("reduction_param")) != null)
                p.reduction_param = ReductionParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("relu_param")) != null)
                p.relu_param = ReLUParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("reshape_param")) != null)
                p.reshape_param = ReshapeParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("scale_param")) != null)
                p.scale_param = ScaleParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("sigmoid_param")) != null)
                p.sigmoid_param = SigmoidParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("softmax_param")) != null)
                p.softmax_param = SoftmaxParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("spp_param")) != null)
                p.spp_param = SPPParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("slice_param")) != null)
                p.slice_param = SliceParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("swish_param")) != null)
                p.swish_param = SwishParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("tanh_param")) != null)
                p.tanh_param = TanhParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("threshold_param")) != null)
                p.threshold_param = ThresholdParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("tile_param")) != null)
                p.tile_param = TileParameter.FromProto(rpp);

            // DEPRECIATED
            if ((rpp = rp.FindChild("lstm_simple_param")) != null)
                p.lstm_simple_param = LSTMSimpleParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("recurrent_param")) != null)
                p.recurrent_param = RecurrentParameter.FromProto(rpp);

            // Alpha layers

            // Beta layers.
            if ((rpp = rp.FindChild("attention_param")) != null)
                p.attention_param = AttentionParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("data_sequence_param")) != null)
                p.data_sequence_param = DataSequenceParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("decode_param")) != null)
                p.decode_param = DecodeParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("gather_param")) != null)
                p.gather_param = GatherParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("interp_param")) != null)
                p.interp_param = InterpParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("knn_param")) != null)
                p.knn_param = KnnParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("lstm_attention_param")) != null)
                p.lstm_attention_param = LSTMAttentionParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("merge_param")) != null)
                p.merge_param = MergeParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("mish_param")) != null)
                p.mish_param = MishParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("normalization_param")) != null)
                p.normalization1_param = Normalization1Parameter.FromProto(rpp);

            if ((rpp = rp.FindChild("serf_param")) != null)
                p.serf_param = SerfParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("squeeze_param")) != null)
                p.squeeze_param = SqueezeParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("model_data_param")) != null)
                p.model_data_param = ModelDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("text_data_param")) != null)
                p.text_data_param = TextDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("triplet_loss_param")) != null)
                p.triplet_loss_param = TripletLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("transpose_param")) != null)
                p.transpose_param = TransposeParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("unpooling_param")) != null)
                p.unpooling_param = UnPoolingParameter.FromProto(rpp);

            // HDF5 layers.
            if ((rpp = rp.FindChild("hdf5_data_param")) != null)
                p.hdf5_data_param = HDF5DataParameter.FromProto(rpp);

            // GPT layers.
            if ((rpp = rp.FindChild("causal_self_attention_param")) != null)
                p.causal_self_attention_param = CausalSelfAttentionParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("multihead_attention_param")) != null)
                p.multihead_attention_param = MultiheadAttentionParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("positional_encoder_param")) != null)
                p.positional_encoder_param = PositionalEncoderParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("gelu_param")) != null)
                p.gelu_param = GeluParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("layer_norm_param")) != null)
                p.layer_norm_param = LayerNormParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("transformer_block_param")) != null)
                p.transformer_block_param = TransformerBlockParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("tokenized_data_param")) != null)
                p.tokenized_data_param = TokenizedDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("tokenized_data_pairs_param")) != null)
                p.tokenized_data_pairs_param = TokenizedDataPairsParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("nll_loss_param")) != null)
                p.nll_loss_param = NLLLossParameter.FromProto(rpp);

            // TFT layers.
            if ((rpp = rp.FindChild("data_temporal_param")) != null)
                p.data_temporal_param = DataTemporalParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("categorical_trans_param")) != null)
                p.categorical_trans_param = CategoricalTransformationParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("numeric_trans_param")) != null)
                p.numeric_trans_param = NumericTransformationParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("gateaddnorm_param")) != null)
                p.gateaddnorm_param = GateAddNormParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("glu_param")) != null)
                p.glu_param = GluParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("grn_param")) != null)
                p.grn_param = GrnParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("varselnet_param")) != null)
                p.varselnet_param = VarSelNetParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("multihead_attention_interp_param")) != null)
                p.multihead_attention_interp_param = MultiHeadAttentionInterpParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("reshape_temporal_param")) != null)
                p.reshape_temporal_param = ReshapeTemporalParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("quantile_loss_param")) != null)
                p.quantile_loss_param = QuantileLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("quantile_accuracy_param")) != null)
                p.quantile_accuracy_param = QuantileAccuracyParameter.FromProto(rpp);

            // Nt layers.
            if ((rpp = rp.FindChild("gram_param")) != null)
                p.gram_param = GramParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("onehot_param")) != null)
                p.onehot_param = OneHotParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("scalar_param")) != null)
                p.scalar_param = ScalarParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("tv_loss_param")) != null)
                p.tv_loss_param = TVLossParameter.FromProto(rpp);

            // Ssd layers.
            if ((rpp = rp.FindChild("annotated_data_param")) != null)
                p.annotated_data_param = AnnotatedDataParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("detection_evaluate_param")) != null)
                p.detection_evaluate_param = DetectionEvaluateParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("detection_output_param")) != null)
                p.detection_output_param = DetectionOutputParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("multiboxloss_param")) != null)
                p.multiboxloss_param = MultiBoxLossParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("normalization2_param")) != null)
                p.normalization2_param = Normalization2Parameter.FromProto(rpp);

            if ((rpp = rp.FindChild("permute_param")) != null)
                p.permute_param = PermuteParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("prior_box_param")) != null)
                p.prior_box_param = PriorBoxParameter.FromProto(rpp);

            if ((rpp = rp.FindChild("video_data_param")) != null)
                p.video_data_param = VideoDataParameter.FromProto(rpp);

            return p;
        }

        private static Phase parsePhase(string strVal)
        {
            switch (strVal)
            {
                case "TEST":
                    return Phase.TEST;

                case "TRAIN":
                    return Phase.TRAIN;

                case "RUN":
                    return Phase.RUN;

                case "NONE":
                    return Phase.NONE;

                default:
                    throw new Exception("Unknown 'phase' value: " + strVal);
            }
        }

        /// <summary>
        /// Converts the string type into a LayerType, or <i>null</i> if no match is found.
        /// </summary>
        /// <param name="strType">Specifies the layer type.</param>
        /// <returns>The LayerType is returned, or <i>null</i> if not found.</returns>
        public static LayerType? GetType(string strType)
        {
            try
            {
                return parseLayerType(strType);
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static LayerType parseLayerType(string str)
        {
            str = str.ToLower();

            switch (str)
            {
                case "absval":
                    return LayerType.ABSVAL;

                case "accuracy":
                    return LayerType.ACCURACY;

                case "accuracydecode":
                case "accuracy_decode":
                    return LayerType.ACCURACY_DECODE;

                case "accuracyencoding":
                case "accuracy_encoding":
                    return LayerType.ACCURACY_ENCODING;

                case "argmax":
                    return LayerType.ARGMAX;

                case "annotateddata":
                    return LayerType.ANNOTATED_DATA;

                case "attention":
                    return LayerType.ATTENTION;

                case "batchnorm":
                    return LayerType.BATCHNORM;

                case "batchreindex":
                    return LayerType.BATCHREINDEX;

                case "bias":
                    return LayerType.BIAS;

                case "bnll":
                    return LayerType.BNLL;

                case "categoricaltrans":
                case "categorical_trans":
                    return LayerType.CATEGORICAL_TRANS;

                case "clip":
                    return LayerType.CLIP;

                case "causalselfattention":
                    return LayerType.CAUSAL_SELF_ATTENTION;

                case "channelembedding":
                    return LayerType.CHANNEL_EMBEDDING;

                case "concat":
                    return LayerType.CONCAT;

                case "constant":
                    return LayerType.CONSTANT;

                case "contrastiveloss":
                case "contrastive_loss":
                    return LayerType.CONTRASTIVE_LOSS;

                case "convolution":
                    return LayerType.CONVOLUTION;

                case "convolutionoctave":
                case "convolution_octave":
                    return LayerType.CONVOLUTION_OCTAVE;

                case "crop":
                    return LayerType.CROP;

                case "copy":
                    return LayerType.COPY;

                case "decode":
                    return LayerType.DECODE;

                case "data":
                    return LayerType.DATA;

                case "datanormalizer":
                case "data_normalizer":
                    return LayerType.DATA_NORMALIZER;

                case "datasequence":
                case "data_sequence":
                    return LayerType.DATA_SEQUENCE;

                case "datatemporal":
                case "data_temporal":
                    return LayerType.DATA_TEMPORAL;

                case "debug":
                    return LayerType.DEBUG;

                case "deconvolution":
                    return LayerType.DECONVOLUTION;

                case "detectionevaluate":
                case "detection_evaluate":
                    return LayerType.DETECTION_EVALUATE;

                case "detectionoutput":
                case "detection_output":
                    return LayerType.DETECTION_OUTPUT;

                case "dropout":
                    return LayerType.DROPOUT;

                case "dummydata":
                    return LayerType.DUMMYDATA;

                case "eltwise":
                    return LayerType.ELTWISE;

                case "elu":
                    return LayerType.ELU;

                case "embed":
                    return LayerType.EMBED;

                case "euclideanloss":
                case "euclidean_loss":
                    return LayerType.EUCLIDEAN_LOSS;

                case "event":
                    return LayerType.EVENT;

                case "exp":
                    return LayerType.EXP;

                case "filter":
                    return LayerType.FILTER;

                case "flatten":
                    return LayerType.FLATTEN;

                case "gather":
                    return LayerType.GATHER;

                case "gateaddnorm":
                    return LayerType.GATEADDNORM;

                case "gelu":
                    return LayerType.GELU;

                case "glu":
                    return LayerType.GLU;

                case "grn":
                    return LayerType.GRN;

                case "globresnet":
                    return LayerType.GLOBRES_NORM;

                case "gsl":
                    return LayerType.GRADIENTSCALER;

                case "gram":
                    return LayerType.GRAM;

                case "hdf5data":
                    return LayerType.HDF5_DATA;

//                case "hdf5output":
//                    return LayerType.HDF5OUTPUT;

                case "hingeloss":
                case "hinge_loss":
                    return LayerType.HINGE_LOSS;

                case "im2col":
                    return LayerType.IM2COL;

                case "imagedata":
                    return LayerType.IMAGE_DATA;

                case "infogainloss":
                case "infogain_loss":
                    return LayerType.INFOGAIN_LOSS;

                case "innerproduct":
                case "inner_product":
                    return LayerType.INNERPRODUCT;

                case "input":
                    return LayerType.INPUT;

                case "interp":
                    return LayerType.INTERP;

                case "knn":
                    return LayerType.KNN;

                case "labelmapping":
                    return LayerType.LABELMAPPING;

                case "layernorm":
                    return LayerType.LAYERNORM;

                case "log":
                    return LayerType.LOG;

                case "lrn":
                    return LayerType.LRN;

                case "mean_error_loss":
                case "meanerrorloss":
                    return LayerType.MEAN_ERROR_LOSS;

                case "math":
                    return LayerType.MATH;

                case "merge":
                    return LayerType.MERGE;

                case "memorydata":
                    return LayerType.MEMORYDATA;

                case "multiboxloss":
                case "multibox_loss":
                    return LayerType.MULTIBOX_LOSS;

                case "multiheadattention":
                    return LayerType.MULTIHEAD_ATTENTION;

                case "multiheadattentioninterp":
                    return LayerType.MULTIHEAD_ATTENTION_INTERP;

                case "memoryloss":
                case "memory_loss":
                    return LayerType.MEMORY_LOSS;

                case "mish":
                    return LayerType.MISH;

                case "multinomiallogisticloss":
                case "multinomiallogistic_loss":
                    return LayerType.MULTINOMIALLOGISTIC_LOSS;

                case "mvn":
                    return LayerType.MVN;

                case "nllloss":
                case "nll_loss":
                    return LayerType.NLL_LOSS;

                case "numerictrans":
                case "numeric_trans":
                    return LayerType.NUMERIC_TRANS;

                case "onehot":
                    return LayerType.ONEHOT;

                case "normalization1":
                    return LayerType.NORMALIZATION1;

                case "normalize":
                case "normalization2":
                    return LayerType.NORMALIZATION2;

                case "parameter":
                    return LayerType.PARAMETER;

                case "permute":
                    return LayerType.PERMUTE;

                case "positionalencoder":
                    return LayerType.POSITIONAL_ENCODER;

                case "pooling":
                    return LayerType.POOLING;

                case "unpooling1":
                    return LayerType.UNPOOLING1;

                case "unpooling":
                    return LayerType.UNPOOLING;

                case "power":
                    return LayerType.POWER;

                case "prelu":
                    return LayerType.PRELU;

                case "priorbox":
                    return LayerType.PRIORBOX;

                case "quantileaccuracy":
                case "quantile_accuracy":
                    return LayerType.QUANTILE_ACCURACY;

                case "quantileloss":
                case "quantile_loss":
                    return LayerType.QUANTILE_LOSS;

                case "reduction":
                    return LayerType.REDUCTION;

                case "relu":
                    return LayerType.RELU;

                case "reshape":
                    return LayerType.RESHAPE;

                case "reshapetemporal":
                    return LayerType.RESHAPE_TEMPORAL;

                case "squeeze":
                    return LayerType.SQUEEZE;

                case "unsqueeze":
                    return LayerType.UNSQUEEZE;

                case "scalar":
                    return LayerType.SCALAR;

                case "scale":
                    return LayerType.SCALE;

                case "serf":
                    return LayerType.SERF;

                case "sigmoid":
                    return LayerType.SIGMOID;

                case "sigmoidcrossentropyloss":
                case "sigmoidcrossentropy_loss":
                    return LayerType.SIGMOIDCROSSENTROPY_LOSS;

                case "softmaxcrossentropyloss":
                case "softmaxcrossentropy_loss":
                    return LayerType.SOFTMAXCROSSENTROPY_LOSS;

                case "softmaxcrossentropy2loss":
                case "softmaxcrossentropy2_loss":
                    return LayerType.SOFTMAXCROSSENTROPY2_LOSS;

                case "silence":
                    return LayerType.SILENCE;

                case "slice":
                    return LayerType.SLICE;

                case "softmax":
                    return LayerType.SOFTMAX;

                case "softmaxwithloss":
                case "softmaxwith_loss":
                case "softmax_loss":
                    return LayerType.SOFTMAXWITH_LOSS;

                case "smoothl1loss":
                case "smoothl1_loss":
                    return LayerType.SMOOTHL1_LOSS;

                case "split":
                    return LayerType.SPLIT;

                case "spp":
                    return LayerType.SPP;

                case "swish":
                    return LayerType.SWISH;

                case "tanh":
                    return LayerType.TANH;

                case "modeldata":
                case "model_data":
                    return LayerType.MODEL_DATA;

                case "textdata":
                case "text_data":
                    return LayerType.TEXT_DATA;

                case "threshold":
                    return LayerType.THRESHOLD;

                case "tile":
                    return LayerType.TILE;

                case "transpose":
                    return LayerType.TRANSPOSE;

                case "transformerblock":
                    return LayerType.TRANSFORMER_BLOCK;

                case "tokenizeddata":
                    return LayerType.TOKENIZED_DATA;

                case "tokenizeddatapairs":
                    return LayerType.TOKENIZED_DATA_PAIRS;

                case "tokenizeddatapairs_py":
                case "tokenizeddatapairspy":
                    return LayerType.TOKENIZED_DATA_PAIRS_PY;

                case "triplet_loss":
                case "tripletloss":
                    return LayerType.TRIPLET_LOSS;

                case "tvloss":
                case "tv_loss":
                    return LayerType.TV_LOSS;

                // case "windowdata":
                //      return LayerType.WINDOWDATA;

                // DEPRECIATED
                case "lstmsimple":
                case "lstm_simple":
                    return LayerType.LSTM_SIMPLE;

                case "lstmattention":
                case "lstm_attention":
                    return LayerType.LSTM_ATTENTION;

                case "rnn":
                    return LayerType.RNN;

                case "lstm":
                    return LayerType.LSTM;

                case "lstm_unit":
                    return LayerType.LSTM_UNIT;

                case "videodata":
                case "video_data":
                    return LayerType.VIDEO_DATA;

                case "varselnet":
                    return LayerType.VARSELNET;

                default:
                    throw new Exception("Unknown 'layertype' value: " + str);
            }
        }

        /// <summary>
        /// Returns a string representation of the LayerParameter.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string strOut = ((use_halfsize) ? "HALF " : "FULL ");

            strOut += m_strName + " (" + m_type.ToString() + ")";
            strOut += " btm = " + Utility.ToString(m_rgstrBottom);
            strOut += " top = " + Utility.ToString(m_rgstrTop);

            return strOut;
        }
    }
}
