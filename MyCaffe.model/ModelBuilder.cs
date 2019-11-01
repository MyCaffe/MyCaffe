using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.param.ssd;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.model namespace contains all classes used to programically create new model scripts.
/// </summary>
namespace MyCaffe.model
{
    /// <summary>
    /// The ModelBuilder is an abstract class that is overridden by a base class used to programically build new models.
    /// </summary>
    public abstract class ModelBuilder
    {
        /// <summary>
        /// Specifies the base directory that contains the data and models.
        /// </summary>
        protected string m_strBaseDir;
        /// <summary>
        /// Specifies the base net to be altered.
        /// </summary>
        protected NetParameter m_net = new NetParameter();
        /// <summary>
        /// Specifies the base solver to use.
        /// </summary>
        protected SolverParameter m_solver = new SolverParameter();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strBaseDir">Specifies the base directory that contains the data and models.</param>
        /// <param name="net">Optionally, specifies the 'base' net parameter that is to be altered (default = null).</param>
        /// <param name="solver">Optionally, specifies the 'base' solver parameter to use (default = null).</param>
        public ModelBuilder(string strBaseDir, NetParameter net = null, SolverParameter solver = null)
        {
            m_strBaseDir = strBaseDir.TrimEnd('\\', '/');
            m_net = net;
            m_solver = solver;
        }

        /// <summary>
        /// Create the base solver to use.
        /// </summary>
        /// <returns>
        /// The solver parameter created is returned.
        /// </returns>
        public abstract SolverParameter CreateSolver();

        /// <summary>
        /// Create the training/testing/deploy model to use.
        /// </summary>
        public abstract NetParameter CreateModel(bool bDeploy = false);

        /// <summary>
        /// Create the deploy model to use.
        /// </summary>
        public abstract NetParameter CreateDeployModel();

        /// <summary>
        /// Returns the full path of the filename using the base directory original set when creating the ModelBuilder.
        /// </summary>
        /// <param name="strFile">Specifies the partial path of the file.</param>
        /// <param name="strSubDir">Specifies the sub-directory off the path (or null for none).</param>
        /// <returns>The full path of the file is returned.</returns>
        protected string getFileName(string strFile, string strSubDir)
        {
            string strOut = m_strBaseDir + "\\";

            if (!string.IsNullOrEmpty(strSubDir))
                strOut += strSubDir + "\\";

            strOut += strFile;

            return strOut;
        }

        /// <summary>
        /// Add extra layers on top of a 'base' network (e.g. VGGNet or Inception)
        /// </summary>
        /// <param name="bUseBatchNorm">Optionally, specifies whether or not to use batch normalization layers (default = <i>true</i>).</param>
        /// <param name="dfLrMult">Optionally, specifies the learning rate multiplier (default = 1.0).</param>
        protected abstract LayerParameter addExtraLayers(bool bUseBatchNorm = true, double dfLrMult = 1.0);

        /// <summary>
        /// Find a layer with a given name.
        /// </summary>
        /// <param name="strName">Specifies the name of the layer to find.</param>
        /// <returns>The layer parameter of the layer with the specified name is returned.</returns>
        protected LayerParameter findLayer(string strName)
        {
            foreach (LayerParameter p in m_net.layer)
            {
                if (p.name == strName)
                    return p;
            }

            return null;
        }

        /// <summary>
        /// Create the base network parameter for the model and set its name to the 'm_strModel' name.
        /// </summary>
        /// <param name="strName">Specifies the model name.</param>
        /// <returns>The NetParameter created is returned.</returns>
        protected NetParameter createNet(string strName)
        {
            NetParameter net = new NetParameter();

            net.name = strName;

            return net;
        }

        /// <summary>
        /// Add the Annotated Data layer.
        /// </summary>
        /// <param name="strSource">Specifies the data source.</param>
        /// <param name="phase">Specifies the phase under which to run the layer (e.g. TRAIN, TEST, RUN).</param>
        /// <param name="nBatchSize">Optionally, specifies the batch size (default = 32).</param>
        /// <param name="bOutputLabel">Optionally, specifies whether or not to output the label (default = true).</param>
        /// <param name="strLabelMapFile">Optionally, specifies the label file (default = "").</param>
        /// <param name="anno_type">Optionally, specifies the annotation type (default = NONE).</param>
        /// <param name="transform">Optionally, specifies the transformation parameter (default = null, ignored).</param>
        /// <param name="rgSampler">Optionally, specifies the list of batch samplers (default = null, ignored).</param>
        /// <returns>The annotated data layer is retunred after it is added to the network.</returns>
        protected LayerParameter addAnnotatedDataLayer(string strSource, Phase phase, int nBatchSize = 32, bool bOutputLabel = true, string strLabelMapFile = "", SimpleDatum.ANNOTATION_TYPE anno_type = SimpleDatum.ANNOTATION_TYPE.NONE, TransformationParameter transform = null, List<BatchSampler> rgSampler = null)
        {
            LayerParameter data = new LayerParameter(LayerParameter.LayerType.ANNOTATED_DATA);

            data.include.Add(new NetStateRule(phase));

            if (transform != null)
                data.transform_param = transform;

            data.annotated_data_param.label_map_file = strLabelMapFile;

            if (rgSampler != null)
                data.annotated_data_param.batch_sampler = rgSampler;

            data.annotated_data_param.anno_type = anno_type;
            data.name = "data";
            data.data_param.batch_size = (uint)nBatchSize;
            data.data_param.source = strSource;

            data.top.Clear();
            data.top.Add("data");

            if (bOutputLabel)
                data.top.Add("label");

            m_net.layer.Add(data);

            return data;
        }

        /// <summary>
        /// Create the multi-box head layers.
        /// </summary>
        /// <param name="data">Specifies the data layer.</param>
        /// <param name="nNumClasses">Specifies the number of classes.</param>
        /// <param name="rgInfo">Specifies the info associated with the layers to connect to.</param>
        /// <param name="rgPriorVariance">Specifies the prior variance.</param>
        /// <param name="bUseObjectness">Optionally, specifies whether or not to use objectness (default = false).</param>
        /// <param name="bUseBatchNorm">Optionally, specifies whether or not to use batch-norm layers (default = true).</param>
        /// <param name="dfLrMult">Optionally, specifies the learning multiplier (default = 1.0).</param>
        /// <param name="bUseScale">Optionally, specifies whether or not to use scale layers (default = true).</param>
        /// <param name="nImageHt">Optionally, specifies the image height (default = 0, ignore).</param>
        /// <param name="nImageWd">Optionally, specifies the image width (default = 0, ignore).</param>
        /// <param name="bShareLocation">Optionally, specifies whether or not to share the location (default = true).</param>
        /// <param name="bFlip">Optionally, specifies whether or not to flip (default = true).</param>
        /// <param name="bClip">Optionally, specifies whether or not to clip (default = true).</param>
        /// <param name="dfOffset">Optionally, specifies the offset (default = 0.5).</param>
        /// <param name="nKernelSize">Optionally, specifies the kernel size (default = 1).</param>
        /// <param name="nPad">Optionally, specifies the pad (default = 0).</param>
        /// <param name="strConfPostfix">Optionally, specifies the confidence postfix (default = "").</param>
        /// <param name="strLocPostfix">Optionally, specifies the location postifix (default = "").</param>
        /// <returns></returns>
        protected List<LayerParameter> createMultiBoxHead(LayerParameter data, int nNumClasses, List<MultiBoxHeadInfo> rgInfo, List<float> rgPriorVariance, bool bUseObjectness = false, bool bUseBatchNorm = true, double dfLrMult = 1.0, bool bUseScale = true, int nImageHt = 0, int nImageWd = 0, bool bShareLocation = true, bool bFlip = true, bool bClip = true, double dfOffset = 0.5, int nKernelSize = 1, int nPad = 0, string strConfPostfix = "", string strLocPostfix = "")
        {
            LayerParameter lastLayer;
            string strName;

            for (int i = 1; i < rgInfo.Count; i++)
            {
                if (!rgInfo[0].Verify(rgInfo[1]))
                    throw new Exception("The multi-bix header info must be consistent across all items.");
            }

            if (nNumClasses <= 0)
                throw new Exception("The number of classes must be > 0.");

            List<string> rgstrLocLayers = new List<string>();
            List<string> rgstrConfLayers = new List<string>();
            List<string> rgstrPriorBoxLayers = new List<string>();
            List<string> rgstrObjLayers = new List<string>();

            for (int i = 0; i < rgInfo.Count; i++)
            {
                LayerParameter fromLayer = findLayer(rgInfo[i].SourceLayer);

                //---------------------------------------------------
                // Get the normalize value.
                //---------------------------------------------------
                if (rgInfo[i].Normalization.HasValue && rgInfo[i].Normalization.Value != -1)
                {
                    LayerParameter norm = new LayerParameter(LayerParameter.LayerType.NORMALIZATION2);
                    norm.name = fromLayer.name + "_norm";
                    norm.normalization2_param.scale_filler = new FillerParameter("constant", rgInfo[i].Normalization.Value);
                    norm.normalization2_param.across_spatial = false;
                    norm.normalization2_param.channel_shared = false;
                    norm.top.Add(norm.name);
                    fromLayer = connectAndAddLayer(fromLayer, norm);
                }

                //---------------------------------------------------
                // Intermediate layers.
                //---------------------------------------------------
                if (rgInfo[i].InterLayerDepth.HasValue && rgInfo[i].InterLayerDepth.Value > 0)
                    fromLayer = addConvBNLayer(fromLayer.name, fromLayer.name + "_inter", bUseBatchNorm, true, (int)rgInfo[i].InterLayerDepth.Value, 3, 1, 1, dfLrMult);

                //---------------------------------------------------
                // Estimate number of priors per location given provided parameters.
                //---------------------------------------------------
                double? dfMinSize = rgInfo[i].MinSize;
                double? dfMaxSize = rgInfo[i].MaxSize;
                double? dfAspectHt = rgInfo[i].AspectRatioHeight;
                double? dfAspectWd = rgInfo[i].AspectRatioWidth;
                double? dfStepWd = rgInfo[i].StepWidth;
                double? dfStepHt = rgInfo[i].StepHeight;
                int nAspectLen = (dfAspectWd == dfAspectHt) ? 1 : 2;
                int nNumPriorsPerLocation = (dfMaxSize.HasValue) ? (2 + nAspectLen) : (1 + nAspectLen);

                if (bFlip)
                    nNumPriorsPerLocation += nAspectLen;

                //---------------------------------------------------
                // Create location prediction layer.
                //---------------------------------------------------
                int nNumLocOutput = nNumPriorsPerLocation * 4;
                if (!bShareLocation)
                    nNumLocOutput *= nNumClasses;

                strName = fromLayer.name + "_mbox_loc" + strLocPostfix;
                lastLayer = addConvBNLayer(fromLayer.name, strName, bUseBatchNorm, false, nNumLocOutput, nKernelSize, nPad, 1, dfLrMult);

                LayerParameter permute = new LayerParameter(LayerParameter.LayerType.PERMUTE);
                permute.name = strName + "_perm";
                permute.permute_param.order = new List<int>() { 0, 2, 3, 1 };
                permute.top.Add(permute.name);
                lastLayer = connectAndAddLayer(lastLayer, permute);

                LayerParameter flatten = new LayerParameter(LayerParameter.LayerType.FLATTEN);
                flatten.name = strName + "_flat";
                flatten.flatten_param.axis = 1;
                flatten.top.Add(flatten.name);
                lastLayer = connectAndAddLayer(lastLayer, flatten);
                rgstrLocLayers.Add(lastLayer.name);

                //---------------------------------------------------
                // Create confidence prediction layer.
                //---------------------------------------------------
                strName = fromLayer.name + "_mbox_conf" + strConfPostfix;
                int nNumConfOutput = nNumPriorsPerLocation * nNumClasses;
                lastLayer = addConvBNLayer(fromLayer.name, strName, bUseBatchNorm, false, nNumConfOutput, nKernelSize, nPad, 1, dfLrMult);

                permute = new LayerParameter(LayerParameter.LayerType.PERMUTE);
                permute.name = strName + "_perm";
                permute.permute_param.order = new List<int>() { 0, 2, 3, 1 };
                permute.top.Add(permute.name);
                lastLayer = connectAndAddLayer(lastLayer, permute);

                flatten = new LayerParameter(LayerParameter.LayerType.FLATTEN);
                flatten.name = strName + "_flat";
                flatten.flatten_param.axis = 1;
                flatten.top.Add(flatten.name);
                lastLayer = connectAndAddLayer(lastLayer, flatten);
                rgstrConfLayers.Add(lastLayer.name);

                //---------------------------------------------------
                // Create prior generation layer.
                //---------------------------------------------------
                strName = fromLayer.name + "_mbox_priorbox";
                LayerParameter priorbox = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
                priorbox.name = strName;
                priorbox.top.Add(priorbox.name);
                priorbox.prior_box_param.min_size.Add((float)dfMinSize.Value);
                priorbox.prior_box_param.clip = bClip;
                priorbox.prior_box_param.variance = rgPriorVariance;
                priorbox.prior_box_param.offset = (float)dfOffset;

                if (dfMaxSize.HasValue)
                    priorbox.prior_box_param.max_size.Add((float)dfMaxSize.Value);

                if (dfAspectWd.HasValue)
                    priorbox.prior_box_param.aspect_ratio.Add((float)dfAspectWd.Value);

                if (dfAspectHt.HasValue)
                    priorbox.prior_box_param.aspect_ratio.Add((float)dfAspectHt.Value);

                if (dfStepWd.HasValue && dfStepHt.HasValue)
                {
                    if (dfStepWd.Value == dfStepHt.Value)
                    {
                        priorbox.prior_box_param.step = (float)dfStepWd.Value;
                    }
                    else
                    {
                        priorbox.prior_box_param.step_h = (float)dfStepHt.Value;
                        priorbox.prior_box_param.step_w = (float)dfStepWd.Value;
                    }
                }

                if (nImageHt != 0 && nImageWd != 0)
                {
                    if (nImageHt == nImageWd)
                    {
                        priorbox.prior_box_param.img_size = (uint)nImageHt;
                    }
                    else
                    {
                        priorbox.prior_box_param.img_h = (uint)nImageHt;
                        priorbox.prior_box_param.img_w = (uint)nImageWd;
                    }
                }

                lastLayer = connectAndAddLayer(fromLayer, priorbox);
                lastLayer.bottom.Add(data.top[0]);
                rgstrPriorBoxLayers.Add(lastLayer.name);

                //---------------------------------------------------
                // Create objectness prediction layer
                //---------------------------------------------------
                if (bUseObjectness)
                {
                    strName = fromLayer.name + "_mbox_objectness";
                    int nNumObjOutput = nNumPriorsPerLocation * 2;
                    lastLayer = addConvBNLayer(fromLayer.name, strName, bUseBatchNorm, false, nNumObjOutput, nKernelSize, nPad, 1, dfLrMult);

                    permute = new LayerParameter(LayerParameter.LayerType.PERMUTE);
                    permute.name = strName + "_perm";
                    permute.permute_param.order = new List<int>() { 0, 2, 3, 1 };
                    lastLayer = connectAndAddLayer(lastLayer, permute);

                    flatten = new LayerParameter(LayerParameter.LayerType.FLATTEN);
                    flatten.name = strName + "_flat";
                    flatten.flatten_param.axis = 1;
                    lastLayer = connectAndAddLayer(lastLayer, flatten);
                    rgstrObjLayers.Add(lastLayer.name);
                }
            }

            //---------------------------------------------------
            // Concatenate priorbox, loc, and conf layers.
            //---------------------------------------------------
            List<LayerParameter> rgMboxLayers = new List<LayerParameter>();
            strName = "mbox_loc";

            LayerParameter concat = new LayerParameter(LayerParameter.LayerType.CONCAT);
            concat.name = strName;
            concat.concat_param.axis = 1;
            concat.bottom = rgstrLocLayers;
            concat.top.Add(concat.name);
            m_net.layer.Add(concat);
            rgMboxLayers.Add(concat);

            strName = "mbox_conf";
            concat = new LayerParameter(LayerParameter.LayerType.CONCAT);
            concat.name = strName;
            concat.concat_param.axis = 1;
            concat.bottom = rgstrConfLayers;
            concat.top.Add(concat.name);
            m_net.layer.Add(concat);
            rgMboxLayers.Add(concat);

            strName = "mbox_priorbox";
            concat = new LayerParameter(LayerParameter.LayerType.CONCAT);
            concat.name = strName;
            concat.concat_param.axis = 2;
            concat.bottom = rgstrPriorBoxLayers;
            concat.top.Add(concat.name);
            m_net.layer.Add(concat);
            rgMboxLayers.Add(concat);

            if (bUseObjectness)
            {
                strName = "mbox_objectness";
                concat = new LayerParameter(LayerParameter.LayerType.CONCAT);
                concat.name = strName;
                concat.concat_param.axis = 1;
                concat.bottom = rgstrObjLayers;
                concat.top.Add(concat.name);
                m_net.layer.Add(concat);
                rgMboxLayers.Add(concat);
            }

            return rgMboxLayers;
        }

        /// <summary>
        /// Add convolution, batch-norm layers.
        /// </summary>
        /// <param name="strInputLayer">Specifies the input layer.</param>
        /// <param name="strOutputLayer">Specifies the output layer.</param>
        /// <param name="bUseBatchNorm">Optionally, specifies whether or not to use a batch-norm layer.</param>
        /// <param name="bUseRelU">Specifies whether or not to add a RelU layer.</param>
        /// <param name="nNumOutput">Specifies the number of output.</param>
        /// <param name="nKernelSize">Specifies the kernel size.</param>
        /// <param name="nPad">Specifies the pad.</param>
        /// <param name="nStride">Specifies the stride.</param>
        /// <param name="dfLrMult">Optionally, specifies the default learning rate multiplier (default = 1.0).</param>
        /// <param name="nDilation">Optionally, specifies the dilation (default = 1).</param>
        /// <param name="bUseScale">Optionally, specifies whether or not to use a ScaleLayer (default = true).</param>
        /// <param name="strConvPrefix">Optionally, specifies the convolution layer name prefix (default = "").</param>
        /// <param name="strConvPostfix">Optionally, specifies the convolution layer name postfix (default = "").</param>
        /// <param name="strBnPrefix">Optionally, specifies the batch-norm layer name prefix (default = "").</param>
        /// <param name="strBnPostfix">Optionally, specifies the batch-norm layer name postfix (default = "_bn").</param>
        /// <param name="strScalePrefix">Optionally, specifies the scale layer name prefix (default = "").</param>
        /// <param name="strScalePostFix">Optionally, specifies the scale layer name postfix (default = "_scale")</param>
        /// <param name="strBiasPrefix">Optionally, specifies the bias layer name prefix (default = "").</param>
        /// <param name="strBiasPostfix">Optionally, specifies the bias layer name postfix (default = "_bias").</param>
        /// <returns>The last layer added is returned.</returns>
        protected LayerParameter addConvBNLayer(string strInputLayer, string strOutputLayer, bool bUseBatchNorm, bool bUseRelU, int nNumOutput, int nKernelSize, int nPad, int nStride, double dfLrMult = 1.0, int nDilation = 1, bool bUseScale = true, string strConvPrefix = "", string strConvPostfix = "", string strBnPrefix = "", string strBnPostfix = "_bn", string strScalePrefix = "", string strScalePostFix = "_scale", string strBiasPrefix = "", string strBiasPostfix = "_bias")
        {
            LayerParameter lastLayer = findLayer(strInputLayer);

            LayerParameter convLayer = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            convLayer.convolution_param.weight_filler = new FillerParameter("xavier");
            convLayer.convolution_param.bias_filler = new FillerParameter("constant", 0);
            convLayer.convolution_param.bias_term = true;
            convLayer.name = strConvPrefix + strOutputLayer + strConvPostfix;
            convLayer.convolution_param.kernel_size.Add((uint)nKernelSize);
            convLayer.convolution_param.pad.Add((uint)nPad);
            convLayer.convolution_param.stride.Add((uint)nStride);
            convLayer.convolution_param.dilation.Add((uint)nDilation);
            convLayer.convolution_param.num_output = (uint)nNumOutput;
            convLayer.top.Add(convLayer.name);

            LayerParameter bnLayer = null;
            LayerParameter scaleLayer = null;
            LayerParameter biasLayer = null;
            LayerParameter reluLayer = null;

            // Setup the BachNorm Layer
            if (bUseBatchNorm)
            {
                convLayer.parameters.Add(new ParamSpec(dfLrMult, 1.0));
                convLayer.convolution_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.01);
                convLayer.convolution_param.bias_term = false;

                bnLayer = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
                bnLayer.name = strBnPrefix + strOutputLayer + strBnPostfix;
                bnLayer.batch_norm_param.eps = 0.001;
                bnLayer.batch_norm_param.moving_average_fraction = 0.999;
                bnLayer.batch_norm_param.use_global_stats = false;
                bnLayer.parameters.Add(new ParamSpec(0.0, 0.0));
                bnLayer.parameters.Add(new ParamSpec(0.0, 0.0));
                bnLayer.parameters.Add(new ParamSpec(0.0, 0.0));
                bnLayer.top.Add(bnLayer.name);

                double dfBnLrMult = dfLrMult;

                // When using global stats, not updating scale/bias parameters.
                if (bnLayer.batch_norm_param.use_global_stats.GetValueOrDefault(false))
                    dfBnLrMult = 0;

                // Parameters for scale bias layer after batchnorm.
                if (bUseScale)
                {
                    scaleLayer = new LayerParameter(LayerParameter.LayerType.SCALE);
                    scaleLayer.name = strScalePrefix + strOutputLayer + strScalePostFix;
                    scaleLayer.scale_param.bias_term = true;
                    scaleLayer.scale_param.filler = new FillerParameter("constant", 1.0);
                    scaleLayer.scale_param.bias_filler = new FillerParameter("constant", 0.0);
                    scaleLayer.parameters.Add(new ParamSpec(dfBnLrMult, 0.0));
                    scaleLayer.parameters.Add(new ParamSpec(dfBnLrMult, 0.0));
                    scaleLayer.top.Add(scaleLayer.name);
                }
                else
                {
                    biasLayer = new LayerParameter(LayerParameter.LayerType.BIAS);
                    biasLayer.name = strBiasPrefix + strOutputLayer + strBiasPostfix;
                    biasLayer.bias_param.filler = new FillerParameter("constant", 0.0);
                    biasLayer.parameters.Add(new ParamSpec(dfBnLrMult, 0.0));
                    biasLayer.top.Add(biasLayer.name);
                }
            }
            else
            {
                convLayer.parameters.Add(new ParamSpec(dfLrMult, 1.0));
                convLayer.parameters.Add(new ParamSpec(dfLrMult * 2, 0.0));
            }

            lastLayer = connectAndAddLayer(lastLayer, convLayer);

            if (bnLayer != null)
                lastLayer = connectAndAddLayer(lastLayer, bnLayer);

            if (scaleLayer != null)
                lastLayer = connectAndAddLayer(lastLayer, scaleLayer);

            if (biasLayer != null)
                lastLayer = connectAndAddLayer(lastLayer, biasLayer);

            if (bUseRelU)
            {
                reluLayer = new LayerParameter(LayerParameter.LayerType.RELU);
                reluLayer.name = convLayer.name + "_relu";
                lastLayer = connectAndAddLayer(lastLayer, reluLayer, true);
            }

            return lastLayer;
        }

        /// <summary>
        /// Connect the from layer to the 'to' layer.
        /// </summary>
        /// <param name="fromLayer">Specifies the layer who's bottom is connected to the toLayer's top.</param>
        /// <param name="toLayer">Specifies the layer who's top is connected to the from layer's bottom.</param>
        /// <param name="bInPlace">Optionally, specifies whether or not to connect both the top and bottom of the toLayer to the top of the from layer.</param>
        /// <param name="bAdd">Optionally, specifies whether or not to add the layer to the network (default = true).</param>
        /// <param name="nTopIdx">Optionally, specifies the top index of the item to connect.</param>
        /// <returns>The toLayer is returned as the next layer.</returns>
        protected LayerParameter connectAndAddLayer(LayerParameter fromLayer, LayerParameter toLayer, bool bInPlace = false, bool bAdd = true, int nTopIdx = 0)
        {
            toLayer.bottom.Clear();
            toLayer.bottom.Add(fromLayer.top[nTopIdx]);

            if (bAdd)
                m_net.layer.Add(toLayer);

            if (bInPlace)
            {
                toLayer.top.Clear();
                toLayer.top.Add(fromLayer.top[nTopIdx]);
            }

            return toLayer;
        }

        /// <summary>
        /// Connect the from layer to the 'to' layer.
        /// </summary>
        /// <param name="rgFromLayer">Specifies a list of layers who's bottoms are connected to the toLayer's top.</param>
        /// <param name="toLayer">Specifies the layer who's top is connected to the from layer's bottom.</param>
        /// <param name="bAdd">Optionally, specifies whether or not to add the layer to the network (default = true).</param>
        /// <returns>The toLayer is returned as the next layer.</returns>
        protected LayerParameter connectAndAddLayer(List<LayerParameter> rgFromLayer, LayerParameter toLayer, bool bAdd = true)
        {
            toLayer.bottom.Clear();

            for (int i = 0; i < rgFromLayer.Count; i++)
            {
                toLayer.bottom.Add(rgFromLayer[i].top[0]);
            }

            if (bAdd)
                m_net.layer.Add(toLayer);

            return toLayer;
        }

        /// <summary>
        /// Create a new convolution layer parameter.
        /// </summary>
        /// <param name="strName">Specifies the layer name.</param>
        /// <param name="nNumOutput">Specifies the number of output.</param>
        /// <param name="nKernelSize">Specifies the kernel size.</param>
        /// <param name="nPad">Optionally, specifies the pad (default = 0).</param>
        /// <param name="nStride">Optionally, specifies the stride (default = 1).</param>
        /// <param name="nDilation">Optionally, specifies the dilation (default = 1).</param>
        /// <returns>The convolution layer parameter is returned.</returns>
        protected LayerParameter createConvolution(string strName, int nNumOutput, int nKernelSize, int nPad = 0, int nStride = 1, int nDilation = 1)
        {
            LayerParameter conv = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            conv.name = strName;
            conv.convolution_param.num_output = (uint)nNumOutput;
            conv.convolution_param.kernel_size.Add((uint)nKernelSize);
            conv.convolution_param.pad.Add((uint)nPad);
            conv.convolution_param.stride.Add((uint)nStride);
            conv.convolution_param.dilation.Add((uint)nDilation);
            conv.convolution_param.weight_filler = new FillerParameter("xavier");
            conv.convolution_param.bias_filler = new FillerParameter("constant", 0.0);
            conv.parameters.Add(new ParamSpec(1.0, 1.0));
            conv.parameters.Add(new ParamSpec(2.0, 0.0));
            conv.top.Add(strName);

            return conv;
        }

        /// <summary>
        /// Create a new pooling layer parameter.
        /// </summary>
        /// <param name="strName">Specifies the layer name.</param>
        /// <param name="method">Specifies the pooling method.</param>
        /// <param name="nKernelSize">Specifies the kernel size.</param>
        /// <param name="nPad">Optionally, specifies the pad (default = 0).</param>
        /// <param name="nStride">Optionally, specifies the stride (default = 1).</param>
        /// <returns>The pooling layer parameter is returned.</returns>
        protected LayerParameter createPooling(string strName, PoolingParameter.PoolingMethod method, int nKernelSize, int nPad = 0, int nStride = 1)
        {
            LayerParameter pool = new LayerParameter(LayerParameter.LayerType.POOLING);
            pool.name = strName;
            pool.pooling_param.kernel_size.Add((uint)nKernelSize);
            pool.pooling_param.stride.Add((uint)nStride);
            pool.pooling_param.pad.Add((uint)nPad);
            pool.pooling_param.pool = method;
            pool.top.Add(strName);

            return pool;
        }

        /// <summary>
        /// Add a new VGG block.
        /// </summary>
        /// <param name="lastLayer">Specifies the last layer that this block is to be connected to.</param>
        /// <param name="nBlockIdx">Specifies the block index.</param>
        /// <param name="nConvIdx">Specifies the convolution index.</param>
        /// <param name="nNumOutput">Specifies the number of outputs.</param>
        /// <param name="nConvCount">Specifies the number of convolution layers to add.</param>
        /// <param name="bNoPool">When adding the last layer, specifies whether or not to add a pooling (false) or convolution (true) layer.  When this parameter is null, the adding of the last layer is skipped.</param>
        /// <param name="bDilatePool">Optionally, specifies whether or not to dilate the last pooling layer (default = false).</param>
        /// <param name="nKernelSize">Optionally, specifies the kernel size (default = 3).</param>
        /// <param name="nPad">Optionally, specifies the pad (default = 1).</param>
        /// <param name="nStride">Optionally, specifies the stride (default = 1).</param>
        /// <param name="nDilation">Optionally, specifies the dilation (default = 1).</param>
        /// <returns></returns>
        protected LayerParameter addVGGBlock(LayerParameter lastLayer, int nBlockIdx, int nConvIdx, int nNumOutput, int nConvCount, bool? bNoPool, bool bDilatePool = false, int nKernelSize = 3, int nPad = 1, int nStride = 1, int nDilation = 1)
        {
            for (int i = 0; i < nConvCount; i++)
            {
                string strConvName = "conv" + nBlockIdx.ToString() + "_" + nConvIdx.ToString();

                LayerParameter conv = createConvolution(strConvName, nNumOutput, nKernelSize, nPad, nStride, nDilation);
                lastLayer = connectAndAddLayer(lastLayer, conv);

                LayerParameter relu = new LayerParameter(LayerParameter.LayerType.RELU);
                relu.name = "relu" + nBlockIdx.ToString();
                lastLayer = connectAndAddLayer(lastLayer, relu, true);

                nConvIdx++;
            }

            if (!bNoPool.HasValue)
                return lastLayer;

            if (bNoPool.Value)
            {
                string strConvName = "conv" + nBlockIdx.ToString() + "_" + nConvIdx.ToString();
                LayerParameter conv = createConvolution(strConvName, nNumOutput, 3, 1, 2);
                lastLayer = connectAndAddLayer(lastLayer, conv);
            }
            else
            {
                string strPoolName = "pool" + nBlockIdx.ToString();
                LayerParameter pool = (bDilatePool) ? createPooling(strPoolName, PoolingParameter.PoolingMethod.MAX, 3, 1, 1) : createPooling(strPoolName, PoolingParameter.PoolingMethod.MAX, 2, 0, 2);
                lastLayer = connectAndAddLayer(lastLayer, pool);
            }

            return lastLayer;
        }

        /// <summary>
        /// Adds the final layers to the network.
        /// </summary>
        /// <param name="lastLayer">Specifies the previous layer to connect the last layers to.</param>
        /// <param name="nBlockIdx">Specifies the block index.</param>
        /// <param name="nConvIdx">Specifies the convolution index.</param>
        /// <param name="nNumOutput">Specifies the number of outputs for the convolution layers.</param>
        /// <param name="nDilation">Specifies the dilation to use for the last fully connected convolution layers (used when bFullConv = true).</param>
        /// <param name="bDilated">Specifies whether or not dialation is used.</param>
        /// <param name="bNoPool">Specifies whether or not pooling is used.</param>
        /// <param name="bFullConv">Specifies whether or not full convolution layers are used instead of inner product layers.</param>
        /// <param name="bReduced">Specifies whether or not the final layers are used to reduce the data.</param>
        /// <param name="bDropout">Specifies whether or not dropout layers are connected.</param>
        /// <returns>The last layer is returned.</returns>
        protected LayerParameter addVGGfc(LayerParameter lastLayer, int nBlockIdx, int nConvIdx, int nNumOutput, int nDilation, bool bDilated, bool bNoPool, bool bFullConv, bool bReduced, bool bDropout)
        {
            string strConvName = "conv" + nBlockIdx.ToString() + "_" + nConvIdx.ToString();
            string strPoolName = "pool" + nBlockIdx.ToString();

            if (bDilated)
            {
                if (bNoPool)
                {
                    LayerParameter conv = createConvolution(strConvName, nNumOutput, 3, 1, 1);
                    lastLayer = connectAndAddLayer(lastLayer, conv);
                }
                else
                {
                    LayerParameter pool = createPooling(strPoolName, PoolingParameter.PoolingMethod.MAX, 3, 1, 1);
                    lastLayer = connectAndAddLayer(lastLayer, pool);
                }
            }
            else
            {
                if (bNoPool)
                {
                    LayerParameter conv = createConvolution(strConvName, nNumOutput, 3, 1, 2);
                    lastLayer = connectAndAddLayer(lastLayer, conv);
                }
                else
                {
                    LayerParameter pool = createPooling(strPoolName, PoolingParameter.PoolingMethod.MAX, 2, 0, 2);
                    lastLayer = connectAndAddLayer(lastLayer, pool);
                }
            }

            if (bFullConv)
            {
                int nKernelSize;

                if (bDilated)
                {
                    if (bReduced)
                    {
                        nDilation *= 6;
                        nKernelSize = 3;
                        nNumOutput = 1024;
                    }
                    else
                    {
                        nDilation *= 2;
                        nKernelSize = 7;
                        nNumOutput = 4096;
                    }
                }
                else
                {
                    if (bReduced)
                    {
                        nDilation *= 3;
                        nKernelSize = 3;
                        nNumOutput = 1024;
                    }
                    else
                    {
                        nKernelSize = 7;
                        nNumOutput = 4096;
                    }
                }

                int nPad = (int)((nKernelSize + (nDilation - 1) * (nKernelSize - 1)) - 1) / 2;
                LayerParameter fc6 = createConvolution("fc6", nNumOutput, nKernelSize, nPad, 1, nDilation);
                lastLayer = connectAndAddLayer(lastLayer, fc6);

                LayerParameter relu = new LayerParameter(LayerParameter.LayerType.RELU);
                relu.name = "relu" + nBlockIdx.ToString();
                lastLayer = connectAndAddLayer(lastLayer, relu, true);

                if (bDropout)
                {
                    LayerParameter dropout = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                    dropout.name = "dropout6";
                    dropout.dropout_param.dropout_ratio = 0.5;
                    lastLayer = connectAndAddLayer(lastLayer, dropout, true);
                }

                LayerParameter fc7 = createConvolution("fc7", nNumOutput, 1);
                lastLayer = connectAndAddLayer(lastLayer, fc7);

                if (bDropout)
                {
                    LayerParameter dropout = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                    dropout.name = "dropout7";
                    dropout.dropout_param.dropout_ratio = 0.5;
                    lastLayer = connectAndAddLayer(lastLayer, dropout, true);
                }
            }
            else
            {
                LayerParameter fc6 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                fc6.inner_product_param.num_output = 4096;
                fc6.name = "fc6";
                lastLayer = connectAndAddLayer(lastLayer, fc6, true);

                if (bDropout)
                {
                    LayerParameter dropout = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                    dropout.name = "dropout6";
                    dropout.dropout_param.dropout_ratio = 0.5;
                    lastLayer = connectAndAddLayer(lastLayer, dropout, true);
                }

                LayerParameter fc7 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                fc6.inner_product_param.num_output = 4096;
                fc6.name = "fc7";
                lastLayer = connectAndAddLayer(lastLayer, fc7);

                if (bDropout)
                {
                    LayerParameter dropout = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                    dropout.name = "dropout7";
                    dropout.dropout_param.dropout_ratio = 0.5;
                    lastLayer = connectAndAddLayer(lastLayer, dropout, true);
                }
            }

            return lastLayer;
        }

        /// <summary>
        /// Adds the full VGG body to the network, connecting it to the 'strFromLayer'.
        /// </summary>
        /// <param name="lastLayer">Specifies the layer to connect the VGG net to.</param>
        /// <param name="bNeedFc">Optionally, specifies whether or not to add the fully connected end layers (default = true).</param>
        /// <param name="bFullConv">Optionally, specifies whether or not full convolution layers are used instead of inner product layers (default = true).</param>
        /// <param name="bReduced">Optionally, specifies whether or not the final layers are used to reduce the data (default = true).</param>
        /// <param name="bDilated">Optionally, specifies whether or not dialation is used (default = true).</param>
        /// <param name="bNoPool">Optionally, specifies whether or not pooling is used (default = false, use pooling).</param>
        /// <param name="bDropout">Optionally, specifies whether or not dropout layers are connected (default = false).</param>
        /// <param name="rgstrFreezeLayers">Optionally, specifies a set of layers who's training is to be frozen (default = null to ignore).</param>
        /// <param name="bDilatePool4">Optionally, specifies whether or not to dilate pool #4 (default = false).</param>
        /// <returns>The last layer is returned.</returns>
        protected LayerParameter addVGGNetBody(LayerParameter lastLayer, bool bNeedFc = true, bool bFullConv = true, bool bReduced = true, bool bDilated = true, bool bNoPool = false, bool bDropout = false, List<string> rgstrFreezeLayers = null, bool bDilatePool4 = false)
        {
            lastLayer = addVGGBlock(lastLayer, 1, 1, 64, 2, bNoPool, false, 3, 1, 1);
            lastLayer = addVGGBlock(lastLayer, 2, 1, 128, 2, bNoPool, false, 3, 1, 1);
            lastLayer = addVGGBlock(lastLayer, 3, 1, 256, 3, bNoPool, false, 3, 1, 1);
            lastLayer = addVGGBlock(lastLayer, 4, 1, 512, 3, bNoPool, bDilatePool4, 3, 1, 1);

            int nDilation = (bDilatePool4) ? 2 : 1;
            int nKernelSize = 3;
            int nPad = (int)((nKernelSize + (nDilation - 1) * (nKernelSize - 1)) - 1) / 2;
            lastLayer = addVGGBlock(lastLayer, 5, 1, 512, 3, null, false, nKernelSize, nPad, 1, nDilation);

            if (bNeedFc)    
                lastLayer = addVGGfc(lastLayer, 5, 4, 512, nDilation, bDilated, bNoPool, bFullConv, bReduced, bDropout);

            if (rgstrFreezeLayers != null)
            {
                foreach (string strFreezeLayer in rgstrFreezeLayers)
                {
                    LayerParameter p = findLayer(strFreezeLayer);
                    if (p != null)
                        p.freeze_learning = true;
                }
            }

            return lastLayer;
        }

        /// <summary>
        /// Returns the base net altered by the model builder.
        /// </summary>
        public NetParameter Net
        {
            get { return m_net; }
        }

        /// <summary>
        /// Returns the base solver.
        /// </summary>
        public SolverParameter Solver
        {
            get { return m_solver; }
        }
    }

    /// <summary>
    /// The MultiBoxHeadInfo contains information used to build the multi-box head of layers.
    /// </summary>
    public class MultiBoxHeadInfo
    {
        string m_strSourceLayer;
        double? m_dfMinSize;
        double? m_dfMaxSize;
        double? m_dfStepWidth;
        double? m_dfStepHeight;
        double? m_dfAspectRatioHeight;
        double? m_dfAspectRatioWidth;
        double? m_dfNormalization;
        double? m_nInterLayerDepth;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strSrcLayer">Specifies the source layer.</param>
        /// <param name="dfMinSize">Specifies the minimum bbox size.</param>
        /// <param name="dfMaxSize">Specifies the maximum bbox size.</param>
        /// <param name="dfStepWidth">Specifies the step width size.</param>
        /// <param name="dfStepHeight">Specifies the step height size.</param>
        /// <param name="dfAspectRatioWidth">Specifies the aspect ratio width.</param>
        /// <param name="dfAspectRatioHeight">Specifies the aspect ratio height.</param>
        /// <param name="dfNormalization">Specifies the normalization to use or -1 to ignore.</param>
        /// <param name="nInterLayerDepth">Specifies the inner layer depth or -1 to ignore.</param>
        public MultiBoxHeadInfo(string strSrcLayer, double? dfMinSize = null, double? dfMaxSize = null, double? dfStepWidth = null, double? dfStepHeight = null, double? dfAspectRatioWidth = null, double? dfAspectRatioHeight = null, double? dfNormalization = null, int? nInterLayerDepth = null)
        {
            m_strSourceLayer = strSrcLayer;
            m_dfMinSize = dfMinSize;
            m_dfMaxSize = dfMaxSize;
            m_dfStepWidth = dfStepWidth;
            m_dfStepHeight = dfStepHeight;
            m_dfAspectRatioHeight = dfAspectRatioHeight;
            m_dfAspectRatioWidth = dfAspectRatioWidth;
            m_dfNormalization = dfNormalization;
            m_nInterLayerDepth = nInterLayerDepth;
        }

        private bool verify(double? df1, double? df2)
        {
            if ((df1.HasValue && !df2.HasValue) || (!df1.HasValue && df2.HasValue))
                return false;

            return true;
        }

        /// <summary>
        /// Verify that all numical values are consistently set (or not) between two info objects.
        /// </summary>
        /// <param name="info">Specifies the info object to compare.</param>
        /// <returns>If the settings are consistent, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Verify(MultiBoxHeadInfo info)
        {
            if (!verify(m_dfAspectRatioHeight, info.m_dfAspectRatioHeight))
                return false;

            if (!verify(m_dfAspectRatioWidth, info.m_dfAspectRatioWidth))
                return false;

            if (!verify(m_dfMaxSize, info.m_dfMaxSize))
                return false;

            if (!verify(m_dfMinSize, info.m_dfMinSize))
                return false;

            if (!verify(m_dfNormalization, info.m_dfNormalization))
                return false;

            if (!verify(m_dfStepWidth, info.m_dfStepWidth))
                return false;

            if (!verify(m_dfStepHeight, info.m_dfStepHeight))
                return false;

            if (!verify(m_nInterLayerDepth, info.m_nInterLayerDepth))
                return false;

            return true;
        }

        /// <summary>
        /// Returns the source layer.
        /// </summary>
        public string SourceLayer
        {
            get { return m_strSourceLayer; }
        }

        /// <summary>
        /// Returns the bbox minimum size.
        /// </summary>
        public double? MinSize
        {
            get { return m_dfMinSize; }
        }

        /// <summary>
        /// Returns the bbox maximum size.
        /// </summary>
        public double? MaxSize
        {
            get { return m_dfMaxSize; }
        }

        /// <summary>
        /// Returns the step eight.
        /// </summary>
        public double? StepHeight
        {
            get { return m_dfStepHeight; }
        }

        /// <summary>
        /// Returns the step width.
        /// </summary>
        public double? StepWidth
        {
            get { return m_dfStepWidth; }
        }

        /// <summary>
        /// Returns the aspect ratio height.
        /// </summary>
        public double? AspectRatioHeight
        {
            get { return m_dfAspectRatioHeight; }
        }

        /// <summary>
        /// Returns the aspect ratio width.
        /// </summary>
        public double? AspectRatioWidth
        {
            get { return m_dfAspectRatioWidth; }
        }

        /// <summary>
        /// Returns the normalization, or -1 to ignore.
        /// </summary>
        public double? Normalization
        {
            get { return m_dfNormalization; }
        }

        /// <summary>
        /// Returns the inner layer depth, or -1 to ignore.
        /// </summary>
        public double? InterLayerDepth
        {
            get { return m_nInterLayerDepth; }
        }
    }
}
