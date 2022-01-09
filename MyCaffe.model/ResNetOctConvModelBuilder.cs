using MyCaffe.basecode;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.model
{
    /// <summary>
    /// The ResNetOctConvModelBuilder is used to build Octave Convolution based RESNET models.
    /// </summary>
    public class ResNetOctConvModelBuilder : ModelBuilder
    {
        int m_nGpuID = 0;
        List<int> m_rgGpuID = new List<int>();
        int m_nBatchSize;
        int m_nIterSize;
        double m_dfBaseLr;
        int m_nTestIter = 100;
        string m_strTrainDataSource;
        string m_strTestDataSource;
        TransformationParameter m_transformTrain = null;
        TransformationParameter m_transformTest = null;
        string m_strModel;
        MODEL m_model = MODEL.RESNET26;
        string m_strDataset;
        List<Tuple<int, bool>> m_rgIpLayers;
        int m_nInplanes = 64;
        int m_nGroups = 1;
        int m_nBaseWidth = 64;
        int m_nExpansion = 4;

        /// <summary>
        /// Defines the type of model to create.
        /// </summary>
        public enum MODEL
        {
            /// <summary>
            /// Create a ResNet26 model.
            /// </summary>
            RESNET26
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strBaseDirectory">Specifies the base directory that contains the data and models.</param>
        /// <param name="strDataset">Specifies the dataset that the model will run on.</param>
        /// <param name="rgIpLayers">Specifies a list of inner product layers added to the end of the network where each entry specifies the number of output and whether or not Noise is enabled for the layer.</param>
        /// <param name="model">Specifies the type of ResNet model to create.</param>
        /// <param name="nBatchSize">Optionally, specifies the batch size (default = 32).</param>
        /// <param name="rgGpuId">Optionally, specifies a set of GPU ID's to use (when null, GPU=0 is used).</param>
        /// <param name="net">Specifies the 'base' net parameter that is to be altered.</param>
        public ResNetOctConvModelBuilder(string strBaseDirectory, string strDataset, List<Tuple<int, bool>> rgIpLayers, MODEL model, int nBatchSize = 32, List<int> rgGpuId = null, NetParameter net = null)
            : base(strBaseDirectory, net)
        {
            if (rgGpuId == null)
                m_rgGpuID.Add(0);
            else
                m_rgGpuID = new List<int>(rgGpuId);

            m_rgIpLayers = rgIpLayers;
            m_model = model;
            m_strModel = model.ToString();
            m_nBatchSize = nBatchSize;

            m_nIterSize = 1;
            m_nGpuID = m_rgGpuID[0];
            m_dfBaseLr = 0.001;

            m_strDataset = strDataset;

            //-------------------------------------------------------
            // Create the transformer for Training.
            //-------------------------------------------------------
            m_transformTrain = new TransformationParameter();
            m_transformTrain.mirror = true;
            m_transformTrain.color_order = TransformationParameter.COLOR_ORDER.BGR; // to support caffe models.
            m_transformTrain.mean_value = new List<double>();
            m_transformTrain.mean_value.Add(104);
            m_transformTrain.mean_value.Add(117);
            m_transformTrain.mean_value.Add(123);

            //-------------------------------------------------------
            // Create the transformer for Testing.
            //-------------------------------------------------------
            m_transformTest = new TransformationParameter();
            m_transformTest.color_order = TransformationParameter.COLOR_ORDER.BGR; // to support caffe models.
            m_transformTest.mean_value = new List<double>();
            m_transformTest.mean_value.Add(104);
            m_transformTest.mean_value.Add(117);
            m_transformTest.mean_value.Add(123);
        }

        /// <summary>
        /// Create the deploy model.
        /// </summary>
        public override NetParameter CreateDeployModel()
        {
            return CreateModel(true);
        }

        /// <summary>
        /// Create the training model.
        /// </summary>
        /// <param name="bDeploy">Optionally, specifies to create a deployment model (default = false).</param>
        public override NetParameter CreateModel(bool bDeploy = false)
        {
            LayerParameter lastLayer = null;
            LayerParameter data = null;

            m_strTrainDataSource = m_strDataset + ".training";
            m_strTestDataSource = m_strDataset + ".testing";

            m_net = createNet(m_strModel);

            string strDataName = "data";
            bool bNamedParams = false;

            if (!bDeploy)
                addDataLayer(m_strTrainDataSource, Phase.TRAIN, m_nBatchSize, true, m_transformTrain, strDataName);
            data = addDataLayer(m_strTestDataSource, Phase.TEST, m_nBatchSize, true, m_transformTest, strDataName);

            lastLayer = addBody(bDeploy, strDataName, bNamedParams);
            LayerParameter output1 = lastLayer;

            if (!bDeploy)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
                loss.name = "loss";
                loss.include.Add(new NetStateRule(Phase.TRAIN));
                loss.top.Add(loss.name);
                connectAndAddLayer(lastLayer, loss);
                loss.bottom.Add(data.top[1]);
            }

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 1;
            softmax.include.Add(new NetStateRule(Phase.TEST));
            softmax.include.Add(new NetStateRule(Phase.RUN));
            softmax.top.Add(softmax.name);
            lastLayer = connectAndAddLayer(lastLayer, softmax);

            if (!bDeploy)
            {
                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.include.Add(new NetStateRule(Phase.TEST));
                accuracy.top.Add(accuracy.name);
                connectAndAddLayer(lastLayer, accuracy);
                accuracy.bottom.Add(data.top[1]);
            }

            return m_net;
        }

        private LayerParameter addBody(bool bDeploy, string strDataName, bool bNamedParams = false, string strLayerPostfix = "", Phase phaseExclude = Phase.NONE)
        {
            LayerParameter lastLayer;

            switch (m_model)
            {
                case MODEL.RESNET26:
                    lastLayer = addResNetOctConvBody(strDataName, new int[] { 2, 2, 2, 2 });
                    break;

                default:
                    throw new Exception("The model type '" + m_model.ToString() + "' is not supported.");
            }

            for (int i = 0; i < m_rgIpLayers.Count; i++)
            {
                LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                string strName = "fc" + (i + 1).ToString();
                ip.name = strName + strLayerPostfix;
                ip.inner_product_param.axis = 1;
                ip.inner_product_param.num_output = (uint)m_rgIpLayers[i].Item1;
                ip.inner_product_param.enable_noise = m_rgIpLayers[i].Item2;
                ip.top.Add(ip.name);
                ip.parameters.Add(new ParamSpec(1, 1, (bNamedParams) ? strName + "_w" : null));
                ip.parameters.Add(new ParamSpec(1, 2, (bNamedParams) ? strName + "_b" : null));
                addExclusion(ip, phaseExclude);
                lastLayer = connectAndAddLayer(lastLayer, ip);
            }

            return lastLayer;
        }

        private LayerParameter addResNetOctConvBody(string strDataName, int[] rgBlocks, bool bNamedParams = false, string strLayerPostfix = "", Phase phaseExclude = Phase.NONE)
        {
            string strConvPrefix = "";
            string strConvPostfix = "";
            string strBnPrefix = "bn_";
            string strBnPostfix = "";
            string strScalePrefix = "scale_";
            string strScalePostfix = "";

            LayerParameter lastLayer = addConvBNLayer(strDataName, "conv1", true, true, m_nInplanes, 7, 3, 2, 1, 1, SCALE_BIAS.NONE, strConvPrefix, strConvPostfix, strBnPrefix, strBnPostfix, strScalePrefix, strScalePostfix, "", "_bias", bNamedParams, strLayerPostfix, phaseExclude);

            LayerParameter pool = createPooling("pool1" + strLayerPostfix, PoolingParameter.PoolingMethod.MAX, 3, 1, 2);
            addExclusion(pool, phaseExclude);
            lastLayer = connectAndAddLayer(lastLayer, pool);

            lastLayer = make_layer(lastLayer, 1, 64, rgBlocks[0], 1, true, false);
            lastLayer = make_layer(lastLayer, 2, 128, rgBlocks[1], 2, false, false);
            lastLayer = make_layer(lastLayer, 3, 256, rgBlocks[2], 2, false, false);
            lastLayer = make_layer(lastLayer, 4, 512, rgBlocks[3], 2, false, true);

            pool = createPooling("pool2" + strLayerPostfix, PoolingParameter.PoolingMethod.AVE, 3, 1, 2);
            addExclusion(pool, phaseExclude);
            lastLayer = connectAndAddLayer(lastLayer, pool);

            LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            string strName = "fc1";
            ip.name = strName + strLayerPostfix;
            ip.inner_product_param.axis = 1;
            ip.inner_product_param.num_output = (uint)(512 * m_nExpansion);
            ip.top.Add(ip.name);
            ip.parameters.Add(new ParamSpec(1, 1, (bNamedParams) ? strName + "_w" : null));
            ip.parameters.Add(new ParamSpec(1, 2, (bNamedParams) ? strName + "_b" : null));
            addExclusion(ip, phaseExclude);
            lastLayer = connectAndAddLayer(lastLayer, ip);

            return lastLayer;
        }

        private LayerParameter make_layer(LayerParameter lastLayer, int nIdx, int nPlanes, int nBlocks, int nStride, bool bInput, bool bOutput)
        {
            int nBlockIdx = 1;
            lastLayer = addBlock(lastLayer, nIdx, ref nBlockIdx, nPlanes, nStride, bInput, bOutput);

            for (int i = 0; i < nBlocks; i++)
            {
                lastLayer = addBlock(lastLayer, nIdx, ref nBlockIdx, nPlanes, nStride, bInput, bOutput);
            }

            return lastLayer;
        }

        private LayerParameter addBlock(LayerParameter lastLayer, int nIdx, ref int nBlockIdx, int nPlanes, int nStride, bool bInput, bool bOutput)
        {
            double dfAlphaIn = (bInput) ? 0.0 : 0.5;
            double dfAlphaOut = 0.5;

            int nWidth = (int)(nPlanes * (m_nBaseWidth / 64.0)) * m_nGroups;

            string strName = "conv1_" + nIdx.ToString() + "_" + nBlockIdx.ToString();
            lastLayer = addOctConvBNLayer(lastLayer.name, strName, nWidth, 1, 0, 1, (bInput) ? 0 : dfAlphaIn, dfAlphaOut);

            dfAlphaIn = 0.5;

            strName = "conv2_" + nIdx.ToString() + "_" + nBlockIdx.ToString();
            lastLayer = addOctConvBNLayer(lastLayer.name, strName, nWidth, 3, 0, 1, dfAlphaIn, dfAlphaOut);

            dfAlphaOut = (bOutput) ? 0.0 : 0.5;

            strName = "conv3_" + nIdx.ToString() + "_" + nBlockIdx.ToString();
            lastLayer = addOctConvBNLayer(lastLayer.name, strName, nPlanes * m_nExpansion, 1, 0, 1, dfAlphaIn, dfAlphaOut);

            if (nStride != 1 || m_nInplanes != nPlanes * m_nExpansion)
            {
                strName = "ds_" + nIdx.ToString() + "_" + nBlockIdx.ToString();
                lastLayer = addOctConvBNLayer(lastLayer.name, strName, nPlanes * m_nExpansion, 1, 0, nStride, dfAlphaIn, dfAlphaOut);
            }

            nBlockIdx++;

            return lastLayer;
        }

        /// <summary>
        /// Add octave convolution, batch-norm layers.
        /// </summary>
        /// <param name="strInputLayer">Specifies the first input layer.</param>
        /// <param name="strOutputLayer">Specifies the output layer.</param>
        /// <param name="nNumOutput">Specifies the number of output.</param>
        /// <param name="nKernelSize">Specifies the kernel size.</param>
        /// <param name="nPad">Specifies the pad.</param>
        /// <param name="nStride">Specifies the stride.</param>
        /// <param name="dfAlphaIn">Specifies the alpha in.</param>
        /// <param name="dfAlphaOut">Specifies the alpha out.</param>
        /// <param name="bUseBias">Optionally, specifies to use bias (default = true).</param>
        /// <param name="strConvPrefix">Optionally, specifies the convolution layer name prefix (default = "").</param>
        /// <param name="strConvPostfix">Optionally, specifies the convolution layer name postfix (default = "").</param>
        /// <param name="strBnPrefix">Optionally, specifies the batch-norm layer name prefix (default = "").</param>
        /// <param name="strBnPostfix">Optionally, specifies the batch-norm layer name postfix (default = "_bn").</param>
        /// <param name="strLayerPostfix">Optionally, specifies a layer name postfix (default = "").</param>
        /// <param name="phaseExclude">Optionally, specifies a phase to exclude (default = NONE).</param>
        /// <returns>The last layer added is returned.</returns>
        protected LayerParameter addOctConvBNLayer(string strInputLayer, string strOutputLayer, int nNumOutput, int nKernelSize, int nPad, int nStride, double dfAlphaIn, double dfAlphaOut, bool bUseBias = true, string strConvPrefix = "", string strConvPostfix = "", string strBnPrefix = "", string strBnPostfix = "_bn", string strLayerPostfix = "", Phase phaseExclude = Phase.NONE)
        {
            double dfLrMult = 1;
            bool bNamedParams = false;

            LayerParameter lastLayer1;
            string strName = strConvPrefix + strOutputLayer + strConvPostfix;

            LayerParameter.LayerType type = LayerParameter.LayerType.CONVOLUTION_OCTAVE;
            LayerParameter convLayer = new LayerParameter(type);
            convLayer.convolution_param.weight_filler = new FillerParameter("xavier");
            convLayer.convolution_param.bias_filler = new FillerParameter("constant", 0);
            convLayer.convolution_param.bias_term = bUseBias;
            convLayer.name = strName + strLayerPostfix;
            convLayer.convolution_param.kernel_size.Add((uint)nKernelSize);
            convLayer.convolution_param.pad.Add((uint)nPad);
            convLayer.convolution_param.stride.Add((uint)nStride);
            convLayer.convolution_param.dilation.Add((uint)1);
            convLayer.convolution_param.num_output = (uint)nNumOutput;
            convLayer.top.Add(convLayer.name);

            convLayer.convolution_octave_param.alpha_in = dfAlphaIn;
            convLayer.convolution_octave_param.alpha_out = dfAlphaOut;

            if (dfAlphaOut > 0)
                convLayer.top.Add(convLayer.name + "_l");

            addExclusion(convLayer, phaseExclude);

            // Setup the BachNorm Layer
            convLayer.parameters.Add(new ParamSpec(dfLrMult, 1.0, (bNamedParams) ? strName + "_w" : null));
            convLayer.convolution_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.01);
            convLayer.convolution_param.bias_term = false;

            LayerParameter bnLayer = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
            strName = strBnPrefix + strOutputLayer + strBnPostfix;
            bnLayer.name = strName + strLayerPostfix;
            bnLayer.batch_norm_param.eps = 0.001;
            bnLayer.batch_norm_param.moving_average_fraction = 0.999;
            bnLayer.batch_norm_param.use_global_stats = false;
            bnLayer.parameters.Add(new ParamSpec(0.0, 0.0, (bNamedParams) ? strName + "_w1" : null));
            bnLayer.parameters.Add(new ParamSpec(0.0, 0.0, (bNamedParams) ? strName + "_w2" : null));
            bnLayer.parameters.Add(new ParamSpec(0.0, 0.0, (bNamedParams) ? strName + "_w3" : null));
            bnLayer.top.Add(bnLayer.name);
            addExclusion(bnLayer, phaseExclude);

            string strInputLayer2 = null;
            if (dfAlphaIn > 0)
                strInputLayer2 = strInputLayer + "_l";

            lastLayer1 = connectAndAddLayer(strInputLayer, convLayer, strInputLayer2);
            lastLayer1 = connectAndAddLayer(lastLayer1, bnLayer, true, true);

            LayerParameter reluLayer = new LayerParameter(LayerParameter.LayerType.RELU);
            reluLayer.name = convLayer.name + "_relu";
            addExclusion(reluLayer, phaseExclude);
            lastLayer1 = connectAndAddLayer(lastLayer1, reluLayer, true, true);

            if (dfAlphaOut > 0)
            {
                bnLayer = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
                strName = strBnPrefix + strOutputLayer + strBnPostfix;
                bnLayer.name = strName + strLayerPostfix + "_l";
                bnLayer.batch_norm_param.eps = 0.001;
                bnLayer.batch_norm_param.moving_average_fraction = 0.999;
                bnLayer.batch_norm_param.use_global_stats = false;
                bnLayer.parameters.Add(new ParamSpec(0.0, 0.0, (bNamedParams) ? strName + "_w1" : null));
                bnLayer.parameters.Add(new ParamSpec(0.0, 0.0, (bNamedParams) ? strName + "_w2" : null));
                bnLayer.parameters.Add(new ParamSpec(0.0, 0.0, (bNamedParams) ? strName + "_w3" : null));
                bnLayer.top.Add(bnLayer.name + "_l");
                addExclusion(bnLayer, phaseExclude);

                LayerParameter lastLayer2 = connectAndAddLayer(convLayer, bnLayer, true, true, 1);

                reluLayer = new LayerParameter(LayerParameter.LayerType.RELU);
                reluLayer.name = convLayer.name + "_relu_l";
                addExclusion(reluLayer, phaseExclude);
                connectAndAddLayer(lastLayer2, reluLayer, true, true);
            }

            return lastLayer1;
        }


        /// <summary>
        /// Create the base solver to use.
        /// </summary>
        /// <returns>
        /// The solver parameter created is returned.
        /// </returns>
        public override SolverParameter CreateSolver()
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
            m_solver.clip_gradients = 1;

            // Test parameters.
            m_solver.test_iter.Add(m_nTestIter);
            m_solver.test_interval = 10000;
            m_solver.test_initialization = false;
            m_solver.eval_type = SolverParameter.EvaluationType.CLASSIFICATION;

            return m_solver;
        }

        /// <summary>
        /// Add an extra layer.
        /// </summary>
        /// <param name="bUseBatchNorm">Not used.</param>
        /// <param name="dfLrMult">Not used.</param>
        /// <returns>Currently, just returns last layer.</returns>
        protected override LayerParameter addExtraLayers(bool bUseBatchNorm = true, double dfLrMult = 1)
        {
            return m_net.layer[m_net.layer.Count - 1];
        }
    }
}
