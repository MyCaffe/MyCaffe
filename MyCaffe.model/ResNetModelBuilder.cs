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
    /// The ResNetModelBuilder adds the extra layers to a 'base' model for the ResNet model.
    /// </summary>
    public class ResNetModelBuilder : ModelBuilder
    {
        int m_nGpuID = 0;
        List<int> m_rgGpuID = new List<int>();
        int m_nBatchSize;
        int m_nAccumBatchSize;
        int m_nBatchSizePerDevice;
        int m_nIterSize;
        double m_dfBaseLr;
        int m_nTestIter = 100;
        string m_strTrainDataSource;
        string m_strTestDataSource;
        TransformationParameter m_transformTrain = null;
        TransformationParameter m_transformTest = null;
        string m_strModel;
        MODEL m_model = MODEL.RESNET101;
        bool m_bUsePool5 = false;
        bool m_bUseDilationConv5 = false;
        string m_strDataset;
        List<Tuple<int, bool>> m_rgIpLayers;
        bool m_bSiamese = false;
        int m_nChannels = 3;

        /// <summary>
        /// Defines the type of model to create.
        /// </summary>
        public enum MODEL
        {
            /// <summary>
            /// Specifies to create a ResNet101 model.
            /// </summary>
            RESNET101,
            /// <summary>
            /// Specifies to create a ResNet152 model.
            /// </summary>
            RESNET152
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strBaseDirectory">Specifies the base directory that contains the data and models.</param>
        /// <param name="strDataset">Specifies the dataset that the model will run on.</param>
        /// <param name="nChannels">Specifies the number of channels in the data set (e.g. color = 3, b/w = 1).</param>
        /// <param name="bSiamese">Specifies whether or not to create a Siamese network."</param> 
        /// <param name="rgIpLayers">Specifies a list of inner product layers added to the end of the network where each entry specifies the number of output and whether or not Noise is enabled for the layer.</param>
        /// <param name="bUsePool5">Specifies whether or not to use the Pool layer as the last layer.</param>
        /// <param name="bUseDilationConv5">Specifies whether or not to use dilation on block 5 layers.</param>
        /// <param name="model">Specifies the type of ResNet model to create.</param>
        /// <param name="nBatchSize">Optionally, specifies the batch size (default = 32).</param>
        /// <param name="nAccumBatchSize">Optionally, specifies the accumulation batch size (default = 32).</param>
        /// <param name="rgGpuId">Optionally, specifies a set of GPU ID's to use (when null, GPU=0 is used).</param>
        /// <param name="net">Specifies the 'base' net parameter that is to be altered.</param>
        public ResNetModelBuilder(string strBaseDirectory, string strDataset, int nChannels, bool bSiamese, List<Tuple<int, bool>> rgIpLayers, bool bUsePool5, bool bUseDilationConv5, MODEL model, int nBatchSize = 32, int nAccumBatchSize = 32, List<int> rgGpuId = null, NetParameter net = null) 
            : base(strBaseDirectory, net)
        {
            if (rgGpuId == null)
                m_rgGpuID.Add(0);
            else
                m_rgGpuID = new List<int>(rgGpuId);

            m_nChannels = nChannels;
            m_bSiamese = bSiamese;
            m_rgIpLayers = rgIpLayers;
            m_model = model;
            m_strModel = model.ToString();
            m_nBatchSize = nBatchSize;
            m_nAccumBatchSize = nAccumBatchSize;
            m_nIterSize = m_nAccumBatchSize / m_nBatchSize;

            m_nBatchSizePerDevice = (m_rgGpuID.Count == 1) ? m_nBatchSize : m_nBatchSize / m_rgGpuID.Count;
            m_nIterSize = (int)Math.Ceiling((float)m_nAccumBatchSize / (m_nBatchSizePerDevice * m_rgGpuID.Count));
            m_nGpuID = m_rgGpuID[0];
            m_dfBaseLr = 0.001;

            m_bUseDilationConv5 = bUseDilationConv5;
            m_bUsePool5 = bUsePool5;
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

            // Test parameters.
            m_solver.test_iter.Add(m_nTestIter);
            m_solver.test_interval = 10000;
            m_solver.test_initialization = false;
            m_solver.eval_type = SolverParameter.EvaluationType.CLASSIFICATION;

            return m_solver;
        }


        /// <summary>
        /// Create the training SSD model for the pascal dataset.
        /// </summary>
        public override NetParameter CreateModel(bool bDeploy = false)
        {
            LayerParameter lastLayer = null;
            LayerParameter data = null;
            string strData1 = null;
            string strData2 = null;

            m_strTrainDataSource = m_strDataset + ".training";
            m_strTestDataSource = m_strDataset + ".testing";

            m_net = createNet(m_strModel);

            string strDataName = "data";
            string strLayerPostfix = "";
            bool bNamedParams = false;

            if (m_bSiamese)
            {
                strDataName = "pair_data";
                strLayerPostfix = "_p";
                bNamedParams = true;
            }

            if (!bDeploy)
                addDataLayer(m_strTrainDataSource, Phase.TRAIN, m_nBatchSize, true, m_transformTrain, strDataName, m_bSiamese);
            data = addDataLayer(m_strTestDataSource, Phase.TEST, m_nBatchSize, true, m_transformTest, strDataName, m_bSiamese);

            if (m_bSiamese)
            {
                LayerParameter slice = new LayerParameter(LayerParameter.LayerType.SLICE);
                slice.slice_param.axis = 1;
                slice.slice_param.slice_point.Add((uint)m_nChannels);
                slice.exclude.Add(new NetStateRule(Phase.RUN));
                lastLayer = connectAndAddLayer(data, slice);
                strData1 = "data";
                strData2 = "data_p";
                slice.top.Add(strData1);
                slice.top.Add(strData2);
                strDataName = "data";
            }

            lastLayer = addBody(bDeploy, strDataName, bNamedParams);
            LayerParameter output1 = lastLayer;

            if (m_bSiamese)
            {
                LayerParameter decode = new LayerParameter(LayerParameter.LayerType.DECODE);
                decode.name = "decode1";
                decode.decode_param.target = param.beta.DecodeParameter.TARGET.CENTROID;
                decode.decode_param.cache_size = 100;
                decode.top.Add(decode.name);
                lastLayer = connectAndAddLayer(lastLayer, decode);
                decode.bottom.Add(data.top[1]);

                if (!bDeploy)
                {
                    LayerParameter silence1 = new LayerParameter(LayerParameter.LayerType.SILENCE);
                    silence1.name = "silence1";
                    silence1.include.Add(new NetStateRule(Phase.TRAIN));
                    connectAndAddLayer(lastLayer, silence1);

                    lastLayer = addBody(false, strDataName + strLayerPostfix, bNamedParams, strLayerPostfix, Phase.RUN);
                    LayerParameter output2 = lastLayer;

                    LayerParameter loss = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
                    loss.name = "loss";
                    loss.contrastive_loss_param.margin = 5;
                    loss.top.Add(loss.name);
                    loss.top.Add("match");
                    connectAndAddLayer(lastLayer, loss);
                    loss.bottom.Clear();
                    loss.bottom.Add(output1.top[0]);
                    loss.bottom.Add(output2.top[0]);
                    loss.bottom.Add(data.top[1]);
                    loss.loss_weight.Add(1);
                    loss.loss_weight.Add(0);
                    addExclusion(loss, Phase.RUN);

                    LayerParameter silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                    silence.name = "silence2";
                    connectAndAddLayer(loss, silence);
                    silence.bottom[0] = "match";
                    addExclusion(silence, Phase.RUN);

                    LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY_DECODE);
                    accuracy.name = "accuracy";
                    accuracy.include.Add(new NetStateRule(Phase.TEST));
                    accuracy.top.Add(accuracy.name);
                    connectAndAddLayer(decode, accuracy);
                    accuracy.bottom.Add(data.top[1]);
                }
            }
            else
            {
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
            }

            return m_net;
        }

        private LayerParameter addBody(bool bDeploy, string strDataName, bool bNamedParams = false, string strLayerPostfix = "", Phase phaseExclude = Phase.NONE)
        {
            LayerParameter lastLayer;

            switch (m_model)
            {
                case MODEL.RESNET101:
                    lastLayer = addResNetBody(strDataName, 4, 23, m_bUsePool5, m_bUseDilationConv5, bNamedParams, strLayerPostfix, phaseExclude);
                    break;

                case MODEL.RESNET152:
                    lastLayer = addResNetBody(strDataName, 8, 36, m_bUsePool5, m_bUseDilationConv5, bNamedParams, strLayerPostfix, phaseExclude);
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

        /// <summary>
        /// Create the testing SSD model for the pascal dataset.
        /// </summary>
        public override NetParameter CreateDeployModel()
        {
            return CreateModel(true);
        }

        /// <summary>
        /// Add extra layers (for SSD with the Pascal dataset) on top of a 'base' network (e.g. VGGNet or Inception)
        /// </summary>
        /// <param name="bUseBatchNorm">Optionally, specifies whether or not to use batch normalization layers (default = <i>true</i>).</param>
        /// <param name="dfLrMult">Optionally, specifies the learning rate multiplier (default = 1.0).</param>
        protected override LayerParameter addExtraLayers(bool bUseBatchNorm = true, double dfLrMult = 1)
        {
            return m_net.layer[m_net.layer.Count - 1];
        }
    }
}
