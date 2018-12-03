using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.solvers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.extras
{
    /// <summary>
    /// The NeuralStyleTransfer object uses the GramLayer, TVLossLayer and LBFGSSolver to perform the neural style transfer algorithm.
    /// </summary>
    /// <remarks>
    /// @see [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) by Marc Schmidt, 2005
    /// @see [ftokarev/caffe-neural-style Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017. 
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// </remarks>
    public class NeuralStyleTransfer<T> : IDisposable
    {
        CudaDnn<T> m_cuda;
        Log m_log;
        int m_nIterations = 1000;
        int m_nDisplayEvery = 100;                // vgg19 settings
        double m_dfTVLossWeight = 0;              // 0.01 to smooth out result -or- 0 to disable.
        double m_dfStyleDataScale1 = 0.0001;      // 0.0001
        double m_dfStyleDataScale2 = 1;           // 1
        double m_dfContentDataScale = 0.0001;     // 0.0001
        double m_dfContentLossScale = 0.0001;     // 0.0001 to 1 (larger make image granier)
        CancelEvent m_evtCancel;
        DataTransformer<T> m_transformer = null;
        TransformationParameter m_transformationParam;
        PersistCaffe<T> m_persist;
        NetParameter m_param;
        byte[] m_rgWeights = null;
        Dictionary<string, Dictionary<string, double>> m_rgLayers = new Dictionary<string, Dictionary<string, double>>();
        List<string> m_rgstrUsedLayers = new List<string>();
        List<double> m_rgMeanValues = new List<double>();
        SolverParameter.SolverType m_solverType = SolverParameter.SolverType.LBFGS;
        double m_dfLearningRate = 1.0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event used to abort processing.</param>
        /// <param name="strModelType">Specifies the model type: 'vgg19', 'vgg16'</param>
        /// <param name="strModel">Specifies the network model to use.</param>
        /// <param name="rgWeights">Optionally, specifies the weights to use (or <i>null</i> to ignore).</param>
        /// <param name="bCaffeModel">Specifies whether or not the weights are in the caffe (<i>true</i>) or mycaffe (<i>false</i>) format.</param>
        /// <param name="solverType">Optionally, specifies the solver type to use (default = LBFGS).</param>
        /// <param name="dfLearningRate">Optionally, specifies the solver learning rate (default = 1.0).</param>
        public NeuralStyleTransfer(CudaDnn<T> cuda, Log log, CancelEvent evtCancel, string strModelType, string strModel, byte[] rgWeights, bool bCaffeModel, SolverParameter.SolverType solverType = SolverParameter.SolverType.LBFGS, double dfLearningRate = 1.0)
        {
            m_cuda = cuda;
            m_log = log;
            m_evtCancel = evtCancel;
            m_rgWeights = rgWeights;
            m_solverType = solverType;
            m_dfLearningRate = dfLearningRate;

            if (m_evtCancel != null)
                m_evtCancel.Reset();

            RawProto proto = RawProto.Parse(strModel);
            m_param = NetParameter.FromProto(proto);

            add_input_layer(m_param);
            m_rgstrUsedLayers = load_layers(strModelType);
            prune(m_param, m_rgstrUsedLayers);
            add_gram_layers(m_param);

            m_transformationParam = new TransformationParameter();
            m_transformationParam.color_order = (bCaffeModel) ? TransformationParameter.COLOR_ORDER.BGR : TransformationParameter.COLOR_ORDER.RGB;
            m_transformationParam.scale = 1.0;
            m_transformationParam.mean_value = m_rgMeanValues;

            m_persist = new PersistCaffe<T>(m_log, false);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
        }

        private void add_input_layer(NetParameter p)
        {
            List<int> rgDelIdx = new List<int>();
            LayerParameter data_param = null;
            LayerParameter input = null;

            for (int i = 0; i < p.layer.Count; i++)
            {
                if (p.layer[i].type == LayerParameter.LayerType.DATA)
                {
                    if (data_param == null)
                        data_param = p.layer[i];

                    rgDelIdx.Add(i);
                }
                else if (p.layer[i].type == LayerParameter.LayerType.INPUT)
                {
                    input = p.layer[i];
                }
            }

            for (int i = rgDelIdx.Count - 1; i >= 0; i--)
            {
                p.layer.RemoveAt(rgDelIdx[i]);
            }

            if (input == null)
            {
                input = new LayerParameter(LayerParameter.LayerType.INPUT);
                int nH = 224;
                int nW = 224;
                input.input_param.shape.Add(new BlobShape(1, 3, nH, nW));
                input.name = data_param.name;
                input.top.Add("input1");

                p.layer.Insert(0, input);
            }
            else
            {
                input.name = "input1";
            }
        }

        private List<string> load_layers(string strName)
        {
            m_rgLayers = new Dictionary<string, Dictionary<string, double>>();
            Dictionary<string, double> rgContent = new Dictionary<string, double>();
            Dictionary<string, double> rgStyle = new Dictionary<string, double>();

            switch (strName)
            {
                case "vgg19":
                case "vgg16":
                    rgContent.Add("conv4_2", 1);
                    rgStyle.Add("conv1_1", 0.2);
                    rgStyle.Add("conv2_1", 0.2);
                    rgStyle.Add("conv3_1", 0.2);
                    rgStyle.Add("conv4_1", 0.2);
                    rgStyle.Add("conv5_1", 0.2);
                    // mean is taken from gist.github.com/ksimonyan/3785162f95cd2d5fee77
                    m_rgMeanValues = new List<double>() { 103.939, 116.779, 123.68 };
                    break;

                case "googlenet":
                    rgContent.Add("conv2/3x3", 2e-4);
                    rgContent.Add("inception_3a/output", 1 - 2e-4);
                    rgStyle.Add("conv1/7x7_s2", 0.2);
                    rgStyle.Add("conv2/3x3", 0.2);
                    rgStyle.Add("inception_3a/output", 0.2);
                    rgStyle.Add("inception_4a/output", 0.2);
                    rgStyle.Add("inception_5a/output", 0.2);
                    m_rgMeanValues = new List<double>() { 104, 117, 123 };
                    break;

                case "caffenet":
                    rgContent.Add("conv4", 1);
                    rgStyle.Add("conv1", 0.2);
                    rgStyle.Add("conv2", 0.2);
                    rgStyle.Add("conv3", 0.2);
                    rgStyle.Add("conv4", 0.2);
                    rgStyle.Add("conv5", 0.2);
                    break;

                default:
                    throw new Exception("Model '" + strName + "' is not supported.");
            }

            m_rgLayers.Add("content", rgContent);
            m_rgLayers.Add("style", rgStyle);

            List<string> rgstrUsedLayers = new List<string>();

            foreach (KeyValuePair<string, double> kv in rgContent)
            {
                rgstrUsedLayers.Add(kv.Key);
            }

            // Add the gram layers
            Dictionary<string, double> rgGram = new Dictionary<string, double>();
            foreach (KeyValuePair<string, double> kv in rgStyle)
            {
                rgstrUsedLayers.Add(kv.Key);
                rgGram.Add("gram_" + kv.Key, kv.Value);
            }

            m_rgLayers.Add("gram", rgGram);

            // Add the input layers
            Dictionary<string, double> rgInput = new Dictionary<string, double>();
            foreach (KeyValuePair<string, double> kv in rgContent)
            {
                rgInput.Add(kv.Key, kv.Value);
            }

            foreach (KeyValuePair<string, double> kv in rgGram)
            {
                rgInput.Add(kv.Key, kv.Value);
            }

            m_rgLayers.Add("input", rgInput);

            rgstrUsedLayers.Sort();

            return rgstrUsedLayers;
        }

        private void prune(NetParameter p, List<string> rgUsedLayers)
        {
            int nPruneFrom = -1;

            // We assume that all layers after the used layers are not useful.
            for (int i = 0; i < p.layer.Count; i++)
            {
                for (int j = 0; j < p.layer[i].top.Count; j++)
                {
                    bool bIsUsed = rgUsedLayers.Contains(p.layer[i].top[j]);

                    if (nPruneFrom >= 0 && bIsUsed)
                    {
                        nPruneFrom = -1;
                        break;
                    }
                    else if (nPruneFrom < 0 && !bIsUsed)
                    {
                        nPruneFrom = i;
                    }
                }
            }

            if (nPruneFrom >= 0)
            {
                for (int i = p.layer.Count - 1; i >= nPruneFrom; i--)
                {
                    m_log.WriteLine("Pruning layer '" + p.layer[i].name);
                    p.layer.RemoveAt(i);
                }
            }
        }

        private void add_gram_layers(NetParameter p)
        {
            List<KeyValuePair<string, double>> lstStyle = m_rgLayers["style"].ToList();
            List<KeyValuePair<string, double>> lstGram = m_rgLayers["gram"].ToList();

            for (int i=0; i<lstStyle.Count; i++)
            {
                LayerParameter layer = new LayerParameter(LayerParameter.LayerType.GRAM);
                string strStyle = lstStyle[i].Key;
                string strGram = lstGram[i].Key;

                layer.name = strGram;

                layer.bottom.Add(strStyle);
                layer.top.Add(strGram);
                layer.gram_param.alpha = m_dfStyleDataScale1;
                layer.gram_param.disable_scaling_on_gradient = true;
                layer.gram_param.beta = m_dfStyleDataScale2;

                p.layer.Add(layer);
            }
        }

        private double get_style_scale(Blob<T> b)
        {
            double df1 = b.shape(0);
            df1 = Math.Pow(df1, -2);
            double df2 = b.count(1);
            df2 = Math.Pow(df2, -2);

            double dfC = (df1 * df2);

            return dfC / 4;
        }

        private double get_content_scale(Blob<T> b)
        {
            return m_dfContentLossScale;
        }

        private void prepare_data_blob(Net<T> net, Bitmap bmp)
        {
            List<int> rgDataShape = new List<int>() { 1, 3, bmp.Height, bmp.Width };
            m_transformer = new DataTransformer<T>(m_log, m_transformationParam, Phase.TEST, 3, bmp.Height, bmp.Width);

            Blob<T> data = net.blob_by_name("data");
            data.Reshape(rgDataShape);
            data.mutable_cpu_data = m_transformer.Transform(ImageData.GetImageData(bmp, 3, false, -1));
        }

        private void prepare_input_param(Net<T> net, Bitmap bmp)
        {
            List<int> rgDataShape = new List<int>() { 1, 3, bmp.Height, bmp.Width };
            m_transformer = new DataTransformer<T>(m_log, m_transformationParam, Phase.TEST, 3, bmp.Height, bmp.Width);

            Blob<T> data = net.param_by_name("input1");
            data.Reshape(rgDataShape);
            data.mutable_cpu_data = m_transformer.Transform(ImageData.GetImageData(bmp, 3, false, -1));
        }

        private Bitmap save(Net<T> net)
        {
            Blob<T> blob = net.param_by_name("input1");
            Datum d = m_transformer.UnTransform(blob);
            return ImageData.GetImage(d);
        }

        /// <summary>
        /// Process the content image by applying the style to it that was learned from the style image.
        /// </summary>
        /// <param name="bmpStyle">Specifies the image used to train the what style to apply to the content.</param>
        /// <param name="bmpContent">Specifies the content image to which the style is to be applied.</param>
        /// <param name="nIterations">Specifies the number of training iterations.</param>
        /// <param name="strResultDir">Optionally, specifies an output directory where intermediate images are stored.</param>
        /// <param name="nIntermediateOutput">Optionally, specifies how often to output an intermediate image.</param>
        /// <param name="dfTvLoss">Optionally, specifies the TV-Loss weight for smoothing (default = 0, which disables this loss).</param>
        /// <returns>The resulting image is returned.</returns>
        public Bitmap Process(Bitmap bmpStyle, Bitmap bmpContent, int nIterations, string strResultDir = null, int nIntermediateOutput = -1, double dfTvLoss = 0)
        {
            Solver<T> solver = null;
            Net<T> net = null;
            BlobCollection<T> colContentActivations = new BlobCollection<T>();
            BlobCollection<T> colGramActivations = new BlobCollection<T>();
            double dfLoss;

            try
            {
                m_dfTVLossWeight = dfTvLoss;
                m_nIterations = nIterations;

                if (bmpStyle.Width != bmpContent.Width ||
                    bmpStyle.Height != bmpContent.Height)
                    bmpStyle = ImageTools.ResizeImage(bmpStyle, bmpContent.Width, bmpContent.Height);

                m_log.WriteLine("Creating input network...");
                m_log.Enable = false;
                net = new Net<T>(m_cuda, m_log, m_param, m_evtCancel, null, Phase.TEST);
                m_log.Enable = true;

                if (m_rgWeights != null)
                    net.LoadWeights(m_rgWeights, m_persist);

                //-----------------------------------------
                //  Get style and content activations.
                //-----------------------------------------

                prepare_data_blob(net, bmpStyle);
                net.Forward(out dfLoss);

                foreach (KeyValuePair<string, double> kvGram in m_rgLayers["gram"])
                {
                    string strGram = kvGram.Key;
                    Blob<T> blobGram = net.blob_by_name(strGram);
                    colGramActivations.Add(blobGram.Clone());
                }

                prepare_data_blob(net, bmpContent);
                net.Forward(out dfLoss);

                foreach (KeyValuePair<string, double> kvContent in m_rgLayers["content"])
                {
                    string strContent = kvContent.Key;
                    Blob<T> blobContent = net.blob_by_name(strContent);
                    colContentActivations.Add(blobContent.Clone());
                }


                //-----------------------------------------
                //  Prepare the network by adding new layers.
                //-----------------------------------------

                NetParameter net_param = m_param;

                foreach (KeyValuePair<string, double> kvInput in m_rgLayers["input"])
                {
                    string strName = kvInput.Key;
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.INPUT);
                    p.name = "input_" + strName;
                    p.top.Add(p.name);

                    Blob<T> blob = net.blob_by_name(strName);
                    p.input_param.shape.Add(new BlobShape(blob.shape()));

                    net_param.layer.Add(p);
                }

                foreach (KeyValuePair<string, double> kvContent in m_rgLayers["content"])
                {
                    string strName = kvContent.Key;
                    string strScale1 = "input_" + strName;
                    string strScale2 = strName;

                    if (m_dfContentDataScale != 1.0)
                    {
                        strScale1 += "b";
                        LayerParameter ps1 = new LayerParameter(LayerParameter.LayerType.SCALAR);
                        ps1.scalar_param.value = m_dfContentDataScale;
                        ps1.scalar_param.operation = ScalarParameter.ScalarOp.MUL;
                        ps1.scalar_param.passthrough_gradient = true;
                        ps1.bottom.Add("input_" + strName);
                        ps1.top.Add(strScale1);

                        net_param.layer.Add(ps1);

                        strScale2 += "b";
                        LayerParameter ps2 = new LayerParameter(LayerParameter.LayerType.SCALAR);
                        ps2.scalar_param.value = m_dfContentDataScale;
                        ps2.scalar_param.operation = ScalarParameter.ScalarOp.MUL;
                        ps2.scalar_param.passthrough_gradient = true;
                        ps2.bottom.Add(strName);
                        ps2.top.Add(strScale2);

                        net_param.layer.Add(ps2);
                    }

                    LayerParameter event_param = new LayerParameter(LayerParameter.LayerType.EVENT);
                    event_param.name = "event_" + strName;
                    event_param.bottom.Add(strScale2);
                    event_param.bottom.Add(strScale1);
                    event_param.top.Add("event_" + strName);

                    net_param.layer.Add(event_param);

                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strName;

                    Blob<T> blobContent = colContentActivations[strName];
                    double dfScale = get_content_scale(blobContent);
                    p.loss_weight.Add(kvContent.Value * dfScale);

                    p.bottom.Add("event_" + strName);
                    p.bottom.Add(strScale1);
                    p.top.Add("loss_" + strName);

                    net_param.layer.Add(p);
                }

                foreach (KeyValuePair<string, double> kvGram in m_rgLayers["gram"].ToList())
                {
                    string strGramName = kvGram.Key;

                    LayerParameter event_param = new LayerParameter(LayerParameter.LayerType.EVENT);
                    event_param.name = "event_" + strGramName;
                    event_param.bottom.Add(strGramName);
                    event_param.bottom.Add("input_" + strGramName);
                    event_param.top.Add("event_" + strGramName);

                    net_param.layer.Add(event_param);

                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strGramName;

                    Blob<T> blobGram = colGramActivations[strGramName];
                    double dfScale = get_style_scale(blobGram);
                    p.loss_weight.Add(kvGram.Value * dfScale);

                    p.bottom.Add("input_" + strGramName);
                    p.bottom.Add("event_" + strGramName);
                    p.top.Add("loss_" + strGramName);

                    net_param.layer.Add(p);
                }

                // Add TV Loss;
                if (m_dfTVLossWeight != 0)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.TV_LOSS);
                    p.name = "loss_tv";

                    double dfWeight = m_dfTVLossWeight;
                    p.loss_weight.Add(dfWeight);

                    p.bottom.Add("data");
                    p.top.Add("loss_tv");

                    net_param.layer.Add(p);
                }

                // Replace InputLayer with ParameterLayer,
                // so that we'll be able to backprop into the image.
                Blob<T> data = net.blob_by_name("data");
                for (int i=0; i<net_param.layer.Count; i++)
                {
                    LayerParameter p = net_param.layer[i];

                    if (p.name == "input1")
                    {
                        net_param.layer[i].SetType(LayerParameter.LayerType.PARAMETER);
                        net_param.layer[i].parameter_param.shape = new BlobShape(data.shape());
                        break;
                    }
                }

                // Disable weights learning.
                List<LayerParameter.LayerType> rgTypes = new List<LayerParameter.LayerType>();
                rgTypes.Add(LayerParameter.LayerType.CONVOLUTION);
                rgTypes.Add(LayerParameter.LayerType.DECONVOLUTION);
                rgTypes.Add(LayerParameter.LayerType.INNERPRODUCT);
                rgTypes.Add(LayerParameter.LayerType.PRELU);
                rgTypes.Add(LayerParameter.LayerType.BIAS);
                rgTypes.Add(LayerParameter.LayerType.EMBED);
                rgTypes.Add(LayerParameter.LayerType.LSTM);
                rgTypes.Add(LayerParameter.LayerType.LSTM_SIMPLE);
                rgTypes.Add(LayerParameter.LayerType.RNN);

                foreach (LayerParameter layer in net_param.layer)
                {
                    if (rgTypes.Contains(layer.type))
                    {
                        layer.parameters = new List<ParamSpec>();
                        layer.parameters.Add(new ParamSpec(0, 0));
                        layer.parameters.Add(new ParamSpec(0, 0));
                    }
                }

                net.Dispose();
                net = null;


                //-----------------------------------------
                //  Create solver and assign inputs.
                //-----------------------------------------

                RawProto proto1 = net_param.ToProto("root");
                string str = proto1.ToString();

                SolverParameter solver_param = new SolverParameter();
                solver_param.display = m_nDisplayEvery;
                solver_param.train_net_param = net_param;
                solver_param.test_iter.Clear();
                solver_param.test_interval = 0;
                solver_param.test_initialization = false;
                solver_param.base_lr = m_dfLearningRate;
                solver_param.type = m_solverType;

                m_log.WriteLine("Creating " + m_solverType.ToString() + " solver with learning rate = " + m_dfLearningRate.ToString() + "...");
                m_log.Enable = false;

                if (m_solverType == SolverParameter.SolverType.LBFGS)
                    solver = new LBFGSSolver<T>(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, m_persist);
                else
                    solver = Solver<T>.Create(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, m_persist);

                m_log.Enable = true;
                solver.OnSnapshot += Solver_OnSnapshot;
                solver.OnTrainingIteration += Solver_OnTrainingIteration;

                foreach (Layer<T> layer in solver.net.layers)
                {
                    if (layer.type == LayerParameter.LayerType.EVENT)
                    {
                        EventLayer<T> eventLayer = layer as EventLayer<T>;
                        eventLayer.OnBackward += EventLayer_OnBackward;
                    }
                }

                prepare_input_param(solver.net, bmpContent);

                foreach (KeyValuePair<string, double> kvContent in m_rgLayers["content"])
                {
                    string strName = kvContent.Key;
                    Blob<T> blobDst = solver.net.blob_by_name("input_" + strName);
                    Blob<T> blobSrc = colContentActivations[strName];
                    blobDst.CopyFrom(blobSrc);
                }

                foreach (KeyValuePair<string, double> kvGram in m_rgLayers["gram"])
                {
                    string strName = kvGram.Key;
                    Blob<T> blobDst = solver.net.blob_by_name("input_" + strName);
                    Blob<T> blobSrc = colGramActivations[strName];
                    blobDst.CopyFrom(blobSrc);
                }

                //-----------------------------------------
                //  Optimize.
                //-----------------------------------------

                int nIterations1 = m_nIterations;
                if (strResultDir != null && nIntermediateOutput > 0)
                    nIterations1 /= nIntermediateOutput;

                if (m_rgWeights != null)
                {
                    Blob<T> blobInput = solver.net.learnable_parameters[0];
                    solver.net.learnable_parameters.RemoveAt(0);
                    solver.net.LoadWeights(m_rgWeights, m_persist);
                    solver.net.learnable_parameters.Insert(0, blobInput);
                }

                if (strResultDir != null)
                {
                    strResultDir = strResultDir.TrimEnd('\\');
                    strResultDir += "\\";
                }

                for (int i = 0; i < nIterations1; i++)
                {
                    if (m_evtCancel.WaitOne(0))
                        break;

                    solver.Step(nIntermediateOutput, TRAIN_STEP.NONE, true, true, true);

                    if (strResultDir != null)
                    {
                        Bitmap bmpTemp = save(solver.net);

                        string strFile = strResultDir + i.ToString() + "_temp.png";
                        if (File.Exists(strFile))
                            File.Delete(strFile);

                        bmpTemp.Save(strFile);
                    }
                }

                Bitmap bmpOutput = save(solver.net);

                return bmpOutput;
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                if (net != null)
                    net.Dispose();

                if (solver != null)
                    solver.Dispose();

                colGramActivations.Dispose();
                colContentActivations.Dispose();
            }
        }

        private void EventLayer_OnBackward(object sender, BackwardArgs<T> e)
        {
            int nCount = e.BottomVec[0].count();
            long hTopDiff0 = e.TopVec[0].mutable_gpu_diff;
            long hBottomData1 = e.BottomVec[1].gpu_data;
            long hBottomDiff1 = e.BottomVec[1].mutable_gpu_diff;
            long hBottomDiff = e.BottomVec[0].mutable_gpu_diff;

            m_cuda.sign(nCount, hBottomData1, hBottomDiff1);
            m_cuda.abs(nCount, hBottomDiff1, hBottomDiff1);
            m_cuda.mul(nCount, hBottomDiff1, hTopDiff0, hBottomDiff);
        }

        private void Solver_OnTrainingIteration(object sender, TrainingIterationArgs<T> e)
        {
            m_log.WriteLine("Iteration = " + e.Iteration.ToString() + " - Loss = " + e.SmoothedLoss.ToString());

            if (double.IsNaN(e.Loss))
            {
                m_log.WriteError(new Exception("Loss = NAN!"));
                m_evtCancel.Set();
                return;
            }

            if (double.IsInfinity(e.Loss))
            {
                m_log.WriteError(new Exception("Loss = Infinity!"));
                m_evtCancel.Set();
                return;
            }
        }

        private void Solver_OnSnapshot(object sender, SnapshotArgs e)
        {
        }
    }
}
