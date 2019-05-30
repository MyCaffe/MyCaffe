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
    /// @see [fzliu/style-transfer Github](https://github.com/fzliu/style-transfer/blob/master/style.py) by Frank Liu and Dylan Paiton, 2015.
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// @see [Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398) by Raymond Yuan, Medium, 2018
    /// @see [Neural Artistic Style Transfer: A Comprehensive Look](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199) by Shubhang Desai, Medium, 2017
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
        int m_nDefaultMaxImageSize = 840;
        string m_strDataBlobName = "data";
        Solver<T> m_solver = null;
        int m_nIntermediateOutput = 0;
        int m_nPartialIteration = 0;
        int m_nPartialIterations1 = 0;
        Net<T> m_netShare = null;
        bool m_bUsingSharedNet = false;
        int m_nLBFGSCorrections = 100;
        bool m_bAllowHalfSize = false;
        bool m_bAllowHalfSizeOnGram = true;
        bool m_bAllowHalfSizeOnEvent = true;
        bool m_bAllowHalfSizeOnLoss = true;
        bool m_bAllowHalfSizeOnScalar = true;
        long m_hWorkspaceData = 0;  
        ulong m_lWorkspaceSize = 0;

        /// <summary>
        /// Specifies the event fired after producing intermediate output (e.g. when m_nIntermediateOutput > 0)
        /// </summary>
        public event EventHandler<NeuralStyleIntermediateOutputArgs> OnIntermediateOutput;

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
        /// <param name="nLBFGSCorrections">Optionally, specifies the number of LBFGS corrections to use (default = 100, only applies when using LBFGS solver).</param>
        /// <param name="dfDataScale">Optionally, specifies the data scaling factor (default = 1.0).</param>
        /// <param name="netShare">Optionally, specifies a net to share.</param>
        public NeuralStyleTransfer(CudaDnn<T> cuda, Log log, CancelEvent evtCancel, string strModelType, string strModel, byte[] rgWeights, bool bCaffeModel, SolverParameter.SolverType solverType = SolverParameter.SolverType.LBFGS, double dfLearningRate = 1.5, int nLBFGSCorrections = 100, double dfDataScale = 1.0, Net<T> netShare = null)
        {
            evtCancel.Reset();

            m_log = log;
            m_evtCancel = evtCancel;
            m_rgWeights = rgWeights;
            m_solverType = solverType;
            m_dfLearningRate = dfLearningRate;
            m_nLBFGSCorrections = nLBFGSCorrections;

            setupNetShare(netShare, cuda);

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
            m_transformationParam.scale = dfDataScale;
            m_transformationParam.mean_value = m_rgMeanValues;

            m_persist = new PersistCaffe<T>(m_log, false);
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event used to abort processing.</param>
        /// <param name="rgLayers">Specifies the layers along with their style and content weights.</param>
        /// <param name="strModelDesc">Specifies the network model descriptor to use.</param>
        /// <param name="rgWeights">Optionally, specifies the weights to use (or <i>null</i> to ignore).</param>
        /// <param name="bCaffeModel">Specifies whether or not the weights are in the caffe (<i>true</i>) or mycaffe (<i>false</i>) format.</param>
        /// <param name="solverType">Optionally, specifies the solver type to use (default = LBFGS).</param>
        /// <param name="dfLearningRate">Optionally, specifies the solver learning rate (default = 1.0).</param>
        /// <param name="nMaxImageSize">Optionally, specifies the default maximum image size (default = 840).</param>
        /// <param name="nLBFGSCorrections">Optionally, specifies the LBFGS Corrections (only used when using the LBFGS solver, default = 100).</param>
        /// <param name="dfDataScale">Optionally, specifies the data scaling factor (default = 1.0).</param>
        /// <param name="netShare">Optionally, specifies a net to share.</param>
        public NeuralStyleTransfer(CudaDnn<T> cuda, Log log, CancelEvent evtCancel, Dictionary<string, Tuple<double, double>> rgLayers, string strModelDesc, byte[] rgWeights, bool bCaffeModel, SolverParameter.SolverType solverType = SolverParameter.SolverType.LBFGS, double dfLearningRate = 1.0, int nMaxImageSize = 840, int nLBFGSCorrections = 100, double dfDataScale = 1.0, Net<T> netShare = null)
        {
            m_log = log;
            m_evtCancel = evtCancel;
            m_rgWeights = rgWeights;
            m_solverType = solverType;
            m_dfLearningRate = dfLearningRate;
            m_nDefaultMaxImageSize = nMaxImageSize;
            m_nLBFGSCorrections = nLBFGSCorrections;

            setupNetShare(netShare, cuda);

            if (m_evtCancel != null)
                m_evtCancel.Reset();

            RawProto proto = RawProto.Parse(strModelDesc);
            m_param = NetParameter.FromProto(proto);

            Dictionary<string, double> rgStyle = new Dictionary<string, double>();
            Dictionary<string, double> rgContent = new Dictionary<string, double>();

            foreach (KeyValuePair<string, Tuple<double, double>> kv in rgLayers)
            {
                if (kv.Value.Item1 != 0)
                    rgStyle.Add(kv.Key, kv.Value.Item1);

                if (kv.Value.Item2 != 0)
                    rgContent.Add(kv.Key, kv.Value.Item2);
            }

            add_input_layer(m_param);
            m_rgstrUsedLayers = load_layers(rgStyle, rgContent);
            prune(m_param, m_rgstrUsedLayers);
            add_gram_layers(m_param);

            m_transformationParam = new TransformationParameter();
            m_transformationParam.color_order = (bCaffeModel) ? TransformationParameter.COLOR_ORDER.BGR : TransformationParameter.COLOR_ORDER.RGB;
            m_transformationParam.scale = dfDataScale;
            m_transformationParam.mean_value = m_rgMeanValues;

            m_persist = new PersistCaffe<T>(m_log, false);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            m_evtCancel.Set();

            if (m_solver != null)
            {
                m_solver.Dispose();
                m_solver = null;
            }

            if (m_hWorkspaceData != 0)
            {
                m_cuda.FreeMemory(m_hWorkspaceData);
                m_hWorkspaceData = 0;
                m_lWorkspaceSize = 0;
            }
        }

        /// <summary>
        /// Setup which layers are allowed to use half-sized memory when their convolution counterparts use it.
        /// </summary>
        /// <param name="bAllowHs">Allow half-size memory.</param>
        /// <param name="bAllowOnGram">Allow half-size on the gram layers.</param>
        /// <param name="bAllowOnEvent">Allow half-size on the event layers.</param>
        /// <param name="bAllowOnLoss">Allow half-size on the loss layers.</param>
        /// <param name="bAllowOnScalar">Allow half-size on the scalar layers.</param>
        public void SetupHalfSize(bool bAllowHs, bool bAllowOnGram, bool bAllowOnEvent, bool bAllowOnLoss, bool bAllowOnScalar)
        {
            m_bAllowHalfSize = bAllowHs;
            m_bAllowHalfSizeOnEvent = bAllowOnEvent;
            m_bAllowHalfSizeOnGram = bAllowOnGram;
            m_bAllowHalfSizeOnLoss = bAllowOnLoss;
            m_bAllowHalfSizeOnScalar = bAllowOnScalar;

            if (!bAllowHs || !m_bAllowHalfSizeOnGram)
            {
                List<string> rgstrHalfLayers = new List<string>();

                foreach (LayerParameter layer1 in m_param.layer)
                {
                    if (layer1.use_halfsize)
                    {
                        if (layer1.name.Contains("gram"))
                            layer1.use_halfsize = false;
                        else
                            rgstrHalfLayers.Add(layer1.name);
                    }
                }

                if (!bAllowHs && rgstrHalfLayers.Count > 0)
                {
                    string strErr = "Half-sized memory not supported!  Disable half-size in the following layers: " + Utility.ToString<string>(rgstrHalfLayers);
                    m_log.FAIL(strErr);                  
                }
            }
        }

        private void setupNetShare(Net<T> net, CudaDnn<T> cuda)
        {
            if (net == null)
            {
                m_cuda = cuda;
                return;
            }

            int nNetDeviceId = net.Cuda.GetDeviceID();
            int nCudaDeviceId = cuda.GetDeviceID();

            if (nNetDeviceId != nCudaDeviceId)
            {
                m_cuda = cuda;
                return;
            }

            m_netShare = net;
            m_cuda = m_netShare.Cuda;
            m_bUsingSharedNet = true;

            return;
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
                    {
                        data_param = p.layer[i];
                        m_strDataBlobName = data_param.top[0];
                    }

                    rgDelIdx.Add(i);
                }
                else if (p.layer[i].type == LayerParameter.LayerType.INPUT)
                {
                    input = p.layer[i];
                    m_strDataBlobName = input.top[0];
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
                input.name = "input1";
                input.top.Add(m_strDataBlobName);

                p.layer.Insert(0, input);
            }
            else
            {
                input.name = "input1";
            }
        }

        private List<string> load_layers(string strName)
        {
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

            return load_layers(rgStyle, rgContent);
        }

        private List<string> load_layers(Dictionary<string, double> rgStyle, Dictionary<string, double> rgContent)
        {
            m_rgLayers = new Dictionary<string, Dictionary<string, double>>();
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
                bool bUseHalfSize = false;
                LayerParameter layer = new LayerParameter(LayerParameter.LayerType.GRAM);
                string strStyle = lstStyle[i].Key;
                string strGram = lstGram[i].Key;

                foreach (LayerParameter layer1 in p.layer)
                {
                    if (layer1.type == LayerParameter.LayerType.CONVOLUTION)
                    {
                        if (layer1.top.Contains(strStyle))
                        {
                            bUseHalfSize = layer1.use_halfsize;
                            break;
                        }
                    }
                }

                layer.name = strGram;

                layer.bottom.Add(strStyle);
                layer.top.Add(strGram);
                layer.gram_param.alpha = m_dfStyleDataScale1;
                layer.gram_param.disable_scaling_on_gradient = true;
                layer.gram_param.beta = m_dfStyleDataScale2;
                layer.use_halfsize = (bUseHalfSize && m_bAllowHalfSizeOnGram);

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

            Blob<T> data = net.blob_by_name(m_strDataBlobName);
            data.Reshape(rgDataShape, data.HalfSize);
            data.mutable_cpu_data = m_transformer.Transform(ImageData.GetImageData(bmp, 3, false, -1));
        }

        private void prepare_input_param(Net<T> net, Bitmap bmp)
        {
            List<int> rgDataShape = new List<int>() { 1, 3, bmp.Height, bmp.Width };
            m_transformer = new DataTransformer<T>(m_log, m_transformationParam, Phase.TEST, 3, bmp.Height, bmp.Width);

            Blob<T> data = net.param_by_name("input1");
            data.Reshape(rgDataShape, data.HalfSize);
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
        /// <param name="nIntermediateOutput">Optionally, specifies how often to output an intermediate image.</param>
        /// <param name="dfTvLoss">Optionally, specifies the TV-Loss weight for smoothing (default = 0, which disables this loss).</param>
        /// <param name="nMaxSize">Optionally, specifies a maximum image size override (default = -1, which uses the default).</param>
        /// <param name="bEnablePartialSolution">Optionally, specifies to run the process only up through the next intermediate image.  Subsequent calls to ProcessNext moves to the next intermediate image until completed.</param>
        /// <returns>Upon completion the resulting final image is returned, otherwise when using bEnablePartionSolution = <i>true</i>, null is returned.</returns>
        public Bitmap Process(Bitmap bmpStyle, Bitmap bmpContent, int nIterations, int nIntermediateOutput = -1, double dfTvLoss = 0, int nMaxSize = -1, bool bEnablePartialSolution = false)
        {
            Solver<T> solver = null;
            Net<T> net = null;
            BlobCollection<T> colContentActivations = new BlobCollection<T>();
            BlobCollection<T> colGramActivations = new BlobCollection<T>();
            double dfLoss;
            bool bDone = true;

            try
            {
                m_dfTVLossWeight = dfTvLoss;
                m_nIterations = nIterations;

                if (nMaxSize == -1)
                    nMaxSize = m_nDefaultMaxImageSize;

                if (bmpContent.Width > nMaxSize ||
                    bmpContent.Height > nMaxSize)
                {
                    double dfAspectRatio = (double)bmpContent.Height / (double)bmpContent.Width;
                    int nWidth = nMaxSize;
                    int nHeight = (int)(nMaxSize * dfAspectRatio);
                    bmpContent = ImageTools.ResizeImage(bmpContent, nWidth, nHeight);
                }

                if (bmpStyle.Width != bmpContent.Width ||
                    bmpStyle.Height != bmpContent.Height)
                    bmpStyle = ImageTools.ResizeImage(bmpStyle, bmpContent.Width, bmpContent.Height);

                m_log.WriteLine("Creating input network...");
                m_log.Enable = false;
                net = new Net<T>(m_cuda, m_log, m_param, m_evtCancel, null, Phase.TEST, null, m_netShare, net_OnGetWorkspace, net_OnSetWorkspace);
                m_log.Enable = true;

                if (m_rgWeights != null && !m_bUsingSharedNet)
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
                    p.use_halfsize = blob.HalfSize;

                    net_param.layer.Add(p);
                }

                foreach (KeyValuePair<string, double> kvContent in m_rgLayers["content"])
                {
                    string strName = kvContent.Key;
                    string strScale1 = "input_" + strName;
                    string strScale2 = strName;
                    Blob<T> blobContent = colContentActivations[strName];

                    if (m_dfContentDataScale != 1.0)
                    {
                        strScale1 += "b";
                        LayerParameter ps1 = new LayerParameter(LayerParameter.LayerType.SCALAR);
                        ps1.scalar_param.value = m_dfContentDataScale;
                        ps1.scalar_param.operation = ScalarParameter.ScalarOp.MUL;
                        ps1.scalar_param.passthrough_gradient = true;
                        ps1.use_halfsize = (blobContent.HalfSize && m_bAllowHalfSizeOnScalar);
                        ps1.bottom.Add("input_" + strName);
                        ps1.top.Add(strScale1);

                        net_param.layer.Add(ps1);

                        strScale2 += "b";
                        LayerParameter ps2 = new LayerParameter(LayerParameter.LayerType.SCALAR);
                        ps2.scalar_param.value = m_dfContentDataScale;
                        ps2.scalar_param.operation = ScalarParameter.ScalarOp.MUL;
                        ps2.scalar_param.passthrough_gradient = true;
                        ps2.use_halfsize = (blobContent.HalfSize && m_bAllowHalfSizeOnScalar);
                        ps2.bottom.Add(strName);
                        ps2.top.Add(strScale2);

                        net_param.layer.Add(ps2);
                    }

                    LayerParameter event_param = new LayerParameter(LayerParameter.LayerType.EVENT);
                    event_param.name = "event_" + strName;
                    event_param.bottom.Add(strScale2);
                    event_param.bottom.Add(strScale1);
                    event_param.use_halfsize = (blobContent.HalfSize && m_bAllowHalfSizeOnEvent);
                    event_param.top.Add("event_" + strName);

                    net_param.layer.Add(event_param);

                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strName;

                    double dfScale = get_content_scale(blobContent);
                    p.loss_weight.Add(kvContent.Value * dfScale);
                    p.use_halfsize = (blobContent.HalfSize && m_bAllowHalfSizeOnLoss);
                    p.bottom.Add("event_" + strName);
                    p.bottom.Add(strScale1);
                    p.top.Add("loss_" + strName);

                    net_param.layer.Add(p);
                }

                foreach (KeyValuePair<string, double> kvGram in m_rgLayers["gram"].ToList())
                {
                    string strGramName = kvGram.Key;
                    Blob<T> blobGram = colGramActivations[strGramName];

                    LayerParameter event_param = new LayerParameter(LayerParameter.LayerType.EVENT);
                    event_param.name = "event_" + strGramName;
                    event_param.use_halfsize = (blobGram.HalfSize && m_bAllowHalfSizeOnEvent);
                    event_param.bottom.Add(strGramName);
                    event_param.bottom.Add("input_" + strGramName);
                    event_param.top.Add("event_" + strGramName);

                    net_param.layer.Add(event_param);

                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strGramName;
                    p.use_halfsize = (blobGram.HalfSize && m_bAllowHalfSizeOnLoss);

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

                    p.bottom.Add(m_strDataBlobName);
                    p.top.Add("loss_tv");

                    net_param.layer.Add(p);
                }

                // Replace InputLayer with ParameterLayer,
                // so that we'll be able to backprop into the image.
                Blob<T> data = net.blob_by_name(m_strDataBlobName);
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

                SolverParameter solver_param = new SolverParameter();
                solver_param.display = m_nDisplayEvery;
                solver_param.train_net_param = net_param;
                solver_param.test_iter.Clear();
                solver_param.test_interval = 0;
                solver_param.test_initialization = false;
                solver_param.base_lr = m_dfLearningRate;
                solver_param.type = m_solverType;
                solver_param.lbgfs_corrections = m_nLBFGSCorrections;

                m_log.WriteLine("Creating " + m_solverType.ToString() + " solver with learning rate = " + m_dfLearningRate.ToString() + "...");
                m_log.Enable = false;

                if (m_solverType == SolverParameter.SolverType.LBFGS)
                    solver = new LBFGSSolver<T>(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, m_persist, 1, 0, m_netShare, net_OnGetWorkspace, net_OnSetWorkspace);
                else
                    solver = Solver<T>.Create(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, m_persist, 1, 0, m_netShare, net_OnGetWorkspace, net_OnSetWorkspace);

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

                colGramActivations.Dispose();
                colGramActivations = null;

                colContentActivations.Dispose();
                colContentActivations = null;


                //-----------------------------------------
                //  Optimize.
                //-----------------------------------------

                if (nIntermediateOutput <= 0 || nIntermediateOutput == m_nIterations)
                {
                    bEnablePartialSolution = false;
                    nIntermediateOutput = m_nIterations;
                }

                int nIterations1 = m_nIterations / nIntermediateOutput;

                if (m_rgWeights != null && !m_bUsingSharedNet)
                {
                    Blob<T> blobInput = solver.net.learnable_parameters[0];
                    solver.net.learnable_parameters.RemoveAt(0);
                    solver.net.LoadWeights(m_rgWeights, m_persist);
                    solver.net.learnable_parameters.Insert(0, blobInput);
                }

                if (bEnablePartialSolution)
                {
                    m_solver = solver;
                    m_nPartialIteration = 0;
                    m_nPartialIterations1 = nIterations1;
                    m_nIntermediateOutput = nIntermediateOutput;
                    bDone = false;
                    return null;
                }
               
                for (int i = 0; i < nIterations1; i++)
                {
                    if (m_evtCancel.WaitOne(0))
                        break;

                    solver.Step(nIntermediateOutput, TRAIN_STEP.NONE, true, true, true);

                    if (!m_evtCancel.WaitOne(0))
                    {
                        if (OnIntermediateOutput != null && nIntermediateOutput > 0 && i < nIterations1 - 1)
                        {
                            Bitmap bmpTemp = save(solver.net);
                            double dfPct = (double)i / (double)nIterations1;
                            OnIntermediateOutput(this, new NeuralStyleIntermediateOutputArgs(i, bmpTemp, dfPct));
                            bmpTemp.Dispose();
                        }
                    }
                }

                return save(solver.net);
            }
            catch (Exception excpt)
            {
                if (solver != null)
                {
                    m_solver = null;
                    solver.Dispose();
                }

                throw excpt;
            }
            finally
            {
                if (net != null)
                    net.Dispose();

                if (colGramActivations != null)
                    colGramActivations.Dispose();

                if (colContentActivations != null)
                    colContentActivations.Dispose();

                if (bDone)
                {
                    if (solver != null)
                        solver.Dispose();
                }
            }
        }

        private void net_OnSetWorkspace(object sender, WorkspaceArgs e)
        {
            if (e.Size < m_lWorkspaceSize)
                return;

            m_lWorkspaceSize = e.Size;
            m_cuda.DisableGhostMemory();

            if (m_hWorkspaceData != 0)
                m_cuda.FreeMemory(m_hWorkspaceData);

            m_hWorkspaceData = m_cuda.AllocMemory((long)m_lWorkspaceSize);
            m_cuda.ResetGhostMemory();
        }

        private void net_OnGetWorkspace(object sender, WorkspaceArgs e)
        {
            e.Data = m_hWorkspaceData;
            e.Size = m_lWorkspaceSize;
        }

        /// <summary>
        /// Process the next partial part of the solution.  This function is only valid after calling Process with bEnablePartialSolution = <i>true</i>.
        /// </summary>
        /// <param name="bmpIntermediate">Returns the intermediate image, if one was created.</param>
        /// <param name="nIntermediateIdx">Returns the intermediate index for the image.</param>
        /// <returns>Upon completion, the final Bitmap is returned, otherwise <i>null</i> is returned.</returns>
        public Bitmap ProcessNext(out Bitmap bmpIntermediate, out int nIntermediateIdx)
        {
            try
            {
                bmpIntermediate = null;
                nIntermediateIdx = m_nPartialIteration * m_nIntermediateOutput;

                if (m_solver == null)
                    throw new Exception("To run the next in process, the solver cannot be null!  You must call Process first.");

                m_solver.Step(m_nIntermediateOutput, TRAIN_STEP.NONE, true, true, true);

                if (m_evtCancel.WaitOne(0))
                    return null;

                m_nPartialIteration++;

                if (m_nIntermediateOutput > 0 && m_nPartialIteration < m_nPartialIterations1)
                {
                    bmpIntermediate = save(m_solver.net);

                    if (OnIntermediateOutput != null)
                    {
                        double dfPct = (double)m_nPartialIteration / (double)m_nPartialIterations1;
                        OnIntermediateOutput(this, new NeuralStyleIntermediateOutputArgs(m_nPartialIteration * m_nIntermediateOutput, bmpIntermediate, dfPct));
                    }
                }

                if (m_nPartialIteration < m_nPartialIterations1)
                    return null;

                return save(m_solver.net);
            }
            catch (Exception excpt)
            {               
                throw excpt;
            }
            finally
            {
                if (m_nPartialIteration == m_nPartialIterations1)
                {
                    if (m_solver != null)
                    {
                        m_solver.Dispose();
                        m_solver = null;
                    }
                }
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
                m_evtCancel.Set();
                m_log.FAIL("Loss = NAN!");
                return;
            }

            if (double.IsInfinity(e.Loss))
            {
                m_evtCancel.Set();
                m_log.FAIL("Loss = Infinity!");
                return;
            }
        }

        private void Solver_OnSnapshot(object sender, SnapshotArgs e)
        {
        }

        /// <summary>
        /// The CreateConfigurationString function packs all deep draw settings into a configuration string.
        /// </summary>
        /// <param name="strSolver">Specifies the type of solver to use.</param>
        /// <param name="dfLearningRate">Specifies the learning rate to use with the solver.</param>
        /// <param name="nMaxImageSize">Specifies the maximum image size to use.</param>
        /// <param name="nIterations">Specifies the number of iterations to run.</param>
        /// <param name="nIntermediateIterations">Specifies how often to output intermediate images if any (a value of 0 disables intermediate output).</param>
        /// <param name="rgWts">Specifies the layers to use and their weights for style and content.</param>
        /// <param name="rgGpuID">Specifies the GPUIDs on which to run the Neural Style.</param>
        /// <param name="nLBFGSCorrections">Specifies the LBFGS corrections to use, only applies when using the LBFGS Solver.</param>
        /// <param name="dfDataScale">Specifies the data scale (default = 1.0).</param>
        /// <param name="bAllowHs">Specivies to allow half sized memory.</param>
        /// <param name="bAllowHsGram">Specifies to allow half sized memory on gram layers.</param>
        /// <param name="bAllowHsEvent">Specifies to allow half sized memory on event layers.</param>
        /// <param name="bAllowHsScalar">Specifies to allow half sized memory on scalar layers.</param>
        /// <param name="bAllowHsLoss">Specifies to allow half sized memory on loss layers.</param>
        /// <returns>The configuration string is returned.</returns>
        public static string CreateConfigurationString(string strSolver, double dfLearningRate, int nMaxImageSize, int nIterations, int nIntermediateIterations, Dictionary<string, Tuple<double, double>> rgWts, List<int> rgGpuID, int nLBFGSCorrections, double dfDataScale, bool bAllowHs, bool bAllowHsGram, bool bAllowHsEvent, bool bAllowHsScalar, bool bAllowHsLoss)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("solver", strSolver);
            rgChildren.Add("learning_rate", dfLearningRate);
            rgChildren.Add("max_image_size", nMaxImageSize);
            rgChildren.Add("iterations", nIterations);
            rgChildren.Add("intermediate_iterations", nIntermediateIterations);

            RawProtoCollection rgLayerWt = new RawProtoCollection();
            foreach (KeyValuePair<string, Tuple<double, double>> kv in rgWts)
            {
                RawProtoCollection layer = new RawProtoCollection();
                layer.Add("name", kv.Key);
                layer.Add("style_wt", kv.Value.Item1);
                layer.Add("content_wt", kv.Value.Item2);

                rgLayerWt.Add(new RawProto("layer", "", layer));
            }

            rgChildren.Add(rgLayerWt);

            RawProtoCollection gpus = new RawProtoCollection();
            foreach (int nGpuID in rgGpuID)
            {
                gpus.Add("gpuid", nGpuID.ToString());
            }

            rgChildren.Add(gpus);
            rgChildren.Add("lbfgs_corrections", nLBFGSCorrections);
            rgChildren.Add("data_scale", dfDataScale);
            rgChildren.Add("allow_hs", bAllowHs);
            rgChildren.Add("allow_hs_gram", bAllowHsGram);
            rgChildren.Add("allow_hs_event", bAllowHsEvent);
            rgChildren.Add("allow_hs_scalar", bAllowHsScalar);
            rgChildren.Add("allow_hs_loss", bAllowHsLoss);

            RawProto proto = new RawProto("root", "", rgChildren);

            return proto.ToString();
        }

        /// <summary>
        /// The ParseConfigurationString method parses a deep draw configuration string into the actual settings.
        /// </summary>
        /// <param name="strConfig">Specifies the configuration string to parse.</param>
        /// <param name="strSolver">Returns the solver to use.</param>
        /// <param name="dfLearningRate">Returns the learning rate to use with the solver.</param>
        /// <param name="nMaxImageSize">Returns the maximum image size.</param>
        /// <param name="nIterations">Returns the number of iterations to run.</param>
        /// <param name="nIntermediateIterations">Returns how often to output intermediate images if any (a value of 0 disables intermediate output).</param>
        /// <param name="rgGpuID">Returns the list of GPUIDs on which to run the Neural Style.</param>
        /// <param name="nLBFGSCorrections">Returns the LBFGS corrections to use, only applies when using the LBFGS Solver.</param>
        /// <param name="dfDataScale">Returns the data scale to use, default = 1.0.</param>
        /// <param name="bAllowHs">Returns whether or not half size memory is allowed.</param>
        /// <param name="bAllowHsGram">Returns whether or not to allow half size memory in the gram layers.</param>
        /// <param name="bAllowHsEvent">Returns whether or not to allow half size memory in the event layers.</param>
        /// <param name="bAllowHsScalar">Returns whether or not to allow half size memory in the scalar layers.</param>
        /// <param name="bAllowHsLoss">Returns whether or not to allow half size memory in the loss layers.</param>
        /// <returns>Returns a list of layers along with their style and content weights.</returns>
        public static Dictionary<string, Tuple<double, double>> ParseConfigurationString(string strConfig, out string strSolver, out double dfLearningRate, out int nMaxImageSize, out int nIterations, out int nIntermediateIterations, out List<int> rgGpuID, out int nLBFGSCorrections, out double dfDataScale, out bool bAllowHs, out bool bAllowHsGram, out bool bAllowHsEvent, out bool bAllowHsScalar, out bool bAllowHsLoss)
        {
            RawProto proto = RawProto.Parse(strConfig);
            string strVal;

            strSolver = null;
            if ((strVal = proto.FindValue("solver")) != null)
                strSolver = strVal;

            dfLearningRate = 0;
            if ((strVal = proto.FindValue("learning_rate")) != null)
                dfLearningRate = double.Parse(strVal);

            nMaxImageSize = 0;
            if ((strVal = proto.FindValue("max_image_size")) != null)
                nMaxImageSize = int.Parse(strVal);

            nIterations = 1000;
            if ((strVal = proto.FindValue("iterations")) != null)
                nIterations = int.Parse(strVal);

            nIntermediateIterations = 0;
            if ((strVal = proto.FindValue("intermediate_iterations")) != null)
                nIntermediateIterations = int.Parse(strVal);

            Dictionary<string, Tuple<double, double>> rgLayers = new Dictionary<string, Tuple<double, double>>();
            RawProtoCollection style = proto.FindChildren("layer");
            foreach (RawProto styleProto in style)
            {
                string strLayer = null;
                if ((strVal = styleProto.FindValue("name")) != null)
                    strLayer = strVal;

                double dfSWt = 0;
                if ((strVal = styleProto.FindValue("style_wt")) != null)
                    dfSWt = double.Parse(strVal);

                double dfCWt = 0;
                if ((strVal = styleProto.FindValue("content_wt")) != null)
                    dfCWt = double.Parse(strVal);

                rgLayers.Add(strLayer, new Tuple<double, double>(dfSWt, dfCWt));
            }

            rgGpuID = new List<int>();
            RawProtoCollection gpus = proto.FindChildren("gpuid");
            foreach (RawProto gpuProto in gpus)
            {
                rgGpuID.Add(int.Parse(gpuProto.Value));
            }

            nLBFGSCorrections = 100;
            if ((strVal = proto.FindValue("lbfgs_corrections")) != null)
                nLBFGSCorrections = int.Parse(strVal);

            dfDataScale = 1.0;
            if ((strVal = proto.FindValue("data_scale")) != null)
                dfDataScale = double.Parse(strVal);

            bAllowHs = false;
            if ((strVal = proto.FindValue("allow_hs")) != null)
                bAllowHs = bool.Parse(strVal);

            bAllowHsGram = true;
            if ((strVal = proto.FindValue("allow_hs_gram")) != null)
                bAllowHsGram = bool.Parse(strVal);

            bAllowHsEvent = true;
            if ((strVal = proto.FindValue("allow_hs_event")) != null)
                bAllowHsEvent = bool.Parse(strVal);

            bAllowHsScalar = true;
            if ((strVal = proto.FindValue("allow_hs_scalar")) != null)
                bAllowHsScalar = bool.Parse(strVal);

            bAllowHsLoss = true;
            if ((strVal = proto.FindValue("allow_hs_loss")) != null)
                bAllowHsLoss = bool.Parse(strVal);

            return rgLayers;
        }
    }

    /// <summary>
    /// The NeuralStyleIntermediateOutputArgs contains the arguments sent to the OnIntermediateOutput event.
    /// </summary>
    public class NeuralStyleIntermediateOutputArgs : EventArgs
    {
        Bitmap m_img;
        int m_nIteration;
        double m_dfPercent;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nIteration">Specifies the current iteration.</param>
        /// <param name="bmp">Specifies the intermediate image.</param>
        /// <param name="dfPct">Specifies the total processing progress.</param>
        public NeuralStyleIntermediateOutputArgs(int nIteration, Bitmap bmp, double dfPct)
        {
            m_nIteration = nIteration;
            m_img = bmp;
            m_dfPercent = dfPct;
        }

        /// <summary>
        /// Returns the current interation.
        /// </summary>
        public int Iteration
        {
            get { return m_nIteration; }
        }

        /// <summary>
        /// Returns the current intermediate image.
        /// </summary>
        public Bitmap Image
        {
            get { return m_img; }
        }

        /// <summary>
        /// Returns the total processing progress.
        /// </summary>
        public double Percent
        {
            get { return m_dfPercent; }
        }
    }
}
