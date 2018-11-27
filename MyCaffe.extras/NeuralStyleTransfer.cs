using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
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
        List<string> m_rgContentLayers;
        List<string> m_rgStyleLayers;
        List<string> m_rgGramLayers;
        int m_nIterations = 200;
        int m_nDisplayEvery = 100;
        double m_dfTVLossWeight = 0.01;  // 0.01 - smaller numbers sharpen the image.
        double m_dfStyleWeight = 100;    // 100 - higher numbers use more style.
        double m_dfContentWeight = 100;    // 5 - higher numbers use more content.
        CancelEvent m_evtCancel;
        DataTransformer<T> m_transformer = null;
        TransformationParameter m_transformationParam;
        PersistCaffe<T> m_persist;
        NetParameter m_net_param;
        byte[] m_rgWeights = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the instance of CudaDnn to use.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="strModel">Specifies the network model to use.</param>
        /// <param name="rgWeights">Optionally, specifies the weights to use (or <i>null</i> to ignore).</param>
        /// <param name="rgContentLayers">Specifies the names of the content layers.</param>
        /// <param name="rgStyleLayers">Specifies the names of the style layers.</param>
        /// <param name="evtCancel">Specifies the cancel event used to abort processing.</param>
        /// <param name="bCaffeModel">Specifies whether or not the weights are in the caffe (<i>true</i>) or mycaffe (<i>false</i>) format.</param>
        /// <param name="dfContentWeight">Optionally, specifies the content weight to use (default = 100).</param>
        /// <param name="dfStyleWeight">Optionally, specifies the style weight to use (default = 5000).</param>
        /// <param name="dfTvWeight">Optionally, specifies the TV weight to use (default = 0.01).</param>
        public NeuralStyleTransfer(CudaDnn<T> cuda, Log log, string strModel, byte[] rgWeights, List<string> rgContentLayers, List<string> rgStyleLayers, CancelEvent evtCancel, bool bCaffeModel, double? dfContentWeight = null, double? dfStyleWeight = null, double? dfTvWeight = null)
        {
            m_cuda = cuda;
            m_log = log;
            m_evtCancel = evtCancel;
            m_rgWeights = rgWeights;

            if (dfContentWeight.HasValue)
                m_dfContentWeight = dfContentWeight.Value;

            if (dfStyleWeight.HasValue)
                m_dfStyleWeight = dfStyleWeight.Value;

            if (dfTvWeight.HasValue)
                m_dfTVLossWeight = dfTvWeight.Value;

            m_rgContentLayers = rgContentLayers;
            m_rgStyleLayers = rgStyleLayers;

            List<string> rgUsedLayers = new List<string>();
            rgUsedLayers.AddRange(m_rgContentLayers);
            rgUsedLayers.AddRange(m_rgStyleLayers);

            RawProto proto = RawProto.Parse(strModel);
            m_net_param = NetParameter.FromProto(proto);
            prune(m_net_param, rgUsedLayers);
            add_gram_layers(m_net_param, m_rgStyleLayers);

            // mean is taken from gist.github.com/ksimonyan/3785162f95cd2d5fee77
            m_transformationParam = new TransformationParameter();
            m_transformationParam.color_order = (bCaffeModel) ? TransformationParameter.COLOR_ORDER.BGR : TransformationParameter.COLOR_ORDER.RGB;
            m_transformationParam.scale = 1.0 / 256.0;
            m_transformationParam.mean_value.AddRange(new List<double>() { 103.939, 116.779, 123.68 });

            m_persist = new PersistCaffe<T>(m_log, false);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
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

        private void add_gram_layers(NetParameter p, List<string> rgStyle)
        {
            m_rgGramLayers = new List<string>();

            for (int i = 0; i < rgStyle.Count; i++)
            {
                LayerParameter layer = new LayerParameter(LayerParameter.LayerType.GRAM);
                string strStyle = rgStyle[i];
                string strGram = "gram_" + strStyle;

                layer.name = strGram;
                m_rgGramLayers.Add(strGram);

                layer.bottom.Add(strStyle);
                layer.top.Add(strGram);

                p.layer.Add(layer);
            }
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

            Blob<T> data = net.param_by_name("input");
            data.Reshape(rgDataShape);
            data.mutable_cpu_data = m_transformer.Transform(ImageData.GetImageData(bmp, 3, false, -1));
        }

        private Bitmap save(Net<T> net)
        {
            Blob<T> blob = net.param_by_name("input");
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
        /// <returns>The resulting image is returned.</returns>
        public Bitmap Process(Bitmap bmpStyle, Bitmap bmpContent, int nIterations, string strResultDir = null, int nIntermediateOutput = -1)
        {
            Solver<T> solver = null;
            Net<T> net = null;
            BlobCollection<T> colContentActivations = new BlobCollection<T>();
            BlobCollection<T> colGramActivations = new BlobCollection<T>();
            double dfLoss;

            try
            {
                m_nIterations = nIterations;

                if (bmpStyle.Width != bmpContent.Width ||
                    bmpStyle.Height != bmpContent.Height)
                    bmpStyle = ImageTools.ResizeImage(bmpStyle, bmpContent.Width, bmpContent.Height);

                net = new Net<T>(m_cuda, m_log, m_net_param, m_evtCancel, null, Phase.TEST);

                if (m_rgWeights != null)
                    net.LoadWeights(m_rgWeights, m_persist);

                //-----------------------------------------
                //  Get style and content activations.
                //-----------------------------------------

                prepare_data_blob(net, bmpStyle);
                net.Forward(out dfLoss);

                foreach (string strGram in m_rgGramLayers)
                {
                    Blob<T> blobGram = net.blob_by_name(strGram);
                    colGramActivations.Add(blobGram.Clone());
                }

                Dictionary<string, int> rgStyleLayerSizes = new Dictionary<string, int>();
                foreach (string strStyle in m_rgStyleLayers)
                {
                    Blob<T> blobStyle = net.blob_by_name(strStyle);
                    rgStyleLayerSizes.Add(strStyle, blobStyle.count());
                }

                prepare_data_blob(net, bmpContent);
                net.Forward(out dfLoss);

                foreach (string strContent in m_rgContentLayers)
                {
                    Blob<T> blobContent = net.blob_by_name(strContent);
                    colContentActivations.Add(blobContent.Clone());
                }


                //-----------------------------------------
                //  Prepare the network by adding new layers.
                //-----------------------------------------

                NetParameter net_param = m_net_param;
                List<string> rgInputLayers = new List<string>();
                rgInputLayers.AddRange(m_rgContentLayers);
                rgInputLayers.AddRange(m_rgGramLayers);

                foreach (string strName in rgInputLayers)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.INPUT);
                    p.name = "input_" + strName;
                    p.top.Add(p.name);

                    Blob<T> blob = net.blob_by_name(strName);
                    BlobShape shape = new BlobShape(blob.shape());

                    p.input_param.shape.Add(shape);

                    net_param.layer.Add(p);
                }

                foreach (string strName in m_rgContentLayers)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strName;

                    double dfWeight = 2 * m_dfContentWeight / colContentActivations[strName].count();
                    p.loss_weight.Add(dfWeight);

                    p.bottom.Add("input_" + strName);
                    p.bottom.Add(strName);
                    p.top.Add("loss_" + strName);

                    net_param.layer.Add(p);
                }

                foreach (string strName in m_rgStyleLayers)
                {
                    string strGramName = "gram_" + strName;

                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strGramName;

                    double dfWeight = 2 * m_dfStyleWeight / colGramActivations[strGramName].count() / Math.Pow(rgStyleLayerSizes[strName], 2.0);
                    p.loss_weight.Add(dfWeight);

                    p.bottom.Add("input_" + strGramName);
                    p.bottom.Add(strGramName);
                    p.top.Add("loss_" + strGramName);

                    net_param.layer.Add(p);
                }

                // Add TV Loss;
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

                    if (p.name == "input")
                    {
                        net_param.layer[i].SetType(LayerParameter.LayerType.PARAMETER);
                        net_param.layer[i].parameter_param.shape = new BlobShape(data.shape());
                        break;
                    }
                }

                // Disable weights learning.
                List<LayerParameter.LayerType> rgTypes = new List<LayerParameter.LayerType>();
                rgTypes.Add(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                rgTypes.Add(LayerParameter.LayerType.TV_LOSS);
                rgTypes.Add(LayerParameter.LayerType.GRAM);
                rgTypes.Add(LayerParameter.LayerType.INPUT);
                rgTypes.Add(LayerParameter.LayerType.PARAMETER);
                rgTypes.Add(LayerParameter.LayerType.POOLING);
                rgTypes.Add(LayerParameter.LayerType.RELU);

                foreach (LayerParameter layer in net_param.layer)
                {
                    if (!rgTypes.Contains(layer.type))
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

                solver = new LBFGSSolver<T>(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, m_persist);
                solver.OnSnapshot += Solver_OnSnapshot;

                prepare_input_param(solver.net, bmpContent);

                foreach (string strName in m_rgContentLayers)
                {
                    Blob<T> blobDst = solver.net.blob_by_name("input_" + strName);
                    Blob<T> blobSrc = colContentActivations[strName];
                    blobDst.CopyFrom(blobSrc);
                }

                foreach (string strName in m_rgGramLayers)
                {
                    Blob<T> blobDst = solver.net.blob_by_name("input_" + strName);
                    Blob<T> blobSrc = colGramActivations[strName];
                    blobDst.CopyFrom(blobSrc);
                }

                //-----------------------------------------
                //  Optimize.
                //-----------------------------------------

                if (strResultDir != null && nIntermediateOutput > 0)
                {
                    int nImageCount = m_nIterations / nIntermediateOutput;

                    solver.Solve(nIntermediateOutput, m_rgWeights);

                    strResultDir = strResultDir.TrimEnd('\\');
                    strResultDir += "\\";

                    for (int i = 0; i < nImageCount; i++)
                    {
                        Bitmap bmpTemp = save(solver.net);

                        string strFile = strResultDir + i.ToString() + "_temp.png";
                        if (File.Exists(strFile))
                            File.Delete(strFile);

                        bmpTemp.Save(strFile);

                        solver.Step(nIntermediateOutput);
                    }
                }
                else
                {
                    solver.Solve(m_nIterations, m_rgWeights);
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

        private void Solver_OnSnapshot(object sender, SnapshotArgs e)
        {
        }
    }
}
