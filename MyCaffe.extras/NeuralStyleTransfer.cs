using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.param;
using MyCaffe.solvers;
using System;
using System.Collections.Generic;
using System.Drawing;
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
        int m_nIterations = 1000;
        int m_nSaveEvery = 100;
        int m_nDisplayEvery = 100;
        double m_dfTVLossWeight = 1e-2;
        double m_dfStyleWeight = 100;
        double m_dfContentWeight = 5;
        int m_nGpuID = 0;
        Net<T> m_net;
        CancelEvent m_evtCancel;
        DataTransformer<T> m_transformer = null;
        TransformationParameter m_transformationParam;
        PersistCaffe<T> m_persist;

        public NeuralStyleTransfer(CudaDnn<T> cuda, Log log, NetParameter p, List<string> rgContentLayers, List<string> rgStyleLayers, CancelEvent evtCancel)
        {
            m_cuda = cuda;
            m_log = log;
            m_evtCancel = evtCancel;

            m_rgContentLayers = rgContentLayers;
            m_rgStyleLayers = rgStyleLayers;

            List<string> rgUsedLayers = new List<string>();
            rgUsedLayers.AddRange(m_rgContentLayers);
            rgUsedLayers.AddRange(m_rgStyleLayers);

            NetParameter netP = p.Clone();
            prune(netP, rgUsedLayers);
            add_gram_layers(netP, m_rgStyleLayers);

            m_net = new Net<T>(m_cuda, m_log, netP, evtCancel, null, Phase.TEST);

            // mean is taken from gist.github.com/ksimonyan/3785162f95cd2d5fee77
            m_transformationParam = new TransformationParameter();
            m_transformationParam.color_order = TransformationParameter.COLOR_ORDER.BGR;
            m_transformationParam.scale = 256.0;
            m_transformationParam.mean_value.AddRange(new List<double>() { 103.939, 116.779, 123.68 });

            m_persist = new PersistCaffe<T>(m_log, false);
        }

        public void Dispose()
        {
            if (m_net != null)
            {
                m_net.Dispose();
                m_net = null;
            }
        }

        private void prune(NetParameter p, List<string> rgUsedLayers)
        {
            int nPruneFrom = -1;

            // We assume that all layers after the used layers are not useful.
            for (int i = 0; i < p.layer.Count; i++)
            {
                for (int j = 0; j < p.layer[i].top.Count; j++)
                {
                    if (rgUsedLayers.Contains(p.layer[i].top[j]))
                    {
                        nPruneFrom = -1;
                        break;
                    }
                    else
                    {
                        if (nPruneFrom < 0)
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
                layer.name = "gram_" + rgStyle[i];
                m_rgGramLayers.Add(layer.name);
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
            Datum d = ImageData.GetImageData<T>(blob.update_cpu_data(), 3, blob.height, blob.width, false);
            return ImageData.GetImage(d);
        }

        public Bitmap Process(Bitmap bmpStyle, Bitmap bmpContent, int nIterations)
        {
            BlobCollection<T> colContentActivations = new BlobCollection<T>();
            BlobCollection<T> colGramActivations = new BlobCollection<T>();
            double dfLoss;

            try
            {
                //-----------------------------------------
                //  Get style and content activations.
                //-----------------------------------------

                prepare_data_blob(m_net, bmpStyle);
                m_net.Forward(out dfLoss);

                foreach (string strGram in m_rgGramLayers)
                {
                    Blob<T> blobGram = m_net.blob_by_name(strGram);
                    colGramActivations.Add(blobGram.Clone());
                }

                Dictionary<string, int> rgStyleLayerSizes = new Dictionary<string, int>();
                foreach (string strStyle in m_rgStyleLayers)
                {
                    Blob<T> blobStyle = m_net.blob_by_name(strStyle);
                    rgStyleLayerSizes.Add(strStyle, blobStyle.count());
                }

                prepare_data_blob(m_net, bmpContent);
                m_net.Forward(out dfLoss);

                foreach (string strContent in m_rgContentLayers)
                {
                    Blob<T> blobContent = m_net.blob_by_name(strContent);
                    colContentActivations.Add(blobContent.Clone());
                }


                //-----------------------------------------
                //  Prepare the network by adding new layers.
                //-----------------------------------------

                NetParameter net_param = m_net.ToProto(false);
                List<string> rgInputLayers = new List<string>();
                rgInputLayers.AddRange(m_rgContentLayers);
                rgInputLayers.AddRange(m_rgGramLayers);

                foreach (string strName in rgInputLayers)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.INPUT);
                    p.name = "input_" + strName;
                    p.top.Add(p.name);

                    Blob<T> blob = m_net.blob_by_name(strName);
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
                    p.top.Add("loss_" + strName);

                    net_param.layer.Add(p);
                }

                foreach (string strName in m_rgStyleLayers)
                {
                    string strGramName = "gram_" + strName;

                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
                    p.name = "loss_" + strGramName;

                    double dfWeight = 2 * m_dfStyleWeight / colGramActivations[strName].count() / Math.Pow(rgStyleLayerSizes[strName], 2.0);
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
                Blob<T> data = m_net.blob_by_name("data");
                for (int i=0; i<net_param.layer.Count; i++)
                {
                    LayerParameter p = net_param.layer[i];

                    if (p.name == "input")
                    {
                        net_param.layer[i] = new LayerParameter(LayerParameter.LayerType.PARAMETER);
                        net_param.layer[i].parameter_param.shape = new BlobShape(data.shape());
                        net_param.layer[i].bottom = Utility.Clone<string>(p.bottom);
                        net_param.layer[i].top = Utility.Clone<string>(p.top);
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
                    layer.parameters = new List<ParamSpec>();
                    layer.parameters.Add(new ParamSpec(0, 0));
                    layer.parameters.Add(new ParamSpec(0, 0));
                }


                //-----------------------------------------
                //  Create solver and assign inputs.
                //-----------------------------------------

                SolverParameter solver_param = new SolverParameter();
                solver_param.display = m_nDisplayEvery;
                solver_param.train_net_param = net_param;

                Solver<T> solver = new LBFGSSolver<T>(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, m_persist);
                solver.OnSnapshot += Solver_OnSnapshot;

                prepare_input_param(solver.net, bmpContent);

                foreach (string strName in m_rgContentLayers)
                {
                    Blob<T> blobDst = solver.net.blob_by_name("input_" + strName);
                    Blob<T> blobSrc = colContentActivations[strName];
                    blobDst.ShareData(blobSrc);
                    blobDst.ShareDiff(blobSrc);
                }

                foreach (string strName in m_rgGramLayers)
                {
                    Blob<T> blobDst = solver.net.blob_by_name("input_" + strName);
                    Blob<T> blobSrc = colGramActivations[strName];
                    blobDst.ShareData(blobSrc);
                    blobDst.ShareDiff(blobSrc);
                }

                //-----------------------------------------
                //  Optimize.
                //-----------------------------------------

                solver.Solve(m_nIterations);
                Bitmap bmpOutput = save(solver.net);

                solver.Dispose();

                return bmpOutput;
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                colGramActivations.Dispose();
                colContentActivations.Dispose();
            }
        }

        private void Solver_OnSnapshot(object sender, SnapshotArgs e)
        {
        }
    }
}
