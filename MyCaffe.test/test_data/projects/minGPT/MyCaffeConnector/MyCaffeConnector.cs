using MyCaffe;
using MyCaffe.basecode;
using MyCaffe.param.gpt;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.common;
using MyCaffe.solvers;
using System.Runtime.InteropServices;
using System.IO;
using System.Reflection;
using MyCaffe.layers;
using System.Reflection.Emit;

namespace MyCaffeConnector
{
    public class MyCaffeConnector : IDisposable
    {
        CancelEvent m_evtCancel = new CancelEvent();
        Log m_log = new Log("MyCaffeConnector");
        MyCaffeControl<float> m_mycaffe;
        Dictionary<string, Layer<float>> m_rgLayers = new Dictionary<string, Layer<float>>();
        Dictionary<string, List<int>> m_rgTopShapes = new Dictionary<string, List<int>>();
        Dictionary<string, List<int>> m_rgBtmShapes = new Dictionary<string, List<int>>();
        BlobCollection<float> m_colTop = new BlobCollection<float>();
        BlobCollection<float> m_colBtm = new BlobCollection<float>();
        Blob<float> m_blobEncIn;
        Blob<float> m_blobDecIn;
        Blob<float> m_blobDecOut;
        Blob<float> m_blobEncMask;
        Blob<float> m_blobDecMask;
        Blob<float> m_blobLoss;
        Blob<float> m_blobBtm;
        Blob<float> m_blobTop;
        float m_fLastAccuracy = 0;
        
        public MyCaffeConnector()
        {
            SettingsCaffe s = new SettingsCaffe() { GpuIds = "0" };
            string strCudaPath = "C:\\Program Files\\SignalPop\\MyCaffe\\cuda_11.7\\CudaDnnDll.11.7.dll";
            m_mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel, null, null, null, null, strCudaPath, true);

            m_blobBtm = m_mycaffe.CreateBlob("btm");
            m_blobTop = m_mycaffe.CreateBlob("top");
        }

        public void Dispose()
        {
            CleanUp();
        }

        public void Test(float[] ptr)
        {
        }

        private string buildSolver()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 1e-4;
            solver.type = SolverParameter.SolverType.ADAM;
            solver.lr_policy = "fixed";
            solver.test_initialization = false;

            return solver.ToProto("root").ToString();
        }

        private string buildModelEx(NetParameter net, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout, bool bAddInput = false, Phase phase = Phase.TRAIN)
        {
            if (bAddInput)
            {
                LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
                input.name = "input";
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize, (int)nBlockSize } });
                input.top.Add("enc");
                input.top.Add("dec");
                input.top.Add("tgt");
                input.top.Add("emsk");
                input.top.Add("dmsk");
                net.layer.Add(input);
            }

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "embed1";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.bottom.Add("enc");
            emb1.top.Add("emb1");
            net.layer.Add(emb1);

            LayerParameter emb2 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb2.name = "embed2";
            emb2.embed_param.bias_term = false;
            emb2.embed_param.input_dim = nDecVocabSize;
            emb2.embed_param.num_output = nEmbed;
            emb2.bottom.Add("dec");
            emb2.top.Add("emb2");
            net.layer.Add(emb2);

            LayerParameter pos1 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            pos1.positional_encoder_param.block_size = nBlockSize;
            pos1.positional_encoder_param.embed = nEmbed;
            pos1.name = "posenc1";
            pos1.bottom.Add("emb1");
            pos1.top.Add("pos1");
            net.layer.Add(pos1);

            LayerParameter pos2 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            pos2.positional_encoder_param.block_size = nBlockSize;
            pos2.positional_encoder_param.embed = nEmbed;
            pos2.name = "posenc2";
            pos2.bottom.Add("emb2");
            pos2.top.Add("pos2");
            net.layer.Add(pos2);

            string strEncBtm = "pos1";
            int nLayers = 6;
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "enc" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
                enc.transformer_block_param.heads = 8;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.bottom.Add(strEncBtm);
                enc.bottom.Add("emsk");
                enc.top.Add(enc.name);
                net.layer.Add(enc);

                strEncBtm = enc.name;
            }

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln1.name = "ln1";
            ln1.layer_norm_param.enable_cuda_impl = false;
            ln1.bottom.Add(strEncBtm);
            ln1.top.Add("ln1");
            net.layer.Add(ln1);

            string strDecBtm = "pos2";
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter dec = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                dec.name = "dec" + (i + 1).ToString();
                dec.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
                dec.transformer_block_param.heads = 8;
                dec.transformer_block_param.embed = nEmbed;
                dec.transformer_block_param.block_size = nBlockSize;
                dec.transformer_block_param.layers = (uint)nLayers;
                dec.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                dec.transformer_block_param.attn_dropout = dfDropout;
                dec.transformer_block_param.resid_dropout = dfDropout;
                dec.bottom.Add(strDecBtm);
                dec.bottom.Add("dmsk");
                dec.bottom.Add("ln1");
                dec.bottom.Add("emsk");
                dec.top.Add(dec.name);
                net.layer.Add(dec);

                strDecBtm = dec.name;
            }

            LayerParameter ln2 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln2.name = "ln2";
            ln2.layer_norm_param.enable_cuda_impl = false;
            ln2.bottom.Add(strDecBtm);
            ln2.top.Add("ln2");
            net.layer.Add(ln2);

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nDecVocabSize;
            ip1.bottom.Add("ln2");
            ip1.top.Add("logits");
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.softmax_param.algorithm = SOFTMAX_ALGORITHM.LOG;
            softmax.softmax_param.algorithm_train = SOFTMAX_ALGORITHM.LOG;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            net.layer.Add(softmax);

            if (phase == Phase.TRAIN)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
                loss.name = "loss";
                loss.nll_loss_param.axis = 2;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.bottom.Add("prob");
                loss.bottom.Add("tgt");
                loss.top.Add("loss");
                loss.include.Add(new NetStateRule(Phase.TRAIN));
                net.layer.Add(loss);
            }

            if (phase == Phase.TRAIN)
            {
                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.axis = 2;
                accuracy.accuracy_param.ignore_labels.Add(0);
                accuracy.accuracy_param.enable_simple_accuracy = true;
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("tgt");
                accuracy.top.Add("accuracy");
                accuracy.include.Add(new NetStateRule(Phase.TRAIN));
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }

        public void Initialize()
        {
            m_colTop.Clear();
            m_colTop.Add(m_blobTop);
            m_colBtm.Clear();
            m_colBtm.Add(m_blobBtm);
        }

        public void InitializeEx(uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout)
        {
            NetParameter net_param = new NetParameter();
            string strSolver = buildSolver();
            string strModel = buildModelEx(net_param, nBatch, nBlockSize, nEmbed, nEncVocabSize, nDecVocabSize, dfDropout, true);

            m_mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
            
            m_blobEncIn = m_mycaffe.CreateBlob("encin");
            m_blobEncIn.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobDecIn = m_mycaffe.CreateBlob("decin");
            m_blobDecIn.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobDecOut = m_mycaffe.CreateBlob("decout");
            m_blobDecOut.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobEncMask = m_mycaffe.CreateBlob("e_mask");
            m_blobEncMask.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobDecMask = m_mycaffe.CreateBlob("d_mask");
            m_blobDecMask.Reshape((int)nBatch, (int)nBlockSize, (int)nBlockSize, 1);
            m_blobLoss = m_mycaffe.CreateBlob("loss");
            m_blobLoss.Reshape(1, 1, 1, 1);
        }

        public static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }

        public float[] softmax_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                p.softmax_param.engine = EngineParameter.Engine.CUDNN;
                p.softmax_param.axis = -1;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);
            
            return m_blobTop.mutable_cpu_data;
        }

        public float[] softmax_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] logsoftmax_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                p.softmax_param.engine = EngineParameter.Engine.CUDNN;
                p.softmax_param.algorithm = SOFTMAX_ALGORITHM.LOG;
                p.softmax_param.axis = 2;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] logsoftmax_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] layernorm_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.layer_norm_param.enable_cuda_impl = true;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] layernorm_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public void innerproduct_setup(string strTag, int nAxis, int nN, int nC, int nIn, int nOut, float[] rgW, float[] rgB)
        {
            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                p.inner_product_param.axis = nAxis;
                p.inner_product_param.num_output = (uint)nOut;
                p.inner_product_param.bias_term = (rgB != null && rgB.Length > 0);
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                m_blobBtm.Reshape(nN, nC, nIn, 1);

                layer1.Setup(m_colBtm, m_colTop);
                layer1.blobs[0].mutable_cpu_data = rgW;
                if (rgB != null && rgB.Length > 0)
                    layer1.blobs[1].mutable_cpu_data = rgB;

                m_rgLayers.Add(strTag, layer1);
            }
            else
            {
                Layer<float> layer = m_rgLayers[strTag];
                layer.blobs[0].mutable_cpu_data = rgW;
                if (rgB != null && rgB.Length > 0)
                    layer.blobs[1].mutable_cpu_data = rgB;
            }
        }

        public float[] innerproduct_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
                throw new Exception("You must call 'innerproduct_setup' first!");

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            if (!m_rgBtmShapes.ContainsKey(strTag))
                m_rgBtmShapes.Add(strTag, Utility.Clone<int>(m_blobBtm.shape()));
            if (!m_rgTopShapes.ContainsKey(strTag))
                m_rgTopShapes.Add(strTag, Utility.Clone<int>(m_blobTop.shape()));

            return m_blobTop.mutable_cpu_data;
        }

        public float[] innerproduct_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(m_rgTopShapes[strTag]);
            m_blobBtm.Reshape(m_rgBtmShapes[strTag]);

            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] channel_sum(int nN, int nC, int nH, float[] rgX)
        {
            List<int> rgShapeB = new List<int>() { nN, nC, nH };
            m_blobBtm.Reshape(rgShapeB);
            List<int> rgShapeT = new List<int>() { nN, nC };
            m_blobTop.Reshape(rgShapeT);
            m_blobBtm.mutable_cpu_data = rgX;
            m_mycaffe.Cuda.channel_sum(nN * nC, nN * nC, nH, 1, m_blobBtm.gpu_data, m_blobTop.mutable_gpu_data);
            return m_blobTop.mutable_cpu_data;
        }

        public void save(int nIter)
        {
            SaveWeights("C:\\temp\\projects\\TransformerTranslator\\TransformerTranslator\\state\\", nIter);
        }

        public void SaveWeights(string strPath, int nIter)
        {
            string strDir = Directory.GetCurrentDirectory();
            try
            {
                Directory.SetCurrentDirectory(AssemblyDirectory);
                
                byte[] rgb = m_mycaffe.GetWeights();
                File.WriteAllBytes(strPath + nIter.ToString() + "_mycaffe.mycaffemodel", rgb);

                Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);
                RawProto proto = net.net_param.ToProto("root");
                string strModel = proto.ToString();
                File.WriteAllText(strPath + "mycaffe.prototxt", strModel);

                Solver<float> solver = m_mycaffe.GetInternalSolver();
                RawProto proto2 = solver.parameter.ToProto("root");
                string strSolver = proto2.ToString();
                File.WriteAllText(strPath + "mycaffe.solver.prototxt", strSolver);
            }
            finally
            {
                Directory.SetCurrentDirectory(strDir);
            }
        }

        public float CurrentLoss
        {
            get
            {
                Solver<float> solver = m_mycaffe.GetInternalSolver();
                return (float)solver.smoothed_loss;
            }
        }

        public float CurrentAccuracy
        {
            get { return m_fLastAccuracy; }
        }

        public float[] Step(int nIter, float[] rgEncIn, float[] rgDecIn, float[] rgDecOut, float[] rgEncMask, float[] rgDecMask)
        {
            m_blobEncIn.mutable_cpu_data = rgEncIn;
            m_blobDecIn.mutable_cpu_data = rgDecIn;
            m_blobDecOut.mutable_cpu_data = rgDecOut;
            m_blobEncMask.mutable_cpu_data = rgEncMask;
            m_blobDecMask.mutable_cpu_data = rgDecMask;

            Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);

            BlobCollection<float> colInput = new BlobCollection<float>();
            colInput.Add(m_blobEncIn);
            colInput.Add(m_blobDecIn);
            colInput.Add(m_blobDecOut);
            colInput.Add(m_blobEncMask);
            colInput.Add(m_blobDecMask);

            net.ClearParamDiffs();

            double dfLoss;
            net.ForwardBackward(colInput, out dfLoss);
            
            Solver<float> solver = m_mycaffe.GetInternalSolver();
            solver.UpdateSmoothedLoss(dfLoss, nIter);
            solver.ApplyUpdate(nIter);

            //if (nIter % 500 == 0)
            //    save(nIter);

            Blob<float> blobAccuracy = net.FindBlob("accuracy");
            m_fLastAccuracy = blobAccuracy.GetData(0);

            Blob<float> blobOutput = net.FindBlob("prob");
            return blobOutput.mutable_cpu_data;                        
        }

        private void dispose(ref Blob<float> b)
        {
            if (b != null)
                b.Dispose();
            b = null;
        }

        public void CleanUp()
        {
            dispose(ref m_blobEncIn);
            dispose(ref m_blobDecIn);
            dispose(ref m_blobDecOut);
            dispose(ref m_blobEncMask);
            dispose(ref m_blobDecMask);
            dispose(ref m_blobLoss);
            dispose(ref m_blobBtm);
            dispose(ref m_blobTop);

            foreach (KeyValuePair<string, Layer<float>> kv in m_rgLayers)
            {
                kv.Value.Dispose();
            }

            if (m_mycaffe != null)
                m_mycaffe.Dispose();            
        }
    }
}
