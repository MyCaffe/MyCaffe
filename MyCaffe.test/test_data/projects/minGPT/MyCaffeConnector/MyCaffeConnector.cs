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
        Dictionary<string, Blob<float>> m_rgBtm = new Dictionary<string, Blob<float>>();
        Dictionary<string, Blob<float>> m_rgTop = new Dictionary<string, Blob<float>>();
        BlobCollection<float> m_colTop = new BlobCollection<float>();
        BlobCollection<float> m_colBtm = new BlobCollection<float>();
        Blob<float> m_blobEncIn;
        Blob<float> m_blobTgt;
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
            return File.ReadAllText("C:\\temp\\projects\\2023.minGpt\\minGPT\\models\\gpt\\solver.prototxt");
        }

        private string buildModel()
        {
            return File.ReadAllText("C:\\temp\\projects\\2023.minGpt\\minGPT\\models\\gpt\\train_test.prototxt");
        }

        public void Initialize()
        {
            m_colTop.Clear();
            m_colTop.Add(m_blobTop);
            m_colBtm.Clear();
            m_colBtm.Add(m_blobBtm);

            for (int i = 0; i < 6; i++)
            {
                string strName = "blk" + i.ToString();
                m_rgBtm.Add(strName, m_mycaffe.CreateBlob(strName));
                m_rgTop.Add(strName, m_mycaffe.CreateBlob(strName));
            }
        }

        public void InitializeEx(uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout)
        {
            string strSolver = buildSolver();
            string strModel = buildModel();

            m_mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
            
            m_blobEncIn = m_mycaffe.CreateBlob("encin");
            m_blobEncIn.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobTgt = m_mycaffe.CreateBlob("tgt");
            m_blobTgt.Reshape((int)nBatch, (int)nBlockSize, 1, 1);

            for (int i = 0; i < 6; i++)
            {
                string strName = "blk" + i.ToString();
                m_rgBtm.Add(strName, m_mycaffe.CreateBlob(strName));
                m_rgTop.Add(strName, m_mycaffe.CreateBlob(strName));
            }
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

        public float[] tfb_fwd(string strTag, int nLayers, int nHeads, int nEmbed, int nBlkSize, int nN, int nC, int nH, int nW, float[] rg)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobBtm.Reshape(rgShape);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
                p.transformer_block_param.heads = (uint)nHeads;
                p.transformer_block_param.embed = (uint)nEmbed;
                p.transformer_block_param.block_size = (uint)nBlkSize;
                p.transformer_block_param.attn_dropout = 0.0;
                p.transformer_block_param.resid_dropout = 0.0;
                p.transformer_block_param.layers = (uint)nLayers;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] tfb_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobTop.Reshape(rgShape);
            m_blobBtm.Reshape(rgShape);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] tfb_fwd_all(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            BlobCollection<float> colBtm = new BlobCollection<float>();
            BlobCollection<float> colTop = new BlobCollection<float>();

            Blob<float> blobLastTop = null;

            for (int i = 0; i < 6; i++)
            {
                string strName = "blk" + i.ToString();
                Blob<float> blobBtm = m_rgBtm[strName];
                Blob<float> blobTop = m_rgTop[strName];

                colBtm.Clear();
                colBtm.Add(blobBtm);
                colTop.Clear();
                colTop.Add(blobTop);

                blobBtm.Reshape(rgShape);

                if (i == 0)
                    blobBtm.mutable_cpu_data = rg;
                else
                    blobBtm.CopyFrom(blobLastTop);    

                if (!m_rgLayers.ContainsKey(strName))
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                    p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
                    p.transformer_block_param.heads = 6;
                    p.transformer_block_param.embed = 192;
                    p.transformer_block_param.block_size = 128;
                    p.transformer_block_param.attn_dropout = 0.0;
                    p.transformer_block_param.resid_dropout = 0.0;
                    p.transformer_block_param.layers = 6;
                    p.name = "tfb" + i.ToString();
                    Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                    layer1.Setup(colBtm, colTop);
                    m_rgLayers.Add(strName, layer1);
                }

                Layer<float> layer = m_rgLayers[strName];
                layer.Forward(colBtm, colTop);

                blobLastTop = blobTop;
            }

            return blobLastTop.mutable_cpu_data;
        }

        public float[] tfb_bwd_all(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            BlobCollection<float> colBtm = new BlobCollection<float>();
            BlobCollection<float> colTop = new BlobCollection<float>();

            Blob<float> blobLastBtm = null;

            for (int i = 0; i < 6; i++)
            {
                string strName = "blk" + i.ToString();
                Blob<float> blobBtm = m_rgBtm[strName];
                Blob<float> blobTop = m_rgTop[strName];

                colBtm.Clear();
                colBtm.Add(blobBtm);
                colTop.Clear();
                colTop.Add(blobTop);

                blobTop.Reshape(rgShape);
                blobBtm.Reshape(rgShape);

                if (i == 0)
                {
                    blobTop.mutable_cpu_diff = rgY;
                    blobTop.mutable_cpu_diff = rgYGrad;
                }
                else
                {
                    blobTop.CopyFrom(blobLastBtm, true);
                }

                Layer<float> layer = m_rgLayers[strName];
                layer.Backward(colTop, new List<bool>() { true }, colBtm);

                blobLastBtm = blobBtm;
            }

            return blobLastBtm.mutable_cpu_diff;
        }


        public float[] softmax_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobBtm.Reshape(rgShape);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                p.softmax_param.engine = EngineParameter.Engine.CAFFE;
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
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobTop.Reshape(rgShape);
            m_blobBtm.Reshape(rgShape);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] logsoftmax_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobBtm.Reshape(rgShape);
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
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobTop.Reshape(rgShape);
            m_blobBtm.Reshape(rgShape);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] layernorm_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobBtm.Reshape(rgShape);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.layer_norm_param.enable_cuda_impl = false;
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
            List<int> rgShape = new List<int>() { nN, nC, nH };
            if (nW > 1)
                rgShape.Add(nW);

            m_blobTop.Reshape(rgShape);
            m_blobBtm.Reshape(rgShape);
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

        public float[] Step(int nIter, float[] rgEncIn, float[] rgTgt)
        {
            m_blobEncIn.mutable_cpu_data = rgEncIn;
            m_blobTgt.mutable_cpu_data = rgTgt;

            Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);

            BlobCollection<float> colInput = new BlobCollection<float>();
            colInput.Add(m_blobEncIn);
            colInput.Add(m_blobTgt);

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
            dispose(ref m_blobTgt);
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
