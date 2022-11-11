using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using System.Diagnostics;
using System.IO;
using MyCaffe.param.gpt;
using MyCaffe.db.image;
using System.Threading;
using MyCaffe.layers.gpt;
using MyCaffe.solvers;
using MyCaffe.basecode.descriptors;

///
/// WORK IN PROGRESS
///
namespace MyCaffe.test
{
    [TestClass]
    public class TestTransformerBlockLayer
    {
        [TestMethod]
        public void TestForwardPico()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForwardPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientPico()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestGradientPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico3()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForwardPico(false, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico3()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(false, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientPico3()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestGradientPico(false, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico3Batch()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForwardPico(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico3Batch()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientPico3Batch()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestGradientPico(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardMini()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForwardMini();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
        
        [TestMethod]
        public void TestTrainingGptMini()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrainingGptMini(100000);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingGptPico()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrainingGptPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingGptPicoBatch3()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrainingGptPico(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITransformerBlockLayerTest : ITest
    {
        void TestForwardPico(bool bBatch, int nHeads);
        void TestBackwardPico(bool bBatch, int nHeads);
        void TestGradientPico(bool bBatch, int nHeads);
        void TestForwardMini();
        void TestTrainingGptMini(int nIter);
        void TestTrainingGptPico(bool bBatch, int nHeads);
    }

    class TransformerBlockLayerTest : TestBase
    {
        public TransformerBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Transformer Block Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TransformerBlockLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new TransformerBlockLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class TransformerBlockLayerTest2<T> : TestEx<T>, ITransformerBlockLayerTest
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();
        AutoResetEvent m_evtForceSnapshot = new AutoResetEvent(false);
        AutoResetEvent m_evtForceTest = new AutoResetEvent(false);
        WaitHandle[] m_rgevtCancel;
        List<int> m_rgGpu = new List<int>();
        Net<T> m_netRun = null;
        TokenizedDataLayer<T> m_dataLayer = null;
        Blob<T> m_blobY;
        Blob<T> m_blobX;
        float[] m_rgTestInput;
        MyCaffeControl<T> m_ctrl = null;

        public TransformerBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobY = new Blob<T>(m_cuda, m_log);
            m_blobX = new Blob<T>(m_cuda, m_log);

            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.AddRange(m_evtCancel.Handles);

            m_rgevtCancel = rgWait.ToArray();
            m_settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_settings.GpuIds = nDeviceID.ToString();
            m_rgGpu.Add(nDeviceID);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        private void dispose1(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        protected override void dispose()
        {
            dispose(ref m_blobY);
            dispose(ref m_blobX);

            base.dispose();
        }

        public Tuple<List<int>, float[]> Fill(string strGpt, string strName, Log log, string strPass = "")
        {
            string strFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\" + strGpt + "\\";

            if (!string.IsNullOrEmpty(strPass))
                strFile += strPass + "\\";

            strFile += strName + ".txt";

            string[] rgstrLines = File.ReadAllLines(strFile);
            string strSize = rgstrLines[0].Trim('#', ' ', '(', ')', ',');
            string[] rgstrSize = strSize.Split(',');
            List<int> rgnShape = new List<int>() { 1 };
            
            if (!string.IsNullOrEmpty(strSize))
                rgnShape = rgstrSize.Select(p1 => int.Parse(p1)).ToList();
            List<float> rgfVal = new List<float>();
            
            while (rgnShape.Count < 4)
            {
                rgnShape.Add(1);
            }

            int nCount = 1;
            foreach (int nDim in rgnShape)
            {
                nCount *= nDim;
            }

            for (int i = 1; i < rgstrLines.Length; i++)
            {
                string[] rgstrVals = rgstrLines[i].Split(' ');

                for (int j = 0; j < rgstrVals.Length; j++)
                {
                    string strVal = rgstrVals[j].Trim();

                    if (!string.IsNullOrEmpty(strVal))
                    {
                        float fVal = float.Parse(strVal);
                        rgfVal.Add(fVal);
                    }
                }
            }

            log.CHECK_EQ(rgfVal.Count, nCount, "The bottom count does not match the number of values read in!");

            float[] rgf = rgfVal.ToArray();

            return new Tuple<List<int>, float[]>(rgnShape, rgf);
        }

        public void TestForwardPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 3;
            p.transformer_block_param.block_size = 4;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-blk";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "1_x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);
                
                Tuple<List<int>, float[]> y = Fill(strModel, "10_y", m_log);
                m_blobY.Reshape(y.Item1);
                m_blobY.mutable_cpu_data = convert(y.Item2);
                
                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log);
                Tuple<List<int>, float[]> attnProjBias = Fill(strModel, "attn_proj_bias", m_log);
                Tuple<List<int>, float[]> attnProjWt = Fill(strModel, "attn_proj_weight", m_log);

                Tuple<List<int>, float[]> fcBias = Fill(strModel, "fc_bias", m_log);
                Tuple<List<int>, float[]> fcWt = Fill(strModel, "fc_weight", m_log);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "proj_bias", m_log);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "proj_weight", m_log);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].mutable_cpu_data = convert(attnWt.Item2);
                layer.blobs[1].mutable_cpu_data = convert(attnBias.Item2);
                layer.blobs[2].mutable_cpu_data = convert(attnProjWt.Item2);
                layer.blobs[3].mutable_cpu_data = convert(attnProjBias.Item2);
                
                layer.blobs[4].mutable_cpu_data = convert(fcWt.Item2);
                layer.blobs[5].mutable_cpu_data = convert(fcBias.Item2);
                layer.blobs[6].mutable_cpu_data = convert(projWt.Item2);
                layer.blobs[7].mutable_cpu_data = convert(projBias.Item2);

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                float[] rgExpected = convertF(m_blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 0.00000001f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackwardPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 3;
            p.transformer_block_param.block_size = 4;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-blk";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "1_x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "grad_1_y", m_log);                
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "grad_10_x", m_log);

                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log);
                Tuple<List<int>, float[]> attnProjBias = Fill(strModel, "attn_proj_bias", m_log);
                Tuple<List<int>, float[]> attnProjWt = Fill(strModel, "attn_proj_weight", m_log);

                Tuple<List<int>, float[]> fcBias = Fill(strModel, "fc_bias", m_log);
                Tuple<List<int>, float[]> fcWt = Fill(strModel, "fc_weight", m_log);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "proj_bias", m_log);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "proj_weight", m_log);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].mutable_cpu_data = convert(attnWt.Item2);
                layer.blobs[1].mutable_cpu_data = convert(attnBias.Item2);
                layer.blobs[2].mutable_cpu_data = convert(attnProjWt.Item2);
                layer.blobs[3].mutable_cpu_data = convert(attnProjBias.Item2);

                layer.blobs[4].mutable_cpu_data = convert(fcWt.Item2);
                layer.blobs[5].mutable_cpu_data = convert(fcBias.Item2);
                layer.blobs[6].mutable_cpu_data = convert(projWt.Item2);
                layer.blobs[7].mutable_cpu_data = convert(projBias.Item2);

                layer.Forward(BottomVec, TopVec);

                m_blob_top.mutable_cpu_diff = convert(y_grad.Item2);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);
                
                // Now, check values
                float[] rgExpected = x_grad.Item2;
                float[] rgActual = convertF(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 1e-3f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 3;
            p.transformer_block_param.block_size = 4;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-blk";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                Tuple<List<int>, float[]> data = Fill(strModel, "1_x", m_log);
                m_blob_bottom.Reshape(data.Item1);
                m_blob_bottom.mutable_cpu_data = convert(data.Item2);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
                checker.CheckGradient(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardMini()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.heads = 6;
            p.transformer_block_param.embed = 192;
            p.transformer_block_param.block_size = 128;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-mini-blk";

                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "1_x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y = Fill(strModel, "10_y", m_log);
                m_blobY.Reshape(y.Item1);
                m_blobY.mutable_cpu_data = convert(y.Item2);

                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log);
                Tuple<List<int>, float[]> attnProjBias = Fill(strModel, "attn_proj_bias", m_log);
                Tuple<List<int>, float[]> attnProjWt = Fill(strModel, "attn_proj_weight", m_log);

                Tuple<List<int>, float[]> fcBias = Fill(strModel, "fc_bias", m_log);
                Tuple<List<int>, float[]> fcWt = Fill(strModel, "fc_weight", m_log);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "proj_bias", m_log);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "proj_weight", m_log);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].mutable_cpu_data = convert(attnWt.Item2);
                layer.blobs[1].mutable_cpu_data = convert(attnBias.Item2);
                layer.blobs[2].mutable_cpu_data = convert(attnProjWt.Item2);
                layer.blobs[3].mutable_cpu_data = convert(attnProjBias.Item2);

                layer.blobs[4].mutable_cpu_data = convert(fcWt.Item2);
                layer.blobs[5].mutable_cpu_data = convert(fcBias.Item2);
                layer.blobs[6].mutable_cpu_data = convert(projWt.Item2);
                layer.blobs[7].mutable_cpu_data = convert(projBias.Item2);

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                float[] rgExpected = convertF(m_blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 1e-5f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private ProjectEx getGptProject(string strModel, int nIter)
        {
            ProjectEx p = new ProjectEx("GPT-mini Project");

            DatasetDescriptor ds = new DatasetDescriptor("MODEL");
            p.SetDataset(ds);
            p.OnOverrideModel += new EventHandler<OverrideProjectArgs>(project_OnOverrideModel);
            p.OnOverrideSolver += new EventHandler<OverrideProjectArgs>(project_OnOverrideSolver);
            
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\gpt-mini\\" + strModel + ".prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\gpt-mini\\" + strModel + "-solver.prototxt");

            p.LoadModelFile(strModelFile);
            RawProto proto = RawProtoFile.LoadFromFile(strSolverFile);

            RawProto iter = proto.FindChild("max_iter");
            iter.Value = nIter.ToString();

            p.SolverDescription = proto.ToString();

            return p;
        }

        void project_OnOverrideSolver(object sender, OverrideProjectArgs e)
        {
            RawProto proto = e.Proto;

            RawProto display = proto.FindChild("display");
            if (display != null)
                display.Value = "100";

            RawProto test_iter = proto.FindChild("test_iter");
            if (test_iter != null)
                test_iter.Value = "100";

            RawProto test_interval = proto.FindChild("test_interval");
            if (test_interval != null)
                test_interval.Value = "100";
        }

        void project_OnOverrideModel(object sender, OverrideProjectArgs e)
        {
        }

        // WORK IN PROGRESS
        public void TestTrainingGptMini(int nIter)
        {
            m_log.EnableTrace = true;
            m_log.WriteHeader("GPT-Mini - Test Train");

            try
            {
                m_ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
                ProjectEx project = getGptProject("gpt-mini", nIter);

                m_ctrl.OnTestingIteration += ctrl_OnTestingIteration;
                m_ctrl.Load(Phase.TRAIN, project, null, null, false, null, false);

                m_netRun = m_ctrl.GetInternalNet(Phase.RUN);
                m_dataLayer = m_ctrl.GetInternalNet(Phase.TEST).layers[0] as TokenizedDataLayer<T>;

                string strTestInput = ("O God, O God!").PadLeft((int)m_dataLayer.layer_param.tokenized_data_param.block_size, ' ');
                m_rgTestInput = new float[strTestInput.Length];
                for (int i = 0; i < strTestInput.Length; i++)
                {
                    m_rgTestInput[i] = (int)strTestInput[i];
                }

                m_blobX.Reshape(1, m_rgTestInput.Length, 1, 1);
                m_blobX.mutable_cpu_data = convert(m_rgTestInput);

                m_ctrl.Train();
            }
            finally
            {
                m_ctrl.Dispose();
                m_ctrl = null;
            }
        }

        private void ctrl_OnTestingIteration(object sender, TestingIterationArgs<T> e)
        {
            if (e.Iteration > 0 && e.Iteration % 500 == 0)
            {
                m_ctrl.UpdateRunWeights();

                generate(m_netRun, m_blobX, m_blobY, 500);
                
                m_dataLayer.Detokenize(m_blobY, m_blobY);
                float[] rgY = convertF(m_blobY.mutable_cpu_data);
                string strOut = "";

                for (int i = 0; i < rgY.Length; i++)
                {
                    strOut += (char)rgY[i];
                }

                m_log.WriteLine(strOut);
            }
        }

        // WORK IN PROGRESS
        private void generate(Net<T> net, Blob<T> blobIdx, Blob<T> blobY, int nMaxNewTokens)
        {
            Blob<T> blobLogits = net.blob_by_name("logits");
            BlobCollection<T> colBottom = new BlobCollection<T>() { blobIdx };
            BlobCollection<T> colTop;
            double dfLoss;
            List<float> rgfIdx = new List<float>();

            float[] rgIdx = convertF(blobIdx.mutable_cpu_data);
            rgfIdx.AddRange(rgIdx);

            for (int i = 0; i < nMaxNewTokens; i++)
            {                
                // Forward pass to get the logits.
                colBottom[0] = blobIdx;
                colTop = net.Forward(colBottom, out dfLoss, true);

                int nIdx = colTop[0].count();
                float fIdxVal = convertF(colTop[0].GetData(nIdx - 1));
                rgfIdx.Add(fIdxVal);
                // Clip to block size.
                rgfIdx.RemoveAt(0);

                blobIdx.mutable_cpu_data = convert(rgfIdx.ToArray());
            }

            blobY.Reshape(1, rgfIdx.Count(), 1, 1);
            blobY.mutable_cpu_data = convert(rgfIdx.ToArray());
        }

        public void TestTrainingGptPico(bool bBatch, int nHeads)
        {
            string strModel = "gpt-pico-model";
            if (nHeads > 1)
                strModel += nHeads.ToString();
            if (bBatch)
                strModel += "B";

            m_log.EnableTrace = true;
            m_log.WriteHeader("GPT-Pico - Test Train");

            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
            ProjectEx project = getGptProject(strModel, 1);

            ctrl.Load(Phase.TRAIN, project, null, null, false, null, false, false);

            Net<T> net = ctrl.GetInternalNet(Phase.TRAIN);
            
            Tuple<List<int>, float[]> idx = Fill(strModel, "1_idx", m_log);
            Tuple<List<int>, float[]> pos = Fill(strModel, "1_pos", m_log);
            Tuple<List<int>, float[]> tok_emb = Fill(strModel, "2_tok_emb", m_log);
            Tuple<List<int>, float[]> pos_emb = Fill(strModel, "3_pos_emb", m_log);
            Tuple<List<int>, float[]> x4 = Fill(strModel, "4_x", m_log);
            Tuple<List<int>, float[]> x5 = Fill(strModel, "5_x", m_log);
            Tuple<List<int>, float[]> x6 = Fill(strModel, "6_x", m_log);
            Tuple<List<int>, float[]> logits = Fill(strModel, "7_logits", m_log);
            Tuple<List<int>, float[]> target = Fill(strModel, "7_targets", m_log);
            Tuple<List<int>, float[]> loss = Fill(strModel, "7_loss", m_log);
            
            Tuple<List<int>, float[]> grad_1_loss = Fill(strModel, "grad_1_loss", m_log);
            Tuple<List<int>, float[]> grad_2_logits = Fill(strModel, "grad_2_logits", m_log);
            Tuple<List<int>, float[]> grad_3_x = Fill(strModel, "grad_3_x", m_log);
            Tuple<List<int>, float[]> grad_4_x = Fill(strModel, "grad_4_x", m_log);
            Tuple<List<int>, float[]> grad_5_x = Fill(strModel, "grad_5_x", m_log);
            Tuple<List<int>, float[]> grad_6_pos_emb = Fill(strModel, "grad_6_pos_emb", m_log);
            Tuple<List<int>, float[]> grad_6_tok_emb = Fill(strModel, "grad_6_tok_emb", m_log);
            
            // Load all weights.

            Tuple<List<int>, float[]> gpt_wte_weight = Fill(strModel, "gpt_wte_weight", m_log);
            net.learnable_parameters[0].mutable_cpu_data = convert(gpt_wte_weight.Item2);

            Tuple<List<int>, float[]> gpt_wpe_weight = Fill(strModel, "gpt_wpe_weight", m_log);
            net.learnable_parameters[1].mutable_cpu_data = convert(gpt_wpe_weight.Item2);

            Tuple<List<int>, float[]> attn_weight = Fill(strModel, "attn_weight", m_log);
            net.learnable_parameters[2].mutable_cpu_data = convert(attn_weight.Item2);

            Tuple<List<int>, float[]> attn_bias = Fill(strModel, "attn_bias", m_log);
            net.learnable_parameters[3].mutable_cpu_data = convert(attn_bias.Item2);

            Tuple<List<int>, float[]> attn_proj_weight = Fill(strModel, "attn_proj_weight", m_log);
            net.learnable_parameters[4].mutable_cpu_data = convert(attn_proj_weight.Item2);

            Tuple<List<int>, float[]> attn_proj_bias = Fill(strModel, "attn_proj_bias", m_log);
            net.learnable_parameters[5].mutable_cpu_data = convert(attn_proj_bias.Item2);
            
            Tuple<List<int>, float[]> fcWt = Fill(strModel, "fc_weight", m_log);
            net.learnable_parameters[6].mutable_cpu_data = convert(fcWt.Item2);

            Tuple<List<int>, float[]> fcBias = Fill(strModel, "fc_bias", m_log);
            net.learnable_parameters[7].mutable_cpu_data = convert(fcBias.Item2);
            
            Tuple<List<int>, float[]> projWt = Fill(strModel, "proj_weight", m_log);
            net.learnable_parameters[8].mutable_cpu_data = convert(projWt.Item2);

            Tuple<List<int>, float[]> projBias = Fill(strModel, "proj_bias", m_log);
            net.learnable_parameters[9].mutable_cpu_data = convert(projBias.Item2);
            
            Tuple<List<int>, float[]> gpt_lm_head_weight = Fill(strModel, "gpt_lm_head_weight", m_log);
            net.learnable_parameters[10].mutable_cpu_data = convert(gpt_lm_head_weight.Item2);

            // Test First Forward Pass            
            net.ForwardFromTo(0, 0); // TokenizedData
            verifyBlob(m_log, net.blobs[0], idx);
            verifyBlob(m_log, net.blobs[1], pos);
            verifyBlob(m_log, net.blobs[2], target);
            
            net.ForwardFromTo(1, 1); // Embed wte
            verifyBlob(m_log, net.blobs[3], tok_emb);

            net.ForwardFromTo(2, 2); // Embed wpe
            verifyBlob(m_log, net.blobs[4], pos_emb);

            net.ForwardFromTo(3, 3); // EltWise x = (tok_emb + pos_emb)
            verifyBlob(m_log, net.blobs[5], x4);

            net.ForwardFromTo(4, 4); // TransformerBlock x = block(x)
            verifyBlob(m_log, net.blobs[6], x5);

            net.ForwardFromTo(5, 5); // LayerNorm x = ln(x)
            verifyBlob(m_log, net.blobs[7], x6);

            net.ForwardFromTo(6, 6); // InnerProduct logit = lm_head(x)
            verifyBlob(m_log, net.blobs[8], logits);

            net.ForwardFromTo(7); // loss = F.cross_entropy(logits, targets)
            verifyBlob(m_log, net.blobs[9], loss);

            // Test Backward Pass
            verifyBlob(m_log, net.blobs[9], grad_1_loss, true);

            net.Backward(7, 7); // loss = F.cross_entropy(logits, targets)
            verifyBlob(m_log, net.blobs[8], grad_2_logits, true); // logits diff

            net.Backward(6, 6); // InnerProduct logit = lm_head(x)
            verifyBlob(m_log, net.blobs[7], grad_3_x, true); // ln x diff

            net.Backward(5, 5); // LayerNorm x = ln(x)
            verifyBlob(m_log, net.blobs[6], grad_4_x, true); // tbf x diff

            net.Backward(4, 4); // TransformerBlock x = block(x)
            verifyBlob(m_log, net.blobs[5], grad_5_x, true); // x diff

            net.Backward(3, 3); // x = tok_emb + pos_emb
            verifyBlob(m_log, net.blobs[4], grad_6_pos_emb, true); // tbf x diff
            verifyBlob(m_log, net.blobs[3], grad_6_tok_emb, true); // tbf x diff

            net.Backward(2, 2);
            net.Backward(1, 1);
            
            // Apply Gradients
            Solver<T> solver = ctrl.GetInternalSolver();
            solver.ApplyUpdate();


            // Load second pass test data.
            Tuple<List<int>, float[]> idx_2 = Fill(strModel, "1_idx", m_log, "pass_2");
            Tuple<List<int>, float[]> pos_2 = Fill(strModel, "1_pos", m_log, "pass_2");
            Tuple<List<int>, float[]> tok_emb_2 = Fill(strModel, "2_tok_emb", m_log, "pass_2");
            Tuple<List<int>, float[]> pos_emb_2 = Fill(strModel, "3_pos_emb", m_log, "pass_2");
            Tuple<List<int>, float[]> x4_2 = Fill(strModel, "4_x", m_log, "pass_2");
            Tuple<List<int>, float[]> x5_2 = Fill(strModel, "5_x", m_log, "pass_2");
            Tuple<List<int>, float[]> x6_2 = Fill(strModel, "6_x", m_log, "pass_2");
            Tuple<List<int>, float[]> logits_2 = Fill(strModel, "7_logits", m_log, "pass_2");
            Tuple<List<int>, float[]> target_2 = Fill(strModel, "7_targets", m_log, "pass_2");
            Tuple<List<int>, float[]> loss_2 = Fill(strModel, "7_loss", m_log, "pass_2");

            // Verify gradient application
            Tuple<List<int>, float[]> gpt_lm_head_weight_2 = Fill(strModel, "gpt_lm_head_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[10], gpt_lm_head_weight_2);

            Tuple<List<int>, float[]> projBias_2 = Fill(strModel, "proj_bias", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[9], projBias_2);

            Tuple<List<int>, float[]> projWt_2 = Fill(strModel, "proj_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[8], projWt_2);

            Tuple<List<int>, float[]> fcBias_2 = Fill(strModel, "fc_bias", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[7], fcBias_2);

            Tuple<List<int>, float[]> fcWt_2 = Fill(strModel, "fc_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[6], fcWt_2);

            Tuple<List<int>, float[]> attn_proj_bias_2 = Fill(strModel, "attn_proj_bias", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[5], attn_proj_bias_2);

            Tuple<List<int>, float[]> attn_proj_weight_2 = Fill(strModel, "attn_proj_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[4], attn_proj_weight_2);

            Tuple<List<int>, float[]> attn_bias_2 = Fill(strModel, "attn_bias", m_log, "pass_2");
/*bug->*/ //verifyBlob(m_log, net.learnable_parameters[3], attn_bias_2);

            Tuple<List<int>, float[]> attn_weight_2 = Fill(strModel, "attn_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[2], attn_weight_2);

            Tuple<List<int>, float[]> gpt_wpe_weight_2 = Fill(strModel, "gpt_wpe_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[1], gpt_wpe_weight_2);

            Tuple<List<int>, float[]> gpt_wte_weight_2 = Fill(strModel, "gpt_wte_weight", m_log, "pass_2");
            verifyBlob(m_log, net.learnable_parameters[0], gpt_wte_weight_2);

            
            // Test Second Forward Pass to see if applied gradients match
            net.ForwardFromTo(0, 0); // TokenizedData
            verifyBlob(m_log, net.blobs[0], idx_2);
            verifyBlob(m_log, net.blobs[1], pos_2);
            verifyBlob(m_log, net.blobs[2], target_2);

            net.ForwardFromTo(1, 1); // Embed wte
            verifyBlob(m_log, net.blobs[3], tok_emb_2);

            net.ForwardFromTo(2, 2); // Embed wpe
            verifyBlob(m_log, net.blobs[4], pos_emb_2);

            net.ForwardFromTo(3, 3); // EltWise x = (tok_emb + pos_emb)
            verifyBlob(m_log, net.blobs[5], x4_2);

            net.ForwardFromTo(4, 4); // TransformerBlock x = block(x)
            verifyBlob(m_log, net.blobs[6], x5_2);

            net.ForwardFromTo(5, 5); // LayerNorm x = ln(x)
            verifyBlob(m_log, net.blobs[7], x6_2);

            net.ForwardFromTo(6, 6); // InnerProduct logit = lm_head(x)
            verifyBlob(m_log, net.blobs[8], logits_2);

            net.ForwardFromTo(7); // loss = F.cross_entropy(logits, targets)
            verifyBlob(m_log, net.blobs[9], loss_2);

            ctrl.Dispose();
        }

        private void verifyBlob(Log log, Blob<T> blob, Tuple<List<int>, float[]> data, bool bDiff = false)
        {
            if (blob.count() != data.Item2.Length)
                m_log.FAIL(blob.Name + ": The blob count does not match the data count!");

            float[] rgData = (bDiff) ? convertF(blob.mutable_cpu_diff) : convertF(blob.mutable_cpu_data);

            for (int i = 0; i < rgData.Length; i++)
            {
                float fActual = rgData[i];
                float fExpected = data.Item2[i];
                float fDiff = fActual - fExpected;

                if (Math.Abs(fDiff) > 0.0001)
                    m_log.FAIL(blob.Name + ": The data at index " + i.ToString() + " does not match!");
            }
        }
    }
}
