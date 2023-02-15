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
using System.IO.Compression;
using System.Net;

namespace MyCaffe.test
{
    [TestClass]
    public class TestTransformerBlockLayer
    {
        [TestMethod]
        public void TestTrain()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrain();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
        [TestMethod]
        public void TestInference()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestInference();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForward(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForward("pico");
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
                    t.TestBackward("pico");
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
                    // Change to 10000 for full training.
                    t.TestTrainingGptMini(500);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingGptMiniTestMany()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    // Change to 10000 for full training.
                    t.TestTrainingGptMini(500);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingGptMini1()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrainingGptMini1(10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        public void TestGptHuggingFaceImport()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTestHuggingFaceImport();
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
        void TestTrain();
        void TestInference();
        void TestForward(string strModel);
        void TestBackward(string strModel);
        void TestGradient();

        void TestForwardPico(bool bBatch, uint nHeads);
        void TestBackwardPico(bool bBatch, uint nHeads);
        void TestGradientPico(bool bBatch, uint nHeads);
        void TestForwardMini();
        void TestTrainingGptMini(int nIter);
        void TestTrainingGptMiniTestMany(int nIter);
        void TestTrainingGptMini1(int nIter);
        void TestTestHuggingFaceImport();
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
        Blob<T> m_blobY = null;
        Blob<T> m_blobX = null;
        Blob<T> m_blobPos = null;
        float[] m_rgTestInput;
        MyCaffeControl<T> m_ctrl = null;
        Random m_random = new Random(3407);
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public TransformerBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;

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
            base.dispose();
        }

        public Tuple<List<int>, float[]> Fill(string strGpt, string strName, Log log, string strPass = "", string strPathOvr = null)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\gpt\\" + strGpt + "\\"; 
            if (!string.IsNullOrEmpty(strPathOvr))
                strPath = strPathOvr;
            string strFile = strPath;

            if (!string.IsNullOrEmpty(strPass))
                strFile += strPass + "\\";

            strFile += strName + ".txt";

            string[] rgstrLines = System.IO.File.ReadAllLines(strFile);
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

        public void TestForwardPico(bool bBatch, uint nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 3;
            p.transformer_block_param.block_size = 4;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

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
                blobY.Reshape(y.Item1);
                blobY.mutable_cpu_data = convert(y.Item2);
                
                Tuple<List<int>, float[]> attnBias = Fill(strModel, "blk_attn_bias", m_log);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "blk_attn_weight", m_log);
                Tuple<List<int>, float[]> attnProjBias = Fill(strModel, "blk_attn_proj_bias", m_log);
                Tuple<List<int>, float[]> attnProjWt = Fill(strModel, "blk_attn_proj_weight", m_log);

                Tuple<List<int>, float[]> fcBias = Fill(strModel, "blk_fc_bias", m_log);
                Tuple<List<int>, float[]> fcWt = Fill(strModel, "blk_fc_weight", m_log);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "blk_proj_bias", m_log);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "blk_proj_weight", m_log);

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
                float[] rgExpected = convertF(blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 1e-7f;
                    float fDiff = fActual - fExpected;

                    if (Math.Abs(fDiff) > fErr)
                        m_log.FAIL("The values are not as expected!");
                }
            }
            finally
            {
                blobY.Dispose();
                layer.Dispose();
            }
        }

        public void TestBackwardPico(bool bBatch, uint nHeads)
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

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "grad_1_y", m_log, "iter_0");                
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "grad_10_x", m_log, "iter_0");

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
                    float fErr = 1e-7f;
                    float fDiff = fActual - fExpected;

                    if (Math.Abs(fDiff) > fErr)
                        m_log.FAIL("The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        // Currently Fails on bBatch=True, nHeads=3
        public void TestGradientPico(bool bBatch, uint nHeads)
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
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

            try
            {
                string strModel = "gpt-mini-blk";

                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "1_x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y = Fill(strModel, "10_y", m_log);
                blobY.Reshape(y.Item1);
                blobY.mutable_cpu_data = convert(y.Item2);

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
                float[] rgExpected = convertF(blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
#warning "TODO: TestTransformerBlock ForwardMini - need to tighten up error threshold and test."                    
                    float fErr = 1e-3f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                blobY.Dispose();
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

                m_blobY = m_ctrl.CreateBlob("results");
                m_blobX = m_ctrl.CreateBlob("data");

                m_netRun = m_ctrl.GetInternalNet(Phase.RUN);
                m_dataLayer = m_ctrl.GetInternalNet(Phase.TEST).layers[0] as TokenizedDataLayer<T>;
                
                string strTestInput = "O God, O God!";
                
                m_rgTestInput = new float[strTestInput.Length];
                for (int i = 0; i < strTestInput.Length; i++)
                {
                    m_rgTestInput[i] = (int)strTestInput[i];
                }

                int[] rgShape = new int[] { 1, m_rgTestInput.Length };
                m_blobX.Reshape(rgShape);
                m_blobX.mutable_cpu_data = convert(m_rgTestInput);

                m_ctrl.Train();
            }
            finally
            {
                dispose1(ref m_blobY);
                dispose1(ref m_blobX);

                m_ctrl.Dispose();
                m_ctrl = null;
            }
        }

        public void TestTrainingGptMiniTestMany(int nIter)
        {
            m_log.EnableTrace = true;
            m_log.WriteHeader("GPT-Mini - Test Train");

            try
            {
                m_ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
                ProjectEx project = getGptProject("gpt-mini", nIter);

                m_ctrl.Load(Phase.TRAIN, project, null, null, false, null, false);


                
                m_blobY = m_ctrl.CreateBlob("results");
                m_blobX = m_ctrl.CreateBlob("data");

                string strTestInput = "O God, O God!";
                
                m_rgTestInput = new float[strTestInput.Length];
                for (int i = 0; i < strTestInput.Length; i++)
                {
                    m_rgTestInput[i] = (int)strTestInput[i];
                }

                int[] rgShape = new int[] { 1, m_rgTestInput.Length };
                m_blobX.Reshape(rgShape);
                m_blobX.mutable_cpu_data = convert(m_rgTestInput);

                PropertySet res;
                m_ctrl.Train(nIter);
                res = m_ctrl.TestMany(new PropertySet("InputData=O God! O God!;Max=500"));
                m_log.WriteLine(res.GetProperty("Results"));
            }
            finally
            {
                dispose1(ref m_blobY);
                dispose1(ref m_blobX);

                m_ctrl.Dispose();
                m_ctrl = null;
            }
        }

        public void TestTrainingGptMini1(int nIter)
        {
            m_log.EnableTrace = true;
            m_log.WriteHeader("GPT-Mini - Test Train");

            try
            {
                m_ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
                ProjectEx project = getGptProject("gpt-mini1", nIter);

                m_ctrl.OnTestingIteration += ctrl_OnTestingIteration;
                m_ctrl.Load(Phase.TRAIN, project, null, null, false, null, false);

                Solver<T> solver = m_ctrl.GetInternalSolver();
                solver.OnStart += Solver_OnStart;
                solver.OnTrainingIteration += Solver_OnTrainingIteration;

                m_blobY = m_ctrl.CreateBlob("results");
                m_blobX = m_ctrl.CreateBlob("data");

                m_netRun = m_ctrl.GetInternalNet(Phase.RUN);
                m_dataLayer = m_ctrl.GetInternalNet(Phase.TEST).layers[0] as TokenizedDataLayer<T>;

                string strTestInput = "O God, O God!";
                
                List<int> rgTokens = ((TokenizedDataLayer<T>)m_dataLayer).Tokenize(strTestInput, false, false);

                m_rgTestInput = new float[rgTokens.Count];
                for (int i = 0; i < rgTokens.Count; i++)
                {
                    m_rgTestInput[i] = rgTokens[i];
                }

                int[] rgShape = new int[] { 1, m_rgTestInput.Length };
                m_blobX.Reshape(rgShape);
                m_blobX.mutable_cpu_data = convert(m_rgTestInput);

                m_ctrl.Train(10);
            }
            finally
            {
                dispose1(ref m_blobY);
                dispose1(ref m_blobX);

                m_dataLayer = null;
                m_netRun = null;

                m_ctrl.Dispose();
                m_ctrl = null;
            }
        }

        private void Solver_OnTrainingIteration(object sender, TrainingIterationArgs<T> e)
        {
            string strModel = "gpt-mini1";
            string strIter = "iter_" + (e.Iteration - 1).ToString();
            Tuple<List<int>, float[]> loss = Fill(strModel, "7_loss", m_log, strIter);

            double dfDiff = Math.Abs(loss.Item2[0] - e.Loss);

            m_log.WriteLine("Loss = " + e.Loss.ToString() + " Difference from expected = " + dfDiff.ToString());
        }

        private void Solver_OnStart(object sender, EventArgs e)
        {
            Solver<T> solver = sender as Solver<T>;

            if (solver.iter == 0)
            {
                string strModel = "gpt-mini1";
                string strPath = null;

                Tuple<List<int>, float[]> gpt_wte_weight = Fill(strModel, "gpt_wte_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> gpt_wpe_weight = Fill(strModel, "gpt_wpe_weight", m_log, "iter_0", strPath);

                Tuple<List<int>, float[]> attn_weight1 = Fill(strModel, "tfb_1_attn_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> attn_bias1 = Fill(strModel, "tfb_1_attn_bias", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> attn_proj_weight1 = Fill(strModel, "tfb_1_attn_proj_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> attn_proj_bias1 = Fill(strModel, "tfb_1_attn_proj_bias", m_log, "iter_0", strPath);

                Tuple<List<int>, float[]> fc_weights1 = Fill(strModel, "tfb_1_fc_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> fc_bias1 = Fill(strModel, "tfb_1_fc_bias", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> proj_weight1 = Fill(strModel, "tfb_1_proj_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> proj_bias1 = Fill(strModel, "tfb_1_proj_bias", m_log, "iter_0", strPath);

                Tuple<List<int>, float[]> attn_weight2 = Fill(strModel, "tfb_2_attn_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> attn_bias2 = Fill(strModel, "tfb_2_attn_bias", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> attn_proj_weight2 = Fill(strModel, "tfb_2_attn_proj_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> attn_proj_bias2 = Fill(strModel, "tfb_2_attn_proj_bias", m_log, "iter_0", strPath);

                Tuple<List<int>, float[]> fc_weights2 = Fill(strModel, "tfb_2_fc_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> fc_bias2 = Fill(strModel, "tfb_2_fc_bias", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> proj_weight2 = Fill(strModel, "tfb_2_proj_weight", m_log, "iter_0", strPath);
                Tuple<List<int>, float[]> proj_bias2 = Fill(strModel, "tfb_2_proj_bias", m_log, "iter_0", strPath);

                Tuple<List<int>, float[]> gpt_lm_head_weight = Fill(strModel, "gpt_lm_head_weight", m_log, "iter_0", strPath);

                Net<T> net = solver.TrainingNet;

                setData(net.learnable_parameters[0], gpt_wte_weight);
                setData(net.learnable_parameters[1], gpt_wpe_weight);
                
                setData(net.learnable_parameters[2], attn_weight1);
                setData(net.learnable_parameters[3], attn_bias1);
                setData(net.learnable_parameters[4], attn_proj_weight1);
                setData(net.learnable_parameters[5], attn_proj_bias1);
                setData(net.learnable_parameters[6], fc_weights1);
                setData(net.learnable_parameters[7], fc_bias1);
                setData(net.learnable_parameters[8], proj_weight1);
                setData(net.learnable_parameters[9], proj_bias1);

                setData(net.learnable_parameters[10], attn_weight2);
                setData(net.learnable_parameters[11], attn_bias2);
                setData(net.learnable_parameters[12], attn_proj_weight2);
                setData(net.learnable_parameters[13], attn_proj_bias2);
                setData(net.learnable_parameters[14], fc_weights2);
                setData(net.learnable_parameters[15], fc_bias2);
                setData(net.learnable_parameters[16], proj_weight2);
                setData(net.learnable_parameters[17], proj_bias2);

                setData(net.learnable_parameters[18], gpt_lm_head_weight);
            }
        }

        private void setData(Blob<T> blob, Tuple<List<int>, float[]> data)
        {
            if (!blob.CompareShape(data.Item1))
                throw new Exception("The shapes do not match!");

            blob.mutable_cpu_data = convert(data.Item2);
        }

        private void ctrl_OnTestingIteration(object sender, TestingIterationArgs<T> e)
        {
            if (e.Iteration > 0 && e.Iteration % 500 == 0)
            {
                m_ctrl.UpdateRunWeights(false, false);
                generate(m_netRun, m_blobX, m_blobY, 500, (int)m_dataLayer.layer_param.tokenized_data_param.block_size, 65, 10);

                string strOut = "";
                float[] rgY = convertF(m_blobY.mutable_cpu_data);
                int nCount = m_blobY.count(1);

                for (int i = 0; i < m_blobY.num; i++)
                {
                    string str1 = m_dataLayer.Detokenize(rgY, i * nCount, nCount);
                    strOut += str1 + Environment.NewLine;
                }

                m_log.WriteLine(strOut);
            }
        }

        private void fillPos(Blob<T> blobX, Blob<T> blobPos)
        {
            int[] rgShape = new int[] { 1, blobX.channels };

            blobPos.Reshape(rgShape);
            float[] rgPos = new float[blobX.channels];

            for (int i = 0; i < rgPos.Length; i++)
            {
                rgPos[i] = i;
            }

            blobPos.mutable_cpu_data = convert(rgPos);
        }

        private int getNextIndex(Blob<T> blob, int nVocabCount, int nTopK, Layer<T> softmax)
        {
            List<Tuple<string, int, double>> rg = m_dataLayer.PostProcessLogitsOutput(-1, blob, softmax, 10);
            return rg[0].Item2;
        }

        private int getNextIndexOld(Blob<T> blob, int nVocabCount, int nTopK, Layer<T> softmax)
        {           
            float[] rgData = convertF(blob.mutable_cpu_data);
            float[] rgLogits = new float[nVocabCount];
            int nIdxStart = blob.count() - nVocabCount;
            Dictionary<int, float> rgTopK = new Dictionary<int, float>();

            for (int i = nIdxStart; i<blob.count(); i++)
            {
                float fVal = rgData[i];
                rgTopK.Add(i - nIdxStart, fVal);

                if (rgTopK.Count > nTopK)
                {
                    float fMin = float.MaxValue;
                    int nMinIdx = -1;

                    foreach (KeyValuePair<int, float> kv in rgTopK)
                    {
                        if (kv.Value < fMin)
                        {
                            fMin = kv.Value;
                            nMinIdx = kv.Key;
                        }
                    }

                    rgTopK.Remove(nMinIdx);
                }                
            }

            for (int i = 0; i < rgLogits.Count(); i++)
            {
                if (rgTopK.ContainsKey(i))
                    rgLogits[i] = rgTopK[i];
                else
                    rgLogits[i] = -float.MaxValue;
            }

            m_blobX.Reshape(1, 1, nVocabCount, 1);
            m_blobX.mutable_cpu_data = convert(rgLogits);

            BlobCollection<T> colBottom = new BlobCollection<T>() { m_blobX };
            BlobCollection<T> colTop = new BlobCollection<T>() { m_blobY };
            softmax.Forward(colBottom, colTop);

            float[] rgProb = convertF(m_blobY.mutable_cpu_data);
            float fRand = (float)m_random.NextDouble();
            float fTotal = 0;

            for (int i = 0; i < rgProb.Length; i++)
            {
                fTotal += rgProb[i];

                if (fTotal >= fRand)
                    return i;
            }

            return rgProb.Length - 1;
        }

        private void generate(Net<T> net, Blob<T> blobIdx, Blob<T> blobY, int nMaxNewTokens, int nBlockSize, int nVocabSize, int nTopK)
        {
            Blob<T> blobLogits = net.blob_by_name("logits");
            Layer<T> softmax = net.FindLastLayer(LayerParameter.LayerType.SOFTMAX);
            BlobCollection<T> colBottom = new BlobCollection<T>() { blobIdx };
            double dfLoss;
            List<float> rgfIdx = new List<float>();
            List<float> rgfIdxOut = new List<float>();
            int[] rgShape;

            float[] rgIdx = convertF(blobIdx.mutable_cpu_data);
            rgfIdx.AddRange(rgIdx);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < nMaxNewTokens; i++)
            {                
                // Forward pass to get the logits.
                net.Forward(colBottom, out dfLoss, true);                                
                float fIdxVal = getNextIndex(blobLogits, nVocabSize, nTopK, softmax);

                rgfIdx.Add(fIdxVal);
                // Clip to block size.
                if (rgfIdx.Count > nBlockSize)
                    rgfIdx.RemoveAt(0);

                rgfIdxOut.Add(fIdxVal);

                rgShape = new int[] { 1, rgfIdx.Count };
                if (blobIdx.channels < rgfIdx.Count)
                    blobIdx.Reshape(rgShape);
                blobIdx.mutable_cpu_data = convert(rgfIdx.ToArray());
                
                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    sw.Restart();
                    double dfPct = (double)i / nMaxNewTokens;
                    m_log.WriteLine("Generating at " + dfPct.ToString("P") + "...");
                }
            }

            rgShape = new int[] { 1, rgfIdxOut.Count };
            blobY.Reshape(rgShape);
            blobY.mutable_cpu_data = convert(rgfIdxOut.ToArray());
        }

        private void verifyBlob(Log log, Blob<T> blob, Tuple<List<int>, float[]> data, bool bDiff = false, double dfErr = 1e-5)
        {
            if (blob.count() != data.Item2.Length)
                m_log.FAIL(blob.Name + ": The blob count does not match the data count!");

            float[] rgData = (bDiff) ? convertF(blob.mutable_cpu_diff) : convertF(blob.mutable_cpu_data);

            for (int i = 0; i < rgData.Length; i++)
            {
                float fActual = rgData[i];
                float fExpected = data.Item2[i];
                float fDiff = fActual - fExpected;

                if (Math.Abs(fDiff) > dfErr)
                    m_log.FAIL(blob.Name + ": The data at index " + i.ToString() + " does not match!");
            }
        }

        // WORK IN PROGRESS
        //
        // TODO: All that remains is to implement the GPT2 tokenizer and load the GPT2 vocabulary file
        // found at:
        // @see [GPT2 Vocabulary File](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json)
        public void TestTestHuggingFaceImport()
        {
            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\huggingface\\gpt2\\";
            string strWtPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\huggingface\\gpt2\\gpt2_weights\\";
            string strSrc = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\input.txt";
            TextInputData input = new TextInputData(strSrc);
            int nVocabSize = 50257; // OpenAi's model vocabulary
            int nBlockSize = 1024;
            int nLayerCount = 12;
            int nHeads = 12;
            int nEmbed = 768;
            int nBatchSize = 1;

            string strModel = createGptModel("GPT2", strSrc, nVocabSize, (uint)nLayerCount, (uint)nHeads, (uint)nEmbed, (uint)nBlockSize, nBatchSize);

            m_log.EnableTrace = true;
            m_log.WriteHeader("GPT-Mini - Test Train");

            try
            {
                m_log.WriteLine("Downloading GPT2 model weights...");
                loadWeightFiles(strModelPath, strModel);

                m_ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
                m_log.WriteLine("Loading GPT2 model...");
                BlobShape shape = new BlobShape(new List<int>() { 1, nBlockSize });
                m_ctrl.LoadToRun(strModel, null, shape, null, null, false, false);

                m_blobY = m_ctrl.CreateBlob("results");
                m_blobX = m_ctrl.CreateBlob("data");
                m_blobPos = m_ctrl.CreateBlob("pos");
                m_blobPos.Reshape(1, 128, 1, 1);
                
                m_netRun = m_ctrl.GetInternalNet(Phase.RUN);
                m_log.WriteLine("Loading GPT2 pre-trained weights into the model...");
                loadWeights(m_netRun, strWtPath);

                if (m_dataLayer == null)
                {
                    LayerParameter param_data = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA);
                    param_data.tokenized_data_param.source = strSrc;
                    param_data.tokenized_data_param.block_size = (uint)nBlockSize;
                    param_data.tokenized_data_param.batch_size = (uint)nBatchSize;
                    param_data.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
                    m_dataLayer = new TokenizedDataLayer<T>(m_cuda, m_log, param_data, null, null);

                    BlobCollection<T> colBottom = new BlobCollection<T>();
                    BlobCollection<T> colTop = new BlobCollection<T>() { m_blobX, m_blobPos, m_blobX };

                    m_dataLayer.LayerSetUp(colBottom, colTop);
                }

                string strInput = System.IO.File.ReadAllText(strSrc);
                int nIdx = m_random.Next(strInput.Length - (2 * nBlockSize));

                List<int> rgTokens = m_dataLayer.Tokenize(strInput.Substring(nIdx, nBlockSize), false, false);

                m_rgTestInput = new float[nBlockSize];
                for (int i = 0; i < nBlockSize; i++)
                {
                    if (i < rgTokens.Count)
                        m_rgTestInput[i] = rgTokens[i];
                }

                int[] rgShape = new int[] { 1, m_rgTestInput.Length };
                m_blobX.Reshape(rgShape);
                m_blobX.mutable_cpu_data = convert(m_rgTestInput);

                generate(m_netRun, m_blobX, m_blobY, 500, nBlockSize, nVocabSize, 10);

                string strOut = "";
                float[] rgY = convertF(m_blobY.mutable_cpu_data);
                int nCount = m_blobY.count(1);

                for (int i = 0; i < m_blobY.num; i++)
                {
                    string str1 = m_dataLayer.Detokenize(rgY, i * nCount, nCount);
                    strOut += str1 + Environment.NewLine;
                }

                m_log.WriteLine(strOut);
            }
            finally
            {
                dispose1(ref m_blobY);
                dispose1(ref m_blobX);
                dispose1(ref m_blobPos);

                if (m_dataLayer != null)
                {
                    m_dataLayer.Dispose();
                    m_dataLayer = null;
                }

                m_ctrl.Dispose();
                m_ctrl = null;
            }
        }

        private void setData(Blob<T> blob, string strPath, string strName)
        {
            m_log.WriteLine("Loading blob '" + strName + "' from '" + strPath + "'...");
            string strNpyFile = strPath + strName + ".npy";
            string strShapeFile = strPath + strName + ".txt";

            List<int> rgShape = new List<int>();
            string[] rgstr = System.IO.File.ReadAllLines(strShapeFile);
            foreach (string s in rgstr)
            {
                rgShape.Add((int)float.Parse(s));
            }

            if (!blob.CompareShape(rgShape))
                throw new Exception("The shape of the blob '" + blob.Name + "' does not match the shape of the data!");

            blob.LoadFromNumpy(strNpyFile);
        }
        
        private void loadWeights(Net<T> net, string strPath)
        {
            Stopwatch sw = new Stopwatch();
            string strModelPath = strPath;
            int nTfBlockCount = 12;
            
            setData(net.learnable_parameters[0], strModelPath, "gpt_wte_weight");
            setData(net.learnable_parameters[1], strModelPath, "gpt_wpe_weight");
            int nIdx = 2;

            sw.Start();
            
            for (int i = 0; i < nTfBlockCount; i++)
            {
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_bias");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_proj_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_proj_bias");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_fc_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_fc_bias");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_proj_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_proj_bias");
                nIdx++;

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    sw.Restart();
                    double dfPct = (double)i / nTfBlockCount;
                    m_log.WriteLine("Loading pre-trained weights: " + dfPct.ToString("P"));
                }
            }

            setData(net.learnable_parameters[nIdx], strModelPath, "gpt_lm_head_weight");
        }

        private void loadWeightFiles(string strPath, string strModel)
        {
            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strModelFile = strPath + "gpt2_model.prototxt";
            if (!System.IO.File.Exists(strModelFile))
                System.IO.File.WriteAllText(strModelFile, strModel);

            string strWtsFile = strPath + "gpt2_weights.zip";
            if (!System.IO.File.Exists(strWtsFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/gpt2_weights.zip";
                    string strFile1 = "gpt2_weights.zip";
                    string strFile = strPath + strFile1;

                    m_swUpdateTimer.Start();
                    m_dfLastProgress = 0;
                    
                    webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                    webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                    webClient.DownloadFileAsync(new Uri(strUrl), strFile, strFile1);

                    m_evtDownloadDone.WaitOne();
                }
            }

            string strFirstTarget = strPath + "gpt2_weights\\gpt_lm_head_weight.npy";
            if (!System.IO.File.Exists(strFirstTarget))
            {
                ZipFile.ExtractToDirectory(strWtsFile, strPath + "gpt2_weights");
            }
        }

        private void WebClient_DownloadFileCompleted(object sender, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            bool bTraceEnabled = m_log.EnableTrace;
            m_log.EnableTrace = true;
            m_log.WriteLine("Downloading done.");
            m_log.EnableTrace = bTraceEnabled;

            m_evtDownloadDone.Set();
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_swUpdateTimer.Elapsed.TotalMilliseconds >= 1000)
            {
                if (m_dfLastProgress != e.ProgressPercentage)
                {
                    m_dfLastProgress = e.ProgressPercentage;
                    string strFile = e.UserState.ToString();
                    bool bTraceEnabled = m_log.EnableTrace;
                    m_log.EnableTrace = true;

                    m_log.Progress = e.ProgressPercentage / 100.0;
                    m_log.WriteLine("Downloading '" + strFile + "' at " + m_log.Progress.ToString("P") + "...");
                    m_log.EnableTrace = bTraceEnabled;
                }

                m_swUpdateTimer.Restart();
            }
        }

        private string createGptModel(string strName, string strSrc, int nVocabSize, uint nLayers, uint nHeads, uint nEmbed, uint nBlockSize, int nBatchSize)
        {
            NetParameter p = new NetParameter();
            p.name = strName;

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { 1, (int)nBlockSize } });
            input.top.Add("tokdata1");
            input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { 1, (int)nBlockSize } });
            input.top.Add("pos");
            p.layer.Add(input);

            LayerParameter embedTok = new LayerParameter(LayerParameter.LayerType.EMBED);
            embedTok.name = "wte";
            embedTok.embed_param.bias_term = false;
            embedTok.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            embedTok.embed_param.input_dim = (uint)nVocabSize;
            embedTok.embed_param.num_output = (uint)nEmbed;
            embedTok.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay
            embedTok.bottom.Add("tokdata1");
            embedTok.top.Add("tok_emb");
            p.layer.Add(embedTok);

            LayerParameter embedPos = new LayerParameter(LayerParameter.LayerType.EMBED);
            embedPos.name = "wpe";
            embedPos.embed_param.bias_term = false;
            embedPos.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            embedPos.embed_param.input_dim = (uint)nBlockSize;
            embedPos.embed_param.num_output = (uint)nEmbed;
            embedPos.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay
            embedPos.bottom.Add("pos");
            embedPos.top.Add("pos_emb");
            p.layer.Add(embedPos);

            LayerParameter add = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            add.name = "eltwise1";
            add.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            add.eltwise_param.allow_single_batch_input = true;
            add.bottom.Add("tok_emb");
            add.bottom.Add("pos_emb");
            add.top.Add("eltwise1");
            p.layer.Add(add);

            string strTop = "eltwise1";
            LayerParameter drop = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            drop.name = "drop1";
            drop.dropout_param.dropout_ratio = 0.1;
            drop.bottom.Add(strTop);
            drop.top.Add(strTop);
            p.layer.Add(drop);

            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter tfb = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                tfb.name = "tfb" + (i + 1).ToString();
                tfb.transformer_block_param.layers = nLayers;
                tfb.transformer_block_param.heads = nHeads;
                tfb.transformer_block_param.embed = nEmbed;
                tfb.transformer_block_param.block_size = nBlockSize;
                tfb.transformer_block_param.attn_dropout = 0.1;
                tfb.transformer_block_param.resid_dropout = 0.1;
                tfb.parameters.Add(new ParamSpec(1, 1)); // mult lr, decay for Attn c_attn weight/bias
                tfb.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay for Attn c_proj weight/bias
                tfb.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay for FC weight/bias
                tfb.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay for Proj weight/bias
                tfb.bottom.Add(strTop);

                strTop = "tfb" + (i + 1).ToString();
                tfb.top.Add(strTop);
                p.layer.Add(tfb);
            }

            LayerParameter ln = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln.name = "ln1";
            ln.bottom.Add(strTop);
            ln.top.Add("ln1");
            p.layer.Add(ln);

            LayerParameter lm_head = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            lm_head.name = "lm_head";
            lm_head.inner_product_param.bias_term = false;
            lm_head.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            lm_head.inner_product_param.num_output = (uint)nVocabSize;
            lm_head.inner_product_param.axis = 2;
            lm_head.parameters.Add(new ParamSpec(1, 1)); // mult lr, decay
            lm_head.bottom.Add("ln1");
            lm_head.top.Add("logits");
            p.layer.Add(lm_head);

            LayerParameter prob = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            prob.name = "prob";
            prob.softmax_param.axis = 2;
            prob.bottom.Add("logits");
            prob.top.Add("prob");
            p.layer.Add(prob);

            RawProto proto = p.ToProto("root");
            return proto.ToString();
        }
        
        private string loadTestData1(string strModel)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb\\";

            if (!string.IsNullOrEmpty(strModel))
                strPath += strModel + "\\";

            string strFileName = "_transformer_test";

            if (!string.IsNullOrEmpty(strModel))
                strFileName += "_" + strModel;

            strFileName += ".zip";

            string strTestPath = "test\\iter_0";
            string strTestFile = "1_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private void load_state(Layer<T> layer, string strPath)
        {
            layer.blobs[0].LoadFromNumpy(strPath + "blk0.attn.c_attn.weight.npy");
            layer.blobs[1].LoadFromNumpy(strPath + "blk0.attn.c_attn.bias.npy");
            layer.blobs[2].LoadFromNumpy(strPath + "blk0.attn.c_proj.weight.npy");
            layer.blobs[3].LoadFromNumpy(strPath + "blk0.attn.c_proj.bias.npy");
            layer.blobs[4].LoadFromNumpy(strPath + "blk0.c_fc.weight.npy");
            layer.blobs[5].LoadFromNumpy(strPath + "blk0.c_fc.bias.npy");
            layer.blobs[6].LoadFromNumpy(strPath + "blk0.c_proj.weight.npy");
            layer.blobs[7].LoadFromNumpy(strPath + "blk0.c_proj.bias.npy");
        }

        /// <summary>
        /// Test the forward pass of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestForward(string strModel)
        {
            string strPath = loadTestData1(strModel);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;

            if (strModel == "pico")
            {
                p.transformer_block_param.heads = 1;
                p.transformer_block_param.embed = 3;
                p.transformer_block_param.block_size = 4;
            }
            else
            {
                p.transformer_block_param.heads = 6;
                p.transformer_block_param.embed = 192;
                p.transformer_block_param.block_size = 128;
            }
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                if (strModel == "pico")
                    blobX.LoadFromNumpy(strPath + "blk0.x.npy");
                else
                    blobX.LoadFromNumpy(strPath + "1_x_emb1.npy");

                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                layer.Forward(colBtm, colTop);

                if (strModel == "pico")
                    blobVal.LoadFromNumpy(strPath + "blk0.y.npy");
                else
                    blobVal.LoadFromNumpy(strPath + "12b_out2.npy");

                double dfErr = (typeof(T) == typeof(float)) ? 1e-12 : 2e-06;
                verify(blobY, blobVal, false, dfErr);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestBackward(string strModel)
        {
            string strPath = "C:\\temp\\projects\\2023.minGpt\\minGPT\\minGPT\\test\\iter_0\\"; // loadTestData1(strModel);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
            if (strModel == "pico")
            {
                p.transformer_block_param.heads = 1;
                p.transformer_block_param.embed = 3;
                p.transformer_block_param.block_size = 4;
            }
            else
            {
                p.transformer_block_param.heads = 6;
                p.transformer_block_param.embed = 192;
                p.transformer_block_param.block_size = 128;
            }
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                if (strModel == "pico")
                    blobX.LoadFromNumpy(strPath + "1_x_emb1.npy");
                else
                    blobX.LoadFromNumpy(strPath + "blk0.x.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                layer.Forward(colBtm, colTop);

                if (strModel == "pico")
                    blobVal.LoadFromNumpy(strPath + "12b_out2.npy");
                else
                    blobVal.LoadFromNumpy(strPath + "blk0.y.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 1e-12 : 2e-06;
                verify(blobY, blobVal, false, dfErr);

                if (strModel == "pico")
                    colTop[0].LoadFromNumpy(strPath + "grad_12b_out2.npy", true);
                else
                    colTop[0].LoadFromNumpy(strPath + "grad_blk0.y.npy", true);

                layer.Backward(colTop, new List<bool>() { true }, colBtm);

                if (strModel == "pico")
                    blobVal.LoadFromNumpy(strPath + "grad_1_x_emb1.npy", true);
                else
                    blobVal.LoadFromNumpy(strPath + "grad_blk0.x.npy", true);

                dfErr = (typeof(T) == typeof(float)) ? 1e-12 : 3e-06;
                verify(colBtm[0], blobVal, true, dfErr);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            string strPath = loadTestData1(null);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
            p.transformer_block_param.heads = 6;
            p.transformer_block_param.embed = 192;
            p.transformer_block_param.block_size = 128;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                blobX.LoadFromNumpy(strPath + "1_x_emb1.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, colBtm, colTop, -1, 200);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                layer.Dispose();
            }
        }

        private string loadTestData2()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb2\\";
            string strFileName = "_transformer_test2";

            strFileName += ".zip";

            string strTestPath = "test";
            string strTestFile = "iter_0\\1_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private string buildModelEx(NetParameter net, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, double dfDropout, Phase phase)
        {
            LayerParameter tok = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA);
            tok.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            tok.tokenized_data_param.vocabulary_type = TokenizedDataParameter.VOCABULARY_TYPE.CHARACTER;
            tok.tokenized_data_param.source = "$ProgramData$\\MyCaffe\\test_data\\data\\text\\input.txt";
            tok.tokenized_data_param.batch_size = nBatch;
            tok.tokenized_data_param.block_size = nBlockSize;
            tok.top.Add("tokdata");
            tok.top.Add("pos");
            if (phase != Phase.RUN)
                tok.top.Add("tgt");
            net.layer.Add(tok);

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "wte";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            emb1.parameters.Add(new ParamSpec(1.0, 0.0));
            emb1.bottom.Add("tokdata");
            emb1.top.Add("tok_emb");
            net.layer.Add(emb1);

            LayerParameter emb2 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb2.name = "wpe";
            emb2.embed_param.bias_term = false;
            emb2.embed_param.input_dim = nBlockSize;
            emb2.embed_param.num_output = nEmbed;
            emb2.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            emb2.parameters.Add(new ParamSpec(1.0, 0.0));
            emb2.bottom.Add("pos");
            emb2.top.Add("tok_pos");
            net.layer.Add(emb2);

            LayerParameter elt = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            elt.name = "eltwise1";
            elt.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            elt.eltwise_param.allow_single_batch_input = true;
            elt.bottom.Add("tok_emb");
            elt.bottom.Add("tok_pos");
            elt.top.Add("eltwise1");
            net.layer.Add(elt);

            string strEncBtm = "eltwise1";
            int nLayers = 1;
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "tfb" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
                enc.transformer_block_param.heads = 1;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.bottom.Add(strEncBtm);
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

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nEncVocabSize;
            ip1.inner_product_param.bias_term = false;
            ip1.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            ip1.parameters.Add(new ParamSpec(1, 1));
            ip1.bottom.Add("ln1");
            ip1.top.Add("logits");
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.softmax_param.algorithm = SOFTMAX_ALGORITHM.ACCURATE;
            softmax.softmax_param.algorithm_train = SOFTMAX_ALGORITHM.LOG;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            net.layer.Add(softmax);

            if (phase != Phase.RUN)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
                loss.name = "loss";
                loss.nll_loss_param.axis = 2;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.bottom.Add("prob");
                loss.bottom.Add("tgt");
                loss.top.Add("loss");
                net.layer.Add(loss);

                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.axis = 2;
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("tgt");
                accuracy.top.Add("accuracy");
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }

        private string buildSolver()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 5e-04;
            solver.weight_decay = 0;
            solver.adamw_decay = 0.1;
            solver.momentum = 0.9;
            solver.momentum2 = 0.95;
            solver.delta = 1e-08;
            solver.type = SolverParameter.SolverType.ADAMW;
            solver.lr_policy = "fixed";
            solver.test_initialization = false;

            return solver.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the training cycle of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False, model_type = 'gpt_nano1'
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestTrain()
        {
            string strPath = loadTestData2();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel);

            try
            {
                NetParameter net_param = new NetParameter();
                string strSolver = buildSolver();
                string strModel = buildModelEx(net_param, 1, 64, 128, 65, 0.0, Phase.TRAIN);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
                Blob<float> blobVal = mycaffe.CreateBlob("val");
                Blob<float> blobWork = mycaffe.CreateBlob("work");

                Net<float> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<float> solver = mycaffe.GetInternalSolver();

                net.learnable_parameters[0].LoadFromNumpy(strPath + "iter_0\\1_wte.weight.npy");
                net.learnable_parameters[1].LoadFromNumpy(strPath + "iter_0\\1_wpe.weight.npy");
                net.learnable_parameters[2].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_attn.weight.npy");
                net.learnable_parameters[3].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_attn.bias.npy");
                net.learnable_parameters[4].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_proj.weight.npy");
                net.learnable_parameters[5].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_proj.bias.npy");
                net.learnable_parameters[6].LoadFromNumpy(strPath + "iter_0\\blk0.c_fc.weight.npy");
                net.learnable_parameters[7].LoadFromNumpy(strPath + "iter_0\\blk0.c_fc.bias.npy");
                net.learnable_parameters[8].LoadFromNumpy(strPath + "iter_0\\blk0.c_proj.weight.npy");
                net.learnable_parameters[9].LoadFromNumpy(strPath + "iter_0\\blk0.c_proj.bias.npy");
                net.learnable_parameters[10].LoadFromNumpy(strPath + "iter_0\\13_lm_head.weight.npy");

                for (int i = 0; i < 1; i++)
                {
                    string strPath1 = strPath + "iter_" + i.ToString() + "\\";

                    blobVal.LoadFromNumpy(strPath1 + "1_wte.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[0], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[1], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[2], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[3], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[4], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[5], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[6], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[7], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[8], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[9], blobWork, false, 1e-08));

                    net.ClearParamDiffs();

                    double dfLoss = 0;
                    for (int j = 0; j < net.layers.Count; j++)
                    {
                        BlobCollection<float> colTop = net.top_vecs[j];
                        BlobCollection<float> colBottom = net.bottom_vecs[j];

                        if (j == 3)
                        {
                            blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                            Trace.Assert(blobVal.Compare(net.layers[j].blobs[0], blobWork, false, 1e-08));
                        }

                        double dfLayerLoss = net.layers[j].Forward(colBottom, colTop);
                        dfLoss += dfLayerLoss;

                        string strTopData = null;
                        string strBottomData = null;
                        string strLayer = net.layers[j].layer_param.name;
                        Trace.WriteLine("LAYER: " + strLayer);
                        switch (j)
                        {
                            case 0:
                                colTop[0].LoadFromNumpy(strPath1 + "1_x.npy");
                                colTop[2].LoadFromNumpy(strPath1 + "1_targets.npy");
                                break;

                            case 2:
                                strBottomData = "1_x.npy";
                                strTopData = "0_tok_emb.npy";
                                break;

                            case 3:
                                strBottomData = "0_pos.npy";
                                strTopData = "0_pos_emb.npy";
                                break;

                            case 4:
                                strTopData = "0_x.npy";
                                break;

                            case 5:
                                strBottomData = "0_x.npy";
                                strTopData = "0_blk_x.npy";
                                break;

                            case 6:
                                strBottomData = "0_blk_x.npy";
                                strTopData = "12_ln_x.npy";
                                break;

                            case 7:
                                strBottomData = "12_ln_x.npy";
                                strTopData = "13_logits.npy";
                                break;
                        }

                        if (!string.IsNullOrEmpty(strBottomData))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strBottomData);
                            Trace.Assert(blobVal.Compare(colBottom[0], blobWork, false, 1e-08));
                        }

                        if (!string.IsNullOrEmpty(strTopData))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strTopData);
                            Trace.Assert(blobVal.Compare(colTop[0], blobWork, false, 1e-08));
                        }
                    }

                    net.blobs[net.blobs.Count - 1].SetDiff(1);

                    for (int j = net.layer_need_backward.Count-1; j >= 0; j--)
                    {
                        if (net.layer_need_backward[j])
                        {
                            BlobCollection<float> colTop = net.top_vecs[j];
                            BlobCollection<float> colBottom = net.bottom_vecs[j];
                            List<bool> rgNeedBackward = net.bottom_need_backward[j];

                            net.layers[j].Backward(colTop, rgNeedBackward, colBottom);

                            string strBottomDiff = null;
                            string strTopData = null;
                            string strBottomData = null;
                            switch (j)
                            {
                                case 10:
                                    strBottomDiff = "grad_14_prob.npy"; // NLLLoss -> prob
                                    break;

                                case 8:
                                    strBottomDiff = "grad_13_logits.npy"; // LogSoftmax -> logits
                                    strBottomData = "13_logits.npy";
                                    strTopData = "14_prob.npy";
                                    break;

                                case 7:
                                    strBottomDiff = "grad_12_ln_x.npy"; // Linear -> LayerNorm
                                    strBottomData = "12_ln_x.npy";
                                    strTopData = "13_logits.npy";
                                    break;
                            }

                            if (!string.IsNullOrEmpty(strBottomDiff))
                            {
                                blobVal.LoadFromNumpy(strPath1 + strBottomDiff, true);
                                Trace.Assert(blobVal.Compare(colBottom[0], blobWork, true, 1e-08));

                                if (!string.IsNullOrEmpty(strTopData))
                                {
                                    blobVal.LoadFromNumpy(strPath1 + strTopData);
                                    Trace.Assert(blobVal.Compare(colTop[0], blobWork, false, 1e-08));
                                }

                                if (!string.IsNullOrEmpty(strBottomData))
                                {
                                    blobVal.LoadFromNumpy(strPath1 + strBottomData);
                                    Trace.Assert(blobVal.Compare(colBottom[0], blobWork, false, 1e-08));
                                }
                            }
                        }
                    }

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_lm_head.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[10], blobWork, true, 1e-08));

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_proj.bias.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[9], blobWork, true, 6e-08)); //<<-bug
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_proj.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[8], blobWork, true, 1e-08));

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_fc.bias.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[7], blobWork, true, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_fc.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[6], blobWork, true, 1e-08));

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_proj.bias.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[5], blobWork, true, 3e-07)); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_proj.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[4], blobWork, true, 1e-08));

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_attn.bias.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[3], blobWork, true, 3e-08)); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_attn.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[2], blobWork, true, 3e-08)); //<<-bug?

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_tfb.wpe.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[1], blobWork, true, 2e-07)); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_tfb.wte.wt.npy", true);
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[0], blobWork, true, 3e-07)); //<<-bug?

                    solver.ApplyUpdate(i);

                    blobVal.LoadFromNumpy(strPath1 + "____.param_lm_head.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[10], blobWork, false, 2e-08));

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_proj.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[9], blobWork));
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_proj.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[8], blobWork));

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_fc.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[7], blobWork));
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_fc.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[6], blobWork, false, 2e-08));

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_proj.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[5], blobWork)); 
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_proj.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[4], blobWork));

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_attn.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[3], blobWork, false, 2e-04)); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_attn.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[2], blobWork, false, 2e-08)); 

                    blobVal.LoadFromNumpy(strPath1 + "____.param_tfb.wpe.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[1], blobWork));
                    blobVal.LoadFromNumpy(strPath1 + "____.param_tfb.wte.wt.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[0], blobWork));
                }
            }
            finally
            {
                mycaffe.Dispose();
            }
        }


        private string loadTestData3()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb3\\";
            string strFileName = "_transformer_test3";

            strFileName += ".zip";

            string strTestPath = "test";
            string strTestFile = "0_blk_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        /// <summary>
        /// Test the inference cycle of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set enable_inference_save_testing = True
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False, model_type = 'gpt_nano1'
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestInference()
        {
            string strPath = loadTestData3();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel);
            Blob<float> blobInput = null;

            try
            {
                NetParameter net_param = new NetParameter();
                string strModel = buildModelEx(net_param, 1, 64, 128, 65, 0.0, Phase.RUN);

                BlobShape shape = new BlobShape(1, 64, 1, 1);

                mycaffe.LoadToRun(strModel, null, shape);
                Blob<float> blobVal = mycaffe.CreateBlob("val");
                Blob<float> blobWork = mycaffe.CreateBlob("work");

                Net<float> net = mycaffe.GetInternalNet(Phase.RUN);

                net.learnable_parameters[0].LoadFromNumpy(strPath + "1_wte.weight.npy");
                net.learnable_parameters[1].LoadFromNumpy(strPath + "1_wpe.weight.npy");
                net.learnable_parameters[2].LoadFromNumpy(strPath + "blk0.csa.c_attn.weight.npy");
                net.learnable_parameters[3].LoadFromNumpy(strPath + "blk0.csa.c_attn.bias.npy");
                net.learnable_parameters[4].LoadFromNumpy(strPath + "blk0.csa.c_proj.weight.npy");
                net.learnable_parameters[5].LoadFromNumpy(strPath + "blk0.csa.c_proj.bias.npy");
                net.learnable_parameters[6].LoadFromNumpy(strPath + "blk0.c_fc.weight.npy");
                net.learnable_parameters[7].LoadFromNumpy(strPath + "blk0.c_fc.bias.npy");
                net.learnable_parameters[8].LoadFromNumpy(strPath + "blk0.c_proj.weight.npy");
                net.learnable_parameters[9].LoadFromNumpy(strPath + "blk0.c_proj.bias.npy");
                net.learnable_parameters[10].LoadFromNumpy(strPath + "13_lm_head.weight.npy");

                // Run forward test pass
                { 
                    string strPath1 = strPath;

                    blobVal.LoadFromNumpy(strPath1 + "1_wte.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[0], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[1], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[2], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[3], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[4], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[5], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[6], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[7], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.weight.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[8], blobWork, false, 1e-08));
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.bias.npy");
                    Trace.Assert(blobVal.Compare(net.learnable_parameters[9], blobWork, false, 1e-08));

                    double dfLoss = 0;
                    for (int j = 0; j < net.layers.Count; j++)
                    {
                        BlobCollection<float> colTop = net.top_vecs[j];
                        BlobCollection<float> colBottom = net.bottom_vecs[j];

                        if (j == 0)
                        {
                            PropertySet input = new PropertySet();
                            input.SetProperty("InputData", "O God, O God!");
                            TokenizedDataLayer<float> toklayer = net.layers[j] as TokenizedDataLayer<float>;

                            int nSeqLen;
                            BlobCollection<float> col = toklayer.PreProcessInput(input, out nSeqLen);
                            blobInput = col[0];
                            colBottom[0].CopyFrom(blobInput, false, true);
                        }

                        if (j == 2)
                        {
                            blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                            Trace.Assert(blobVal.Compare(net.layers[j].blobs[0], blobWork, false, 1e-08));
                        }

                        double dfLayerLoss = net.layers[j].Forward(colBottom, colTop);
                        dfLoss += dfLayerLoss;

                        string strTopData0 = null;
                        double dfTopErr0 = 1e-08;
                        string strTopData1 = null;
                        double dfTopErr1 = 1e-08;
                        string strBottomData = null;
                        double dfBtmErr = 1e-08;
                        string strLayer = net.layers[j].layer_param.name;
                        Trace.WriteLine("LAYER: " + strLayer);
                        switch (j)
                        {
                            case 0:
                                strTopData0 = "1_x.npy";
                                strTopData1 = "0_pos.npy";
                                break;

                            case 1:
                                strBottomData = "1_x.npy";
                                strTopData0 = "0_tok_emb.npy";
                                break;

                            case 2:
                                strBottomData = "0_pos.npy";
                                strTopData0 = "0_pos_emb.npy";
                                break;

                            case 3:
                                strTopData0 = "0_x.npy";
                                break;

                            case 4:
                                strBottomData = "0_x.npy";
                                strTopData0 = "0_blk_x.npy";
                                dfTopErr0 = 5e-07;
                                break;

                            case 5:
                                strBottomData = "0_blk_x.npy";
                                dfBtmErr = 5e-07;
                                strTopData0 = "12_ln_x.npy";
                                dfTopErr0 = 8e-07;
                                break;

                            case 6:
                                strBottomData = "12_ln_x.npy";
                                dfBtmErr = 8e-07;
                                strTopData0 = "13_logits.npy";
                                dfTopErr0 = 2e-06;
                                break;
                        }

                        if (!string.IsNullOrEmpty(strBottomData))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strBottomData);
                            Trace.Assert(blobVal.Compare(colBottom[0], blobWork, false, dfBtmErr));
                        }

                        if (!string.IsNullOrEmpty(strTopData0))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strTopData0);
                            Trace.Assert(blobVal.Compare(colTop[0], blobWork, false, dfTopErr0));
                        }

                        if (!string.IsNullOrEmpty(strTopData1))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strTopData1);
                            Trace.Assert(blobVal.Compare(colTop[1], blobWork, false, dfTopErr1));
                        }
                    }
                }
            }
            finally
            {
                if (blobInput != null)
                    blobInput.Dispose();

                mycaffe.Dispose();
            }
        }
    }
}
