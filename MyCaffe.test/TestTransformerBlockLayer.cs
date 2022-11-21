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
using System.Dynamic;
using static MyCaffe.param.beta.DecodeParameter;
using System.Collections.Concurrent;

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
    }

    interface ITransformerBlockLayerTest : ITest
    {
        void TestForwardPico(bool bBatch, int nHeads);
        void TestBackwardPico(bool bBatch, int nHeads);
        void TestGradientPico(bool bBatch, int nHeads);
        void TestForwardMini();
        void TestTrainingGptMini(int nIter);
        void TestTrainingGptMini1(int nIter);
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

                Solver<T> solver = m_ctrl.GetInternalSolver();

                m_blobY = m_ctrl.CreateBlob("results");
                m_blobX = m_ctrl.CreateBlob("data");
                m_blobPos = m_ctrl.CreateBlob("pos");
                m_blobPos.Reshape(1, 128, 1, 1);

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
                dispose1(ref m_blobPos);

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
                m_blobPos = m_ctrl.CreateBlob("pos");
                m_blobPos.Reshape(1, 128, 1, 1);

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

                m_ctrl.Train(10);
            }
            finally
            {
                dispose1(ref m_blobY);
                dispose1(ref m_blobX);
                dispose1(ref m_blobPos);

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

                fillPos(m_blobX, m_blobPos);
                m_dataLayer.Tokenize(m_blobX, m_blobX);
                generate(m_netRun, m_blobX, m_blobY, m_blobPos, 500, (int)m_dataLayer.layer_param.tokenized_data_param.block_size, 65, 10);
                
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

        private void generate(Net<T> net, Blob<T> blobIdx, Blob<T> blobY, Blob<T> blobPos, int nMaxNewTokens, int nBlockSize, int nVocabSize, int nTopK)
        {
            Blob<T> blobLogits = net.blob_by_name("logits");
            Layer<T> softmax = net.FindLastLayer(LayerParameter.LayerType.SOFTMAX);
            BlobCollection<T> colBottom = new BlobCollection<T>() { blobIdx, blobPos };
            double dfLoss;
            List<float> rgfIdx = new List<float>();
            List<float> rgfIdxOut = new List<float>();
            int[] rgShape;

            float[] rgIdx = convertF(blobIdx.mutable_cpu_data);
            rgfIdx.AddRange(rgIdx);
                       
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
                
                if (blobPos.count() != blobIdx.count(0, 2))
                    fillPos(blobIdx, blobPos);
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
    }
}
