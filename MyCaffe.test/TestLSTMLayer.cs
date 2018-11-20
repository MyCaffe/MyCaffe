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
using MyCaffe.db.image;

namespace MyCaffe.test
{
    [TestClass]
    public class TestLSTMLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestSetup();
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
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestForward();
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
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
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
        public void TestGradientNonZeroCont()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestGradientNonZeroCont();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientNonZeroContBufferSize2()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestGradientNonZeroContBufferSize2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientNonZeroContBufferSize2WithStaticInput()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestGradientNonZeroContBufferSize2WithStaticInput();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLSTMUnitSetup()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestLSTMUnitSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLSTMUnitGradient()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestLSTMUnitGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLSTMUnitGradientNonZeroCont()
        {
            LSTMLayerTest test = new LSTMLayerTest();

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestLSTMUnitGradientNonZeroCont();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupCuDnn()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardCuDnn()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestForward(Phase.TEST);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardCuDnnTraining()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestForward(Phase.TRAIN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCuDnn()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
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
        public void TestCuDnn()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestCuDnn();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ILSTMLayerTest : ITest
    {
        void TestCuDnn();
        void TestSetup();
        void TestForward(Phase phase = Phase.NONE);
        void TestGradient();
        void TestGradientNonZeroCont();
        void TestGradientNonZeroContBufferSize2();
        void TestGradientNonZeroContBufferSize2WithStaticInput();

        void TestLSTMUnitSetup();
        void TestLSTMUnitGradient();
        void TestLSTMUnitGradientNonZeroCont();
    }

    class LSTMLayerTest : TestBase
    {
        public LSTMLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("LSTM Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LSTMLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LSTMLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LSTMLayerTest<T> : TestEx<T>, ILSTMLayerTest
    {
        Blob<T> m_blob_bottom_cont;
        Blob<T> m_blob_bottom_static;
        int m_nNumOutput = 7;
        LayerParameter m_param;
        CancelEvent m_evtCancel = new CancelEvent();
        Blob<T> m_blobUnit_bottom_cont;
        Blob<T> m_blobUnit_bottom_c_prev;
        Blob<T> m_blobUnit_bottom_x;
        Blob<T> m_blobUnit_top_c;
        Blob<T> m_blobUnit_top_h;
        BlobCollection<T> m_colUnitBottomVec = new BlobCollection<T>();
        BlobCollection<T> m_colUnitTopVec = new BlobCollection<T>();

        public LSTMLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom_cont = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_static = new Blob<T>(m_cuda, m_log);

            BottomVec.Add(m_blob_bottom_cont);

            m_blobUnit_bottom_cont = new Blob<T>(m_cuda, m_log);
            m_blobUnit_bottom_c_prev = new Blob<T>(m_cuda, m_log);
            m_blobUnit_bottom_x = new Blob<T>(m_cuda, m_log);
            m_blobUnit_top_c = new Blob<T>(m_cuda, m_log);
            m_blobUnit_top_h = new Blob<T>(m_cuda, m_log);

            m_colUnitBottomVec.Add(m_blobUnit_bottom_c_prev);
            m_colUnitBottomVec.Add(m_blobUnit_bottom_x);
            m_colUnitBottomVec.Add(m_blobUnit_bottom_cont);
            m_colUnitTopVec.Add(m_blobUnit_top_c);
            m_colUnitTopVec.Add(m_blobUnit_top_h);

            ReshapeBlobs(1, 3);

            m_param = new LayerParameter(LayerParameter.LayerType.LSTM);
            m_param.recurrent_param.num_output = (uint)m_nNumOutput;
            m_param.recurrent_param.weight_filler = new FillerParameter("gaussian");
            m_param.recurrent_param.weight_filler.std = 0.2;
            m_param.recurrent_param.bias_filler = new FillerParameter("gaussian");
            m_param.recurrent_param.bias_filler.std = 0.1;

            m_param.phase = Phase.TEST;
        }

        protected override void dispose()
        {
            m_blob_bottom_cont.Dispose();
            m_blob_bottom_static.Dispose();
            m_blobUnit_bottom_cont.Dispose();
            m_blobUnit_bottom_c_prev.Dispose();
            m_blobUnit_bottom_x.Dispose();
            m_blobUnit_top_c.Dispose();
            m_blobUnit_top_h.Dispose();
            base.dispose();
        }

        public void ReshapeBlobs(int nNumTimesteps, int nNumInstances)
        {
            m_blob_bottom.Reshape(nNumTimesteps, nNumInstances, 3, 2);
            m_blob_bottom_static.Reshape(nNumInstances, 2, 3, 4);
            List<int> rgShape = new List<int>() { nNumTimesteps, nNumInstances };
            m_blob_bottom_cont.Reshape(rgShape);
            rgShape.Add(m_nNumOutput);

            rgShape[0] = 1;
            rgShape[1] = nNumInstances;
            rgShape[2] = 4 * m_nNumOutput;
            m_blobUnit_bottom_x.Reshape(rgShape);

            rgShape[0] = 1;
            rgShape[1] = nNumInstances;
            rgShape[2] = m_nNumOutput;
            m_blobUnit_bottom_c_prev.Reshape(rgShape);

            rgShape = new List<int>();
            rgShape.Add(1);
            rgShape.Add(nNumInstances);
            m_blobUnit_bottom_cont.Reshape(rgShape);

            FillerParameter filler_param = new FillerParameter("uniform");
            filler_param.min = -1;
            filler_param.max = 1;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);

            filler.Fill(m_blob_bottom);
            filler.Fill(m_blobUnit_bottom_c_prev);
            filler.Fill(m_blobUnit_bottom_x);
        }

        public void TestSetup()
        {
            m_param.recurrent_param.engine = m_engine;
            LSTMLayer<T> layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);

            layer.Setup(BottomVec, TopVec);

            List<int> rgExpectedTopShape = Utility.Clone<int>(m_blob_bottom.shape(), 3);
            rgExpectedTopShape[2] = m_nNumOutput;

            if (m_blob_top.num_axes == 4 && m_blob_top.shape(3) == 1 && rgExpectedTopShape.Count == 3)
                rgExpectedTopShape.Add(1);

            m_log.CHECK(Utility.Compare<int>(m_blob_top.shape(), rgExpectedTopShape), "The top shape is not as expected.");

            layer.Dispose();
        }

        public void TestForward(Phase phase)
        {
            int kNumTimesteps = 3;
            int nNum = m_blob_bottom.shape(1);
            ReshapeBlobs(kNumTimesteps, nNum);


            // Fill the cont blob with <0, 1, 1, ..., 1>,
            //  indicating a sequence that begins at the first timestep
            //  the continues for the rest of the sequence.
            double[] rgData = convert(m_blob_bottom_cont.mutable_cpu_data);

            for (int t = 0; t < kNumTimesteps; t++)
            {
                for (int n = 0; n < nNum; n++)
                {
                    rgData[t * nNum + n] = (t > 0) ? 1 : 0;
                }
            }

            m_blob_bottom_cont.mutable_cpu_data = convert(rgData);


            // Process the full sequence in each single batch.
            FillerParameter filler_param = new FillerParameter("gaussian");
            filler_param.mean = 0;
            filler_param.std = 1;
            Filler<T> sequence_filler = Filler<T>.Create(m_cuda, m_log, filler_param);
            m_cuda.rng_setseed(1);
            sequence_filler.Fill(m_blob_bottom);

            m_param.recurrent_param.engine = m_engine;

            if (phase != Phase.NONE)
                m_param.phase = phase;

            LSTMLayer<T> layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            m_cuda.rng_setseed(1701);
            layer.Setup(BottomVec, TopVec);

            m_log.WriteLine("Calling forward for full sequence LSTM");
            layer.Forward(BottomVec, TopVec);

            // Copy the inputs and outputs to reuse/check them later.
            Blob<T> bottom_copy = new Blob<T>(m_cuda, m_log, m_blob_bottom.shape());
            bottom_copy.CopyFrom(m_blob_bottom);
            Blob<T> top_copy = new Blob<T>(m_cuda, m_log, m_blob_top.shape());
            top_copy.CopyFrom(m_blob_top);

            // Process the batch one step at a time;
            //  check that we get the same result.
            ReshapeBlobs(1, nNum);

            layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            m_cuda.rng_setseed(1701);
            layer.Setup(BottomVec, TopVec);

            int nBottomCount = m_blob_bottom.count();
            int nTopCount = m_blob_top.count();
            double kEpsilon = 1e-5;

            for (int t = 0; t < kNumTimesteps; t++)
            {
                m_cuda.copy(nBottomCount, bottom_copy.gpu_data, m_blob_bottom.mutable_gpu_data, t * nBottomCount);

                double[] rgCont = convert(m_blob_bottom_cont.mutable_cpu_data);

                for (int n = 0; n < nNum; n++)
                {
                    rgCont[n] = (t > 0) ? 1 : 0;
                }

                m_blob_bottom_cont.mutable_cpu_data = convert(rgCont);

                m_log.WriteLine("Calling forward for LSTM timestep " + t.ToString());
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                double[] rgTopCopy = convert(top_copy.update_cpu_data());

                for (int i = 0; i < nTopCount; i++)
                {
                    m_log.CHECK_LT(t * nTopCount + i, top_copy.count(), "The top count is incorrect.");
                    m_log.EXPECT_NEAR(rgTop[i], rgTopCopy[t * nTopCount + i], kEpsilon, "t = " + t.ToString() + "; i = " + i.ToString());
                }
            }

            // Process the batch one timestep at a time with all cont blobs set to 0.
            //  Check that we get a different result, except in the first timestep.
            m_cuda.rng_setseed(1701);
            layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            layer.Setup(BottomVec, TopVec);

            for (int t = 0; t < kNumTimesteps; t++)
            {
                m_cuda.copy(nBottomCount, bottom_copy.gpu_data, m_blob_bottom.mutable_gpu_data, t * nBottomCount);

                double[] rgCont = convert(m_blob_bottom_cont.mutable_cpu_data);

                for (int n = 0; n < nNum; n++)
                {
                    rgCont[n] = 0;
                }

                m_blob_bottom_cont.mutable_cpu_data = convert(rgCont);

                m_log.WriteLine("Calling forward for LSTM timestep " + t.ToString());
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                double[] rgTopCopy = convert(top_copy.update_cpu_data());

                for (int i = 0; i < nTopCount; i++)
                {
                    if (t == 0)
                        m_log.EXPECT_NEAR(rgTop[i], rgTopCopy[t * nTopCount + i], kEpsilon, "t = " + t.ToString() + "; i = " + i.ToString());
                    else
                        m_log.CHECK_NE(rgTop[i], rgTopCopy[t * nTopCount + i], "t = " + t.ToString() + "; i = " + i.ToString());
                }
            }
        }

        public void TestLSTMUnitSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_UNIT);
            LSTMUnitLayer<T> layer = new LSTMUnitLayer<T>(m_cuda, m_log, p);
            layer.Setup(m_colUnitBottomVec, m_colUnitTopVec);
            int nNumAxes = m_blobUnit_bottom_c_prev.num_axes;

            m_log.CHECK_EQ(nNumAxes, m_blobUnit_top_c.num_axes, "The blobUnit_bottom_c_prev must have the same axes as blobUnit_top_c.");
            m_log.CHECK_EQ(nNumAxes, m_blobUnit_top_h.num_axes, "The blobUnit_bottom_c_prev must have the same axes as blobUnit_top_h.");

            for (int i = 0; i < nNumAxes; i++)
            {
                m_log.CHECK_EQ(m_blobUnit_bottom_c_prev.shape(i), m_blobUnit_top_c.shape(i), "The blobUnit_bottom_c.shape(" + i.ToString() + ") must have the same shape as blobUnit_top_c.shape(" + i.ToString() + ")");
                m_log.CHECK_EQ(m_blobUnit_bottom_c_prev.shape(i), m_blobUnit_top_h.shape(i), "The blobUnit_bottom_c.shape(" + i.ToString() + ") must have the same shape as blobUnit_top_h.shape(" + i.ToString() + ")");
            }
        }

        public void TestLSTMUnitGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_UNIT);
            LSTMUnitLayer<T> layer = new LSTMUnitLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgContData = convert(m_blobUnit_bottom_cont.mutable_cpu_data);
            rgContData[0] = 0;
            rgContData[1] = 0;
            rgContData[2] = 0;
            m_blobUnit_bottom_cont.mutable_cpu_data = convert(rgContData);

            checker.CheckGradientExhaustive(layer, m_colUnitBottomVec, m_colUnitTopVec, 0);
            checker.CheckGradientExhaustive(layer, m_colUnitBottomVec, m_colUnitTopVec, 1);
        }

        public void TestLSTMUnitGradientNonZeroCont()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_UNIT);
            LSTMUnitLayer<T> layer = new LSTMUnitLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgContData = convert(m_blobUnit_bottom_cont.mutable_cpu_data);
            rgContData[0] = 1;
            rgContData[1] = 0;
            rgContData[2] = 1;
            m_blobUnit_bottom_cont.mutable_cpu_data = convert(rgContData);

            checker.CheckGradientExhaustive(layer, m_colUnitBottomVec, m_colUnitTopVec, 0);
            checker.CheckGradientExhaustive(layer, m_colUnitBottomVec, m_colUnitTopVec, 1);
        }

        public void TestGradient()
        {
            m_param.recurrent_param.engine = m_engine;
            m_param.phase = Phase.TRAIN;
            LSTMLayer<T> layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestGradientNonZeroCont()
        {
            LSTMLayer<T> layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgCont = convert(m_blob_bottom_cont.mutable_cpu_data);
            for (int i = 0; i < m_blob_bottom_cont.count(); i++)
            {
                rgCont[i] = (i > 2) ? 1 : 0;
            }
            m_blob_bottom_cont.mutable_cpu_data = convert(rgCont);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestGradientNonZeroContBufferSize2()
        {
            ReshapeBlobs(2, 2);

            // fill the values.
            FillerParameter filler_param = new FillerParameter("uniform");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);
            filler.Fill(m_blob_bottom);

            LSTMLayer<T> layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgCont = convert(m_blob_bottom_cont.mutable_cpu_data);

            for (int i = 0; i < m_blob_bottom_cont.count(); i++)
            {
                rgCont[i] = (i > 2) ? 1 : 0;
            }

            m_blob_bottom_cont.mutable_cpu_data = convert(rgCont);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestGradientNonZeroContBufferSize2WithStaticInput()
        {
            ReshapeBlobs(2, 2);

            // fill the values.
            FillerParameter filler_param = new FillerParameter("uniform");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);
            filler.Fill(m_blob_bottom);
            filler.Fill(m_blob_bottom_static);

            BottomVec.Add(m_blob_bottom_static);

            LSTMLayer<T> layer = new LSTMLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgCont = convert(m_blob_bottom_cont.mutable_cpu_data);

            for (int i = 0; i < m_blob_bottom_cont.count(); i++)
            {
                rgCont[i] = (i > 2) ? 1 : 0;
            }

            m_blob_bottom_cont.mutable_cpu_data = convert(rgCont);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 2);
        }

        public void TestCuDnn()
        {
            int nBatchSize = 64;
            int nSeqLen = 20;
            int nNumLayers = 2;
            int nHiddenSize = 512;
            int nInputSize = nHiddenSize;
            long hCuDnn = 0;
            long hXDesc = 0;
            long hYDesc = 0;
            long hHxDesc = 0;
            long hCxDesc = 0;
            long hHyDesc = 0;
            long hCyDesc = 0;
            long hRnnDesc = 0;
            long hWtDesc = 0;
            long hDropoutDesc = 0;
            long hStates = 0;
            long hWorkspace = 0;
            int nWorkspaceCount = 0;
            long hReserved = 0;
            int nReservedCount = 0;
            Blob<T> blobX = null;
            Blob<T> blobHx = null;
            Blob<T> blobCx = null;
            Blob<T> blobY = null;
            Blob<T> blobHy = null;
            Blob<T> blobCy = null;
            Blob<T> blobWt = null;

            try
            {
                // Create cudnn context
                hCuDnn = m_cuda.CreateCuDNN();

                // Setup inputs and outputs.
                blobX = new Blob<T>(m_cuda, m_log, nSeqLen, nBatchSize, nInputSize, 1);
                blobHx = new Blob<T>(m_cuda, m_log, nNumLayers, nBatchSize, nHiddenSize, 1);
                blobCx = new Blob<T>(m_cuda, m_log, nNumLayers, nBatchSize, nHiddenSize, 1);

                blobY = new Blob<T>(m_cuda, m_log, nSeqLen, nBatchSize, nHiddenSize, 1);
                blobHy = new Blob<T>(m_cuda, m_log, nNumLayers, nBatchSize, nHiddenSize, 1);
                blobCy = new Blob<T>(m_cuda, m_log, nNumLayers, nBatchSize, nHiddenSize, 1);

                // Setup tensor descriptors where there is one tensor per time step (tensors are set internall by m_cuda)
                hXDesc = m_cuda.CreateRnnDataDesc();
                m_cuda.SetRnnDataDesc(hXDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nInputSize);
                hYDesc = m_cuda.CreateRnnDataDesc();
                m_cuda.SetRnnDataDesc(hYDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                int[] rgDimA = new int[3];
                rgDimA[0] = nNumLayers;
                rgDimA[1] = nBatchSize;
                rgDimA[2] = nHiddenSize;

                int[] rgStrideA = new int[3];
                rgStrideA[0] = rgDimA[2] * rgDimA[1];
                rgStrideA[1] = rgDimA[2];
                rgStrideA[2] = 1;

                hHxDesc = m_cuda.CreateTensorDesc();
                m_cuda.SetTensorNdDesc(hHxDesc, rgDimA, rgStrideA);
                hCxDesc = m_cuda.CreateTensorDesc();
                m_cuda.SetTensorNdDesc(hCxDesc, rgDimA, rgStrideA);
                hHyDesc = m_cuda.CreateTensorDesc();
                m_cuda.SetTensorNdDesc(hHyDesc, rgDimA, rgStrideA);
                hCyDesc = m_cuda.CreateTensorDesc();
                m_cuda.SetTensorNdDesc(hCyDesc, rgDimA, rgStrideA);

                //-------------------------------------------------
                // Set the dropout layer.
                //-------------------------------------------------

                hDropoutDesc = m_cuda.CreateDropoutDesc();
                ulong ulStateCount;
                ulong ulReservedCount;
                m_cuda.GetDropoutInfo(hCuDnn, 0, out ulStateCount, out ulReservedCount);
                hStates = m_cuda.AllocMemory((int)ulStateCount);
                m_cuda.SetDropoutDesc(hCuDnn, hDropoutDesc, 0, hStates, 1337);

                // Setup the RNN descriptor.
                hRnnDesc = m_cuda.CreateRnnDesc();
                m_cuda.SetRnnDesc(hCuDnn, hRnnDesc, nHiddenSize, nNumLayers, hDropoutDesc, RNN_MODE.LSTM);

                // Setup the parameters.  This needs to be done after the RNN descriptor is set
                // otherwise we don't know how many parameters we have to allocate.

                int nAllWtCount = m_cuda.GetRnnParamCount(hCuDnn, hRnnDesc, hXDesc);

                int[] rgDimW = new int[3];
                rgDimW[0] = nAllWtCount;
                rgDimW[1] = 1;
                rgDimW[2] = 1;

                hWtDesc = m_cuda.CreateFilterDesc();
                m_cuda.SetFilterNdDesc(hWtDesc, rgDimW);

                blobWt = new Blob<T>(m_cuda, m_log, new List<int>() { nAllWtCount });

                // Setup the workspace and reserved memory.
                nWorkspaceCount = m_cuda.GetRnnWorkspaceCount(hCuDnn, hRnnDesc, hXDesc, out nReservedCount);
                hWorkspace = m_cuda.AllocMemory(nWorkspaceCount);
                hReserved = m_cuda.AllocMemory(nReservedCount);


                //-------------------------------------------------
                // Initialize weights and inputs.
                //-------------------------------------------------

                // Initialize something simple.
                // Matrices are initialized to 1/matrixSize, biases to 1, data is 1.
                blobX.SetData(1.0);
                blobHx.SetData(1.0);
                blobCx.SetData(1.0);

                blobY.SetDiff(1.0);
                blobHy.SetDiff(1.0);
                blobCy.SetDiff(1.0);

                // Weights
                int nNumLinLayers = 8; // LSTM.
                Filler<T> fillerBias = Filler<T>.Create(m_cuda, m_log, new FillerParameter("constant", 1.0));

                for (int i = 0; i < nNumLayers; i++)
                {
                    for (int j = 0; j < nNumLinLayers; j++)
                    {
                        int nWtCount;
                        long hWtPtr;
                        int nBiasCount;
                        long hBiasPtr;

                        m_cuda.GetRnnLinLayerParams(hCuDnn, hRnnDesc, i, hXDesc, hWtDesc, blobWt.gpu_data, j, out nWtCount, out hWtPtr, out nBiasCount, out hBiasPtr);

                        double dfFill = 1.0 / (double)nWtCount;
                        Filler<T> fillerWt = Filler<T>.Create(m_cuda, m_log, new FillerParameter("constant", dfFill));

                        fillerWt.Fill(nWtCount, hWtPtr);
                        fillerBias.Fill(nBiasCount, hBiasPtr);

                        m_cuda.FreeMemoryPointer(hWtPtr);
                        m_cuda.FreeMemoryPointer(hBiasPtr);
                    }
                }

                //-------------------------------------------------
                // At this point all of the setup is done.  We now need
                // to pass through the RNN.
                //-------------------------------------------------

                m_cuda.SynchronizeDevice();

                // Added here for testing - use non training when not training
                //m_cuda.RnnForward(hCuDnn,
                //                  hRnnDesc,
                //                  hXDesc,
                //                  blobX.gpu_data,
                //                  hHxDesc,
                //                  blobHx.gpu_data,
                //                  hCxDesc,
                //                  blobCx.gpu_data,
                //                  hWtDesc,
                //                  blobWt.gpu_data,
                //                  hYDesc,
                //                  blobY.mutable_gpu_data,
                //                  hHyDesc,
                //                  blobHy.mutable_gpu_data,
                //                  hCyDesc,
                //                  blobCy.mutable_gpu_data,
                //                  hWorkspace,
                //                  nWorkspaceCount,
                //                  hReserved,
                //                  nReservedCount,
                //                  false);

                // Use 'bTraining=true' when training.
                m_cuda.RnnForward(hCuDnn,
                                  hRnnDesc,
                                  hXDesc,
                                  blobX.gpu_data,
                                  hHxDesc,
                                  blobHx.gpu_data,
                                  hCxDesc,
                                  blobCx.gpu_data,
                                  hWtDesc,
                                  blobWt.gpu_data,
                                  hYDesc,
                                  blobY.mutable_gpu_data,
                                  hHyDesc,
                                  blobHy.mutable_gpu_data,
                                  hCyDesc,
                                  blobCy.mutable_gpu_data,
                                  hWorkspace,
                                  nWorkspaceCount,
                                  hReserved,
                                  nReservedCount,
                                  true);

                m_cuda.RnnBackwardData(hCuDnn,
                                  hRnnDesc,
                                  hYDesc,
                                  blobY.gpu_data,
                                  blobY.gpu_diff,
                                  hHyDesc,
                                  blobHy.gpu_diff,
                                  hCyDesc,
                                  blobCy.gpu_diff,
                                  hWtDesc,
                                  blobWt.gpu_data,
                                  hHxDesc,
                                  blobHx.gpu_data,
                                  hCxDesc,
                                  blobCx.gpu_data,
                                  hXDesc,
                                  blobX.mutable_gpu_diff,
                                  hHxDesc,
                                  blobHx.mutable_gpu_diff,
                                  hCxDesc,
                                  blobCx.mutable_gpu_diff,
                                  hWorkspace,
                                  nWorkspaceCount,
                                  hReserved,
                                  nReservedCount);

                // RnnBackwardWeights adds to the data in dw.
                blobWt.SetDiff(0);

                m_cuda.RnnBackwardWeights(hCuDnn,
                                  hRnnDesc,
                                  hXDesc,
                                  blobX.gpu_data,
                                  hHxDesc,
                                  blobHx.gpu_data,
                                  hYDesc,
                                  blobY.gpu_data,
                                  hWorkspace,
                                  nWorkspaceCount,
                                  hWtDesc,
                                  blobWt.mutable_gpu_diff,
                                  hReserved,
                                  nReservedCount);

                // Make double sure everything is finished before result checking.
                m_cuda.SynchronizeDevice();

                //-------------------------------------------------
                // Print check sums.
                //-------------------------------------------------
                printCheckSums(blobY, blobHy, blobCy, nSeqLen, nBatchSize, nHiddenSize, nNumLayers, false);
                printCheckSums(blobY, blobHy, blobCy, nSeqLen, nBatchSize, nHiddenSize, nNumLayers, true);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                if (blobX != null)
                    blobX.Dispose();

                if (blobHx != null)
                    blobHx.Dispose();

                if (blobCx != null)
                    blobCx.Dispose();

                if (blobY != null)
                    blobY.Dispose();

                if (blobHy != null)
                    blobHy.Dispose();

                if (blobCy != null)
                    blobCy.Dispose();

                if (blobWt != null)
                    blobWt.Dispose();

                if (hWorkspace != 0)
                    m_cuda.FreeMemory(hWorkspace);

                if (hReserved != 0)
                    m_cuda.FreeMemory(hReserved);

                if (hDropoutDesc != 0)
                    m_cuda.FreeDropoutDesc(hDropoutDesc);

                if (hStates != 0)
                    m_cuda.FreeMemory(hStates);

                if (hWtDesc != 0)
                    m_cuda.FreeFilterDesc(hWtDesc);

                if (hRnnDesc != 0)
                    m_cuda.FreeRnnDesc(hRnnDesc);

                if (hHxDesc != 0)
                    m_cuda.FreeTensorDesc(hHxDesc);

                if (hCxDesc != 0)
                    m_cuda.FreeTensorDesc(hCxDesc);

                if (hHyDesc != 0)
                    m_cuda.FreeTensorDesc(hHyDesc);

                if (hCyDesc != 0)
                    m_cuda.FreeTensorDesc(hCyDesc);

                if (hXDesc != 0)
                    m_cuda.FreeRnnDataDesc(hXDesc);

                if (hYDesc != 0)
                    m_cuda.FreeRnnDataDesc(hYDesc);

                if (hCuDnn != 0)
                    m_cuda.FreeCuDNN(hCuDnn);
            }
        }

        void printCheckSums(Blob<T> blobY, Blob<T> blobHy, Blob<T> blobCy, int nSeqLen, int nBatchSize, int nHiddenSize, int nNumLayers, bool bDiff)
        {
            double[] rgY = convert((bDiff) ? blobY.update_cpu_diff() : blobY.update_cpu_data());
            double[] rgHy = convert((bDiff) ? blobHy.update_cpu_diff() : blobHy.update_cpu_data());
            double[] rgCy = convert((bDiff) ? blobCy.update_cpu_diff() : blobCy.update_cpu_data());

            double dfCheckSumi = 0;
            double dfCheckSumh = 0;
            double dfCheckSumc = 0;

            for (int m = 0; m < nBatchSize; m++)
            {
                double dfLocalSumi = 0;
                double dfLocalSumh = 0;
                double dfLocalSumc = 0;

                for (int j = 0; j < nSeqLen; j++)
                {
                    for (int i = 0; i < nHiddenSize; i++)
                    {
                        dfLocalSumi += rgY[j * nBatchSize * nHiddenSize + m * nHiddenSize + i];
                    }
                }

                for (int j = 0; j < nNumLayers; j++)
                {
                    for (int i = 0; i < nHiddenSize; i++)
                    {
                        dfLocalSumh += rgHy[j * nHiddenSize * nBatchSize + m * nHiddenSize + i];
                        dfLocalSumc += rgCy[j * nHiddenSize * nBatchSize + m * nHiddenSize + i];
                    }
                }

                dfCheckSumi += dfLocalSumi;
                dfCheckSumh += dfLocalSumh;
                dfCheckSumc += dfLocalSumc;
            }

            string strType = (bDiff) ? "diff" : "data";
            m_log.WriteLine("i " + strType + " checksum " + dfCheckSumi.ToString());
            m_log.WriteLine("c " + strType + " checksum " + dfCheckSumc.ToString());
            m_log.WriteLine("h " + strType + " checksum " + dfCheckSumh.ToString());
        }

        void printCheckSums(Blob<T> blobWt)
        {
            double[] rgWt = convert(blobWt.update_cpu_diff());

            double dfCheckSumw = 0;

            for (int i = 0; i < blobWt.count(); i++)
            {
                dfCheckSumw += rgWt[i];
            }

            m_log.WriteLine("weight diff checksum " + dfCheckSumw.ToString());
        }
    }
}
