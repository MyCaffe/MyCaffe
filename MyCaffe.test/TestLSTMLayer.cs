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
using System.IO;
using System.Diagnostics;
using MyCaffe.solvers;

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
                    t.TestCuDnn(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCuDnnWithTensorCores()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestCuDnn(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCuDnn()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 1, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCuDnnTwoStreams()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 2, 1, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCuDnnTwoInputOneOutputStreams()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 1, 1, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCaffe()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 1, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCuDnnBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 10, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCuDnnTwoStreamsBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 2, 10, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCuDnnTwoInputOneOutputStreamsBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 1, 10, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSineWaveWithGradientCaffeBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 10, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCuDnn()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 1, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCuDnnTwoStreams()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 2, 1, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCuDnnTwoInputOneOutputStreams()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 1, 1, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCaffe()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 1, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCuDnnBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 10, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCuDnnTwoStreamsBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 2, 10, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCuDnnTwoInputOneOutputStreamsBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 2, 15, 10, 1, 10, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRandomWaveWithGradientCaffeBatch()
        {
            LSTMLayerTest test = new LSTMLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILSTMLayerTest t in test.Tests)
                {
                    t.TestTraining(4000, 100, 1, 15, 10, 1, 10, true, 4000, DataStream.DATASTREAM_TYPE.RANDOM_STDDEV);
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
        void TestCuDnn(bool bUseTensorCores);
        void TestSetup();
        void TestForward(Phase phase = Phase.NONE);
        void TestGradient();
        void TestGradientNonZeroCont();
        void TestGradientNonZeroContBufferSize2();
        void TestGradientNonZeroContBufferSize2WithStaticInput();

        void TestLSTMUnitSetup();
        void TestLSTMUnitGradient();
        void TestLSTMUnitGradientNonZeroCont();

        void TestTraining(int nTotalDataLength, int nSteps, int nNumInputs, int nNumHidden, int nNumOutputs, int nNumItemsPerOutput, int nBatch, bool bShortModel, int nMaxIter = 4000, DataStream.DATASTREAM_TYPE type = DataStream.DATASTREAM_TYPE.SINWAVE);
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
        const double PREDICTION_STUB = -2;

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

                    double dfTop1 = rgTop[i];
                    double dfTop0 = rgTopCopy[t * nTopCount + i];

                    m_log.EXPECT_NEAR(dfTop1, dfTop0, kEpsilon, "t = " + t.ToString() + "; i = " + i.ToString());
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
                    double dfTop1 = rgTop[i];
                    double dfTop0 = rgTopCopy[t * nTopCount + i];

                    if (t == 0)
                        m_log.EXPECT_NEAR(dfTop1, dfTop0, kEpsilon, "t = " + t.ToString() + "; i = " + i.ToString());
                    else
                        m_log.CHECK_NE(dfTop1, dfTop0, "t = " + t.ToString() + "; i = " + i.ToString());
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

        public void TestCuDnn(bool bUseTensorCores)
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
            ulong nWorkspaceCount = 0;
            long hReserved = 0;
            ulong nReservedCount = 0;
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
                m_cuda.SetRnnDesc(hCuDnn, hRnnDesc, nHiddenSize, nNumLayers, hDropoutDesc, RNN_MODE.LSTM, bUseTensorCores);

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
                hWorkspace = m_cuda.AllocMemory((long)nWorkspaceCount);
                hReserved = m_cuda.AllocMemory((long)nReservedCount);


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
                printCheckSums(blobX, blobHx, blobCx, nSeqLen, nBatchSize, nInputSize, nNumLayers, true);
                printCheckSums(blobWt);
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

        void printCheckSums(Blob<T> blob, Blob<T> blobH, Blob<T> blobC, int nSeqLen, int nBatchSize, int nHiddenSize, int nNumLayers, bool bDiff)
        {
            double[] rg = convert((bDiff) ? blob.update_cpu_diff() : blob.update_cpu_data());
            double[] rgH = convert((bDiff) ? blobH.update_cpu_diff() : blobH.update_cpu_data());
            double[] rgC = convert((bDiff) ? blobC.update_cpu_diff() : blobC.update_cpu_data());

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
                        dfLocalSumi += rg[j * nBatchSize * nHiddenSize + m * nHiddenSize + i];
                    }
                }

                for (int j = 0; j < nNumLayers; j++)
                {
                    for (int i = 0; i < nHiddenSize; i++)
                    {
                        dfLocalSumh += rgH[j * nHiddenSize * nBatchSize + m * nHiddenSize + i];
                        dfLocalSumc += rgC[j * nHiddenSize * nBatchSize + m * nHiddenSize + i];
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

        /// <summary>
        /// Test the training portion of the LSTM, predict the next in the output based on the sequence.
        /// </summary>
        /// <param name="nTotalDataLength">Specifies the data length.</param>
        /// <param name="nSteps">Specifies the number of time steps.</param>
        /// <param name="nNumInputs">Specifies the number of inputs.</param>
        /// <param name="nNumHidden">Specifies the number of LSTM hidden outputs.</param>
        /// <param name="nNumOutputs">Specifies the number of output predictions.</param>
        /// <param name="nBatch">Specifies the batch size.</param>
        /// <param name="bShortModel">Specifies whether or not to use a short model specification.</param>
        /// <param name="nMaxIter">Specifies the maximum number of iterations.</param>
        public void TestTraining(int nTotalDataLength, int nSteps, int nNumInputs, int nNumHidden, int nNumOutputs, int nNumItemsPerOutput, int nBatch, bool bShortModel, int nMaxIter, DataStream.DATASTREAM_TYPE type)
        {
            string strType = (typeof(T) == typeof(double)) ? "DOUBLE" : "FLOAT";
            string strPath = "c:\\temp\\test_results\\";
            int nDeviceID = m_cuda.GetDeviceID();
            T tOne = (T)Convert.ChangeType(1.0, typeof(T));
            T tZero = (T)Convert.ChangeType(0.0, typeof(T));
            NetParameter net_param = getTrainModel(nSteps, nNumInputs, nNumHidden, nNumOutputs, nNumItemsPerOutput, nBatch, type);
            SolverParameter solver_param = getSolver(net_param, nDeviceID, nMaxIter);

            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strModel1 = solver_param.net_param.ToProto("root").ToString();

            // Set device id
            m_log.WriteLine("Running test on " + m_cuda.GetDeviceName(solver_param.device_id) + " with ID = " + solver_param.device_id.ToString());
            m_log.WriteLine("type: " + typeof(T).ToString() + " num_output: " + nNumHidden.ToString() + " batch: " + nBatch.ToString() + " MaxIter: " + nMaxIter.ToString());
            m_cuda.SetDeviceID(solver_param.device_id);

            Solver<T> solver = Solver<T>.Create(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, null);
            Net<T> train_net = solver.net;
            Net<T> test_net = solver.test_nets[0];

            Assert.AreEqual(true, train_net.has_blob("data"));
            Assert.AreEqual(true, train_net.has_blob("clip"));
            Assert.AreEqual(true, train_net.has_blob("label"));
            Assert.AreEqual(true, test_net.has_blob("data"));
            Assert.AreEqual(true, test_net.has_blob("clip"));

            Blob<T> train_data_blob = train_net.blob_by_name("data");
            Blob<T> train_label_blob = train_net.blob_by_name("label");
            Blob<T> train_clip_blob = train_net.blob_by_name("clip");

            Blob<T> test_data_blob = test_net.blob_by_name("data");
            Blob<T> test_clip_blob = test_net.blob_by_name("clip");

            int seq_length = train_data_blob.shape(0);
            Assert.AreEqual(0, nTotalDataLength % seq_length);

            // Initialize bias for the forget gate to 1 to speed up LSTM learning.
            for (int i = 0; i < train_net.layers.Count; i++)
            {
                if (train_net.layers[i].type != LayerParameter.LayerType.LSTM)
                    continue;

                if (train_net.layers[i].blobs.Count <= 1) // cuDNN does not have a bias blob.
                    continue;

                int h = (int)train_net.layers[i].layer_param.recurrent_param.num_output;
                Blob<T> bias = train_net.layers[i].blobs[1];

                double[] rgBias = convert(bias.mutable_cpu_data);

                for (int j = 0; j < h; j++)
                {
                    rgBias[h + j] = 1.0;
                }

                bias.mutable_cpu_data = convert(rgBias);

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }


            //-------------------------------------------------------
            //  Training
            //-------------------------------------------------------

            Stopwatch swStatus = new Stopwatch();
            Stopwatch sw1 = new Stopwatch();
            TestingProgressSet progress = new TestingProgressSet();
            double dfTotalTime = 0;

            swStatus.Start();

            // Set clip to 0 for first batch, 1 otherwise.
            train_clip_blob.SetData(1.0);
            train_clip_blob.SetData(0.0, 0, nBatch);


            // Load the initial data with seq_length in data, and next nNumOutputs in label.
            double[] rgData;
            double[] rgLabel;
            DataStream data = new DataStream(nTotalDataLength, seq_length, nNumInputs, nNumOutputs, nNumItemsPerOutput, nBatch, type, strPath, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);

            data.CreateArrays(out rgData, out rgLabel);
            data.LoadData(true, rgData, rgLabel, m_param.type, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);

            // Training loop;
            int iter = 0;
            while (iter < solver_param.max_iter)
            {
                train_data_blob.mutable_cpu_data = convert(rgData.ToArray());
                train_label_blob.mutable_cpu_data = convert(rgLabel.ToArray());

                sw1.Start();
                solver.Step(1);

                sw1.Stop();
                dfTotalTime += sw1.Elapsed.TotalMilliseconds;
                sw1.Reset();

                iter++;

                if (swStatus.Elapsed.TotalMilliseconds > 2000)
                {
                    m_log.Progress = (double)iter / (double)solver_param.max_iter;
                    m_log.WriteLine("iteration = " + iter.ToString() + "  (" + m_log.Progress.ToString("P") + ")  ave solver time = " + (dfTotalTime / iter).ToString() + " ms.");
                    swStatus.Restart();

                    progress.SetProgress(m_log.Progress);
                }

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }

                data.LoadData(false, rgData, rgLabel, m_param.type, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);
            }

            m_log.WriteLine("Solving completed.");


            //-------------------------------------------------------
            //  Testing 
            //-------------------------------------------------------

            // Copy the trained weights to the test net.
            test_net.ShareTrainedLayersWith(train_net);

            string strFileName = strPath + "lstm_" + strType + "_" + type.ToString() + "_" + m_engine.ToString() + "_next_TEST_results_num_in_" + nNumInputs.ToString() + "_num_out_" + nNumHidden.ToString() + "_items_" + nNumItemsPerOutput.ToString() + "_batch_" + nBatch.ToString() + "_maxiter_" + nMaxIter.ToString() + ".csv";
            if (File.Exists(strFileName))
                File.Delete(strFileName);

            swStatus.Restart();

            // Set the clip mask
            test_clip_blob.SetData(1.0);
            test_clip_blob.SetData(0.0, 0, nBatch);

            // Load the first sequence data.
            data.Reset(true);
            data.LoadData(true, rgData, rgLabel, m_param.type, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);

            using (StreamWriter sw = new StreamWriter(strFileName))
            {
                sw.WriteLine("Running Test Network with: " + m_engine.ToString() + ", batch=" + nBatch.ToString() + ", numHidden=" + nNumHidden.ToString());

                string strHeader = "";

                for (int i = 0; i < nNumOutputs; i++)
                {
                    for (int j = 0; j < nNumItemsPerOutput; j++)
                    {
                        strHeader += "expected" + i.ToString() + "_" + j.ToString() + ",predicted" + i.ToString() + "_" + j.ToString();
                    }

                    strHeader += ",,";
                }

                sw.WriteLine(strHeader.TrimEnd(','));

                for (int i = 0; i < nTotalDataLength; i++)
                {
                    double dfLoss;

                    test_data_blob.mutable_cpu_data = convert(rgData);
                    test_net.Forward(out dfLoss);

                    Blob<T> pred = test_net.FindBlob("ip1");
                    m_log.CHECK_EQ(pred.count(), (nSteps + nNumOutputs) * nNumItemsPerOutput * nBatch, "The result blob should equal the number of data + predictions.");

                    double[] rgPredictions = convert(pred.mutable_cpu_data);
                    double[] rgPredictions2 = null;

                    if (type != DataStream.DATASTREAM_TYPE.SINWAVE)
                    {
                        Blob<T> pred2 = test_net.FindBlob("power1");
                        rgPredictions2 = convert(pred2.mutable_cpu_data);
                    }

                    string strLine = "";

                    for (int j = 0; j < nNumOutputs; j++)
                    {
                        for (int k = 0; k < nNumItemsPerOutput; k++)
                        {
                            double dfPredicted = rgPredictions[((nBatch - 1) * (seq_length + nNumOutputs) * nNumItemsPerOutput) + (seq_length + j) * nNumItemsPerOutput + k];  // predictions are the last 'nNumOutput' count of the outputs.
                            double dfExpected = rgLabel[((nBatch - 1) * nNumOutputs * nNumItemsPerOutput) + (j * nNumItemsPerOutput) + k];

                            strLine += dfExpected.ToString() + "," + dfPredicted.ToString() + ",";

                            if (rgPredictions2 != null)
                            {
                                double dfPredicted2 = rgPredictions2[((nBatch - 1) * (seq_length + nNumOutputs) * nNumItemsPerOutput) + (seq_length + j) * nNumItemsPerOutput + k];  // predictions are the last 'nNumOutput' count of the outputs.
                                strLine += dfPredicted2.ToString() + ",";
                            }
                        }

                        strLine += ",";
                    }

                    sw.WriteLine(strLine);

                    if (swStatus.Elapsed.TotalMilliseconds > 2000)
                    {
                        m_log.Progress = (double)i / nTotalDataLength;
                        m_log.WriteLine("Testing iteration " + i.ToString() + " (" + m_log.Progress.ToString("P") + ")");
                        swStatus.Restart();
                    }

                    if (m_evtCancel.WaitOne(0))
                    {
                        m_log.WriteLine("Aborted.");
                        return;
                    }

                    data.LoadData(false, rgData, rgLabel, m_param.type, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);
                }
            }

            m_log.WriteLine("Testing completed.");


            //-------------------------------------------------------
            //  Running 
            //-------------------------------------------------------

            NetParameter net_param_deploy = getDeployModel(nSteps, nNumInputs, nNumHidden, nNumOutputs, nNumItemsPerOutput, nBatch, type);
            Net<T> deploy_net = new Net<T>(m_cuda, m_log, net_param_deploy, m_evtCancel, null, Phase.RUN);

            string strModel2 = net_param_deploy.ToProto("root").ToString();

            // Copy the trained weights to the deploy net.
            train_net.CopyTrainedLayersTo(deploy_net);

            strFileName = strPath + "lstm_" + strType + "_" + type.ToString() + "_" + m_engine.ToString() + "_next_DEPLOY_results_num_in_" + nNumInputs.ToString() + "_num_out_" + nNumHidden.ToString() + "_items_" + nNumItemsPerOutput.ToString() + "_maxiter_" + nMaxIter.ToString() + ".csv";
            if (File.Exists(strFileName))
                File.Delete(strFileName);

            Blob<T> deploy_data_blob = deploy_net.blob_by_name("data");
            Blob<T> deploy_clip_blob = deploy_net.blob_by_name("clip");

            // Load the first sequence data.
            data = new DataStream(nTotalDataLength, seq_length + nNumOutputs, nNumInputs, nNumOutputs, nNumItemsPerOutput, nBatch, type, strPath, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);
            data.CreateArrays(out rgData, out rgLabel);
            data.LoadData(true, rgData, rgLabel, m_param.type, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);

            swStatus.Restart();

            using (StreamWriter sw = new StreamWriter(strFileName))
            {
                sw.WriteLine("Running Deploy Network");
                string strHeader = "";

                for (int i = 0; i < nNumOutputs; i++)
                {
                    for (int j = 0; j < nNumItemsPerOutput; j++)
                    {
                        strHeader += "expected" + i.ToString() + "_" + j.ToString() + ",predicted" + i.ToString() + "_" + j.ToString();
                    }

                    strHeader += ",,";
                }

                sw.WriteLine(strHeader.TrimEnd(','));

                for (int i = 0; i < nTotalDataLength; i++)
                {
                    List<double> rgPredictions = new List<double>();
                    double dfLoss;

                    deploy_clip_blob.SetData(0.0);

                    for (int j = 0; j < seq_length; j++)
                    {
                        for (int k = 0; k < nNumInputs; k++)
                        {
                            deploy_data_blob.SetData(rgData[(j*nNumInputs) + k], k);
                        }

                        deploy_net.Forward(out dfLoss);
                        deploy_clip_blob.SetData(1.0);
                    }

                    for (int j = 0; j < nNumOutputs; j++)
                    {
                        for (int k = 0; k < nNumInputs; k++)
                        {
                            deploy_data_blob.SetData(PREDICTION_STUB, k);
                        }

                        BlobCollection<T> results = deploy_net.Forward(out dfLoss);
                        deploy_clip_blob.SetData(1.0);

                        m_log.CHECK_EQ(results.Count, 1, "There should only be one result blob.");
                        m_log.CHECK_EQ(results[0].count(), nNumItemsPerOutput, "There should be 1 outputs in the result.");
                        double[] rgPred = convert(results[0].mutable_cpu_data);

                        for (int k = 0; k < nNumItemsPerOutput; k++)
                        {
                            rgPredictions.Add(rgPred[k]);
                        }
                    }

                    string strLine = "";

                    for (int j = 0; j < nNumOutputs; j++)
                    {
                        for (int k = 0; k < nNumItemsPerOutput; k++)
                        {
                            double dfExpected = rgData[(seq_length + j) * nNumInputs + k];
                            double dfPredicted = rgPredictions[(j * nNumItemsPerOutput) + k];

                            strLine += dfExpected.ToString() + "," + dfPredicted.ToString() + ",";
                        }

                        strLine += ",";
                    }

                    sw.WriteLine(strLine.TrimEnd(','));

                    if (swStatus.Elapsed.TotalMilliseconds > 2000)
                    {
                        m_log.Progress = (double)i / nTotalDataLength;
                        m_log.WriteLine("Running iteration " + i.ToString() + " (" + m_log.Progress.ToString("P") + ")");
                        swStatus.Restart();
                    }

                    if (m_evtCancel.WaitOne(0))
                    {
                        m_log.WriteLine("Aborted.");
                        return;
                    }

                    data.LoadData(false, rgData, rgLabel, m_param.type, (type == DataStream.DATASTREAM_TYPE.SINWAVE) ? false : true);
                }
            }


            //-------------------------------------------------------
            //  Cleanup 
            //-------------------------------------------------------

            solver.Dispose();
            deploy_net.Dispose();
        }

        /// <summary>
        /// Get the solver descriptor.
        /// </summary>
        /// <param name="nMaxIter">Specifies the maximum number of iterations to run.</param>
        /// <returns>The solver parameter is returned.</returns>
        private SolverParameter getSolver(NetParameter net_param, int nDeviceID, int nMaxIter = 4000)
        {
            SolverParameter solver_param = new SolverParameter();

            solver_param.test_interval = 500;
            solver_param.test_iter.Add(100);
            solver_param.test_initialization = false;
            solver_param.max_iter = nMaxIter;
            solver_param.snapshot = nMaxIter;
            solver_param.display = 200;
            solver_param.type = SolverParameter.SolverType.ADAM;
            solver_param.base_lr = 0.002;
            solver_param.device_id = nDeviceID;
            solver_param.net_param = net_param;

            return solver_param;
        }

        /// <summary>
        /// Build the model for testing.
        /// </summary>
        /// <param name="nSteps">Specifies the number of time-steps.</param>
        /// <param name="nNumInputs">Specifies the number of input items (per time-step)</param>
        /// <param name="nNumHidden">Specifies the number of hidden outputs of the LSTM</param>
        /// <param name="nNumOutputs">Specifies the number of predicted outputs.</param>
        /// <param name="nBatch">Specifies the batch size.</param>
        /// <remarks>
        /// When using the non clockwork model to create forward predictions, a the data input is concatenated with dummy data to create inputs that then
        /// match the final labels which are a concatenation of the data + labels (containing future data).
        /// 
        /// For example:
        ///   timesteps = 100 (past items to use for prediction)
        ///   batch = 1
        ///   input = 1
        ///   hidden = 23
        ///   output = 10 (future items to predict)
        ///   
        /// Data Size = { 100, 1, 1 }
        /// Clip Size = { 100, 1 }
        /// Label Size = { 10, 1, 1 }
        /// 
        /// Dummy Data Size = { 10, 1, 1 } (to match label predictions)
        /// 
        /// Concat1 Size = { 110, 1, 1 } (Data + Dummy) - fed into LSTM
        /// 
        /// Concat2 Size = { 110, 1, 1 } (Data + Label) - fed into LOSS
        /// 
        /// This is similar to the way CorvusCorax loads the LSTM layer int he 'Caffe LSTM Example on Sin(t) Waveform Pediction'
        /// @see [GitHub: Caffe LSTM Example on Sin(t) Waveform Prediction](https://github.com/CorvusCorax/Caffe-LSTM-Mini-Tutorial) by CorvusCorax, 2019.
        /// </remarks>
        /// <returns>The string representing the model is returned.</returns>
        private NetParameter getTrainModel(int nSteps, int nNumInputs, int nNumHidden, int nNumOutputs, int nNumItemsPerOutput, int nBatch, DataStream.DATASTREAM_TYPE type)
        {
            NetParameter net_param = new NetParameter();

            net_param.name = "LSTM";

            LayerParameter input_layer = new LayerParameter(LayerParameter.LayerType.INPUT, "data");
            input_layer.top.Add("data");
            input_layer.top.Add("clip");
            input_layer.top.Add("label");
            input_layer.input_param.shape.Add(new BlobShape(new List<int>() { nSteps, nBatch, nNumInputs }));      // data
            input_layer.input_param.shape.Add(new BlobShape(new List<int>() { nSteps + nNumOutputs, nBatch }));    // clip
            input_layer.input_param.shape.Add(new BlobShape(new List<int>() { nNumOutputs, nBatch, nNumItemsPerOutput })); // label
            net_param.layer.Add(input_layer);

            LayerParameter dummy_data_layer = new LayerParameter(LayerParameter.LayerType.DUMMYDATA, "dummydata");
            dummy_data_layer.top.Add("dummy");
            dummy_data_layer.dummy_data_param.shape.Add(new BlobShape(new List<int>() { nNumOutputs, nBatch, nNumInputs })); // should match data batch and inputs.
            dummy_data_layer.dummy_data_param.data_filler.Add(new FillerParameter("constant", PREDICTION_STUB));
            net_param.layer.Add(dummy_data_layer);

            LayerParameter concat_layer1 = new LayerParameter(LayerParameter.LayerType.CONCAT, "concat1");
            concat_layer1.bottom.Add("data");
            concat_layer1.bottom.Add("dummy");
            concat_layer1.top.Add("fulldata");
            concat_layer1.concat_param.axis = 0;
            net_param.layer.Add(concat_layer1);

            LayerParameter lstm_layer = new LayerParameter(LayerParameter.LayerType.LSTM, "lstm1");
            lstm_layer.bottom.Add("fulldata");
            lstm_layer.bottom.Add("clip");
            lstm_layer.top.Add("lstm1");
            lstm_layer.recurrent_param.num_output = (uint)nNumHidden;
            lstm_layer.recurrent_param.engine = m_engine;
            lstm_layer.recurrent_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.05);
            lstm_layer.recurrent_param.bias_filler = new FillerParameter("constant", 0);
            if (m_engine == EngineParameter.Engine.CUDNN)
            {
                lstm_layer.recurrent_param.num_layers = 1;
                lstm_layer.recurrent_param.dropout_ratio = 0;
                lstm_layer.recurrent_param.dropout_seed = 0;
            }
            net_param.layer.Add(lstm_layer);

            LayerParameter innerproduct_layer = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ip1");
            innerproduct_layer.bottom.Add("lstm1");
            innerproduct_layer.top.Add("ip1");
            innerproduct_layer.inner_product_param.num_output = (uint)nNumItemsPerOutput;
            innerproduct_layer.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.1);
            innerproduct_layer.inner_product_param.bias_filler = new FillerParameter("constant", 0);
            innerproduct_layer.inner_product_param.axis = 2;
            net_param.layer.Add(innerproduct_layer);

            if (type != DataStream.DATASTREAM_TYPE.SINWAVE)
            {
                LayerParameter power_layer = new LayerParameter(LayerParameter.LayerType.POWER, "power1");
                power_layer.bottom.Add("ip1");
                power_layer.top.Add("power1");
                power_layer.power_param.scale = 2.0;
                power_layer.power_param.power = 1.0;
                power_layer.power_param.shift = 0;
                power_layer.include.Add(new NetStateRule(Phase.TEST));
                power_layer.propagate_down.Add(false);
                net_param.layer.Add(power_layer);
            }

            string strData = "data";
            if (nNumInputs > nNumItemsPerOutput)
            {
                LayerParameter slice_layer = new LayerParameter(LayerParameter.LayerType.SLICE, "slice");
                slice_layer.bottom.Add("data");
                slice_layer.top.Add("slice1");
                slice_layer.top.Add("slice2");
                slice_layer.slice_param.slice_point.Add((uint)nNumItemsPerOutput);
                slice_layer.slice_param.axis = 2;
                net_param.layer.Add(slice_layer);
                strData = "slice1";

                LayerParameter silence_layer = new LayerParameter(LayerParameter.LayerType.SILENCE, "silence");
                silence_layer.bottom.Add("slice2");
                net_param.layer.Add(silence_layer);
            }

            LayerParameter concat_layer2 = new LayerParameter(LayerParameter.LayerType.CONCAT, "concat2");
            concat_layer2.bottom.Add(strData);
            concat_layer2.bottom.Add("label");
            concat_layer2.top.Add("fulllabel");
            concat_layer2.concat_param.axis = 0;
            net_param.layer.Add(concat_layer2);

            LayerParameter euclidean_loss_layer = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS, "loss1");
            euclidean_loss_layer.bottom.Add("ip1");
            euclidean_loss_layer.bottom.Add("fulllabel");
            euclidean_loss_layer.top.Add("loss");
            net_param.layer.Add(euclidean_loss_layer);

            return net_param;
        }

        /// <summary>
        /// Build the model for running.
        /// </summary>
        /// <param name="nSteps">Specifies the number of time-steps.</param>
        /// <param name="nNumInputs">Specifies the number of input items (per time-step)</param>
        /// <param name="nNumHidden">Specifies the number of hidden outputs of the LSTM</param>
        /// <param name="nNumOutputs">Specifies the number of predicted outputs.</param>
        /// <param name="nBatch">Specifies the batch size.</param>
        /// <remarks>
        /// When using the non clockwork model to create forward predictions, a the data input is concatenated with dummy data to create inputs that then
        /// match the final labels which are a concatenation of the data + labels (containing future data).
        /// 
        /// For example:
        ///   timesteps = 100 (past items to use for prediction)
        ///   batch = 1
        ///   input = 1
        ///   hidden = 23
        ///   output = 10 (future items to predict)
        ///   
        /// Data Size = { 100, 1, 1 }
        /// Clip Size = { 100, 1 }
        /// Label Size = { 10, 1, 1 }
        /// 
        /// Dummy Data Size = { 10, 1, 1 } (to match label predictions)
        /// 
        /// Concat1 Size = { 110, 1, 1 } (Data + Dummy) - fed into LSTM
        /// 
        /// Concat2 Size = { 110, 1, 1 } (Data + Label) - fed into LOSS
        /// 
        /// This is similar to the way CorvusCorax loads the LSTM layer int he 'Caffe LSTM Example on Sin(t) Waveform Pediction'
        /// @see [GitHub: Caffe LSTM Example on Sin(t) Waveform Prediction](https://github.com/CorvusCorax/Caffe-LSTM-Mini-Tutorial) by CorvusCorax, 2019.
        /// </remarks>
        /// <returns>The string representing the model is returned.</returns>
        private NetParameter getDeployModel(int nSteps, int nNumInputs, int nNumHidden, int nNumOutputs, int nNumItemsPerOutput, int nBatch, DataStream.DATASTREAM_TYPE type)
        {
            NetParameter net_param = new NetParameter();

            net_param.name = "LSTM";

            LayerParameter input_layer = new LayerParameter(LayerParameter.LayerType.INPUT, "data");
            input_layer.top.Add("data");
            input_layer.top.Add("clip");
            input_layer.input_param.shape.Add(new BlobShape(new List<int>() { 1, 1, nNumInputs }));     // data
            input_layer.input_param.shape.Add(new BlobShape(new List<int>() { 1, 1 }));                 // clip
            net_param.layer.Add(input_layer);

            LayerParameter lstm_layer = new LayerParameter(LayerParameter.LayerType.LSTM, "lstm1");
            lstm_layer.bottom.Add("data");
            lstm_layer.bottom.Add("clip");
            lstm_layer.top.Add("lstm1");
            lstm_layer.recurrent_param.num_output = (uint)nNumHidden;
            lstm_layer.recurrent_param.engine = m_engine;
            lstm_layer.recurrent_param.weight_filler = new FillerParameter("xavier");
            lstm_layer.recurrent_param.bias_filler = new FillerParameter("constant", 0);
            if (m_engine == EngineParameter.Engine.CUDNN)
            {
                lstm_layer.recurrent_param.num_layers = 1;
                lstm_layer.recurrent_param.dropout_ratio = 0;
                lstm_layer.recurrent_param.dropout_seed = 0;
            }
            net_param.layer.Add(lstm_layer);

            LayerParameter innerproduct_layer = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ip1");
            innerproduct_layer.bottom.Add("lstm1");
            innerproduct_layer.top.Add("ip1");
            innerproduct_layer.inner_product_param.num_output = (uint)nNumItemsPerOutput;
            innerproduct_layer.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.1);
            innerproduct_layer.inner_product_param.bias_filler = new FillerParameter("constant", 0);
            innerproduct_layer.inner_product_param.axis = 2;
            net_param.layer.Add(innerproduct_layer);

            if (type != DataStream.DATASTREAM_TYPE.SINWAVE)
            {
                LayerParameter power_layer = new LayerParameter(LayerParameter.LayerType.POWER, "power1");
                power_layer.bottom.Add("ip1");
                power_layer.top.Add("power1");
                power_layer.power_param.scale = 2.0;
                power_layer.power_param.power = 1.0;
                power_layer.power_param.shift = 0;
                power_layer.propagate_down.Add(false);
                net_param.layer.Add(power_layer);
            }

            return net_param;
        }
    }

    public class DataStream
    {
        int m_nDataIdx = 0;
        int m_nLabelIdx = 0;
        List<double[]> m_rgrgSequence = new List<double[]>();
        List<List<double>> m_rgrgData1 = new List<List<double>>();
        List<List<double>> m_rgrgLabel1 = new List<List<double>>();
        int m_nBatch;
        int m_nSequenceLength;
        int m_nNumOutput;
        int m_nNumItemsPerOutput;
        int m_nNumInput;
        CryptoRandom m_random = new CryptoRandom();
        double m_dfAccumulator = 0;
        List<double> m_rgAccumulations = new List<double>();

        public enum DATASTREAM_TYPE
        {
            SINWAVE,
            RANDOM,
            RANDOM_STDDEV
        }

        delegate double fn(double dfT);

        public DataStream(int nTotalDataLength, int nSequenceLength, int nNumInput, int nNumOutput, int nNumItemsPerOutput, int nBatch, DATASTREAM_TYPE type, string strPath, bool bForceToRange = true, bool bCenter = false)
        {
            fn fn1;
            fn fn2;

            switch (type)
            {
                case DATASTREAM_TYPE.RANDOM:
                    fn1 = f_rand;
                    fn2 = f_sma;
                    break;

                case DATASTREAM_TYPE.RANDOM_STDDEV:
                    fn1 = f_rand_stdev;
                    fn2 = f_sma;
                    break;

                default:
                    fn1 = f_x1;
                    fn2 = f_x2;
                    break;
            }

            for (int i = 0; i < nNumInput; i++)
            {
                m_rgrgSequence.Add(new double[nTotalDataLength]);
            }

            // Construct the data.
            double[] rgdfMin = new double[nNumInput];
            double[] rgdfMax = new double[nNumInput];
            double[] rgdfMean = new double[nNumInput];

            for (int j = 0; j < nNumInput; j++)
            {
                rgdfMin[j] = double.MaxValue;
                rgdfMax[j] = -double.MaxValue;
            }

            for (int i = 0; i < nTotalDataLength; i++)
            {
                double df = fn1(i * 0.01);

                for (int j = 0; j < nNumInput; j++)
                {
                    m_rgrgSequence[j][i] = df;
                    rgdfMin[j] = Math.Min(rgdfMin[j], df);
                    rgdfMax[j] = Math.Max(rgdfMax[j], df);
                    df = fn2(df);
                }
            }

            if (bForceToRange)
            {
                for (int i = 0; i < nTotalDataLength; i++)
                {
                    for (int j = 0; j < nNumInput; j++)
                    {
                        double dfVal = m_rgrgSequence[j][i];

                        dfVal = (dfVal - rgdfMin[j]) / (rgdfMax[j] - rgdfMin[j]);
                        dfVal *= 2;
                        dfVal += -1;
                        rgdfMean[j] += dfVal;
                        m_rgrgSequence[j][i] = dfVal;
                    }
                }

                if (bCenter)
                {
                    for (int i = 0; i < nTotalDataLength; i++)
                    {
                        for (int j = 0; j < nNumInput; j++)
                        {
                            if (i == 0)
                                rgdfMean[j] /= nTotalDataLength;

                            m_rgrgSequence[j][i] -= rgdfMean[j];
                        }
                    }
                }
            }
#if DEBUG
            using (StreamWriter sw = new StreamWriter(strPath + "values.csv"))
            {
                for (int i = 0; i < nTotalDataLength; i++)
                {
                    string strLine = "";

                    for (int j = 0; j < nNumInput; j++)
                    {
                        strLine += m_rgrgSequence[j][i].ToString();
                        strLine += ",";
                    }

                    sw.WriteLine(strLine.TrimEnd(','));
                }
            }
#endif

            m_nSequenceLength = nSequenceLength;
            m_nNumOutput = nNumOutput;
            m_nNumInput = nNumInput;
            m_nNumItemsPerOutput = nNumItemsPerOutput;
            m_nBatch = nBatch;

            Reset();
        }

        private double f_x1(double dfT)
        {
            return 0.5 * Math.Sin(2 * dfT) - 0.05 * Math.Cos(17 * dfT + 0.8) + 0.05 * Math.Sin(25 * dfT + 10) - 0.02 * Math.Cos(45 * dfT + 0.3);
        }

        private double f_x2(double df)
        {
            return Math.Cos(df);
        }

        private double f_rand(double dfT)
        {
            double dfVal = m_random.NextDouble();

            dfVal -= 0.5;

            if (dfT == 0)
            {
                m_dfAccumulator = dfVal;
                m_rgAccumulations.Clear();
            }
            else
            {
                m_dfAccumulator += dfVal;
            }

            m_rgAccumulations.Add(m_dfAccumulator);

            if (m_rgAccumulations.Count > 20)
                m_rgAccumulations.RemoveAt(0);

            return m_dfAccumulator;
        }

        private double f_rand_stdev(double dfT)
        {
            double dfVal = m_random.NextDouble();
            dfVal -= 0.5;

            int nIdx = (int)dfT;
            int nIdxStdDev = nIdx % 30;
            int nRandStdDev = m_random.Next(nIdxStdDev);
            double dfVal2 = m_random.NextDouble();

            dfVal2 -= 0.5;
            dfVal2 *= ((double)nRandStdDev / 30.0);

            dfVal += dfVal2;

            if (dfT == 0)
            {
                m_dfAccumulator = dfVal;
                m_rgAccumulations.Clear();
            }
            else
            {
                m_dfAccumulator += dfVal;
            }

            m_rgAccumulations.Add(m_dfAccumulator);

            if (m_rgAccumulations.Count > 20)
                m_rgAccumulations.RemoveAt(0);

            return m_dfAccumulator;
        }

        private double f_sma(double df)
        {
            if (m_rgAccumulations.Count == 0)
                return 0;

            double dfTotal = 0;

            foreach (double df1 in m_rgAccumulations)
            {
                dfTotal += df1;
            }

            return dfTotal / m_rgAccumulations.Count;
        }

        public void Reset(bool bRandom = false)
        {
            m_nDataIdx = 0;

            if (bRandom)
                m_nDataIdx = m_random.Next(m_rgrgSequence[0].Length);

            m_nLabelIdx = m_nDataIdx + m_nSequenceLength;

            m_rgrgData1.Clear();
            m_rgrgLabel1.Clear();

            for (int j = 0; j < m_nNumInput; j++)
            {
                m_rgrgData1.Add(new List<double>());
                m_rgrgLabel1.Add(new List<double>());
            }
        }

        public void CreateArrays(out double[] rgData, out double[] rgLabel)
        {
            rgData = new double[m_nBatch * m_nSequenceLength * m_nNumInput];
            rgLabel = new double[m_nBatch * m_nNumOutput * m_nNumItemsPerOutput];
        }

        public void LoadData(bool bInitial, double[] rgData, double[] rgLabel, LayerParameter.LayerType type, bool bNormalize)
        {
            if (rgData == null || rgData.Length != m_nBatch * m_nSequenceLength * m_nNumInput)
                throw new Exception("The data length is incorrect!");

            if (rgLabel == null || rgLabel.Length != m_nBatch * m_nNumOutput * m_nNumItemsPerOutput)
                throw new Exception("The label length is incorrect!");

            for (int i = 0; i < m_nBatch; i++)
            {
                if (bInitial && i == 0)
                    loadData(i, m_rgrgData1, m_rgrgLabel1);
                else
                    loadNext(i, m_rgrgData1, m_rgrgLabel1);

                for (int j = 0; j < m_nSequenceLength; j++)
                {
                    // LSTM: Create input data, the data must be in the order
                    // seq1_val1, seq2_val1, ..., seqBatch_Size_val1, seq1_val2, seq2_val2, ..., seqBatch_Size_valSequence_Length
                    int nIdxData = (m_nBatch * j + i) * m_nNumInput;

                    for (int k = 0; k < m_nNumInput; k++)
                    {
                        rgData[nIdxData + k] = m_rgrgData1[k][j];
                    }
                }

                for (int j = 0; j < m_nNumOutput; j++)
                {
                    // LSTM: Create input data, the data must be in the order
                    // seq1_val1, seq2_val1, ..., seqBatch_Size_val1, seq1_val2, seq2_val2, ..., seqBatch_Size_valSequence_Length
                    int nIdxLabel = (m_nBatch * j + i) * m_nNumItemsPerOutput;

                    for (int k = 0; k < m_nNumItemsPerOutput; k++)
                    {
                        rgLabel[nIdxLabel + k] = m_rgrgLabel1[k][j];
                    }
                }
            }

            // Normalize across both data and label combined.
            if (bNormalize)
                normalize(rgData, rgLabel);

            // Transpose data for CAFFE ordering.
            //if (type == LayerParameter.LayerType.LSTM && m_nBatch > 1)
            //{
            //    double[] rgDataT = new double[rgData.Length];
            //    double[] rgLabelT = new double[rgLabel.Length];
            //    int nSrcIdx = 0;
            //    int nDstIdx = 0;

            //    for (int i = 0; i < m_nBatch; i++) // batch
            //    {
            //        for (int j = 0; j < m_nSequenceLength; j++)  // sequence
            //        {
            //            nDstIdx = m_nBatch * j + i;
            //            nSrcIdx++;
            //            rgDataT[nDstIdx] = rgData[nSrcIdx];
            //        }

            //        for (int j = 0; j < m_nNumOutput; j++)  // sequence
            //        {
            //            nDstIdx = m_nBatch * j + i;
            //            nSrcIdx++;
            //            rgLabelT[nDstIdx] = rgLabel[nSrcIdx];
            //        }
            //    }

            //    Array.Copy(rgDataT, rgData, rgData.Length);
            //    Array.Copy(rgLabelT, rgLabel, rgLabel.Length);
            //}
        }

        private void loadData(int nBatchIdx, List<List<double>> rgrgData, List<List<double>> rgrgLabel)
        {
            for (int j = 0; j < m_nNumInput; j++)
            {
                rgrgData[j].Clear();
                rgrgLabel[j].Clear();
            }

            for (int i = 0; i < m_nSequenceLength; i++)
            {
                for (int j = 0; j < m_nNumInput; j++)
                {
                    rgrgData[j].Add(m_rgrgSequence[j][m_nDataIdx]);
                }

                m_nDataIdx++;

                if (m_nDataIdx == m_rgrgSequence[0].Length)
                    m_nDataIdx = 0;
            }

            m_nLabelIdx = m_nDataIdx;

            for (int i = 0; i < m_nNumOutput; i++)
            {
                for (int j = 0; j < m_nNumItemsPerOutput; j++)
                {
                    rgrgLabel[j].Add(m_rgrgSequence[j][m_nLabelIdx]);
                }

                m_nLabelIdx++;

                if (m_nLabelIdx == m_rgrgSequence[0].Length)
                    m_nLabelIdx = 0;
            }
        }

        private void loadNext(int nBatchIdx, List<List<double>> rgrgData, List<List<double>> rgrgLabel)
        {
            for (int j = 0; j < m_nNumInput; j++)
            {
                rgrgData[j].Add(m_rgrgSequence[j][m_nDataIdx]);
            }

            m_nDataIdx++;

            if (m_nDataIdx == m_rgrgSequence[0].Length)
                m_nDataIdx = 0;

            for (int j = 0; j < m_nNumInput; j++)
            {
                rgrgData[j].RemoveAt(0);
            }

            for (int j = 0; j < m_nNumItemsPerOutput; j++)
            {
                rgrgLabel[j].Add(m_rgrgSequence[j][m_nLabelIdx]);
            }

            m_nLabelIdx++;

            if (m_nLabelIdx == m_rgrgSequence[0].Length)
                m_nLabelIdx = 0;

            for (int j = 0; j < m_nNumItemsPerOutput; j++)
            {
                rgrgLabel[j].RemoveAt(0);
            }
        }

        private void normalize(double[] rgData, double[] rgLabel)
        {
            double dfMin = double.MaxValue;
            double dfMax = -double.MaxValue;
            double dfSum = rgData.Sum(p => p) + rgLabel.Sum(p => p);
            double dfAve = dfSum / (rgData.Length + rgLabel.Length);

            for (int i = 0; i < rgData.Length; i++)
            {
                rgData[i] -= dfAve;
                dfMin = Math.Min(dfMin, rgData[i]);
                dfMax = Math.Max(dfMax, rgData[i]);
            }

            for (int i = 0; i < rgLabel.Length; i++)
            {
                rgLabel[i] -= dfAve;
                dfMin = Math.Min(dfMin, rgLabel[i]);
                dfMax = Math.Max(dfMax, rgLabel[i]);
            }

            double dfRange = dfMax - dfMin;

            for (int i = 0; i < rgData.Length; i++)
            {
                rgData[i] = (rgData[i] - dfMin) / dfRange;
            }

            for (int i = 0; i < rgLabel.Length; i++)
            {
                rgLabel[i] = (rgLabel[i] - dfMin) / dfRange;
            }
        }
    }
}
