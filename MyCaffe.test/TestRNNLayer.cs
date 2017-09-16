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
using MyCaffe.imagedb;

namespace MyCaffe.test
{
    [TestClass]
    public class TestRNNLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            RNNLayerTest test = new RNNLayerTest();

            try
            {
                foreach (IRNNLayerTest t in test.Tests)
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
            RNNLayerTest test = new RNNLayerTest();

            try
            {
                foreach (IRNNLayerTest t in test.Tests)
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
            RNNLayerTest test = new RNNLayerTest();

            try
            {
                foreach (IRNNLayerTest t in test.Tests)
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
            RNNLayerTest test = new RNNLayerTest();

            try
            {
                foreach (IRNNLayerTest t in test.Tests)
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
            RNNLayerTest test = new RNNLayerTest();

            try
            {
                foreach (IRNNLayerTest t in test.Tests)
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
            RNNLayerTest test = new RNNLayerTest();

            try
            {
                foreach (IRNNLayerTest t in test.Tests)
                {
                    t.TestGradientNonZeroContBufferSize2WithStaticInput();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IRNNLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestGradient();
        void TestGradientNonZeroCont();
        void TestGradientNonZeroContBufferSize2();
        void TestGradientNonZeroContBufferSize2WithStaticInput();
    }

    class RNNLayerTest : TestBase
    {
        public RNNLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("RNN Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new RNNLayerTest<double>(strName, nDeviceID, engine);
            else
                return new RNNLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class RNNLayerTest<T> : TestEx<T>, IRNNLayerTest
    {
        Blob<T> m_blob_bottom_cont;
        Blob<T> m_blob_bottom_static;
        int m_nNumOutput = 7;
        LayerParameter m_param;
        CancelEvent m_evtCancel = new CancelEvent();

        public RNNLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom_cont = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_static = new Blob<T>(m_cuda, m_log);

            BottomVec.Add(m_blob_bottom_cont);

            ReshapeBlobs(1, 3);

            m_param = new LayerParameter(LayerParameter.LayerType.RNN);
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
            base.dispose();
        }

        public void ReshapeBlobs(int nNumTimesteps, int nNumInstances)
        {
            m_blob_bottom.Reshape(nNumTimesteps, nNumInstances, 3, 2);
            m_blob_bottom_static.Reshape(nNumInstances, 2, 3, 4);
            List<int> rgShape = new List<int>() { nNumTimesteps, nNumInstances };
            m_blob_bottom_cont.Reshape(rgShape);

            FillerParameter filler_param = new FillerParameter("uniform");
            filler_param.min = -1;
            filler_param.max = 1;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);

            filler.Fill(m_blob_bottom);
        }

        public void TestSetup()
        {
            RNNLayer<T> layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);

            layer.Setup(BottomVec, TopVec);

            List<int> rgExpectedTopShape = Utility.Clone<int>(m_blob_bottom.shape(), 3);
            rgExpectedTopShape[2] = m_nNumOutput;
            m_log.CHECK(Utility.Compare<int>(m_blob_top.shape(), rgExpectedTopShape), "The top shape is not as expected.");

            layer.Dispose();
        }

        public void TestForward()
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
            sequence_filler.Fill(m_blob_bottom);

            RNNLayer<T> layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            m_cuda.rng_setseed(1701);
            layer.Setup(BottomVec, TopVec);

            m_log.WriteLine("Calling forward for full sequence RNN");
            layer.Forward(BottomVec, TopVec);

            // Copy the inputs and outputs to reuse/check them later.
            Blob<T> bottom_copy = new Blob<T>(m_cuda, m_log, m_blob_bottom.shape());
            bottom_copy.CopyFrom(m_blob_bottom);
            Blob<T> top_copy = new Blob<T>(m_cuda, m_log, m_blob_top.shape());
            top_copy.CopyFrom(m_blob_top);

            // Process the batch one step at a time;
            //  check that we get the same result.
            ReshapeBlobs(1, nNum);

            layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
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

                m_log.WriteLine("Calling forward for RNN timestep " + t.ToString());
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
            layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
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

                m_log.WriteLine("Calling forward for RNN timestep " + t.ToString());
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

        public void TestGradient()
        {
            RNNLayer<T> layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestGradientNonZeroCont()
        {
            RNNLayer<T> layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
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

            RNNLayer<T> layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
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

            RNNLayer<T> layer = new RNNLayer<T>(m_cuda, m_log, m_param, m_evtCancel);
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
    }
}
