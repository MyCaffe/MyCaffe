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

namespace MyCaffe.test
{
    [TestClass]
    public class TestBatchNormLayer
    {
        [TestMethod]
        public void TestForward()
        {
            BatchNormLayerTest test = new BatchNormLayerTest();

            try
            {
                foreach (IBatchNormLayerTest t in test.Tests)
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
        public void TestForwardInplace()
        {
            BatchNormLayerTest test = new BatchNormLayerTest();

            try
            {
                foreach (IBatchNormLayerTest t in test.Tests)
                {
                    t.TestForwardInplace();
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
            BatchNormLayerTest test = new BatchNormLayerTest();

            try
            {
                foreach (IBatchNormLayerTest t in test.Tests)
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
        public void TestForwardCuDnn()
        {
            BatchNormLayerTest test = new BatchNormLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IBatchNormLayerTest t in test.Tests)
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
        public void TestForwardInplaceCuDnn()
        {
            BatchNormLayerTest test = new BatchNormLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IBatchNormLayerTest t in test.Tests)
                {
                    t.TestForwardInplace();
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
            BatchNormLayerTest test = new BatchNormLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IBatchNormLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IBatchNormLayerTest : ITest
    {
        void TestForward();
        void TestForwardInplace();
        void TestGradient();
    }

    class BatchNormLayerTest : TestBase
    {
        public BatchNormLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("BatchNorm Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BatchNormLayerTest<double>(strName, nDeviceID, engine);
            else
                return new BatchNormLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BatchNormLayerTest<T> : TestEx<T>, IBatchNormLayerTest
    {
        public BatchNormLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 5, 2, 3, 4 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
            p.batch_norm_param.engine = m_engine;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test mean
            int nNum = Bottom.num;
            int nChannels = Bottom.channels;
            int nHeight = Bottom.height;
            int nWidth = Bottom.width;

            for (int j = 0; j < nChannels; j++)
            {
                double dfSum = 0;
                double dfVar = 0;

                for (int i = 0; i < nNum; i++)
                {
                    for (int k=0; k<nHeight; k++)
                    {
                        for (int l=0; l<nWidth; l++)
                        {
                            T fData = Top.data_at(i, j, k, l);
                            double dfData = convert(fData);

                            dfSum += dfData;
                            dfVar += dfData * dfData;
                        }
                    }
                }

                dfSum /= nHeight * nWidth * nNum;
                dfVar /= nHeight * nWidth * nNum;

                double dfKErrorBound = 0.001;

                // expect zero mean
                m_log.EXPECT_NEAR(0.0, dfSum, dfKErrorBound);
                // expect unit variance
                m_log.EXPECT_NEAR(1.0, dfVar, dfKErrorBound);
            }
        }

        public void TestForwardInplace()
        {
            Blob<T> blobInPlace = new Blob<T>(m_cuda, m_log, 5, 2, 3, 4);
            BlobCollection<T> colBottom = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
            p.batch_norm_param.engine = m_engine;
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(blobInPlace);

            colBottom.Add(blobInPlace);
            colTop.Add(blobInPlace);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(colBottom, colTop);
            layer.Forward(colBottom, colTop);

            // Test mean
            int nNum = blobInPlace.num;
            int nChannels = blobInPlace.channels;
            int nHeight = blobInPlace.height;
            int nWidth = blobInPlace.width;

            for (int j = 0; j < nChannels; j++)
            {
                double dfSum = 0;
                double dfVar = 0;

                for (int i = 0; i < nNum; i++)
                {
                    for (int k = 0; k < nHeight; k++)
                    {
                        for (int l = 0; l < nWidth; l++)
                        {
                            T fData = blobInPlace.data_at(i, j, k, l);
                            double dfData = convert(fData);

                            dfSum += dfData;
                            dfVar += dfData * dfData;
                        }
                    }
                }

                dfSum /= nHeight * nWidth * nNum;
                dfVar /= nHeight * nWidth * nNum;

                double dfKErrorBound = 0.001;

                // expect zero mean
                m_log.EXPECT_NEAR(0.0, dfSum, dfKErrorBound);
                // expect unit variance
                m_log.EXPECT_NEAR(1.0, dfVar, dfKErrorBound);
            }

            blobInPlace.Dispose();
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
            p.batch_norm_param.engine = m_engine;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-4);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
