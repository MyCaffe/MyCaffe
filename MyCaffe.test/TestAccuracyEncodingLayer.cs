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
    public class TestAccuracyEncodingLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            AccuracyEncodingLayerTest test = new AccuracyEncodingLayerTest();

            try
            {
                foreach (IAccuracyEncodingLayerTest t in test.Tests)
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
            AccuracyEncodingLayerTest test = new AccuracyEncodingLayerTest();

            try
            {
                foreach (IAccuracyEncodingLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IAccuracyEncodingLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
    }

    class AccuracyEncodingLayerTest : TestBase
    {
        public AccuracyEncodingLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("AccuracyEncoding Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new AccuracyEncodingLayerTest<double>(strName, nDeviceID, engine);
            else
                return new AccuracyEncodingLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class AccuracyEncodingLayerTest<T> : TestEx<T>, IAccuracyEncodingLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_label;
        int m_nNum = 3 * 42;
        int m_nEncodingDim = 3;

        public AccuracyEncodingLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);
            m_engine = engine;

            m_blob_bottom_label = new Blob<T>(m_cuda, m_log);

            List<int> rgShape = new List<int>() { m_nNum, m_nEncodingDim, 1, 1 };
            m_blob_bottom.Reshape(rgShape);
            rgShape = new List<int>() { m_nNum, 3, 1, 1 };
            m_blob_bottom_label.Reshape(rgShape);

            FillBottoms();

            BottomVec.Add(m_blob_bottom_label);
        }

        public void FillBottoms()
        {
            double[] rgData = convert(Bottom.mutable_cpu_data);
            double[] rgLabel = convert(m_blob_bottom_label.mutable_cpu_data);
            int nLabels = 3;
            int nIdx = 0;
            List<int> rgnLabel = new List<int>();

            m_log.CHECK_EQ(nLabels, m_nEncodingDim, "Encoding dim should be set to the number of labels = " + nLabels.ToString());

            // Fill the embeddings.
            for (int i = 0; i < m_nNum/nLabels; i++)
            {
                for (int j = 0; j < nLabels; j++)
                {
                    double dfVal0 = m_random.NextDouble();
                    double dfVal1 = m_random.NextDouble();

                    rgData[nIdx * m_nEncodingDim + 0] = ((j == 0) ? 0.8 + 0.2 * dfVal1 : 0.2 * dfVal0);
                    rgData[nIdx * m_nEncodingDim + 1] = ((j == 1) ? 0.8 + 0.2 * dfVal1 : 0.2 * dfVal0);
                    rgData[nIdx * m_nEncodingDim + 2] = ((j == 2) ? 0.8 + 0.2 * dfVal1 : 0.2 * dfVal0);

                    rgnLabel.Add(j);
                    nIdx++;
                }
            }

            nIdx = 0;
            for (int i = 0; i < m_blob_bottom_label.num; i++)
            {
                rgLabel[i * 3 + 0] = 0;
                rgLabel[i * 3 + 1] = rgnLabel[nIdx];
                rgLabel[i * 3 + 2] = 0;
                nIdx++;
            }

            m_blob_bottom.mutable_cpu_data = convert(rgData);
            m_blob_bottom_label.mutable_cpu_data = convert(rgLabel);
        }

        protected override void dispose()
        {
            m_blob_bottom_label.Dispose();
            base.dispose();
        }

        public Blob<T> BottomLabel
        {
            get { return m_blob_bottom_label; }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY_ENCODING);
            AccuracyEncodingLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as AccuracyEncodingLayer<T>;

            m_log.CHECK(layer != null, "The Accuracy Encoding layer is null.");

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(1, Top.num, "The top num should equal 1.");
            m_log.CHECK_EQ(1, Top.channels, "The top channels should equal 1.");
            m_log.CHECK_EQ(1, Top.height, "The top height should equal 1.");
            m_log.CHECK_EQ(1, Top.width, "The top width should equal 1.");
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY_ENCODING);
            AccuracyEncodingLayer<T> layer = new AccuracyEncodingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            // repeat the forward
            for (int iter = 0; iter < 3; iter++)
            {
                FillBottoms();
                layer.Forward(BottomVec, TopVec);

                double dfAccuracy = convert(TopVec[0].GetData(0));
                m_log.EXPECT_NEAR(dfAccuracy, 1.0, 1e-3, "Accuracy is not accurate.");
            }
        }
    }
}
