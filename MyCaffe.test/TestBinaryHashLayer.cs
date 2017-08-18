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
using MyCaffe.layers.alpha;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBinaryHashLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            BinaryHashLayerTest test = new BinaryHashLayerTest();

            try
            {
                foreach (IBinaryHashLayerTest t in test.Tests)
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
            BinaryHashLayerTest test = new BinaryHashLayerTest();

            try
            {
                foreach (IBinaryHashLayerTest t in test.Tests)
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
        public void TestRun()
        {
            BinaryHashLayerTest test = new BinaryHashLayerTest();

            try
            {
                foreach (IBinaryHashLayerTest t in test.Tests)
                {
                    t.TestRun();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IBinaryHashLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestRun();
    }

    class BinaryHashLayerTest : TestBase
    {
        public BinaryHashLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Binary Hash Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BinaryHashLayerTest<double>(strName, nDeviceID, engine);
            else
                return new BinaryHashLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BinaryHashLayerTest<T> : TestEx<T>, IBinaryHashLayerTest
    {
        Blob<T> m_blobFC6; // fc6 in AlexNet
        Blob<T> m_blobFC7; // fc7 in AlexNet
        Blob<T> m_blobFC8; // fc8 in AlexNet
        Blob<T> m_blobLabels;

        public BinaryHashLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blobFC6 = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 4096 });
            m_blobFC7 = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 128 });
            m_blobFC8 = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 10 });
            m_blobLabels = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 1 });

            BottomVec.Clear();
            BottomVec.Add(m_blobFC8);
            BottomVec.Add(m_blobFC7);
            BottomVec.Add(m_blobFC6);
            BottomVec.Add(m_blobLabels);
        }

        protected override void dispose()
        {
            m_blobFC6.Dispose();
            m_blobFC7.Dispose();
            m_blobFC8.Dispose();
            base.dispose();
        }

        private void fill(int nIdx, int nMax)
        {
            fill(m_blobFC8, nIdx, nMax);
            fill(m_blobFC7, nIdx, nMax);
            fill(m_blobFC6, nIdx, nMax);
            m_blobLabels.SetData(nIdx);
        }

        private void fill(Blob<T> blob, int nIdx, int nMax)
        {
            /// Fill the blob first with the signal data (0.4 to 0.9) range.
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = 0.4;
            fp.max = 0.9;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(blob);

            // Extract a 'segment's worth of signal
            int nSegment = (blob.count() / blob.num) / nMax;
            double[] rgData = convert(blob.mutable_cpu_data);
            double[] rgSignal = new double[nSegment];

            for (int i = 0; i < nSegment; i++)
            {
                rgSignal[i] = rgData[i];
            }

            // Re-initialize data with random data (0.0 to 0.6) range
            fp.min = 0.0;
            fp.max = 0.6;
          
            filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(blob);

            // Fill the specific segement with the signal data.
            rgData = convert(blob.mutable_cpu_data);
            nIdx *= nSegment;

            for (int i = 0; i < nSegment; i++)
            {
                rgData[i + nIdx] = rgSignal[i];
            }

            blob.mutable_cpu_data = convert(rgData);

            // Final data shoul be: rand rand signal rand rand... where the signal is at the index position.
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BINARYHASH);
            BinaryHashLayer<T> layer = new BinaryHashLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(3, layer.blobs.Count, "There should be 3 blobs for the layer.");

            int nLabelCount = m_blobFC8.count() / m_blobFC8.num;
            int nCacheDepth = p.binary_hash_param.cache_depth;
            int nLayer2Dim = m_blobFC7.count() / m_blobFC7.num;
            int nLayer3Dim = m_blobFC6.count() / m_blobFC6.num;

            // Verify cache #1 (for first pass)
            m_log.CHECK_EQ(nLabelCount, layer.blobs[0].shape(0), "There should be shape(0)=" + nLabelCount.ToString() + " in the blob[0].");
            m_log.CHECK_EQ(nCacheDepth, layer.blobs[0].shape(1), "There should be shape(1)=" + nCacheDepth.ToString() + " in the blob[0].");
            m_log.CHECK_EQ(nLayer2Dim, layer.blobs[0].shape(2), "There should be shape(2)=" + nLayer2Dim.ToString() + " in the blob[0].");

            // Verify cache #2 (for second, fine-grain pass)
            m_log.CHECK_EQ(nLabelCount, layer.blobs[1].shape(0), "There should be shape(0)=" + nLabelCount.ToString() + " in the blob[1].");
            m_log.CHECK_EQ(nCacheDepth, layer.blobs[1].shape(1), "There should be shape(1)=" + nCacheDepth.ToString() + " in the blob[1].");
            m_log.CHECK_EQ(nLayer3Dim, layer.blobs[1].shape(2), "There should be shape(2)=" + nLayer3Dim + " in the blob[1].");

            m_log.CHECK(BottomVec[0].shape_string == TopVec[0].shape_string, "The bottom(0) and top(0) should have the same shape.");
        }

        public void TestForward()
        {
            testForward(100);
        }

        private BinaryHashLayer<T> testForward(int nIterations)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BINARYHASH);
            p.phase = Phase.TRAIN;
            BinaryHashLayer<T> layer = new BinaryHashLayer<T>(m_cuda, m_log, p);
            Random rand = new Random();

            layer.Setup(BottomVec, TopVec);

            for (int j = 0; j < nIterations; j++)
            {
                for (int i = 0; i < 10; i++)
                {
                    double dfMin = i;
                    double dfMax = dfMin + rand.NextDouble();

                    fill(i, 10);
                    layer.Forward(BottomVec, TopVec);
                }
            }

            int nLabelCount = m_blobFC8.count() / m_blobFC8.num;
            int nCacheDepth = p.binary_hash_param.cache_depth;
            int nLayer2Dim = m_blobFC7.count() / m_blobFC7.num;
            int nLayer3Dim = m_blobFC6.count() / m_blobFC6.num;
            int nSegment2 = nLayer2Dim / nLabelCount;
            int nSegment3 = nLayer3Dim / nLabelCount;

            double[] rgBlobs0 = convert(layer.blobs[0].mutable_cpu_data);

            for (int i = 0; i < nLabelCount; i++)
            {
                for (int k = 0; k < nCacheDepth; k++)
                {
                    for (int j = 0; j < nLayer2Dim; j++)
                    {
                        int nIdx = (i * nCacheDepth * nLayer2Dim) + (k * nLayer2Dim) + j;
                        double dfVal = rgBlobs0[nIdx];

                        double dfMin = 0.0;
                        double dfMax = 0.6;

                        if (j >= (i * nSegment2) && j < ((i+1) * nSegment2))
                        {
                            dfMin = 0.4;
                            dfMax = 0.9;
                        }

                        m_log.CHECK_GE(dfVal, dfMin, "The blob(0) value at " + nIdx.ToString("N0") + " is less than the expected minimum of " + dfMin.ToString());
                        m_log.CHECK_LT(dfVal, dfMax, "The blob(0) value at " + nIdx.ToString("N0") + " is greater than the expected maximum of " + dfMax.ToString());
                    }
                }
            }

            double[] rgBlob1 = convert(layer.blobs[1].mutable_cpu_data);

            for (int i = 0; i < nLabelCount; i++)
            {
                for (int k = 0; k < nCacheDepth; k++)
                {
                    for (int j = 0; j < nLayer3Dim; j++)
                    {
                        int nIdx = (i * nCacheDepth * nLayer3Dim) + (k * nLayer3Dim) + j;
                        double dfVal = rgBlob1[nIdx];

                        double dfMin = 0.0;
                        double dfMax = 0.6;

                        if (j >= (i * nSegment3) && j < ((i + 1) * nSegment3))
                        {
                            dfMin = 0.4;
                            dfMax = 0.9;
                        }

                        m_log.CHECK_GE(dfVal, dfMin, "The blob(1) value at " + nIdx.ToString("N0") + " is less than the expected minimum of " + dfMin.ToString());
                        m_log.CHECK_LT(dfVal, dfMax, "The blob(1) value at " + nIdx.ToString("N0") + " is greater than the expected maximum of " + dfMax.ToString());
                    }
                }
            }

            return layer;
        }

        public void TestRun()
        {
            BinaryHashLayer<T> layer = testForward(1000);

            layer.SetPhase(Phase.RUN);

            for (int i = 0; i < 10; i++)
            {
                double dfMin = i;
                double dfMax = dfMin + 0.9;

                fill(i, 10);
                layer.Forward(BottomVec, TopVec);

                double[] rgData = convert(TopVec[0].update_cpu_data());
                int nLabel = findLabel(rgData);

                m_log.CHECK_EQ(i, nLabel, "The label should be " + i.ToString());
            }
        }

        private int findLabel(double[] rg)
        {
            int nIdx = -1;
            double dfMax = -double.MaxValue;

            for (int i = 0; i < rg.Length; i++)
            {
                if (rg[i] > dfMax)
                {
                    dfMax = rg[i];
                    nIdx = i;
                }
            }

            return nIdx;
        }
    }
}
