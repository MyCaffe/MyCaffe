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
using MyCaffe.param.beta;

///
/// WORK IN PROGRESS
///
namespace MyCaffe.test
{
    [TestClass]
    public class TestLayerNormLayer
    {
        [TestMethod]
        public void TestForward()
        {
            LayerNormLayerTest test = new LayerNormLayerTest();

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
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
            LayerNormLayerTest test = new LayerNormLayerTest();

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
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
            LayerNormLayerTest test = new LayerNormLayerTest();

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
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

    interface ILayerNormLayerTest : ITest
    {
        void TestForward();
        void TestForwardInplace();
        void TestGradient();
    }

    class LayerNormLayerTest : TestBase
    {
        public LayerNormLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("LayerNorm Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LayerNormLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LayerNormLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LayerNormLayerTest<T> : TestEx<T>, ILayerNormLayerTest
    {
        public LayerNormLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 3, 1 }, nDeviceID)
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

        private double[] calculateLayerNorm(Blob<T> b, LayerNormParameter p)
        {
            double[] rgData = convert(b.update_cpu_data());
            double[] rgNorm = new double[rgData.Length];
            int nSpatialDim = b.height * b.width;

            for (int n = 0; n < b.num; n++)
            {
                for (int c = 0; c < b.channels; c++)
                {
                    double dfTotal = 0;
                    
                    for (int i = 0; i < nSpatialDim; i++)
                    {
                        int nIdx = n * b.channels * nSpatialDim + c * nSpatialDim + i;
                        dfTotal += rgData[nIdx];                        
                    }

                    double dfMean = dfTotal / nSpatialDim;

                    dfTotal = 0;

                    for (int i = 0; i < nSpatialDim; i++)
                    {
                        int nIdx = n * b.channels * nSpatialDim + c * nSpatialDim + i;
                        double dfMeanDiff = rgData[nIdx] - dfMean;
                        double dfMeanDiffSq = dfMeanDiff * dfMeanDiff;
                        dfTotal += dfMeanDiffSq;
                    }

                    double dfVar = dfTotal / nSpatialDim;
                    double dfStd = Math.Sqrt(dfVar + p.epsilon);

                    for (int i = 0; i < nSpatialDim; i++)
                    {
                        int nIdx = n * b.channels * nSpatialDim + c * nSpatialDim + i;
                        double dfNorm = (rgData[nIdx] - dfMean) / dfStd;
                        rgNorm[nIdx] = dfNorm;
                    }
                }
            }

            return rgNorm;
        }

        public void TestForward()
        {
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                m_filler.Fill(m_blob_bottom);
                m_blob_top.SetData(0);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, m_blob_bottom.num, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.channels, m_blob_bottom.channels, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.height, m_blob_bottom.height, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.width, m_blob_bottom.width, "The num does not match!");

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                double[] rgExpected = calculateLayerNorm(m_blob_bottom, p.layer_norm_param);

                for (int i = 0; i < rgTop.Count(); i++)
                {
                    double dfActual = rgTop[i];
                    double dfExpected = rgExpected[i];
                    double dfErr = 1e-5;

                    m_log.EXPECT_NEAR(dfActual, dfExpected, dfErr, "The top data does not match the expected data!");
                }
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestForwardInplace()
        {
            Layer<T> layer = null;
            Blob<T> blobInPlace = null;

            try
            {
                blobInPlace = new Blob<T>(m_cuda, m_log, 2, 3, 3, 1);
                BlobCollection<T> colBottom = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                FillerParameter fp = new FillerParameter("gaussian");
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(blobInPlace);

                m_blob_bottom.CopyFrom(blobInPlace, false, true);

                colBottom.Add(blobInPlace);
                colTop.Add(blobInPlace);

                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                layer.Setup(colBottom, colTop);
                layer.Forward(colBottom, colTop);

                double[] rgTop = convert(blobInPlace.update_cpu_data());
                double[] rgExpected = calculateLayerNorm(m_blob_bottom, p.layer_norm_param);
                
                for (int i = 0; i < rgTop.Count(); i++)
                {
                    double dfActual = rgTop[i];
                    double dfExpected = rgExpected[i];
                    double dfErr = 1e-5;

                    m_log.EXPECT_NEAR(dfActual, dfExpected, dfErr, "The top data does not match the expected data!");
                }
            }
            finally
            {
                if (blobInPlace != null)
                   blobInPlace.Dispose();

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradient()
        {
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.batch_norm_param.engine = m_engine;
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-4);

                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
