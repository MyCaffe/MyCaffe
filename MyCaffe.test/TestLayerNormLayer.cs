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
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.batch_norm_param.engine = m_engine;
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // TBD
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

            try
            {
                Blob<T> blobInPlace = new Blob<T>(m_cuda, m_log, 5, 2, 3, 4);
                BlobCollection<T> colBottom = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.batch_norm_param.engine = m_engine;
                FillerParameter fp = new FillerParameter("gaussian");
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(blobInPlace);

                colBottom.Add(blobInPlace);
                colTop.Add(blobInPlace);

                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
                layer.Setup(colBottom, colTop);
                layer.Forward(colBottom, colTop);

                // Test Data: TBD

                blobInPlace.Dispose();
            }
            finally
            {
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
