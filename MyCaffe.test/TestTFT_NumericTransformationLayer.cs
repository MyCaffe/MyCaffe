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
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.tft;

/// <summary>
/// Testing the Numeric Transformation layer.
/// 
/// NumericTransformation Layer - layer converts inputs into embeddings using inner product layers.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_NumericTransformationLayer
    {
        [TestMethod]
        public void TestForward()
        {
            NumericTransformationLayerTest test = new NumericTransformationLayerTest();

            try
            {
                foreach (INumericTransformationLayerTest t in test.Tests)
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
            NumericTransformationLayerTest test = new NumericTransformationLayerTest();

            try
            {
                foreach (INumericTransformationLayerTest t in test.Tests)
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

    interface INumericTransformationLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class NumericTransformationLayerTest : TestBase
    {
        public NumericTransformationLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("NumericTransformation Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NumericTransformationLayerTest<double>(strName, nDeviceID, engine);
            else
                return new NumericTransformationLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class NumericTransformationLayerTest<T> : TestEx<T>, INumericTransformationLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public NumericTransformationLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 5;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;

            try
            {
                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                layer.Setup(BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 5;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;

            try
            {
                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                layer.Setup(BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
