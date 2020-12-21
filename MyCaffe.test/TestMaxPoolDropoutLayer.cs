using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMaxPoolDropoutLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            MaxPoolDropoutLayerTest test = new MaxPoolDropoutLayerTest();

            try
            {
                foreach (IMaxPoolDropoutLayerTest t in test.Tests)
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
            MaxPoolDropoutLayerTest test = new MaxPoolDropoutLayerTest();

            try
            {
                foreach (IMaxPoolDropoutLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestBackward()
        {
            MaxPoolDropoutLayerTest test = new MaxPoolDropoutLayerTest();

            try
            {
                foreach (IMaxPoolDropoutLayerTest t in test.Tests)
                {
                    t.TestBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IMaxPoolDropoutLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestBackward();
    }

    class MaxPoolDropoutLayerTest : TestBase
    {
        public MaxPoolDropoutLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ArgMax Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MaxPoolDropoutLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MaxPoolDropoutLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MaxPoolDropoutLayerTest<T> : TestEx<T>, IMaxPoolDropoutLayerTest
    {
        public MaxPoolDropoutLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("constant", 1.0);
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            PoolingLayer<T> max_layer = new PoolingLayer<T>(m_cuda, m_log, p);
            max_layer.Setup(BottomVec, TopVec);
            LayerParameter pd = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            DropoutLayer<T> dropout_layer = new DropoutLayer<T>(m_cuda, m_log, pd);
            dropout_layer.Setup(TopVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "The top and bottom should have the same num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "The top and bottom should have the same channels.");

            if (p.pooling_param.reshape_algorithm == PoolingParameter.PoolingReshapeAlgorithm.ONNX)
                m_log.CHECK_EQ(Top.height, 2, "The top should have height = 2.");
            else
                m_log.CHECK_EQ(Top.height, 3, "The top should have height = 3.");

            m_log.CHECK_EQ(Top.width, 2, "The top should have width = 2.");
        }


        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfSum = convert(Top.asum_data());

            m_log.CHECK_EQ(dfSum, Top.count(), "The sum of values should equal the count.");

            // Dropout in-place.
            LayerParameter pd = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            DropoutLayer<T> dropout_layer = new DropoutLayer<T>(m_cuda, m_log, pd);
            dropout_layer.Setup(TopVec, TopVec);
            dropout_layer.Forward(TopVec, TopVec);

            dfSum = convert(Top.asum_data());
            double dfScale = 1.0 / (1.0 - pd.dropout_param.dropout_ratio);

            m_log.CHECK_GE(dfSum, 0, "The asum should be positive.");
            m_log.CHECK_LE(dfSum, Top.count() * dfScale, "The asum should be less than or equal to the top count * " + dfScale.ToString());
        }

        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.phase = Phase.TRAIN;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Top.SetDiff(1.0);

            List<bool> rgbPropagateDown = Utility.Create<bool>(BottomVec.Count, true);
            layer.Backward(TopVec, rgbPropagateDown, BottomVec);

            double dfSum = convert(Bottom.asum_diff());
 
            m_log.CHECK_EQ(dfSum, Top.count(), "The sum of values should equal the count.");

            // Dropout in-place.
            LayerParameter pd = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            DropoutLayer<T> dropout_layer = new DropoutLayer<T>(m_cuda, m_log, pd);
            dropout_layer.Setup(TopVec, TopVec);
            dropout_layer.Forward(TopVec, TopVec);
            layer.Backward(TopVec, rgbPropagateDown, BottomVec);

            double dfSumWithDropout = convert(Bottom.asum_diff());

            m_log.CHECK_GE(dfSumWithDropout, dfSum, "The sum with dropout should be >= the sum without.");
        }
    }
}
