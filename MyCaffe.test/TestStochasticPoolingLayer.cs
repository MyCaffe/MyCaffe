using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestStochasticPoolingLayer
    {
        #region CAFFE Tests

        [TestMethod]
        public void TestSetup()
        {
            PoolingStochasticLayerTest test = new PoolingStochasticLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingStochasticLayerTest t in test.Tests)
                {
                    t.TestSetup(PoolingParameter.PoolingReshapeAlgorithm.CAFFE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupUseOnnx()
        {
            PoolingStochasticLayerTest test = new PoolingStochasticLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingStochasticLayerTest t in test.Tests)
                {
                    t.TestSetup(PoolingParameter.PoolingReshapeAlgorithm.ONNX);
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
        public void TestStochastic()
        {
            PoolingStochasticLayerTest test = new PoolingStochasticLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingStochasticLayerTest t in test.Tests)
                {
                    t.TestStochastic();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestStochasticTestPhase()
        {
            PoolingStochasticLayerTest test = new PoolingStochasticLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingStochasticLayerTest t in test.Tests)
                {
                    t.TestStochasticTestPhase();
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
        public void TestGradient()
        {
            PoolingStochasticLayerTest test = new PoolingStochasticLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingStochasticLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion
    }


    class PoolingStochasticLayerTest : TestBase
    {
        public PoolingStochasticLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Pooling Stochastic Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PoolingStochasticLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PoolingStochasticLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IPoolingStochasticLayerTest : ITest
    {
        void TestSetup(PoolingParameter.PoolingReshapeAlgorithm alg);
        void TestStochastic();
        void TestStochasticTestPhase();
        void TestGradient();
    }

    class PoolingStochasticLayerTest<T> : TestEx<T>, IPoolingStochasticLayerTest
    {
        public PoolingStochasticLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_cuda.rng_setseed(1701);
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            p.min = 0.1;
            p.max = 1.0;
            return p;
        }

        public void TestSetup(PoolingParameter.PoolingReshapeAlgorithm alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);

            p.pooling_param.reshape_algorithm = alg;
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.STOCHASTIC;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(BottomVec.Count, TopVec.Count, "The top and bottom vecs should have the same count.");
            m_log.CHECK_EQ(Bottom.num, Top.num, "The top and bottom should have the same num.");
            m_log.CHECK_EQ(Bottom.channels, Top.channels, "The top and bottom should have the same channels.");

            if (p.pooling_param.reshape_algorithm == PoolingParameter.PoolingReshapeAlgorithm.ONNX)
                m_log.CHECK_EQ(2, Top.height, "The top height should = 2.");
            else
                m_log.CHECK_EQ(3, Top.height, "The top height should = 3.");

            m_log.CHECK_EQ(2, Top.width, "The top and bottom should have the same width.");
        }

        public void TestStochastic()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.phase = basecode.Phase.TRAIN;
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.STOCHASTIC;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check if the output is correct - it should do random sampling.
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());
            double dfTotal = 0;

            for (int n = 0; n < Top.num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int ph = 0; ph < Top.height; ph++)
                    {
                        for (int pw = 0; pw < Top.width; pw++)
                        {
                            double dfPooled = rgTop[Top.offset(n, c, ph, pw)];
                            dfTotal += dfPooled;
                            int hStart = ph * 2;
                            int hEnd = Math.Min(hStart + 3, Bottom.height);
                            int wStart = pw * 2;
                            int wEnd = Math.Min(wStart + 3, Bottom.width);
                            bool bHasEqual = false;

                            for (int h = hStart; h < hEnd; h++)
                            {
                                for (int w = wStart; w < wEnd; w++)
                                {
                                    int nIdx = Bottom.offset(n, c, h, w);
                                    double dfBottom = rgBottom[nIdx];
                                    float fBottom = (float)dfBottom;
                                    float fPooled = (float)dfPooled;

                                    if (fPooled == fBottom)
                                        bHasEqual = true;
                                }
                            }

                            m_log.CHECK(bHasEqual, "Expected there to be an equal value.");
                        }
                    }
                }
            }

            // When we are doing stochastic pooling, the average we get should be higher
            // than the simple data average since we are weighting more on higher-valued
            // ones.
            m_log.CHECK_GE(dfTotal / Top.count(), 0.55, "The average should be greater than 0.55");
        }

        public void TestStochasticTestPhase()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.engine = m_engine;
            p.phase = basecode.Phase.TEST;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.STOCHASTIC;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check if the output is correct - it should do random sampling.
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());
            double dfTotal = 0;

            for (int n = 0; n < Top.num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int ph = 0; ph < Top.height; ph++)
                    {
                        for (int pw = 0; pw < Top.width; pw++)
                        {
                            double dfPooled = rgTop[Top.offset(n, c, ph, pw)];
                            dfTotal += dfPooled;
                            int hStart = ph * 2;
                            int hEnd = Math.Min(hStart + 3, Bottom.height);
                            int wStart = pw * 2;
                            int wEnd = Math.Min(wStart + 3, Bottom.width);
                            bool bSmallerThanMax = false;

                            for (int h = hStart; h < hEnd; h++)
                            {
                                for (int w = wStart; w < wEnd; w++)
                                {
                                    int nIdx = Bottom.offset(n, c, h, w);
                                    double dfBottom = rgBottom[nIdx];

                                    if (dfPooled <= dfBottom)
                                        bSmallerThanMax = true;
                                }
                            }

                            m_log.CHECK(bSmallerThanMax, "Expected there to be at least one smaller than max.");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.engine = m_engine;
            p.phase = basecode.Phase.TRAIN;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.STOCHASTIC;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
            checker.CheckGradient(layer, BottomVec, TopVec);
        }
    }
}
