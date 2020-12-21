using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPoolingLayer
    {
        #region CAFFE Tests

        [TestMethod]
        public void TestSetup()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
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
        public void TestSetupPadded()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupPadded(PoolingParameter.PoolingReshapeAlgorithm.CAFFE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupUseOnnxAlg()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetup(PoolingParameter.PoolingReshapeAlgorithm.ONNX);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupUseOnnxAlgPadded()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupPadded(PoolingParameter.PoolingReshapeAlgorithm.ONNX);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupGlobalPooling()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupGlobalPooling();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardMax()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardMax();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardMaxTopMask()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardMaxTopMask();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardMaxPadded()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardMaxPadded();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAve()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardAve();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientMax()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientMax();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientMaxTopMask()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientMaxTopMask();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAve()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientAve();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAvePadded()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);
            
            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientAvePadded();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion

        #region CuDNN Tests

        [TestMethod]
        public void TestSetupCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupCuDNN(PoolingParameter.PoolingReshapeAlgorithm.CAFFE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupPaddedCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupPaddedCuDNN(PoolingParameter.PoolingReshapeAlgorithm.CAFFE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupCuDNNUseOnnxAlg()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupCuDNN(PoolingParameter.PoolingReshapeAlgorithm.ONNX);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupPaddedCuDNNUseOnnxAlg()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestSetupPaddedCuDNN(PoolingParameter.PoolingReshapeAlgorithm.ONNX);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardMaxCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardMaxCuDNN();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientMaxCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientMaxCuDNN();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardMaxPaddedCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardMaxPaddedCuDNN();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAveCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardAveCuDNN();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAveCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientAveCuDNN();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAvePaddedCuDNN()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IPoolingLayerTest t in test.Tests)
                {
                    t.TestGradientAvePaddedCuDNN();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion
    }

    class PoolingLayerTest : TestBase
    {
        public PoolingLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Pooling Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PoolingLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PoolingLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IPoolingLayerTest : ITest
    {
        void TestSetup(PoolingParameter.PoolingReshapeAlgorithm alg);
        void TestSetupPadded(PoolingParameter.PoolingReshapeAlgorithm alg);
        void TestSetupGlobalPooling();
        void TestForwardMax();
        void TestForwardMaxTopMask();
        void TestForwardMaxPadded();
        void TestForwardAve();
        void TestGradientMax();
        void TestGradientMaxTopMask();
        void TestGradientAve();
        void TestGradientAvePadded();
        void TestSetupCuDNN(PoolingParameter.PoolingReshapeAlgorithm alg);
        void TestSetupPaddedCuDNN(PoolingParameter.PoolingReshapeAlgorithm alg);
        void TestForwardMaxCuDNN();
        void TestGradientMaxCuDNN();
        void TestForwardMaxPaddedCuDNN();
        void TestForwardAveCuDNN();
        void TestGradientAveCuDNN();
        void TestGradientAvePaddedCuDNN();
    }

    class PoolingLayerTest<T> : TestEx<T>, IPoolingLayerTest
    {
        Blob<T> Top_mask;

        public PoolingLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_cuda.rng_setseed(1701);
            Top_mask = new Blob<T>(m_cuda, m_log);
            m_engine = engine;
        }

        protected override void dispose()
        {
            Top_mask.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("gaussian");
            p.value = 1.0;
            return p;
        }

        public Blob<T> TopMask
        {
            get { return Top_mask; }
        }

        public void TestForwardSquare()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.kernel_size.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            p.pooling_param.engine = m_engine;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 3, 5);
            // Input: 2x2 channels of:
            //  [1 2 5 2 3]
            //  [9 4 1 4 8]
            //  [1 2 5 2 3]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            for (int i = 0; i < 15 * nNum * nChannels; i += 15)
            {
                rgBottom[i + 0] = 1;
                rgBottom[i + 1] = 2;
                rgBottom[i + 2] = 5;
                rgBottom[i + 3] = 2;
                rgBottom[i + 4] = 3;
                rgBottom[i + 5] = 9;
                rgBottom[i + 6] = 4;
                rgBottom[i + 7] = 1;
                rgBottom[i + 8] = 4;
                rgBottom[i + 9] = 8;
                rgBottom[i + 10] = 1;
                rgBottom[i + 11] = 2;
                rgBottom[i + 12] = 5;
                rgBottom[i + 13] = 2;
                rgBottom[i + 14] = 3;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(2, Top.height, "The top height should equal 2.");
            m_log.CHECK_EQ(4, Top.width, "The top width should equal 4.");

            if (TopVec.Count > 1)
            {
                m_log.CHECK_EQ(nNum, Top_mask.num, "The top mask num should equal " + nNum.ToString());
                m_log.CHECK_EQ(nChannels, Top_mask.channels, "The top mask channels should equal " + nChannels.ToString());
                m_log.CHECK_EQ(2, Top_mask.height, "The top mask height should equal 2.");
                m_log.CHECK_EQ(4, Top_mask.width, "The top mask width should equal 4.");
            }

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:
            //  [9 5 5 8]
            //  [9 5 5 8]
            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 8 * nNum * nChannels; i += 8)
            {
                m_log.CHECK_EQ(rgTop[i + 0], 9, "The top element at " + (i + 0).ToString() + " should be 9");
                m_log.CHECK_EQ(rgTop[i + 1], 5, "The top element at " + (i + 1).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 2], 5, "The top element at " + (i + 2).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 3], 8, "The top element at " + (i + 3).ToString() + " should be 8");
                m_log.CHECK_EQ(rgTop[i + 4], 9, "The top element at " + (i + 4).ToString() + " should be 9");
                m_log.CHECK_EQ(rgTop[i + 5], 5, "The top element at " + (i + 5).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 6], 5, "The top element at " + (i + 6).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 7], 8, "The top element at " + (i + 7).ToString() + " should be 8");
            }

            if (TopVec.Count > 1)
            {
                // Expected output: 2x2 channels of:
                //  [5  2  2 9]
                //  [5 12 12 9]
                double[] rgMask = convert(Top_mask.update_cpu_data());
                for (int i = 0; i < 8 * nNum * nChannels; i += 8)
                {
                    m_log.CHECK_EQ(rgMask[i + 0], 5, "The mask element at " + (i + 0).ToString() + " should be 5");
                    m_log.CHECK_EQ(rgMask[i + 1], 2, "The mask element at " + (i + 1).ToString() + " should be 2");
                    m_log.CHECK_EQ(rgMask[i + 2], 2, "The mask element at " + (i + 2).ToString() + " should be 2");
                    m_log.CHECK_EQ(rgMask[i + 3], 9, "The mask element at " + (i + 3).ToString() + " should be 9");
                    m_log.CHECK_EQ(rgMask[i + 4], 5, "The mask element at " + (i + 4).ToString() + " should be 5");
                    m_log.CHECK_EQ(rgMask[i + 5], 12, "The mask element at " + (i + 5).ToString() + " should be 12");
                    m_log.CHECK_EQ(rgMask[i + 6], 12, "The mask element at " + (i + 6).ToString() + " should be 12");
                    m_log.CHECK_EQ(rgMask[i + 7], 9, "The mask element at " + (i + 7).ToString() + " should be 9");
                }
            }
        }

        public void TestForwardRectHigh()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.kernel_h = 3;
            p.pooling_param.kernel_w = 2;
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            p.pooling_param.engine = m_engine;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 6, 6);
            // Input: 2x2 channels of:
            //  [35  1  6 26 19 24]
            //  [ 3 32  7 21 23 25] 
            //  [31  9  2 22 27 20]
            //  [ 8 28 33 17 10 15]
            //  [30  5 34 12 14 16]
            //  [ 4 36 29 13 18 11]
            // (this is generated by magic(6) in MATLAB)
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            for (int i = 0; i < 36 * nNum * nChannels; i += 36)
            {
                rgBottom[i + 0] = 35;
                rgBottom[i + 1] = 1;
                rgBottom[i + 2] = 6;
                rgBottom[i + 3] = 26;
                rgBottom[i + 4] = 19;
                rgBottom[i + 5] = 24;
                rgBottom[i + 6] = 3;
                rgBottom[i + 7] = 32;
                rgBottom[i + 8] = 7;
                rgBottom[i + 9] = 21;
                rgBottom[i + 10] = 23;
                rgBottom[i + 11] = 25;
                rgBottom[i + 12] = 31;
                rgBottom[i + 13] = 9;
                rgBottom[i + 14] = 2;
                rgBottom[i + 15] = 22;
                rgBottom[i + 16] = 27;
                rgBottom[i + 17] = 20;
                rgBottom[i + 18] = 8;
                rgBottom[i + 19] = 28;
                rgBottom[i + 20] = 33;
                rgBottom[i + 21] = 17;
                rgBottom[i + 22] = 10;
                rgBottom[i + 23] = 15;
                rgBottom[i + 24] = 30;
                rgBottom[i + 25] = 5;
                rgBottom[i + 26] = 34;
                rgBottom[i + 27] = 12;
                rgBottom[i + 28] = 14;
                rgBottom[i + 29] = 16;
                rgBottom[i + 30] = 4;
                rgBottom[i + 31] = 36;
                rgBottom[i + 32] = 29;
                rgBottom[i + 33] = 13;
                rgBottom[i + 34] = 18;
                rgBottom[i + 35] = 11;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(4, Top.height, "The top height should equal 4.");
            m_log.CHECK_EQ(5, Top.width, "The top width should equal 5.");

            if (TopVec.Count > 1)
            {
                m_log.CHECK_EQ(nNum, Top_mask.num, "The top mask num should equal " + nNum.ToString());
                m_log.CHECK_EQ(nChannels, Top_mask.channels, "The top mask channels should equal " + nChannels.ToString());
                m_log.CHECK_EQ(4, Top_mask.height, "The top mask height should equal 4.");
                m_log.CHECK_EQ(5, Top_mask.width, "The top mask width should equal 5.");
            }

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:
            //  [35 32 26 27 27]
            //  [32 33 33 27 27]
            //  [31 34 34 27 27]
            //  [36 36 34 18 18]
            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 20 * nNum * nChannels; i += 20)
            {
                m_log.CHECK_EQ(rgTop[i + 0], 35, "The top element at " + (i + 0).ToString() + " should be 35");
                m_log.CHECK_EQ(rgTop[i + 1], 32, "The top element at " + (i + 1).ToString() + " should be 32");
                m_log.CHECK_EQ(rgTop[i + 2], 26, "The top element at " + (i + 2).ToString() + " should be 26");
                m_log.CHECK_EQ(rgTop[i + 3], 27, "The top element at " + (i + 3).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 4], 27, "The top element at " + (i + 4).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 5], 32, "The top element at " + (i + 5).ToString() + " should be 32");
                m_log.CHECK_EQ(rgTop[i + 6], 33, "The top element at " + (i + 6).ToString() + " should be 33");
                m_log.CHECK_EQ(rgTop[i + 7], 33, "The top element at " + (i + 7).ToString() + " should be 33");
                m_log.CHECK_EQ(rgTop[i + 8], 27, "The top element at " + (i + 8).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 9], 27, "The top element at " + (i + 9).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 10], 31, "The top element at " + (i + 10).ToString() + " should be 31");
                m_log.CHECK_EQ(rgTop[i + 11], 34, "The top element at " + (i + 11).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 12], 34, "The top element at " + (i + 12).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 13], 27, "The top element at " + (i + 13).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 14], 27, "The top element at " + (i + 14).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 15], 36, "The top element at " + (i + 15).ToString() + " should be 36");
                m_log.CHECK_EQ(rgTop[i + 16], 36, "The top element at " + (i + 16).ToString() + " should be 36");
                m_log.CHECK_EQ(rgTop[i + 17], 34, "The top element at " + (i + 17).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 18], 18, "The top element at " + (i + 18).ToString() + " should be 18");
                m_log.CHECK_EQ(rgTop[i + 19], 18, "The top element at " + (i + 19).ToString() + " should be 18");
            }

            if (TopVec.Count > 1)
            {
                // Expected output: 2x2 channels of:
                //  [ 1  8  4 17 17]
                //  [ 8 21 21 17 17]
                //  [13 27 27 17 17]
                //  [32 32 27 35 35]
                double[] rgMask = convert(Top_mask.update_cpu_data());
                for (int i = 0; i < 20 * nNum * nChannels; i += 20)
                {
                    m_log.CHECK_EQ(rgMask[i + 0], 0, "The top element at " + (i + 0).ToString() + " should be 0");
                    m_log.CHECK_EQ(rgMask[i + 1], 7, "The top element at " + (i + 1).ToString() + " should be 7");
                    m_log.CHECK_EQ(rgMask[i + 2], 3, "The top element at " + (i + 2).ToString() + " should be 3");
                    m_log.CHECK_EQ(rgMask[i + 3], 16, "The top element at " + (i + 3).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 4], 16, "The top element at " + (i + 4).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 5], 7, "The top element at " + (i + 5).ToString() + " should be 7");
                    m_log.CHECK_EQ(rgMask[i + 6], 20, "The top element at " + (i + 6).ToString() + " should be 20");
                    m_log.CHECK_EQ(rgMask[i + 7], 20, "The top element at " + (i + 7).ToString() + " should be 20");
                    m_log.CHECK_EQ(rgMask[i + 8], 16, "The top element at " + (i + 8).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 9], 16, "The top element at " + (i + 9).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 10], 12, "The top element at " + (i + 10).ToString() + " should be 12");
                    m_log.CHECK_EQ(rgMask[i + 11], 26, "The top element at " + (i + 11).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 12], 26, "The top element at " + (i + 12).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 13], 16, "The top element at " + (i + 13).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 14], 16, "The top element at " + (i + 14).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 15], 31, "The top element at " + (i + 15).ToString() + " should be 31");
                    m_log.CHECK_EQ(rgMask[i + 16], 31, "The top element at " + (i + 16).ToString() + " should be 31");
                    m_log.CHECK_EQ(rgMask[i + 17], 26, "The top element at " + (i + 17).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 18], 34, "The top element at " + (i + 18).ToString() + " should be 34");
                    m_log.CHECK_EQ(rgMask[i + 19], 34, "The top element at " + (i + 19).ToString() + " should be 34");
                }
            }
        }

        public void TestForwardRectWide()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.kernel_h = 2;
            p.pooling_param.kernel_w = 3;
            p.pooling_param.engine = m_engine;
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 6, 6);
            // Input: 2x2 channels of:
            //  [35  1  6 26 19 24]
            //  [ 3 32  7 21 23 25] 
            //  [31  9  2 22 27 20]
            //  [ 8 28 33 17 10 15]
            //  [30  5 34 12 14 16]
            //  [ 4 36 29 13 18 11]
            // (this is generated by magic(6) in MATLAB)
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            for (int i = 0; i < 36 * nNum * nChannels; i += 36)
            {
                rgBottom[i + 0] = 35;
                rgBottom[i + 1] = 1;
                rgBottom[i + 2] = 6;
                rgBottom[i + 3] = 26;
                rgBottom[i + 4] = 19;
                rgBottom[i + 5] = 24;
                rgBottom[i + 6] = 3;
                rgBottom[i + 7] = 32;
                rgBottom[i + 8] = 7;
                rgBottom[i + 9] = 21;
                rgBottom[i + 10] = 23;
                rgBottom[i + 11] = 25;
                rgBottom[i + 12] = 31;
                rgBottom[i + 13] = 9;
                rgBottom[i + 14] = 2;
                rgBottom[i + 15] = 22;
                rgBottom[i + 16] = 27;
                rgBottom[i + 17] = 20;
                rgBottom[i + 18] = 8;
                rgBottom[i + 19] = 28;
                rgBottom[i + 20] = 33;
                rgBottom[i + 21] = 17;
                rgBottom[i + 22] = 10;
                rgBottom[i + 23] = 15;
                rgBottom[i + 24] = 30;
                rgBottom[i + 25] = 5;
                rgBottom[i + 26] = 34;
                rgBottom[i + 27] = 12;
                rgBottom[i + 28] = 14;
                rgBottom[i + 29] = 16;
                rgBottom[i + 30] = 4;
                rgBottom[i + 31] = 36;
                rgBottom[i + 32] = 29;
                rgBottom[i + 33] = 13;
                rgBottom[i + 34] = 18;
                rgBottom[i + 35] = 11;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(5, Top.height, "The top height should equal 4.");
            m_log.CHECK_EQ(4, Top.width, "The top width should equal 5.");

            if (TopVec.Count > 1)
            {
                m_log.CHECK_EQ(nNum, Top_mask.num, "The top mask num should equal " + nNum.ToString());
                m_log.CHECK_EQ(nChannels, Top_mask.channels, "The top mask channels should equal " + nChannels.ToString());
                m_log.CHECK_EQ(5, Top_mask.height, "The top mask height should equal 4.");
                m_log.CHECK_EQ(4, Top_mask.width, "The top mask width should equal 5.");
            }

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:
            //  [35 32 26 26]
            //  [32 32 27 27]
            //  [33 33 33 27]
            //  [34 34 34 17]
            //  [36 36 34 18]
            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 20 * nNum * nChannels; i += 20)
            {
                m_log.CHECK_EQ(rgTop[i + 0], 35, "The top element at " + (i + 0).ToString() + " should be 35");
                m_log.CHECK_EQ(rgTop[i + 1], 32, "The top element at " + (i + 1).ToString() + " should be 32");
                m_log.CHECK_EQ(rgTop[i + 2], 26, "The top element at " + (i + 2).ToString() + " should be 26");
                m_log.CHECK_EQ(rgTop[i + 3], 26, "The top element at " + (i + 3).ToString() + " should be 26");
                m_log.CHECK_EQ(rgTop[i + 4], 32, "The top element at " + (i + 4).ToString() + " should be 32");
                m_log.CHECK_EQ(rgTop[i + 5], 32, "The top element at " + (i + 5).ToString() + " should be 32");
                m_log.CHECK_EQ(rgTop[i + 6], 27, "The top element at " + (i + 6).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 7], 27, "The top element at " + (i + 7).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 8], 33, "The top element at " + (i + 8).ToString() + " should be 33");
                m_log.CHECK_EQ(rgTop[i + 9], 33, "The top element at " + (i + 9).ToString() + " should be 33");
                m_log.CHECK_EQ(rgTop[i + 10], 33, "The top element at " + (i + 10).ToString() + " should be 33");
                m_log.CHECK_EQ(rgTop[i + 11], 27, "The top element at " + (i + 11).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 12], 34, "The top element at " + (i + 12).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 13], 34, "The top element at " + (i + 13).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 14], 34, "The top element at " + (i + 14).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 15], 17, "The top element at " + (i + 15).ToString() + " should be 17");
                m_log.CHECK_EQ(rgTop[i + 16], 36, "The top element at " + (i + 16).ToString() + " should be 36");
                m_log.CHECK_EQ(rgTop[i + 17], 36, "The top element at " + (i + 17).ToString() + " should be 36");
                m_log.CHECK_EQ(rgTop[i + 18], 34, "The top element at " + (i + 18).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 19], 18, "The top element at " + (i + 19).ToString() + " should be 18");
            }

            if (TopVec.Count > 1)
            {
                // Expected output: 2x2 channels of:
                //  [ 1  8  4  4]
                //  [ 8  8 17 17]
                //  [21 21 21 17]
                //  [27 27 27 22]
                //  [32 32 27 35]
                double[] rgMask = convert(Top_mask.update_cpu_data());
                for (int i = 0; i < 20 * nNum * nChannels; i += 20)
                {
                    m_log.CHECK_EQ(rgMask[i + 0], 0, "The top element at " + (i + 0).ToString() + " should be 0");
                    m_log.CHECK_EQ(rgMask[i + 1], 7, "The top element at " + (i + 1).ToString() + " should be 7");
                    m_log.CHECK_EQ(rgMask[i + 2], 3, "The top element at " + (i + 2).ToString() + " should be 3");
                    m_log.CHECK_EQ(rgMask[i + 3], 3, "The top element at " + (i + 3).ToString() + " should be 3");
                    m_log.CHECK_EQ(rgMask[i + 4], 7, "The top element at " + (i + 4).ToString() + " should be 7");
                    m_log.CHECK_EQ(rgMask[i + 5], 7, "The top element at " + (i + 5).ToString() + " should be 7");
                    m_log.CHECK_EQ(rgMask[i + 6], 16, "The top element at " + (i + 6).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 7], 16, "The top element at " + (i + 7).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 8], 20, "The top element at " + (i + 8).ToString() + " should be 20");
                    m_log.CHECK_EQ(rgMask[i + 9], 20, "The top element at " + (i + 9).ToString() + " should be 20");
                    m_log.CHECK_EQ(rgMask[i + 10], 20, "The top element at " + (i + 10).ToString() + " should be 20");
                    m_log.CHECK_EQ(rgMask[i + 11], 16, "The top element at " + (i + 11).ToString() + " should be 16");
                    m_log.CHECK_EQ(rgMask[i + 12], 26, "The top element at " + (i + 12).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 13], 26, "The top element at " + (i + 13).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 14], 26, "The top element at " + (i + 14).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 15], 21, "The top element at " + (i + 15).ToString() + " should be 21");
                    m_log.CHECK_EQ(rgMask[i + 16], 31, "The top element at " + (i + 16).ToString() + " should be 31");
                    m_log.CHECK_EQ(rgMask[i + 17], 31, "The top element at " + (i + 17).ToString() + " should be 31");
                    m_log.CHECK_EQ(rgMask[i + 18], 26, "The top element at " + (i + 18).ToString() + " should be 26");
                    m_log.CHECK_EQ(rgMask[i + 19], 34, "The top element at " + (i + 19).ToString() + " should be 34");
                }
            }
        }

        public void TestSetup(PoolingParameter.PoolingReshapeAlgorithm alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);

            p.pooling_param.reshape_algorithm = alg;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.engine = m_engine;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "Top and bottom should have the same num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "Top and bottom should have the same channels.");

            if (p.pooling_param.reshape_algorithm == PoolingParameter.PoolingReshapeAlgorithm.ONNX)
                m_log.CHECK_EQ(2, Top.height, "The top height should be 2.");
            else
                m_log.CHECK_EQ(3, Top.height, "The top height should be 3.");

            m_log.CHECK_EQ(2, Top.width, "The top width should be 2.");
        }

        public void TestSetupPadded(PoolingParameter.PoolingReshapeAlgorithm alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);

            p.pooling_param.reshape_algorithm = alg;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pad.Add(1);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
            p.pooling_param.engine = m_engine;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "Top and bottom should have the same num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "Top and bottom should have the same channels.");

            if (p.pooling_param.reshape_algorithm == PoolingParameter.PoolingReshapeAlgorithm.ONNX)
                m_log.CHECK_EQ(3, Top.height, "The top height should be 3.");
            else
                m_log.CHECK_EQ(4, Top.height, "The top height should be 4.");

            m_log.CHECK_EQ(3, Top.width, "The top width should be 2.");
        }

        public void TestSetupGlobalPooling()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.global_pooling = true;
            p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
            p.pooling_param.engine = m_engine;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "Top and bottom should have the same num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "Top and bottom should have the same channels.");
            m_log.CHECK_EQ(1, Top.height, "The top height should be 1.");
            m_log.CHECK_EQ(1, Top.width, "The top width should be 1.");
        }

        public void TestForwardMax()
        {
            TestForwardSquare();
            TestForwardRectHigh();
            TestForwardRectWide();
        }

        public void TestForwardMaxTopMask()
        {
            TopVec.Add(TopMask);
            TestForwardSquare();
            TestForwardRectHigh();
            TestForwardRectWide();
        }

        public void TestForwardMaxPadded()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pad.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            Bottom.Reshape(1, 1, 3, 3);

            // Input:
            // [1 2 4]
            // [2 3 2]
            // [4 2 1]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            rgBottom[0] = 1;
            rgBottom[1] = 2;
            rgBottom[2] = 4;
            rgBottom[3] = 2;
            rgBottom[4] = 3;
            rgBottom[5] = 2;
            rgBottom[6] = 4;
            rgBottom[7] = 2;
            rgBottom[8] = 1;
            Bottom.mutable_cpu_data = convert(rgBottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(1, Top.num, "The top num should = 1");
            m_log.CHECK_EQ(1, Top.channels, "The top channels should = 1");
            m_log.CHECK_EQ(3, Top.height, "The top height should = 3");
            m_log.CHECK_EQ(3, Top.width, "The top width should = 3");

            layer.Forward(BottomVec, TopVec);

            double dfEpsilon = 1e-8;
            // Output:
            // [1 4 4]
            // [4 4 4]
            // [4 4 1]
            double[] rgTop = convert(Top.update_cpu_data());
            m_log.EXPECT_NEAR(rgTop[0], 1, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[1], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[2], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[3], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[4], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[5], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[6], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[7], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[8], 1, dfEpsilon);
        }

        public void TestForwardAve()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(1);
            p.pooling_param.pad.Add(1);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;

            Bottom.Reshape(1, 1, 3, 3);
            FillerParameter fp = new FillerParameter("constant");
            fp.value = 2;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(1, Top.num, "Top num should be 1");
            m_log.CHECK_EQ(1, Top.channels, "Top channels should be 1");
            m_log.CHECK_EQ(3, Top.height, "Top height should be 3");
            m_log.CHECK_EQ(3, Top.width, "Top width should be 3");

            layer.Forward(BottomVec, TopVec);

            double dfEpsilon = 1e-5;
            double[] rgTop = convert(Top.update_cpu_data());

            m_log.EXPECT_NEAR(rgTop[0], 8.0 / 9, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[1], 4.0 / 3, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[2], 8.0 / 9, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[3], 4.0 / 3, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[4], 2.0, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[5], 4.0 / 3, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[6], 8.0 / 9, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[7], 4.0 / 3, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[8], 8.0 / 9, dfEpsilon);
        }

        public void TestGradientMax()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    p.pooling_param.pad.Add(1);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
                }
            }
        }

        public void TestGradientMaxTopMask()
        {
            PoolingLayerTest test = new PoolingLayerTest(EngineParameter.Engine.CAFFE);

            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    TopVec.Add(TopMask);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);

                    TopVec.RemoveAt(TopVec.Count - 1);
                }
            }
        }

        public void TestGradientAve()
        {
            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
                }
            }
        }

        public void TestGradientAvePadded()
        {
            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    p.pooling_param.pad.Add(2);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
                }
            }
        }

        public void TestSetupCuDNN(PoolingParameter.PoolingReshapeAlgorithm alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);

            p.pooling_param.reshape_algorithm = alg;
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "Top and bottom should have the same num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "Top and bottom should have the same channels.");

            if (p.pooling_param.reshape_algorithm == PoolingParameter.PoolingReshapeAlgorithm.ONNX)
                m_log.CHECK_EQ(2, Top.height, "The top height should be 2.");
            else
                m_log.CHECK_EQ(3, Top.height, "The top height should be 3.");

            m_log.CHECK_EQ(2, Top.width, "The top width should be 2.");
        }

        public void TestSetupPaddedCuDNN(PoolingParameter.PoolingReshapeAlgorithm alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);

            p.pooling_param.reshape_algorithm = alg;
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pad.Add(1);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "Top and bottom should have the same num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "Top and bottom should have the same channels.");

            if (p.pooling_param.reshape_algorithm == PoolingParameter.PoolingReshapeAlgorithm.ONNX)
                m_log.CHECK_EQ(3, Top.height, "The top height should be 3.");
            else
                m_log.CHECK_EQ(4, Top.height, "The top height should be 4.");

            m_log.CHECK_EQ(3, Top.width, "The top width should be 3.");
        }

        public void TestForwardMaxCuDNN()
        {
            TestForwardSquare();
            TestForwardRectHigh();
            TestForwardRectWide();
        }

        public void TestGradientMaxCuDNN()
        {
            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    // currently, cuDNN pooling does not support padding.
                    p.pooling_param.pad.Add(0);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
                }
            }
        }

        public void TestForwardMaxPaddedCuDNN()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pad.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            Bottom.Reshape(1, 1, 3, 3);

            // Input:
            // [1 2 4]
            // [2 3 2]
            // [4 2 1]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            rgBottom[0] = 1;
            rgBottom[1] = 2;
            rgBottom[2] = 4;
            rgBottom[3] = 2;
            rgBottom[4] = 3;
            rgBottom[5] = 2;
            rgBottom[6] = 4;
            rgBottom[7] = 2;
            rgBottom[8] = 1;
            Bottom.mutable_cpu_data = convert(rgBottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(1, Top.num, "The top num should = 1");
            m_log.CHECK_EQ(1, Top.channels, "The top channels should = 1");
            m_log.CHECK_EQ(3, Top.height, "The top height should = 3");
            m_log.CHECK_EQ(3, Top.width, "The top width should = 3");

            layer.Forward(BottomVec, TopVec);

            double dfEpsilon = 1e-8;
            // Output:
            // [1 4 4]
            // [4 4 4]
            // [4 4 1]
            double[] rgTop = convert(Top.update_cpu_data());
            m_log.EXPECT_NEAR(rgTop[0], 1, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[1], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[2], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[3], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[4], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[5], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[6], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[7], 4, dfEpsilon);
            m_log.EXPECT_NEAR(rgTop[8], 1, dfEpsilon);
        }

        public void TestForwardAveCuDNN()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
            p.pooling_param.engine = m_engine;
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(1);
            // currently, cuDNN does not support padding so we use
            // a simplified version of this 
            p.pooling_param.pad.Add(0);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;

            Bottom.Reshape(1, 1, 3, 3);
            FillerParameter fp = new FillerParameter("constant");
            fp.value = 2;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(1, Top.num, "Top num should be 1");
            m_log.CHECK_EQ(1, Top.channels, "Top channels should be 1");
            m_log.CHECK_EQ(1, Top.height, "Top height should be 3");
            m_log.CHECK_EQ(1, Top.width, "Top width should be 3");

            layer.Forward(BottomVec, TopVec);

            double dfEpsilon = 1e-5;
            double[] rgTop = convert(Top.update_cpu_data());

            m_log.EXPECT_NEAR(rgTop[0], 2.0, dfEpsilon);
        }

        public void TestGradientAveCuDNN()
        {
            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
                }
            }
        }

        public void TestGradientAvePaddedCuDNN()
        {
            for (int kernel_h = 3; kernel_h <= 4; kernel_h++)
            {
                for (int kernel_w = 3; kernel_w <= 4; kernel_w++)
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.POOLING);
                    p.pooling_param.engine = m_engine;
                    p.pooling_param.kernel_h = (uint)kernel_h;
                    p.pooling_param.kernel_w = (uint)kernel_w;
                    p.pooling_param.stride.Add(2);
                    p.pooling_param.pad.Add(0);
                    p.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
                    PoolingLayer<T> layer = new PoolingLayer<T>(m_cuda, m_log, p);

                    GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                    checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
                }
            }
        }
    }
}
