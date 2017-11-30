using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.layers.alpha;

namespace MyCaffe.test
{
    [TestClass]
    public class TestUnPoolingLayer
    {
        #region CAFFE Tests UnPooling1

        [TestMethod]
        public void TestForwardSquare1()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardSquare(LayerParameter.LayerType.UNPOOLING1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRectHigh1()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardRectHigh(LayerParameter.LayerType.UNPOOLING1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRectWithPad1()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardRectWithPad(LayerParameter.LayerType.UNPOOLING1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardSquareWithPad1()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardSquareWithPad(LayerParameter.LayerType.UNPOOLING1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion

        #region CAFFE Tests UnPooling2

        [TestMethod]
        public void TestForwardSquare2()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardSquare(LayerParameter.LayerType.UNPOOLING2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRectHigh2()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardRectHigh(LayerParameter.LayerType.UNPOOLING2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRectWithPad2()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardRectWithPad(LayerParameter.LayerType.UNPOOLING2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardSquareWithPad2()
        {
            UnPoolingLayerTest test = new UnPoolingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IUnPoolingLayerTest t in test.Tests)
                {
                    t.TestForwardSquareWithPad(LayerParameter.LayerType.UNPOOLING2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion
    }

    class UnPoolingLayerTest : TestBase
    {
        public UnPoolingLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("UnPooling Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new UnPoolingLayerTest<double>(strName, nDeviceID, engine);
            else
                return new UnPoolingLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IUnPoolingLayerTest : ITest
    {
        void TestForwardSquare(LayerParameter.LayerType type);
        void TestForwardRectHigh(LayerParameter.LayerType type);
        void TestForwardRectWithPad(LayerParameter.LayerType type);
        void TestForwardSquareWithPad(LayerParameter.LayerType type);
    }

    class UnPoolingLayerTest<T> : TestEx<T>, IUnPoolingLayerTest
    {
        Blob<T> m_blob_bottom_mask;

        public UnPoolingLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 2, 2, 4 }, nDeviceID)
        {
            m_cuda.rng_setseed(1701);
            m_blob_bottom_mask = new Blob<T>(m_cuda, m_log, 2, 2, 2, 4);
            m_engine = engine;

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom_mask);

            BottomVec.Add(m_blob_bottom_mask);
        }

        protected override void dispose()
        {
            m_blob_bottom_mask.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("gaussian");
            p.value = 1.0;
            return p;
        }

        public Blob<T> BottomMask
        {
            get { return m_blob_bottom_mask; }
        }

        /// <summary>
        /// Test for 2x2 square unpooling layer
        /// </summary>
        public void TestForwardSquare(LayerParameter.LayerType type)
        {
            LayerParameter p = new LayerParameter(type);
            p.pooling_param.kernel_size.Add(2);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            p.pooling_param.engine = m_engine;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 2, 4);
            BottomMask.Reshape(nNum, nChannels, 2, 4);
            // Input: 2x2 channels of:
            //  [9 5 5 8]
            //  [9 5 5 8]
            // Mask: 2x2 channels of:
            //  [5  2  2  9]
            //  [5 12 12  9]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            double[] rgBottomMask = convert(BottomMask.mutable_cpu_data);

            for (int i = 0; i < 8 * nNum * nChannels; i += 8)
            {
                rgBottom[i + 0] = 9;
                rgBottom[i + 1] = 5;
                rgBottom[i + 2] = 5;
                rgBottom[i + 3] = 8;
                rgBottom[i + 4] = 9;
                rgBottom[i + 5] = 5;
                rgBottom[i + 6] = 5;
                rgBottom[i + 7] = 8;
                // And the mask
                rgBottomMask[i + 0] = 5;
                rgBottomMask[i + 1] = 2;
                rgBottomMask[i + 2] = 2;
                rgBottomMask[i + 3] = 9;
                rgBottomMask[i + 4] = 5;
                rgBottomMask[i + 5] = 12;
                rgBottomMask[i + 6] = 12;
                rgBottomMask[i + 7] = 9;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);
            BottomMask.mutable_cpu_data = convert(rgBottomMask);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new basecode.CancelEvent());
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(3, Top.height, "The top height should equal 3.");
            m_log.CHECK_EQ(5, Top.width, "The top width should equal 5.");

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:
            //  [0 0 5 0 0]
            //  [9 0 0 0 8]
            //  [0 0 5 0 0]
            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 15 * nNum * nChannels; i += 15)
            {
                m_log.CHECK_EQ(rgTop[i +  0], 0, "The top element at " + (i +  0).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  1], 0, "The top element at " + (i +  1).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  2], 5, "The top element at " + (i +  2).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i +  3], 0, "The top element at " + (i +  3).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  4], 0, "The top element at " + (i +  4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  5], 9, "The top element at " + (i +  5).ToString() + " should be 9");
                m_log.CHECK_EQ(rgTop[i +  6], 0, "The top element at " + (i +  6).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  7], 0, "The top element at " + (i +  7).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  8], 0, "The top element at " + (i +  8).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i +  9], 8, "The top element at " + (i +  9).ToString() + " should be 8");
                m_log.CHECK_EQ(rgTop[i + 10], 0, "The top element at " + (i + 10).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 11], 0, "The top element at " + (i + 11).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 12], 5, "The top element at " + (i + 12).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 13], 0, "The top element at " + (i + 13).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 14], 0, "The top element at " + (i + 14).ToString() + " should be 0");
            }
        }

        public void TestForwardRectHigh(LayerParameter.LayerType type)
        {
            LayerParameter p = new LayerParameter(type);
            p.pooling_param.kernel_h = 3;
            p.pooling_param.kernel_w = 2;
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            p.pooling_param.engine = m_engine;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 4, 5);
            BottomMask.Reshape(nNum, nChannels, 4, 5);
            // Input: 2x2 channels of:
            //  [35 32 26 27 27]
            //  [32 33 33 27 27]
            //  [31 34 34 27 27]
            //  [32 32 27 18 18]
            //
            // Mask: 2x2 channels of:
            //  [  1  8  4 17 17]
            //  [  8 21 21 17 17]
            //  [ 13 27 27 17 17]
            //  [ 32 32 27 35 35]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            double[] rgBottomMask = convert(BottomMask.mutable_cpu_data);

            for (int i = 0; i < 20 * nNum * nChannels; i += 20)
            {
                rgBottom[i + 0] = 35;
                rgBottom[i + 1] = 32;
                rgBottom[i + 2] = 26;
                rgBottom[i + 3] = 27;
                rgBottom[i + 4] = 27;
                rgBottom[i + 5] = 32;
                rgBottom[i + 6] = 32;
                rgBottom[i + 7] = 33;
                rgBottom[i + 8] = 27;
                rgBottom[i + 9] = 27;
                rgBottom[i + 10] = 31;
                rgBottom[i + 11] = 34;
                rgBottom[i + 12] = 34;
                rgBottom[i + 13] = 27;
                rgBottom[i + 14] = 27;
                rgBottom[i + 15] = 36;
                rgBottom[i + 16] = 36;
                rgBottom[i + 17] = 34;
                rgBottom[i + 18] = 18;
                rgBottom[i + 19] = 18;

                // For the mask

                rgBottomMask[i + 0] = 0;
                rgBottomMask[i + 1] = 7;
                rgBottomMask[i + 2] = 3;
                rgBottomMask[i + 3] = 16;
                rgBottomMask[i + 4] = 16;
                rgBottomMask[i + 5] = 7;
                rgBottomMask[i + 6] = 20;
                rgBottomMask[i + 7] = 20;
                rgBottomMask[i + 8] = 16;
                rgBottomMask[i + 9] = 16;
                rgBottomMask[i + 10] = 12;
                rgBottomMask[i + 11] = 26;
                rgBottomMask[i + 12] = 26;
                rgBottomMask[i + 13] = 16;
                rgBottomMask[i + 14] = 16;
                rgBottomMask[i + 15] = 31;
                rgBottomMask[i + 16] = 31;
                rgBottomMask[i + 17] = 26;
                rgBottomMask[i + 18] = 34;
                rgBottomMask[i + 19] = 34;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);
            BottomMask.mutable_cpu_data = convert(rgBottomMask);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new basecode.CancelEvent());
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(6, Top.height, "The top height should equal 4.");
            m_log.CHECK_EQ(6, Top.width, "The top width should equal 5.");

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:
            //  [35  0  0 26  0  0]
            //  [ 0 32  0  0  0  0]
            //  [31  0  0  0 27  0]
            //  [ 0  0 33  0  0  0]
            //  [ 0  0 34  0  0  0]
            //  [ 0 36  0  0 18  0]
            // (this is generated by magic(6) in MATLAB)

            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 36 * nNum * nChannels; i += 36)
            {
                m_log.CHECK_EQ(rgTop[i + 0], 35, "The top element at " + (i + 0).ToString() + " should be 35");
                m_log.CHECK_EQ(rgTop[i + 1],  0, "The top element at " + (i + 1).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 2],  0, "The top element at " + (i + 2).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 3], 26, "The top element at " + (i + 3).ToString() + " should be 26");
                m_log.CHECK_EQ(rgTop[i + 4],  0, "The top element at " + (i + 4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 5],  0, "The top element at " + (i + 5).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 6],  0, "The top element at " + (i + 6).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 7], 32, "The top element at " + (i + 7).ToString() + " should be 32");
                m_log.CHECK_EQ(rgTop[i + 8],  0, "The top element at " + (i + 8).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 9],  0, "The top element at " + (i + 9).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 10],  0, "The top element at " + (i + 10).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 11],  0, "The top element at " + (i + 11).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 12], 31, "The top element at " + (i + 12).ToString() + " should be 31");
                m_log.CHECK_EQ(rgTop[i + 13],  0, "The top element at " + (i + 13).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 14],  0, "The top element at " + (i + 14).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 15],  0, "The top element at " + (i + 15).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 16], 27, "The top element at " + (i + 16).ToString() + " should be 27");
                m_log.CHECK_EQ(rgTop[i + 17],  0, "The top element at " + (i + 17).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 18],  0, "The top element at " + (i + 18).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 19],  0, "The top element at " + (i + 19).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 20], 33, "The top element at " + (i + 20).ToString() + " should be 33");
                m_log.CHECK_EQ(rgTop[i + 21],  0, "The top element at " + (i + 21).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 22],  0, "The top element at " + (i + 22).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 23],  0, "The top element at " + (i + 23).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 24],  0, "The top element at " + (i + 24).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 25],  0, "The top element at " + (i + 25).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 26], 34, "The top element at " + (i + 26).ToString() + " should be 34");
                m_log.CHECK_EQ(rgTop[i + 27],  0, "The top element at " + (i + 27).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 28],  0, "The top element at " + (i + 28).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 29],  0, "The top element at " + (i + 29).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 30],  0, "The top element at " + (i + 30).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 31], 36, "The top element at " + (i + 31).ToString() + " should be 36");
                m_log.CHECK_EQ(rgTop[i + 32],  0, "The top element at " + (i + 32).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 33],  0, "The top element at " + (i + 33).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 34], 18, "The top element at " + (i + 34).ToString() + " should be 18");
                m_log.CHECK_EQ(rgTop[i + 35],  0, "The top element at " + (i + 35).ToString() + " should be 0");
            }
        }

        /// <summary>
        /// Test for 2x4 rectangular unpooling layer
        /// </summary>
        public void TestForwardRectWithPad(LayerParameter.LayerType type)
        {
            LayerParameter p = new LayerParameter(type);
            p.pooling_param.kernel_size.Add(3);
            p.pooling_param.stride.Add(2);
            p.pooling_param.pad.Add(1);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            p.pooling_param.engine = m_engine;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 2, 4);
            BottomMask.Reshape(nNum, nChannels, 2, 4);
            // Input: 2x2 channels of:
            //  [9 5 5 8]
            //  [9 5 5 8]
            // Mask: 2x2 channels of:
            //  [ 8 11 11 14 ]
            //  [ 8 11 11 14 ]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            double[] rgBottomMask = convert(BottomMask.mutable_cpu_data);

            for (int i = 0; i < 8 * nNum * nChannels; i += 8)
            {
                rgBottom[i + 0] = 9;
                rgBottom[i + 1] = 5;
                rgBottom[i + 2] = 5;
                rgBottom[i + 3] = 8;
                rgBottom[i + 4] = 9;
                rgBottom[i + 5] = 5;
                rgBottom[i + 6] = 5;
                rgBottom[i + 7] = 8;
                // And the mask
                rgBottomMask[i + 0] = 8;
                rgBottomMask[i + 1] = 11;
                rgBottomMask[i + 2] = 11;
                rgBottomMask[i + 3] = 14;
                rgBottomMask[i + 4] = 8;
                rgBottomMask[i + 5] = 11;
                rgBottomMask[i + 6] = 11;
                rgBottomMask[i + 7] = 14;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);
            BottomMask.mutable_cpu_data = convert(rgBottomMask);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new basecode.CancelEvent());
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(4, Top.height, "The top height should equal 4.");
            m_log.CHECK_EQ(8, Top.width, "The top width should equal 8.");

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:
            //  [ 0  0  0  0  0  0  0  0 ]
            //  [ 9  0  0  5  0  0  8  0 ]
            //  [ 0  0  0  0  0  0  0  0 ]
            //  [ 0  0  0  0  0  0  0  0 ]
            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 32 * nNum * nChannels; i += 32)
            {
                int l = 0;
                m_log.CHECK_EQ(rgTop[i + l + 0], 0, "The top element at " + (i + l + 0).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 1], 0, "The top element at " + (i + l + 1).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 2], 0, "The top element at " + (i + l + 2).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 3], 0, "The top element at " + (i + l + 3).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 4], 0, "The top element at " + (i + l + 4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 5], 0, "The top element at " + (i + l + 5).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 6], 0, "The top element at " + (i + l + 6).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 7], 0, "The top element at " + (i + l + 7).ToString() + " should be 0");

                l += 8;
                m_log.CHECK_EQ(rgTop[i + l + 0], 9, "The top element at " + (i + l + 0).ToString() + " should be 9");
                m_log.CHECK_EQ(rgTop[i + l + 1], 0, "The top element at " + (i + l + 1).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 2], 0, "The top element at " + (i + l + 2).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 3], 5, "The top element at " + (i + l + 3).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + l + 4], 0, "The top element at " + (i + l + 4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 5], 0, "The top element at " + (i + l + 5).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 6], 8, "The top element at " + (i + l + 6).ToString() + " should be 8");
                m_log.CHECK_EQ(rgTop[i + l + 7], 0, "The top element at " + (i + l + 7).ToString() + " should be 0");

                l += 8;
                m_log.CHECK_EQ(rgTop[i + l + 0], 0, "The top element at " + (i + l + 0).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 1], 0, "The top element at " + (i + l + 1).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 2], 0, "The top element at " + (i + l + 2).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 3], 0, "The top element at " + (i + l + 3).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 4], 0, "The top element at " + (i + l + 4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 5], 0, "The top element at " + (i + l + 5).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 6], 0, "The top element at " + (i + l + 6).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 7], 0, "The top element at " + (i + l + 7).ToString() + " should be 0");

                l += 8;
                m_log.CHECK_EQ(rgTop[i + l + 0], 0, "The top element at " + (i + l + 0).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 1], 0, "The top element at " + (i + l + 1).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 2], 0, "The top element at " + (i + l + 2).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 3], 0, "The top element at " + (i + l + 3).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 4], 0, "The top element at " + (i + l + 4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 5], 0, "The top element at " + (i + l + 5).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 6], 0, "The top element at " + (i + l + 6).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + l + 7], 0, "The top element at " + (i + l + 7).ToString() + " should be 0");
            }
        }

        public void TestForwardSquareWithPad(LayerParameter.LayerType type)
        {
            LayerParameter p = new LayerParameter(type);
            p.pooling_param.kernel_h = 2;
            p.pooling_param.kernel_w = 2;
            p.pooling_param.stride.Add(2);
            p.pooling_param.pad.Add(1);
            p.pooling_param.pool = PoolingParameter.PoolingMethod.MAX;
            p.pooling_param.engine = m_engine;
            int nNum = 2;
            int nChannels = 2;

            Bottom.Reshape(nNum, nChannels, 3, 3);
            BottomMask.Reshape(nNum, nChannels, 3, 3);
            // Input: 2x2 channels of:
            //  [ 3 12  5]
            //  [ 8  9  2]
            //  [ 5  8  3]
            //
            // Mask: 2x2 channels of:
            //  [ 0  1  3]
            //  [ 5 12  9]
            //  [15 17 24]
            double[] rgBottom = convert(Bottom.mutable_cpu_data);
            double[] rgBottomMask = convert(BottomMask.mutable_cpu_data);

            for (int i = 0; i < 9 * nNum * nChannels; i += 9)
            {
                rgBottom[i + 0] = 3;
                rgBottom[i + 1] = 12;
                rgBottom[i + 2] = 5;
                rgBottom[i + 3] = 8;
                rgBottom[i + 4] = 9;
                rgBottom[i + 5] = 2;
                rgBottom[i + 6] = 5;
                rgBottom[i + 7] = 8;
                rgBottom[i + 8] = 3;

                // For the mask

                rgBottomMask[i + 0] = 0;        
                rgBottomMask[i + 1] = 1;
                rgBottomMask[i + 2] = 3;
                rgBottomMask[i + 3] = 5;
                rgBottomMask[i + 4] = 12;
                rgBottomMask[i + 5] = 9;
                rgBottomMask[i + 6] = 15;
                rgBottomMask[i + 7] = 17;
                rgBottomMask[i + 8] = 24;
            }
            Bottom.mutable_cpu_data = convert(rgBottom);
            BottomMask.mutable_cpu_data = convert(rgBottomMask);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new basecode.CancelEvent());
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(nNum, Top.num, "The top num should equal " + nNum.ToString());
            m_log.CHECK_EQ(nChannels, Top.channels, "The top channels should equal " + nChannels.ToString());
            m_log.CHECK_EQ(5, Top.height, "The top height should equal 5.");
            m_log.CHECK_EQ(5, Top.width, "The top width should equal 5.");

            layer.Forward(BottomVec, TopVec);

            // Expected output: 2x2 channels of:                 (   index values   )
            //  [  .  .  .  .  .  .  .]
            //  [  .  3 12  0  5  0  .]   eg  [ 3 12][ 0  5][ 0  .]   [ 0  1][ 2  3][ 4  .]
            //  [  .  8  0  0  0  2  .]      _[ 8  0][ 0  0][ 2  .]_ _[ 5  6][ 7  8][ 9  .]_
            //  [  .  0  0  9  0  0  .]       [ 0  0][ 9  0][ 0  .]   [10 11][12 13][14  .]
            //  [  .  5  0  8  0  0  .]      _[ 5  0][ 8  0][ 0  .]_ _[15 16][17 18][19  .]_
            //  [  .  0  0  0  0  3  .]       [ 0  0][ 0  0][ 3  .]   [20 21][22 23][24  .]
            //  [  .  .  .  .  .  .  .]       [ .  .][ .  .][ .  .]   [ .  .][ .  .][ .  .]

            double[] rgTop = convert(Top.update_cpu_data());
            for (int i = 0; i < 25 * nNum * nChannels; i += 25)
            {
                m_log.CHECK_EQ(rgTop[i + 0], 3, "The top element at " + (i + 0).ToString() + " should be 3");
                m_log.CHECK_EQ(rgTop[i + 1], 12, "The top element at " + (i + 1).ToString() + " should be 12");
                m_log.CHECK_EQ(rgTop[i + 2], 0, "The top element at " + (i + 2).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 3], 5, "The top element at " + (i + 3).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 4], 0, "The top element at " + (i + 4).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 5], 8, "The top element at " + (i + 5).ToString() + " should be 8");
                m_log.CHECK_EQ(rgTop[i + 6], 0, "The top element at " + (i + 6).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 7], 0, "The top element at " + (i + 7).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 8], 0, "The top element at " + (i + 8).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 9],  2, "The top element at " + (i + 9).ToString() + " should be 2");
                m_log.CHECK_EQ(rgTop[i + 10], 0, "The top element at " + (i + 10).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 11], 0, "The top element at " + (i + 11).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 12], 9, "The top element at " + (i + 12).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 13], 0, "The top element at " + (i + 13).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 14], 0, "The top element at " + (i + 14).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 15], 5, "The top element at " + (i + 15).ToString() + " should be 5");
                m_log.CHECK_EQ(rgTop[i + 16], 0, "The top element at " + (i + 16).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 17], 8, "The top element at " + (i + 17).ToString() + " should be 8");
                m_log.CHECK_EQ(rgTop[i + 18], 0, "The top element at " + (i + 18).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 19], 0, "The top element at " + (i + 19).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 20], 0, "The top element at " + (i + 20).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 21], 0, "The top element at " + (i + 21).ToString() + " should be 9");
                m_log.CHECK_EQ(rgTop[i + 22], 0, "The top element at " + (i + 22).ToString() + " should be 9");
                m_log.CHECK_EQ(rgTop[i + 23], 0, "The top element at " + (i + 23).ToString() + " should be 0");
                m_log.CHECK_EQ(rgTop[i + 24], 3, "The top element at " + (i + 24).ToString() + " should be 3");
            }
        }
    }
}
