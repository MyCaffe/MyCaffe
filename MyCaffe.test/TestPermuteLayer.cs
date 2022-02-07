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
using MyCaffe.layers.beta;
using MyCaffe.layers.ssd;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPermuteLayer
    {

        [TestMethod]
        public void TestSetup()
        {
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
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
        public void TestSetupIdentity()
        {
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
                {
                    t.TestSetupIdentity();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardIdentity()
        {
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
                {
                    t.TestForwardIdentity();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward2D()
        {
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
                {
                    t.TestForward2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward3D()
        {
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
                {
                    t.TestForward3D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTwoPermute()
        {
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
                {
                    t.TestTwoPermute();
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
            PermuteLayerTest test = new PermuteLayerTest();

            try
            {
                foreach (IPermuteLayerTest t in test.Tests)
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

    interface IPermuteLayerTest : ITest
    {
        void TestSetup();
        void TestSetupIdentity();
        void TestForwardIdentity();
        void TestForward2D();
        void TestForward3D();
        void TestTwoPermute();
        void TestGradient();
    }

    class PermuteLayerTest : TestBase
    {
        public PermuteLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Permute Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PermuteLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PermuteLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PermuteLayerTest<T> : TestEx<T>, IPermuteLayerTest
    {
        float m_fEps = 1e-6f;

        public PermuteLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 2, 2, 3 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("constant", 1.0);
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            p.permute_param.order.Add(0);
            p.permute_param.order.Add(2);
            p.permute_param.order.Add(3);
            p.permute_param.order.Add(1);
            PermuteLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;

            try
            { 
                m_blob_bottom.Reshape(2, 3, 4, 5);
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, 2, "The top num should equal 2.");
                m_log.CHECK_EQ(m_blob_top.channels, 4, "The top channels should equal 4.");
                m_log.CHECK_EQ(m_blob_top.height, 5, "The top height should equal 5.");
                m_log.CHECK_EQ(m_blob_top.width, 3, "The top width should equal 3.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupIdentity()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            PermuteLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;

            try
            { 
                m_blob_bottom.Reshape(2, 3, 4, 5);
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, 2, "The top num should equal 2.");
                m_log.CHECK_EQ(m_blob_top.channels, 3, "The top channels should equal 3.");
                m_log.CHECK_EQ(m_blob_top.height, 4, "The top height should equal 4.");
                m_log.CHECK_EQ(m_blob_top.width, 5, "The top width should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardIdentity()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            PermuteLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;

            try
            {
                m_blob_bottom.Reshape(2, 3, 4, 5);
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, 2, "The top num should equal 2.");
                m_log.CHECK_EQ(m_blob_top.channels, 3, "The top channels should equal 3.");
                m_log.CHECK_EQ(m_blob_top.height, 4, "The top height should equal 4.");
                m_log.CHECK_EQ(m_blob_top.width, 5, "The top width should equal 5.");

                for (int i = 0; i < m_blob_bottom.count(); i++)
                {
                    double dfBtm = Utility.ConvertVal<T>(m_blob_bottom.GetData(i));
                    double dfTop = Utility.ConvertVal<T>(m_blob_top.GetData(i));
                    m_log.EXPECT_NEAR(dfBtm, dfTop, m_fEps, "The top and bottom values do not match!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward2D()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            p.permute_param.order.Add(0);
            p.permute_param.order.Add(1);
            p.permute_param.order.Add(3);
            p.permute_param.order.Add(2);
            PermuteLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;

            try
            {
                int nNum = 2;
                int nChannels = 3;
                int nHeight = 2;
                int nWidth = 3;
                m_blob_bottom.Reshape(nNum, nChannels, nHeight, nWidth);

                // Input: 2 x 3 channels of:
                //    [1 2 3]
                //    [4 5 6]
                for (int i = 0; i < nHeight * nWidth * nNum * nChannels; i += nHeight * nWidth)
                {
                    m_blob_bottom.SetData(1, i + 0);
                    m_blob_bottom.SetData(2, i + 1);
                    m_blob_bottom.SetData(3, i + 2);
                    m_blob_bottom.SetData(4, i + 3);
                    m_blob_bottom.SetData(5, i + 4);
                    m_blob_bottom.SetData(6, i + 5);
                }

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Expected output: 2 x 3 channels of:
                //    [1 4]
                //    [2 5]
                //    [3 6]
                for (int i = 0; i < nHeight * nWidth * nNum * nChannels; i += nHeight * nWidth)
                {
                    double dfVal;

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 0));
                    m_log.CHECK_EQ(dfVal, 1, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 1));
                    m_log.CHECK_EQ(dfVal, 4, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 2));
                    m_log.CHECK_EQ(dfVal, 2, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 3));
                    m_log.CHECK_EQ(dfVal, 5, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 4));
                    m_log.CHECK_EQ(dfVal, 3, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 5));
                    m_log.CHECK_EQ(dfVal, 6, "The value is not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward3D()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            p.permute_param.order.Add(0);
            p.permute_param.order.Add(2);
            p.permute_param.order.Add(3);
            p.permute_param.order.Add(1);
            PermuteLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;

            try
            { 
                int nNum = 2;
                int nChannels = 2;
                int nHeight = 2;
                int nWidth = 3;
                m_blob_bottom.Reshape(nNum, nChannels, nHeight, nWidth);

                // Input: 2 of:
                //    [1 2 3]
                //    [4 5 6]
                //    =======
                //    [7 8 9]
                //    [10 11 12]
                int nInnerDim = nChannels * nHeight * nWidth;
                for (int i = 0; i < nNum; i++)
                {
                    for (int j = 0; j < nInnerDim; j++)
                    {
                        m_blob_bottom.SetData(j + 1, i * nInnerDim + j);
                    }
                }

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Expected output: 2 of:
                //    [1 7]
                //    [2 8]
                //    [3 9]
                //    =====
                //    [4 10]
                //    [5 11]
                //    [6 12]
                for (int i = 0; i < nNum * nInnerDim; i += nInnerDim)
                {
                    double dfVal;

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 0));
                    m_log.CHECK_EQ(dfVal, 1, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 1));
                    m_log.CHECK_EQ(dfVal, 7, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 2));
                    m_log.CHECK_EQ(dfVal, 2, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 3));
                    m_log.CHECK_EQ(dfVal, 8, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 4));
                    m_log.CHECK_EQ(dfVal, 3, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 5));
                    m_log.CHECK_EQ(dfVal, 9, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 6));
                    m_log.CHECK_EQ(dfVal, 4, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 7));
                    m_log.CHECK_EQ(dfVal, 10, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 8));
                    m_log.CHECK_EQ(dfVal, 5, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 9));
                    m_log.CHECK_EQ(dfVal, 11, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 10));
                    m_log.CHECK_EQ(dfVal, 6, "The value is not as expected!");

                    dfVal = Utility.ConvertVal<T>(m_blob_top.GetData(i + 11));
                    m_log.CHECK_EQ(dfVal, 12, "The value is not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestTwoPermute()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            p.permute_param.order.Add(0);
            p.permute_param.order.Add(2);
            p.permute_param.order.Add(3);
            p.permute_param.order.Add(1);
            PermuteLayer<T> layer1 = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;
            PermuteLayer<T> layer2 = null;

            try
            {
                Blob<T> input1 = new Blob<T>(m_cuda, m_log, 2, 3, 4, 5);
                FillerParameter fp = new FillerParameter("gaussian");
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(input1);

                Blob<T> output1 = new Blob<T>(m_cuda, m_log);

                BlobCollection<T> bottom_vec = new BlobCollection<T>();
                BlobCollection<T> top_vec = new BlobCollection<T>();
                bottom_vec.Add(input1);
                top_vec.Add(output1);

                layer1.Setup(bottom_vec, top_vec);
                layer1.Forward(bottom_vec, top_vec);

                m_log.CHECK_EQ(output1.num, 2, "The outpt1 num should be 2");
                m_log.CHECK_EQ(output1.channels, 4, "The outpt1 channels should be 4");
                m_log.CHECK_EQ(output1.height, 5, "The outpt1 height should be 5");
                m_log.CHECK_EQ(output1.width, 3, "The outpt1 width should be 3");

                // Create second permute layer which transfers back to the original order.
                p.permute_param.order.Clear();
                p.permute_param.order.Add(0);
                p.permute_param.order.Add(3);
                p.permute_param.order.Add(1);
                p.permute_param.order.Add(2);
                layer2 = Layer<T>.Create(m_cuda, m_log, p, null) as PermuteLayer<T>;

                Blob<T> output2 = new Blob<T>(m_cuda, m_log);
                bottom_vec.Clear();
                top_vec.Clear();
                bottom_vec.Add(output1);
                top_vec.Add(output2);

                layer2.Setup(bottom_vec, top_vec);
                layer2.Forward(bottom_vec, top_vec);

                m_log.CHECK_EQ(output2.num, 2, "The outpt2 num should be 2");
                m_log.CHECK_EQ(output2.channels, 3, "The outpt2 channels should be 3");
                m_log.CHECK_EQ(output2.height, 4, "The outpt2 height should be 4");
                m_log.CHECK_EQ(output2.width, 5, "The outpt2 width should be 5");

                for (int i = 0; i < output2.count(); i++)
                {
                    double df1 = Utility.ConvertVal<T>(input1.GetData(i));
                    double df2 = Utility.ConvertVal<T>(output2.GetData(i));
                    m_log.EXPECT_NEAR(df1, df2, m_fEps, "The output values are not the same!");
                }

                input1.Dispose();
                output2.Dispose();
                output2.Dispose();
            }
            finally
            {
                layer1.Dispose();

                if (layer2 != null)
                    layer2.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            p.permute_param.order.Add(0);
            p.permute_param.order.Add(2);
            p.permute_param.order.Add(3);
            p.permute_param.order.Add(1);
            PermuteLayer<T> layer = new PermuteLayer<T>(m_cuda, m_log, p);

            try
            { 
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
