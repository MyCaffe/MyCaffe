using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.basecode;
using System.Drawing;
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestInnerProductLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
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
        public void TestSetupTransposeFalse()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestSetupTransposeFalse();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupTransposeTrue()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestSetupTransposeTrue();
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
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
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
        public void TestForwardTranspose()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestForwardTranspose();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNoBatch()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestForwardNoBatch();
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
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientTranspose()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestGradientTranspose();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardTranspose()
        {
            InnerProductLayerTest test = new InnerProductLayerTest();

            try
            {
                foreach (IInnerProductLayerTest t in test.Tests)
                {
                    t.TestBackwardTranspose();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    class InnerProductLayerTest : TestBase
    {
        public InnerProductLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("InnerProduct Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new InnerProductLayerTest<double>(strName, nDeviceID, engine);
            else
                return new InnerProductLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IInnerProductLayerTest : ITest
    {
        void TestSetup();
        void TestSetupTransposeFalse();
        void TestSetupTransposeTrue();
        void TestForward();
        void TestForwardTranspose();
        void TestForwardNoBatch();
        void TestGradient();
        void TestGradientTranspose();
        void TestBackwardTranspose();
    }

    class InnerProductLayerTest<T> : TestEx<T>, IInnerProductLayerTest
    {
        Blob<T> m_blob_bottom_nobatch;

        public InnerProductLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_blob_bottom_nobatch = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 2, 3, 4 });
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_blob_bottom_nobatch.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public Blob<T> BottomNoBatch
        {
            get { return m_blob_bottom_nobatch; }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 2, "Expected top num to equal 2.");
                m_log.CHECK_EQ(Top.height, 1, "Expected top height to equal 1.");
                m_log.CHECK_EQ(Top.width, 1, "Expected top width to equal 1.");
                m_log.CHECK_EQ(Top.channels, 10, "Expected top channels to equal 10.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupTransposeFalse()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.transpose = false;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 2, "Expected top num to equal 2.");
                m_log.CHECK_EQ(Top.height, 1, "Expected top height to equal 1.");
                m_log.CHECK_EQ(Top.width, 1, "Expected top width to equal 1.");
                m_log.CHECK_EQ(Top.channels, 10, "Expected top channels to equal 10.");
                m_log.CHECK_EQ(2, layer.blobs[0].num_axes, "The blob[0] should have 2 axes.");
                m_log.CHECK_EQ(10, layer.blobs[0].shape(0), "The blob[0] shape(0) should be 10.");
                m_log.CHECK_EQ(60, layer.blobs[0].shape(1), "The blob[0] shape(1) should be 60.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupTransposeTrue()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.transpose = true;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 2, "Expected top num to equal 2.");
                m_log.CHECK_EQ(Top.height, 1, "Expected top height to equal 1.");
                m_log.CHECK_EQ(Top.width, 1, "Expected top width to equal 1.");
                m_log.CHECK_EQ(Top.channels, 10, "Expected top channels to equal 10.");
                m_log.CHECK_EQ(2, layer.blobs[0].num_axes, "The blob[0] should have 2 axes.");
                m_log.CHECK_EQ(60, layer.blobs[0].shape(0), "The blob[0] shape(0) should be 60.");
                m_log.CHECK_EQ(10, layer.blobs[0].shape(1), "The blob[0] shape(1) should be 10.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.weight_filler.type = "uniform";
            p.inner_product_param.bias_filler.type = "uniform";
            p.inner_product_param.bias_filler.min = 1.0;
            p.inner_product_param.bias_filler.max = 2.0;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];

                    m_log.CHECK_GE(dfTop, 1.0, "Expected top value at " + i.ToString() + " to be greater than or equal 1.0");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardTranspose()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.weight_filler.type = "uniform";
            p.inner_product_param.bias_filler.type = "uniform";
            p.inner_product_param.bias_filler.min = 1.0;
            p.inner_product_param.bias_filler.max = 2.0;
            p.inner_product_param.transpose = false;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);
            InnerProductLayer<T> ip_t = null;
            Blob<T> blobTop = null;
            Blob<T> blobTop2 = null;
            Blob<T> blobTopT = null;

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                int nCount = Top.count();
                blobTop = new Blob<T>(m_cuda, m_log);
                blobTop.ReshapeLike(Top);
                m_cuda.copy(nCount, Top.gpu_data, blobTop.mutable_gpu_data);

                blobTop2 = new Blob<T>(m_cuda, m_log);
                TopVec.Clear();
                TopVec.Add(blobTop2);

                p.inner_product_param.transpose = true;
                ip_t = new InnerProductLayer<T>(m_cuda, m_log, p);
                ip_t.Setup(BottomVec, TopVec);
                int nCountW = layer.blobs[0].count();

                m_log.CHECK_EQ(nCountW, ip_t.blobs[0].count(), "The ip and ip_t layers should have blobs[0] with the same count.");

                // manually copy and transpose the weights from 1st IP layer into 2nd.
                T[] rgW = layer.blobs[0].update_cpu_data();
                T[] rgWt = ip_t.blobs[0].mutable_cpu_data;
                int nWidth = layer.blobs[0].shape(1);
                int nWidthT = ip_t.blobs[0].shape(1);

                for (int i = 0; i < nCountW; i++)
                {
                    int nR = i / nWidth;
                    int nC = i % nWidth;
                    rgWt[nC * nWidthT + nR] = rgW[nR * nWidth + nC]; // copy while transposing
                }

                ip_t.blobs[0].mutable_cpu_data = rgWt;

                // copy bias from 1st IP layer to 2nd layer
                m_log.CHECK_EQ(layer.blobs[1].count(), ip_t.blobs[1].count(), "The first and second ip layers do not have the same blob[1] count.");
                m_cuda.copy(layer.blobs[1].count(), layer.blobs[1].gpu_data, ip_t.blobs[1].mutable_gpu_data);

                ip_t.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(nCount, Top.count(), "Invalid count for top blob for IP with transpose.");

                blobTopT = new Blob<T>(m_cuda, m_log);
                blobTopT.ReshapeLike(TopVec[0]);
                m_cuda.copy(nCount, TopVec[0].gpu_data, blobTopT.mutable_gpu_data);

                double[] rgData = convert(blobTop.update_cpu_data());
                double[] rgDataT = convert(blobTopT.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    m_log.EXPECT_EQUAL<float>(rgData[i], rgDataT[i], "The data at " + i.ToString() + " should be equal.");
                }
            }
            finally
            {
                layer.Dispose();

                if (ip_t != null)
                    ip_t.Dispose();

                if (blobTop != null)
                    blobTop.Dispose();

                if (blobTop2 != null)
                    blobTop2.Dispose();

                if (blobTopT != null)
                    blobTopT.Dispose();
            }
        }

        public void TestForwardNoBatch()
        {
            BottomVec.Clear();
            BottomVec.Add(BottomNoBatch);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.weight_filler.type = "uniform";
            p.inner_product_param.bias_filler.type = "uniform";
            p.inner_product_param.bias_filler.min = 1.0;
            p.inner_product_param.bias_filler.max = 2.0;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];

                    m_log.CHECK_GE(dfTop, 1.0, "Expected top value at " + i.ToString() + " to be greater than or equal 1.0");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.weight_filler.type = "uniform";
            p.inner_product_param.bias_filler.type = "uniform";
            p.inner_product_param.bias_filler.min = 1.0;
            p.inner_product_param.bias_filler.max = 2.0;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientTranspose()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.weight_filler.type = "uniform";
            p.inner_product_param.bias_filler.type = "uniform";
            p.inner_product_param.bias_filler.min = 1.0;
            p.inner_product_param.bias_filler.max = 2.0;
            p.inner_product_param.transpose = true;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackwardTranspose()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = 10;
            p.inner_product_param.weight_filler.type = "uniform";
            p.inner_product_param.bias_filler.type = "uniform";
            p.inner_product_param.bias_filler.min = 1.0;
            p.inner_product_param.bias_filler.max = 2.0;
            p.inner_product_param.transpose = false;
            InnerProductLayer<T> layer = new InnerProductLayer<T>(m_cuda, m_log, p);
            InnerProductLayer<T> ip_t = null;
            Blob<T> blobTop = null;
            Blob<T> blobDiff = null;
            Blob<T> blobW = null;
            Blob<T> blobBottomDiff = null;
            Blob<T> blobNewTop = null;

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // copy top blob
                blobTop = new Blob<T>(m_cuda, m_log);
                blobTop.CopyFrom(Top, false, true);

                // fake top diff
                blobDiff = new Blob<T>(m_cuda, m_log);
                blobDiff.ReshapeLike(Top);
                {
                    FillerParameter fp = new FillerParameter("uniform");
                    Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                    filler.Fill(blobDiff);
                }

                m_cuda.copy(TopVec[0].count(), blobDiff.gpu_data, TopVec[0].mutable_gpu_diff);
                List<bool> rgbPropagateDown = Utility.Create<bool>(1, true);

                layer.Backward(TopVec, rgbPropagateDown, BottomVec);

                // copy first ip's weights and their diffs.
                blobW = new Blob<T>(m_cuda, m_log);
                blobW.CopyFrom(layer.blobs[0], false, true);
                blobW.CopyFrom(layer.blobs[0], true, true);

                // copy bottom diffs
                blobBottomDiff = new Blob<T>(m_cuda, m_log);
                blobBottomDiff.CopyFrom(BottomVec[0], true, true);

                // repeat original top with transposed ip.
                blobNewTop = new Blob<T>(m_cuda, m_log);
                TopVec.Clear();
                TopVec.Add(blobNewTop);

                p.inner_product_param.transpose = true;
                ip_t = new InnerProductLayer<T>(m_cuda, m_log, p);

                ip_t.Setup(BottomVec, TopVec);

                // manually copy and transpose the weights from 1st IP layer into 2nd
                {
                    T[] rgWsrc = blobW.update_cpu_data();
                    T[] rgWt = ip_t.blobs[0].mutable_cpu_data;
                    int nWidth = layer.blobs[0].shape(1);
                    int nWidthT = ip_t.blobs[0].shape(1);

                    for (int i = 0; i < layer.blobs[0].count(); i++)
                    {
                        int nR = i / nWidth;
                        int nC = i % nWidth;
                        rgWt[nC * nWidthT + nR] = rgWsrc[nR * nWidth + nC]; // copy while transposing
                    }

                    ip_t.blobs[0].mutable_cpu_data = rgWt;

                    // copy bias from 1st IP layer to 2nd IP layer
                    m_log.CHECK_EQ(layer.blobs[1].count(), ip_t.blobs[1].count(), "The first and second layer blobs[1] should have the same count.");
                    m_cuda.copy(layer.blobs[1].count(), layer.blobs[1].gpu_data, ip_t.blobs[1].mutable_gpu_data);
                }

                ip_t.Forward(BottomVec, TopVec);
                m_cuda.copy(TopVec[0].count(), blobDiff.gpu_data, TopVec[0].mutable_gpu_diff);
                ip_t.Backward(TopVec, rgbPropagateDown, BottomVec);

                double[] rgData = convert(blobW.update_cpu_diff());
                double[] rgDataT = convert(ip_t.blobs[0].update_cpu_diff());
                int nWidth1 = layer.blobs[0].shape(1);
                int nWidthT1 = ip_t.blobs[0].shape(1);

                for (int i = 0; i < layer.blobs[0].count(); i++)
                {
                    int nR = i / nWidth1;
                    int nC = i % nWidth1;

                    double dfData1 = rgData[nR * nWidth1 + nC];
                    double dfDataT1 = rgDataT[nC * nWidthT1 + nR];

                    m_log.CHECK_NE(0.0, dfData1, "The data at " + i.ToString() + " should not be zero.");
                    m_log.EXPECT_EQUAL<float>(dfData1, dfDataT1, "The data items at " + i.ToString() + " should be equal.");
                }

                rgData = convert(blobBottomDiff.update_cpu_diff());
                rgDataT = convert(BottomVec[0].update_cpu_diff());

                for (int i = 0; i < BottomVec[0].count(); i++)
                {
                    double dfData1 = rgData[i];
                    double dfDataT1 = rgDataT[i];

                    m_log.CHECK_NE(0.0, dfData1, "The data at " + i.ToString() + " should not be zero.");
                    m_log.EXPECT_EQUAL<float>(dfData1, dfDataT1, "The data items at " + i.ToString() + " should be equal.");
                }
            }
            finally
            {
                layer.Dispose();

                if (blobNewTop != null)
                    blobNewTop.Dispose();

                if (blobBottomDiff != null)
                    blobBottomDiff.Dispose();

                if (blobW != null)
                    blobW.Dispose();

                if (blobTop != null)
                    blobTop.Dispose();

                if (blobDiff != null)
                    blobDiff.Dispose();

                if (ip_t != null)
                    ip_t.Dispose();
            }
        }
    }
}
