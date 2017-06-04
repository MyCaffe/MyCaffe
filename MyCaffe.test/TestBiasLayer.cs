using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBiasLayer
    {
        [TestMethod]
        public void TestForwardEltwise()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardEltwise();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardEltwiseInPlace()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardEltwiseInPlace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardEltwiseInPlace()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestBackwardEltwiseInPlace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardEltwiseWithParam()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardEltwiseWithParam();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBroadcastBegin()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBroadcastBegin();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBroadcastMiddle()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBroadcastMiddle();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBroadcastMiddleInPlace()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBroadcastMiddleInPlace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardBroadcastMiddleInPlace()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestBackwardBroadcastMiddleInPlace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBroadcastMiddleWithParam()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBroadcastMiddleWithParam();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBroadcastEnd()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBroadcastEnd();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBias()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBias();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBiasAxis2()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestForwardBias();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientEltwise()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientEltwise();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientEltwiseWithParam()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientEltwise();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBroadcastBegin()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientBroadcastBegin();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBroadcastMiddle()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientBroadcastMiddle();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBroadcastMiddleWithParam()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientBroadcastMiddleWithParam();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBroadcastEnd()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientBroadcastEnd();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBias()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientBias();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBiasAxis2()
        {
            BiasLayerTest test = new BiasLayerTest();

            try
            {
                foreach (IBiasLayerTest t in test.Tests)
                {
                    t.TestGradientBiasAxis2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IBiasLayerTest : ITest
    {
        void TestForwardEltwise();
        void TestForwardEltwiseInPlace();
        void TestBackwardEltwiseInPlace();
        void TestForwardEltwiseWithParam();
        void TestForwardBroadcastBegin();
        void TestForwardBroadcastMiddle();
        void TestForwardBroadcastMiddleInPlace();
        void TestBackwardBroadcastMiddleInPlace();
        void TestForwardBroadcastMiddleWithParam();
        void TestForwardBroadcastEnd();
        void TestForwardBias();
        void TestForwardBiasAxis2();
        void TestGradientEltwise();
        void TestGradientEltwiseWithParam();
        void TestGradientBroadcastBegin();
        void TestGradientBroadcastMiddle();
        void TestGradientBroadcastMiddleWithParam();
        void TestGradientBroadcastEnd();
        void TestGradientBias();
        void TestGradientBiasAxis2();
    }

    class BiasLayerTest : TestBase
    {
        public BiasLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Bias Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BiasLayerTest<double>(strName, nDeviceID, engine);
            else
                return new BiasLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BiasLayerTest<T> : TestEx<T>, IBiasLayerTest
    {
        Blob<T> m_blob_bottom_eltwise;
        Blob<T> m_blob_bottom_broadcast_0;
        Blob<T> m_blob_bottom_broadcast_1;
        Blob<T> m_blob_bottom_broadcast_2;
        Blob<T> m_blob_bottom_bias;

        public BiasLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_blob_bottom_eltwise = new Blob<T>(m_cuda, m_log, Bottom);
            m_blob_bottom_broadcast_0 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_broadcast_1 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_broadcast_2 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_bias = new Blob<T>(m_cuda, m_log, new List<int>());
            m_engine = engine;

            m_cuda.rng_setseed(1701);
            List<int> rgShape0 = new List<int>() { 2, 3 };
            m_blob_bottom_broadcast_0.Reshape(rgShape0);
            List<int> rgShape1 = new List<int>() { 3, 4 };
            m_blob_bottom_broadcast_1.Reshape(rgShape1);
            List<int> rgShape2 = new List<int>() { 4, 5 };
            m_blob_bottom_broadcast_2.Reshape(rgShape2);
            
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom_eltwise);
            filler.Fill(m_blob_bottom_broadcast_0);
            filler.Fill(m_blob_bottom_broadcast_1);
            filler.Fill(m_blob_bottom_broadcast_2);
            filler.Fill(m_blob_bottom_bias);
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = 1;
            fp.max = 10;
            return fp;
        }

        protected override void dispose()
        {
            m_blob_bottom_eltwise.Dispose();
            m_blob_bottom_broadcast_0.Dispose();
            m_blob_bottom_broadcast_1.Dispose();
            m_blob_bottom_broadcast_2.Dispose();
            m_blob_bottom_bias.Dispose();
            base.dispose();
        }

        public Blob<T> BottomEltwise
        {
            get { return m_blob_bottom_eltwise; }
        }

        public Blob<T> BottomBroadcast0
        {
            get { return m_blob_bottom_broadcast_0; }
        }

        public Blob<T> BottomBroadcast1
        {
            get { return m_blob_bottom_broadcast_1; }
        }

        public Blob<T> BottomBroadcast2
        {
            get { return m_blob_bottom_broadcast_2; }
        }

        public Blob<T> BottomBias
        {
            get { return m_blob_bottom_bias; }
        }

        public void TestForwardEltwise()
        {
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());
            int nCount = Top.count();
            double[] rgInDataA = convert(Bottom.update_cpu_data());
            double[] rgInDataB = convert(BottomEltwise.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] + rgInDataB[i];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestForwardEltwiseInPlace()
        {
            TopVec[0] = Bottom; // in-place computation.
            Blob<T> blobOriginalBottom = new Blob<T>(m_cuda, m_log, Bottom.shape());
            blobOriginalBottom.CopyFrom(Bottom);
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Bottom.update_cpu_data());
            int nCount = Bottom.count();
            double[] rgInDataA = convert(blobOriginalBottom.update_cpu_data());
            double[] rgInDataB = convert(BottomEltwise.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] + rgInDataB[i];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestBackwardEltwiseInPlace()
        {
            Blob<T> blobOriginalBottom = new Blob<T>(m_cuda, m_log, Bottom.shape());
            blobOriginalBottom.CopyFrom(Bottom);
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            Blob<T> blobTopDiff = new Blob<T>(m_cuda, m_log, Bottom.shape());
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = 1.0;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(blobTopDiff);

            List<bool> rgPropagateDown = Utility.Create<bool>(2, true);
            // Run forward + backward without in-place computation;
            // save resulting bottom diffs.

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);
            m_cuda.copy(blobTopDiff.count(), blobTopDiff.gpu_data, Top.mutable_gpu_diff);
            layer.Backward(TopVec, rgPropagateDown, BottomVec);

            bool kReshape = true;
            bool kCopyDiff = true;
            Blob<T> blobOriginalBottomDiff = new Blob<T>(m_cuda, m_log);
            blobOriginalBottomDiff.CopyFrom(Bottom, kCopyDiff, kReshape);
            Blob<T> blobOriginalBiasDiff = new Blob<T>(m_cuda, m_log);
            blobOriginalBiasDiff.CopyFrom(BottomEltwise, kCopyDiff, kReshape);

            // Rerun forward + backward with in-place computation;
            // check that resulting bottom diffs are the same.
            TopVec[0] = Bottom; // in-place computation.
            layer.Forward(BottomVec, TopVec);
            m_cuda.copy(blobTopDiff.count(), blobTopDiff.gpu_data, Bottom.mutable_gpu_diff);
            layer.Backward(TopVec, rgPropagateDown, BottomVec);

            double[] rgOriginalBottomDiff = convert(blobOriginalBottomDiff.update_cpu_diff());
            double[] rgBottomDiff = convert(Bottom.update_cpu_diff());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfOriginalBottomDiff = rgOriginalBottomDiff[i];
                double dfBottomDiff = rgBottomDiff[i];

                m_log.EXPECT_NEAR(dfOriginalBottomDiff, dfBottomDiff, 1e-5);
            }

            double[] rgOriginalBiasDiff = convert(blobOriginalBiasDiff.update_cpu_diff());
            double[] rgBottomBiasDiff = convert(BottomEltwise.update_cpu_diff());

            for (int i = 0; i < BottomEltwise.count(); i++)
            {
                double dfOriginalBiasDiff = rgOriginalBiasDiff[i];
                double dfBottomBiasDiff = rgBottomBiasDiff[i];

                m_log.EXPECT_NEAR(dfOriginalBiasDiff, dfBottomBiasDiff, 1e-5);
            }
        }

        public void TestForwardEltwiseWithParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            p.bias_param.num_axes = -1;
            p.bias_param.filler = new FillerParameter("gaussian");
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());
            int nCount = Top.count();
            double[] rgInDataA = convert(Bottom.update_cpu_data());
            double[] rgInDataB = convert(layer.blobs[0].update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] + rgInDataB[i];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestForwardBroadcastBegin()
        {
            BottomVec.Add(BottomBroadcast0);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c, h, w));
                            double dfBottom = convert(Bottom.data_at(n, c, h, w));
                            double dfBroadcast = convert(BottomBroadcast0.data_at(n, c, 0, 0));

                            m_log.EXPECT_NEAR(dfTop, dfBottom + dfBroadcast, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardBroadcastMiddle()
        {
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 1;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c, h, w));
                            double dfBottom = convert(Bottom.data_at(n, c, h, w));
                            double dfBroadcast = convert(BottomBroadcast1.data_at(c, h, 0, 0));

                            m_log.EXPECT_NEAR(dfTop, dfBottom + dfBroadcast, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardBroadcastMiddleInPlace()
        {
            TopVec[0] = Bottom; // in-place computation.
            Blob<T> blobOriginalBottom = new Blob<T>(m_cuda, m_log, Bottom.shape());
            blobOriginalBottom.CopyFrom(Bottom);

            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 1;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            double dfTop = convert(Bottom.data_at(n, c, h, w));
                            double dfBottom = convert(blobOriginalBottom.data_at(n, c, h, w));
                            double dfBroadcast = convert(BottomBroadcast1.data_at(c, h, 0, 0));

                            m_log.EXPECT_NEAR(dfTop, dfBottom + dfBroadcast, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestBackwardBroadcastMiddleInPlace()
        {
            Blob<T> blobOriginalBottom = new Blob<T>(m_cuda, m_log, Bottom.shape());
            blobOriginalBottom.CopyFrom(Bottom);
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 1;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            Blob<T> blobTopDiff = new Blob<T>(m_cuda, m_log, Bottom.shape());
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = 1.0;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(blobTopDiff);

            List<bool> rgPropagateDown = Utility.Create<bool>(2, true);
            // Run forward + backward without in-place computation;
            // save resulting bottom diffs.

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);
            m_cuda.copy(blobTopDiff.count(), blobTopDiff.gpu_data, Top.mutable_gpu_diff);
            layer.Backward(TopVec, rgPropagateDown, BottomVec);

            bool kReshape = true;
            bool kCopyDiff = true;
            Blob<T> blobOriginalBottomDiff = new Blob<T>(m_cuda, m_log);
            blobOriginalBottomDiff.CopyFrom(Bottom, kCopyDiff, kReshape);
            Blob<T> blobOriginalBiasDiff = new Blob<T>(m_cuda, m_log);
            blobOriginalBiasDiff.CopyFrom(BottomBroadcast1, kCopyDiff, kReshape);

            // Rerun forward + backward with in-place computation;
            // check that resulting bottom diffs are the same.
            TopVec[0] = Bottom; // in-place computation.
            layer.Forward(BottomVec, TopVec);
            m_cuda.copy(blobTopDiff.count(), blobTopDiff.gpu_data, Bottom.mutable_gpu_diff);
            layer.Backward(TopVec, rgPropagateDown, BottomVec);

            double[] rgOriginalBottomDiff = convert(blobOriginalBottomDiff.update_cpu_diff());
            double[] rgBottomDiff = convert(Bottom.update_cpu_diff());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfOriginalBottomDiff = rgOriginalBottomDiff[i];
                double dfBottomDiff = rgBottomDiff[i];

                m_log.EXPECT_NEAR(dfOriginalBottomDiff, dfBottomDiff, 1e-5);
            }

            double[] rgOriginalBiasDiff = convert(blobOriginalBiasDiff.update_cpu_diff());
            double[] rgBottomBroadcastDiff = convert(BottomBroadcast1.update_cpu_diff());

            for (int i = 0; i < BottomBroadcast1.count(); i++)
            {
                double dfOriginalBiasDiff = rgOriginalBiasDiff[i];
                double dfBottomBroadcastDiff = rgBottomBroadcastDiff[i];

                m_log.EXPECT_NEAR(dfOriginalBiasDiff, dfBottomBroadcastDiff, 1e-5);
            }
        }

        public void TestForwardBroadcastMiddleWithParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 1;
            p.bias_param.num_axes = 2;
            p.bias_param.filler = new FillerParameter("gaussian");
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c, h, w));
                            double dfBottom = convert(Bottom.data_at(n, c, h, w));
                            double dfParam = convert(layer.blobs[0].data_at(c, h, 0, 0));

                            m_log.EXPECT_NEAR(dfTop, dfBottom + dfParam, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardBroadcastEnd()
        {
            BottomVec.Add(BottomBroadcast2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 2;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c, h, w));
                            double dfBottom = convert(Bottom.data_at(n, c, h, w));
                            double dfBroadcast = convert(BottomBroadcast2.data_at(h, w, 0, 0));

                            m_log.EXPECT_NEAR(dfTop, dfBottom + dfBroadcast, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardBias()
        {
            BottomVec.Add(BottomBias);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());
            int nCount = Top.count();
            double[] rgInDataA = convert(Bottom.update_cpu_data());
            double[] rgBias = convert(BottomBias.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] + rgBias[0];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestForwardBiasAxis2()
        {
            BottomVec.Add(BottomBias);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 2;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());
            int nCount = Top.count();
            double[] rgInDataA = convert(Bottom.update_cpu_data());
            double[] rgBias = convert(BottomBias.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] + rgBias[0];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestGradientEltwise()
        {
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestGradientEltwiseWithParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            p.bias_param.num_axes = -1;
            p.bias_param.filler = new FillerParameter("gaussian");
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastBegin()
        {
            BottomVec.Add(BottomBroadcast0);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 0;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastMiddle()
        {
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 1;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastMiddleWithParam()
        {
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 1;
            p.bias_param.num_axes = 2;
            p.bias_param.filler = new FillerParameter("gaussian");
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastEnd()
        {
            BottomVec.Add(BottomBroadcast2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 2;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBias()
        {
            BottomVec.Add(BottomBias);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBiasAxis2()
        {
            BottomVec.Add(BottomBias);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BIAS);
            p.bias_param.axis = 2;
            BiasLayer<T> layer = new BiasLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
