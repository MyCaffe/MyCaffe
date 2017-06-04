using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.common;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestScaleLayer
    {
        [TestMethod]
        public void TestForwardEltwise()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
        public void TestForwardBroadcastMiddleWithParamAndBias()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
                {
                    t.TestForwardBroadcastMiddleWithParamAndBias();
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
        public void TestForwardScale()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
                {
                    t.TestForwardScale();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardScaleAxis2()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
                {
                    t.TestForwardScale();
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
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
        public void TestGradientScale()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
                {
                    t.TestGradientScale();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientScaleAndBias()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
                {
                    t.TestGradientScaleAndBias();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientScaleAxis2()
        {
            ScaleLayerTest test = new ScaleLayerTest();

            try
            {
                foreach (IScaleLayerTest t in test.Tests)
                {
                    t.TestGradientScaleAxis2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IScaleLayerTest : ITest
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
        void TestForwardBroadcastMiddleWithParamAndBias();
        void TestForwardBroadcastEnd();
        void TestForwardScale();
        void TestForwardScaleAxis2();
        void TestGradientEltwise();
        void TestGradientEltwiseWithParam();
        void TestGradientBroadcastBegin();
        void TestGradientBroadcastMiddle();
        void TestGradientBroadcastMiddleWithParam();
        void TestGradientBroadcastEnd();
        void TestGradientScale();
        void TestGradientScaleAndBias();
        void TestGradientScaleAxis2();
    }

    class ScaleLayerTest : TestBase
    {
        public ScaleLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Scale Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ScaleLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ScaleLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ScaleLayerTest<T> : TestEx<T>, IScaleLayerTest
    {
        Blob<T> m_blob_bottom_eltwise;
        Blob<T> m_blob_bottom_broadcast_0;
        Blob<T> m_blob_bottom_broadcast_1;
        Blob<T> m_blob_bottom_broadcast_2;
        Blob<T> m_blob_bottom_scale;

        public ScaleLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_blob_bottom_eltwise = new Blob<T>(m_cuda, m_log, Bottom);
            m_blob_bottom_broadcast_0 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_broadcast_1 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_broadcast_2 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom_scale = new Blob<T>(m_cuda, m_log, new List<int>());
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
            filler.Fill(m_blob_bottom_scale);
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
            m_blob_bottom_scale.Dispose();
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

        public Blob<T> BottomScale
        {
            get { return m_blob_bottom_scale; }
        }

        public void TestForwardEltwise()
        {
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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
                double dfInData = rgInDataA[i] * rgInDataB[i];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestForwardEltwiseInPlace()
        {
            TopVec[0] = Bottom; // in-place computation.
            Blob<T> blobOriginalBottom = new Blob<T>(m_cuda, m_log, Bottom.shape());
            blobOriginalBottom.CopyFrom(Bottom);
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Bottom.update_cpu_data());
            int nCount = Bottom.count();
            double[] rgInDataA = convert(blobOriginalBottom.update_cpu_data());
            double[] rgInDataB = convert(BottomEltwise.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] * rgInDataB[i];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestBackwardEltwiseInPlace()
        {
            Blob<T> blobOriginalBottom = new Blob<T>(m_cuda, m_log, Bottom.shape());
            blobOriginalBottom.CopyFrom(Bottom);
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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
            Blob<T> blobOriginalScaleDiff = new Blob<T>(m_cuda, m_log);
            blobOriginalScaleDiff.CopyFrom(BottomEltwise, kCopyDiff, kReshape);

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

            double[] rgOriginalScaleDiff = convert(blobOriginalScaleDiff.update_cpu_diff());
            double[] rgBottomScaleDiff = convert(BottomEltwise.update_cpu_diff());

            for (int i = 0; i < BottomEltwise.count(); i++)
            {
                double dfOriginalScaleDiff = rgOriginalScaleDiff[i];
                double dfBottomScaleDiff = rgBottomScaleDiff[i];

                m_log.EXPECT_NEAR(dfOriginalScaleDiff, dfBottomScaleDiff, 1e-5);
            }
        }

        public void TestForwardEltwiseWithParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            p.scale_param.num_axes = -1;
            p.scale_param.filler = new FillerParameter("gaussian");
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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
                double dfInData = rgInDataA[i] * rgInDataB[i];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestForwardBroadcastBegin()
        {
            BottomVec.Add(BottomBroadcast0);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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

                            m_log.EXPECT_NEAR(dfTop, dfBottom * dfBroadcast, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardBroadcastMiddle()
        {
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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

                            m_log.EXPECT_NEAR(dfTop, dfBottom * dfBroadcast, 1e-5);
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

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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

                            m_log.EXPECT_NEAR(dfTop, dfBottom * dfBroadcast, 1e-5);
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

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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
            Blob<T> blobOriginalScaleDiff = new Blob<T>(m_cuda, m_log);
            blobOriginalScaleDiff.CopyFrom(BottomBroadcast1, kCopyDiff, kReshape);

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

            double[] rgOriginalScaleDiff = convert(blobOriginalScaleDiff.update_cpu_diff());
            double[] rgBottomBroadcastDiff = convert(BottomBroadcast1.update_cpu_diff());

            for (int i = 0; i < BottomBroadcast1.count(); i++)
            {
                double dfOriginalScaleDiff = rgOriginalScaleDiff[i];
                double dfBottomBroadcastDiff = rgBottomBroadcastDiff[i];

                m_log.EXPECT_NEAR(dfOriginalScaleDiff, dfBottomBroadcastDiff, 1e-5);
            }
        }

        public void TestForwardBroadcastMiddleWithParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            p.scale_param.num_axes = 2;
            p.scale_param.filler = new FillerParameter("gaussian");
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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

                            m_log.EXPECT_NEAR(dfTop, dfBottom * dfParam, 1e-5);
                        }
                    }
                }
            }
        }
        public void TestForwardBroadcastMiddleWithParamAndBias()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            p.scale_param.num_axes = 2;
            p.scale_param.filler = new FillerParameter("gaussian");
            p.scale_param.bias_term = true;
            p.scale_param.bias_filler = new FillerParameter("gaussian");
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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
                            double dfParam1 = convert(layer.blobs[0].data_at(c, h, 0, 0));
                            double dfParam2 = convert(layer.blobs[1].data_at(c, h, 0, 0));

                            m_log.EXPECT_NEAR(dfTop, dfBottom * dfParam1 + dfParam2, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardBroadcastEnd()
        {
            BottomVec.Add(BottomBroadcast2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 2;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

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

                            m_log.EXPECT_NEAR(dfTop, dfBottom * dfBroadcast, 1e-5);
                        }
                    }
                }
            }
        }

        public void TestForwardScale()
        {
            BottomVec.Add(BottomScale);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());
            int nCount = Top.count();
            double[] rgInDataA = convert(Bottom.update_cpu_data());
            double[] rgScale = convert(BottomScale.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] * rgScale[0];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestForwardScaleAxis2()
        {
            BottomVec.Add(BottomScale);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 2;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should have the same shape.");
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());
            int nCount = Top.count();
            double[] rgInDataA = convert(Bottom.update_cpu_data());
            double[] rgScale = convert(BottomScale.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                double dfData = rgTopData[i];
                double dfInData = rgInDataA[i] * rgScale[0];

                m_log.EXPECT_NEAR(dfData, dfInData, 1e-5);
            }
        }

        public void TestGradientEltwise()
        {
            BottomVec.Add(BottomEltwise);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestGradientEltwiseWithParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            p.scale_param.num_axes = -1;
            p.scale_param.filler = new FillerParameter("gaussian");
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastBegin()
        {
            BottomVec.Add(BottomBroadcast0);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 0;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastMiddle()
        {
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastMiddleWithParam()
        {
            BottomVec.Add(BottomBroadcast1);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 1;
            p.scale_param.num_axes = 2;
            p.scale_param.filler = new FillerParameter("gaussian");
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientBroadcastEnd()
        {
            BottomVec.Add(BottomBroadcast2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 2;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientScale()
        {
            BottomVec.Add(BottomScale);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientScaleAndBias()
        {
            BottomVec.Add(BottomScale);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.bias_term = true;
            p.scale_param.bias_filler = new FillerParameter("gaussian");
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientScaleAxis2()
        {
            BottomVec.Add(BottomScale);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SCALE);
            p.scale_param.axis = 2;
            ScaleLayer<T> layer = new ScaleLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
