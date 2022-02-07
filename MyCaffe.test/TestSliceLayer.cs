using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSliceLayer
    {
        [TestMethod]
        public void TestSetupNum()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestSetupNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupChannels()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestSetupChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrivialSlice()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestTrivialSlice();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSliceAcrossNum()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestSliceAcrossNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSliceAcrossChannels()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestSliceAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSliceAcrossChannelsWithAxis()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestSliceAcrossChannelsWithAxis();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientTrivial()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestGradientTrivial();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAcrossNum()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestGradientAcrossNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAcrossChannels()
        {
            SliceLayerTest test = new SliceLayerTest();

            try
            {
                foreach (ISliceLayerTest t in test.Tests)
                {
                    t.TestGradientAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ISliceLayerTest : ITest
    {
        void TestSetupNum();
        void TestSetupChannels();
        void TestTrivialSlice();
        void TestSliceAcrossNum();
        void TestSliceAcrossChannels();
        void TestSliceAcrossChannelsWithAxis();
        void TestGradientTrivial();
        void TestGradientAcrossNum();
        void TestGradientAcrossChannels();
    }

    class SliceLayerTest : TestBase
    {
        public SliceLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Slice Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SliceLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SliceLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SliceLayerTest<T> : TestEx<T>, ISliceLayerTest
    {
        Blob<T> m_blob_top_1;
        Blob<T> m_blob_top_2;
        BlobCollection<T> m_colTop1 = new BlobCollection<T>();

        public SliceLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 6, 12, 2, 3 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_top_1 = new Blob<T>(m_cuda, m_log, Bottom);
            m_blob_top_2 = new Blob<T>(m_cuda, m_log, Bottom);

            FillerParameter fp1 = getFillerParam();
            fp1.value = 2.0;
            Filler<T> filler1 = Filler<T>.Create(m_cuda, m_log, fp1);
            filler1.Fill(m_blob_top_1);

            FillerParameter fp2 = getFillerParam();
            fp2.value = 3.0;
            Filler<T> filler2 = Filler<T>.Create(m_cuda, m_log, fp2);
            filler2.Fill(m_blob_top_2);

            TopVec.Add(m_blob_top_1);
            TopVec1.Add(Top);
            TopVec1.Add(m_blob_top_1);
            TopVec1.Add(m_blob_top_2);
        }

        protected override void dispose()
        {
            m_blob_top_1.Dispose();
            m_blob_top_2.Dispose();
            base.dispose();
        }

        public virtual void ReduceBottomBlobSize()
        {
            Bottom.Reshape(4, 5, 2, 2);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(Bottom);
        }

        public Blob<T> Top1
        {
            get { return m_blob_top_1; }
        }

        public Blob<T> Top2
        {
            get { return m_blob_top_2; }
        }

        public BlobCollection<T> TopVec1
        {
            get { return m_colTop1; }
        }

        public void TestSetupNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            p.slice_param.axis = 0;
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec1);

                m_log.CHECK_EQ(3 * Top.num, Bottom.num, "The bottom num should equal 3 * the top num.");
                m_log.CHECK_EQ(Top.num, Top1.num, "The top0 and top1 should have the same num.");
                m_log.CHECK_EQ(Top.num, Top2.num, "The top0 and top2 should have the same num.");
                m_log.CHECK_EQ(Top.channels, Bottom.channels, "The top and bottom should have the same number of channels.");
                m_log.CHECK_EQ(Top.height, Bottom.height, "The top and bottom should have the same height.");
                m_log.CHECK_EQ(Top.width, Bottom.width, "The top and bottom should have the same width.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            p.slice_param.slice_point.Add(3);
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, Bottom.num, "The bottom num should equal the top num.");
                m_log.CHECK_EQ(3, Top.channels, "The top0 should have 3 channels.");
                m_log.CHECK_EQ(9, Top1.channels, "The top0 should have 9 channels.");
                m_log.CHECK_EQ(Bottom.channels, Top.channels + Top1.channels, "The bottom channels should equal the top0 + top1 channels.");
                m_log.CHECK_EQ(Top.height, Bottom.height, "The top and bottom should have the same height.");
                m_log.CHECK_EQ(Top.width, Bottom.width, "The top and bottom should have the same width.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// Test the trivial (single output) 'slice' operation --
        /// should be the identity.
        /// </summary>
        public void TestTrivialSlice()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                TopVec.RemoveAt(1);

                layer.Setup(BottomVec, TopVec);

                m_log.CHECK(Utility.Compare<int>(Bottom.shape(), Top.shape()), "The top and bottom should hav the same shape.");

                double[] rgBottom = convert(Bottom.update_cpu_data());
                double[] rgTop = convert(Top.update_cpu_data());

                for (int i = 0; i < Bottom.count(); i++)
                {
                    double dfBottom = rgBottom[i];
                    double dfTop = rgTop[i];

                    m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values at " + i.ToString() + " should be the same.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSliceAcrossNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            p.slice_param.axis = 0;
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                int nTopNum = Bottom.num / 2;

                m_log.CHECK_EQ(nTopNum, Top.num, "The top nums should match.");
                m_log.CHECK_EQ(nTopNum, Top1.num, "The top nums should match.");

                layer.Forward(BottomVec, TopVec);

                for (int n = 0; n < nTopNum; n++)
                {
                    for (int c = 0; c < Top.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n, c, h, w));
                                double dfTop = convert(Top.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }

                    for (int c = 0; c < Top1.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n + 3, c, h, w));
                                double dfTop = convert(Top1.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSliceAcrossChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            int nSlicePoint0 = 2;
            int nSlicePoint1 = 8;
            p.slice_param.slice_point.Add((uint)nSlicePoint0);
            p.slice_param.slice_point.Add((uint)nSlicePoint1);
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec1);

                m_log.CHECK_EQ(nSlicePoint0, Top.channels, "The top should have 2 channels.");
                m_log.CHECK_EQ(nSlicePoint1 - nSlicePoint0, Top1.channels, "The top1 should have 6 channels.");
                m_log.CHECK_EQ(Bottom.channels - nSlicePoint1, Top2.channels, "The bottom channels - 8 should equal the top2 channels.");

                layer.Forward(BottomVec, TopVec1);

                for (int n = 0; n < Bottom.num; n++)
                {
                    for (int c = 0; c < Top.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n, c, h, w));
                                double dfTop = convert(Top.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }

                    for (int c = 0; c < Top1.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n, c + nSlicePoint0, h, w));
                                double dfTop = convert(Top1.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }

                    for (int c = 0; c < Top2.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n, c + nSlicePoint1, h, w));
                                double dfTop = convert(Top2.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSliceAcrossChannelsWithAxis()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            p.slice_param.axis = 0;
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec1);
                layer.Forward(BottomVec, TopVec1);

                for (int n = 0; n < Top.num; n++)
                {
                    for (int c = 0; c < Top.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n + 0, c, h, w));
                                double dfTop = convert(Top.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }

                    for (int c = 0; c < Top1.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n + 2, c, h, w));
                                double dfTop = convert(Top1.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }

                    for (int c = 0; c < Top2.channels; c++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                double dfBottom = convert(Bottom.data_at(n + 4, c, h, w));
                                double dfTop = convert(Top2.data_at(n, c, h, w));

                                m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom values should be equal.");
                            }
                        }
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// Test the trivial (single output) 'slice' operation --
        /// should be the identity.
        /// </summary>
        public void TestGradientTrivial()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

            try
            {
                TopVec.RemoveAt(1);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientAcrossNum()
        {
            // Gradient checks are slow; reduce blob size.
            ReduceBottomBlobSize();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            p.slice_param.axis = 0;
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

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

        public void TestGradientAcrossChannels()
        {
            // Gradient checks are slow; reduce blob size.
            ReduceBottomBlobSize();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SLICE);
            int nSlicePoint = 4;
            p.slice_param.slice_point.Add((uint)nSlicePoint);
            SliceLayer<T> layer = new SliceLayer<T>(m_cuda, m_log, p);

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
    }
}
