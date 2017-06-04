using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestFilterLayer
    {
        [TestMethod]
        public void TestReshape()
        {
            FilterLayerTest test = new FilterLayerTest();

            try
            {
                foreach (IFilterLayerTest t in test.Tests)
                {
                    t.TestReshape();
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
            FilterLayerTest test = new FilterLayerTest();

            try
            {
                foreach (IFilterLayerTest t in test.Tests)
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
            FilterLayerTest test = new FilterLayerTest();

            try
            {
                foreach (IFilterLayerTest t in test.Tests)
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


    interface IFilterLayerTest : ITest
    {
        void TestReshape();
        void TestForward();
        void TestGradient();
    }

    class FilterLayerTest : TestBase
    {
        public FilterLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Filter Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new FilterLayerTest<double>(strName, nDeviceID, engine);
            else
                return new FilterLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class FilterLayerTest<T> : TestEx<T>, IFilterLayerTest
    {
        Blob<T> m_blob_bottom_labels;
        Blob<T> m_blob_bottom_selector;
        Blob<T> m_blob_top_labels;

        public FilterLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 4, 3, 6, 4 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_labels = new Blob<T>(m_cuda, m_log, 4, 1, 1, 1);
            m_blob_bottom_selector = new Blob<T>(m_cuda, m_log, 4, 1, 1, 1);
            m_blob_top_labels = new Blob<T>(m_cuda, m_log);

            Setup();
        }

        protected override void dispose()
        {
            if (m_blob_bottom_labels != null)
            {
                m_blob_bottom_labels.Dispose();
                m_blob_bottom_labels = null;
            }

            if (m_blob_bottom_selector != null)
            {
                m_blob_bottom_selector.Dispose();
                m_blob_bottom_selector = null;
            }

            if (m_blob_top_labels != null)
            {
                m_blob_top_labels.Dispose();
                m_blob_top_labels = null;
            }

            base.dispose();
        }

        public Blob<T> TopLabels
        {
            get { return m_blob_top_labels; }
        }

        public Blob<T> BottomLabels
        {
            get { return m_blob_bottom_labels; }
        }

        public Blob<T> BottomSelector
        {
            get { return m_blob_bottom_selector; }
        }

        public void Setup()
        {
            // fill the values
            Random random = new Random(1890);
            m_cuda.rng_setseed(1890);
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            // fill the selector blob.
            double[] rgBottomSelector = convert(m_blob_bottom_selector.mutable_cpu_data);
            rgBottomSelector[0] = 0;
            rgBottomSelector[1] = 1;
            rgBottomSelector[2] = 1;
            rgBottomSelector[3] = 0;
            m_blob_bottom_selector.mutable_cpu_data = convert(rgBottomSelector);

            // fill the other bottom blobs.
            filler.Fill(m_blob_bottom);

            double[] rgBottomLabels = convert(m_blob_bottom_labels.mutable_cpu_data);
            for (int i = 0; i < rgBottomLabels.Length; i++)
            {
                rgBottomLabels[i] = random.Next(5);
            }
            m_blob_bottom_labels.mutable_cpu_data = convert(rgBottomLabels);

            BottomVec.Add(m_blob_bottom_labels);
            BottomVec.Add(m_blob_bottom_selector);
            TopVec.Add(m_blob_top_labels);
        }

        public void TestReshape()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FILTER);
            FilterLayer<T> layer = new FilterLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);

            // In the test first and last items should have been filtered
            // so we just expect 2 remaining items.
            m_log.CHECK_EQ(Top.shape(0), 2, "The top data should have shape(0) = 2.");
            m_log.CHECK_EQ(TopLabels.shape(0), 2, "The top labels should have shape(0) = 2");
            m_log.CHECK_GT(Bottom.shape(0), Top.shape(0), "The bottom.shape(0) should be greater than the top.shape(0).");
            m_log.CHECK_GT(BottomLabels.shape(0), TopLabels.shape(0), "The bottom_labels.shape(0) should be greater than the top_labels.shape(0).");

            for (int i = 1; i < BottomLabels.num_axes; i++)
            {
                m_log.CHECK_EQ(BottomLabels.shape(i), TopLabels.shape(i), "The bottomLabels.shape(" + i.ToString() + ") should be equal to the topLabels.shape(" + i.ToString() + ").");
            }
        }


        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FILTER);
            FilterLayer<T> layer = new FilterLayer<T>(m_cuda, m_log, p);
            
            layer.Setup(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double fTop = convert(TopLabels.data_at(0, 0, 0, 0));
            double fBtm = convert(BottomLabels.data_at(1, 0, 0, 0));

            m_log.CHECK_EQ(fTop, fBtm, "The top data_at(0, 0, 0, 0) should equal bottom data_at(1, 0, 0, 0).");

            fTop = convert(TopLabels.data_at(1, 0, 0, 0));
            fBtm = convert(BottomLabels.data_at(2, 0, 0, 0));

            m_log.CHECK_EQ(fTop, fBtm, "The top data_at(1, 0, 0, 0) should equal bottom data_at(2, 0, 0, 0).");

            int nDim = Top.count() / Top.shape(0);
            int nTopOffset = 0;
            int nBtmOffset = 0;
            double[] rgTopData = convert(Top.update_cpu_data());
            double[] rgBtmData = convert(Bottom.update_cpu_data());

            // selector is 0 1 1 0, so we need to compare bottom(1, c, h, w)
            // with top(0, c, h, w), and bottom(2, c, h, w) with top(1, c, h, w)

            nBtmOffset += nDim; // bottom(1, c, h, w)
            for (int n = 0; n < nDim; n++)
            {
                m_log.CHECK_EQ(rgTopData[n + nTopOffset], rgBtmData[n + nBtmOffset], "The top and bottom values should be the same.");
            }

            nBtmOffset += nDim; // bottom(2, c, h, w)
            nTopOffset += nDim; // top(1, c, h, w)
            for (int n = 0; n < nDim; n++)
            {
                m_log.CHECK_EQ(rgTopData[n + nTopOffset], rgBtmData[n + nBtmOffset], "The top and bottom values should be the same.");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FILTER);
            FilterLayer<T> layer = new FilterLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            // check only input 0 (data) because labels and selector
            // don't need backpropagation
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
