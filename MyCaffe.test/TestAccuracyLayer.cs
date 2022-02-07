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

namespace MyCaffe.test
{
    [TestClass]
    public class TestAccuracyLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
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
        public void TestSetupTopK()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestSetupTopK();
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
        public void TestSetupOutputPerClass()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestSetupOutputPerClass();
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
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
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
        public void TestForwardWithSpatialAxes()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestForwardWithSpatialAxes();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardIgnoreLabel()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestForwardIgnoreLabel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTopK()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestForwardTopK();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPerClass()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestForwardPerClass();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPerClassWithIgnoreLabel()
        {
            AccuracyLayerTest test = new AccuracyLayerTest();

            try
            {
                foreach (IAccuracyLayerTest t in test.Tests)
                {
                    t.TestForwardPerClassWithIgnoreLabel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IAccuracyLayerTest : ITest
    {
        void TestSetup();
        void TestSetupTopK();
        void TestSetupOutputPerClass();
        void TestForward();
        void TestForwardWithSpatialAxes();
        void TestForwardIgnoreLabel();
        void TestForwardTopK();
        void TestForwardPerClass();
        void TestForwardPerClassWithIgnoreLabel();
    }

    class AccuracyLayerTest : TestBase
    {
        public AccuracyLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Accuracy Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new AccuracyLayerTest<double>(strName, nDeviceID, engine);
            else
                return new AccuracyLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class AccuracyLayerTest<T> : TestEx<T>, IAccuracyLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_label;
        Blob<T> m_blob_top_per_class;
        BlobCollection<T> m_colTopPerClass = new BlobCollection<T>();
        int m_nTopK = 3;

        public AccuracyLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom_label = new Blob<T>(m_cuda, m_log);
            m_blob_top_per_class = new Blob<T>(m_cuda, m_log);

            List<int> rgShape = new List<int>() { 100, 10 };
            Bottom.Reshape(rgShape);
            rgShape.RemoveAt(1);
            m_blob_bottom_label.Reshape(rgShape);

            FillBottoms();

            BottomVec.Add(m_blob_bottom_label);
            m_colTopPerClass.Add(Top);
            m_colTopPerClass.Add(m_blob_top_per_class);
        }

        public void FillBottoms()
        {
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(Bottom);

            double[] rgLabelData = convert(m_blob_bottom_label.mutable_cpu_data);

            for (int i = 0; i < m_blob_bottom_label.count(); i++)
            {
                rgLabelData[i] = m_random.Next(int.MaxValue) % 10;
            }

            m_blob_bottom_label.mutable_cpu_data = convert(rgLabelData);
        }

        protected override void dispose()
        {
            m_blob_bottom_label.Dispose();
            m_blob_top_per_class.Dispose();
            base.dispose();
        }

        public Blob<T> BottomLabel
        {
            get { return m_blob_bottom_label; }
        }

        public Blob<T> TopPerClass
        {
            get { return m_blob_top_per_class; }
        }

        public BlobCollection<T> TopPerClassVec
        {
            get { return m_colTopPerClass; }
        }

        public int TopK
        {
            get { return m_nTopK; }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(1, Top.num, "The top num should equal 1.");
                m_log.CHECK_EQ(1, Top.channels, "The top channels should equal 1.");
                m_log.CHECK_EQ(1, Top.height, "The top height should equal 1.");
                m_log.CHECK_EQ(1, Top.width, "The top width should equal 1.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupTopK()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            p.accuracy_param.top_k = 5;
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(1, Top.num, "The top num should equal 1.");
                m_log.CHECK_EQ(1, Top.channels, "The top channels should equal 1.");
                m_log.CHECK_EQ(1, Top.height, "The top height should equal 1.");
                m_log.CHECK_EQ(1, Top.width, "The top width should equal 1.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupOutputPerClass()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopPerClassVec);

                m_log.CHECK_EQ(1, Top.num, "The top num should equal 1.");
                m_log.CHECK_EQ(1, Top.channels, "The top channels should equal 1.");
                m_log.CHECK_EQ(1, Top.height, "The top height should equal 1.");
                m_log.CHECK_EQ(1, Top.width, "The top width should equal 1.");
                m_log.CHECK_EQ(10, TopPerClass.num, "The top per class num should equal 10.");
                m_log.CHECK_EQ(1, TopPerClass.channels, "The top per class channels should equal 1.");
                m_log.CHECK_EQ(1, TopPerClass.height, "The top per class height should equal 1.");
                m_log.CHECK_EQ(1, TopPerClass.width, "The top per class width should equal 1.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                // repeat the forward
                for (int iter = 0; iter < 3; iter++)
                {
                    layer.Forward(BottomVec, TopVec);

                    double dfMaxVal;
                    int nMaxId;
                    int nNumCorrectLabels = 0;

                    for (int i = 0; i < 100; i++)
                    {
                        dfMaxVal = -double.MaxValue;
                        nMaxId = 0;

                        for (int j = 0; j < 10; j++)
                        {
                            double dfBottom = convert(Bottom.data_at(i, j, 0, 0));

                            if (dfBottom > dfMaxVal)
                            {
                                dfMaxVal = dfBottom;
                                nMaxId = j;
                            }
                        }

                        int nIdx = (int)convert(BottomLabel.data_at(i, 0, 0, 0));
                        if (nMaxId == nIdx)
                            nNumCorrectLabels++;
                    }

                    double dfTop = convert(Top.data_at(0, 0, 0, 0));
                    double dfExpected = nNumCorrectLabels / 100.0;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardWithSpatialAxes()
        {
            Bottom.Reshape(2, 10, 4, 5);
            List<int> rgLabelShape = new List<int>() { 2, 4, 5 };
            BottomLabel.Reshape(rgLabelShape);
            FillBottoms();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            p.accuracy_param.axis = 1;
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                // repeat the forward
                for (int iter = 0; iter < 3; iter++)
                {
                    layer.Forward(BottomVec, TopVec);

                    double dfMaxVal;
                    int nNumLabels = BottomLabel.count();
                    int nMaxId;
                    int nNumCorrectLabels = 0;
                    int[] rgLabelOffset = new int[3];

                    for (int n = 0; n < Bottom.num; n++)
                    {
                        for (int h = 0; h < Bottom.height; h++)
                        {
                            for (int w = 0; w < Bottom.width; w++)
                            {
                                dfMaxVal = -double.MaxValue;
                                nMaxId = 0;

                                for (int c = 0; c < Bottom.channels; c++)
                                {
                                    double dfPredVal = convert(Bottom.data_at(n, c, h, w));

                                    if (dfPredVal > dfMaxVal)
                                    {
                                        dfMaxVal = dfPredVal;
                                        nMaxId = c;
                                    }
                                }

                                rgLabelOffset[0] = n;
                                rgLabelOffset[1] = h;
                                rgLabelOffset[2] = w;

                                int nCorrectLabel = (int)convert(BottomLabel.data_at(new List<int>(rgLabelOffset)));

                                if (nMaxId == nCorrectLabel)
                                    nNumCorrectLabels++;
                            }
                        }
                    }

                    double dfTop = convert(Top.data_at(0, 0, 0, 0));
                    double dfExpected = nNumCorrectLabels / (double)nNumLabels;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardIgnoreLabel()
        {
            int kIgnoreLabelValue = -1;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            p.accuracy_param.ignore_label = kIgnoreLabelValue;
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                // Manually set some labels to the ignore lable value (-1).
                BottomLabel.SetData(kIgnoreLabelValue, 2);
                BottomLabel.SetData(kIgnoreLabelValue, 5);
                BottomLabel.SetData(kIgnoreLabelValue, 32);

                layer.Setup(BottomVec, TopVec);

                // repeat the forward
                for (int iter = 0; iter < 3; iter++)
                {
                    layer.Forward(BottomVec, TopVec);

                    double dfMaxVal;
                    int nMaxId;
                    int nNumCorrectLabels = 0;
                    int nCount = 0;

                    for (int i = 0; i < 100; i++)
                    {
                        double dfBottomLabel = convert(BottomLabel.data_at(i, 0, 0, 0));

                        if (kIgnoreLabelValue == (int)dfBottomLabel)
                            continue;

                        nCount++;
                        dfMaxVal = -double.MaxValue;
                        nMaxId = 0;

                        for (int j = 0; j < 10; j++)
                        {
                            double dfBottom = convert(Bottom.data_at(i, j, 0, 0));

                            if (dfBottom > dfMaxVal)
                            {
                                dfMaxVal = dfBottom;
                                nMaxId = j;
                            }
                        }

                        int nIdx = (int)convert(BottomLabel.data_at(i, 0, 0, 0));
                        if (nMaxId == nIdx)
                            nNumCorrectLabels++;
                    }

                    m_log.CHECK_EQ(nCount, 97, "Expected to count 97 tests.");

                    double dfTop = convert(Top.data_at(0, 0, 0, 0));
                    double dfExpected = nNumCorrectLabels / (double)nCount;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardTopK()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            p.accuracy_param.top_k = (uint)TopK;
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                // repeat the forward
                for (int iter = 0; iter < 3; iter++)
                {
                    layer.Forward(BottomVec, TopVec);

                    double dfCurrentValue;
                    int nCurrentRank;
                    int nNumCorrectLabels = 0;

                    for (int i = 0; i < 100; i++)
                    {
                        for (int j = 0; j < 10; j++)
                        {
                            dfCurrentValue = convert(Bottom.data_at(i, j, 0, 0));
                            nCurrentRank = 0;

                            for (int k = 0; k < 10; k++)
                            {
                                double dfBottom = convert(Bottom.data_at(i, k, 0, 0));

                                if (dfBottom > dfCurrentValue)
                                    nCurrentRank++;
                            }

                            int nIdx = (int)convert(BottomLabel.data_at(i, 0, 0, 0));

                            if (nCurrentRank < TopK && j == nIdx)
                                nNumCorrectLabels++;
                        }
                    }

                    double dfTop = convert(Top.data_at(0, 0, 0, 0));
                    double dfExpected = nNumCorrectLabels / (double)100.0;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardPerClass()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopPerClassVec);

                // repeat the forward
                for (int iter = 0; iter < 3; iter++)
                {
                    layer.Forward(BottomVec, TopPerClassVec);

                    double dfMaxVal;
                    int nMaxId;
                    int nNumCorrectLabels = 0;
                    int nNumClass = TopPerClass.num;
                    int[] rgCorrectPerClass = new int[nNumClass];
                    int[] rgNumPerClass = new int[nNumClass];

                    for (int i = 0; i < 100; i++)
                    {
                        dfMaxVal = -double.MaxValue;
                        nMaxId = 0;

                        for (int j = 0; j < 10; j++)
                        {
                            double dfBottom = convert(Bottom.data_at(i, j, 0, 0));

                            if (dfBottom > dfMaxVal)
                            {
                                dfMaxVal = dfBottom;
                                nMaxId = j;
                            }
                        }

                        int nIdx = (int)convert(BottomLabel.data_at(i, 0, 0, 0));
                        rgNumPerClass[nIdx]++;

                        if (nMaxId == nIdx)
                        {
                            rgCorrectPerClass[nIdx]++;
                            nNumCorrectLabels++;
                        }
                    }

                    double dfTop = convert(Top.data_at(0, 0, 0, 0));
                    double dfExpected = nNumCorrectLabels / 100.0;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);

                    for (int i = 0; i < nNumClass; i++)
                    {
                        double dfAccuracyPerClass = 0;

                        if (rgNumPerClass[i] > 0)
                            dfAccuracyPerClass = rgCorrectPerClass[i] / (double)rgNumPerClass[i];

                        double dfTopPerClass = convert(TopPerClass.data_at(i, 0, 0, 0));

                        m_log.EXPECT_NEAR(dfTopPerClass, dfAccuracyPerClass, 1e-4);
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardPerClassWithIgnoreLabel()
        {
            int kIgnoreLabelValue = -1;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            p.accuracy_param.ignore_label = kIgnoreLabelValue;
            AccuracyLayer<T> layer = new AccuracyLayer<T>(m_cuda, m_log, p);

            try
            {
                // Manually set some labels to the ignore lable value (-1).
                BottomLabel.SetData(kIgnoreLabelValue, 2);
                BottomLabel.SetData(kIgnoreLabelValue, 5);
                BottomLabel.SetData(kIgnoreLabelValue, 32);

                layer.Setup(BottomVec, TopPerClassVec);

                // repeat the forward
                for (int iter = 0; iter < 3; iter++)
                {
                    layer.Forward(BottomVec, TopPerClassVec);

                    double dfMaxVal;
                    int nMaxId;
                    int nNumCorrectLabels = 0;
                    int nNumClass = TopPerClass.num;
                    int[] rgCorrectPerClass = new int[nNumClass];
                    int[] rgNumPerClass = new int[nNumClass];
                    int nCount = 0;

                    for (int i = 0; i < 100; i++)
                    {
                        double dfBottomLabel = convert(BottomLabel.data_at(i, 0, 0, 0));

                        if (kIgnoreLabelValue == (int)dfBottomLabel)
                            continue;

                        nCount++;
                        dfMaxVal = -double.MaxValue;
                        nMaxId = 0;

                        for (int j = 0; j < 10; j++)
                        {
                            double dfBottom = convert(Bottom.data_at(i, j, 0, 0));

                            if (dfBottom > dfMaxVal)
                            {
                                dfMaxVal = dfBottom;
                                nMaxId = j;
                            }
                        }

                        int nIdx = (int)convert(BottomLabel.data_at(i, 0, 0, 0));
                        rgNumPerClass[nIdx]++;

                        if (nMaxId == nIdx)
                        {
                            rgCorrectPerClass[nIdx]++;
                            nNumCorrectLabels++;
                        }
                    }

                    m_log.CHECK_EQ(nCount, 97, "Expected to count 97 tests.");

                    double dfTop = convert(Top.data_at(0, 0, 0, 0));
                    double dfExpected = nNumCorrectLabels / (double)nCount;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);

                    for (int i = 0; i < nNumClass; i++)
                    {
                        double dfAccuracyPerClass = 0;

                        if (rgNumPerClass[i] > 0)
                            dfAccuracyPerClass = rgCorrectPerClass[i] / (double)rgNumPerClass[i];

                        double dfTopPerClass = convert(TopPerClass.data_at(i, 0, 0, 0));

                        m_log.EXPECT_NEAR(dfTopPerClass, dfAccuracyPerClass, 1e-4);
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
