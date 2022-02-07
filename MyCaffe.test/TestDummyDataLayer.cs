using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestDummyDataLayer
    {
        [TestMethod]
        public void TestOneTopConstant()
        {
            DummyDataLayerTest test = new DummyDataLayerTest();

            try
            {
                foreach (IDummyDataLayerTest t in test.Tests)
                {
                    t.TestOneTopConstant();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTwoTopConstant()
        {
            DummyDataLayerTest test = new DummyDataLayerTest();

            try
            {
                foreach (IDummyDataLayerTest t in test.Tests)
                {
                    t.TestTwoTopConstant();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestThreeTopConstantGaussianConstant()
        {
            DummyDataLayerTest test = new DummyDataLayerTest();

            try
            {
                foreach (IDummyDataLayerTest t in test.Tests)
                {
                    t.TestThreeTopConstantGaussianConstant();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IDummyDataLayerTest : ITest
    {
        void TestOneTopConstant();
        void TestTwoTopConstant();
        void TestThreeTopConstantGaussianConstant();
    }

    class DummyDataLayerTest : TestBase
    {
        public DummyDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("DummyData Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DummyDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new DummyDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class DummyDataLayerTest<T> : TestEx<T>, IDummyDataLayerTest
    {
        Blob<T> m_blob_top_b;
        Blob<T> m_blob_top_c;

        public DummyDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_top_b = new Blob<T>(m_cuda, m_log);
            m_blob_top_c = new Blob<T>(m_cuda, m_log);

            TopVec.Add(m_blob_top_b);
            TopVec.Add(m_blob_top_c);

            BottomVec.Clear();
        }

        protected override void dispose()
        {
            m_blob_top_b.Dispose();
            m_blob_top_c.Dispose();
            base.dispose();
        }

        public Blob<T> TopB
        {
            get { return m_blob_top_b; }
        }

        public Blob<T> TopC
        {
            get { return m_blob_top_c; }
        }

        public void TestOneTopConstant()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DUMMYDATA);
            p.dummy_data_param.num.Add(5);
            p.dummy_data_param.channels.Add(3);
            p.dummy_data_param.height.Add(2);
            p.dummy_data_param.width.Add(4);

            TopVec.RemoveAt(TopVec.Count - 1);
            TopVec.RemoveAt(TopVec.Count - 1);

            DummyDataLayer<T> layer = new DummyDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(5, Top.num, "The top num should equal 5.");
                m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
                m_log.CHECK_EQ(2, Top.height, "The top height should equal 2.");
                m_log.CHECK_EQ(4, Top.width, "The top width should equal 4.");
                m_log.CHECK_EQ(0, TopB.count(), "The top_b should have count() = 0.");
                m_log.CHECK_EQ(0, TopC.count(), "The top_c should have count() = 0.");

                for (int i = 0; i < TopVec.Count; i++)
                {
                    double[] rgTopData = convert(TopVec[i].update_cpu_data());

                    for (int j = 0; j < TopVec[i].count(); j++)
                    {
                        m_log.CHECK_EQ(0, rgTopData[j], "The value at index " + j.ToString() + " should be 0.");
                    }
                }

                layer.Forward(BottomVec, TopVec);

                for (int i = 0; i < TopVec.Count; i++)
                {
                    double[] rgTopData = convert(TopVec[i].update_cpu_data());

                    for (int j = 0; j < TopVec[i].count(); j++)
                    {
                        m_log.CHECK_EQ(0, rgTopData[j], "The value at index " + j.ToString() + " should be 0.");
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestTwoTopConstant()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DUMMYDATA);
            p.dummy_data_param.num.Add(5);
            p.dummy_data_param.channels.Add(3);
            p.dummy_data_param.height.Add(2);
            p.dummy_data_param.width.Add(4);
            p.dummy_data_param.num.Add(5);
            // Don't explicitly set number of channels or height for 2nd top blob; should
            // default to first channels and height (as we check later).
            p.dummy_data_param.height.Add(1);
            FillerParameter data_filler_param = new FillerParameter("constant");
            data_filler_param.value = 7;
            p.dummy_data_param.data_filler.Add(data_filler_param);

            TopVec.RemoveAt(TopVec.Count - 1);

            DummyDataLayer<T> layer = new DummyDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(5, Top.num, "The top num should equal 5.");
                m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
                m_log.CHECK_EQ(2, Top.height, "The top height should equal 2.");
                m_log.CHECK_EQ(4, Top.width, "The top width should equal 4.");
                m_log.CHECK_EQ(5, TopB.num, "The top_b num should equal 5.");
                m_log.CHECK_EQ(3, TopB.channels, "The top_b channels should equal 3.");
                m_log.CHECK_EQ(1, TopB.height, "The top_b height should equal 1.");
                m_log.CHECK_EQ(4, TopB.width, "The top_b width should equal 4.");
                m_log.CHECK_EQ(0, TopC.count(), "The top_c should have count() = 0.");

                for (int i = 0; i < TopVec.Count; i++)
                {
                    double[] rgTopData = convert(TopVec[i].update_cpu_data());

                    for (int j = 0; j < TopVec[i].count(); j++)
                    {
                        m_log.CHECK_EQ(7, rgTopData[j], "The value at index " + j.ToString() + " should be 0.");
                    }
                }

                layer.Forward(BottomVec, TopVec);

                for (int i = 0; i < TopVec.Count; i++)
                {
                    double[] rgTopData = convert(TopVec[i].update_cpu_data());

                    for (int j = 0; j < TopVec[i].count(); j++)
                    {
                        m_log.CHECK_EQ(7, rgTopData[j], "The value at index " + j.ToString() + " should be 0.");
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestThreeTopConstantGaussianConstant()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DUMMYDATA);
            p.dummy_data_param.num.Add(5);
            p.dummy_data_param.channels.Add(3);
            p.dummy_data_param.height.Add(2);
            p.dummy_data_param.width.Add(4);            
            FillerParameter data_filler_param_a = new FillerParameter("constant");
            data_filler_param_a.value = 7;
            p.dummy_data_param.data_filler.Add(data_filler_param_a);
            FillerParameter data_filler_param_b = new FillerParameter("gaussian");
            double gaussian_mean = 3.0;
            double gaussian_std = 0.01;
            data_filler_param_b.mean = gaussian_mean;
            data_filler_param_b.std = gaussian_std;
            p.dummy_data_param.data_filler.Add(data_filler_param_b);
            FillerParameter data_filler_param_c = new FillerParameter("constant");
            data_filler_param_c.value = 9;
            p.dummy_data_param.data_filler.Add(data_filler_param_c);

            DummyDataLayer<T> layer = new DummyDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(5, Top.num, "The top num should equal 5.");
                m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
                m_log.CHECK_EQ(2, Top.height, "The top height should equal 2.");
                m_log.CHECK_EQ(4, Top.width, "The top width should equal 4.");
                m_log.CHECK_EQ(5, TopB.num, "The top_b num should equal 5.");
                m_log.CHECK_EQ(3, TopB.channels, "The top_b channels should equal 3.");
                m_log.CHECK_EQ(2, TopB.height, "The top_b height should equal 2.");
                m_log.CHECK_EQ(4, TopB.width, "The top_b width should equal 4.");
                m_log.CHECK_EQ(5, TopC.num, "The top_c num should equal 5.");
                m_log.CHECK_EQ(3, TopC.channels, "The top_c channels should equal 3.");
                m_log.CHECK_EQ(2, TopC.height, "The top_c height should equal 2.");
                m_log.CHECK_EQ(4, TopC.width, "The top_c width should equal 4.");

                double[] rgTopA = convert(Top.update_cpu_data());

                for (int i = 0; i < Top.count(); i++)
                {
                    m_log.CHECK_EQ(7, rgTopA[i], "The top_a value at " + i.ToString() + " should be 7.");
                }

                // Blob b uses a Gaussian filler, so Setup shoudl not hvae initialized it.
                // Blob b's data should threfore be the default Blob data value: 0

                double[] rgTopB = convert(TopB.update_cpu_data());

                for (int i = 0; i < TopB.count(); i++)
                {
                    m_log.CHECK_EQ(0, rgTopB[i], "The top_b value at " + i.ToString() + " should be 0.");
                }

                double[] rgTopC = convert(TopC.update_cpu_data());

                for (int i = 0; i < TopC.count(); i++)
                {
                    m_log.CHECK_EQ(9, rgTopC[i], "The top_c value at " + i.ToString() + " should be 9.");
                }

                // Do a Forward pass to fill in blob b with Gaussian data.
                layer.Forward(BottomVec, TopVec);

                rgTopA = convert(Top.update_cpu_data());

                for (int i = 0; i < Top.count(); i++)
                {
                    m_log.CHECK_EQ(7, rgTopA[i], "The top_a value at " + i.ToString() + " should be 7.");
                }

                // Check that the Gaussian's data has been filled in with values within
                // 10 standard deviations of the mean.  Record the first and last sample
                // to check that they're different after the next Forward pass.
                rgTopB = convert(TopB.update_cpu_data());

                for (int i = 0; i < TopB.count(); i++)
                {
                    m_log.EXPECT_NEAR(gaussian_mean, rgTopB[i], gaussian_std * 10);
                }

                double first_gaussian_sample = rgTopB[0];
                double last_gaussian_sample = rgTopB[TopB.count() - 1];

                rgTopC = convert(TopC.update_cpu_data());

                for (int i = 0; i < TopC.count(); i++)
                {
                    m_log.CHECK_EQ(9, rgTopC[i], "The top_c value at " + i.ToString() + " should be 9.");
                }

                // Do another forward pass to fill in Blob b with Gaussian data again,
                // checking that we get different values.
                layer.Forward(BottomVec, TopVec);

                rgTopA = convert(Top.update_cpu_data());

                for (int i = 0; i < Top.count(); i++)
                {
                    m_log.CHECK_EQ(7, rgTopA[i], "The top_a value at " + i.ToString() + " should be 7.");
                }

                // Check that the Gaussian's data has been filled in with values within
                // 10 standard deviations of the mean.  Record the first and last sample
                // to check that they're different after the next Forward pass.
                rgTopB = convert(TopB.update_cpu_data());

                for (int i = 0; i < TopB.count(); i++)
                {
                    m_log.EXPECT_NEAR(gaussian_mean, rgTopB[i], gaussian_std * 10);
                }

                m_log.CHECK_NE(first_gaussian_sample, rgTopB[0], "The the first data item in TopB should differ between the two forward passes.");
                m_log.CHECK_NE(last_gaussian_sample, rgTopB[TopB.count() - 1], "The the first data item in TopB should differ between the two forward passes.");

                rgTopC = convert(TopC.update_cpu_data());

                for (int i = 0; i < TopC.count(); i++)
                {
                    m_log.CHECK_EQ(9, rgTopC[i], "The top_c value at " + i.ToString() + " should be 9.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
