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
using MyCaffe.layers.nt;

namespace MyCaffe.test
{
    [TestClass]
    public class TestOneHotLayer
    {
        [TestMethod]
        public void TestForward()
        {
            OneHotLayerTest test = new OneHotLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IOneHotLayerTest t in test.Tests)
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
            OneHotLayerTest test = new OneHotLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IOneHotLayerTest t in test.Tests)
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

    interface IOneHotLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class OneHotLayerTest : TestBase
    {
        public OneHotLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("OneHot Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new OneHotLayerTest<double>(strName, nDeviceID, engine);
            else
                return new OneHotLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class OneHotLayerTest<T> : TestEx<T>, IOneHotLayerTest
    {
        Blob<T> m_blobTemp;

        public OneHotLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobTemp = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = 0;
            fp.max = 1;

            return fp;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ONEHOT);
            p.onehot_param.axis = 2;
            p.onehot_param.num_output = 8;
            p.onehot_param.min = 0;
            p.onehot_param.max = 1;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            layer.Setup(BottomVec, TopVec);

            int nCount1 = TopVec[0].count(p.onehot_param.axis);
            m_log.CHECK_EQ(nCount1, (int)p.onehot_param.num_output, "The top size is incorrect!");

            layer.Forward(BottomVec, TopVec);

            // Test values
            List<double> rgBuckets = new List<double>();
            double dfStep = (p.onehot_param.max - p.onehot_param.min) / p.onehot_param.num_output;
            double dfTotal = p.onehot_param.min;

            for (int i = 0; i < (int)p.onehot_param.num_output; i++)
            {
                dfTotal += dfStep;
                rgBuckets.Add(dfTotal);
            }

            double[] rgOneHot = new double[rgBuckets.Count];
            double[] rgTop = convert(TopVec[0].mutable_cpu_data);
            double[] rgBottom = convert(BottomVec[0].mutable_cpu_data);
            int nCount = BottomVec[0].count(0, p.onehot_param.axis + 1);

            for (int i = 0; i < nCount; i++)
            {
                double dfBottom = rgBottom[i];
                int nExpectedIdx = rgBuckets.Count-1;

                for (int j = 0; j < rgBuckets.Count; j++)
                {
                    if (dfBottom < rgBuckets[j])
                    {
                        nExpectedIdx = j;
                        break;
                    }
                }

                Array.Copy(rgTop, i * rgOneHot.Length, rgOneHot, 0, rgOneHot.Length);

                for (int j = 0; j < rgOneHot.Length; j++)
                {
                    if (j == nExpectedIdx)
                        m_log.CHECK_EQ(rgOneHot[j], 1, "The one hot expected index should be 1!");
                    else
                        m_log.CHECK_EQ(rgOneHot[j], 0, "The one hot expected index should be 0!");
                }
            }
        }

        public void TestGradient()
        {
            RawProto proto = RawProto.Parse("name: \"onehot\" type: \"ONEHOT\" onehot_param { axis: 2 num_output: 8 min: 0 max: 1 }");
            LayerParameter p = LayerParameter.FromProto(proto);
            OneHotLayer<T> layer = new OneHotLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Blob<T> b = new Blob<T>(m_cuda, m_log, TopVec[0]);
            m_filler.Fill(b);
            m_cuda.copy(b.count(), b.gpu_data, TopVec[0].mutable_gpu_diff);

            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            double[] rgBottomDiff = convert(BottomVec[0].mutable_cpu_diff);
            double[] rgTopDiff = convert(TopVec[0].mutable_cpu_diff);
            double[] rgTopData = convert(TopVec[0].mutable_cpu_data);

            int nCount = BottomVec[0].count(0, p.onehot_param.axis);
            int nItemCount = TopVec[0].count() / nCount;

            for (int i = 0; i < nCount; i++)
            {
                double dfDiffSum = 0;

                for (int j = 0; j < nItemCount; j++)
                {
                    if (rgTopData[i * nItemCount + j] == 1.0)
                        dfDiffSum += rgTopDiff[i * nItemCount + j];
                    else
                        dfDiffSum -= rgTopDiff[i * nItemCount + j];
                }

                dfDiffSum /= nItemCount;

                m_log.EXPECT_NEAR(dfDiffSum, rgBottomDiff[i], 0.001, "The bottom diff does not match the expected diff!");
            }
        }
    }
}
