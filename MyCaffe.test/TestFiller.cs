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
    public class TestFiller
    {
        [TestMethod]
        public void TestConstantFiller()
        {
            ConstantFillerTest test = new ConstantFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestUniformFiller()
        {
            UniformFillerTest test = new UniformFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPositiveUnitballFiller()
        {
            PositiveUnitballFillerTest test = new PositiveUnitballFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGaussianFiller()
        {
            GaussianFillerTest test = new GaussianFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestXavierFillerFanIn()
        {
            XavierFillerTest test = new XavierFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    double dfN = 2 * 4 * 5;
                    t.TestFill(FillerParameter.VarianceNorm.FAN_IN, dfN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestXavierFillerFanOut()
        {
            XavierFillerTest test = new XavierFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    double dfN = 1000 * 4 * 5;
                    t.TestFill(FillerParameter.VarianceNorm.FAN_OUT, dfN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestXavierFillerAverage()
        {
            XavierFillerTest test = new XavierFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    double dfN = (2 * 4 * 5 + 1000 * 4 * 5) / 2.0;
                    t.TestFill(FillerParameter.VarianceNorm.AVERAGE, dfN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMSRAFillerFanIn()
        {
            MSRAFillerTest test = new MSRAFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    double dfN = 2 * 4 * 5;
                    t.TestFill(FillerParameter.VarianceNorm.FAN_IN, dfN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMSRAFillerFanOut()
        {
            MSRAFillerTest test = new MSRAFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    double dfN = 1000 * 4 * 5;
                    t.TestFill(FillerParameter.VarianceNorm.FAN_OUT, dfN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMSRAFillerAverage()
        {
            MSRAFillerTest test = new MSRAFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    double dfN = (2 * 4 * 5 + 1000 * 4 * 5) / 2.0;
                    t.TestFill(FillerParameter.VarianceNorm.AVERAGE, dfN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IFillerTest : ITest
    {
        void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0);
    }

    class ConstantFillerTest : TestBase
    {
        public ConstantFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Constant Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ConstantFillerTest<double>(strName, nDeviceID, engine);
            else
                return new ConstantFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ConstantFillerTest<T> : TestEx<T>, IFillerTest
    {
        FillerParameter m_fp;

        public ConstantFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            m_fp = new FillerParameter("constant", 10.0);
            return m_fp;
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_GE(rgData[i], m_fp.value, "The filler value should be equal to " + m_fp.value.ToString());
            }
        }
    }


    class UniformFillerTest : TestBase
    {
        public UniformFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Uniform Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new UniformFillerTest<double>(strName, nDeviceID, engine);
            else
                return new UniformFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class UniformFillerTest<T> : TestEx<T>, IFillerTest
    {
        FillerParameter m_fp;

        public UniformFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            m_fp = new FillerParameter("uniform");
            m_fp.min = 1.0;
            m_fp.max = 2.0;
            return m_fp;
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_GE(rgData[i], m_fp.min, "The filler value should be greater than or equal to " + m_fp.min.ToString());
                m_log.CHECK_LE(rgData[i], m_fp.max, "The filler value should be less than or equal to " + m_fp.max.ToString());
            }
        }
    }


    class PositiveUnitballFillerTest : TestBase
    {
        public PositiveUnitballFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PositiveUnitball Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PositiveUnitballFillerTest<double>(strName, nDeviceID, engine);
            else
                return new PositiveUnitballFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PositiveUnitballFillerTest<T> : TestEx<T>, IFillerTest
    {
        FillerParameter m_fp;

        public PositiveUnitballFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            m_fp = new FillerParameter("positive_unitball");
            return m_fp;
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            int nNum = Bottom.num;
            int nCount = Bottom.count();
            int nDim = nCount / nNum;
            double[] rgData = convert(Bottom.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_GE(rgData[i], 0.0, "The filler value should be greater than or equal to 0.0");
                m_log.CHECK_LE(rgData[i], 1.0, "The filler value should be less than or equal to 1.0");
            }

            for (int i = 0; i < nNum; i++)
            {
                double dfSum = 0;

                for (int j = 0; j < nDim; j++)
                {
                    dfSum += rgData[i * nDim + j];
                }

                m_log.CHECK_GE(dfSum, 0.999, "The sum should be greater than or equal to 0.999");
                m_log.CHECK_LE(dfSum, 1.001, "The sum should be less than or equal to 1.0001");
            }
        }
    }


    class GaussianFillerTest : TestBase
    {
        public GaussianFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Gaussian Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GaussianFillerTest<double>(strName, nDeviceID, engine);
            else
                return new GaussianFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class GaussianFillerTest<T> : TestEx<T>, IFillerTest
    {
        FillerParameter m_fp;

        public GaussianFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            m_fp = new FillerParameter("gaussian");
            m_fp.mean = 10.0;
            m_fp.std = 0.1;
            return m_fp;
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());
            double dfMean = 0;
            double dfVar = 0;

            for (int i = 0; i < nCount; i++)
            {
                dfMean += rgData[i];
                dfVar += (rgData[i] - m_fp.mean) * (rgData[i] - m_fp.mean);
            }

            dfMean /= nCount;
            dfVar /= nCount;

            // Very loose test.
            m_log.CHECK_GE(dfMean, m_fp.mean - m_fp.std * 5, "The mean is not as expected.");
            m_log.CHECK_LE(dfMean, m_fp.mean + m_fp.std * 5, "The mean is not as expected.");
            double dfTarget = m_fp.std * m_fp.std;
            m_log.CHECK_GE(dfVar, dfTarget / 5.0, "The variance is not as expected.");
            m_log.CHECK_LE(dfVar, dfTarget * 5.0, "The variance is not as expected.");
        }
    }


    class XavierFillerTest : TestBase
    {
        public XavierFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Xavier Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new XavierFillerTest<double>(strName, nDeviceID, engine);
            else
                return new XavierFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class XavierFillerTest<T> : TestEx<T>, IFillerTest
    {
        FillerParameter m_fp;

        public XavierFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 1000, 2, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            m_fp = new FillerParameter("xavier");
            m_fp.variance_norm = varNorm;

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            filler.Fill(Bottom);

            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());
            double dfMean = 0;
            double dfEx2 = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                dfMean += rgData[i];
                dfEx2 += rgData[i] * rgData[i];
            }

            dfMean /= nCount;
            dfEx2 /= nCount;

            double dfStd = Math.Sqrt(dfEx2 - dfMean * dfMean);
            double dfTargetStd = Math.Sqrt(2.0 / dfN);

            m_log.EXPECT_NEAR(dfMean, 0, 0.1);
            m_log.EXPECT_NEAR(dfStd, dfTargetStd, 0.1);
        }
    }


    class MSRAFillerTest : TestBase
    {
        public MSRAFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MSRA Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MSRAFillerTest<double>(strName, nDeviceID, engine);
            else
                return new MSRAFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MSRAFillerTest<T> : TestEx<T>, IFillerTest
    {
        FillerParameter m_fp;

        public MSRAFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 1000, 2, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            m_fp = new FillerParameter("msra");
            m_fp.variance_norm = varNorm;

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            filler.Fill(Bottom);

            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());
            double dfMean = 0;
            double dfEx2 = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                dfMean += rgData[i];
                dfEx2 += rgData[i] * rgData[i];
            }

            dfMean /= nCount;
            dfEx2 /= nCount;

            double dfStd = Math.Sqrt(dfEx2 - dfMean * dfMean);
            double dfTargetStd = Math.Sqrt(2.0 / dfN);

            m_log.EXPECT_NEAR(dfMean, 0, 0.1);
            m_log.EXPECT_NEAR(dfStd, dfTargetStd, 0.1);
        }
    }
}
