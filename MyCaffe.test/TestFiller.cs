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
        public void TestConstantFiller1D()
        {
            ConstantFillerTest test = new ConstantFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill1D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstantFiller2D()
        {
            ConstantFillerTest test = new ConstantFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstantFiller5D()
        {
            ConstantFillerTest test = new ConstantFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill5D();
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
        public void TestUniformFiller1D()
        {
            UniformFillerTest test = new UniformFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill1D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestUniformFiller2D()
        {
            UniformFillerTest test = new UniformFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestUniformFiller5D()
        {
            UniformFillerTest test = new UniformFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill5D();
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
        public void TestPositiveUnitballFiller1D()
        {
            PositiveUnitballFillerTest test = new PositiveUnitballFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill1D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPositiveUnitballFiller2D()
        {
            PositiveUnitballFillerTest test = new PositiveUnitballFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPositiveUnitballFiller5D()
        {
            PositiveUnitballFillerTest test = new PositiveUnitballFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill5D();
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
        public void TestGaussianFiller1D()
        {
            GaussianFillerTest test = new GaussianFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill1D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGaussianFiller2D()
        {
            GaussianFillerTest test = new GaussianFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGaussianFiller5D()
        {
            GaussianFillerTest test = new GaussianFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill5D();
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
        public void TestXavierFiller1D()
        {
            XavierFillerTest test = new XavierFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill1D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestXavierFiller2D()
        {
            XavierFillerTest test = new XavierFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestXavierFiller5D()
        {
            XavierFillerTest test = new XavierFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill5D();
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

        [TestMethod]
        public void TestMSRAFiller1D()
        {
            MSRAFillerTest test = new MSRAFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill1D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMSRAFiller2D()
        {
            MSRAFillerTest test = new MSRAFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMSRAFiller5D()
        {
            MSRAFillerTest test = new MSRAFillerTest();

            try
            {
                foreach (IFillerTest t in test.Tests)
                {
                    t.TestFill5D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBilinearFillOdd()
        {
            BilinearFillerTest test = new BilinearFillerTest();

            try
            {
                foreach (IFillerTest2 t in test.Tests)
                {
                    t.TestFillOdd();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBilinearFillEven()
        {
            BilinearFillerTest test = new BilinearFillerTest();

            try
            {
                foreach (IFillerTest2 t in test.Tests)
                {
                    t.TestFillEven();
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
        void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0);
        void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0);
        void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0);
    }

    interface IFillerTest2 : ITest
    {
        void TestFillOdd();
        void TestFillEven();
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

        protected virtual void test_params(List<int> rgShape)
        {
            Bottom.Reshape(rgShape);
            m_filler.Fill(Bottom);

            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_EQ(rgData[i], m_fp.value, "The filler value should be equal to " + m_fp.value.ToString());
            }
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5 };
            test_params(rgShape);
        }

        public void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 15 };
            test_params(rgShape);
        }

        public void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 8, 3 };
            test_params(rgShape);
        }

        public void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5, 2 };
            test_params(rgShape);
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

        protected virtual void test_params(List<int> rgShape)
        {
            Bottom.Reshape(rgShape);
            m_filler.Fill(Bottom);

            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_GE(rgData[i], m_fp.min, "The filler value should be greater than or equal to " + m_fp.min.ToString());
                m_log.CHECK_LE(rgData[i], m_fp.max, "The filler value should be less than or equal to " + m_fp.max.ToString());
            }
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5 };
            test_params(rgShape);
        }

        public void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 15 };
            test_params(rgShape);
        }

        public void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 8, 3 };
            test_params(rgShape);
        }

        public void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5, 2 };
            test_params(rgShape);
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

        protected virtual void test_params(List<int> rgShape)
        {
            Bottom.Reshape(rgShape);
            m_filler.Fill(Bottom);

            int nNum = Bottom.shape(0);
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

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5 };
            test_params(rgShape);
        }

        public void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 15 };
            test_params(rgShape);
        }

        public void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 8, 3 };
            test_params(rgShape);
        }

        public void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5, 2 };
            test_params(rgShape);
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

        protected virtual void test_params(List<int> rgShape, double dfTolerance = 5, int nRepetitions = 100)
        {
            // Test for statistical properties should be run multiple times.
            Bottom.Reshape(rgShape);

            for (int i = 0; i < nRepetitions; i++)
            {
                test_params_iter(dfTolerance);
            }
        }

        protected virtual void test_params_iter(double dfTolerance)
        {
            // This test has a configurable tolerance perameter - by default it is
            // equal to 5.0 which is very loose - allowing some tuning (e.g. for tests
            // on smaller blobs the actual variance will be larger than desired, so the
            // tolerance can be increased to account for that.
            m_filler.Fill(Bottom);

            int nCount = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());
            double dfMean = 0;
            double dfVar = 0;

            for (int i = 0; i < nCount; i++)
            {
                dfMean += rgData[i];
                dfVar += rgData[i] * rgData[i];
            }

            dfMean /= nCount;
            dfVar /= nCount;
            dfVar -= dfMean * dfMean;

            m_log.CHECK_GE(dfMean, m_fp.mean - m_fp.std * dfTolerance, "The mean is not as expected.");
            m_log.CHECK_LE(dfMean, m_fp.mean + m_fp.std * dfTolerance, "The mean is not as expected.");
            double dfTarget = m_fp.std * m_fp.std;
            m_log.CHECK_GE(dfVar, dfTarget / dfTolerance, "The variance is not as expected.");
            m_log.CHECK_LE(dfVar, dfTarget * dfTolerance, "The variance is not as expected.");
        }

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5 };
            test_params(rgShape, 3);
        }

        public void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 1, 125 };
            test_params(rgShape, 3);
        }

        public void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 8, 15 };
            test_params(rgShape, 3);
        }

        public void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5, 2 };
            test_params(rgShape, 2);
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

        protected virtual void test_params(FillerParameter.VarianceNorm varNorm, double dfN, List<int> rgShape, int nRepetitions = 100)
        {
            Bottom.Reshape(rgShape);

            for (int i = 0; i < nRepetitions; i++)
            {
                test_params_iter(varNorm, dfN);
            }
        }

        protected virtual void test_params_iter(FillerParameter.VarianceNorm varNorm, double dfN)
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

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 1000, 2, 4, 5 };
            test_params(varNorm, dfN, rgShape);
        }

        public void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 25 };
            Bottom.Reshape(rgShape);
            m_fp = new FillerParameter("xavier");
            m_fp.variance_norm = varNorm;
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            m_filler.Fill(Bottom);
        }

        public void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 8, 3 };
            Bottom.Reshape(rgShape);
            m_fp = new FillerParameter("xavier");
            m_fp.variance_norm = varNorm;
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            m_filler.Fill(Bottom);
        }

        public void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5, 2 };
            Bottom.Reshape(rgShape);
            m_fp = new FillerParameter("xavier");
            m_fp.variance_norm = varNorm;
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            m_filler.Fill(Bottom);
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

        protected virtual void test_params(FillerParameter.VarianceNorm varNorm, double dfN, List<int> rgShape, int nRepetitions = 100)
        {
            Bottom.Reshape(rgShape);

            for (int i = 0; i < nRepetitions; i++)
            {
                test_params_iter(varNorm, dfN);
            }
        }

        protected virtual void test_params_iter(FillerParameter.VarianceNorm varNorm, double dfN)
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

        public void TestFill(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 1000, 2, 4, 5 };
            test_params(varNorm, dfN, rgShape);
        }

        public void TestFill1D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 25 };
            Bottom.Reshape(rgShape);
            m_fp = new FillerParameter("msra");
            m_fp.variance_norm = varNorm;
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            m_filler.Fill(Bottom);
        }

        public void TestFill2D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 8, 3 };
            Bottom.Reshape(rgShape);
            m_fp = new FillerParameter("msra");
            m_fp.variance_norm = varNorm;
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            m_filler.Fill(Bottom);
        }

        public void TestFill5D(FillerParameter.VarianceNorm varNorm = FillerParameter.VarianceNorm.AVERAGE, double dfN = 0)
        {
            List<int> rgShape = new List<int>() { 2, 3, 4, 5, 2 };
            Bottom.Reshape(rgShape);
            m_fp = new FillerParameter("msra");
            m_fp.variance_norm = varNorm;
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            m_filler.Fill(Bottom);
        }
    }

    class BilinearFillerTest : TestBase
    {
        public BilinearFillerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Bilinear Filler Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BilinearFillerTest<double>(strName, nDeviceID, engine);
            else
                return new BilinearFillerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BilinearFillerTest<T> : TestEx<T>, IFillerTest2
    {
        FillerParameter m_fp;

        public BilinearFillerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
        }

        protected virtual void test_params(List<int> rgShape)
        {
            m_fp = new FillerParameter("bilinear");
            m_filler = Filler<T>.Create(m_cuda, m_log, m_fp);

            Bottom.Reshape(rgShape);
            m_filler.Fill(Bottom);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, m_fp);
            filler.Fill(Bottom);

            m_log.CHECK_EQ(Bottom.num_axes, 4, "The bilinear test expects 4 axes.");

            int nOuterNum = Bottom.count(0, 2);
            int nInnerNum = Bottom.count(2, 4);
            double[] rgData = convert(Bottom.update_cpu_data());
            int nF = (int)Math.Ceiling(Bottom.shape(3) / 2.0);
            double dfC = (Bottom.shape(3) - 1.0) / (2.0 * nF);

            for (int i = 0; i < nOuterNum; i++)
            {
                for (int j = 0; j < nInnerNum; j++)
                {
                    double dfX = j % Bottom.shape(3);
                    double dfY = (j / Bottom.shape(3)) % Bottom.shape(2);
                    double dfExpectedVal = (1 - Math.Abs(dfX / nF - dfC)) * (1 - Math.Abs(dfY / nF - dfC));
                    double dfActualVal = rgData[i * nInnerNum + j];
                    m_log.EXPECT_NEAR(dfExpectedVal, dfActualVal, 0.01);
                }
            }
        }

        public void TestFillOdd()
        {
            const int n = 7;
            List<int> rgShape = new List<int>() { 1000, 2, n, n };
            test_params(rgShape);
        }

        public void TestFillEven()
        {
            const int n = 6;
            List<int> rgShape = new List<int>() { 1000, 2, n, n };
            test_params(rgShape);
        }
    }
}
