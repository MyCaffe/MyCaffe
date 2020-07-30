using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.test;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMathFunctions
    {
        [TestMethod]
        public void TestAsum()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestAsum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSign()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestSign();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFabs()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestFabs();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestScale()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestScale();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopy()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestCopy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMinMax()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestMinMax();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMinMaxVec()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestMinMaxVec(5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNanInf()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestNanInf();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMinMaxOneElm()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestMinMaxOneElm();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNanInfOneElm()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestNanInfOneElm();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSort()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestSort();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSort2()
        {
            MathFunctionsTest test = new MathFunctionsTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMathFunctionsTest t in test.Tests)
                {
                    t.TestSort2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IMathFunctionsTest : ITest
    {
        void TestAsum();
        void TestSign();
        void TestFabs();
        void TestScale();
        void TestCopy();
        void TestMinMax();
        void TestMinMaxVec(int nK);
        void TestNanInf();
        void TestMinMaxOneElm();
        void TestNanInfOneElm();
        void TestSort();
        void TestSort2();
    }

    class MathFunctionsTest : TestBase
    {
        public MathFunctionsTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Math Functions Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MathFunctionsTest<double>(strName, nDeviceID, engine);
            else
                return new MathFunctionsTest<float>(strName, nDeviceID, engine);
        }
    }

    class MathFunctionsTest<T> : TestEx<T>, IMathFunctionsTest
    {
        public MathFunctionsTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            Bottom.Reshape(11, 17, 19, 23);
            Top.Reshape(11, 17, 19, 23);

            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(Bottom);
            filler.Fill(Top);
        }

        public void TestAsum()
        {
            int n = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());
            double dfStdAsum = 0;

            for (int i = 0; i < n; i++)
            {
                dfStdAsum += Math.Abs(rgData[i]);
            }

            double dfAsum = convert(Bottom.asum_data());

            m_log.CHECK_LT((dfAsum - dfStdAsum) / dfStdAsum, 1e-2, "The asum is not as expected.");
        }

        public void TestSign()
        {
            int n = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());

            m_cuda.sign(n, Bottom.gpu_data, Bottom.mutable_gpu_diff);
            double[] rgSigns = convert(Bottom.update_cpu_diff());

            for (int i = 0; i < n; i++)
            {
                m_log.CHECK_EQ(rgSigns[i], rgData[i] > 0 ? 1 : (rgData[i] < 0 ? -1 : 0), "The signs do not match.");
            }
        }

        public void TestFabs()
        {
            int n = Bottom.count();
            double[] rgData = convert(Bottom.update_cpu_data());

            m_cuda.abs(n, Bottom.gpu_data, Bottom.mutable_gpu_diff);
            double[] rgAbs = convert(Bottom.update_cpu_diff());

            for (int i = 0; i < n; i++)
            {
                m_log.CHECK_EQ(rgAbs[i], rgData[i] > 0 ? rgData[i] : -rgData[i], "The abs do not match.");
            }
        }

        public void TestScale()
        {
            Random random = new Random(1701);
            int n = Bottom.count();
            double[] rgDiff = convert(Bottom.update_cpu_diff());
            double dfAlpha = rgDiff[random.Next() % Bottom.count()];

            m_cuda.scale(n, dfAlpha, Bottom.gpu_data, Bottom.mutable_gpu_diff);
            double[] rgScaled = convert(Bottom.update_cpu_diff());
            double[] rgData = convert(Bottom.update_cpu_data());

            for (int i = 0; i < n; i++)
            {
                m_log.CHECK_EQ(rgScaled[i], rgData[i] * dfAlpha, "The scaled values do not match.");
            }
        }

        public void TestCopy()
        {
            Random random = new Random(1701);
            int n = Bottom.count();

            m_cuda.copy(n, Bottom.gpu_data, Top.mutable_gpu_data);

            double[] rgBtm = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < n; i++)
            {
                m_log.CHECK_EQ(rgBtm[i], rgTop[i], "The top and bottom do not match.");
            }
        }

        public void TestMinMax()
        {
            double dfFree;
            double dfUsed;
            bool bCudaCall;
            m_cuda.GetDeviceMemory(out dfFree, out dfUsed, out bCudaCall);
            int nSize = (dfFree < 3.0) ? 124 : 512;

            Blob<T> data = new Blob<T>(m_cuda, m_log);
            Blob<T> work = new Blob<T>(m_cuda, m_log);

            data.Reshape(nSize, 3, 224, 224);
            Tuple<double, double, double, double> workSize = m_cuda.minmax(data.count(), 0, 0, 0);
            work.Reshape((int)workSize.Item1, 1, 1, 1);

            int nCount = data.count();
            T[] rgData = new T[nCount];
            double dfMin = double.MaxValue;
            double dfMax = -double.MaxValue;
            int nCycle = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfVal = Math.Sin(i) * Math.Cos(i);
                rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));

                dfMax = Math.Max(dfMax, dfVal);
                dfMin = Math.Min(dfMin, dfVal);

                if ((i % (int)workSize.Item1) == 0)
                {
                    nCycle++;
                    dfVal = dfMax * (nCycle + 1);
                    dfMax = dfVal;
                    rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
                }

                if (((i+1) % (int)workSize.Item1) == 0)
                {
                    nCycle++;
                    dfVal = dfMin - (dfMax * (nCycle + 1));
                    dfMin = dfVal;
                    rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
                }
            }

            data.mutable_cpu_data = rgData;

            Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff);

            m_log.EXPECT_EQUAL<float>(minmax.Item1, dfMin);
            m_log.EXPECT_EQUAL<float>(minmax.Item2, dfMax);
            m_log.CHECK_EQ(minmax.Item3, 0, "The max value should be zero.");
            m_log.CHECK_EQ(minmax.Item4, 0, "The inf value should be zero.");

            data.Dispose();
            work.Dispose();
        }

        public void TestMinMaxVec(int nK)
        {
            double dfFree;
            double dfUsed;
            bool bCudaCall;
            m_cuda.GetDeviceMemory(out dfFree, out dfUsed, out bCudaCall);
            int nSize = (dfFree < 3.0) ? 124 : 512;

            Blob<T> data = new Blob<T>(m_cuda, m_log);
            Blob<T> work = new Blob<T>(m_cuda, m_log);

            long hMin = m_cuda.AllocHostBuffer(nK);
            long hMax = m_cuda.AllocHostBuffer(nK);

            data.Reshape(nSize, 3, 224, 224);
            m_cuda.minmax(data.count(), 0, 0, 0, nK, hMin, hMax);
            double[] rgMin = m_cuda.GetHostMemoryDouble(hMin);
            double[] rgMax = m_cuda.GetHostMemoryDouble(hMax);
            work.Reshape((int)rgMin[0], 1, 1, 1);
            m_filler.Fill(data);

            m_cuda.minmax(data.count(), data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, nK, hMin, hMax);

            rgMin = m_cuda.GetHostMemoryDouble(hMin);
            rgMax = m_cuda.GetHostMemoryDouble(hMax);

            List<double> rgMin1 = new List<double>(rgMin);
            List<double> rgMax1 = new List<double>(rgMax);
            rgMin1 = rgMin1.OrderBy(p => p).ToList();
            rgMax1 = rgMax1.OrderByDescending(p => p).ToList();

            double[] rgData = convert(data.update_cpu_data());
            List<double> rgData1 = new List<double>(rgData);
            List<double> rgMin2 = new List<double>();
            List<double> rgMax2 = new List<double>();

            for (int i = 0; i < nK; i++)
            {
                double dfMin = double.MaxValue;
                double dfMax = -double.MaxValue;

                for (int j = 0; j < rgData1.Count; j++)
                { 
                    dfMin = Math.Min(dfMin, rgData1[j]);
                    dfMax = Math.Max(dfMax, rgData1[j]);
                }

                rgMin2.Add(dfMin);
                rgMax2.Add(dfMax);

                rgData1.Remove(dfMin);
                rgData1.Remove(dfMax);
            }

            rgMin2 = rgMin2.OrderBy(p => p).ToList();
            rgMax2 = rgMax2.OrderByDescending(p => p).ToList();

            double dfMaxTotal1 = 0;
            double dfMinTotal1 = 0;
            double dfMaxTotal2 = 0;
            double dfMinTotal2 = 0;

            for (int i = 0; i < nK; i++)
            {
                dfMaxTotal1 += rgMax1[i];
                dfMinTotal1 += rgMin1[i];
                dfMaxTotal2 += rgMax2[i];
                dfMinTotal2 += rgMin2[i];
            }

            double dfMinAve1 = dfMinTotal1 / nK;
            double dfMaxAve1 = dfMaxTotal1 / nK;
            double dfMinAve2 = dfMinTotal2 / nK;
            double dfMaxAve2 = dfMaxTotal2 / nK;

            double dfMinDiff = Math.Abs(dfMinAve1 - dfMinAve2);
            double dfMaxDiff = Math.Abs(dfMaxAve1 - dfMaxAve2);

            m_log.CHECK_LE(dfMinDiff, 1.0, "Min difference too high.");
            m_log.CHECK_LE(dfMaxDiff, 1.0, "Max difference too high.");

            data.Dispose();
            work.Dispose();

            m_cuda.FreeHostBuffer(hMin);
            m_cuda.FreeHostBuffer(hMax);
        }

        public void TestNanInf()
        {
            double dfFree;
            double dfUsed;
            bool bCudaCall;
            m_cuda.GetDeviceMemory(out dfFree, out dfUsed, out bCudaCall);
            int nSize = (dfFree < 3.0) ? 124 : 512;

            Blob<T> data = new Blob<T>(m_cuda, m_log);
            Blob<T> work = new Blob<T>(m_cuda, m_log);

            data.Reshape(nSize, 3, 224, 224);
            Tuple<double, double, double, double> workSize = m_cuda.minmax(data.count(), 0, 0, 0, true);
            work.Reshape((int)workSize.Item1, 1, 1, 1);

            int nCount = data.count();
            T[] rgData = new T[nCount];
            double dfMin = double.MaxValue;
            double dfMax = -double.MaxValue;
            int nCycle = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfVal = Math.Sin(i) * Math.Cos(i);
                rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));

                dfMax = Math.Max(dfMax, dfVal);
                dfMin = Math.Min(dfMin, dfVal);

                if ((i % (int)workSize.Item1) == 0)
                {
                    nCycle++;
                    dfVal = dfMax * (nCycle + 1);
                    dfMax = dfVal;
                    rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
                }

                if (((i + 1) % (int)workSize.Item1) == 0)
                {
                    nCycle++;
                    dfVal = dfMin - (dfMax * (nCycle + 1));
                    dfMin = dfVal;
                    rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
                }
            }

            data.mutable_cpu_data = rgData;
            Blob<T> data2 = new Blob<T>(m_cuda, m_log, data);
            m_cuda.copy(data2.count(), data.gpu_data, data2.mutable_gpu_data);

            Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.EXPECT_EQUAL<float>(minmax.Item1, dfMin);
            m_log.EXPECT_EQUAL<float>(minmax.Item2, dfMax);
            m_log.CHECK_EQ(minmax.Item3, 0, "There should be no nans.");
            m_log.CHECK_EQ(minmax.Item4, 0, "There should be no inf values.");

            // test Nan count.
            data.SetData(double.NaN, 10);
            data.SetData(double.NaN, 22);
            data.SetData(double.NaN, 55);

            minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.CHECK_EQ(minmax.Item3, 3, "There should be 3 nans.");
            m_log.CHECK_EQ(minmax.Item4, 0, "There should be no inf.");

            // test inf count.
            m_cuda.copy(data.count(), data2.gpu_data, data.mutable_gpu_data);

            data.SetData(float.NegativeInfinity, 4);
            data.SetData(float.PositiveInfinity, 88);
            data.SetData(double.PositiveInfinity, 120);
            data.SetData(double.PositiveInfinity, 121);

            minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.CHECK_EQ(minmax.Item3, 0, "There should be 0 nans.");
            m_log.CHECK_EQ(minmax.Item4, 4, "There should be 4 inf.");

            // test both nan and inf
            m_cuda.copy(data.count(), data2.gpu_data, data.mutable_gpu_data);

            data.SetData(double.NaN, 10);
            data.SetData(double.NaN, 22);
            data.SetData(double.NaN, 55);
            data.SetData(double.NegativeInfinity, 4);
            data.SetData(double.PositiveInfinity, 88);
            data.SetData(double.PositiveInfinity, 120);
            data.SetData(double.PositiveInfinity, 121);

            minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.CHECK_EQ(minmax.Item3, 3, "There should be 3 nans.");
            m_log.CHECK_EQ(minmax.Item4, 4, "There should be 4 inf.");

            data.Dispose();
            work.Dispose();
        }

        public void TestMinMaxOneElm()
        {
            Blob<T> data = new Blob<T>(m_cuda, m_log);
            Blob<T> work = new Blob<T>(m_cuda, m_log);

            data.Reshape(1, 1, 1, 1);
            Tuple<double, double, double, double> workSize = m_cuda.minmax(data.count(), 0, 0, 0);
            work.Reshape((int)workSize.Item1, 1, 1, 1);

            int nCount = data.count();
            T[] rgData = new T[nCount];
            double dfMin = double.MaxValue;
            double dfMax = -double.MaxValue;

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfVal = Math.Sin(i + 1) * Math.Cos(i + 1);
                rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));

                dfMax = Math.Max(dfMax, dfVal);
                dfMin = Math.Min(dfMin, dfVal);
            }

            data.mutable_cpu_data = rgData;

            Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff);

            m_log.EXPECT_EQUAL<float>(minmax.Item1, dfMin);
            m_log.EXPECT_EQUAL<float>(minmax.Item2, dfMax);
            m_log.CHECK_EQ(minmax.Item3, 0, "The max value should be zero.");
            m_log.CHECK_EQ(minmax.Item4, 0, "The inf value should be zero.");

            data.Dispose();
            work.Dispose();
        }

        public void TestNanInfOneElm()
        {
            Blob<T> data = new Blob<T>(m_cuda, m_log);
            Blob<T> work = new Blob<T>(m_cuda, m_log);

            data.Reshape(1, 1, 1, 1);
            Tuple<double, double, double, double> workSize = m_cuda.minmax(data.count(), 0, 0, 0, true);
            work.Reshape((int)workSize.Item1, 1, 1, 1);

            int nCount = data.count();
            T[] rgData = new T[nCount];
            double dfMin = double.MaxValue;
            double dfMax = -double.MaxValue;

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfVal = Math.Sin(i + 1) * Math.Cos(i + 1);
                rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));

                dfMax = Math.Max(dfMax, dfVal);
                dfMin = Math.Min(dfMin, dfVal);
            }

            data.mutable_cpu_data = rgData;
            Blob<T> data2 = new Blob<T>(m_cuda, m_log, data);
            m_cuda.copy(data2.count(), data.gpu_data, data2.mutable_gpu_data);

            Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.EXPECT_EQUAL<float>(minmax.Item1, dfMin);
            m_log.EXPECT_EQUAL<float>(minmax.Item2, dfMax);
            m_log.CHECK_EQ(minmax.Item3, 0, "There should be no nans.");
            m_log.CHECK_EQ(minmax.Item4, 0, "There should be no inf values.");

            // test Nan count.
            data.SetData(double.NaN, 0);

            minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.CHECK_EQ(minmax.Item3, 1, "There should be 1 nans.");
            m_log.CHECK_EQ(minmax.Item4, 0, "There should be no inf.");

            // test inf count.
            m_cuda.copy(data.count(), data2.gpu_data, data.mutable_gpu_data);

            data.SetData(float.NegativeInfinity, 0);

            minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, true);

            m_log.CHECK_EQ(minmax.Item3, 0, "There should be 0 nans.");
            m_log.CHECK_EQ(minmax.Item4, 1, "There should be 4 inf.");

            data.Dispose();
            work.Dispose();
        }

        public void TestSort()
        {
            double dfFree;
            double dfUsed;
            bool bCudaCall;
            m_cuda.GetDeviceMemory(out dfFree, out dfUsed, out bCudaCall);
            int nSize = (dfFree < 3.0) ? 124 : 512;

            Blob<T> data = new Blob<T>(m_cuda, m_log);
            Blob<T> work = new Blob<T>(m_cuda, m_log);

            data.Reshape(nSize, 3, 224, 224);
            Tuple<double, double, double, double> workSize = m_cuda.minmax(data.count(), 0, 0, 0);
            work.Reshape((int)workSize.Item1, 1, 1, 1);

            int nCount = data.count();
            T[] rgData = new T[nCount];
            double dfMin = double.MaxValue;
            double dfMax = -double.MaxValue;
            int nCycle = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                double dfVal = Math.Sin(i) * Math.Cos(i);
                rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));

                dfMax = Math.Max(dfMax, dfVal);
                dfMin = Math.Min(dfMin, dfVal);

                if ((i % (int)workSize.Item1) == 0)
                {
                    nCycle++;
                    dfVal = dfMax * (nCycle + 1);
                    dfMax = dfVal;
                    rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
                }

                if (((i + 1) % (int)workSize.Item1) == 0)
                {
                    nCycle++;
                    dfVal = dfMin - (dfMax * (nCycle + 1));
                    dfMin = dfVal;
                    rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
                }
            }

            data.mutable_cpu_data = rgData;

            Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, data.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff);

            m_log.EXPECT_EQUAL<float>(minmax.Item1, dfMin);
            m_log.EXPECT_EQUAL<float>(minmax.Item2, dfMax);
            m_log.CHECK_EQ(minmax.Item3, 0, "The max value should be zero.");
            m_log.CHECK_EQ(minmax.Item4, 0, "The inf value should be zero.");


            m_cuda.sort(nCount, data.mutable_gpu_data);

            double[] rgData1 = m_cuda.GetMemoryDouble(data.gpu_data);
            m_log.EXPECT_EQUAL<float>(dfMin, rgData1[0]);
            m_log.EXPECT_EQUAL<float>(dfMax, rgData1[rgData1.Length - 1]);

            data.Dispose();
            work.Dispose();
        }

        public void TestSort2()
        {
            Blob<T> blob = new Blob<T>(m_cuda, m_log);

            blob.Reshape(6400, 1, 1, 1);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, new FillerParameter());
            filler.Fill(blob);

            double dfMin = blob.min_data;
            double dfMax = blob.max_data;

            m_cuda.sort(blob.count(), blob.mutable_gpu_data);

            double[] rgData = m_cuda.GetMemoryDouble(blob.gpu_data);

            m_log.EXPECT_EQUAL<float>(dfMin, rgData[0]);
            m_log.EXPECT_EQUAL<float>(dfMax, rgData[rgData.Length - 1]);
        }
    }
}
