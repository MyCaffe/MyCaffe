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
using System.Diagnostics;
using MyCaffe.db.image;
using System.Drawing;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPCA
    {
        [TestMethod]
        public void TestSetup()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
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
        public void TestSumSq()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestSumSq(1000, 784, 784 * 10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumSqDiff()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestSumSqDiff(1000, 784, 784 * 10, 784 * 20);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumSqDiff1()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestSumSqDiff(1000, 784, 784 * 10, 784 * 11);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleDot()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestDot(3, 3, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleNonSquareDot()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestDot(3, 2, 4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBigNonSquareDot()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    int nM = 20000;
                    int nN = 784;
                    int nK = 50;

                    if (t.DataType == DataType.FLOAT)
                    {
                        nM = 1;
                        nK = 10;
                    }

                    t.TestDot(nM, nN, nK);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleTranspose()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestTranspose(3, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleLargeTranspose()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestTranspose(300, 300);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNonSquareTranspose()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestTranspose(3, 2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNonSquareLargeTranspose()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestTranspose(40, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleMeanCenter()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMeanCenter(3, 3, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleLargeMeanCenter()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    int nN = 300;
                    int nD = 300;

                    if (t.DataType == DataType.FLOAT)
                    {
                        nN = 100;
                        nD = 100;
                    }

                    t.TestMeanCenter(nN, nD, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNonSquareMeanCenter()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMeanCenter(3, 2, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNonSquareLargeMeanCenter()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    int nN = 60000;
                    int nD = 784;

                    if (t.DataType == DataType.FLOAT)
                    {
                        nN = 100;
                        nD = 78;
                    }

                    t.TestMeanCenter(nN, nD, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleMeanCenterNormalize()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMeanCenter(3, 3, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleLargeMeanCenterNormalize()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    int nN = 300;
                    int nD = 300;

                    if (t.DataType == DataType.FLOAT)
                    {
                        nN = 100;
                        nD = 100;
                    }

                    t.TestMeanCenter(nN, nD, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNonSquareMeanCenterNormalize()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMeanCenter(3, 2, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNonSquareLargeMeanCenterNormalize()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMeanCenter(3000, 784, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleWidth()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestWidth(3, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBigWidth()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    int nN = 60000;
                    int nD = 784;

                    if (t.DataType == DataType.FLOAT)
                    {
                        nN = 100;
                        nD = 784;
                    }

                    t.TestWidth(nN, nD);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimplePCA()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestSimplePCA();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimplePCA_randomdata()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestSimplePCA_randomdata();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMaxVal()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMaxVal();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMinVal()
        {
            PCATest test = new PCATest();

            try
            {
                foreach (IPCATest t in test.Tests)
                {
                    t.TestMinVal();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IPCATest : ITest
    {
        void TestSetup();
        void TestSumSq(int nN, int nD, int nAOff);
        void TestSumSqDiff(int nN, int nD, int nAOff, int nBOff);
        void TestDot(int nM, int nN, int nK);
        void TestTranspose(int nM, int nN);
        void TestMeanCenter(int nN, int nD, bool bNormalize);
        void TestWidth(int nN, int nD);
        void TestSimplePCA_randomdata();
        void TestSimplePCA();
        void TestMaxVal();
        void TestMinVal();
    }

    class PCATest : TestBase
    {
        public PCATest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PCA Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PCATest<double>(strName, nDeviceID, engine);
            else
                return new PCATest<float>(strName, nDeviceID, engine);
        }
    }

    class PCATest<T> : TestEx<T>, IPCATest
    {
        public PCATest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 20, 3, 1, 1 }, nDeviceID)
        {
            m_engine = engine;

            Top.ReshapeLike(Bottom);

            FillerParameter p1 = new FillerParameter("gaussian");
            p1.std = 1.0;
            p1.mean = 0.0;
            Filler<T> f1 = Filler<T>.Create(m_cuda, m_log, p1);

            FillerParameter p2 = new FillerParameter("gaussian");
            p2.std = 1.0;
            p2.mean = 1.0;
            Filler<T> f2 = Filler<T>.Create(m_cuda, m_log, p2);

            f1.Fill(Bottom);
            f2.Fill(Top);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestSetup()
        {
            int nMaxIter = 100;
            int nM = 40;
            int nN = 3;
            int nK = 2;
            int nDataCount;
            long hData = m_cuda.AllocPCAData(nM, nN, nK, out nDataCount);
            int nScoresCount;
            long hScores = m_cuda.AllocPCAScores(nM, nN, nK, out nScoresCount);
            int nLoadsCount;
            long hLoads = m_cuda.AllocPCALoads(nM, nN, nK, out nLoadsCount);

            Assert.AreEqual(nDataCount, nM * nN);
            Assert.AreEqual(nScoresCount, nM * nK);
            Assert.AreEqual(nLoadsCount, nN * nK);

            long hPCA = m_cuda.CreatePCA(nMaxIter, nM, nN, nK, hData, hScores, hLoads);

            m_cuda.FreePCA(hPCA);

            m_cuda.FreeMemory(hData);
            m_cuda.FreeMemory(hScores);
            m_cuda.FreeMemory(hLoads);
        }

        public void TestSumSq(int nN, int nD, int nAOff)
        {
            int nCount = nN * nD;
            double dfErr = (m_dt == common.DataType.DOUBLE) ? 0.01 : 0.1;
            double[] rgData = new double[nCount];
            double dfFactor = (nCount < 1000) ? 1.0 : 0.001;

            for (int i = 0; i < nCount; i++)
            {
                rgData[i] = i * dfFactor;
            }

            long hData = m_cuda.AllocMemory(rgData);
            long hTemp = m_cuda.AllocMemory(rgData.Length);

            double dfSumSq = m_cuda.sumsq(nD, hTemp, hData, nAOff);
            double dfSumSq1 = 0;

            for (int d = 0; d < nD; d++)
            {
                double dfItem = rgData[nAOff + d];
                dfSumSq1 += (dfItem * dfItem);
            }

            m_log.EXPECT_NEAR(dfSumSq, dfSumSq1, dfErr);

            m_cuda.FreeMemory(hTemp);
            m_cuda.FreeMemory(hData);
        }

        public void TestSumSqDiff(int nN, int nD, int nAOff, int nBOff)
        {
            int nCount = nN * nD;
            double dfErr = (m_dt == common.DataType.DOUBLE) ? 0.01 : 0.1;
            double[] rgData = new double[nCount];
            double dfFactor = (nCount < 1000) ? 1.0 : 0.001;

            for (int i = 0; i < nCount; i++)
            {
                rgData[i] = i * dfFactor;
            }

            long hData = m_cuda.AllocMemory(rgData);
            long hTemp = m_cuda.AllocMemory(rgData.Length);

            double dfSumSq = m_cuda.sumsqdiff(nD, hTemp, hData, hData, nAOff, nBOff);
            dfSumSq = m_cuda.sumsqdiff(nD, hTemp, hData, hData, nAOff, nBOff);
            dfSumSq = m_cuda.sumsqdiff(nD, hTemp, hData, hData, nAOff, nBOff);
            double dfSumSq1 = 0;

            for (int d = 0; d < nD; d++)
            {
                double dfDiff = rgData[nAOff + d] - rgData[nBOff + d];
                dfSumSq1 += (dfDiff * dfDiff);
            }

            m_log.EXPECT_NEAR(dfSumSq, dfSumSq1, dfErr);

            m_cuda.FreeMemory(hTemp);
            m_cuda.FreeMemory(hData);
        }

        public void TestDot(int nM, int nN, int nK)
        {
            if (m_dt == DataType.DOUBLE)
                TestDotD(nM, nN, nK);
            else
                TestDotF(nM, nN, nK);
        }

        private void TestDotD(int nM, int nN, int nK)
        {
            double dfErr = (m_dt == common.DataType.DOUBLE) ? 0.01 : 0.1;
            if (nM > 1000)
                dfErr = 1.0;

            double[] rgA = new double[nM * nN];
            double[] rgB = new double[nN * nK];
            double[] rgC = new double[nM * nK];
            int i = 1;

            for (int y = 0; y < nM; y++)
            {
                for (int x = 0; x < nN; x++)
                {
                    int nIdx = y * nN + x;
                    rgA[nIdx] = i;
                    i++;

                    if (i % 3 == 0)
                        rgA[nIdx] = 0;
                }
            }

            i = 1;
            for (int y = 0; y < nN; y++)
            {
                for (int x = 0; x < nK; x++)
                {
                    int nIdx = y * nK + x;
                    rgB[nIdx] = (double)i / 1000.0;
                    i++;

                    if (i % 2 == 0)
                        rgB[nIdx] = 0;
                }
            }

            for (int m = 0; m < nM; m++)
            {
                for (int k = 0; k < nK; k++)
                {
                    double dfVal = 0;

                    for (int n = 0; n < nN; n++)
                    {
                        int nIdxA = m * nN + n;
                        int nIdxB = n * nK + k;
                        dfVal += rgA[nIdxA] * rgB[nIdxB];
                    }

                    int nIdxC = m * nK + k;
                    rgC[nIdxC] = dfVal;
                }
            }

            long hA = m_cuda.AllocMemory(convert(rgA));
            long hB = m_cuda.AllocMemory(convert(rgB));
            long hC = m_cuda.AllocMemory(nK * nM);

            m_cuda.matrix_dot(nM, nN, nK, hA, hB, hC);
            double[] rgC1 = m_cuda.GetMemoryDouble(hC);

            for (int y = 0; y < nM; y++)
            {
                for (int x = 0; x < nK; x++)
                {
                    int nIdx = y * nK + x;
                    double dfC = rgC[nIdx];
                    double dfC1 = rgC1[nIdx];

                    m_log.EXPECT_NEAR(dfC, dfC1, dfErr);
                }
            }
        }

        private void TestDotF(int nM, int nN, int nK)
        {
            float dfErr = 10.0f;
            float[] rgA = new float[nM * nN];
            float[] rgB = new float[nN * nK];
            float[] rgC = new float[nM * nK];
            int i = 1;

            for (int y = 0; y < nM; y++)
            {
                for (int x = 0; x < nN; x++)
                {
                    int nIdx = y * nN + x;
                    rgA[nIdx] = i;
                    i++;

                    if (i % 3 == 0)
                        rgA[nIdx] = 0;
                }
            }

            i = 1;
            for (int y = 0; y < nN; y++)
            {
                for (int x = 0; x < nK; x++)
                {
                    int nIdx = y * nK + x;
                    rgB[nIdx] = (float)i / 1000.0f;
                    i++;

                    if (i % 2 == 0)
                        rgB[nIdx] = 0;
                }
            }

            for (int m = 0; m < nM; m++)
            {
                for (int k = 0; k < nK; k++)
                {
                    float dfVal = 0;

                    for (int n = 0; n < nN; n++)
                    {
                        int nIdxA = m * nN + n;
                        int nIdxB = n * nK + k;
                        dfVal += rgA[nIdxA] * rgB[nIdxB];
                    }

                    int nIdxC = m * nK + k;
                    rgC[nIdxC] = dfVal;
                }
            }

            long hA = m_cuda.AllocMemory(convert(rgA));
            long hB = m_cuda.AllocMemory(convert(rgB));
            long hC = m_cuda.AllocMemory(nK * nM);

            m_cuda.matrix_dot(nM, nN, nK, hA, hB, hC);
            float[] rgC1 = m_cuda.GetMemoryFloat(hC);

            for (int y = 0; y < nM; y++)
            {
                for (int x = 0; x < nK; x++)
                {
                    int nIdx = y * nK + x;
                    float dfC = rgC[nIdx];
                    float dfC1 = rgC1[nIdx];

                    m_log.EXPECT_NEAR(dfC, dfC1, dfErr);
                }
            }
        }

        public void TestTranspose(int nM, int nN)
        {
            double[] rgData = new double[nM * nN];
            int i = 1;

            for (int y=0; y<nM; y++)
            {
                for (int x=0; x<nN; x++)
                {
                    int nIdx = y * nN + x;
                    rgData[nIdx] = i;
                    i++;
                }
            }

            long hData = m_cuda.AllocMemory(rgData);
            long hDataT = m_cuda.AllocMemory(rgData.Length);
            
            m_cuda.matrix_transpose(nN, nM, hData, hDataT);

            double[] rgDataT = m_cuda.GetMemoryDouble(hDataT);

            Assert.AreEqual(true, rgData.Length <= rgDataT.Length);

            i = 1;

            for (int x = 0; x < nM; x++ )
            {
                for (int y = 0; y < nN; y++)
                {
                    int nIdx = y * nM + x;
                    Assert.AreEqual(rgDataT[nIdx], i);
                    i++;
                }
            }

            m_cuda.FreeMemory(hData);
            m_cuda.FreeMemory(hDataT);
        }

        public void TestMeanCenter(int nN, int nD, bool bNormalize)
        {
            double dfErr = (m_dt == common.DataType.DOUBLE) ? 0.01 : 0.1;
            if (nN > 1000)
                dfErr = 2.0;

            double[] rgData = new double[nN * nD];
            int i = 1;
            double dfFactor = (nN > 1000) ? 0.0000001 : 1.0;
            double dfMax = -double.MaxValue;

            for (int y = 0; y < nN; y++)
            {
                for (int x = 0; x < nD; x++)
                {
                    int nIdx = y * nD + x;
                    rgData[nIdx] = i * dfFactor;
                    i++;
                }
            }

            long hData = m_cuda.AllocMemory(rgData);


            //-------------------------------------------------------
            //  Test by column.
            //-------------------------------------------------------

            long hDataC = m_cuda.AllocMemory(rgData.Length);
            long hSumC = m_cuda.AllocMemory(nD);    // contains the sum of each column.

            m_cuda.matrix_meancenter_by_column(nD, nN, hData, hSumC, hDataC, bNormalize);
            double[] rgMC_by_col = m_cuda.GetMemoryDouble(hDataC);
            double[] rgColSums = m_cuda.GetMemoryDouble(hSumC);
            double[] rgColSums1 = new double[nD];

            for (int n = 0; n < nN; n++)
            {
                for (int d = 0; d < nD; d++)
                {
                    int nIdx = n * nD + d;
                    rgColSums1[d] += rgData[nIdx];
                }
            }

            for (int d= 0; d < nD; d++)
            {
                double dfSum1 = rgColSums[d];
                double dfSum2 = rgColSums1[d];
                m_log.EXPECT_NEAR(dfSum1, dfSum2, dfErr);
            }

            double[] rgMC_by_col1 = new double[nN * nD];

            for (int n = 0; n < nN; n++)
            {
                for (int d = 0; d < nD; d++)
                {
                    int nIdx = n * nD + d;
                    rgMC_by_col1[nIdx] = (rgData[nIdx] - (rgColSums1[d] / nN));
                    dfMax = Math.Max(dfMax, rgMC_by_col1[nIdx]);
                }
            }

            for (int n = 0; n < nN; n++)
            {
                for (int d = 0; d < nD; d++)
                {
                    int nIdx = n * nD + d;
                    double dfV1 = rgMC_by_col[nIdx];
                    double dfV2 = rgMC_by_col1[nIdx] / ((bNormalize) ? dfMax : 1.0);
                    m_log.EXPECT_NEAR(dfV1, dfV2, dfErr);
                }
            }


            //-------------------------------------------------------
            //  Test by row.
            //-------------------------------------------------------

            long hDataT = m_cuda.AllocMemory(rgData.Length);
            m_cuda.matrix_transpose(nD, nN, hData, hDataT);
            double[] rgDataT = m_cuda.GetMemoryDouble(hDataT);

            int nN1 = nD;
            int nD1 = nN;

            long hDataR = m_cuda.AllocMemory(rgData.Length);
            long hSumR = m_cuda.AllocMemory(nN);    // contains the sum of each row.

            m_cuda.matrix_meancenter_by_column(nN, nD, hDataT, hSumR, hDataR, bNormalize);
            double[] rgMC_by_row = m_cuda.GetMemoryDouble(hDataR);
            double[] rgRowSums = m_cuda.GetMemoryDouble(hSumR);
            double[] rgRowSums1 = new double[nD1];

            for (int n = 0; n < nN1; n++)
            {
                for (int d = 0; d < nD1; d++)
                {
                    int nIdx = n * nD1 + d;
                    rgRowSums1[d] += rgDataT[nIdx];
                }
            }

            for (int n = 0; n < nN; n++)
            {
                double dfSum1 = rgRowSums[n];
                double dfSum2 = rgRowSums1[n];
                m_log.EXPECT_NEAR(dfSum1, dfSum2, dfErr);
            }

            double[] rgMC_by_row1 = new double[nN * nD];
            dfMax = -double.MaxValue;

            for (int n = 0; n < nN1; n++)
            {
                for (int d = 0; d < nD1; d++)
                {
                    int nIdx = n * nD1 + d;
                    rgMC_by_row1[nIdx] = (rgDataT[nIdx] - (rgRowSums1[d] / nD));
                    dfMax = Math.Max(dfMax, rgMC_by_row1[nIdx]);
                }
            }

            for (int n = 0; n < nN1; n++)
            {
                for (int d = 0; d < nD1; d++)
                {
                    int nIdx = n * nD1 + d;
                    double dfV1 = rgMC_by_row[nIdx];
                    double dfV2 = rgMC_by_row1[nIdx] / ((bNormalize) ? dfMax : 1.0);
                    m_log.EXPECT_NEAR(dfV1, dfV2, dfErr);
                }
            }

            m_cuda.FreeMemory(hDataT);
            m_cuda.FreeMemory(hData);
            m_cuda.FreeMemory(hDataR);
            m_cuda.FreeMemory(hDataC);
            m_cuda.FreeMemory(hSumR);
            m_cuda.FreeMemory(hSumC);
        }

        public void TestWidth(int nN, int nD)
        {
            double[] rgData = new double[nN * nD];
            int i = 1;

            for (int y = 0; y < nN; y++)
            {
                for (int x = 0; x < nD; x++)
                {
                    int nIdx = y * nD + x;
                    rgData[nIdx] = i;
                    i++;
                }
            }

            double dfErr = (m_dt == common.DataType.DOUBLE) ? 0.01 : 0.3;
            if (nN > 1000)
                dfErr = 5.0;

            long hData = m_cuda.AllocMemory(rgData);
            long hData1 = m_cuda.AllocMemory(rgData.Length);
            long hMean = m_cuda.AllocMemory(nD);

            m_cuda.copy(nN * nD, hData, hData1);
            double[] rgData2 = m_cuda.GetMemoryDouble(hData1);
            m_cuda.matrix_meancenter_by_column(nD, nN, hData1, hMean, hData1);    // hData1 mean centered on columns, 
                                                                                  // hMean contains sums of each column
            double[] rgData1 = m_cuda.GetMemoryDouble(hData1);
            double[] rgSums = m_cuda.GetMemoryDouble(hMean);

            m_cuda.scal(nD, 1.0 / nN, hMean);
            double[] rgMeans = m_cuda.GetMemoryDouble(hMean);
            List<double> rgMeans1 = new List<double>();

            //----------------------------------------------
            // example
            //
            //  input data     mean centered (by column)
            // [  1  2  3 ]    [ -3 -3 -3 ]  <- 1 - 4 = -3 (first column, first row)
            // [  4  5  6 ] >> [  0  0  0 ]  <- 4 - 4 =  0 (first column, second row)
            // [  7  8  9 ]    [  3  3  3 ]  <- 7 - 4 =  3 (first column, third row)
            //   12 15 18   <- hMean after .matrix_meancenter() call.
            //    4  5  6   <- hMean after scal(1/N=3)
            //----------------------------------------------

            for (int x = 0; x < nD; x++)
            {
                double dfTotal = 0;

                for (int y = 0; y < nN; y++)
                {
                    int nIdx = y * nD + x;
                    dfTotal += rgData[nIdx];
                }

                rgMeans1.Add(dfTotal / nN);
            }

            for (int n = 0; n < nD; n++)
            {
                m_log.EXPECT_NEAR(rgMeans[n], rgMeans1[n], dfErr);
            }

            for (int y = 0; y < nN; y++)
            {
                for (int x = 0; x < nD; x++)
                {
                    int nIdx = y * nD + x;
                    double dfMC = rgData1[nIdx];
                    double dfMC1 = rgData[nIdx] - rgMeans1[x];

                    m_log.EXPECT_NEAR(dfMC, dfMC1, dfErr);
                }
            }

            //----------------------------------------------
            //  Test Min/Max
            //----------------------------------------------

            List<double> rgMin = new List<double>();
            List<double> rgMax = new List<double>();
            long hDataT = m_cuda.AllocMemory(rgData.Length);

            m_cuda.matrix_transpose(nD, nN, hData, hDataT);  // hDataT(nD rows x nN cols)

            for (int d = 0; d < nD; d++)
            {
                long lPos;
                double dfMin = m_cuda.min(nN, hDataT, out lPos, d * nN);
                double dfMax = m_cuda.max(nN, hDataT, out lPos, d * nN);

                rgMin.Add(dfMin);
                rgMax.Add(dfMax);
            }

            long hMin = m_cuda.AllocMemory(rgMin);
            long hMax = m_cuda.AllocMemory(rgMax);

            List<double> rgMin1 = new List<double>();
            List<double> rgMax1 = new List<double>();
            double[] rgDataT = m_cuda.GetMemoryDouble(hDataT);

            for (int d = 0; d < nD; d++)
            {
                double dfMin = double.MaxValue;
                double dfMax = double.MinValue;

                for (int n = 0; n < nN; n++)
                {
                    int nIdx = d * nN + n;
                    double df = rgDataT[nIdx];
                    dfMin = Math.Min(dfMin, df);
                    dfMax = Math.Max(dfMax, df);
                }

                rgMin1.Add(dfMin);
                rgMax1.Add(dfMax);
            }

            double[] rgMin2 = m_cuda.GetMemoryDouble(hMin);
            double[] rgMax2 = m_cuda.GetMemoryDouble(hMax);

            for (int d = 0; d < nD; d++)
            {
                double dfMin1 = rgMin1[d];
                double dfMin2 = rgMin2[d];
                double dfMax1 = rgMax1[d];
                double dfMax2 = rgMax2[d];

                m_log.CHECK_EQ(dfMin1, dfMin2, "Min 1 and 2 are not equal!");
                m_log.CHECK_EQ(dfMax1, dfMax2, "Max 1 and 2 are not equal!");
            }


            //----------------------------------------------
            //  Test Width
            //----------------------------------------------

            long hWidth = m_cuda.AllocMemory(nD);
            m_cuda.width(nD, hMean, hMin, hMax, 0, hWidth);
            double[] rgWidth = m_cuda.GetMemoryDouble(hWidth);
            List<double> rgWidth1 = new List<double>();

            for (int d = 0; d < nD; d++)
            {
                double dfMax = rgMax1[d];
                double dfMin = rgMin1[d];
                double dfMean = rgMeans1[d];
                double dfWidth = Math.Max(dfMax - dfMean, dfMean - dfMin);
                rgWidth1.Add(dfWidth);
            }

            for (int d = 0; d < nD; d++)
            {
                double dfWidth = rgWidth[d];
                double dfWidth1 = rgWidth1[d];
                m_log.EXPECT_NEAR(dfWidth, dfWidth1, dfErr);
            }


            //----------------------------------------------
            //  Test contains point
            //----------------------------------------------

            bool bContainsPoint = false;
            List<double> rgPoint = new List<double>();

            for (int d = 0; d < nD; d++)
            {
                rgPoint.Add(float.MaxValue);
            }

            long hPoint = m_cuda.AllocMemory(rgPoint);
            long hWork = m_cuda.AllocMemory(rgPoint.Count);

            bContainsPoint = m_cuda.contains_point(nD, hMean, hWidth, hPoint, hWork);
            m_log.CHECK(!bContainsPoint, "The point should not be contained.");

            for (int d = 0; d < nD; d++)
            {
                rgPoint[d] = rgMeans1[d];
            }

            m_cuda.SetMemory(hPoint, rgPoint);
            bContainsPoint = m_cuda.contains_point(nD, hMean, hWidth, hPoint, hWork);
            m_log.CHECK(bContainsPoint, "The point should be contained.");

            CryptoRandom rand = new CryptoRandom();

            for (int d = 0; d < nD; d++)
            {
                double dfRand = rand.NextDouble(0, 2);
                double df = rgWidth1[d] * dfRand;

                rgPoint[d] = rgMeans1[d] + df;
                m_cuda.SetMemory(hPoint, rgPoint);
                bContainsPoint = m_cuda.contains_point(nD, hMean, hWidth, hPoint, hWork);

                if (dfRand >= 1.0)
                    m_log.CHECK(!bContainsPoint, "The point should not be contained.");
                else
                    m_log.CHECK(bContainsPoint, "The point should be contained.");

                rgPoint[d] = rgMeans[d];
            }

            m_cuda.FreeMemory(hPoint);
            m_cuda.FreeMemory(hWork);
            m_cuda.FreeMemory(hWidth);
            m_cuda.FreeMemory(hData);
            m_cuda.FreeMemory(hData1);
            m_cuda.FreeMemory(hDataT);
            m_cuda.FreeMemory(hMean);
            m_cuda.FreeMemory(hMin);
            m_cuda.FreeMemory(hMax);
        }

        public void TestSimplePCA()
        {
            int nMaxIter = 1000;
            int nM = 5;
            int nN = 5;
            int nK = 5;
            int nDataCount;
            long hData = m_cuda.AllocPCAData(nM, nN, nK, out nDataCount);


            //-----------------------------------
            //  Copy all samples into Data.
            //-----------------------------------

            double[] rgData = m_cuda.GetMemoryDouble(hData);

            rgData[0] = 1; rgData[1] = -1; rgData[2] = 1; rgData[3] = 0; rgData[4] = 0;
            rgData[5] = 2; rgData[6] = 9; rgData[7] = 1; rgData[8] = 0; rgData[9] = 0; 
            rgData[10] = 3; rgData[11] = -10; rgData[12] = 1; rgData[13] = 0; rgData[14] = 0;
            rgData[15] = 4; rgData[16] = 2; rgData[17] = 1; rgData[18] = 0; rgData[19] = 0;
            rgData[20] = 5; rgData[21] = 40; rgData[22] = 1; rgData[23] = 0; rgData[24] = 0;

            m_cuda.SetMemory(hData, rgData);

            TestSimplePCA(nMaxIter, nM, nN, nK, hData);

            m_cuda.FreeMemory(hData);
        }

        public void TestSimplePCA_randomdata()
        {
            int nMaxIter = 100;
            int nM = 40;
            int nN = 3;
            int nK = 3;
            int nDataCount;
            long hData = m_cuda.AllocPCAData(nM, nN, nK, out nDataCount);


            //-----------------------------------
            //  Copy all samples into Data.
            //-----------------------------------

            m_cuda.copy(Bottom.count(), Bottom.gpu_data, hData);
            double[] rgData1 = m_cuda.GetMemoryDouble(hData);

            m_cuda.copy(Top.count(), Top.gpu_data, hData, 0, Bottom.count());
            double[] rgData2 = m_cuda.GetMemoryDouble(hData);

            TestSimplePCA(nMaxIter, nM, nN, nK, hData);

            m_cuda.FreeMemory(hData);
        }

        public void TestSimplePCA(int nMaxIter, int nM, int nN, int nK, long hData)
        {
            int nResidualCount;
            long hResiduals = m_cuda.AllocPCAData(nM, nN, nK, out nResidualCount);
            int nScoresCount;
            long hScores = m_cuda.AllocPCAScores(nM, nN, nK, out nScoresCount);
            int nLoadsCount;
            long hLoads = m_cuda.AllocPCALoads(nM, nN, nK, out nLoadsCount);
            long hPCA = m_cuda.CreatePCA(nMaxIter, nM, nN, nK, hData, hScores, hLoads, hResiduals);


            //-----------------------------------
            //  Calculate the means (for debugging)
            //-----------------------------------

            double[] rgData = m_cuda.GetMemoryDouble(hData);
            double[] rgMeans = new double[nN];

            for (int i = 0; i < nM; i++)
            {
                int nIdx = (i * nN);

                for (int j = 0; j < nN; j++)
                {
                    rgMeans[j] += rgData[nIdx + j];
                }
            }

            for (int j = 0; j < nN; j++)
            {
                rgMeans[j] /= nM;
            }

            Trace.Write("Mean values: ");

            for (int j = 0; j < nN; j++)
            {
                Trace.Write(rgMeans[j].ToString());

                if (j < nN - 1)
                    Trace.Write(", ");
            }

            //-----------------------------------
            //  Run the PCA
            //-----------------------------------

            int nCurrentK = 0;
            int nCurrentIteration = 0;
            int nCount = 0;

            while (!m_cuda.RunPCA(hPCA, 1, out nCurrentK, out nCurrentIteration))
            {
                nCount++;
            }

            Trace.WriteLine("Ran " + nCount.ToString() + " cycles");


            //-----------------------------------
            //  View the data.
            //-----------------------------------

            Trace.WriteLine("SCORES (" + nM.ToString() + "x" + nK.ToString() + ")");

            double[] rgScores = m_cuda.GetMemoryDouble(hScores);

            for (int i = 0; i < nM; i++)
            {
                Trace.Write("{ ");

                for (int j = 0; j < nK; j++)
                {
                    Trace.Write(rgScores[(i * nK) + j].ToString());

                    if (j < nK - 1)
                        Trace.Write(",");
                }

                Trace.WriteLine(" }");
            }

            Trace.WriteLine("");
            Trace.WriteLine("LOADS (" + nN.ToString() + "x" + nK.ToString() + ")");

            double[] rgLoads = m_cuda.GetMemoryDouble(hLoads);

            for (int i = 0; i < nN; i++)
            {
                Trace.Write("{ ");

                for (int j = 0; j < nK; j++)
                {
                    Trace.Write(rgLoads[(i * nK) + j].ToString());

                    if (j < nK - 1)
                        Trace.Write(",");
                }

                Trace.WriteLine(" }");
            }


            //-----------------------------------
            //  Free resources.
            //-----------------------------------

            m_cuda.FreePCA(hPCA);
            m_cuda.FreeMemory(hScores);
            m_cuda.FreeMemory(hLoads);
        }

        public void TestMaxVal()
        {
            int n = 1000000;
            long hData = m_cuda.AllocMemory(n);

            m_cuda.rng_gaussian(n, 0, 1, hData);

            long lPos;
            double dfVal = m_cuda.max(n, hData, out lPos);
            double[] rgData = m_cuda.GetMemoryDouble(hData);

            double dfMax = -double.MaxValue;
            int nMaxIdx = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                if (dfMax < rgData[i])
                {
                    nMaxIdx = i;
                    dfMax = rgData[i];
                }
            }

            Assert.AreEqual(dfMax, dfVal);
            Assert.AreEqual(lPos, nMaxIdx);
        }

        public void TestMinVal()
        {
            int n = 1000000;
            long hData = m_cuda.AllocMemory(n);

            m_cuda.rng_gaussian(n, 0, 1, hData);

            long lPos;
            double dfVal = m_cuda.min(n, hData, out lPos);
            double[] rgData = m_cuda.GetMemoryDouble(hData);

            double dfMin = double.MaxValue;
            int nMinIdx = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                if (dfMin > rgData[i])
                {
                    nMinIdx = i;
                    dfMin = rgData[i];
                }
            }

            Assert.AreEqual(dfMin, dfVal);
            Assert.AreEqual(lPos, nMinIdx);
        }
    }
}
