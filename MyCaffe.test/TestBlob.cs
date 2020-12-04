using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.basecode;
using System.Drawing;
using System.IO;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBlob
    {
        [TestMethod]
        public void TestInitialization()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestInitialization();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPointersCPUGPU()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestPointersCPUGPU();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReshape()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
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
        public void TestLegacyBlobProtoShapeEquals()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestLegacyBlobProtoShapeEquals();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyFrom()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyFrom();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyFromChannels()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyFromChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_SumOfSquares()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMath_SumOfSquares();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_Asum()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMath_Asum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_Scale()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMath_Scale();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResize1()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestResize1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResize2()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestResize2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResize3()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestResize3();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class BlobSimpleTest : TestBase
    {
        public BlobSimpleTest()
            : base("Blob Simple Teset")
        {
        }

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            if (dt == DataType.DOUBLE)
                return new BlobSimpleTest<double>(strName, nDeviceID);
            else
                return new BlobSimpleTest<float>(strName, nDeviceID);
        }
    }

    interface IBlobSimpleTest 
    {
        void TestInitialization();
        void TestPointersCPUGPU();
        void TestReshape();
        void TestLegacyBlobProtoShapeEquals();
        void TestCopyFrom();
        void TestCopyFromChannels();
        void TestMath_SumOfSquares();
        void TestMath_Asum();
        void TestMath_Scale();
        void TestResize1();
        void TestResize2();
        void TestResize3();
    }

    class BlobSimpleTest<T> : Test<T>, IBlobSimpleTest 
    {
        Blob<T> m_blob;
        Blob<T> m_blob_preshaped;
        double m_fEpsilon;

        public BlobSimpleTest(string strName, int nDeviceID)
            : base(strName, nDeviceID)
        {
            m_blob = new Blob<T>(m_cuda, m_log);
            m_blob_preshaped = new Blob<T>(m_cuda, m_log, 2, 3, 4, 5);
            m_fEpsilon = 1e-6;
        }

        protected override void dispose()
        {
            m_blob.Dispose();
            m_blob_preshaped.Dispose();
            base.dispose();
        }

        public Blob<T> Blob
        {
            get { return m_blob; }
        }

        public Blob<T> BlobPreshaped
        {
            get { return m_blob_preshaped; }
        }

        public double Epsilon
        {
            get { return m_fEpsilon; }
        }

        public void TestInitialization()
        {
            m_log.CHECK(m_blob != null, "The blob should not be null.");
            m_log.CHECK(m_blob_preshaped != null, "The preshaped blob should not be null.");
            m_log.CHECK_EQ(2, m_blob_preshaped.num, "The preshaped blob should have num = 2.");
            m_log.CHECK_EQ(3, m_blob_preshaped.channels, "The preshaped blob should have channels = 2.");
            m_log.CHECK_EQ(4, m_blob_preshaped.height, "The preshaped blob should have height = 2.");
            m_log.CHECK_EQ(5, m_blob_preshaped.width, "The preshaped blob should have width = 2.");
            m_log.CHECK_EQ(120, m_blob_preshaped.count(), "The preshaped blob should have count() = 120.");
            m_log.CHECK_EQ(0, m_blob.num_axes, "The blob should have 0 axes.");
            m_log.CHECK_EQ(0, m_blob.count(), "The blob count() should be 0.");
        }

        public void TestPointersCPUGPU()
        {
            m_log.CHECK(m_blob_preshaped.gpu_data != 0, "The blob_preshaped gpu data should not be 0!");
            m_log.CHECK(m_blob_preshaped.gpu_diff != 0, "The blob_preshaped gpu diff should not be 0!");
            m_log.CHECK(m_blob_preshaped.mutable_gpu_data != 0, "The blob_preshaped mutable gpu data should not be 0!");
            m_log.CHECK(m_blob_preshaped.mutable_gpu_diff != 0, "The blob_preshaped mutable gpu diff should not be 0!");

            T[] rgData = m_blob_preshaped.update_cpu_data();
            T[] rgDiff = m_blob_preshaped.update_cpu_diff();

            m_log.CHECK_EQ(120, rgData.Length, "The data should have 120 elements.");
            m_log.CHECK_EQ(120, rgDiff.Length, "The data should have 120 elements.");
        }

        public void TestReshape()
        {
            m_blob.Reshape(2, 3, 4, 5);
            m_log.CHECK_EQ(2, m_blob.num, "The blob should have num = 2.");
            m_log.CHECK_EQ(3, m_blob.channels, "The blob should have channels = 2.");
            m_log.CHECK_EQ(4, m_blob.height, "The blob should have height = 2.");
            m_log.CHECK_EQ(5, m_blob.width, "The blob should have width = 2.");
            m_log.CHECK_EQ(120, m_blob.count(), "The blob should have count() = 120.");

            m_blob.ReshapeLike(m_blob_preshaped);
            m_log.CHECK_EQ(2, m_blob.num, "The blob should have num = 2.");
            m_log.CHECK_EQ(3, m_blob.channels, "The blob should have channels = 2.");
            m_log.CHECK_EQ(4, m_blob.height, "The blob should have height = 2.");
            m_log.CHECK_EQ(5, m_blob.width, "The blob should have width = 2.");
            m_log.CHECK_EQ(120, m_blob.count(), "The blob should have count() = 120.");
        }

        public void TestLegacyBlobProtoShapeEquals()
        {
            BlobProto bp = new BlobProto();

            // Reshape to (3 x 2)
            List<int> rgShape = new List<int>() { 3, 2 };
            m_blob.Reshape(rgShape);

            // (3 x 2) blob == (1 x 1 x 3 x 2) legacy blob.
            bp.num = 1;
            bp.channels = 1;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == true, "The blob shape does not equal the blob proto but should!");

            // (3 x 2) blob != (0 x 1 x 3 x 2) legacy blob.
            bp.num = 0;
            bp.channels = 1;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == false, "The blob shape should not equal the blob proto but does!");

            // (3 x 2) blob != (3 x 1 x 3 x 2) legacy blob.
            bp.num = 3;
            bp.channels = 1;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == false, "The blob shape should not equal the blob proto but does!");

            // (3 x 2) blob != (1 x 3 x 3 x 2) legacy blob.
            bp.num = 1;
            bp.channels = 3;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == false, "The blob shape should not equal the blob proto but does!");

            // Reshape to (1 x 3 x 2).
            rgShape.Insert(0, 1);
            m_blob.Reshape(rgShape);

            // (1 x 3 x 2) blob == (1 x 1 x 3 x 2) legacy blob.
            bp.num = 1;
            bp.channels = 1;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == true, "The blob shape does not equal the blob proto but should!");

            // Reshape to (2 x 3 x 2).
            rgShape[0] = 2;
            m_blob.Reshape(rgShape);

            // (2 x 3 x 2) blob != (1 x 1 x 3 x 2) legacy blob.
            bp.num = 1;
            bp.channels = 1;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == false, "The blob shape does not equal the blob proto but should!");

            // (2 x 3 x 2) blob == (1 x 2 x 3 x 2) legacy blob.
            bp.num = 1;
            bp.channels = 2;
            bp.height = 3;
            bp.width = 2;
            m_log.CHECK(m_blob.ShapeEquals(bp) == true, "The blob shape does not equal the blob proto but should!");
        }

        public void TestCopyFrom()
        {
            FillerParameter p = new FillerParameter("uniform");
            p.min = -3;
            p.max = 3;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, p);
            filler.Fill(m_blob_preshaped);

            m_blob.CopyFrom(m_blob_preshaped, false, true);

            m_log.CHECK_EQ(m_blob.num, m_blob_preshaped.num, "The blob nums should be the same.");
            m_log.CHECK_EQ(m_blob.channels, m_blob_preshaped.channels, "The blob channels should be the same.");
            m_log.CHECK_EQ(m_blob.height, m_blob_preshaped.height, "The blob height should be the same.");
            m_log.CHECK_EQ(m_blob.width, m_blob_preshaped.width, "The blob width should be the same.");
            m_log.CHECK_EQ(m_blob.count(), m_blob_preshaped.count(), "The blob counts should be the same.");

            T[] rgData1 = m_blob.update_cpu_data();
            T[] rgData2 = m_blob_preshaped.update_cpu_data();

            for (int n = 0; n < m_blob.num; n++)
            {
                for (int c = 0; c < m_blob.channels; c++)
                {
                    for (int h = 0; h < m_blob.height; h++)
                    {
                        for (int w = 0; w < m_blob.width; w++)
                        {
                            List<int> rgIdx = new List<int>() { n, c, h, w };

                            int nIdx1 = m_blob.offset(n, c, h, w);
                            int nIdx2 = m_blob.offset(rgIdx);

                            T fDataA1 = m_blob.data_at(n, c, h, w);
                            T fDataA2 = m_blob_preshaped.data_at(n, c, h, w);

                            T fDataB1 = m_blob.data_at(rgIdx);
                            T fDataB2 = m_blob_preshaped.data_at(rgIdx);

                            T fDataC1 = rgData1[nIdx1];
                            T fDataC2 = rgData2[nIdx1];

                            T fDataD1 = rgData1[nIdx2];
                            T fDataD2 = rgData2[nIdx2];

                            List<T> rgTest = new List<T>() { fDataA1, fDataA2, fDataB1, fDataB2, fDataC1, fDataC2, fDataD1, fDataD2 };
                            List<double> rgData = new List<double>();

                            foreach (T t in rgTest)
                            {
                                rgData.Add((double)Convert.ChangeType(t, typeof(double)));
                            }

                            for (int i = 1; i < rgData.Count; i++)
                            {
                                m_log.CHECK_EQ(rgData[i - 1], rgData[i], "The items at indexes " + (i - 1).ToString() + " and " + i.ToString() + " should be the same.");
                            }
                        }
                    }
                }
            }

            m_cuda.rng_gaussian(m_blob_preshaped.count(), 0, 2.0, m_blob_preshaped.mutable_gpu_diff);
            m_blob.CopyFrom(m_blob_preshaped, true, true);

            m_log.CHECK_EQ(m_blob.num, m_blob_preshaped.num, "The blob nums should be the same.");
            m_log.CHECK_EQ(m_blob.channels, m_blob_preshaped.channels, "The blob channels should be the same.");
            m_log.CHECK_EQ(m_blob.height, m_blob_preshaped.height, "The blob height should be the same.");
            m_log.CHECK_EQ(m_blob.width, m_blob_preshaped.width, "The blob width should be the same.");
            m_log.CHECK_EQ(m_blob.count(), m_blob_preshaped.count(), "The blob counts should be the same.");

            T[] rgDiff1 = m_blob.update_cpu_diff();
            T[] rgDiff2 = m_blob_preshaped.update_cpu_diff();

            for (int n = 0; n < m_blob.num; n++)
            {
                for (int c = 0; c < m_blob.channels; c++)
                {
                    for (int h = 0; h < m_blob.height; h++)
                    {
                        for (int w = 0; w < m_blob.width; w++)
                        {
                            List<int> rgIdx = new List<int>() { n, c, h, w };

                            int nIdx1 = m_blob.offset(n, c, h, w);
                            int nIdx2 = m_blob.offset(rgIdx);

                            T fDataA1 = m_blob.diff_at(n, c, h, w);
                            T fDataA2 = m_blob_preshaped.diff_at(n, c, h, w);

                            T fDataB1 = m_blob.diff_at(rgIdx);
                            T fDataB2 = m_blob_preshaped.diff_at(rgIdx);

                            T fDataC1 = rgDiff1[nIdx1];
                            T fDataC2 = rgDiff2[nIdx1];

                            T fDataD1 = rgDiff1[nIdx2];
                            T fDataD2 = rgDiff2[nIdx2];

                            List<T> rgTest = new List<T>() { fDataA1, fDataA2, fDataB1, fDataB2, fDataC1, fDataC2, fDataD1, fDataD2 };
                            List<double> rgData = new List<double>();

                            foreach (T t in rgTest)
                            {
                                rgData.Add((double)Convert.ChangeType(t, typeof(double)));
                            }

                            for (int i = 1; i < rgData.Count; i++)
                            {
                                m_log.CHECK_EQ(rgData[i - 1], rgData[i], "The items at indexes " + (i - 1).ToString() + " and " + i.ToString() + " should be the same.");
                            }
                        }
                    }
                }
            }
        }

        public void TestCopyFromChannels()
        {
            Blob<T> blobSrc = new Blob<T>(m_cuda, m_log, 100, 1, 20, 20);
            Blob<T> blobDst = new Blob<T>(m_cuda, m_log, 100, 3, 20, 20);

            try
            {
                FillerParameter p = new FillerParameter("uniform");
                p.min = -3;
                p.max = 3;
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, p);
                filler.Fill(blobSrc);

                for (int i = 0; i < 3; i++)
                {
                    blobDst.CopyFrom(blobSrc, 0, i);
                }

                for (int n = 0; n < blobSrc.num; n++)
                {
                    for (int c = 0; c < blobDst.channels; c++)
                    {
                        for (int h = 0; h < blobDst.height; h++)
                        {
                            for (int w = 0; w < blobDst.width; w++)
                            {
                                double dfDst = Utility.ConvertVal<T>(blobDst.data_at(n, c, h, w));
                                double dfSrc = Utility.ConvertVal<T>(blobSrc.data_at(n, 0, h, w));
                                m_log.CHECK_EQ(dfDst, dfSrc, "The data values do not match!");
                            }
                        }
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                blobSrc.Dispose();
                blobDst.Dispose();
            }
        }

        public void TestMath_SumOfSquares()
        {
            double fVal;
            m_blob.ReshapeLike(m_blob_preshaped);

            // Uninitialized Blob should have sum of squares == 0.
            fVal = (double)Convert.ChangeType(m_blob.sumsq_data(), typeof(double));
            m_log.CHECK_EQ(0, fVal, "The data sum of squares should be 0!");

            fVal = (double)Convert.ChangeType(m_blob.sumsq_diff(), typeof(double));
            m_log.CHECK_EQ(0, fVal, "The diff sum of squares should be 0!");

            FillerParameter p = new FillerParameter("uniform");
            p.min = -3;
            p.max = 3;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, p);
            filler.Fill(m_blob);
           
            double dfExpectedSumSq = 0;
            T[] rgData = m_blob.update_cpu_data();

            for (int i = 0; i < m_blob.count(); i++)
            {
                double dfVal = (double)Convert.ChangeType(rgData[i], typeof(double));
                dfExpectedSumSq += (dfVal * dfVal);
            }

            T fSumSqData = m_blob.sumsq_data();
            T fSumSqDiff = m_blob.sumsq_diff();

            m_log.EXPECT_NEAR(dfExpectedSumSq, (double)Convert.ChangeType(fSumSqData, typeof(double)), m_fEpsilon * dfExpectedSumSq);
            m_log.CHECK_EQ(0.0, (double)Convert.ChangeType(fSumSqDiff, typeof(double)), "The sum of squares for diff should be 0.");

            // Check sumsq_diff too.
            double dfDiffScaleFactor = 7.0;
            T fDiffScaleFactor = (T)Convert.ChangeType(dfDiffScaleFactor, typeof(T));

            m_cuda.scale(m_blob.count(), fDiffScaleFactor, m_blob.gpu_data, m_blob.mutable_gpu_diff);
            fSumSqDiff = m_blob.sumsq_diff();

            double dfExpectedSumSqDiff = dfExpectedSumSq * dfDiffScaleFactor * dfDiffScaleFactor;           
            double dfSumSqDiff = (double)Convert.ChangeType(fSumSqDiff, typeof(double));

            m_log.EXPECT_NEAR(dfExpectedSumSqDiff, dfSumSqDiff, m_fEpsilon * dfExpectedSumSqDiff);
        }

        public void TestMath_Asum()
        {
            double fVal;
            m_blob.ReshapeLike(m_blob_preshaped);

            // Uninitialized Blob should have asum == 0.
            fVal = (double)Convert.ChangeType(m_blob.asum_data(), typeof(double));
            m_log.CHECK_EQ(0, fVal, "The data asum should be 0!");

            fVal = (double)Convert.ChangeType(m_blob.asum_diff(), typeof(double));
            m_log.CHECK_EQ(0, fVal, "The diff asum should be 0!");

            FillerParameter p = new FillerParameter("uniform");
            p.min = -3;
            p.max = 3;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, p);
            filler.Fill(m_blob);

            double dfExpectedAsum = 0;

            T[] rgData = m_blob.update_cpu_data();

            for (int i = 0; i < m_blob.count(); i++)
            {
                double dfVal = (double)Convert.ChangeType(rgData[i], typeof(double));
                dfExpectedAsum += Math.Abs(dfVal);
            }

            T fAsumData = m_blob.asum_data();
            T fAsumDiff = m_blob.asum_diff();

            m_log.EXPECT_NEAR(dfExpectedAsum, (double)Convert.ChangeType(fAsumData, typeof(double)), m_fEpsilon * dfExpectedAsum);
            m_log.CHECK_EQ(0.0, (double)Convert.ChangeType(fAsumDiff, typeof(double)), "The asum for diff should be 0.");

            // Check sumsq_diff too.
            double dfDiffScaleFactor = 7.0;
            T fDiffScaleFactor = (T)Convert.ChangeType(dfDiffScaleFactor, typeof(T));

            m_cuda.scale(m_blob.count(), fDiffScaleFactor, m_blob.gpu_data, m_blob.mutable_gpu_diff);
            fAsumDiff = m_blob.asum_diff();

            double dfExpectedAsumDiff = dfExpectedAsum * dfDiffScaleFactor;
            double dfAsumDiff = (double)Convert.ChangeType(fAsumDiff, typeof(double));

            m_log.EXPECT_NEAR(dfExpectedAsumDiff, dfAsumDiff, m_fEpsilon * dfExpectedAsumDiff);
        }

        public void TestMath_Scale()
        {
            double fVal;
            m_blob.ReshapeLike(m_blob_preshaped);

            // Uninitialized Blob should have asum == 0.
            fVal = (double)Convert.ChangeType(m_blob.asum_data(), typeof(double));
            m_log.CHECK_EQ(0, fVal, "The data asum should be 0!");

            fVal = (double)Convert.ChangeType(m_blob.asum_diff(), typeof(double));
            m_log.CHECK_EQ(0, fVal, "The diff asum should be 0!");

            FillerParameter p = new FillerParameter("uniform");
            p.min = -3;
            p.max = 3;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, p);
            filler.Fill(m_blob);

            double dfExpectedAsum = 0;

            T[] rgData = m_blob.update_cpu_data();

            for (int i = 0; i < m_blob.count(); i++)
            {
                double dfVal = (double)Convert.ChangeType(rgData[i], typeof(double));
                dfExpectedAsum += Math.Abs(dfVal);
            }

            T fAsumDataBeforeScale = m_blob.asum_data();
            T fAsumDiffBeforeScale = m_blob.asum_diff();

            m_log.EXPECT_NEAR(dfExpectedAsum, (double)Convert.ChangeType(fAsumDataBeforeScale, typeof(double)), m_fEpsilon * dfExpectedAsum);
            m_log.CHECK_EQ(0.0, (double)Convert.ChangeType(fAsumDiffBeforeScale, typeof(double)), "The asum for diff should be 0.");

            double dfDataScaleFactor = 3.0;
            T fDataScaleFactor = (T)Convert.ChangeType(dfDataScaleFactor, typeof(T));
            m_blob.scale_data(fDataScaleFactor);

            T fAsumDataAfterScale = m_blob.asum_data();
            T fAsumDiffAfterScale = m_blob.asum_diff();

            m_log.EXPECT_NEAR(dfExpectedAsum * dfDataScaleFactor, (double)Convert.ChangeType(fAsumDataAfterScale, typeof(double)), m_fEpsilon * dfExpectedAsum);
            m_log.CHECK_EQ(0.0, (double)Convert.ChangeType(fAsumDiffAfterScale, typeof(double)), "The asum for diff should be 0.");

            // Check scale diff too.

            double dfDataToDiffScaleFactor = 7.0;
            T fDataToDiffScaleFactor = (T)Convert.ChangeType(dfDataToDiffScaleFactor, typeof(T));
            m_cuda.scale(m_blob.count(), fDataToDiffScaleFactor, m_blob.gpu_data, m_blob.mutable_gpu_diff);

            double dfExpectedAsumBeforeScale = (double)Convert.ChangeType(fAsumDataBeforeScale, typeof(double)) * dfDataScaleFactor;
            double dfAsumData = (double)Convert.ChangeType(m_blob.asum_data(), typeof(double));

            m_log.EXPECT_NEAR(dfExpectedAsumBeforeScale, dfAsumData, m_fEpsilon * dfExpectedAsumBeforeScale);

            double dfExpectedDiffAsumBeforeScale = (double)Convert.ChangeType(fAsumDataBeforeScale, typeof(double)) * dfDataScaleFactor * dfDataToDiffScaleFactor;
            double dfAsumDiff = (double)Convert.ChangeType(m_blob.asum_diff(), typeof(double));

            m_log.EXPECT_NEAR(dfExpectedDiffAsumBeforeScale, dfAsumDiff, m_fEpsilon * dfExpectedDiffAsumBeforeScale);

            double dfDiffScaleFactor = 3.0;
            T fDiffScaleFactor = (T)Convert.ChangeType(dfDiffScaleFactor, typeof(T));

            m_blob.scale_diff(fDiffScaleFactor);

            T fAsumData = m_blob.asum_data();
            double dfAsumData2 = (double)Convert.ChangeType(fAsumData, typeof(double));
            T fAsumDiff = m_blob.asum_diff();
            double dfAsumDiff2 = (double)Convert.ChangeType(fAsumDiff, typeof(double));

            double dfExpectedAsumData = (double)Convert.ChangeType(fAsumDataBeforeScale, typeof(double)) * dfDiffScaleFactor;
            double dfExpectedAsumDiff = dfExpectedDiffAsumBeforeScale * dfDiffScaleFactor;

            m_log.EXPECT_NEAR(dfExpectedAsumData, dfAsumData2, m_fEpsilon * dfExpectedAsumData);
            m_log.EXPECT_NEAR(dfExpectedAsumDiff, dfAsumDiff2, m_fEpsilon * dfExpectedAsumDiff);
        }

        private void Fill1(Blob<T> b)
        {
            double[] rgData = Utility.ConvertVec<T>(b.mutable_cpu_data);

            for (int n = 0; n < b.num; n++)
            {
                for (int c = 0; c < b.channels; c++)
                {
                    for (int h = 0; h < b.height; h++)
                    {
                        for (int w = 0; w < b.width; w++)
                        {
                            int nIdx = (n * b.channels * b.height * b.width) + (c * b.height * b.width) + (h * b.width) + w;
                            float fVal = (w - 0) / (float)b.width * 255.0f;
                            rgData[nIdx] = fVal;
                        }
                    }
                }
            }

            b.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
        }

        private void Fill2(Blob<T> b)
        {
            double[] rgData = Utility.ConvertVec<T>(b.mutable_cpu_data);

            for (int n = 0; n < b.num; n++)
            {
                for (int c = 0; c < b.channels; c++)
                {
                    for (int h = 0; h < b.height; h++)
                    {
                        for (int w = 0; w < b.width; w++)
                        {
                            int nIdx = (n * b.channels * b.height * b.width) + (c * b.height * b.width) + (h * b.width) + w;
                            float fVal = (h - 0) / (float)b.width * 255.0f;
                            rgData[nIdx] = fVal;
                        }
                    }
                }
            }

            b.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
        }

        private void Fill3(Blob<T> b)
        {
            double[] rgData = Utility.ConvertVec<T>(b.mutable_cpu_data);

            for (int n = 0; n < b.num; n++)
            {
                for (int c = 0; c < b.channels; c++)
                {
                    for (int h = 0; h < b.height; h++)
                    {
                        for (int w = 0; w < b.width; w++)
                        {
                            int nIdx = (n * b.channels * b.height * b.width) + (c * b.height * b.width) + (h * b.width) + w;
                            int nVal = Math.Max(h, w);
                            float fVal = (nVal - 0) / (float)b.width * 255.0f;
                            rgData[nIdx] = fVal;
                        }
                    }
                }
            }

            b.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
        }

        private void SaveImage(Blob<T> b, string strFile)
        {
            Datum d = ImageData.GetImageData(b.mutable_cpu_data, b.channels, b.height, b.width, false);
            Image img = ImageData.GetImage(d);
            img.Save(strFile);
            img.Dispose();
            return;
        }

        public void TestResize1()
        {
            m_blob_preshaped.Reshape(2, 3, 28, 28);
            Fill1(m_blob_preshaped);

            string strTemp = Path.GetTempPath();

            SaveImage(m_blob_preshaped, strTemp + "test1_preshaped.png");

            m_blob.Dispose();
            m_blob = m_blob_preshaped.Resize(new List<int>() { 2, 3, 14, 14 });

            SaveImage(m_blob, strTemp + "test1_14x14.png");

            m_blob.Dispose();
            m_blob = m_blob_preshaped.Resize(new List<int>() { 2, 3, 56, 56 });

            SaveImage(m_blob, strTemp + "test1_56x56.png");
        }

        public void TestResize2()
        {
            m_blob_preshaped.Reshape(2, 3, 28, 28);
            Fill2(m_blob_preshaped);

            string strTemp = Path.GetTempPath();

            SaveImage(m_blob_preshaped, strTemp + "test2_preshaped.png");

            m_blob.Dispose();
            m_blob = m_blob_preshaped.Resize(new List<int>() { 2, 3, 14, 14 });

            SaveImage(m_blob, strTemp + "test2_14x14.png");

            m_blob.Dispose();
            m_blob = m_blob_preshaped.Resize(new List<int>() { 2, 3, 56, 56 });

            SaveImage(m_blob, strTemp + "test2_56x56.png");
        }

        public void TestResize3()
        {
            m_blob_preshaped.Reshape(2, 3, 28, 28);
            Fill3(m_blob_preshaped);

            string strTemp = Path.GetTempPath();

            SaveImage(m_blob_preshaped, strTemp + "test3_preshaped.png");

            m_blob.Dispose();
            m_blob = m_blob_preshaped.Resize(new List<int>() { 2, 3, 14, 14 });

            SaveImage(m_blob, strTemp + "test3_14x14.png");

            m_blob.Dispose();
            m_blob = m_blob_preshaped.Resize(new List<int>() { 2, 3, 56, 56 });

            SaveImage(m_blob, strTemp + "test3_56x56.png");
        }
    }
}
