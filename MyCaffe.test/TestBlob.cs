﻿using System;
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
using MyCaffe.layers;
using System.Text.RegularExpressions;
using System.Diagnostics;

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
        public void TestCleanup()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCleanup();
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
        public void TestChannelDuplicate()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestChannelDuplicate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyAndTransposeHeightWidth()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyTransposeHeightWidth(1, 1, 4, 4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyAndTransposeHeightWidthBatch()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyTransposeHeightWidth(3, 1, 4, 4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyAndTransposeHeightWidthMgtrN()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyTransposeHeightWidth(1, 1, 4, 2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyAndTransposeHeightWidthBatchMgtrN()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyTransposeHeightWidth(3, 1, 4, 2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyAndTransposeHeightWidthMltN()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyTransposeHeightWidth(1, 1, 2, 4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyAndTransposeHeightWidthBatchMltN()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestCopyTransposeHeightWidth(3, 1, 2, 4);
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
        public void TestMath_ScaleToRange()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMath_ScaleToRange();
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

        [TestMethod]
        public void TestSetPixel()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestSetPixel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetPixelAtOffset()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestSetPixelAtOffset();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLoadFromNumpy()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestLoadFromNumpy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSaveToNumpy()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestSaveToNumpy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMul()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMatMul(1, 1, false, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMulEx()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMatMulEx();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMulBatch()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMatMul(3, 1, false, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatMulGrad()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMatMulGrad(1, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
        [TestMethod]
        public void TestMatMulGradBatch()
        {
            BlobSimpleTest test = new BlobSimpleTest();

            try
            {
                foreach (IBlobSimpleTest t in test.Tests)
                {
                    t.TestMatMulGrad(3, 1);
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
        void TestCleanup();
        void TestPointersCPUGPU();
        void TestReshape();
        void TestLegacyBlobProtoShapeEquals();
        void TestCopyFrom();
        void TestCopyFromChannels();
        void TestCopyTransposeHeightWidth(int nBatch, int nChannels, int nM, int nN);
        void TestMath_SumOfSquares();
        void TestMath_Asum();
        void TestMath_Scale();
        void TestMath_ScaleToRange();
        void TestResize1();
        void TestResize2();
        void TestResize3();
        void TestSetPixel();
        void TestSetPixelAtOffset();
        void TestLoadFromNumpy();
        void TestSaveToNumpy();
        void TestChannelDuplicate();
        void TestMatMul(int nBatch, int nChannels, bool bTransA, bool bTransB);
        void TestMatMulEx();
        void TestMatMulGrad(int nBatch, int nChannels);
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

        public void TestCleanup()
        {
            CudaDnn<T> cuda = new CudaDnn<T>(0);
            cuda.debug();

            Blob<T> blob = new Blob<T>(cuda, m_log);

            blob.Reshape(10, 10, 1000, 100);

            cuda.Dispose();

            try
            {
                // Dispose of blob AFTER disposing underlying CudaDnn.
                blob.Dispose();
            }
            catch (Exception excpt)
            {
                // This exception is expected.
            }
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

        public void TestMath_ScaleToRange()
        {
            Blob<T> blob = new Blob<T>(m_cuda, m_log, 100, 1, 20, 20);
            Blob<T> blobScaled = new Blob<T>(m_cuda, m_log, 100, 3, 20, 20);

            try
            {
                FillerParameter p = new FillerParameter("uniform");
                p.min = -3;
                p.max = 3;
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, p);
                filler.Fill(blob);

                blobScaled.CopyFrom(blob, false, true);
                blobScaled.scale_to_range(-1, 1);

                double[] rgDfS = Utility.ConvertVec<T>(blobScaled.mutable_cpu_data);
                double[] rgDfO = Utility.ConvertVec<T>(blob.mutable_cpu_data);
                double dfMin = blob.min_data;
                double dfMax = blob.max_data;
                double dfRange = dfMax - dfMin;
                double dfMinS = -1;
                double dfMaxS = 1;
                double dfRangeS = dfMaxS - dfMinS;
                double dfScale = dfRangeS / dfRange;

                for (int i = 0; i < rgDfS.Length; i++)
                {
                    double dfValS = rgDfS[i];
                    double dfValO = rgDfO[i];

                    double dfVal = (((dfValO - dfMin) * dfScale) + dfMinS);

                    m_log.EXPECT_NEAR(dfVal, dfValS, 0.000001, "The values are not as expected.");
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                blob.Dispose();
                blobScaled.Dispose();
            }
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

        public void TestSetPixel()
        {
            m_blob_preshaped.Reshape(1, 3, 28, 28);
            Fill3(m_blob_preshaped);

            double[] rgdf = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    testPixel(x, y, TransformationParameter.COLOR_ORDER.RGB);
                    testPixel(x, y, TransformationParameter.COLOR_ORDER.BGR);
                }
            }

            double[] rgdf1 = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);

            for (int i = 0; i < rgdf.Length; i++)
            {
                m_log.CHECK_EQ(rgdf[i], rgdf1[i], "The values are not as expected!");
            }
        }

        private void testPixel(int nX, int nY, TransformationParameter.COLOR_ORDER order)
        {
            Tuple<T, T, T> pixel = m_blob_preshaped.SetPixel(nX, nY, 10, 8, 9, order);

            double dfR = Utility.ConvertVal<T>(pixel.Item1);
            double dfG = Utility.ConvertVal<T>(pixel.Item2);
            double dfB = Utility.ConvertVal<T>(pixel.Item3);

            Tuple<T, T, T> pixel2 = m_blob_preshaped.SetPixel(nX, nY, (byte)dfR, (byte)dfG, (byte)dfB, order);

            double dfR2 = Utility.ConvertVal<T>(pixel2.Item1);
            double dfG2 = Utility.ConvertVal<T>(pixel2.Item2);
            double dfB2 = Utility.ConvertVal<T>(pixel2.Item3);

            m_log.CHECK_EQ(dfR2, 10, "The values are not as expected!");
            m_log.CHECK_EQ(dfG2, 8, "The values are not as expected!");
            m_log.CHECK_EQ(dfB2, 9, "The values are not as expected!");

            m_blob_preshaped.SetPixel(nX, nY, pixel, false, order);

            pixel2 = m_blob_preshaped.SetPixel(nX, nY, (byte)dfR, (byte)dfG, (byte)dfB, order);

            dfR2 = Utility.ConvertVal<T>(pixel2.Item1);
            dfG2 = Utility.ConvertVal<T>(pixel2.Item2);
            dfB2 = Utility.ConvertVal<T>(pixel2.Item3);

            m_log.CHECK_EQ(dfR2, dfR, "The values are not as expected!");
            m_log.CHECK_EQ(dfG2, dfG, "The values are not as expected!");
            m_log.CHECK_EQ(dfB2, dfB, "The values are not as expected!");

            m_blob_preshaped.SetPixel(nX, nY, pixel, false, order);
        }


        public void TestSetPixelAtOffset()
        {
            int nOffset = 3 * 28 * 28;
            m_blob_preshaped.Reshape(2, 3, 28, 28);
            Fill3(m_blob_preshaped);

            double[] rgdf = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    testPixel(nOffset, x, y, TransformationParameter.COLOR_ORDER.RGB, rgdf);
                    testPixel(nOffset, x, y, TransformationParameter.COLOR_ORDER.BGR, rgdf);
                }
            }

            double[] rgdf1 = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);

            for (int i = 0; i < rgdf.Length; i++)
            {
                m_log.CHECK_EQ(rgdf[i], rgdf1[i], "The values are not as expected!");
            }
        }

        private void testPixel(int nOffset, int nX, int nY, TransformationParameter.COLOR_ORDER order, double[] rgdf)
        {
            Tuple<T, T, T> val = new Tuple<T, T, T>(Utility.ConvertVal<T>(10.0), Utility.ConvertVal<T>(8.0), Utility.ConvertVal<T>(9.0));
            Tuple<T, T, T> pixel = m_blob_preshaped.SetPixel(nX, nY, val, true, order, nOffset);

            double[] rgdf1 = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);
            for (int i = 0; i < nOffset; i++)
            {
                m_log.CHECK_EQ(rgdf[i], rgdf1[i], "The values pre offset should not change!");
            }

            Tuple<T, T, T> pixel2 = m_blob_preshaped.SetPixel(nX, nY, pixel, true, order, nOffset);

            rgdf1 = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);
            for (int i = 0; i < nOffset; i++)
            {
                m_log.CHECK_EQ(rgdf[i], rgdf1[i], "The values pre offset should not change!");
            }

            double dfR2 = Utility.ConvertVal<T>(pixel2.Item1);
            double dfG2 = Utility.ConvertVal<T>(pixel2.Item2);
            double dfB2 = Utility.ConvertVal<T>(pixel2.Item3);

            m_log.CHECK_EQ(dfR2, 10, "The values are not as expected!");
            m_log.CHECK_EQ(dfG2, 8, "The values are not as expected!");
            m_log.CHECK_EQ(dfB2, 9, "The values are not as expected!");

            m_blob_preshaped.SetPixel(nX, nY, pixel, false, order, nOffset);

            rgdf1 = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);
            for (int i = 0; i < nOffset; i++)
            {
                m_log.CHECK_EQ(rgdf[i], rgdf1[i], "The values pre offset should not change!");
            }

            pixel2 = m_blob_preshaped.SetPixel(nX, nY, val, true, order, nOffset);

            rgdf1 = Utility.ConvertVec<T>(m_blob_preshaped.mutable_cpu_data);
            for (int i = 0; i < nOffset; i++)
            {
                m_log.CHECK_EQ(rgdf[i], rgdf1[i], "The values pre offset should not change!");
            }

            dfR2 = Utility.ConvertVal<T>(pixel2.Item1);
            dfG2 = Utility.ConvertVal<T>(pixel2.Item2);
            dfB2 = Utility.ConvertVal<T>(pixel2.Item3);

            double dfR = Utility.ConvertVal<T>(pixel.Item1);
            double dfG = Utility.ConvertVal<T>(pixel.Item2);
            double dfB = Utility.ConvertVal<T>(pixel.Item3);

            m_log.CHECK_EQ(dfR2, dfR, "The values are not as expected!");
            m_log.CHECK_EQ(dfG2, dfG, "The values are not as expected!");
            m_log.CHECK_EQ(dfB2, dfB, "The values are not as expected!");

            m_blob_preshaped.SetPixel(nX, nY, pixel, false, order, nOffset);
        }

        public void TestLoadFromNumpy()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\numpy\\";
            Blob<T> blob = new Blob<T>(m_cuda, m_log);

            try
            {
                blob.LoadFromNumpy(strPath + "numpy_1.npy");

                string[] rgstrLines = File.ReadAllLines(strPath + "numpy_1.txt");
                List<int> rgShape = new List<int>();

                for (int i=0; i<rgstrLines.Length; i++)
                { 
                    float fVal = float.Parse(rgstrLines[i]);
                    rgShape.Add((int)fVal);
                }

                if (!blob.CompareShape(rgShape))
                    m_log.FAIL("The numpy shape and expected shape do not match!");
            }
            finally
            {
                blob.Dispose();
            }
        }

        public void TestSaveToNumpy()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\numpy\\";
            Blob<T> blob = new Blob<T>(m_cuda, m_log);
            Blob<T> blob2 = new Blob<T>(m_cuda, m_log);

            try
            {
                string strTempFile = strPath + "temp.npy";
                if (File.Exists(strTempFile))
                    File.Delete(strTempFile);
                
                blob.LoadFromNumpy(strPath + "numpy_1.npy");

                string[] rgstrLines = File.ReadAllLines(strPath + "numpy_1.txt");
                List<int> rgShape = new List<int>();

                for (int i = 0; i < rgstrLines.Length; i++)
                {
                    float fVal = float.Parse(rgstrLines[i]);
                    rgShape.Add((int)fVal);
                }

                if (!blob.CompareShape(rgShape))
                    m_log.FAIL("The numpy shape and expected shape do not match!");

                // Change the data at the start of each channel.
                int nSpatialDim = blob.count(1);
                for (int i = 0; i < blob.num; i++)
                {
                    int nIdx = i * nSpatialDim;
                    blob.SetData(i, nIdx);
                }

                blob.SaveToNumpy(strTempFile);
                blob2.LoadFromNumpy(strTempFile);

                if (!blob.CompareShape(blob2.shape()))
                    m_log.FAIL("The numpy shape and expected shape do not match!");

                for (int i = 0; i < blob2.num; i++)
                {
                    int nIdx = i * nSpatialDim;
                    float fVal = Utility.ConvertValF<T>(blob.GetData(nIdx));
                    
                    if ((int)fVal != i)
                        m_log.FAIL("The data at index '" + i.ToString() + "' does not match!");
                }
            }
            finally
            {
                blob.Dispose();
                blob2.Dispose();
            }
        }

        public void TestChannelDuplicate()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\numpy\\";
            string strFile = strPath + "channel_duplicate_test.npy";
            Blob<T> blob = new Blob<T>(m_cuda, m_log);
            Blob<T> blobDup = new Blob<T>(m_cuda, m_log);
            Blob<T> blobExp = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);

            try
            {
                blob.LoadFromNumpy(strFile);
                float[] rgData = Utility.ConvertVecF<T>(blob.mutable_cpu_data);
                int nDupCount = blob.channels;
                float[] rgExpected = new float[rgData.Length * nDupCount];

                for (int n = 0; n < blob.num; n++)
                {
                    for (int c = 0; c < blob.channels; c++)
                    {
                        int nSrcIdx = blob.offset(n, c);
                        float fVal = rgData[nSrcIdx];

                        for (int d = 0; d < nDupCount; d++)
                        {
                            int nDstIdx = n * nDupCount * blob.channels + d * blob.channels + c;
                            rgExpected[nDstIdx] = fVal;
                        }
                    }
                }

                string strDbgPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\mycaffe\\test_data\\results\\tokenized_data_test\\";
                if (!Directory.Exists(strDbgPath))
                    Directory.CreateDirectory(strDbgPath);

                blob.SaveToImage(strDbgPath + "__original.png", false);
                
                blobExp.Reshape(blob.num, blob.channels, nDupCount, 1);
                blobExp.mutable_cpu_data = Utility.ConvertVec<T>(rgExpected);
                blobExp.SaveToImage(strDbgPath + "__exp_dup.png", false);

                blobDup.ReshapeLike(blobExp);
                int nCount1 = blobDup.count(2);
                m_cuda.channel_duplicate(blobDup.count(), blob.num, blob.channels, nCount1, blob.gpu_data, blobDup.mutable_gpu_data);

                blobDup.SaveToImage(strDbgPath + "__actual_dup.png", false);
                m_log.CHECK(blobDup.Compare(blobExp, blobWork), "The blob is not as expected!");
            }
            finally
            {
                blob.Dispose();
                blobDup.Dispose();
                blobExp.Dispose();
                blobWork.Dispose();
            }
        }

        private float[] matmul(int nBatch, int nChannels, int nM, int nK, int nN, T[] rgA, T[] rgB)
        {
            return matmul(nBatch, nChannels, nM, nN, nK,
                Utility.ConvertVecF<T>(rgA),
                Utility.ConvertVecF<T>(rgB));
        }

        private float[] matmul(int nBatch, int nChannels, int nM, int nK, int nN, float[] rgA, float[] rgB)
        {
            int nCount = nBatch * nChannels;
            float[] rgC = new float[nCount * nM * nN];

            for (int i = 0; i < nCount; i++)
            {
                for (int m = 0; m < nM; m++)
                {
                    for (int n = 0; n < nN; n++)
                    {
                        int nIdxC = (i * nM * nN) + m * nN + n;
                        float fSum = 0;

                        for (int k = 0; k < nK; k++)
                        {
                            int nIdxA = (i * nM * nK) + m * nK + k;
                            int nIdxB = (i * nK * nN) + k * nN + n;

                            fSum += rgA[nIdxA] * rgB[nIdxB];
                        }

                        rgC[nIdxC] = fSum;
                    }
                }
            }

            return rgC;
        }

        private void expand(Blob<T> b, int nBatch, int nChannels)
        {
            int nExpand = nBatch * nChannels;
            if (nExpand == 1)
                return;

            int nCount = b.count();
          
            List<int> rgShape = Utility.Clone<int>(b.shape());
            rgShape[0] = nBatch;
            rgShape[1] = nChannels;

            float[] rgData = Utility.ConvertVecF<T>(b.mutable_cpu_data);
            float[] rgDiff = Utility.ConvertVecF<T>(b.mutable_cpu_diff);

            b.Reshape(rgShape);

            float[] rgData1 = new float[rgData.Length * nExpand];
            float[] rgDiff1 = new float[rgDiff.Length * nExpand];

            for (int i = 0; i < nExpand; i++)
            {
                Array.Copy(rgData, 0, rgData1, i * nCount, nCount);
                Array.Copy(rgDiff, 0, rgDiff1, i * nCount, nCount);
            }

            b.mutable_cpu_data = Utility.ConvertVec<T>(rgData1);
            b.mutable_cpu_diff = Utility.ConvertVec<T>(rgDiff1);
        }

        private void compare(T[] rg1, T[] rg2)
        {
            compare(Utility.ConvertVecF<T>(rg1),
                    Utility.ConvertVecF<T>(rg2));
        }

        private void compare(float[] rg1, float[] rg2)
        {
            m_log.CHECK_EQ(rg1.Length, rg2.Length, "The lengths should be equal!");

            for (int i = 0; i < rg1.Length; i++)
            {
                m_log.CHECK_EQ(rg1[i], rg2[i], "The items at index '" + i.ToString() + "' should be equal!");
            }
        }

        private void compare(double[] rg1, float[] rg2)
        {
            m_log.CHECK_EQ(rg1.Length, rg2.Length, "The lengths should be equal!");

            for (int i = 0; i < rg1.Length; i++)
            {
                m_log.CHECK_EQ(rg1[i], rg2[i], "The items at index '" + i.ToString() + "' should be equal!");
            }
        }

        public void TestCopyTransposeHeightWidth(int nBatch, int nChannels, int nM, int nN)
        {            
            Blob<T> blobA = new Blob<T>(m_cuda, m_log);
            Blob<T> blobB = new Blob<T>(m_cuda, m_log);
            Blob<T> blobC = new Blob<T>(m_cuda, m_log);

            LayerParameter transpose_param = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, "transpose");
            transpose_param.transpose_param.dim[2] = 3;
            transpose_param.transpose_param.dim[3] = 2;
            Layer<T> transpose = Layer<T>.Create(m_cuda, m_log, transpose_param, null);

            try
            {
                // Create the A and B blobs in row-major ordering
                blobA.Reshape(nBatch, nChannels, nM, nN);
                blobB.Reshape(nBatch, nChannels, nN, nM);
                blobB.SetData(0.0);
                
                // Check Data
                float[] rgA = Utility.ConvertVecF<T>(blobA.mutable_cpu_data);
                for (int i = 0; i < rgA.Length; i++)
                {
                    rgA[i] = i;
                }
                blobA.mutable_cpu_data = Utility.ConvertVec<T>(rgA);

                blobB.CopyFromAndTransposeHeightWidth(blobA, false, true);

                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                colBtm.Add(blobA);
                colTop.Add(blobC);

                transpose.Forward(colBtm, colTop);

                float[] rgB = Utility.ConvertVecF<T>(blobB.mutable_cpu_data);
                float[] rgC = Utility.ConvertVecF<T>(blobB.mutable_cpu_data);
                compare(rgB, rgC);

                // Check Diff
                rgC = Utility.ConvertVecF<T>(blobC.mutable_cpu_diff);
                for (int i = 0; i < rgC.Length; i++)
                {
                    rgC[i] = i * 100;
                }
                blobC.mutable_cpu_diff = Utility.ConvertVec<T>(rgC);

                blobB.CopyFromAndTransposeHeightWidth(blobC, true, true);

                colBtm.Clear();
                colBtm.Add(blobA);
                colTop.Clear();
                colTop.Add(blobC);
                
                transpose.Backward(colTop, new List<bool>() { true }, colBtm);

                rgA = Utility.ConvertVecF<T>(blobB.mutable_cpu_diff);
                rgB = Utility.ConvertVecF<T>(blobB.mutable_cpu_diff);
                compare(rgA, rgB);
            }
            finally
            {
                blobA.Dispose();
                blobB.Dispose();
                blobC.Dispose();
                transpose.Dispose();
            }
        }

        public void TestMatMul(int nBatch, int nChannels, bool bTransA, bool bTransB)
        {
            int nM = 4;
            int nN = 4;
            int nK = 2;
            Blob<T> blobA = new Blob<T>(m_cuda, m_log);
            Blob<T> blobB = new Blob<T>(m_cuda, m_log);
            Blob<T> blobC = new Blob<T>(m_cuda, m_log);

            try
            {
                // Create the A and B blobs in row-major ordering
                blobA.Reshape(nBatch, nChannels, nM, nK);
                blobB.Reshape(nBatch, nChannels, nK, nN);

                float[] rgA = Utility.ConvertVecF<T>(blobA.mutable_cpu_data);
                float[] rgB = Utility.ConvertVecF<T>(blobB.mutable_cpu_data);
                blobC.SetData(0);

                for (int i = 0; i < rgA.Length; i++)
                {
                    rgA[i] = i;
                }
                blobA.mutable_cpu_data = Utility.ConvertVec<T>(rgA);

                for (int i = 0; i < rgB.Length; i++)
                {
                    rgB[i] = i + 100;
                }
                blobB.mutable_cpu_data = Utility.ConvertVec<T>(rgB);

                expand(blobA, nBatch, nChannels);
                expand(blobB, nBatch, nChannels);


                //==================================
                // MatMul Operation
                //==================================

                blobC.MatMul(blobA, blobB, true);

                // Verify values.
                float[] rgC = Utility.ConvertVecF<T>(blobC.mutable_cpu_data);
                m_log.CHECK_EQ(rgC.Length, nM * nN * nBatch * nChannels, "The C matrix is not the expected size!");

                List<int> rgShape = new List<int>() { nBatch, nChannels, nM, nN };
                m_log.CHECK(blobC.CompareShape(rgShape), "The C matrix shape is not the expected size!");

                float[] rgExpected = matmul(nBatch, nChannels, nM, nK, nN, rgA, rgB);
                compare(rgExpected, rgC);
            }
            finally
            {
                blobA.Dispose();
                blobB.Dispose();
                blobC.Dispose();
            }
        }

        public void TestMatMulGrad(int nBatch, int nChannels)
        {
            int nM = 4;
            int nN = 4;
            int nK = 2;
            Blob<T> blobA = new Blob<T>(m_cuda, m_log);
            Blob<T> blobB = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);
            Blob<T> blobAt = new Blob<T>(m_cuda, m_log);
            Blob<T> blobBt = new Blob<T>(m_cuda, m_log);
            Blob<T> blobC = new Blob<T>(m_cuda, m_log);

            LayerParameter transpose_param = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, "transpose");
            transpose_param.transpose_param.dim[2] = 3;
            transpose_param.transpose_param.dim[3] = 2;
            Layer<T> transpose = Layer<T>.Create(m_cuda, m_log, transpose_param, null);

            try
            {
                // Create the A and B blobs in row-major ordering
                blobA.Reshape(1, 1, nM, nK);
                blobB.Reshape(1, 1, nK, nN);

                float[] rgA = Utility.ConvertVecF<T>(blobA.mutable_cpu_data);
                float[] rgB = Utility.ConvertVecF<T>(blobB.mutable_cpu_data);
                blobC.SetData(0);

                for (int i = 0; i < rgA.Length; i++)
                {
                    rgA[i] = i;
                }
                blobA.mutable_cpu_data = Utility.ConvertVec<T>(rgA);

                for (int i = 0; i < rgB.Length; i++)
                {
                    rgB[i] = i + 100;
                }
                blobB.mutable_cpu_data = Utility.ConvertVec<T>(rgB);

                expand(blobA, nBatch, nChannels);
                expand(blobB, nBatch, nChannels);

                List<int> rgShape = new List<int>() { nBatch, nChannels, nM, nN };
                blobC.Reshape(rgShape);
                blobC.SetData(0.0);
                blobC.SetDiff(1.0);

                //==================================
                // MatMul Operation
                //==================================

                blobC.MatMulGrad(blobA, blobB, blobWork);

                // Verify values.
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                colBtm.Clear();
                colBtm.Add(blobB);
                colTop.Clear();
                colTop.Add(blobBt);
                transpose.Setup(colBtm, colTop);
                transpose.Forward(colBtm, colTop);

                float[] rgAgrad = matmul(nBatch, nChannels, nM, nK, nN, blobC.mutable_cpu_diff, blobBt.mutable_cpu_data);
                compare(rgAgrad, Utility.ConvertVecF<T>(blobA.mutable_cpu_diff));

                colBtm.Clear();
                colBtm.Add(blobA);
                colTop.Clear();
                colTop.Add(blobAt);
                transpose.Setup(colBtm, colTop);
                transpose.Forward(colBtm, colTop);

                float[] rgBgrad = matmul(nBatch, nChannels, 2, 4, 4, blobAt.mutable_cpu_data, blobC.mutable_cpu_diff);
                compare(rgBgrad, Utility.ConvertVecF<T>(blobB.mutable_cpu_diff));
            }
            finally
            {
                blobA.Dispose();
                blobB.Dispose();
                blobC.Dispose();
                blobAt.Dispose();
                blobBt.Dispose();
                blobWork.Dispose();

                transpose.Dispose();
            }
        }

        public void TestMatMulEx()
        {
            Blob<T> blobA = new Blob<T>(m_cuda, m_log);
            Blob<T> blobB = new Blob<T>(m_cuda, m_log);
            Blob<T> blobC = new Blob<T>(m_cuda, m_log);

            try
            {
                int nM = 13;
                int nN = 13;
                int nK = 32;

                blobA.Reshape(1, 1, nM, nK);
                blobB.Reshape(1, 1, nK, nN);

                float[] rgA = Utility.ConvertVecF<T>(blobA.mutable_cpu_data);
                float[] rgB = Utility.ConvertVecF<T>(blobB.mutable_cpu_data);

                for (int i = 0; i < nM * nK; i++)
                {
                    rgA[i] = i;
                }

                for (int i = 0; i < nK * nN; i++)
                {
                    rgB[i] = i * 2;
                }

                blobA.mutable_cpu_data = Utility.ConvertVec<T>(rgA);
                blobB.mutable_cpu_data = Utility.ConvertVec<T>(rgB);

                blobC.MatMul(blobA, blobB, true);

                float[] rgC = Utility.ConvertVecF(blobC.mutable_cpu_data);
                float[] rgC1 = matmul(1, 1, nM, nK, nN, rgA, rgB);

                for (int i = 0; i < rgC.Length; i++)
                {
                    float fC = rgC[i];
                    float fC1 = rgC1[i];
                    float fDiff = Math.Abs(fC - fC1);
                    float fErr = 1e-08f;

                    if (fDiff > fErr)
                        m_log.FAIL("The values of the matmul are not as expected!");
                }
            }
            finally
            {
                blobA.Dispose();
                blobB.Dispose();
                blobC.Dispose();
            }
        }
    }
}
