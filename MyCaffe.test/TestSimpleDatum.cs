using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSimpleDatum
    {
        [TestMethod]
        public void TestConstructorFloatArray()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestConstructorFloatArray();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstructorDoubleArray()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestConstructorDoubleArray();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstructorByteArray()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestConstructorByteArray();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstructorFloatList()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestConstructorFloatList();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstructorDoubleList()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestConstructorDoubleList();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConstructorByteList()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestConstructorByteList();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetDataFloat()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestSetDataFloat();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetDataDouble()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestSetDataDouble();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetDataByte()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestSetDataByte();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTranspose()
        {
            SimpleDatumTest test = new SimpleDatumTest();

            try
            {
                foreach (ISimpleDatumTest t in test.Tests)
                {
                    t.TestTranspose();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ISimpleDatumTest : ITest
    {
        void TestConstructorFloatArray();
        void TestConstructorDoubleArray();
        void TestConstructorByteArray();
        void TestConstructorFloatList();
        void TestConstructorDoubleList();
        void TestConstructorByteList();
        void TestSetDataFloat();
        void TestSetDataDouble();
        void TestSetDataByte();
        void TestTranspose();
    }

    class SimpleDatumTest : TestBase
    {
        public SimpleDatumTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SimpleDatum Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SimpleDatumTest<double>(strName, nDeviceID, engine);
            else
                return new SimpleDatumTest<float>(strName, nDeviceID, engine);
        }
    }

    class SimpleDatumTest<T> : TestEx<T>, ISimpleDatumTest
    {
        int m_nChannel = 3;
        int m_nHeight = 10;
        int m_nWidth = 10;
        float[] m_rgfData;
        double[] m_rgdfData;
        byte[] m_rgbData;

        public SimpleDatumTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
            Fill();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("gaussian");
            p.mean = 0;
            p.std = 1.0;
            return p;
        }

        public void Fill()
        {
            int nLen = m_nChannel * m_nHeight * m_nWidth;
            m_rgfData = new float[nLen];
            m_rgdfData = new double[nLen];
            m_rgbData = new byte[nLen];

            for (int c = 0; c < m_nChannel; c++)
            {
                for (int h = 0; h < m_nHeight; h++)
                {
                    for (int w = 0; w < m_nWidth; w++)
                    {
                        int nIdx = c * m_nHeight * m_nWidth + h * m_nWidth + w;
                        m_rgdfData[nIdx] = ((double)nIdx / nLen) * 255.0;
                        m_rgfData[nIdx] = (float)m_rgdfData[nIdx];
                        m_rgbData[nIdx] = (byte)m_rgfData[nIdx];
                    }
                }
            }
        }

        public void TestConstructorFloatArray()
        {
            SimpleDatum sd = new SimpleDatum(true, 10, 10, 10, 1, DateTime.MinValue, m_rgfData, 0, false, -1);

            m_log.CHECK(sd.IsRealData, "The data should be real!");
            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF != null, "The real data 'f' should NOT be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgfData[i], sd.RealDataF[i], "The data items should be equal!");

                float fVal = sd.GetDataAtF(i);
                m_log.CHECK_EQ(m_rgfData[i], fVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                fVal = (float)Convert.ChangeType(tVal, typeof(float));
                m_log.CHECK_EQ(m_rgfData[i], fVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            float[] rgfData = convertF(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                m_log.CHECK_EQ(rgfData[i], m_rgfData[i], "The data items should be equal!");
            }
        }

        public void TestConstructorDoubleArray()
        {
            SimpleDatum sd = new SimpleDatum(true, 10, 10, 10, 1, DateTime.MinValue, m_rgdfData, 0, false, -1);

            m_log.CHECK(sd.IsRealData, "The data should be real!");
            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");
            m_log.CHECK(sd.RealDataD != null, "The real data 'd' should NOT be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgdfData[i], sd.RealDataD[i], "The data items should be equal!");

                double dfVal = sd.GetDataAtD(i);
                m_log.CHECK_EQ(m_rgdfData[i], dfVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                dfVal = (double)Convert.ChangeType(tVal, typeof(double));
                m_log.EXPECT_EQUAL<T>(m_rgdfData[i], dfVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            double[] rgdfData = convert(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                m_log.EXPECT_EQUAL<T>(rgdfData[i], m_rgdfData[i], "The data items should be equal!");
            }
        }

        public void TestConstructorByteArray()
        {
            SimpleDatum sd = new SimpleDatum(false, 10, 10, 10, 1, DateTime.MinValue, m_rgbData, 0, false, -1);

            m_log.CHECK(!sd.IsRealData, "The data should not be real!");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should NOT be null.");
            m_log.CHECK(sd.ByteData != null, "The byte data should be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgbData[i], sd.ByteData[i], "The data items should be equal!");

                byte bVal = sd.GetDataAtByte(i);
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                bVal = (byte)Convert.ChangeType(tVal, typeof(byte));
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            for (int i = 0; i < rgData.Length; i++)
            {
                T tVal = rgData[i];
                byte bVal = (byte)Convert.ChangeType(tVal, typeof(byte));
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");
            }
        }

        public void TestConstructorFloatList()
        {
            SimpleDatum sd = new SimpleDatum(true, 10, 10, 10, 1, DateTime.MinValue, new List<float>(m_rgfData), 0, false, -1);

            m_log.CHECK(sd.IsRealData, "The data should be real!");
            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF != null, "The real data 'f' should NOT be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgfData[i], sd.RealDataF[i], "The data items should be equal!");

                float fVal = sd.GetDataAtF(i);
                m_log.CHECK_EQ(m_rgfData[i], fVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                fVal = (float)Convert.ChangeType(tVal, typeof(float));
                m_log.CHECK_EQ(m_rgfData[i], fVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            float[] rgfData = convertF(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                m_log.CHECK_EQ(rgfData[i], m_rgfData[i], "The data items should be equal!");
            }
        }

        public void TestConstructorDoubleList()
        {
            SimpleDatum sd = new SimpleDatum(true, 10, 10, 10, 1, DateTime.MinValue, new List<double>(m_rgdfData), 0, false, -1);

            m_log.CHECK(sd.IsRealData, "The data should be real!");
            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");
            m_log.CHECK(sd.RealDataD != null, "The real data 'd' should NOT be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgdfData[i], sd.RealDataD[i], "The data items should be equal!");

                double dfVal = sd.GetDataAtD(i);
                m_log.CHECK_EQ(m_rgdfData[i], dfVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                dfVal = (double)Convert.ChangeType(tVal, typeof(double));
                m_log.EXPECT_EQUAL<T>(m_rgdfData[i], dfVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            double[] rgdfData = convert(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                m_log.EXPECT_EQUAL<T>(rgdfData[i], m_rgdfData[i], "The data items should be equal!");
            }
        }

        public void TestConstructorByteList()
        {
            SimpleDatum sd = new SimpleDatum(false, 10, 10, 10, 1, DateTime.MinValue, new List<byte>(m_rgbData), 0, false, -1);

            m_log.CHECK(!sd.IsRealData, "The data should not be real!");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should NOT be null.");
            m_log.CHECK(sd.ByteData != null, "The byte data should be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgbData[i], sd.ByteData[i], "The data items should be equal!");

                byte bVal = sd.GetDataAtByte(i);
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                bVal = (byte)Convert.ChangeType(tVal, typeof(byte));
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            for (int i = 0; i < rgData.Length; i++)
            {
                T tVal = rgData[i];
                byte bVal = (byte)Convert.ChangeType(tVal, typeof(byte));
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");
            }
        }

        public void TestSetDataFloat()
        {
            SimpleDatum sd = new SimpleDatum(m_nChannel, m_nWidth, m_nHeight);

            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");

            sd.SetData(m_rgfData, 1); // set to real, float
            m_log.CHECK(sd.IsRealData, "The data should be real!");

            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF != null, "The real data 'f' should NOT be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgfData[i], sd.RealDataF[i], "The data items should be equal!");

                float fVal = sd.GetDataAtF(i);
                m_log.CHECK_EQ(m_rgfData[i], fVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                fVal = (float)Convert.ChangeType(tVal, typeof(float));
                m_log.CHECK_EQ(m_rgfData[i], fVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            float[] rgfData = convertF(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                m_log.CHECK_EQ(rgfData[i], m_rgfData[i], "The data items should be equal!");
            }
        }

        public void TestSetDataDouble()
        {
            SimpleDatum sd = new SimpleDatum(m_nChannel, m_nWidth, m_nHeight);

            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");

            sd.SetData(m_rgdfData, 1); // set to real, double
            m_log.CHECK(sd.IsRealData, "The data should be real!");

            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD != null, "The real data 'd' should NOT be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");

            for (int i = 0; i < m_rgfData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgdfData[i], sd.RealDataD[i], "The data items should be equal!");

                double dfVal = sd.GetDataAtD(i);
                m_log.CHECK_EQ(m_rgdfData[i], dfVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                dfVal = (double)Convert.ChangeType(tVal, typeof(double));
                m_log.EXPECT_EQUAL<T>(m_rgdfData[i], dfVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgdfData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            double[] rgdfData = convert(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                m_log.EXPECT_EQUAL<T>(rgdfData[i], m_rgdfData[i], "The data items should be equal!");
            }
        }

        public void TestSetDataByte()
        {
            SimpleDatum sd = new SimpleDatum(m_nChannel, m_nWidth, m_nHeight);

            m_log.CHECK(sd.ByteData == null, "The byte data should be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");

            sd.SetData(m_rgbData, 1); // set to non-real, byte
            m_log.CHECK(!sd.IsRealData, "The data should not be real!");

            m_log.CHECK(sd.ByteData != null, "The byte data should NOT be null.");
            m_log.CHECK(sd.RealDataD == null, "The real data 'd' should be null.");
            m_log.CHECK(sd.RealDataF == null, "The real data 'f' should be null.");

            for (int i = 0; i < m_rgbData.Length; i++)
            {
                m_log.CHECK_EQ(m_rgbData[i], sd.ByteData[i], "The data items should be equal!");

                byte bVal = sd.GetDataAtByte(i);
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");

                T tVal = sd.GetDataAt<T>(i);
                bVal = (byte)Convert.ChangeType(tVal, typeof(byte));
                m_log.CHECK_EQ(m_rgbData[i], bVal, "The data items should be equal!");
            }

            T[] rgData = sd.GetData<T>();
            m_log.CHECK_EQ(rgData.Length, m_rgbData.Length, "The data length is incorrect.");
            m_log.CHECK_EQ(sd.ItemCount, rgData.Length, "The data length is incorrect.!");

            float[] rgfData = convertF(rgData);
            for (int i = 0; i < rgData.Length; i++)
            {
                byte bVal = (byte)rgfData[i];
                m_log.CHECK_EQ(bVal, m_rgbData[i], "The data items should be equal!");
            }
        }

        public void TestTranspose()
        {
            double[] rgf = new double[2 * 3 * 4] { 1.1, 1.2, 1.3, 1.4,
                                                   2.1, 2.2, 2.3, 2.4,
                                                   3.1, 3.2, 3.3, 3.4,
                                                   4.1, 4.2, 4.3, 4.4,
                                                   5.1, 5.2, 5.3, 5.4,
                                                   6.1, 6.2, 6.3, 6.4 };
            double[] rgfT = new double[2 * 3 * 4] { 1.1, 1.2, 1.3, 1.4,
                                                    4.1, 4.2, 4.3, 4.4,
                                                    2.1, 2.2, 2.3, 2.4,
                                                    5.1, 5.2, 5.3, 5.4,
                                                    3.1, 3.2, 3.3, 3.4,
                                                    6.1, 6.2, 6.3, 6.4 };
            double[] rgTa = new double[2 * 3 * 4];

            int nH = 2;
            int nW = 3;
            int nDim = 4;
            rgTa = SimpleDatum.Transpose(rgf, nH, nW, nDim);

            for (int i = 0; i < rgTa.Length; i++)
            {
                m_log.CHECK_EQ(rgTa[i], rgfT[i], "The values are not as expected!");
            }
        }
    }
}
