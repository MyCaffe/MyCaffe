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
using System.Threading.Tasks;
using System.Threading;
using System.Reflection;
using System.IO;

namespace MyCaffe.test
{
    [TestClass]
    public class TestExtensionSsd
    {
        [TestMethod]
        public void TestCreate()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestCreate(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Size()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Size(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Bounds()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Bounds(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_DivBounds()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_DivBounds(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Clip()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Clip(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Decode()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Decode(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Encode()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Encode(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Intersect()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Intersect(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_JaccardOverlap()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_JaccardOverlap(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Match()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Match(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFindMatches()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestFindMatches(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCountMatches()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestCountMatches(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSoftMax()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestSoftMax(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestComputeConfLoss()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestComputeConfLoss(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestComputeLocLoss()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestComputeLocLoss(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetTopKScores()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestGetTopKScores(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestApplyNMS()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestApplyNMS(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestApplyNMSSimple()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestApplyNMS(-1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMineHardExamples()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestMineHardExamples(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ISsdExtensionTest : ITest
    {
        void TestCreate(int nConfig);
        void TestBBOX_Size(int nConfig);
        void TestBBOX_Bounds(int nConfig);
        void TestBBOX_DivBounds(int nConfig);
        void TestBBOX_Clip(int nConfig);
        void TestBBOX_Decode(int nConfig);
        void TestBBOX_Encode(int nConfig);
        void TestBBOX_Intersect(int nConfig);
        void TestBBOX_JaccardOverlap(int nConfig);
        void TestBBOX_Match(int nConfig);
        void TestFindMatches(int nConfig);
        void TestCountMatches(int nConfig);
        void TestSoftMax(int nConfig);
        void TestComputeConfLoss(int nConfig);
        void TestComputeLocLoss(int nConfig);
        void TestGetTopKScores(int nConfig);
        void TestApplyNMS(int nConfig);
        void TestMineHardExamples(int nConfig);
    }

    class SsdExtensionTest : TestBase
    {
        public SsdExtensionTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SSD Extension Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SsdExtensionTest<double>(strName, nDeviceID, engine);
            else
                return new SsdExtensionTest<float>(strName, nDeviceID, engine);
        }
    }

    class SsdExtensionTest<T> : TestEx<T>, ISsdExtensionTest
    {
        public SsdExtensionTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        public enum TEST
        {
            CREATE = 1,

            BBOX_SIZE = 2,
            BBOX_BOUNDS = 3,
            BBOX_DIVBOUNDS = 4,
            BBOX_CLIP = 5,

            BBOX_DECODE = 6,
            BBOX_ENCODE = 7,
            BBOX_INTERSECT = 8,
            BBOX_JACCARDOVERLAP = 9,
            BBOX_MATCH = 10,

            FINDMATCHES = 11,
            COUNTMATCHES = 12,
            SOFTMAX = 13,
            COMPUTE_CONF_LOSS = 14,
            COMPUTE_LOC_LOSS = 15,
            GET_TOPK_SCORES = 16,
            APPLYNMS = 17,
            MINE_HARD_EXAMPLES = 18
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }

        private void runTest(long hExtension, TEST tst, params int[] rg)
        {
            List<double> rgdf = new List<double>() { (double)(int)tst };

            for (int i = 0; i < rg.Length; i++)
            {
                rgdf.Add(rg[i]);
            }

            T[] rg1 = m_cuda.RunExtension(hExtension, 3, Utility.ConvertVec<T>(rgdf.ToArray()));
            double[] rgdf1 = Utility.ConvertVec<T>(rg1);
            long lErr = (long)rgdf1[0];
            m_log.CHECK_EQ(lErr, 0, "The SSD test " + tst.ToString() + " failed with error " + lErr.ToString());
        }

        public void TestCreate(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.CREATE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Size(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_SIZE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Bounds(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_BOUNDS, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_DivBounds(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_DIVBOUNDS, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Clip(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_CLIP, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Decode(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_DECODE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }


        public void TestBBOX_Encode(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_ENCODE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }


        public void TestBBOX_Intersect(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_INTERSECT, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_JaccardOverlap(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_JACCARDOVERLAP, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Match(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_MATCH, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestFindMatches(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.FINDMATCHES, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestCountMatches(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.COUNTMATCHES, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestSoftMax(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.SOFTMAX, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestComputeConfLoss(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.COMPUTE_CONF_LOSS, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestComputeLocLoss(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.COMPUTE_LOC_LOSS, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestGetTopKScores(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.GET_TOPK_SCORES, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestApplyNMS(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.APPLYNMS, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestMineHardExamples(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.MINE_HARD_EXAMPLES, nConfig);

            m_cuda.FreeExtension(hExtension);
        }
    }
}
