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
        public void TestBBOX_Decode1_Corner()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Decode1_Corner(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Decode1_CenterSize()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Decode1_CenterSize(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_DecodeN_Corner()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_DecodeN_Corner(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_DecodeN_CenterSize()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_DecodeN_CenterSize(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Encode_Corner()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Encode_Corner(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Encode_CenterSize()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Encode_CenterSize(0);
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
        public void TestBBOX_Match_OneBipartite()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Match_OneBipartite(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Match_AllBipartite()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Match_AllBipartite(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Match_OnePerPrediction()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Match_OnePerPrediction(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Match_AllPerPrediction()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Match_AllPerPrediction(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBOX_Match_AllPerPredictionEx()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestBBOX_Match_AllPerPredictionEx(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetGt()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestGetGt(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetLocPredShared()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestGetLocPredShared(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetLocPredUnShared()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestGetLocPredUnShared(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetConfScores()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestGetConfScores(0);
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
        void TestBBOX_Decode1_Corner(int nConfig);
        void TestBBOX_Decode1_CenterSize(int nConfig);
        void TestBBOX_DecodeN_Corner(int nConfig);
        void TestBBOX_DecodeN_CenterSize(int nConfig);
        void TestBBOX_Encode_Corner(int nConfig);
        void TestBBOX_Encode_CenterSize(int nConfig);
        void TestBBOX_Intersect(int nConfig);
        void TestBBOX_JaccardOverlap(int nConfig);
        void TestBBOX_Match_OneBipartite(int nConfig);
        void TestBBOX_Match_AllBipartite(int nConfig);
        void TestBBOX_Match_OnePerPrediction(int nConfig);
        void TestBBOX_Match_AllPerPrediction(int nConfig);
        void TestBBOX_Match_AllPerPredictionEx(int nConfig);
        void TestGetGt(int nConfig);
        void TestGetLocPredShared(int nConfig);
        void TestGetLocPredUnShared(int nConfig);
        void TestGetConfScores(int nConfig);
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

            BBOX_DECODE1_CORNER = 6,
            BBOX_DECODE1_CENTER_SIZE = 7,
            BBOX_DECODEN_CORNER = 8,
            BBOX_DECODEN_CENTER_SIZE = 9,
            BBOX_ENCODE_CORNER = 10,
            BBOX_ENCODE_CENTER_SIZE = 11,
            BBOX_INTERSECT = 12,
            BBOX_JACCARDOVERLAP = 13,
            BBOX_MATCH_ONEBIPARTITE = 14,
            BBOX_MATCH_ALLBIPARTITE = 15,
            BBOX_MATCH_ONEPERPREDICTION = 16,
            BBOX_MATCH_ALLPERPREDICTION = 17,
            BBOX_MATCH_ALLPERPREDICTIONEX = 18,

            GET_GT = 19,
            GET_LOCPRED_SHARED = 21,
            GET_LOCPRED_UNSHARED = 22,
            GET_CONF_SCORES = 23,

            FINDMATCHES = 24,
            COUNTMATCHES = 25,
            SOFTMAX = 26,
            COMPUTE_CONF_LOSS = 27,
            COMPUTE_LOC_LOSS = 28,
            GET_TOPK_SCORES = 29,
            APPLYNMS = 30,
            MINE_HARD_EXAMPLES = 31
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

        public void TestBBOX_Decode1_Corner(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_DECODE1_CORNER, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Decode1_CenterSize(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_DECODE1_CENTER_SIZE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_DecodeN_Corner(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_DECODEN_CORNER, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_DecodeN_CenterSize(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_DECODEN_CENTER_SIZE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Encode_Corner(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_ENCODE_CORNER, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Encode_CenterSize(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_ENCODE_CENTER_SIZE, nConfig);

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

        public void TestBBOX_Match_OneBipartite(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_MATCH_ONEBIPARTITE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Match_AllBipartite(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_MATCH_ALLBIPARTITE, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Match_OnePerPrediction(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_MATCH_ONEPERPREDICTION, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Match_AllPerPrediction(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_MATCH_ALLPERPREDICTION, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestBBOX_Match_AllPerPredictionEx(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.BBOX_MATCH_ALLPERPREDICTIONEX, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestGetGt(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.GET_GT, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestGetLocPredShared(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.GET_LOCPRED_SHARED, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestGetLocPredUnShared(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.GET_LOCPRED_UNSHARED, nConfig);

            m_cuda.FreeExtension(hExtension);
        }

        public void TestGetConfScores(int nConfig)
        {
            string strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
            if (!File.Exists(strPath))
                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";

            long hExtension = m_cuda.CreateExtension(strPath);

            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
            runTest(hExtension, TEST.GET_CONF_SCORES, nConfig);

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
