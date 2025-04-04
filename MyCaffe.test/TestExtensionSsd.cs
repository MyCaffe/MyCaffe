﻿using System;
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
        public void TestGetPriorBBoxes()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestGetPriorBBoxes(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSoftmax()
        {
            SsdExtensionTest test = new SsdExtensionTest();

            try
            {
                foreach (ISsdExtensionTest t in test.Tests)
                {
                    t.TestSoftmax(0);
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

        //[TestMethod]
        //public void TestComputeLocLoss()
        //{
        //    SsdExtensionTest test = new SsdExtensionTest();

        //    try
        //    {
        //        foreach (ISsdExtensionTest t in test.Tests)
        //        {
        //            t.TestComputeLocLoss(0);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        //[TestMethod]
        //public void TestFindMatches()
        //{
        //    SsdExtensionTest test = new SsdExtensionTest();

        //    try
        //    {
        //        foreach (ISsdExtensionTest t in test.Tests)
        //        {
        //            t.TestFindMatches(0);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        //[TestMethod]
        //public void TestCountMatches()
        //{
        //    SsdExtensionTest test = new SsdExtensionTest();

        //    try
        //    {
        //        foreach (ISsdExtensionTest t in test.Tests)
        //        {
        //            t.TestCountMatches(0);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        //[TestMethod]
        //public void TestMineHardExamples()
        //{
        //    SsdExtensionTest test = new SsdExtensionTest();

        //    try
        //    {
        //        foreach (ISsdExtensionTest t in test.Tests)
        //        {
        //            t.TestMineHardExamples(0);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}
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
        void TestGetPriorBBoxes(int nConfig);
        void TestSoftmax(int nConfig);
        void TestApplyNMS(int nConfig);
        void TestComputeConfLoss(int nConfig);
        void TestFindMatches(int nConfig);
        void TestCountMatches(int nConfig);
        void TestComputeLocLoss(int nConfig);
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
            GET_PRIOR_BBOXES = 24,

            SOFTMAX = 25,
            COMPUTE_CONF_LOSS = 26,
            APPLYNMS = 27,

            COMPUTE_LOC_LOSS = 28,
            FINDMATCHES = 29,
            COUNTMATCHES = 30,
            MINE_HARD_EXAMPLES = 32
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

        private void runTest(int nConfig, TEST tst, params int[] rg)
        {
            long hExtension = 0;

            try
            {
                hExtension = m_cuda.CreateExtension(DllPath);
                m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");
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
            catch (Exception excpt)
            {
                throw excpt;
            }
            catch
            {
                throw new Exception("Non-CLS Exception thrown.");
            }
            finally
            {
                if (hExtension != 0)
                    m_cuda.FreeExtension(hExtension);
            }
        }

        private string DllPath
        {
            get
            {
                string strVersion = m_cuda.Path;
                int nPos = strVersion.LastIndexOf('.');
                if (nPos > 0)
                    strVersion = strVersion.Substring(0, nPos);

                string strTarget = "CudaDnnDll.";
                nPos = strVersion.IndexOf(strTarget);
                if (nPos >= 0)
                    strVersion = strVersion.Substring(nPos + strTarget.Length);

                string strPath;
                if (strVersion.Length > 0)
                {
                    if (strVersion != "12.3" && strVersion.Contains("12.3"))
                        strVersion = "12.3";

                    strPath = AssemblyDirectory + "\\MyCaffe.test.extension." + strVersion + ".dll";
                }
                else
                {
                    strPath = AssemblyDirectory + "\\MyCaffe.test.extension.12.3.dll";
                    if (!File.Exists(strPath))
                    {
                        strPath = AssemblyDirectory + "\\MyCaffe.test.extension.12.2.dll";
                        if (!File.Exists(strPath))
                        {
                            strPath = AssemblyDirectory + "\\MyCaffe.test.extension.12.1.dll";
                            if (!File.Exists(strPath))
                            {
                                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.12.0.dll";
                                if (!File.Exists(strPath))
                                {
                                    strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.8.dll";
                                    if (!File.Exists(strPath))
                                    {
                                        strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.7.dll";
                                        if (!File.Exists(strPath))
                                        {
                                            strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.6.dll";
                                            if (!File.Exists(strPath))
                                            {
                                                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.5.dll";
                                                if (!File.Exists(strPath))
                                                {
                                                    strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.4.dll";
                                                    if (!File.Exists(strPath))
                                                    {
                                                        strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.3.dll";
                                                        if (!File.Exists(strPath))
                                                        {
                                                            strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.2.dll";
                                                            if (!File.Exists(strPath))
                                                            {
                                                                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.1.dll";
                                                                if (!File.Exists(strPath))
                                                                {
                                                                    strPath = AssemblyDirectory + "\\MyCaffe.test.extension.11.0.dll";
                                                                    if (!File.Exists(strPath))
                                                                    {
                                                                        strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.2.dll";
                                                                        if (!File.Exists(strPath))
                                                                        {
                                                                            strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.1.dll";
                                                                            if (!File.Exists(strPath))
                                                                                strPath = AssemblyDirectory + "\\MyCaffe.test.extension.10.0.dll";
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return strPath;
            }
        }

        public void TestCreate(int nConfig)
        {
            runTest(nConfig, TEST.CREATE);
        }

        public void TestBBOX_Size(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_SIZE);
        }

        public void TestBBOX_Bounds(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_BOUNDS);
        }

        public void TestBBOX_DivBounds(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_DIVBOUNDS);
        }

        public void TestBBOX_Clip(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_CLIP);
        }

        public void TestBBOX_Decode1_Corner(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_DECODE1_CORNER);
        }

        public void TestBBOX_Decode1_CenterSize(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_DECODE1_CENTER_SIZE);
        }

        public void TestBBOX_DecodeN_Corner(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_DECODEN_CORNER);
        }

        public void TestBBOX_DecodeN_CenterSize(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_DECODEN_CENTER_SIZE);
        }

        public void TestBBOX_Encode_Corner(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_ENCODE_CORNER);
        }

        public void TestBBOX_Encode_CenterSize(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_ENCODE_CENTER_SIZE);
        }

        public void TestBBOX_Intersect(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_INTERSECT);
        }

        public void TestBBOX_JaccardOverlap(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_JACCARDOVERLAP);
        }

        public void TestBBOX_Match_OneBipartite(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_MATCH_ONEBIPARTITE);
        }

        public void TestBBOX_Match_AllBipartite(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_MATCH_ALLBIPARTITE);
        }

        public void TestBBOX_Match_OnePerPrediction(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_MATCH_ONEPERPREDICTION);
        }

        public void TestBBOX_Match_AllPerPrediction(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_MATCH_ALLPERPREDICTION);
        }

        public void TestBBOX_Match_AllPerPredictionEx(int nConfig)
        {
            runTest(nConfig, TEST.BBOX_MATCH_ALLPERPREDICTIONEX);
        }

        public void TestGetGt(int nConfig)
        {
            runTest(nConfig, TEST.GET_GT);
        }

        public void TestGetLocPredShared(int nConfig)
        {
            runTest(nConfig, TEST.GET_LOCPRED_SHARED);
        }

        public void TestGetLocPredUnShared(int nConfig)
        {
            runTest(nConfig, TEST.GET_LOCPRED_UNSHARED);
        }

        public void TestGetConfScores(int nConfig)
        {
            runTest(nConfig, TEST.GET_CONF_SCORES);
        }

        public void TestGetPriorBBoxes(int nConfig)
        {
            runTest(nConfig, TEST.GET_PRIOR_BBOXES);
        }

        public void TestFindMatches(int nConfig)
        {
            runTest(nConfig, TEST.FINDMATCHES);
        }

        public void TestCountMatches(int nConfig)
        {
            runTest(nConfig, TEST.COUNTMATCHES);
        }

        public void TestSoftmax(int nConfig)
        {
            runTest(nConfig, TEST.SOFTMAX);
        }

        public void TestComputeConfLoss(int nConfig)
        {
            runTest(nConfig, TEST.COMPUTE_CONF_LOSS);
        }

        public void TestComputeLocLoss(int nConfig)
        {
            runTest(nConfig, TEST.COMPUTE_LOC_LOSS);
        }

        public void TestApplyNMS(int nConfig)
        {
            runTest(nConfig, TEST.APPLYNMS);
        }

        public void TestMineHardExamples(int nConfig)
        {
            runTest(nConfig, TEST.MINE_HARD_EXAMPLES);
        }
    }
}
