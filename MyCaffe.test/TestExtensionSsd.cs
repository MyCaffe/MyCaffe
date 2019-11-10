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
        public void TestApplyNMSSimple()
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
    }

    interface ISsdExtensionTest : ITest
    {
        void TestCreate(int nConfig);
        void TestApplyNMS(int nConfig);
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
            APPLYNMS = 2
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
    }
}
