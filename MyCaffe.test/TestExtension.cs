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
    public class TestExtension
    {
        [TestMethod]
        public void TestRun()
        {
            ExtensionTest test = new ExtensionTest();

            try
            {
                foreach (IExtensionTest t in test.Tests)
                {
                    t.TestRun();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IExtensionTest : ITest
    {
        void TestRun();
    }

    class ExtensionTest : TestBase
    {
        public ExtensionTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Extension Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ExtensionTest<double>(strName, nDeviceID, engine);
            else
                return new ExtensionTest<float>(strName, nDeviceID, engine);
        }
    }

    class ExtensionTest<T> : TestEx<T>, IExtensionTest
    {
        public ExtensionTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 1000, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
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
                    if (strVersion != "11.6" && strVersion.Contains("11.6"))
                        strVersion = "11.6";

                    strPath = AssemblyDirectory + "\\MyCaffe.test.extension." + strVersion + ".dll";
                }
                else
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

                return strPath;
            }
        }

        public void TestRun()
        {
            long hExtension = m_cuda.CreateExtension(DllPath);
            m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");

            try
            {
                List<double> rgdf = new List<double>() { 1, 2, 3 };
                T[] rg1 = m_cuda.RunExtension(hExtension, 1, Utility.ConvertVec<T>(rgdf.ToArray()));
                double[] rgdf1 = Utility.ConvertVec<T>(rg1);

                m_log.CHECK_EQ(rgdf1.Length, 3, "There should be three items returned.");
                for (int i = 0; i < rgdf1.Length; i++)
                {
                    m_log.CHECK_EQ(rgdf[i] * rgdf[i], rgdf1[i], "The item at index #" + i.ToString() + " is incorrect.");
                }

                List<float> rgf = new List<float>() { 1, 2, 3 };
                T[] rg2 = m_cuda.RunExtension(hExtension, 2, Utility.ConvertVec<T>(rgf.ToArray()));
                double[] rgdf2 = Utility.ConvertVec<T>(rg2);

                m_log.CHECK_EQ(rgdf1.Length, 3, "There should be three items returned.");
                for (int i = 0; i < rgdf2.Length; i++)
                {
                    m_log.CHECK_EQ(rgf[i] * rgf[i] * rgf[i], rgdf2[i], "The item at index #" + i.ToString() + " is incorrect.");
                }
            }
            finally
            {
                if (hExtension != 0)
                    m_cuda.FreeExtension(hExtension);
            }
        }
    }
}
