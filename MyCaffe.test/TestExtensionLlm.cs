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
using MyCaffe.common;

namespace MyCaffe.test
{
    [TestClass]
    public class TestExtensionLlm
    {
        [TestMethod]
        public void TestLoad()
        {
            ExtenionTestLlm test = new ExtenionTestLlm();

            try
            {
                foreach (IExtensionTestLlm t in test.Tests)
                {
                    // Llama tests only support float.
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestLoad();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGenerate()
        {
            ExtenionTestLlm test = new ExtenionTestLlm();

            try
            {
                foreach (IExtensionTestLlm t in test.Tests)
                {
                    // Llama tests only support float.
                    if (t.DataType == DataType.DOUBLE)
                        continue;
                    t.TestGenerate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IExtensionTestLlm : ITest
    {
        void TestLoad();
        void TestGenerate();
    }

    class ExtenionTestLlm : TestBase
    {
        public ExtenionTestLlm(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Extension Test LLM", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ExtenionTestLlm<double>(strName, nDeviceID, engine);
            else
                return new ExtenionTestLlm<float>(strName, nDeviceID, engine);
        }
    }

    class ExtenionTestLlm<T> : TestEx<T>, IExtensionTestLlm
    {
        public ExtenionTestLlm(string strName, int nDeviceID, EngineParameter.Engine engine)
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

                string strPath1 = m_cuda.Path;
                nPos = strPath1.LastIndexOf('\\');
                if (nPos >= 0)
                    strPath1 = strPath1.Substring(0, nPos);

                string strPath;

                if (strVersion.Length > 0)
                {
                    if (strVersion != "12.3" && strVersion.Contains("12.3"))
                        strVersion = "12.3";

                    strPath = strPath1 + "\\CudaExtension.llm16." + strVersion + ".dll";
                }
                else
                {
                    strPath = strPath1 + "\\CudaExtension.llm16.12.3.dll";
                    if (!File.Exists(strPath))
                    {
                        strPath = strPath1 + "\\CudaExtension.llm16.12.2.dll";
                        if (!File.Exists(strPath))
                        {
                            strPath = strPath1 + "\\CudaExtension.llm16.12.1.dll";
                            if (!File.Exists(strPath))
                            {
                                strPath = strPath1 + "\\CudaExtension.llm16.12.0.dll";
                                if (!File.Exists(strPath))
                                {
                                    strPath = strPath1 + "\\CudaExtension.llm16.11.8.dll";
                                    if (!File.Exists(strPath))
                                    {
                                        throw new Exception("Could not find the CudaExtension.llm16.xx.dll file!");
                                    }
                                }
                            }
                        }
                    }
                }

                return strPath;
            }
        }

        public void TestLoad()
        {
            long hExtension = 0;

            try
            {
                hExtension = m_cuda.CreateExtension(DllPath);
                m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");

                string strModelFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\llama7b\\llama2_7b_chat.bin";
                string strTokenizerFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\llama7b\\tokenizer.bin";
                string strInput = strModelFile + ";" + strTokenizerFile;

                T[] rgParam = new T[3];
                rgParam[0] = Utility.ConvertVal<T>(1.0); // temperature
                rgParam[1] = Utility.ConvertVal<T>(0.9); // topp
                rgParam[2] = Utility.ConvertVal<T>(0);   // seed

                T[] rgLlm = m_cuda.RunExtension(hExtension, (int)CUDAFN_EXTENSION_LLM.CREATE, rgParam);
                m_cuda.RunExtensionEx(hExtension, (int)CUDAFN_EXTENSION_LLM.LOAD, rgLlm, strInput);

                float fStatus = 0;
                while (fStatus < 1)
                {
                    T[] rgStatus = m_cuda.RunExtension(hExtension, (int)CUDAFN_EXTENSION_LLM.QUERY_STATUS, rgLlm);
                    fStatus = Utility.ConvertValF<T>(rgStatus[0]);
                }

                m_cuda.RunExtension(hExtension, (int)CUDAFN_EXTENSION_LLM.DESTROY, rgLlm);
            }
            finally
            {
                if (hExtension != 0)
                    m_cuda.FreeExtension(hExtension);
            }
        }

        public void TestGenerate()
        {
            long hExtension = 0;

            try
            {
                hExtension = m_cuda.CreateExtension(DllPath);
                m_log.CHECK(hExtension != 0, "The extension handle should be non zero.");

                string strModelFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\llama7b\\llama2_7b_chat.bin";
                string strTokenizerFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\llama7b\\tokenizer.bin";
                string strInput = strModelFile + ";" + strTokenizerFile;

                T[] rgParam = new T[3];
                rgParam[0] = Utility.ConvertVal<T>(1.0); // temperature
                rgParam[1] = Utility.ConvertVal<T>(0.9); // topp
                rgParam[2] = Utility.ConvertVal<T>(0);   // seed

                T[] rgLlm = m_cuda.RunExtension(hExtension, (int)CUDAFN_EXTENSION_LLM.CREATE, rgParam);
                m_cuda.RunExtensionEx(hExtension, (int)CUDAFN_EXTENSION_LLM.LOAD, rgLlm, strInput);

                float fStatus = 0;
                while (fStatus < 1)
                {
                    T[] rgStatus = m_cuda.RunExtension(hExtension, (int)CUDAFN_EXTENSION_LLM.QUERY_STATUS, rgLlm);
                    fStatus = Utility.ConvertValF<T>(rgStatus[0]);
                }

                m_cuda.RunExtensionEx(hExtension, (int)CUDAFN_EXTENSION_LLM.GENERATE, rgLlm, "[INST]What is your name?[/INST]");

                string strResponse = "";
                string strText = " ";
                while (!strText.EndsWith("\n[END]"))
                {
                    int[] rgLlm1 = new int[1];
                    rgLlm1[0] = (int)Utility.ConvertValF(rgLlm[0]);
                    string[] rgText = m_cuda.QueryExtensionStrings(hExtension, (int)CUDAFN_EXTENSION_LLM.QUERY_RESPONSE, rgLlm1);
                    strText = rgText[0];

                    if (!string.IsNullOrEmpty(strText))
                        strResponse += strText; 
                }

                m_log.WriteLine(strResponse);

                m_cuda.RunExtension(hExtension, (int)CUDAFN_EXTENSION_LLM.DESTROY, rgLlm);
            }
            finally
            {
                if (hExtension != 0)
                    m_cuda.FreeExtension(hExtension);
            }
        }
    }
}
