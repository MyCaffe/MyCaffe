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
using MyCaffe.preprocessor;
using System.ServiceModel;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffePreProcessor
    {
        [TestMethod]
        public void TestCudaPreProcDLL()
        {
            MyCaffePreProcessorTest test = new MyCaffePreProcessorTest();

            try
            {
                foreach (IMyCaffePreProcessorTest t in test.Tests)
                {
                    t.TestCudaPreProcDLL();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMyCaffePreProcessorTest : ITest
    {
        void TestCudaPreProcDLL();
    }

    class MyCaffePreProcessorTest : TestBase
    {
        public MyCaffePreProcessorTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MyCaffePreProcessor Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MyCaffePreProcessorTest<double>(strName, nDeviceID, engine);
            else
                return new MyCaffePreProcessorTest<float>(strName, nDeviceID, engine);
        }
    }

    class MyCaffePreProcessorTest<T> : TestEx<T>, IMyCaffePreProcessorTest
    {
        public MyCaffePreProcessorTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        public void TestCudaPreProcDLL()
        {
            SettingsCaffe settings = new SettingsCaffe();
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, m_log, evtCancel, null, null, null, null, "", true);
            Extension<T> extension = new Extension<T>(mycaffe as IXMyCaffeExtension<T>);

            string strCudaPath = mycaffe.Cuda.Path;
            string strPath = Path.GetDirectoryName(strCudaPath);
            string strFile = Path.GetFileName(strCudaPath);


            string strTarget = "CudaDnnDll.";
            int nPos = strFile.IndexOf(strTarget);
            if (nPos < 0)
                throw new Exception("Invalid Cuda Path!");

            strFile = "CudaPreProcDll." + strFile.Substring(nPos + strTarget.Length);
            strPath += "\\" + strFile;

            if (!File.Exists(strPath))
                throw new Exception("Could not find extension DLL '" + strPath + "'!");

            extension.Initialize(strPath);

            //
            // Run the Initialize extension function
            //
            int nFields = 2;
            int nOutFields = 0;
            int nDepth = 10;
            bool bCallProcessAfterAdd = false;

            if (typeof(T) == typeof(double))
            {
                List<double> rgParam = new List<double>() { nFields, nDepth, (bCallProcessAfterAdd) ? 1 : 0 };
                double[] rg = extension.Run(Extension<T>.FUNCTION.INITIALIZE, rgParam.ToArray());
                nOutFields = (int)rg[0];
            }
            else
            {
                List<float> rgParam = new List<float>() { nFields, nDepth, (bCallProcessAfterAdd) ? 1 : 0 };
                float[] rg = extension.Run(Extension<T>.FUNCTION.INITIALIZE, rgParam.ToArray());
                nOutFields = (int)rg[0];
            }

            Blob<T> blobInput = new Blob<T>(mycaffe.Cuda, m_log);
            blobInput.Reshape(1, 1, nFields, nDepth);

            Blob<T> blobOutput = new Blob<T>(mycaffe.Cuda, m_log);
            blobOutput.Reshape(1, 1, nOutFields, nDepth);

            //
            // Setup the Memory
            //
            if (typeof(T) == typeof(double))
            {
                List<double> rgParam = new List<double>() { blobInput.count(), blobInput.mutable_gpu_data, blobInput.count(), blobInput.mutable_gpu_diff, blobOutput.count(), blobOutput.mutable_gpu_data, blobOutput.count(), blobOutput.mutable_gpu_diff };
                extension.Run(Extension<T>.FUNCTION.SETMEMORY, rgParam.ToArray());
            }
            else
            {
                List<float> rgParam = new List<float>() { blobInput.count(), blobInput.mutable_gpu_data, blobInput.count(), blobInput.mutable_gpu_diff, blobOutput.count(), blobOutput.mutable_gpu_data, blobOutput.count(), blobOutput.mutable_gpu_diff };
                extension.Run(Extension<T>.FUNCTION.SETMEMORY, rgParam.ToArray());
            }


            List<double> rgData = new List<double>();
            for (int i = 0; i < nFields; i++)
            {
                for (int j = 0; j < nDepth; j++)
                {
                    rgData.Add(j);
                }
            }

            //
            // Add new data
            //
            if (typeof(T) == typeof(double))
            {
                List<double> rgParam = rgData;
                extension.Run(Extension<T>.FUNCTION.ADDDATA, rgParam.ToArray());

                for (int i = 10; i < 15; i++)
                {
                    rgParam = new List<double>() { i, i };
                    extension.Run(Extension<T>.FUNCTION.ADDDATA, rgParam.ToArray());
                }
            }
            else
            {
                List<float> rgParam = rgData.Select(p => (float)p).ToList();
                extension.Run(Extension<T>.FUNCTION.ADDDATA, rgParam.ToArray());

                for (int i = 10; i < 15; i++)
                {
                    rgParam = new List<float>() { i, i };
                    extension.Run(Extension<T>.FUNCTION.ADDDATA, rgParam.ToArray());
                }
            }

            //
            // Verify the results
            //
            double[] rgOut = convert(blobInput.update_cpu_data());

            for (int i = 0; i < nFields; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    int nIdx = i * nDepth;

                    rgData[nIdx + j] = rgData[nIdx + j + 5];

                    if (i == 0)
                        rgData[nIdx + j + 5] = j + 10;
                    else
                        rgData[nIdx + j + 5] = j + 10;
                }
            }

            m_log.CHECK_EQ(rgData.Count, rgOut.Length, "The data lengths should be equal.");

            for (int i = 0; i < rgData.Count; i++)
            {
                m_log.CHECK_EQ(rgData[i], rgOut[i], "The items at index #" + i.ToString() + " should be equal!");
            }

            //
            // Process the data
            //
            extension.Run(Extension<T>.FUNCTION.PROCESSDATA);

            for (int i = 0; i < nFields; i++)
            {
                // Calculate Mean
                double dfSum = 0;

                for (int j = 0; j < nDepth; j++)
                {
                    int nIdx = i * nDepth + j;
                    dfSum += rgData[nIdx];
                }

                double dfMean = dfSum / nDepth;

                // Subtract Mean
                for (int j = 0; j < nDepth; j++)
                {
                    int nIdx = i * nDepth + j;
                    rgData[nIdx] -= dfMean;
                }

                // Calculate StdDev
                dfSum = 0;

                for (int j = 0; j < nDepth; j++)
                {
                    int nIdx = i * nDepth + j;
                    dfSum += rgData[nIdx] * rgData[nIdx];
                }

                double dfStdDev = Math.Sqrt(dfSum / nDepth);

                // Divide by StdDev
                for (int j = 0; j < nDepth; j++)
                {
                    int nIdx = i * nDepth + j;
                    rgData[nIdx] /= dfStdDev;
                }
            }

            rgOut = convert(blobOutput.update_cpu_data());
            m_log.CHECK_EQ(rgData.Count, rgOut.Length, "The data lengths should be equal.");

            for (int i = 0; i < rgData.Count; i++)
            {
                m_log.EXPECT_EQUAL<float>(rgData[i], rgOut[i], "The items at index #" + i.ToString() + " should be equal!");
            }

            blobInput.Dispose();
            blobOutput.Dispose();
            extension.Dispose();
            mycaffe.Dispose();
        }
    }
}
