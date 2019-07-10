using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using System.Diagnostics;
using MyCaffe.param;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestCudaDnn
    {
        [TestMethod]
        public void TestDeviceID()
        {
            CudaDnnTest test = new CudaDnnTest();
            int nDeviceID = TestBase.DEFAULT_DEVICE_ID;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    t.Cuda.SetDeviceID(1);
                    nDeviceID = t.Cuda.GetDeviceID();
                    t.Log.CHECK_EQ(1, nDeviceID, "The deviceID should be equal to 1.");

                    t.Cuda.SetDeviceID(0);
                    nDeviceID = t.Cuda.GetDeviceID();
                    t.Log.CHECK_EQ(0, nDeviceID, "The deviceID should be equal to 0.");

                    t.Cuda.SetDeviceID(TestBase.DEFAULT_DEVICE_ID);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetDeviceName()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    int nDeviceCount = t.Cuda.GetDeviceCount();
                    int nDeviceID = t.Cuda.GetDeviceID();

                    for (int i = 0; i < nDeviceCount; i++)
                    {
                        string strGpu = t.Cuda.GetDeviceName(i);
                        Trace.WriteLine(i.ToString() + " => " + strGpu);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetDeviceMemory()
        {
            Log log = new Log("TestCudnn");
            CudaDnn<double> cuda1 = new CudaDnn<double>(TestBase.DEFAULT_DEVICE_ID, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);

            try
            {
                bool bEstimate;
                double dfFree = 0;
                double dfUsed = 0;
                double dfTotal = cuda1.GetDeviceMemory(out dfFree, out dfUsed, out bEstimate);

                log.EXPECT_EQUAL<float>(dfUsed + dfFree, dfTotal, "used + free expected to equal total.");

                CudaDnn<float> cuda2 = new CudaDnn<float>(TestBase.DEFAULT_DEVICE_ID, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);

                try
                {
                    bool bEstimate2;
                    double dfFree2 = 0;
                    double dfUsed2 = 0;
                    double dfTotal2 = cuda2.GetDeviceMemory(out dfFree2, out dfUsed2, out bEstimate2);

                    float fActual = (float)(dfUsed2 + dfFree2);
                    float fTotal = (float)dfTotal2;

                    log.EXPECT_EQUAL<float>(fActual, fTotal, "Actual expected to equal total.");
                    log.EXPECT_EQUAL<float>(dfUsed, dfUsed2, "Used expected to eaual Used2.");
                    log.EXPECT_EQUAL<float>(dfFree, dfFree2, "Free expected to equal Free2.");
                    log.EXPECT_EQUAL<float>(dfTotal, dfTotal2, "Total expected to equal Total2.");
                }
                finally
                {
                    cuda2.Dispose();
                }
            }
            finally
            {
                cuda1.Dispose();
            }
        }

        [TestMethod]
        public void TestKernelMemCpyDouble()
        {
            CudaDnn<double> cuda1 = new CudaDnn<double>(0, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);
            int nDevCount = cuda1.GetDeviceCount();
            CudaDnn<double> cuda2 = null;

            if (nDevCount > 1)
                cuda2 = new CudaDnn<double>(1, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);

            try
            {
                if (nDevCount <= 1)
                    return;

                copyMemTest(cuda1, cuda2, 0, 1);

                if (nDevCount > 2)
                    copyMemTest(cuda1, cuda2, 1, 2);

                copyMemTest(cuda1, cuda2, 0, 0);
                copyMemTest(cuda1, cuda2, 1, 1);
            }
            finally
            {
                cuda1.Dispose();

                if (cuda2 != null)
                    cuda2.Dispose();
            }
        }

        [TestMethod]
        public void TestKernelMemCpyFloat()
        {
            CudaDnn<float> cuda1 = new CudaDnn<float>(0, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);
            int nDevCount = cuda1.GetDeviceCount();
            CudaDnn<float> cuda2 = null;

            if (nDevCount > 1)
                cuda2 = new CudaDnn<float>(1, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);

            try
            {
                if (nDevCount <= 1)
                    return;

                copyMemTest(cuda1, cuda2, 0, 1);

                if (nDevCount > 2)
                    copyMemTest(cuda1, cuda2, 1, 2);

                copyMemTest(cuda1, cuda2, 0, 0);
                copyMemTest(cuda1, cuda2, 1, 1);
            }
            finally
            {
                cuda1.Dispose();

                if (cuda2 != null)
                    cuda2.Dispose();
            }
        }

        private void copyMemTest(CudaDnn<double> cuda1, CudaDnn<double> cuda2, int nDevice1, int nDevice2)
        {
            long hHost = cuda1.AllocHostBuffer(4);
            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            long hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
            cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
            long hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });

            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            cuda1.KernelCopy(4, hMem2, 0, cuda2.KernelHandle, hMem1, 0, hHost);

            double[] rgData1 = cuda1.GetMemory(hMem1);
            double[] rgData2 = cuda2.GetMemory(hMem2);

            Assert.AreEqual(rgData1.Length, rgData2.Length);

            for (int i = 0; i < rgData1.Length; i++)
            {
                Assert.AreEqual(rgData1[i], rgData2[i]);
            }
        }

        private void copyMemTest(CudaDnn<float> cuda1, CudaDnn<float> cuda2, int nDevice1, int nDevice2)
        {
            long hHost = cuda1.AllocHostBuffer(4);
            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            long hMem1 = cuda1.AllocMemory(new List<float>() { 1, 2, 3, 4 });
            cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
            long hMem2 = cuda2.AllocMemory(new List<float>() { 10, 20, 30, 40 });

            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            cuda1.KernelCopy(4, hMem2, 0, cuda2.KernelHandle, hMem1, 0, hHost);

            float[] rgData1 = cuda1.GetMemory(hMem1);
            float[] rgData2 = cuda2.GetMemory(hMem2);

            Assert.AreEqual(rgData1.Length, rgData2.Length);

            for (int i = 0; i < rgData1.Length; i++)
            {
                Assert.AreEqual(rgData1[i], rgData2[i]);
            }
        }

        [TestMethod]
        public void TestKernelAddDouble()
        {
            CudaDnn<double> cuda1 = new CudaDnn<double>(0, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);
            int nDevCount = cuda1.GetDeviceCount();
            CudaDnn<double> cuda2 = null;

            if (nDevCount > 1)
                cuda2 = new CudaDnn<double>(1, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);

            try
            {
                if (nDevCount > 1)
                {
                    string strP2P1 = cuda1.GetDeviceP2PInfo(0);
                    string strP2P2 = cuda2.GetDeviceP2PInfo(1);

                    if (strP2P1.Contains("P2P Capable = YES") &&
                        strP2P2.Contains("P2P Capable = YES"))
                        addMemTest(cuda1, cuda2, 0, 1);

                    addMemTest(cuda1, cuda2, 0, 0);
                }
            }
            finally
            {
                cuda1.Dispose();

                if (cuda2 != null)
                    cuda2.Dispose();
            }
        }

        private void addMemTest(CudaDnn<double> cuda1, CudaDnn<double> cuda2, int nDevice1, int nDevice2)
        {
            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            long hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
            cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
            long hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });
            long hMem3 = cuda2.AllocMemory(new List<double>() { 0, 0, 0, 0 });

            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            cuda1.KernelAdd(4, hMem1, cuda2.KernelHandle, hMem2, hMem3);

            double[] rgData1 = cuda1.GetMemory(hMem1);
            double[] rgData2 = cuda2.GetMemory(hMem2);
            double[] rgData3 = cuda2.GetMemory(hMem3);

            Assert.AreEqual(rgData1.Length, rgData2.Length);
            Assert.AreEqual(rgData1.Length, rgData3.Length);

            for (int i = 0; i < rgData1.Length; i++)
            {
                Assert.AreEqual(rgData3[i], rgData1[i] + rgData2[i]);
            }
        }

        [TestMethod]
        public void TestKernelAddFloat()
        {
            CudaDnn<float> cuda1 = new CudaDnn<float>(0, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);
            int nDevCount = cuda1.GetDeviceCount();
            CudaDnn<float> cuda2 = null;

            if (nDevCount > 1)
                cuda2 = new CudaDnn<float>(1, DEVINIT.CUBLAS | DEVINIT.CURAND, null, TestBase.CudaPath);

            try
            {
                if (nDevCount > 1)
                {
                    string strP2P1 = cuda1.GetDeviceP2PInfo(0);
                    string strP2P2 = cuda2.GetDeviceP2PInfo(1);

                    if (strP2P1.Contains("P2P Capable = YES") &&
                        strP2P2.Contains("P2P Capable = YES"))
                        addMemTest(cuda1, cuda2, 0, 1);

                    addMemTest(cuda1, cuda2, 0, 0);
                }
            }
            finally
            {
                cuda1.Dispose();

                if (cuda2 != null)
                    cuda2.Dispose();
            }
        }

        private void addMemTest(CudaDnn<float> cuda1, CudaDnn<float> cuda2, int nDevice1, int nDevice2)
        {
            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            long hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
            cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
            long hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });
            long hMem3 = cuda2.AllocMemory(new List<double>() { 0, 0, 0, 0 });

            cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
            cuda1.KernelAdd(4, hMem1, cuda2.KernelHandle, hMem2, hMem3);

            float[] rgData1 = cuda1.GetMemory(hMem1);
            float[] rgData2 = cuda2.GetMemory(hMem2);
            float[] rgData3 = cuda2.GetMemory(hMem3);

            Assert.AreEqual(rgData1.Length, rgData2.Length);
            Assert.AreEqual(rgData1.Length, rgData3.Length);

            for (int i = 0; i < rgData1.Length; i++)
            {
                Assert.AreEqual(rgData3[i], rgData1[i] + rgData2[i]);
            }
        }

        [TestMethod]
        public void TestP2PAccess()
        {
            string strPath = TestBase.CudaPath;
            CudaDnn<double> cuda1 = new CudaDnn<double>(0, DEVINIT.CUBLAS | DEVINIT.CURAND, null, strPath);
            CudaDnn<double> cuda2 = new CudaDnn<double>(0, DEVINIT.CUBLAS | DEVINIT.CURAND, null, strPath);

            try
            {
                int nDeviceCount = cuda1.GetDeviceCount();

                for (int i = 0; i < nDeviceCount; i++)
                {
                    string strInfo = cuda1.GetDeviceP2PInfo(i);
                    Trace.WriteLine(strInfo);

                    strInfo = cuda1.GetDeviceInfo(i, false);
                    Trace.WriteLine(strInfo);

                    strInfo = cuda1.GetDeviceInfo(i, true);
                    Trace.WriteLine(strInfo);
                }

                List<int> rgDevices = new List<int>();

                for (int i = 1; i < nDeviceCount; i++)
                {
                    int nDevice1 = i - 1;
                    int nDevice2 = i;

                    bool bAccessFwd = cuda1.DeviceCanAccessPeer(nDevice1, nDevice2);
                    bool bAccessBwd = cuda1.DeviceCanAccessPeer(nDevice2, nDevice1);
                    string strAccessFwd = ((bAccessFwd) ? " can" : " CANNOT");
                    string strAccessBwd = ((bAccessBwd) ? " can" : " CANNOT");

                    Trace.WriteLine("---");
                    Trace.WriteLine("Device " + nDevice1.ToString() + strAccessFwd + " access " + nDevice2.ToString());
                    Trace.WriteLine("Device " + nDevice2.ToString() + strAccessBwd + " access " + nDevice1.ToString());

                    if (bAccessBwd && bAccessFwd)
                    {
                        if (!rgDevices.Contains(nDevice1))
                            rgDevices.Add(nDevice1);

                        if (!rgDevices.Contains(nDevice2))
                            rgDevices.Add(nDevice2);
                    }
                }

                string strAccessibleDevices = "";

                foreach (int nDev in rgDevices)
                {
                    strAccessibleDevices += nDev.ToString() + ",";
                }

                strAccessibleDevices = strAccessibleDevices.TrimEnd(',');

                Trace.WriteLine("----");
                Trace.WriteLine("P2P Accessible Devices = " + strAccessibleDevices);
                Trace.WriteLine("----");
                Trace.WriteLine("");

                Trace.WriteLine("Memcpy Test");

                for (int i = 1; i < rgDevices.Count; i++)
                {
                    Trace.WriteLine("testing " + rgDevices[i - 1].ToString() + " -> " + rgDevices[i].ToString());

                    cuda1.SetDeviceID(rgDevices[i - 1]);
                    long hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
                    long hHost = cuda1.AllocHostBuffer(4);

                    cuda2.SetDeviceID(rgDevices[i]);
                    long hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });

                    cuda1.SetDeviceID();
                    cuda1.KernelCopy(4, hMem1, 0, cuda2.KernelHandle, hMem2, 0, hHost);

                    double[] rgData1 = cuda1.GetMemory(hMem1);
                    cuda1.FreeMemory(hMem1);
                    hMem1 = 0;
                    cuda2.FreeHostBuffer(hHost);
                    hHost = 0;

                    cuda2.SetDeviceID();
                    double[] rgData2 = cuda2.GetMemory(hMem2);
                    cuda2.FreeMemory(hMem2);
                    hMem2 = 0;

                    Assert.AreEqual(rgData1.Length, rgData2.Length);

                    for (int j = 0; j < rgData1.Length; j++)
                    {
                        Assert.AreEqual(rgData1[j], rgData2[j]);
                        Assert.AreEqual(rgData1[j], j + 1);
                        Assert.AreEqual(rgData2[j], j + 1);
                    }

                    Trace.WriteLine("testing " + rgDevices[i - 1].ToString() + " <- " + rgDevices[i].ToString());

                    cuda1.SetDeviceID(rgDevices[i]);
                    hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
                    hHost = cuda1.AllocHostBuffer(4);

                    cuda2.SetDeviceID(rgDevices[i - 1]);
                    hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });

                    cuda1.SetDeviceID();
                    cuda1.KernelCopy(4, hMem1, 0, cuda2.KernelHandle, hMem2, 0, hHost);

                    rgData1 = cuda1.GetMemory(hMem1);
                    cuda1.FreeMemory(hMem1);
                    hMem1 = 0;
                    cuda1.FreeHostBuffer(hHost);
                    hHost = 0;

                    cuda2.SetDeviceID();
                    rgData2 = cuda2.GetMemory(hMem2);
                    cuda2.FreeMemory(hMem2);
                    hMem2 = 0;

                    Assert.AreEqual(rgData1.Length, rgData2.Length);

                    for (int j = 0; j < rgData1.Length; j++)
                    {
                        Assert.AreEqual(rgData1[j], rgData2[j]);
                        Assert.AreEqual(rgData1[j], j + 1);
                        Assert.AreEqual(rgData2[j], j + 1);
                    }
                }
            }
            finally
            {
                cuda1.Dispose();
                cuda2.Dispose();
            }
        }

        [TestMethod]
        public void TestHostBuffer()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long hMem = t.Cuda.AllocHostBuffer(1000);
                    double[] rgdfData = t.Cuda.GetHostMemoryDouble(hMem);
                    float[] rgfData = t.Cuda.GetHostMemoryFloat(hMem);
                    t.Cuda.FreeHostBuffer(hMem);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllocFreeGet()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            long hMemD = 0;
            long hMemF = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hMemD = t.Cuda.AllocMemory(rgdf);
                    t.Log.CHECK_NE(0, hMemD, "The hMemD handle should not be null!");

                    double[] rgdf1 = t.Cuda.GetMemoryDouble(hMemD);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgdf1.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgdf1[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.GetMemoryFloat(hMemD);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgf1.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgf1[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    hMemF = t.Cuda.AllocMemory(rgf);
                    t.Log.CHECK_NE(0, hMemF, "The hMemF handle should not be null!");

                    double[] rgdf2 = t.Cuda.GetMemoryDouble(hMemF);
                    t.Log.CHECK_EQ(rgdf2.Length, rgf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgdf2.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgdf2[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf2 = t.Cuda.GetMemoryFloat(hMemD);
                    t.Log.CHECK_EQ(rgf2.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgf2.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgf2[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    t.Cuda.FreeMemory(hMemD);
                    t.Cuda.FreeMemory(hMemF);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllocFreeSetDouble()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            long hMemD = 0;
            long hMemF = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hMemD = t.Cuda.AllocMemory(4);
                    t.Log.CHECK_NE(0, hMemD, "The hMemD handle should not be null!");

                    t.Cuda.SetMemory(hMemD, rgdf);

                    double[] rgdf1 = t.Cuda.GetMemoryDouble(hMemD);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgdf1.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgdf1[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.GetMemoryFloat(hMemD);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgf1.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgf1[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    hMemF = t.Cuda.AllocMemory(4);
                    t.Log.CHECK_NE(0, hMemF, "The hMemF handle should not be null!");

                    t.Cuda.SetMemory(hMemF, rgdf);

                    double[] rgdf2 = t.Cuda.GetMemoryDouble(hMemF);
                    t.Log.CHECK_EQ(rgdf2.Length, rgf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgdf2.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgdf2[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf2 = t.Cuda.GetMemoryFloat(hMemD);
                    t.Log.CHECK_EQ(rgf2.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgf2.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgf2[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    t.Cuda.FreeMemory(hMemD);
                    t.Cuda.FreeMemory(hMemF);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllocFreeSetFloat()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            long hMemD = 0;
            long hMemF = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hMemD = t.Cuda.AllocMemory(4);
                    t.Log.CHECK_NE(0, hMemD, "The hMemD handle should not be null!");

                    t.Cuda.SetMemory(hMemD, rgf);

                    double[] rgdf1 = t.Cuda.GetMemoryDouble(hMemD);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgdf1.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgdf1[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.GetMemoryFloat(hMemD);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgf1.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgf1[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    hMemF = t.Cuda.AllocMemory(4);
                    t.Log.CHECK_NE(0, hMemF, "The hMemF handle should not be null!");

                    t.Cuda.SetMemory(hMemF, rgf);

                    double[] rgdf2 = t.Cuda.GetMemoryDouble(hMemF);
                    t.Log.CHECK_EQ(rgdf2.Length, rgf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgdf2.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgdf2[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf2 = t.Cuda.GetMemoryFloat(hMemD);
                    t.Log.CHECK_EQ(rgf2.Length, rgdf.Count, "The data arrays should have the same number of items!");

                    for (int i = 0; i < rgf2.Length; i++)
                    {
                        t.Log.CHECK_EQ(rgf2[i], rgdf[i], "The data items at " + i.ToString() + " are not equal!");
                    }

                    t.Cuda.FreeMemory(hMemD);
                    t.Cuda.FreeMemory(hMemF);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllocHalf()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>();
            long hMemF = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;

                    for (float f = 0; f < 1.0f; f += 0.01f)
                    {
                        rgf.Add((float)Math.Sin((double)f));
                    }

                    hMemF = t.Cuda.AllocMemory(rgf.Count, true);
                    t.Log.CHECK_NE(hMemF, 0, "The memory should not be null!");
                    t.Cuda.SetMemory(hMemF, rgf);

                    float[] rgf1 = t.Cuda.GetMemoryFloat(hMemF);

                    t.Log.CHECK_GE(rgf1.Length, rgf.Count, "The data returned is not the same count as the data set.");

                    for (int i = 0; i < rgf.Count; i++)
                    {
                        float f1 = rgf1[i];
                        float f = rgf[i];

                        t.Log.EXPECT_NEAR_FLOAT(f1, f, 0.001, "The values should be the same!");
                    }

                    t.Cuda.FreeMemory(hMemF);
                    hMemF = 0;
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllocHalf2()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>();
            long hMemF = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;

                    for (float f = 0; f < 1.0f; f += 0.01f)
                    {
                        rgf.Add((float)Math.Sin((double)f));
                    }

                    hMemF = t.Cuda.AllocMemory(rgf.Count, true);
                    t.Log.CHECK_NE(hMemF, 0, "The memory should not be null!");

                    for (int i = 0; i < rgf.Count; i++)
                    {
                        t.Cuda.SetMemoryAt(hMemF, new float[] { rgf[i] }, i);
                    }

                    float[] rgf1 = t.Cuda.GetMemoryFloat(hMemF);

                    t.Log.CHECK_GE(rgf1.Length, rgf.Count, "The data returned is not the same count as the data set.");

                    for (int i = 0; i < rgf.Count; i++)
                    {
                        float f1 = rgf1[i];
                        float f = rgf[i];

                        t.Log.EXPECT_NEAR_FLOAT(f1, f, 0.001, "The values should be the same!");
                    }

                    t.Cuda.FreeMemory(hMemF);
                    hMemF = 0;
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAxpyHalf()
        { 
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>();
            long hMemF1 = 0;
            long hMemF2 = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        continue;

                    long lCount = 20948400;
                    hMemF1 = t.Cuda.AllocMemory(lCount, true);
                    hMemF2 = t.Cuda.AllocMemory(lCount, true);

                    t.Cuda.axpy((int)lCount, -1.0, hMemF1, hMemF2);

                    t.Cuda.FreeMemory(hMemF1);
                    hMemF1 = 0;

                    t.Cuda.FreeMemory(hMemF2);
                    hMemF2 = 0;
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_sumsqdiff()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long h1 = 0;
                    long h2 = 0;
                    long h3 = 0;

                    try
                    {
                        List<double> rg1 = new List<double>();
                        List<double> rg2 = new List<double>();
                        int nCount = 1000;
                    
                        for (int i=0; i<nCount; i++)
                        {
                            rg1.Add(i * 0.2);
                            rg2.Add(i * 0.1);
                        }

                        h1 = t.Cuda.AllocMemory(rg1);
                        h2 = t.Cuda.AllocMemory(rg2);
                        h3 = t.Cuda.AllocMemory(nCount);
                        int nPtA = nCount / 4;
                        int nPtB = nCount / 2;

                        double dfSumSqDiff1 = t.Cuda.sumsqdiff(nCount, h3, h1, h2);
                        double dfSumSqDiff2 = t.Cuda.sumsqdiff(nCount - nPtB, h3, h1, h2, nPtA, nPtB);

                        t.Log.CHECK_GT(dfSumSqDiff1, dfSumSqDiff2, "The first sumsqdiff should be greater than the second.");

                        double dfSumSqDiff1B = 0;
                        double dfSumSqDiff2B = 0;

                        for (int i = 0; i < nCount; i++)
                        {
                            double dfDiff = rg1[i] - rg2[i];
                            dfSumSqDiff1B += dfDiff * dfDiff;

                            if (i < nCount - nPtB)
                            {
                                dfDiff = rg1[i + nPtA] - rg2[i + nPtB];
                                dfSumSqDiff2B += dfDiff * dfDiff;
                            }
                        }

                        if (t.DataType == DataType.FLOAT)
                        {
                            t.Log.EXPECT_NEAR(dfSumSqDiff1, dfSumSqDiff1B, 0.5);
                            t.Log.EXPECT_NEAR(dfSumSqDiff2, dfSumSqDiff2B, 0.5);
                        }
                        else
                        {
                            t.Log.EXPECT_NEAR(dfSumSqDiff1, dfSumSqDiff1B, 0.001);
                            t.Log.EXPECT_NEAR(dfSumSqDiff2, dfSumSqDiff2B, 0.001);
                        }
                    }
                    finally
                    {
                        if (h1 > 0)
                            t.Cuda.FreeMemory(h1);

                        if (h2 > 0)
                            t.Cuda.FreeMemory(h2);

                        if (h3 > 0)
                            t.Cuda.FreeMemory(h3);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_sumsq()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long h1 = 0;
                    long h2 = 0;

                    try
                    {
                        List<double> rg1 = new List<double>();
                        int nCount = 1000;

                        for (int i = 0; i < nCount; i++)
                        {
                            rg1.Add(i * 0.1);
                        }

                        h1 = t.Cuda.AllocMemory(rg1);
                        h2 = t.Cuda.AllocMemory(nCount);
                        int nPtA = nCount / 4;

                        double dfSumSq1 = t.Cuda.sumsq(nCount, h2, h1);
                        double dfSumSq2 = t.Cuda.sumsq(nCount - nPtA, h2, h1, nPtA);

                        t.Log.CHECK_LT(dfSumSq2, dfSumSq1, "The first sumsq should be greater than the second.");

                        double dfSumSq1B = 0;
                        double dfSumSq2B = 0;

                        for (int i = 0; i < nCount; i++)
                        {
                            dfSumSq1B += rg1[i] * rg1[i];

                            if (i < nCount - nPtA)
                            {
                                dfSumSq2B += rg1[i + nPtA] * rg1[i + nPtA];
                            }
                        }

                        if (t.DataType == DataType.FLOAT)
                        {
                            t.Log.EXPECT_NEAR(dfSumSq1, dfSumSq1B, 0.5);
                            t.Log.EXPECT_NEAR(dfSumSq2, dfSumSq2B, 0.5);
                        }
                        else
                        {
                            t.Log.EXPECT_NEAR(dfSumSq1, dfSumSq1B, 0.001);
                            t.Log.EXPECT_NEAR(dfSumSq2, dfSumSq2B, 0.001);
                        }
                    }
                    finally
                    {
                        if (h1 > 0)
                            t.Cuda.FreeMemory(h1);

                        if (h2 > 0)
                            t.Cuda.FreeMemory(h2);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_sqrt_scale()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long h1 = 0;
                    long h2 = 0;

                    try
                    {
                        List<double> rg1 = new List<double>();
                        int nCount = 1000;

                        for (int i = 0; i < nCount; i++)
                        {
                            rg1.Add((i - nCount/2) * 0.1);
                        }

                        h1 = t.Cuda.AllocMemory(rg1);
                        h2 = t.Cuda.AllocMemory(nCount);

                        t.Cuda.sqrt_scale(nCount, h1, h2);

                        if (t.DataType == DataType.DOUBLE)
                        {
                            double[] rg2 = t.Cuda.get_double(nCount, h2);

                            for (int i = 0; i < rg1.Count; i++)
                            {
                                double dfSqrtScale = Math.Abs(rg1[i]);
                                dfSqrtScale = Math.Sqrt(dfSqrtScale);
                                dfSqrtScale *= Math.Sign(rg1[i]);

                                t.Log.CHECK_EQ(dfSqrtScale, rg2[i], "The scaled values at i=" + i.ToString() + " are not the same!");
                            }
                        }
                        else
                        {
                            float[] rg2 = t.Cuda.get_float(nCount, h2);

                            for (int i = 0; i < rg1.Count; i++)
                            {
                                float dfSqrtScale = (float)Math.Abs(rg1[i]);
                                dfSqrtScale = (float)Math.Sqrt(dfSqrtScale);
                                dfSqrtScale *= Math.Sign(rg1[i]);

                                t.Log.CHECK_EQ(dfSqrtScale, rg2[i], "The scaled values at i=" + i.ToString() + " are not the same!");
                            }
                        }
                    }
                    finally
                    {
                        if (h1 > 0)
                            t.Cuda.FreeMemory(h1);

                        if (h2 > 0)
                            t.Cuda.FreeMemory(h2);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_ger()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long hX = 0;
                    long hY = 0;
                    long hA = 0;

                    try
                    {
                        List<double> rgX = new List<double>();
                        List<double> rgY = new List<double>();
                        int nM = 12;
                        int nN = 12;

                        for (int i = 0; i < nM; i++)
                        {
                            rgX.Add((i - nM / 2) * 0.1);
                        }

                        for (int i = 0; i < nN; i++)
                        {
                            rgY.Add((i - nN / 4) * 0.4);
                        }

                        hX = t.Cuda.AllocMemory(rgX);
                        hY = t.Cuda.AllocMemory(rgY);
                        hA = t.Cuda.AllocMemory(nM * nN);

                        t.Cuda.ger(nM, nN, 1.0f, hX, hY, hA);

                        if (t.DataType == DataType.DOUBLE)
                        {
                            double[] rgX1 = t.Cuda.get_double(nM, hX);
                            double[] rgY1 = t.Cuda.get_double(nN, hY);
                            double[] rgA1 = t.Cuda.get_double(nM * nN, hA);

                            for (int i = 0; i < rgY1.Length; i++)
                            {
                                for (int j = 0; j < rgX1.Length; j++)
                                {
                                    double dfValExpected = rgY1[i] * rgX1[j];
                                    double dfVal = rgA1[i * rgX1.Length + j];
                                    t.Log.EXPECT_NEAR(dfVal, dfValExpected, 0.00001, "The value is not as expected.");
                                }
                            }
                        }
                        else
                        {
                            double[] rgX1 = t.Cuda.get_double(nM, hX);
                            double[] rgY1 = t.Cuda.get_double(nN, hY);
                            double[] rgA1 = t.Cuda.get_double(nM * nN, hA);

                            for (int i = 0; i < rgY1.Length; i++)
                            {
                                for (int j = 0; j < rgX1.Length; j++)
                                {
                                    double dfValExpected = rgY1[i] * rgX1[j];
                                    double dfVal = rgA1[i * rgX1.Length + j];
                                    t.Log.EXPECT_NEAR(dfVal, dfValExpected, 0.00001, "The value is not as expected.");
                                }
                            }
                        }
                    }
                    finally
                    {
                        if (hX > 0)
                            t.Cuda.FreeMemory(hX);

                        if (hY > 0)
                            t.Cuda.FreeMemory(hY);

                        if (hA > 0)
                            t.Cuda.FreeMemory(hA);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_setget()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            int nCount = rgf.Count;
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hSrc = t.Cuda.AllocMemory(4);
                    t.Log.CHECK_NE(0, hSrc, "The source should have a valid handle.");
                    t.Cuda.SetMemory(hSrc, rgdf.ToArray());

                    double[] rgdf1 = t.Cuda.get_double(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.get_float(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf1[i], "The values at " + i.ToString() + " are not equal!");
                    }


                    hDst = t.Cuda.AllocMemory(4);
                    t.Log.CHECK_NE(0, hDst, "The source should have a valid handle.");
                    t.Cuda.set(nCount, hDst, 0);

                    for (int i = 0; i < nCount; i++)
                    {
                        double[] rgD = t.Cuda.get_double(nCount, hDst, i);
                        t.Log.CHECK_EQ(1, rgD.Length, "The return array should only have 1 element.");
                        t.Log.CHECK_EQ(0, rgD[0], "The item returned should be zero.");

                        float[] rgF = t.Cuda.get_float(nCount, hDst, i);
                        t.Log.CHECK_EQ(1, rgF.Length, "The return array should only have 1 element.");
                        t.Log.CHECK_EQ(0, rgF[0], "The item returned should be zero.");

                        t.Cuda.set(nCount, hDst, (i + 1) * 2.5, i);
                        rgD = t.Cuda.get_double(nCount, hDst, i);
                        t.Log.CHECK_EQ(1, rgD.Length, "The return array should only have 1 element.");
                        t.Log.CHECK_EQ((i + 1) * 2.5, rgD[0], "The item returned should be " + ((i + 1) * 2.5).ToString());

                        t.Cuda.set(nCount, hDst, (i + 1) * 2.7, i);
                        rgF = t.Cuda.get_float(nCount, hDst, i);
                        t.Log.CHECK_EQ(1, rgF.Length, "The return array should only have 1 element.");
                        t.Log.CHECK_EQ((i + 1) * 2.7f, rgF[0], "The item returned should be " + ((i + 1) * 2.7).ToString());
                    }

                    t.Cuda.FreeMemory(hSrc);
                    t.Cuda.FreeMemory(hDst);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_copy()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            int nCount = rgf.Count;
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hSrc = t.Cuda.AllocMemory(nCount);
                    t.Log.CHECK_NE(0, hSrc, "The source should have a valid handle.");
                    t.Cuda.SetMemory(hSrc, rgdf.ToArray());

                    double[] rgdf1 = t.Cuda.get_double(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.get_float(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    hDst = t.Cuda.AllocMemory(nCount);
                    t.Cuda.set(nCount, hDst, 0);

                    t.Cuda.copy(nCount, hSrc, hDst);

                    double[] rgdf2 = t.Cuda.get_double(nCount, hDst, -1);
                    t.Log.CHECK_EQ(rgdf2.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf2[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf2 = t.Cuda.get_float(nCount, hDst, -1);
                    t.Log.CHECK_EQ(rgf2.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf2[i], "The values at " + i.ToString() + " are not equal!");
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_copy_smallset()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            int nCount = rgf.Count;
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hSrc = t.Cuda.AllocMemory(nCount);
                    t.Log.CHECK_NE(0, hSrc, "The source should have a valid handle.");
                    t.Cuda.SetMemory(hSrc, rgdf.ToArray());

                    double[] rgdf1 = t.Cuda.get_double(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.get_float(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    hDst = t.Cuda.AllocMemory(nCount);
                    t.Cuda.set(nCount, hDst, 0);

                    t.Cuda.copy(nCount - 2, hSrc, hDst);

                    double[] rgdf2 = t.Cuda.get_double(nCount, hDst, -1);
                    t.Log.CHECK_EQ(rgdf2.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount - 2; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf2[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    for (int i = nCount - 2; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(0, rgdf2[i], "The values at " + i.ToString() + " should be zero!");
                    }

                    float[] rgf2 = t.Cuda.get_float(nCount, hDst, -1);
                    t.Log.CHECK_EQ(rgf2.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount - 2; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf2[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    for (int i = nCount - 2; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(0, rgf2[i], "The values at " + i.ToString() + " should be zero!");
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_copy_smallset_zerogarbage()
        {
            CudaDnnTest test = new CudaDnnTest();
            List<float> rgf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            List<float> rgdf = new List<float>() { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            int nCount = rgf.Count;
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hSrc = t.Cuda.AllocMemory(nCount);
                    t.Log.CHECK_NE(0, hSrc, "The source should have a valid handle.");
                    t.Cuda.SetMemory(hSrc, rgdf.ToArray());

                    double[] rgdf1 = t.Cuda.get_double(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgdf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    float[] rgf1 = t.Cuda.get_float(nCount, hSrc, -1);
                    t.Log.CHECK_EQ(rgf1.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf1[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    hDst = t.Cuda.AllocMemory(nCount);
                    t.Cuda.set(nCount, hDst, 9);
                    t.Cuda.copy(nCount - 2, hSrc, hDst);

                    double[] rgdf2 = t.Cuda.get_double(nCount, hDst, -1);
                    t.Log.CHECK_EQ(rgdf2.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount - 2; i++)
                    {
                        t.Log.CHECK_EQ(rgdf[i], rgdf2[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    for (int i = nCount - 2; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(9, rgdf2[i], "The values at " + i.ToString() + " should be 9!");
                    }

                    float[] rgf2 = t.Cuda.get_float(nCount, hDst, -1);
                    t.Log.CHECK_EQ(rgf2.Length, rgdf.Count, "The arrays should have the same count.");

                    for (int i = 0; i < nCount - 2; i++)
                    {
                        t.Log.CHECK_EQ(rgf[i], rgf2[i], "The values at " + i.ToString() + " are not equal!");
                    }

                    for (int i = nCount - 2; i < nCount; i++)
                    {
                        t.Log.CHECK_EQ(9, rgf2[i], "The values at " + i.ToString() + " should be 9!");
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestStream()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hStream = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hStream = t.Cuda.CreateStream();
                    t.Log.CHECK_NE(0, hStream, "The stream handle is null!");
                    t.Cuda.FreeStream(hStream);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCuDNN()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hCuDNN = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hCuDNN = t.Cuda.CreateCuDNN();
                    t.Log.CHECK_NE(0, hCuDNN, "The cudnn handle is null!");
                    t.Cuda.FreeCuDNN(hCuDNN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTensorDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hTensor = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hTensor = t.Cuda.CreateTensorDesc();
                    t.Log.CHECK_NE(0, hTensor, "The tensor handle is null!");

                    t.Cuda.SetTensorDesc(hTensor, 10, 20, 30, 40);
                    t.Cuda.SetTensorDesc(hTensor, 20, 30, 40, 50, 2, 3, 4, 5);

                    t.Cuda.FreeTensorDesc(hTensor);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hFilter = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hFilter = t.Cuda.CreateFilterDesc();
                    t.Log.CHECK_NE(0, hFilter, "The filter handle is null!");

                    t.Cuda.SetFilterDesc(hFilter, 10, 20, 30, 40);
                    t.Cuda.FreeFilterDesc(hFilter);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConvolutionDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hConv = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hConv = t.Cuda.CreateConvolutionDesc();
                    t.Log.CHECK_NE(0, hConv, "The convolution handle is null!");

                    t.Cuda.SetConvolutionDesc(hConv, 10, 10, 2, 3);
                    t.Cuda.FreeConvolutionDesc(hConv);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPoolingDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hPool = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hPool = t.Cuda.CreatePoolingDesc();
                    t.Log.CHECK_NE(0, hPool, "The pooling handle is null!");

                    t.Cuda.SetPoolingDesc(hPool, PoolingMethod.AVE, 10, 20, 2, 3, 4, 5);
                    t.Cuda.SetPoolingDesc(hPool, PoolingMethod.MAX, 10, 20, 2, 3, 4, 5);

                    t.Cuda.FreePoolingDesc(hPool);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLRNDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hLRN = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hLRN = t.Cuda.CreateLRNDesc();
                    t.Log.CHECK_NE(0, hLRN, "The LRN handle is null!");

                    t.Cuda.SetLRNDesc(hLRN, 2, 1.2, 2.1, 0.2);

                    t.Cuda.FreeLRNDesc(hLRN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRnnDataDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hDesc = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hDesc = t.Cuda.CreateRnnDataDesc();
                    t.Log.CHECK_NE(0, hDesc, "The hDesc handle should not be null!");

                    int nMaxSeqLen = 10;
                    int nBatchLen = 5;
                    int nVectorLen = 2;

                    t.Cuda.SetRnnDataDesc(hDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nMaxSeqLen, nBatchLen, nVectorLen);
                    t.Cuda.FreeRnnDataDesc(hDesc);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRnnDesc()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hDesc = 0;
            long hCudnn = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hCudnn = t.Cuda.CreateCuDNN();
                    hDesc = t.Cuda.CreateRnnDesc();

                    int nHiddenSize = 10;
                    int nNumLayers = 1;

                    t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM);

                    t.Cuda.FreeRnnDesc(hDesc);
                    t.Cuda.FreeCuDNN(hCudnn);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRnnGetParamCount()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hDesc = 0;
            long hDataDesc = 0;
            long hCudnn = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hCudnn = t.Cuda.CreateCuDNN();
                    hDesc = t.Cuda.CreateRnnDesc();
                    hDataDesc = t.Cuda.CreateRnnDataDesc();

                    int nHiddenSize = 10;
                    int nSeqLen = 10;
                    int nBatchSize = 5;
                    int nNumLayers = 1;

                    t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM);
                    t.Cuda.SetRnnDataDesc(hDataDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                    int nCount = t.Cuda.GetRnnParamCount(hCudnn, hDesc, hDataDesc);

                    t.Cuda.FreeRnnDesc(hDesc);
                    t.Cuda.FreeRnnDataDesc(hDataDesc);
                    t.Cuda.FreeCuDNN(hCudnn);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRnnGetWorkspaceCount()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hDesc = 0;
            long hDataDesc = 0;
            long hCudnn = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hCudnn = t.Cuda.CreateCuDNN();
                    hDesc = t.Cuda.CreateRnnDesc();
                    hDataDesc = t.Cuda.CreateRnnDataDesc();

                    int nSeqLen = 10;
                    int nHiddenSize = 3;
                    int nBatchSize = 5;
                    int nNumLayers = 1;

                    t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM);
                    t.Cuda.SetRnnDataDesc(hDataDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                    int nReservedCount;
                    int nWorkspaceCount = t.Cuda.GetRnnWorkspaceCount(hCudnn, hDesc, hDataDesc, out nReservedCount);

                    t.Cuda.FreeRnnDesc(hDesc);
                    t.Cuda.FreeRnnDataDesc(hDataDesc);
                    t.Cuda.FreeCuDNN(hCudnn);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRnnGetLinLayerParams()
        {
            CudaDnnTest test = new CudaDnnTest();
            long hDesc = 0;
            long hDataDesc = 0;
            long hCudnn = 0;
            long hWtData = 0;
            long hWtDesc = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hCudnn = t.Cuda.CreateCuDNN();
                    hDesc = t.Cuda.CreateRnnDesc();
                    hDataDesc = t.Cuda.CreateRnnDataDesc();

                    int nSeqLen = 10;
                    int nHiddenSize = 3;
                    int nBatchSize = 5;
                    int nNumLayers = 1;

                    t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM);
                    t.Cuda.SetRnnDataDesc(hDataDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                    int nAllWtCount = t.Cuda.GetRnnParamCount(hCudnn, hDesc, hDataDesc);
                    hWtDesc = t.Cuda.CreateFilterDesc();
                    hWtData = t.Cuda.AllocMemory(nAllWtCount);

                    int[] rgDimWt = new int[3];
                    rgDimWt[0] = nAllWtCount;
                    rgDimWt[1] = 1;
                    rgDimWt[2] = 1;

                    t.Cuda.SetFilterNdDesc(hWtDesc, rgDimWt);
                    Exception err = null;

                    try
                    {
                        int nLinLayers = 8; // LSTM
                        for (int i = 0; i < nNumLayers; i++)
                        {
                            for (int j = 0; j < nLinLayers; j++)
                            {
                                int nWtCount;
                                long hWt;
                                int nBiasCount;
                                long hBias;

                                t.Cuda.GetRnnLinLayerParams(hCudnn, hDesc, i, hDataDesc, hWtDesc, hWtData, j, out nWtCount, out hWt, out nBiasCount, out hBias);

                                Assert.AreNotEqual(nWtCount, 0, "The weight data count should not be zero!");
                                Assert.AreNotEqual(hWt, 0, "The weight handle should not be zero!");
                                Assert.AreNotEqual(nBiasCount, 0, "The bias data count should not be zero!");
                                Assert.AreNotEqual(hBias, 0, "The bias handle should not be zero!");

                                t.Cuda.FreeMemoryPointer(hWt);
                                t.Cuda.FreeMemoryPointer(hBias);
                            }
                        }
                    }
                    catch (Exception excpt)
                    {
                        err = excpt;
                    }

                    t.Cuda.FreeMemory(hWtData);
                    t.Cuda.FreeFilterDesc(hWtDesc);
                    t.Cuda.FreeRnnDesc(hDesc);
                    t.Cuda.FreeRnnDataDesc(hDataDesc);
                    t.Cuda.FreeCuDNN(hCudnn);

                    if (err != null)
                        throw err;
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMemoryTestByBlock()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMemoryTestByBlock();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMemoryTestAll()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMemoryTestAll();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMemoryPointers()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMemoryPointers();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAdd()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestAdd();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGemm()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestGemm();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGemv()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestGemv();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSum()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestSum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestHammingDistance()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestHammingDistance();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatrix()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMatrix();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    public interface ITestCudaDnn : ITest
    {
        void TestAdd();
        void TestGemm();
        void TestGemv();
        void TestSum();
        void TestMemoryTestByBlock();
        void TestMemoryTestAll();
        void TestMemoryPointers();
        void TestHammingDistance();
        void TestMatrix();
    }

    class CudaDnnTest : TestBase
    {
        public CudaDnnTest()
            : base("CudaDnn Test")
        {
        }

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            if (dt == common.DataType.DOUBLE)
                return new CudaDnnTest<double>(strName, nDeviceID, engine);
            else
                return new CudaDnnTest<float>(strName, nDeviceID, engine);
        }
    }

    class CudaDnnTest<T> : TestEx<T>, ITestCudaDnn
    {
        Blob<T> m_temp;
        Blob<T> m_A = null;
        Blob<T> m_B = null;
        Blob<T> m_C = null;
        Blob<T> m_x = null;
        Blob<T> m_y = null;

        public CudaDnnTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_temp = new Blob<T>(m_cuda, m_log);
            m_A = new Blob<T>(m_cuda, m_log, 1, 1, 2, 3);
            m_B = new Blob<T>(m_cuda, m_log, 1, 1, 3, 4);
            m_C = new Blob<T>(m_cuda, m_log, 1, 1, 2, 4);
            m_x = new Blob<T>(m_cuda, m_log, 1, 1, 1, 3);
            m_y = new Blob<T>(m_cuda, m_log, 1, 1, 1, 2);
        }

        protected override void dispose()
        {
            if (m_temp != null)
            {
                m_temp.Dispose();
                m_temp = null;
            }

            if (m_A != null)
            {
                m_A.Dispose();
                m_A = null;
            }

            if (m_B != null)
            {
                m_B.Dispose();
                m_B = null;
            }

            if (m_C != null)
            {
                m_C.Dispose();
                m_C = null;
            }

            if (m_x != null)
            {
                m_x.Dispose();
                m_x = null;
            }

            if (m_y != null)
            {
                m_y.Dispose();
                m_y = null;
            }
            
            base.dispose();
        }

        public void TestAdd()
        {
            double[] rgData1 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            double[] rgData2 = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.12, 0.11, 0.121 };

            m_A.Reshape(rgData1.Length, 1, 1, 1);
            m_B.ReshapeLike(m_A);
            m_C.ReshapeLike(m_B);

            m_A.mutable_cpu_data = convert(rgData1.ToArray());
            m_B.mutable_cpu_data = convert(rgData2.ToArray());
            m_C.SetData(0);

            m_cuda.add(m_A.count(), m_A.gpu_data, m_B.gpu_data, m_C.mutable_gpu_data);

            double[] rgResult = convert(m_C.mutable_cpu_data);

            for (int i = 0; i < rgData1.Length; i++)
            {
                double dfExpected = rgData1[i] + rgData2[i];
                double dfActual = rgResult[i];

                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.001, "The expected and actual are not the same.");
            }

            ulong lWorkSize = m_B.GetConversionWorkSize(true);
            long hWorkMem = m_cuda.AllocMemory((long)lWorkSize);

            m_A.ConvertToHalf(hWorkMem, lWorkSize, true, false);
            m_B.ConvertToHalf(hWorkMem, lWorkSize, true, false);
            m_C.ConvertToHalf(hWorkMem, lWorkSize, true, false);

            m_A.mutable_cpu_data = convert(rgData1.ToArray());
            m_B.mutable_cpu_data = convert(rgData2.ToArray());
            m_C.SetData(0);

            m_cuda.add(m_A.count(), m_A.gpu_data, m_B.gpu_data, m_C.mutable_gpu_data);

            rgResult = convert(m_C.mutable_cpu_data);

            for (int i = 0; i < rgData1.Length; i++)
            {
                double dfExpected = rgData1[i] + rgData2[i];
                double dfActual = rgResult[i];

                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.01, "The expected and actual are not the same.");
            }

            m_cuda.FreeMemory(hWorkMem);
        }

        public void TestGemm()
        {
            double[] rgData = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            double[] A_reshape_data = new double[] { 1, 4, 2, 5, 3, 6 };
            double[] B_reshape_data = new double[] { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
            double[] rgResult = new double[] { 38, 44, 50, 56, 83, 98, 113, 128 };

            m_temp.Reshape(new List<int>() { rgData.Length, 1, 1, 1 });
            m_temp.mutable_cpu_data = convert(rgData);
            m_cuda.copy(6, m_temp.gpu_data, m_A.mutable_gpu_data);
            m_cuda.copy(12, m_temp.gpu_data, m_B.mutable_gpu_data);

            // [1, 2, 3; 4, 5, 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
            m_cuda.gemm(false, false, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            m_cuda.gemm(false, false, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            // Test when we have a transposed A.
            m_A.Reshape(1, 1, 3, 2);
            m_temp.mutable_cpu_data = convert(A_reshape_data);
            m_cuda.copy(6, m_temp.gpu_data, m_A.mutable_gpu_data);

            m_cuda.gemm(true, false, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            m_cuda.gemm(true, false, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }


            // Test when we have a transposed A and a transposed B too.
            m_B.Reshape(1, 1, 4, 3);
            m_temp.mutable_cpu_data = convert(B_reshape_data);
            m_cuda.copy(12, m_temp.gpu_data, m_B.mutable_gpu_data);

            m_cuda.gemm(true, true, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            m_cuda.gemm(true, true, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            // Test when we have a transposed B.
            m_A.Reshape(1, 1, 2, 3);
            m_temp.mutable_cpu_data = convert(rgData);
            m_cuda.copy(6, m_temp.gpu_data, m_A.mutable_gpu_data);

            m_cuda.gemm(false, true, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            m_cuda.gemm(false, true, 2, 4, 3, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            for (int i = 0; i < 8; i++)
            {
                double dfVal = convert(m_C.GetData(i));
                double dfRes = rgResult[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }
        }

        public void TestGemv()
        {
            double[] rgData = new double[] { 1, 2, 3, 4, 5, 6 };
            double[] result_2 = new double[] { 14, 32 };
            double[] result_3 = new double[] { 9, 12, 15 };

            m_temp.Reshape(new List<int>() { rgData.Length, 1, 1, 1 });
            m_temp.mutable_cpu_data = convert(rgData);
            m_cuda.copy(6, m_temp.gpu_data, m_A.mutable_gpu_data);
            m_cuda.copy(3, m_temp.gpu_data, m_x.mutable_gpu_data);

            m_cuda.gemv(false, 2, 3, 1.0, m_A.gpu_data, m_x.gpu_data, 0.0, m_y.mutable_gpu_data);

            for (int i = 0; i < 2; i++)
            {
                double dfVal = convert(m_y.GetData(i));
                double dfRes = result_2[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            m_cuda.gemv(false, 2, 3, 1.0, m_A.gpu_data, m_x.gpu_data, 0.0, m_y.mutable_gpu_data);

            for (int i = 0; i < 2; i++)
            {
                double dfVal = convert(m_y.GetData(i));
                double dfRes = result_2[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }


            // Test transpose case
            m_cuda.copy(2, m_temp.gpu_data, m_y.mutable_gpu_data);

            m_cuda.gemv(true, 2, 3, 1.0, m_A.gpu_data, m_y.gpu_data, 0.0, m_x.mutable_gpu_data);

            for (int i = 0; i < 3; i++)
            {
                double dfVal = convert(m_x.GetData(i));
                double dfRes = result_3[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }

            m_cuda.gemv(true, 2, 3, 1.0, m_A.gpu_data, m_y.gpu_data, 0.0, m_x.mutable_gpu_data);

            for (int i = 0; i < 3; i++)
            {
                double dfVal = convert(m_x.GetData(i));
                double dfRes = result_3[i];
                m_log.CHECK_EQ(dfVal, dfRes, "The value does not match the expected result.");
            }
        }

        public void TestSum()
        {
            double[] rgData = new double[]
            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                10, 20, 30, 40, 50, 60, 70, 80, 90, 100,

                1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1,
                10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 90.1, 100.1,

                1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2,
                10.2, 20.2, 30.2, 40.2, 50.2, 60.2, 70.2, 80.2, 90.2, 100.2
            };

            m_A.Reshape(3, 2, 10, 1);
            m_A.mutable_cpu_data = convert(rgData);

            m_cuda.sum(m_A.count(), 6, 10, m_A.gpu_data, m_A.mutable_gpu_diff);
            double[] rgResult = convert(m_A.mutable_cpu_diff);
            double[] rgExpected = new double[6];

            for (int i = 0; i < 6; i++)
            {
                rgExpected[i] = 0;

                for (int j = 0; j < 10; j++)
                {
                    rgExpected[i] += rgData[(i * 10) + j];
                }
            }

            for (int i = 0; i < 6; i++)
            {
                m_log.EXPECT_NEAR_FLOAT(rgExpected[i], rgResult[i], 0.0001, "The values do not match at i = " + i.ToString());
            }
        }

        public void TestMemoryTestByBlock()
        {
            ulong ulTotalBlocks;
            double dfTotalMemAllocated;
            ulong ulMemStartAddr;
            ulong ulBlockSize;
            long hMemTest = m_cuda.CreateMemoryTest(out ulTotalBlocks, out dfTotalMemAllocated, out ulMemStartAddr, out ulBlockSize);

            for (ulong ulIdx = 0; ulIdx < ulTotalBlocks; ulIdx++)
            {
                T[] rg = m_cuda.RunMemoryTest(hMemTest, MEMTEST_TYPE.MOV_INV_8, ulIdx, 1, true, true, true, true);
                double[] rgData = convert(rg);

                ulong ulStartAddr = (ulong)rgData[0];
                ulong ulAddrCount = (ulong)rgData[1];
                int nErrCount = (int)rgData[2];

                Trace.WriteLine("Start address = 0x" + ulStartAddr.ToString("X"));
                Trace.WriteLine("address count = " + ulAddrCount.ToString("N0"));
                Trace.WriteLine("error count = " + nErrCount.ToString("N0"));

                for (int i = 0; i < nErrCount; i++)
                {
                    Trace.WriteLine("error address " + i.ToString() + " = " + rgData[3 + i].ToString("X"));
                }

                double dfPct = (double)ulIdx / (double)ulTotalBlocks;
                Trace.WriteLine("Percentage Complete: " + dfPct.ToString("P"));
            }

            m_cuda.FreeMemoryTest(hMemTest);
        }

        public void TestMemoryTestAll()
        {
            ulong ulTotalBlocks;
            double dfTotalMemAllocated;
            ulong ulMemStartAddr;
            ulong ulBlockSize;
            long hMemTest = m_cuda.CreateMemoryTest(out ulTotalBlocks, out dfTotalMemAllocated, out ulMemStartAddr, out ulBlockSize);

            T[] rg = m_cuda.RunMemoryTest(hMemTest, MEMTEST_TYPE.MOV_INV_8, 0, ulTotalBlocks, true, true, true, true);
            double[] rgData = convert(rg);

            ulong ulStartAddr = (ulong)rgData[0];
            ulong ulAddrCount = (ulong)rgData[1];
            int nErrCount = (int)rgData[2];

            Trace.WriteLine("Start address = 0x" + ulStartAddr.ToString("X"));
            Trace.WriteLine("address count = " + ulAddrCount.ToString("N0"));
            Trace.WriteLine("error count = " + nErrCount.ToString("N0"));

            for (int i = 0; i < nErrCount; i++)
            {
                Trace.WriteLine("error address " + i.ToString() + " = " + rgData[3 + i].ToString("X"));
            }

            m_cuda.FreeMemoryTest(hMemTest);
        }

        public void TestMemoryPointers()
        {
            long hMem = m_cuda.AllocMemory(1000);
            List<long> rghMem = new List<long>();

            m_cuda.set(1000, hMem, 0);
            long lOffset = 0;

            for (int i = 0; i < 10; i++)
            {
                long hMem1 = m_cuda.CreateMemoryPointer(hMem, lOffset, 100);
                m_cuda.set(100, hMem1, i + 1);
                rghMem.Add(hMem1);
                lOffset += 100;
            }

            for (int i = 0; i < 10; i++)
            {
                long hMem1 = rghMem[i];
                double[] rgData = convert(m_cuda.GetMemory(hMem1));

                Assert.AreEqual(rgData.Length, 100);

                for (int j = 0; j < rgData.Length; j++)
                {
                    Assert.AreEqual(rgData[j], i + 1);
                }
            }

            for (int i = 0; i < 10; i++)
            {
                long hMem1 = rghMem[i];
                m_cuda.FreeMemoryPointer(hMem1);
            }

            double[] rgData1 = convert(m_cuda.GetMemory(hMem));
            Assert.AreEqual(rgData1.Length, 1000);

            for (int i = 0; i < 10; i++)
            {
                lOffset = i * 100;

                for (int j = 0; j < 100; j++)
                {
                    Assert.AreEqual(rgData1[lOffset + j], i + 1);
                }
            }
        }

        public void TestHammingDistance()
        {
            double dfThreshold = 0.5;
            double[] rgDataA = new double[] { 3.2, 4.5, 8.3, -1.2, 0.03, 0.22 };    // binarified(thresh = 0.5): 1, 1, 1,  0, 0,  0
            double[] rgDataB = new double[] { 0.2, 4.9, 8.1,  9.3,  0.1, 0.88 };    // binarified(thresh = 0.5): 0, 1, 1,  1, 0,  1
                                                                                    // hamming difference:       1, 0, 0, -1, 0, -1 = 3 hamming distance (sum of abs value)
            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { rgDataA.Length });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { rgDataB.Length });   // both A and B must have same length.
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.ReshapeLike(m_A);

            m_A.mutable_cpu_data = convert(rgDataA);
            m_B.mutable_cpu_data = convert(rgDataB);

            double dfHammingDistance = m_cuda.hamming_distance(rgDataA.Length, dfThreshold, m_A.gpu_data, m_B.gpu_data, m_C.mutable_gpu_data);

            m_log.CHECK_EQ(3.0, dfHammingDistance, "The hamming distance should = 3.0");

            double[] rgDataC = convert(m_C.mutable_cpu_data);

            m_log.CHECK_EQ(rgDataC.Length, rgDataA.Length, "The length of C is incorrect.");

            // Calculate the hamming distance.
            List<double> rgExpected = new List<double>();

            for (int i = 0; i < rgDataA.Length; i++)
            {
                double dfA = (rgDataA[i] > dfThreshold) ? 1 : 0;
                double dfB = (rgDataB[i] > dfThreshold) ? 1 : 0;
                double dfDiff = dfA - dfB;
                rgExpected.Add(dfDiff);
            }

            for (int i = 0; i < rgDataC.Length; i++)
            {
                m_log.CHECK_EQ(rgExpected[i], rgDataC[i], "The values at " + i.ToString() + " are not as expected.");
            }
        }

        public void TestMatrix()
        {
            double[] rgDataA = new double[] { 3.2, 4.5, 8.3, -1.2, 0.03, 0.22, 0.2, 4.9, 8.1, 9.3, 0.1, 0.88 };
            double[] rgDataB;
                
            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 1, 2, rgDataA.Length/2 });
            m_B = new Blob<T>(m_cuda, m_log);
            m_B.ReshapeLike(m_A);
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.Reshape(1, 1, 1, m_A.shape(3));
            m_C.SetData(1);

            // Sum all columns and test the results.
            m_A.mutable_cpu_data = convert(rgDataA);
            m_cuda.matrix_aggregate_cols(AGGREGATIONS.SUM, m_A.width, m_A.height, m_A.gpu_data, m_B.mutable_gpu_data);
            rgDataB = convert(m_B.mutable_cpu_data);

            double[] rgExpected = new double[rgDataA.Length / 2];
            for (int j = 0; j < m_A.width; j++)
            { 
                for (int i = 0; i < m_A.height; i++)
                {
                    rgExpected[j] += rgDataA[i * m_A.width + j];
                }

                double dfExpected = rgExpected[j];
                double dfActual = rgDataB[j];
                m_log.EXPECT_NEAR(dfExpected, dfActual, 0.00001, "The expected and actual are not the same!");
            }

            // Sum all rows and test the results.
            m_A.mutable_cpu_data = convert(rgDataA);
            m_cuda.matrix_aggregate_rows(AGGREGATIONS.SUM, m_A.width, m_A.height, m_A.gpu_data, m_C.gpu_data, m_B.mutable_gpu_data);
            rgDataB = convert(m_B.mutable_cpu_data);

            rgExpected = new double[2];
            for (int i = 0; i < m_A.height; i++)
            {
                for (int j = 0; j < m_A.width; j++)
                {
                    rgExpected[i] += rgDataA[i * m_A.width + j];
                }

                double dfExpected = rgExpected[i];
                double dfActual = rgDataB[i];
                m_log.EXPECT_NEAR(dfExpected, dfActual, 0.00001, "The expected and actual are not the same!");
            }
        }
    }
}
