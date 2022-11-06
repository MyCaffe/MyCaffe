using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using System.Diagnostics;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.fillers;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestCudaDnn
    {
        [TestMethod]
        public void TestDeviceID()
        {
            CudaDnnTest test = new CudaDnnTest();
            int nDeviceID = -1;
            int nOriginalDeviceID = -1;
            int nDeviceCount = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    nDeviceCount = t.Cuda.GetDeviceCount();
                    nOriginalDeviceID = t.Cuda.GetDeviceID();

                    if (nDeviceCount > 1)
                    {
                        t.Cuda.SetDeviceID(1);
                        nDeviceID = t.Cuda.GetDeviceID();
                        t.Log.CHECK_EQ(1, nDeviceID, "The deviceID should be equal to 1.");
                    }

                    t.Cuda.SetDeviceID(0);
                    nDeviceID = t.Cuda.GetDeviceID();
                    t.Log.CHECK_EQ(0, nDeviceID, "The deviceID should be equal to 0.");

                    t.Cuda.SetDeviceID(nOriginalDeviceID);
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
            PreTest.Init();

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
            PreTest.Init();

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
                if (cuda2 != null)
                    cuda2.Dispose();

                cuda1.Dispose();
            }
        }

        [TestMethod]
        public void TestKernelMemCpyFloat()
        {
            PreTest.Init();

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
                if (cuda2 != null)
                    cuda2.Dispose();

                cuda1.Dispose();
            }
        }

        private void copyMemTest(CudaDnn<double> cuda1, CudaDnn<double> cuda2, int nDevice1, int nDevice2)
        {
            long hHost = 0;
            long hMem1 = 0;
            long hMem2 = 0;

            try
            {
                hHost = cuda1.AllocHostBuffer(4);
                cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
                cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });

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
            finally
            {
                if (hHost != 0)
                    cuda1.FreeHostBuffer(hHost);

                if (hMem2 != 0)
                {
                    cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                    cuda2.FreeMemory(hMem2);
                }

                if (hMem1 != 0)
                {
                    cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                    cuda1.FreeMemory(hMem1);
                }
            }
        }

        private void copyMemTest(CudaDnn<float> cuda1, CudaDnn<float> cuda2, int nDevice1, int nDevice2)
        {
            long hHost = 0;
            long hMem1 = 0;
            long hMem2 = 0;

            try
            {
                hHost = cuda1.AllocHostBuffer(4);
                cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                hMem1 = cuda1.AllocMemory(new List<float>() { 1, 2, 3, 4 });
                cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                hMem2 = cuda2.AllocMemory(new List<float>() { 10, 20, 30, 40 });

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
            finally
            {
                if (hHost != 0)
                    cuda1.FreeHostBuffer(hHost);

                if (hMem2 != 0)
                {
                    cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                    cuda2.FreeMemory(hMem2);
                }

                if (hMem1 != 0)
                {
                    cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                    cuda1.FreeMemory(hMem1);
                }
            }
        }

        [TestMethod]
        public void TestKernelAddDouble()
        {
            PreTest.Init();

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
                if (cuda2 != null)
                    cuda2.Dispose();

                cuda1.Dispose();
            }
        }

        private void addMemTest(CudaDnn<double> cuda1, CudaDnn<double> cuda2, int nDevice1, int nDevice2)
        {
            long hMem1 = 0;
            long hMem2 = 0;
            long hMem3 = 0;

            try
            {
                cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
                cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });
                hMem3 = cuda2.AllocMemory(new List<double>() { 0, 0, 0, 0 });

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
            finally
            {
                if (hMem2 != 0)
                {
                    cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                    cuda2.FreeMemory(hMem2);
                }

                if (hMem3 != 0)
                {
                    cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                    cuda2.FreeMemory(hMem3);
                }

                if (hMem1 != 0)
                {
                    cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                    cuda1.FreeMemory(hMem1);
                }
            }
        }

        [TestMethod]
        public void TestKernelAddFloat()
        {
            PreTest.Init();

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
                if (cuda2 != null)
                    cuda2.Dispose();

                cuda1.Dispose();
            }
        }

        private void addMemTest(CudaDnn<float> cuda1, CudaDnn<float> cuda2, int nDevice1, int nDevice2)
        {
            long hMem1 = 0;
            long hMem2 = 0;
            long hMem3 = 0;

            try
            {
                cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
                cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });
                hMem3 = cuda2.AllocMemory(new List<double>() { 0, 0, 0, 0 });

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
            finally
            {
                if (hMem2 != 0)
                {
                    cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                    cuda2.FreeMemory(hMem2);
                }

                if (hMem3 != 0)
                {
                    cuda2.SetDeviceID(nDevice2, DEVINIT.NONE);
                    cuda2.FreeMemory(hMem3);
                }

                if (hMem1 != 0)
                {
                    cuda1.SetDeviceID(nDevice1, DEVINIT.NONE);
                    cuda1.FreeMemory(hMem1);
                }
            }
        }

        [TestMethod]
        public void TestP2PAccess()
        {
            PreTest.Init();

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
                    long hMem1 = 0;
                    long hMem2 = 0;
                    long hHost = 0;
                    double[] rgData1 = null;
                    double[] rgData2 = null;

                    Trace.WriteLine("testing " + rgDevices[i - 1].ToString() + " -> " + rgDevices[i].ToString());

                    try
                    {
                        cuda1.SetDeviceID(rgDevices[i - 1]);
                        hMem1 = cuda1.AllocMemory(new List<double>() { 1, 2, 3, 4 });
                        hHost = cuda1.AllocHostBuffer(4);

                        cuda2.SetDeviceID(rgDevices[i]);
                        hMem2 = cuda2.AllocMemory(new List<double>() { 10, 20, 30, 40 });

                        cuda1.SetDeviceID();
                        cuda1.KernelCopy(4, hMem1, 0, cuda2.KernelHandle, hMem2, 0, hHost);

                        rgData1 = cuda1.GetMemory(hMem1);
                        cuda1.FreeMemory(hMem1);
                        hMem1 = 0;
                        cuda2.FreeHostBuffer(hHost);
                        hHost = 0;

                        cuda2.SetDeviceID();
                        rgData2 = cuda2.GetMemory(hMem2);
                        cuda2.FreeMemory(hMem2);
                        hMem2 = 0;
                    }
                    catch (Exception excpt)
                    {
                        throw excpt;
                    }
                    finally
                    {
                        if (hMem1 != 0)
                        {
                            cuda1.SetDeviceID(rgDevices[i - 1]);
                            cuda1.FreeMemory(hMem1);
                            hMem1 = 0;
                        }

                        if (hMem2 != 0)
                        {
                            cuda2.SetDeviceID(rgDevices[i]);
                            cuda2.FreeMemory(hMem2);
                            hMem2 = 0;
                        }

                        if (hHost != 0)
                        {
                            cuda1.FreeHostBuffer(hHost);
                            hHost = 0;
                        }
                    }

                    Assert.AreEqual(rgData1.Length, rgData2.Length);

                    for (int j = 0; j < rgData1.Length; j++)
                    {
                        Assert.AreEqual(rgData1[j], rgData2[j]);
                        Assert.AreEqual(rgData1[j], j + 1);
                        Assert.AreEqual(rgData2[j], j + 1);
                    }

                    Trace.WriteLine("testing " + rgDevices[i - 1].ToString() + " <- " + rgDevices[i].ToString());

                    try
                    {
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
                    }
                    catch (Exception excpt)
                    {
                        throw excpt;
                    }
                    finally
                    {
                        if (hMem1 != 0)
                        {
                            cuda1.SetDeviceID(rgDevices[i - 1]);
                            cuda1.FreeMemory(hMem1);
                            hMem1 = 0;
                        }

                        if (hMem2 != 0)
                        {
                            cuda2.SetDeviceID(rgDevices[i]);
                            cuda2.FreeMemory(hMem2);
                            hMem2 = 0;
                        }

                        if (hHost != 0)
                        {
                            cuda1.FreeHostBuffer(hHost);
                            hHost = 0;
                        }
                    }

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
                cuda2.Dispose();
                cuda1.Dispose();
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

                    try
                    {
                        double[] rgdfData = t.Cuda.GetHostMemoryDouble(hMem);
                        float[] rgfData = t.Cuda.GetHostMemoryFloat(hMem);
                    }
                    finally
                    {
                        t.Cuda.FreeHostBuffer(hMem);
                    }
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
                    try
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
                    }
                    finally
                    {
                        if (hMemD != 0)
                        {
                            t.Cuda.FreeMemory(hMemD);
                            hMemD = 0;
                        }

                        if (hMemF != 0)
                        {
                            t.Cuda.FreeMemory(hMemF);
                            hMemF = 0;
                        }
                    }
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
                    try
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
                    }
                    finally
                    {
                        if (hMemD != 0)
                        {
                            t.Cuda.FreeMemory(hMemD);
                            hMemD = 0;
                        }

                        if (hMemF != 0)
                        {
                            t.Cuda.FreeMemory(hMemF);
                            hMemF = 0;
                        }
                    }
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
                    try
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
                    }
                    finally
                    {
                        if (hMemD != 0)
                        {
                            t.Cuda.FreeMemory(hMemD);
                            hMemD = 0;
                        }

                        if (hMemF != 0)
                        {
                            t.Cuda.FreeMemory(hMemF);
                            hMemF = 0;
                        }
                    }
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
                    try
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
                    }
                    finally
                    {
                        if (hMemF != 0)
                        {
                            t.Cuda.FreeMemory(hMemF);
                            hMemF = 0;
                        }
                    }
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
                    try
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
                    }
                    finally
                    {
                        if (hMemF != 0)
                        {
                            t.Cuda.FreeMemory(hMemF);
                            hMemF = 0;
                        }
                    }
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
                    try
                    {
                        if (t.DataType == DataType.DOUBLE)
                            continue;

                        long lCount = 20948400;
                        hMemF1 = t.Cuda.AllocMemory(lCount, true);
                        hMemF2 = t.Cuda.AllocMemory(lCount, true);

                        t.Cuda.axpy((int)lCount, -1.0, hMemF1, hMemF2);
                    }
                    finally
                    {
                        if (hMemF1 != 0)
                        {
                            t.Cuda.FreeMemory(hMemF1);
                            hMemF1 = 0;
                        }

                        if (hMemF2 != 0)
                        {
                            t.Cuda.FreeMemory(hMemF2);
                            hMemF2 = 0;
                        }
                    }
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
                        {
                            t.Cuda.FreeMemory(h1);
                            h1 = 0;
                        }

                        if (h2 > 0)
                        {
                            t.Cuda.FreeMemory(h2);
                            h2 = 0;
                        }

                        if (h3 > 0)
                        {
                            t.Cuda.FreeMemory(h3);
                            h3 = 0;
                        }
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
                        {
                            t.Cuda.FreeMemory(h1);
                            h1 = 0;
                        }

                        if (h2 > 0)
                        {
                            t.Cuda.FreeMemory(h2);
                            h2 = 0;
                        }
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
                        {
                            t.Cuda.FreeMemory(h1);
                            h1 = 0;
                        }

                        if (h2 > 0)
                        {
                            t.Cuda.FreeMemory(h2);
                            h2 = 0;
                        }
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
                        {
                            t.Cuda.FreeMemory(hX);
                            hX = 0;
                        }

                        if (hY > 0)
                        {
                            t.Cuda.FreeMemory(hY);
                            hY = 0;
                        }

                        if (hA > 0)
                        {
                            t.Cuda.FreeMemory(hA);
                            hA = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_sub()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Math sub");
            List<double> rgdf = new List<double>();
            long hSrc = 0;
            long hDst = 0;

            try
            {
                for (int i = 0; i < 20; i++)
                {
                    rgdf.Add(i * 0.1);
                }

                int nCount = rgdf.Count();

                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        hSrc = t.Cuda.AllocMemory(rgdf);
                        hDst = t.Cuda.AllocMemory(rgdf.Count);

                        for (int j = 0; j < 4; j++)
                        {
                            t.Cuda.sub(nCount, hSrc, hSrc, hDst, 0, j * 5, 0, 5);

                            List<double> rgDst = new List<double>();
                            for (int i = 0; i < nCount; i++)
                            {
                                int nIdx = j * 5 + i % 5;
                                rgDst.Add(rgdf[i] - rgdf[nIdx]);
                            }

                            double[] rgdf2 = t.Cuda.GetMemoryDouble(hDst);

                            for (int i = 0; i < nCount; i++)
                            {
                                log.EXPECT_EQUAL<float>(rgdf2[i], rgDst[i]);
                            }
                        }

                        t.Cuda.sub(nCount, hSrc, hSrc, hDst);
                        double dfAsum = t.Cuda.asum_double(nCount, hDst);

                        log.EXPECT_EQUAL<float>(dfAsum, 0.0);
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
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
                    try
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
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_mask()
        {
            CudaDnnTest test = new CudaDnnTest();
            float[] rgf = new float[] { 1.1f, 2.2f, 0.0000099f, 999999.888f };
            int nCount = rgf.Length;
            float[] rgfMask = new float[nCount];
            double[] rgfExpected = new double[nCount];

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long hData = t.Cuda.AllocMemory(rgf);
                    long hMask = t.Cuda.AllocMemory(nCount);
                    long hDst = t.Cuda.AllocMemory(nCount);
                    
                    try
                    {
                        double dfReplacement = double.NegativeInfinity;

                        for (int i = 0; i < rgf.Length; i++)
                        {
                            for (int j = 0; j < rgfMask.Length; j++)
                            {
                                rgfMask[j] = 1.0f;
                                rgfExpected[j] = rgf[j];
                            }

                            for (int j = i; j < rgfMask.Length; j++)
                            {
                                rgfMask[j] = 0.0f;
                                rgfExpected[j] = dfReplacement;
                            }

                            t.Cuda.SetMemory(hMask, rgfMask);
                            t.Cuda.set(nCount, hDst, 0);
                            t.Cuda.mask(nCount, nCount, 0.0, dfReplacement, hData, hMask, hDst);

                            double[] rgfRes = t.Cuda.get_double(nCount, hDst);

                            t.Log.CHECK_EQ(rgfRes.Length, nCount, "The data length returned is not correct!");

                            for (int j = 0; j < rgfRes.Length; j++)
                            {
                                double dfExpected = rgfExpected[j];
                                double dfActual = rgfRes[j];

                                t.Log.EXPECT_NEAR(dfExpected, dfActual, 0.000001, "The expected and actual are not as expected!");
                            }
                        }
                    }
                    finally
                    {
                        if (hData != 0)
                        {
                            t.Cuda.FreeMemory(hData);
                            hData = 0;
                        }
                        
                        if (hMask != 0)
                        {
                            t.Cuda.FreeMemory(hMask);
                            hMask = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_mask2()
        {
            // Test masking over two sets of data (e.g., 2 * mask len)
            CudaDnnTest test = new CudaDnnTest();
            float[] rgf = new float[] { 1.1f, 2.2f, 0.0000099f, 999999.888f, 8.5f, 10.2f, -3.2f, 1.999f };
            int nCount = rgf.Length;
            int nMaskCount = nCount / 2;
            float[] rgfMask = new float[nMaskCount];
            double[] rgfExpected = new double[nCount];

            try
            {
                foreach (ITest t in test.Tests)
                {
                    long hData = t.Cuda.AllocMemory(rgf);
                    long hMask = t.Cuda.AllocMemory(nMaskCount);
                    long hDst = t.Cuda.AllocMemory(nCount);

                    try
                    {
                        double dfReplacement = double.NegativeInfinity;

                        for (int i = 0; i < rgf.Length; i++)
                        {
                            for (int j = 0; j < rgfExpected.Length; j++)
                            {
                                if (j < rgfMask.Length)
                                    rgfMask[j] = 1.0f;
                                rgfExpected[j] = rgf[j];
                            }

                            for (int j = i; j < rgfExpected.Length; j++)
                            {
                                if (j < rgfMask.Length)
                                    rgfMask[j] = 0.0f;

                                if (rgfMask[j % nMaskCount] == 0.0)
                                    rgfExpected[j] = dfReplacement;
                            }
                            
                            t.Cuda.SetMemory(hMask, rgfMask);
                            t.Cuda.set(nCount, hDst, 0);
                            t.Cuda.mask(nCount, nMaskCount, 0.0, dfReplacement, hData, hMask, hDst);

                            double[] rgfRes = t.Cuda.get_double(nCount, hDst);

                            t.Log.CHECK_EQ(rgfRes.Length, nCount, "The data length returned is not correct!");

                            for (int j = 0; j < rgfRes.Length; j++)
                            {
                                double dfExpected = rgfExpected[j];
                                double dfActual = rgfRes[j];

                                t.Log.EXPECT_NEAR(dfExpected, dfActual, 0.000001, "The expected and actual are not as expected!");
                            }
                        }
                    }
                    finally
                    {
                        if (hData != 0)
                        {
                            t.Cuda.FreeMemory(hData);
                            hData = 0;
                        }

                        if (hMask != 0)
                        {
                            t.Cuda.FreeMemory(hMask);
                            hMask = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
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
                    try
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
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
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
                    try
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
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
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
                    try
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
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_copy_sim()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Copy");
            long hSrc1 = 0;
            long hSrc2 = 0;
            long hDst = 0;
            long hSim = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 3;
                        int nCount = nNum * nDim;

                        List<double> rgdfSrc1 = new List<double>() { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 };
                        List<double> rgdfSrc2 = new List<double>() { 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1 };
                        List<double> rgdfSim = new List<double>() { 1, 0, 1, 0 };

                        hSrc1 = t.Cuda.AllocMemory(rgdfSrc1);
                        hSrc2 = t.Cuda.AllocMemory(rgdfSrc2);
                        hSim = t.Cuda.AllocMemory(rgdfSim);
                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.copy(nCount, nNum, nDim, hSrc1, hSrc2, hDst, hSim);
                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            int nIdx = i / nDim;
                            bool bSimilar = (rgdfSim[nIdx] == 0) ? false : true;

                            if (bSimilar)
                                log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc1[i], "The values of src1 and dst are not the same at index = " + i.ToString());
                            else
                                log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc2[i], "The values of src2 and dst are not the same at index = " + i.ToString());
                        }

                        t.Cuda.FreeMemory(hSim);
                        rgdfSim = new List<double>() { 0, 1, 0, 1 };
                        hSim = t.Cuda.AllocMemory(rgdfSim);

                        t.Cuda.copy(nCount, nNum, nDim, hSrc1, hSrc2, hDst, hSim);
                        rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            int nIdx = i / nDim;
                            bool bSimilar = (rgdfSim[nIdx] == 0) ? false : true;

                            if (bSimilar)
                                log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc1[i], "The values of src1 and dst are not the same at index = " + i.ToString());
                            else
                                log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc2[i], "The values of src2 and dst are not the same at index = " + i.ToString());
                        }

                        t.Cuda.FreeMemory(hSim);
                        rgdfSim = new List<double>() { 1, 1, 1, 1 };
                        hSim = t.Cuda.AllocMemory(rgdfSim);

                        t.Cuda.copy(nCount, nNum, nDim, hSrc1, hSrc2, hDst, hSim);
                        rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc1[i], "The values of src1 and dst are not the same at index = " + i.ToString());
                        }

                        t.Cuda.FreeMemory(hSim);
                        rgdfSim = new List<double>() { 0, 0, 0, 0 };
                        hSim = t.Cuda.AllocMemory(rgdfSim);

                        t.Cuda.copy(nCount, nNum, nDim, hSrc1, hSrc2, hDst, hSim);
                        rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc2[i], "The values of src1 and dst are not the same at index = " + i.ToString());
                        }
                    }
                    finally
                    {
                        if (hSrc1 != 0)
                        {
                            t.Cuda.FreeMemory(hSrc1);
                            hSrc1 = 0;
                        }

                        if (hSrc2 != 0)
                        {
                            t.Cuda.FreeMemory(hSrc2);
                            hSrc2 = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }

                        if (hSim != 0)
                        {
                            t.Cuda.FreeMemory(hSim);
                            hSim = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_copy_fill()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Copy Fill");
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 3;
                        int nCount = nNum * nDim;

                        List<double> rgdfSrc = new List<double>() { 0.0, 0.1, 0.2, 0.3, 0.5 };

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);
                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.fill(nNum, nDim, hSrc, 0, nCount, hDst);

                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            int nIdx = i % nDim;
                            log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc[nIdx], "The values of src and dst are not the same at index = " + i.ToString());
                        }

                        t.Cuda.set(nCount, hDst, 0);
                        t.Cuda.fill(nNum, nDim, hSrc, 1, nCount, hDst);

                        rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            int nIdx = i % nDim;
                            log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc[nIdx + 1], "The values of src and dst are not the same at index = " + i.ToString());
                        }

                        t.Cuda.set(nCount, hDst, 0);
                        t.Cuda.fill(nNum, nDim, hSrc, 2, nCount, hDst);

                        rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            int nIdx = i % nDim;
                            log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc[nIdx + 2], "The values of src and dst are not the same at index = " + i.ToString());
                        }
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_copy_expand()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Copy Expand");
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 3;
                        int nCount = nNum * nDim;

                        List<double> rgdfSrc = new List<double>() { 0.1, 0.2, 0.3, 0.4 };

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);

                        double[] rgSrc = t.Cuda.GetMemoryDouble(hSrc);

                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.copy_expand(nCount, nNum, nDim, hSrc, hDst);

                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            int nIdx = i / nDim;
                            log.EXPECT_EQUAL<float>(rgDst[i], rgdfSrc[nIdx], "The values of src and dst are not the same at index = " + i.ToString());
                        }
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_channel_compare()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel Compare");
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 2;
                        int nCount = nNum * nDim;

                        List<double> rgdfSrc = new List<double>() { 0, 0, 1, 0, 2, 3, 3, 3 };

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);
                        hDst = t.Cuda.AllocMemory(nNum);

                        t.Cuda.channel_compare(nCount, nNum, nDim, 1, hSrc, hDst);
                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        log.CHECK_EQ(rgDst.Length, nNum, "The destination length is wrong.");

                        for (int i = 0; i < nNum; i++)
                        {
                            int nIdx = i * nDim;
                            double df1 = rgdfSrc[nIdx + 0];
                            double df2 = rgdfSrc[nIdx + 1];
                            int nCompare = (df1 == df2) ? 1 : 0;

                            log.CHECK_EQ(nCompare, rgDst[i], "The values do not match!");
                        }
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_channel_fill()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel Fill");
            long hSrc = 0;
            long hLabels = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 3;
                        int nCount = nNum * nDim;

                        // Source contains 3 element encodings, one per label 0, 1, and 2 listed in order of the labels.
                        List<double> rgdfSrc = new List<double>() { 0.1, 0.11, 0.12, 0.2, 0.21, 0.22, 0.3, 0.31, 0.32 };
                        // Label contains labels where one or more label are grouped by channel, but only the first label per channel is used.
                        List<double> rgdfLabel = new List<double>() { 1, 5, 2, 5, 0, 5, 1, 5 };
                        // Destination is expected to have label encodings corresponding to the labels of Label placed in the same order as Label.
                        List<double> rgdfDst = new List<double>() { 0.2, 0.21, 0.22, 0.3, 0.31, 0.32, 0.1, 0.11, 0.12, 0.2, 0.21, 0.22 };

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);
                        hLabels = t.Cuda.AllocMemory(rgdfLabel);
                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.channel_fill(nCount, nNum, nDim, 1, hSrc, 2, hLabels, hDst);
                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            double df1 = rgdfDst[i];
                            double df2 = rgDst[i];

                            log.EXPECT_EQUAL<float>(df1, df2, "The values do not match!");
                        }

                        rgdfLabel = new List<double>() { 1, 2, 0, 1 };

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);
                        hLabels = t.Cuda.AllocMemory(rgdfLabel);
                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.channel_fill(nCount, nNum, nDim, 1, hSrc, 1, hLabels, hDst);
                        rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            double df1 = rgdfDst[i];
                            double df2 = rgDst[i];

                            log.EXPECT_EQUAL<float>(df1, df2, "The values do not match!");
                        }

                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }

                        if (hLabels != 0)
                        {
                            t.Cuda.FreeMemory(hLabels);
                            hLabels = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_channel_fillfrom()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel FillFrom");
            long hSrc = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nN = 2;
                        int nC = 3;
                        int nSpatial = 5;
                        int nCount = nN * nC * nSpatial;

                        List<double> rgdfSrc = new List<double>() 
                        { 
                            1.0, 2.0, 3.0, 
                            4.0, 5.0, 6.0 
                        };
                        List<double> rgdfExpected = new List<double>()
                        {
                            1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0
                        };

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);
                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.channel_fillfrom(nCount, nN, nC, nSpatial, hSrc, hDst);

                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i=0; i<nCount; i++)
                        {
                            double dfExpected = rgdfExpected[i];
                            double dfActual = rgDst[i];
                            double dfErr = 0.00000001;

                            log.EXPECT_NEAR(dfExpected, dfActual, dfErr, "The values do not match!");
                        }
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_channel_copy()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel Copy");
            long hDataA = 0;
            long hDataB = 0;
            long hDataC = 0;
            long hDataAll = 0;


            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 2;
                        int nChannels = 2;
                        int nEmbed = 3;
                        int nBlocks = 3;
                        int nCount = nNum * nChannels * nEmbed;
                        //                                              |-n0-------------------------------------|-n1--------------------------------------|
                        //                                              |-c0-----------------|-c1----------------|-c0-----------------|-c1-----------------|
                        List<double> rgdfExpectedA = new List<double>() { 00.01, 00.02, 00.03, 01.01, 01.02, 01.03, 10.01, 10.02, 10.03, 11.01, 11.02, 11.03 };
                        List<double> rgdfExpectedB = new List<double>() { 00.11, 00.12, 00.13, 01.11, 01.12, 01.13, 10.11, 10.12, 10.13, 11.11, 11.12, 11.13 };
                        List<double> rgdfExpectedC = new List<double>() { 00.21, 00.22, 00.23, 01.21, 01.22, 01.23, 10.21, 10.22, 10.23, 11.21, 11.22, 11.23 };
                        List<double> rgdfAll = new List<double>() { 
                        // |-blk0--------------|-blk1---------------|-blk2---------------|
                            00.01, 00.02, 00.03, 00.11, 00.12, 00.13, 00.21, 00.22, 00.23, // n0, c0
                            01.01, 01.02, 01.03, 01.11, 01.12, 01.13, 01.21, 01.22, 01.23, // n0, c1

                            10.01, 10.02, 10.03, 10.11, 10.12, 10.13, 10.21, 10.22, 10.23, // n1, c0
                            11.01, 11.02, 11.03, 11.11, 11.12, 11.13, 11.21, 11.22, 11.23, // n1, c1
                        };

                        hDataA = t.Cuda.AllocMemory(rgdfExpectedA.Count);
                        hDataB = t.Cuda.AllocMemory(rgdfExpectedB.Count);
                        hDataC = t.Cuda.AllocMemory(rgdfExpectedC.Count);
                        hDataAll = t.Cuda.AllocMemory(rgdfAll);

                        // Test FWD copy from X(3) -> Ya, Yb, Yc
                        t.Cuda.channel_copy(nCount, nNum, nChannels, nBlocks, nEmbed, 0, hDataAll, hDataA, DIR.FWD);
                        t.Cuda.channel_copy(nCount, nNum, nChannels, nBlocks, nEmbed, 1, hDataAll, hDataB, DIR.FWD);
                        t.Cuda.channel_copy(nCount, nNum, nChannels, nBlocks, nEmbed, 2, hDataAll, hDataC, DIR.FWD);

                        double[] rgDataA = t.Cuda.GetMemoryDouble(hDataA);
                        double[] rgDataB = t.Cuda.GetMemoryDouble(hDataB);
                        double[] rgDataC = t.Cuda.GetMemoryDouble(hDataC);

                        verifyData(log, rgdfExpectedA, rgDataA);
                        verifyData(log, rgdfExpectedB, rgDataB);
                        verifyData(log, rgdfExpectedC, rgDataC);

                        // Test BWD copy from Ya, Yb, Yc -> X(3)
                        t.Cuda.FreeMemory(hDataAll);
                        hDataAll = t.Cuda.AllocMemory(rgdfAll.Count);
                        double[] rgDataAll1 = t.Cuda.GetMemoryDouble(hDataAll);

                        t.Cuda.channel_copy(nCount, nNum, nChannels, nBlocks, nEmbed, 0, hDataAll, hDataA, DIR.BWD);
                        t.Cuda.channel_copy(nCount, nNum, nChannels, nBlocks, nEmbed, 1, hDataAll, hDataB, DIR.BWD);
                        t.Cuda.channel_copy(nCount, nNum, nChannels, nBlocks, nEmbed, 2, hDataAll, hDataC, DIR.BWD);

                        double[] rgDataAll = t.Cuda.GetMemoryDouble(hDataAll);

                        verifyData(log, rgdfAll, rgDataAll);
                    }
                    finally
                    {
                        if (hDataA != 0)
                        {
                            t.Cuda.FreeMemory(hDataA);
                            hDataA = 0;
                        }

                        if (hDataB != 0)
                        {
                            t.Cuda.FreeMemory(hDataB);
                            hDataB = 0;
                        }

                        if (hDataC != 0)
                        {
                            t.Cuda.FreeMemory(hDataC);
                            hDataC = 0;
                        }

                        if (hDataAll != 0)
                        {
                            t.Cuda.FreeMemory(hDataAll);
                            hDataAll = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        private bool verifyData(Log log, List<double> rgdf, double[] rgData)
        {
            log.CHECK_EQ(rgdf.Count, rgData.Length, "The length of Data does not match the count of rgdf!");

            for (int i = 0; i < rgdf.Count; i++)
            {
                double dfA1 = rgdf[i];
                double dfA2 = rgData[i];

                log.EXPECT_NEAR(dfA1, dfA2, 0.000001);
            }

            return true;
        }

        [TestMethod]
        public void TestMath_channel_scale1()
        {
            int nT = 3;
            int nB = 2;
            int nI = 3;

            // Input contains contains 3 steps, 2 batches with data size = 3 per item.
            double[] rgdfInput = new double[] { 0.01, 0.02, 0.03,  // t0
                                                0.11, 0.12, 0.13,
                                                1.01, 1.02, 1.03,  // t1
                                                1.11, 1.12, 1.13,
                                                2.01, 2.02, 2.03,  // t2
                                                2.11, 2.12, 2.13 };
            // Clip contains values 1 for active, 0 for inactive where one value is set per nT * nB.
            double[] rgdfClip = new double[] { 1, 1,   // t0
                                               1, 0,   // t1
                                               0, 0 }; // t2
                                                                             // Destination is expected to have label encodings corresponding to the labels of Label placed in the same order as Label.
            double[] rgdfOutput = new double[] { 0.01, 0.02, 0.03,  // t0   
                                                 0.11, 0.12, 0.13,
                                                 1.01, 1.02, 1.03,  // t1
                                                 0.00, 0.00, 0.00,
                                                 0.00, 0.00, 0.00,  // t2
                                                 0.00, 0.00, 0.00 };

            TestMath_channel_scale(nT, nB, nI, rgdfInput, rgdfClip, rgdfOutput);
        }

        [TestMethod]
        public void TestMath_channel_scale2()
        {
            int nT = 3;
            int nB = 2;
            int nI = 3;

            // Input contains contains 3 steps, 2 batches with data size = 3 per item.
            double[] rgdfInput = new double[] { 0.01, 0.02, 0.03,  // t0
                                                0.11, 0.12, 0.13,
                                                1.01, 1.02, 1.03,  // t1
                                                1.11, 1.12, 1.13,
                                                2.01, 2.02, 2.03,  // t2
                                                2.11, 2.12, 2.13 };
            // Clip contains values 1 for active, 0 for inactive where one value is set per nT * nB.
            double[] rgdfClip = new double[] { 1, 0,   // t0
                                               0, 1,   // t1
                                               1, 1 }; // t2
                                                       // Destination is expected to have label encodings corresponding to the labels of Label placed in the same order as Label.
            double[] rgdfOutput = new double[] { 0.01, 0.02, 0.03,  // t0   
                                                 0.00, 0.00, 0.00,
                                                 0.00, 0.00, 0.00,  // t1
                                                 1.11, 1.12, 1.13,
                                                 2.01, 2.02, 2.03,  // t2
                                                 2.11, 2.12, 2.13 };

            TestMath_channel_scale(nT, nB, nI, rgdfInput, rgdfClip, rgdfOutput);
        }

        [TestMethod]
        public void TestMath_channel_scale2b()
        {
            int nT = 3;
            int nB = 2;
            int nI = 3;

            // Input contains contains 3 steps, 2 batches with data size = 3 per item.
                                                // t0                t1                   t2
            double[] rgdfInput = new double[] { 0.01, 0.02, 0.03,    1.01, 1.02, 1.03,    2.01, 2.02, 2.03,   // b0
                                                0.11, 0.12, 0.13,    1.11, 1.12, 1.13,    2.11, 2.12, 2.13 }; // b1
            // Clip contains values 1 for active, 0 for inactive where one value is set per nT * nB.
                                             // t0                t1                   t2
            double[] rgdfClip = new double[] {  1,                   0,                   1,   // b0
                                                1,                   1,                   0 }; // b1
            // Destination is expected to have label encodings corresponding to the labels of Label placed in the same order as Label.
                                              // t0                t1                   t2
            double[] rgdfOutput = new double[] { 0.01, 0.02, 0.03,    0.00, 0.00, 0.00,    2.01, 2.02, 2.03,   // b0
                                                 0.11, 0.12, 0.13,    1.11, 1.12, 1.13,    0.00, 0.00, 0.00 }; // b1

            rgdfInput = SimpleDatum.Transpose(rgdfInput, nB, nT, nI);
            rgdfClip = SimpleDatum.Transpose(rgdfClip, nB, nT, 1);
            rgdfOutput = SimpleDatum.Transpose(rgdfOutput, nB, nT, nI);

            TestMath_channel_scale(nT, nB, nI, rgdfInput, rgdfClip, rgdfOutput);
        }

        [TestMethod]
        public void TestMath_channel_scale3()
        {
            int nT = 3;
            int nB = 2;
            int nI = 3;

            // Input contains contains 3 steps, 2 batches with data size = 3 per item.
            double[] rgdfInput = new double[] { 0.01, 0.02, 0.03,  // t0
                                                0.11, 0.12, 0.13,
                                                1.01, 1.02, 1.03,  // t1
                                                1.11, 1.12, 1.13,
                                                2.01, 2.02, 2.03,  // t2
                                                2.11, 2.12, 2.13 };
            // Clip contains values 1 for active, 0 for inactive where one value is set per nT * nB.
            double[] rgdfClip = new double[] { 1, 0,   // t0
                                               0, 1,   // t1
                                               1, 1 }; // t2
                                                       // Destination is expected to have label encodings corresponding to the labels of Label placed in the same order as Label.
            double[] rgdfOutput = new double[] { 0.01, 0.02, 0.03,  // t0   
                                                 0.00, 0.00, 0.00,
                                                 0.00, 0.00, 0.00,  // t1
                                                 1.11, 1.12, 1.13,
                                                 2.01, 2.02, 2.03,  // t2
                                                 2.11, 2.12, 2.13 };

            TestMath_channel_scale(nT, nB, nI, rgdfInput, rgdfClip, rgdfOutput);
        }

        [TestMethod]
        public void TestMath_channel_scale3b()
        {
            int nT = 3;
            int nB = 2;
            int nI = 3;

            // Input contains contains 3 steps, 2 batches with data size = 3 per item.
                                             // t0                   t1                   t2
            double[] rgdfInput = new double[] { 0.01, 0.02, 0.03,    1.01, 1.02, 1.03,    2.01, 2.02, 2.03,   // b0
                                                0.11, 0.12, 0.13,    1.11, 1.12, 1.13,    2.11, 2.12, 2.13 }; // b1
            // Clip contains values 1 for active, 0 for inactive where one value is set per nT * nB.
                                             // t0                   t1                   t2
            double[] rgdfClip = new double[] {  1,                   1,                   0,   // b0
                                                1,                   1,                   0 }; // b1
            // Destination is expected to have label encodings corresponding to the labels of Label placed in the same order as Label.
                                              // t0                t1                   t2
            double[] rgdfOutput = new double[] { 0.01, 0.02, 0.03,    1.01, 1.02, 1.03,    0.00, 0.00, 0.00,   // b0
                                                 0.11, 0.12, 0.13,    1.11, 1.12, 1.13,    0.00, 0.00, 0.00 }; // b1

            rgdfInput = SimpleDatum.Transpose(rgdfInput, nB, nT, nI);
            rgdfClip = SimpleDatum.Transpose(rgdfClip, nB, nT, 1);
            rgdfOutput = SimpleDatum.Transpose(rgdfOutput, nB, nT, nI);

            TestMath_channel_scale(nT, nB, nI, rgdfInput, rgdfClip, rgdfOutput);
        }

        public void TestMath_channel_scale(int nT, int nB, int nI, double[] rgdfSrc, double[] rgdfClip, double[] rgdfExpected)
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel Scale");
            long hSrc = 0;
            long hClip = 0;
            long hDst = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nCount = nT * nB * nI;

                        hSrc = t.Cuda.AllocMemory(rgdfSrc);
                        hClip = t.Cuda.AllocMemory(rgdfClip);
                        hDst = t.Cuda.AllocMemory(nCount);

                        t.Cuda.channel_scale(nCount, nT, nB, nI, hSrc, hClip, hDst);
                        double[] rgDst = t.Cuda.GetMemoryDouble(hDst);

                        for (int i = 0; i < nCount; i++)
                        {
                            double df1 = rgdfExpected[i];
                            double df2 = rgDst[i];

                            log.EXPECT_EQUAL<float>(df1, df2, "The values do not match!");
                        }
                    }
                    finally
                    {
                        if (hSrc != 0)
                        {
                            t.Cuda.FreeMemory(hSrc);
                            hSrc = 0;
                        }

                        if (hDst != 0)
                        {
                            t.Cuda.FreeMemory(hDst);
                            hDst = 0;
                        }

                        if (hClip != 0)
                        {
                            t.Cuda.FreeMemory(hClip);
                            hClip = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_channel_mulv()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel Mulv");
            long hScaleVector = 0;
            long hDataMatrix = 0;
            long hDstMatrix = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 3;
                        int nCount = nNum * nDim;

                        // Scale vector contains 'nDim' items
                        List<double> rgScaleVector = new List<double>() { 0.1, 0.2, 0.3 };
                        // Data matrix is an nNum x nDim matrix.  Dst matrix is the same size.
                        List<double> rgDataMatrix = new List<double>() { 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 4.0, 4.1, 4.2 };
                        hScaleVector = t.Cuda.AllocMemory(rgScaleVector);
                        hDataMatrix = t.Cuda.AllocMemory(rgDataMatrix);
                        hDstMatrix = t.Cuda.AllocMemory(nCount);

                        t.Cuda.channel_mulv(nCount, 1, nNum, nDim, hDataMatrix, hScaleVector, hDstMatrix);
                        double[] rgDst = t.Cuda.GetMemoryDouble(hDstMatrix);

                        double[] rgExpected = new double[nCount];
                        for (int i = 0; i < nNum; i++)
                        {
                            for (int j = 0; j < nDim; j++)
                            {
                                int nIdxData = (i * nDim) + j;
                                rgExpected[nIdxData] = rgDataMatrix[nIdxData] * rgScaleVector[j];
                            }
                        }

                        for (int i = 0; i < nCount; i++)
                        {
                            double dfExpected = rgExpected[i];
                            double dfActual = rgDst[i];

                            log.EXPECT_EQUAL<float>(dfExpected, dfActual, "The values do not match!");
                        }
                    }
                    finally
                    {
                        if (hScaleVector != 0)
                        {
                            t.Cuda.FreeMemory(hScaleVector);
                            hScaleVector = 0;
                        }

                        if (hDataMatrix != 0)
                        {
                            t.Cuda.FreeMemory(hDataMatrix);
                            hDataMatrix = 0;
                        }

                        if (hDstMatrix != 0)
                        {
                            t.Cuda.FreeMemory(hDstMatrix);
                            hDstMatrix = 0;
                        }
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMath_channel_sum()
        {
            CudaDnnTest test = new CudaDnnTest();
            Log log = new Log("Test Channel Sum across channels");
            long hDataMatrix = 0;
            long hDstVector = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    try
                    {
                        int nNum = 4;
                        int nDim = 3;
                        int nCount = nNum * nDim;

                        // Data matrix is an nNum x nDim matrix.  Dst matrix is the same size.
                        List<double> rgDataMatrix = new List<double>() { 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 4.0, 4.1, 4.2 };
                        hDataMatrix = t.Cuda.AllocMemory(rgDataMatrix);
                        hDstVector = t.Cuda.AllocMemory(nNum);

                        t.Cuda.channel_sum(nCount, nNum, nDim, 1, hDataMatrix, hDstVector);

                        double[] rgDst = t.Cuda.GetMemoryDouble(hDstVector);

                        double[] rgExpected = new double[nNum];
                        for (int i = 0; i < nDim; i++)
                        {
                            for (int j = 0; j < nNum; j++)
                            {
                                int nIdxData = j * nDim + i;
                                rgExpected[j] += rgDataMatrix[nIdxData];
                            }
                        }

                        for (int i = 0; i < nNum; i++)
                        {
                            double dfExpected = rgExpected[i];
                            double dfActual = rgDst[i];

                            log.EXPECT_EQUAL<float>(dfExpected, dfActual, "The values do not match!");
                        }
                    }
                    finally
                    {
                        if (hDataMatrix != 0)
                        {
                            t.Cuda.FreeMemory(hDataMatrix);
                            hDataMatrix = 0;
                        }

                        if (hDstVector != 0)
                        {
                            t.Cuda.FreeMemory(hDstVector);
                            hDstVector = 0;
                        }
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

                    try
                    {
                        t.Cuda.SetTensorDesc(hTensor, 10, 20, 30, 40);
                        t.Cuda.SetTensorDesc(hTensor, 20, 30, 40, 50, 2, 3, 4, 5);
                    }
                    finally
                    {
                        t.Cuda.FreeTensorDesc(hTensor);
                    }
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

                    try
                    {
                        t.Cuda.SetFilterDesc(hFilter, 10, 20, 30, 40);
                    }
                    finally
                    {
                        t.Cuda.FreeFilterDesc(hFilter);
                    }
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
            TestConvolutionDesc(false);
        }

        [TestMethod]
        public void TestConvolutionDescWithTensors()
        {
            TestConvolutionDesc(true);
        }

        private void TestConvolutionDesc(bool bUseTensorCores)
        {
            CudaDnnTest test = new CudaDnnTest();
            long hConv = 0;

            try
            {
                foreach (ITest t in test.Tests)
                {
                    hConv = t.Cuda.CreateConvolutionDesc();
                    t.Log.CHECK_NE(0, hConv, "The convolution handle is null!");

                    try
                    {
                        t.Cuda.SetConvolutionDesc(hConv, 10, 10, 2, 3, 1, 1, bUseTensorCores);
                    }
                    finally
                    {
                        t.Cuda.FreeConvolutionDesc(hConv);
                    }
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

                    try
                    {
                        t.Cuda.SetPoolingDesc(hPool, PoolingMethod.AVE, 10, 20, 2, 3, 4, 5);
                        t.Cuda.SetPoolingDesc(hPool, PoolingMethod.MAX, 10, 20, 2, 3, 4, 5);
                    }
                    finally
                    {
                        t.Cuda.FreePoolingDesc(hPool);
                    }
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

                    try
                    {
                        t.Cuda.SetLRNDesc(hLRN, 2, 1.2, 2.1, 0.2);
                    }
                    finally
                    {
                        t.Cuda.FreeLRNDesc(hLRN);
                    }
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

                    try
                    {
                        t.Cuda.SetRnnDataDesc(hDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nMaxSeqLen, nBatchLen, nVectorLen);
                    }
                    finally
                    {
                        t.Cuda.FreeRnnDataDesc(hDesc);
                    }
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
            TestRnnDesc(false);
        }

        [TestMethod]
        public void TestRnnDescWithTensorCores()
        {
            TestRnnDesc(true);
        }

        private void TestRnnDesc(bool bUseTensorCores)
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

                    try
                    {
                        t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM, bUseTensorCores);
                    }
                    finally
                    {
                        t.Cuda.FreeRnnDesc(hDesc);
                        t.Cuda.FreeCuDNN(hCudnn);
                    }
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
            TestRnnGetParamCount(false);
        }

        [TestMethod]
        public void TestRnnGetParamCountWithTensorCores()
        {
            TestRnnGetParamCount(true);
        }

        private void TestRnnGetParamCount(bool bUseTensorCores)
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

                    try
                    {
                        t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM, bUseTensorCores);
                        t.Cuda.SetRnnDataDesc(hDataDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                        int nCount = t.Cuda.GetRnnParamCount(hCudnn, hDesc, hDataDesc);
                    }
                    finally
                    {
                        t.Cuda.FreeRnnDesc(hDesc);
                        t.Cuda.FreeRnnDataDesc(hDataDesc);
                        t.Cuda.FreeCuDNN(hCudnn);
                    }
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
            TestRnnGetWorkspaceCount(false);
        }

        [TestMethod]
        public void TestRnnGetWorkspaceCountWithTensorCores()
        {
            TestRnnGetWorkspaceCount(true);
        }

        private void TestRnnGetWorkspaceCount(bool bUseTensorCores)
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

                    try
                    {
                        t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM, bUseTensorCores);
                        t.Cuda.SetRnnDataDesc(hDataDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                        ulong nReservedCount;
                        ulong nWorkspaceCount = t.Cuda.GetRnnWorkspaceCount(hCudnn, hDesc, hDataDesc, out nReservedCount);
                    }
                    finally
                    {
                        t.Cuda.FreeRnnDesc(hDesc);
                        t.Cuda.FreeRnnDataDesc(hDataDesc);
                        t.Cuda.FreeCuDNN(hCudnn);
                    }
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
            TestRnnGetLinLayerParams(false);
        }

        [TestMethod]
        public void TestRnnGetLinLayerParamsWithTensorCores()
        {
            TestRnnGetLinLayerParams(true);
        }

        private void TestRnnGetLinLayerParams(bool bUseTensorCores)
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
                    long hWt = 0;
                    long hBias = 0;

                    try
                    {
                        t.Cuda.SetRnnDesc(hCudnn, hDesc, nHiddenSize, nNumLayers, 0, RNN_MODE.LSTM, bUseTensorCores);
                        t.Cuda.SetRnnDataDesc(hDataDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, nSeqLen, nBatchSize, nHiddenSize);

                        int nAllWtCount = t.Cuda.GetRnnParamCount(hCudnn, hDesc, hDataDesc);
                        hWtDesc = t.Cuda.CreateFilterDesc();
                        hWtData = t.Cuda.AllocMemory(nAllWtCount);

                        int[] rgDimWt = new int[3];
                        rgDimWt[0] = nAllWtCount;
                        rgDimWt[1] = 1;
                        rgDimWt[2] = 1;

                        t.Cuda.SetFilterNdDesc(hWtDesc, rgDimWt);

                        int nLinLayers = 8; // LSTM
                        for (int i = 0; i < nNumLayers; i++)
                        {
                            for (int j = 0; j < nLinLayers; j++)
                            {
                                int nWtCount;
                                int nBiasCount;

                                t.Cuda.GetRnnLinLayerParams(hCudnn, hDesc, i, hDataDesc, hWtDesc, hWtData, j, out nWtCount, out hWt, out nBiasCount, out hBias);

                                Assert.AreNotEqual(nWtCount, 0, "The weight data count should not be zero!");
                                Assert.AreNotEqual(hWt, 0, "The weight handle should not be zero!");
                                Assert.AreNotEqual(nBiasCount, 0, "The bias data count should not be zero!");
                                Assert.AreNotEqual(hBias, 0, "The bias handle should not be zero!");

                                t.Cuda.FreeMemoryPointer(hWt);
                                hWt = 0;

                                t.Cuda.FreeMemoryPointer(hBias);
                                hBias = 0;
                            }
                        }
                    }
                    finally
                    {
                        if (hWt != 0)
                            t.Cuda.FreeMemoryPointer(hWt);

                        if (hBias != 0)
                            t.Cuda.FreeMemoryPointer(hBias);

                        t.Cuda.FreeMemory(hWtData);
                        t.Cuda.FreeFilterDesc(hWtDesc);
                        t.Cuda.FreeRnnDesc(hDesc);
                        t.Cuda.FreeRnnDataDesc(hDataDesc);
                        t.Cuda.FreeCuDNN(hCudnn);
                    }
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
        public void TestGeam()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestGeam();
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
        public void TestSum2()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestSum2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInterp2()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestInterp2();
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

        [TestMethod]
        public void TestMatrixMeanStdev()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMatrixMeanStdev();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatrixCorrelation1()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMatrixCorrelation1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatrixCorrelation2()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMatrixCorrelation2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyBatch()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopyBatch();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopySequenceK0()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopySequenceK0();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestCopySequenceK1()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopySequenceK1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopySequenceK1Combine()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopySequenceK1Combine();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopySequenceK2()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopySequenceK2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopySequence2()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopySequence2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopySequence2PortionSpatialDomain()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestCopySequence2PortionSpatialDomain();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLoss_mae_loss_bwd()
        {
            CudaDnnTest test = new CudaDnnTest();

            try
            {
                foreach (ITestCudaDnn t in test.Tests)
                {
                    t.TestMAELossBwd();
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
        void TestGeam();
        void TestGemv();
        void TestSum();
        void TestSum2();
        void TestInterp2();
        void TestMemoryTestByBlock();
        void TestMemoryTestAll();
        void TestMemoryPointers();
        void TestHammingDistance();
        void TestMatrix();
        void TestMatrixMeanStdev();
        void TestMatrixCorrelation1();
        void TestMatrixCorrelation2();
        void TestCopyBatch();
        void TestCopySequenceK0();
        void TestCopySequenceK1();
        void TestCopySequenceK1Combine();
        void TestCopySequenceK2();
        void TestCopySequence2();
        void TestCopySequence2PortionSpatialDomain();
        void TestMAELossBwd();
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
        int m_nOriginalDeviceID = 0;
        long m_hCursors = 0;
        long m_hWork = 0;

        public CudaDnnTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_nOriginalDeviceID = m_cuda.GetDeviceID();
            m_temp = new Blob<T>(m_cuda, m_log);
            m_A = new Blob<T>(m_cuda, m_log, 1, 1, 2, 3);
            m_B = new Blob<T>(m_cuda, m_log, 1, 1, 3, 4);
            m_C = new Blob<T>(m_cuda, m_log, 1, 1, 2, 4);
            m_x = new Blob<T>(m_cuda, m_log, 1, 1, 1, 3);
            m_y = new Blob<T>(m_cuda, m_log, 1, 1, 1, 2);
        }

        protected override void dispose()
        {
            m_cuda.SetDeviceID(m_nOriginalDeviceID);

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

            if (m_hCursors != 0)
            {
                m_cuda.FreeHostBuffer(m_hCursors);
                m_hCursors = 0;
            }

            if (m_hWork != 0)
            {
                m_cuda.FreeHostBuffer(m_hWork);
                m_hWork = 0;
            }
            
            base.dispose();
        }

        public void TestMAELossBwd()
        {
            m_A.Reshape(3, 2, 10, 1);
            m_B.Reshape(3, 2, 10, 1);

            FillerParameter fp = new FillerParameter("gaussian", 0, 0, 3);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_A);
            filler.Fill(m_B);

            m_cuda.mean_error_loss_bwd(m_A.count(), m_A.gpu_data, m_B.gpu_data, m_A.mutable_gpu_diff, MEAN_ERROR.MAE);

            double[] rgPredicted = convert(m_A.mutable_cpu_data);
            double[] rgTarget = convert(m_B.mutable_cpu_data);
            double[] rgGrad = convert(m_A.mutable_cpu_diff);

            for (int i = 0; i < rgPredicted.Length; i++)
            {
                double dfPredicted = rgPredicted[i];
                double dfTarget = rgTarget[i];
                double dfGrad = rgGrad[i];

                if (dfPredicted > dfTarget)
                    m_log.CHECK_EQ(dfGrad, 1, "The gradient is incorrect at i = " + i.ToString() + "!");
                else if (dfPredicted < dfTarget)
                    m_log.CHECK_EQ(dfGrad, -1, "The gradient is incorrect at i = " + i.ToString() + "!");
                else
                    m_log.CHECK_EQ(dfGrad, 0, "The gradient is incorrect at i = " + i.ToString() + "!");
            }
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

        private void fill(double[] rg, double dfStart, double dfStep)
        {
            for (int i = 0; i < rg.Length; i++)
            {
                rg[i] = dfStart + dfStep * i;
            }
        }

        public void TestGeam()
        {
            int nM = 256;
            int nN = 80;
            int nK = 1;

            double[] rgA = new double[nM * nK];
            double[] rgB = new double[nK * nN];
            double[] rgC = new double[nM * nN];

            fill(rgA, 0.0, 0.01);
            fill(rgB, 1.0, 0.0);

            m_A.Reshape(1, nM, nK, 1);
            m_B.Reshape(1, nK, nN, 1);
            m_C.Reshape(1, nM, nN, 1);

            m_A.mutable_cpu_data = convert(rgA);
            m_B.mutable_cpu_data = convert(rgB);

            // Create nM x nN matrix.
            m_cuda.gemm(true, false, nM, nN, nK, 1.0, m_A.gpu_data, m_B.gpu_data, 0.0, m_C.mutable_gpu_data);

            double[] rgC1 = convert(m_C.mutable_cpu_data);

            m_y.Reshape(1, nN, nM, 1);

            // Transpose C.
            m_cuda.geam(true, false, nM, nN, 1.0, m_C.gpu_data, m_C.gpu_data, 0.0, m_y.mutable_gpu_data);

            double[] rgY1 = convert(m_y.mutable_cpu_data);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim[1] = 2;
            p.transpose_param.dim[2] = 1;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            BlobCollection<T> colBtm = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();
            colBtm.Add(m_C);
            colTop.Add(m_y);

            layer.Setup(colBtm, colTop);

            // Convert to nN x nM
            layer.Forward(colBtm, colTop);

            double[] rgY = convert(m_y.mutable_cpu_data);

            // Verify the results.
            for (int i = 0; i < nN; i++)
            {
                for (int j = 0; j < nM; j++)
                {
                    int nIdx = i * nM + j;

                    double dfExpected = rgA[j];
                    double dfActual = rgY[nIdx];
                    double dfActual1 = rgY1[nIdx];

                    m_log.EXPECT_NEAR(dfExpected, dfActual, 0.00001, "The numbers are not as expected.");
                    m_log.EXPECT_NEAR(dfActual, dfActual1, 0.00001, "The numbers are not as expected.");
                }
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

        public void TestSum2()
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

            m_cuda.sum(m_A.count(), 1, m_A.count(), m_A.gpu_data, m_A.mutable_gpu_diff);
            double[] rgResult = convert(m_A.mutable_cpu_diff);
            double dfExpected = rgData.Sum();

            m_log.EXPECT_NEAR_FLOAT(dfExpected, rgResult[0], 0.002, "The values to not match!");
        }

        public void TestInterp2()
        {
            double[] rgData = new double[]
            {
                1, 2, 3, 4, 5,

                1.1, 2.1, 3.1, 4.1, 5.1,

                1.2, 2.2, 3.2, 4.2, 5.2,
            };

            int nH = 2;
            int nW = 10;
            int nH1 = 1;
            int nW1 = 5;
            int nNum = 1;
            int nChannels = 3;

            m_B.Reshape(nNum, nChannels, nH, nW);
            m_A.Reshape(nNum, nChannels, nH1, nW1);
            m_A.mutable_cpu_data = convert(rgData);

            m_cuda.interp2(nNum * nChannels, m_B.gpu_data, 0, 0, nH, nW, nH, nW, m_A.mutable_gpu_data, 0, 0, nH1, nW1, nH1, nW1, true);
            m_cuda.interp2(nNum * nChannels, m_B.gpu_data, 0, 0, nH1, nW1, nH1, nW1, m_A.mutable_gpu_diff, 0, 0, nH, nW, nH, nW, false);

            double[] rgB = convert(m_B.mutable_cpu_data);
            double[] rgA1 = convert(m_A.mutable_cpu_data);
            double[] rgA2 = convert(m_A.mutable_cpu_diff);
        }

        public void TestMemoryTestByBlock()
        {
            ulong ulTotalBlocks;
            double dfTotalMemAllocated;
            ulong ulMemStartAddr;
            ulong ulBlockSize;
            long hMemTest = m_cuda.CreateMemoryTest(out ulTotalBlocks, out dfTotalMemAllocated, out ulMemStartAddr, out ulBlockSize);

            try
            {
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
            }
            finally
            {
                m_cuda.FreeMemoryTest(hMemTest);
            }
        }

        public void TestMemoryTestAll()
        {
            ulong ulTotalBlocks;
            double dfTotalMemAllocated;
            ulong ulMemStartAddr;
            ulong ulBlockSize;
            long hMemTest = m_cuda.CreateMemoryTest(out ulTotalBlocks, out dfTotalMemAllocated, out ulMemStartAddr, out ulBlockSize);

            try
            {
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
            }
            finally
            {
                m_cuda.FreeMemoryTest(hMemTest);
            }
        }

        public void TestMemoryPointers()
        {
            long hMem = m_cuda.AllocMemory(1000);

            try
            {
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
            finally
            {
                m_cuda.FreeMemory(hMem);
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
            int nWid = rgDataA.Length / 2;
            int nHt = 2;
                
            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 1, nHt, nWid });
            m_B = new Blob<T>(m_cuda, m_log);
            m_B.ReshapeLike(m_A);
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.Reshape(1, 1, 1, m_A.shape(3));
            m_C.SetData(1);

            // Sum all columns and test the results.
            m_A.mutable_cpu_data = convert(rgDataA);
            m_cuda.matrix_aggregate_cols(AGGREGATIONS.SUM, m_A.width, m_A.height, m_A.gpu_data, m_B.mutable_gpu_data);
            rgDataB = convert(m_B.mutable_cpu_data);

            double[] rgExpected = new double[nWid];
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

            rgExpected = new double[nHt];
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

            //---------------------------------------------
            // Calculate the Standard Deviation of each row
            //---------------------------------------------

            // Clip to actual values (remaining values are temporary)
            m_B.Reshape(1, 1, nHt, 1);

            // Calcualte row means.
            m_cuda.mul_scalar(m_B.count(), 1/(double)nWid, m_B.mutable_gpu_data);

            // Subtract row means from each row.
            m_cuda.matrix_add_vector(ORIENTATION.COL, nWid, nHt, -1.0, m_A.gpu_data, m_B.gpu_data, m_A.mutable_gpu_diff);
            m_cuda.copy(m_A.count(), m_A.gpu_diff, m_A.mutable_gpu_data);

            // Calculate square of each item.
            m_cuda.powx(m_A.count(), m_A.gpu_data, 2.0, m_A.mutable_gpu_data);

            // Sum the rows again
            m_B.ReshapeLike(m_A);
            m_cuda.matrix_aggregate_rows(AGGREGATIONS.SUM, nWid, nHt, m_A.gpu_data, m_C.gpu_data, m_B.mutable_gpu_data);
            m_B.Reshape(1, 1, nHt, 1);

            // Divide by n-1
            m_cuda.mul_scalar(m_B.count(), 1 / (double)(nWid - 1), m_B.mutable_gpu_data);

            // Calculate sqrt to get standard deviation of each row.
            m_cuda.sqrt(m_B.count(), m_B.gpu_data, m_B.mutable_gpu_data);


            //---------------------------------------------
            // Verify the Standard Deviation of each row
            //---------------------------------------------

            double[] rgRowMean = calculateRowMeans(nWid, nHt, rgDataA);
            double[] rgRowStdev = calculateRowStdev(nWid, nHt, rgDataA, rgRowMean);

            // Verify the data.
            rgDataB = convert(m_B.mutable_cpu_data);
            for (int i = 0; i < nHt; i++)
            {
                double dfExpected = rgRowStdev[i];
                double dfActual = rgDataB[i];
                m_log.EXPECT_NEAR(dfExpected, dfActual, 0.00001, "The expected and actual are not the same!");
            }
        }

        public void TestMatrixMeanStdev()
        {
            double[] rgDataA = new double[] { 3.2, 4.5, 8.3, -1.2, 0.03, 0.22, 0.2, 4.9, 8.1, 9.3, 0.1, 0.88 };
            double[] rgDataB;
            int nWid = rgDataA.Length / 2;
            int nHt = 2;

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 1, nHt, nWid });
            m_B = new Blob<T>(m_cuda, m_log);
            m_B.ReshapeLike(m_A);
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.Reshape(1, 1, 1, m_A.shape(3));
            m_C.SetData(1);
            m_y = new Blob<T>(m_cuda, m_log);
            m_y.ReshapeLike(m_A);

            //----------------------------------------
            //  Calculate the row means
            //
            //  Results are placed in the first of
            //  the resulting memory (m_B.gpu_data)
            //  and each entry represents the mean
            //  of a row.
            //----------------------------------------
            m_A.mutable_cpu_data = convert(rgDataA);
            m_cuda.matrix_mean_rows(nWid, nHt, m_A.gpu_data, m_C.gpu_data, 1.0, m_B.mutable_gpu_data);
            double[] rgRowMean = calculateRowMeans(nWid, nHt, rgDataA);

            // Verify the row means.
            rgDataB = convert(m_B.mutable_cpu_data);
            for (int i = 0; i < nHt; i++)
            {
                double dfExpected = rgRowMean[i];
                double dfActual = rgDataB[i];
                m_log.EXPECT_NEAR(dfExpected, dfActual, 0.00001, "The expected and actual are not the same!");
            }

            //----------------------------------------
            //  Calculate the stddev
            //
            //  Results are placed in the first of
            //  the resulting memory (m_y.gpu_data)
            //  and each entry represents the stddev
            //  of a row.
            //----------------------------------------
            m_cuda.matrix_stdev_rows(nWid, nHt, m_A.gpu_data, m_C.gpu_data, m_B.gpu_data, m_A.mutable_gpu_diff, m_y.mutable_gpu_data);
            double[] rgRowStdev = calculateRowStdev(nWid, nHt, rgDataA, rgRowMean);

            // Verify the row stdev.
            double[] rgDataY = convert(m_y.mutable_cpu_data);
            for (int i = 0; i < nHt; i++)
            {
                double dfActual = rgDataY[i];
                double dfExpected = rgRowStdev[i];
                m_log.EXPECT_NEAR(dfExpected, dfActual, 0.00001, "The expected and actual are not the same!");
            }
        }

        public void TestMatrixCorrelation1()
        {
            double[] rgDataA = new double[] { 3.2, 4.5, 8.3, -1.2, 0.03, 0.22, 0.2, 4.9, 8.1, 9.3, 0.1, 0.88, 8.1, 3.9, 0.3, 5.0, -2.1, 1.1 };
            int nHt = 3;
            int nWid = rgDataA.Length / nHt;

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 1, nHt, nWid });
            m_B = new Blob<T>(m_cuda, m_log);
            m_B.ReshapeLike(m_A);
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.Reshape(1, 1, 1, m_A.shape(3));
            m_C.SetData(1);
            m_y = new Blob<T>(m_cuda, m_log);
            m_y.ReshapeLike(m_A);

            // Load the input data (3 x 6)
            m_A.mutable_cpu_data = convert(rgDataA);

            // Calculate the mean of each row.
            m_cuda.matrix_mean_rows(nWid, nHt, m_A.gpu_data, m_C.gpu_data, 1.0, m_B.mutable_gpu_data);

            // Calculate the stdev of each row.
            m_cuda.matrix_stdev_rows(nWid, nHt, m_A.gpu_data, m_C.gpu_data, m_B.gpu_data, m_A.mutable_gpu_diff, m_y.mutable_gpu_data);
            m_y.Reshape(1, 1, nHt, 1);

            // Subtract row means from each row.
            m_cuda.matrix_add_vector(ORIENTATION.COL, nWid, nHt, -1.0, m_A.gpu_data, m_B.gpu_data, m_A.mutable_gpu_diff);
            m_x.ReshapeLike(m_A);

            //----------------------------------------
            //  Calculate the correlation between
            //  the first row and all other rows.
            //----------------------------------------

            //          = (x - xmean)(y - ymean)
            // m_x.diff = (m_A(0).data - m_A(0).diff)(m_A(1).data - m_A(1).diff) 
            int nOffset = nWid;
            for (int i = 1; i < nHt; i++)
            {
                m_cuda.mul(nWid, m_A.gpu_diff, m_A.gpu_diff, m_x.mutable_gpu_diff, 0, nOffset, nOffset);
                nOffset += nWid;
            }

            // Sum the rows of x diff and place in x data.
            m_cuda.matrix_aggregate_rows(AGGREGATIONS.SUM, nWid, nHt, m_x.gpu_diff, m_C.gpu_data, m_x.mutable_gpu_data);
            m_x.Reshape(1, 1, nHt, 1);

            //          = sx * sy
            // m_y.diff = (m_y.data * m_y.data)
            double[] rgStdev = convert(m_y.mutable_cpu_data);
            double[] rgStdevXY = new double[rgStdev.Length];
            for (int i = 1; i < nHt; i++)
            {
                rgStdevXY[i] = rgStdev[0] * rgStdev[i];
            }

            rgStdevXY[0] = 1;
            m_y.mutable_cpu_diff = convert(rgStdevXY);

            // Divide Sum by StdevXY
            m_cuda.div(m_x.count(), m_x.gpu_data, m_y.gpu_diff, m_y.mutable_gpu_data);
            m_cuda.mul_scalar(m_x.count(), 1.0 / (double)(nWid - 1), m_y.mutable_gpu_data);
            m_y.SetData(1, 0);

            // Verify the results.
            double[] rgCorrelation = convert(m_y.mutable_cpu_data);
            double[] rgRowMean = calculateRowMeans(nWid, nHt, rgDataA);
            double[] rgRowStdev = calculateRowStdev(nWid, nHt, rgDataA, rgRowMean);
            double[] rgExpected = calculateCorrelations(nWid, nHt, rgDataA, rgRowMean, rgRowStdev);

            for (int i = 0; i < rgCorrelation.Length; i++)
            {
                double dfActual = rgCorrelation[i];
                double dfExpected = rgExpected[i];
                m_log.EXPECT_NEAR(dfActual, dfExpected, 0.00001, "The correlations do not match!");
            }
        }

        public void TestMatrixCorrelation2()
        {
            double[] rgDataA = new double[] { 3.2, 4.5, 8.3, -1.2, 0.03, 0.22, 0.2, 4.9, 8.1, 9.3, 0.1, 0.88, 8.1, 3.9, 0.3, 5.0, -2.1, 1.1 };
            int nHt = 3;
            int nWid = rgDataA.Length / nHt;

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 1, 1, nHt, nWid });
            m_B = new Blob<T>(m_cuda, m_log);
            m_B.ReshapeLike(m_A);
            m_C = new Blob<T>(m_cuda, m_log);
            m_C.Reshape(1, 1, 1, m_A.shape(3));
            m_C.SetData(1);
            m_y = new Blob<T>(m_cuda, m_log);
            m_y.ReshapeLike(m_A);

            // Load the input data (3 x 6)
            m_A.mutable_cpu_data = convert(rgDataA);

            // Calculate the mean of each row.
            m_cuda.matrix_mean_rows(nWid, nHt, m_A.gpu_data, m_C.gpu_data, 1.0, m_B.mutable_gpu_data);

            // Calculate the stdev of each row.
            m_cuda.matrix_stdev_rows(nWid, nHt, m_A.gpu_data, m_C.gpu_data, m_B.gpu_data, m_A.mutable_gpu_diff, m_y.mutable_gpu_data);

            // Calculate the correlations between the first row and all other rows.
            m_cuda.matrix_correlations(nWid, nHt, m_A.gpu_data, m_C.gpu_data, m_B.gpu_data, m_y.gpu_data, m_A.mutable_gpu_diff, m_y.mutable_gpu_diff);
            m_y.Reshape(1, 1, nHt, 1);

            // Verify the results.
            double[] rgCorrelation = convert(m_y.mutable_cpu_diff);
            double[] rgRowMean = calculateRowMeans(nWid, nHt, rgDataA);
            double[] rgRowStdev = calculateRowStdev(nWid, nHt, rgDataA, rgRowMean);
            double[] rgExpected = calculateCorrelations(nWid, nHt, rgDataA, rgRowMean, rgRowStdev);

            for (int i = 0; i < rgCorrelation.Length; i++)
            {
                double dfActual = rgCorrelation[i];
                double dfExpected = rgExpected[i];
                m_log.EXPECT_NEAR(dfActual, dfExpected, 0.00001, "The correlations do not match!");
            }
        }

        private double[] calculateRowMeans(int nWid, int nHt, double[] rg)
        {
            // Sum all rows.
            double[] rgRowMean = new double[nHt];
            for (int i = 0; i < nHt; i++)
            {
                for (int j = 0; j < m_A.width; j++)
                {
                    rgRowMean[i] += rg[i * nWid + j];
                }
            }

            // Calculate the row mean
            for (int i = 0; i < nHt; i++)
            {
                rgRowMean[i] /= (double)nWid;
            }

            return rgRowMean;
        }

        private double[] calculateRowStdev(int nWid, int nHt, double[] rg, double[] rgRowMean)
        {
            double[] rgXY = new double[nHt * nWid];
            // Subtract mean and square
            for (int i = 0; i < nHt; i++)
            {
                for (int j = 0; j < nWid; j++)
                {
                    rgXY[i * nWid + j] = Math.Pow(rg[i * nWid + j] - rgRowMean[i], 2.0);
                }
            }

            // Sum the rows.
            double[] rgRowStdev = new double[nHt];
            for (int i = 0; i < nHt; i++)
            {
                for (int j = 0; j < nWid; j++)
                {
                    rgRowStdev[i] += rgXY[i * m_A.width + j];
                }
            }

            // Divide by n-1 and Square to get expected stddev
            for (int i = 0; i < nHt; i++)
            {
                rgRowStdev[i] = Math.Sqrt(rgRowStdev[i] / (double)(nWid - 1));
            }

            return rgRowStdev;
        }

        private double[] calculateCorrelations(int nWid, int nHt, double[] rg, double[] rgRowMean, double[] rgRowStdev)
        {
            double[] rgXY = new double[nHt * nWid];
            // Subtract mean
            for (int i = 0; i < nHt; i++)
            {
                for (int j = 0; j < nWid; j++)
                {
                    rgXY[i * nWid + j] = rg[i * nWid + j] - rgRowMean[i];
                }
            }

            // Mutltiply first row of XY by all others
            for (int i = 1; i < nHt; i++)
            {
                for (int j = 0; j < nWid; j++)
                {
                    rgXY[i * nWid + j] *= rgXY[j];
                }
            }

            // Sum the rows.
            double[] rgRowCor = new double[nHt];
            for (int i = 0; i < nHt; i++)
            {
                for (int j = 0; j < nWid; j++)
                {
                    rgRowCor[i] += rgXY[i * nWid + j];
                }
            }

            // Calculate sx*sy
            double[] rgSxSy = new double[nHt];
            for (int i = 1; i < nHt; i++)
            {
                rgSxSy[i] = rgRowStdev[0] * rgRowStdev[i];
            }
            rgSxSy[0] = 1;

            // Divide Row Correlation by Sx*Sy
            for (int i = 0; i < nHt; i++)
            {
                rgRowCor[i] /= rgSxSy[i];
            }

            rgRowCor[0] = 1;

            // Divide by 1/(nWid-1)
            for (int i = 1; i < nHt; i++)
            {
                rgRowCor[i] /= (double)(nWid - 1);
            }

            return rgRowCor;
        }

        public void TestCopyBatch()
        {
            int nCacheSize = 4;

            double[] rgSrcData = new double[] { 1, 2, 11, 12, 21, 22, 31, 32, 41, 42 }; // 5 x 2 x 1
            double[] rgSrcLabel = new double[] { 1, 2, 3, 2, 1 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 2 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 5 });
            m_C = new Blob<T>(m_cuda, m_log, new List<int>() { 3, nCacheSize, 2 });

            m_A.mutable_cpu_data = convert(rgSrcData);
            m_B.mutable_cpu_data = convert(rgSrcLabel);

            m_hCursors = m_cuda.AllocHostBuffer(3 * 2);
            m_hWork = m_cuda.AllocHostBuffer(5);

            m_cuda.copy_batch(10, 5, 2, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.mutable_gpu_data, m_C.mutable_gpu_diff, 1, 3, nCacheSize, m_hCursors, m_hWork);

            double[] rgDst = convert(m_C.mutable_cpu_data);

            m_log.CHECK_EQ(rgDst.Length, nCacheSize * 3 * 2, "The dst cache size is incorrect!");

            // Label 1
            m_log.CHECK_EQ(rgDst[0], 1, "data incorrect!");
            m_log.CHECK_EQ(rgDst[1], 2, "data incorrect!");
            m_log.CHECK_EQ(rgDst[2], 41, "data incorrect!");
            m_log.CHECK_EQ(rgDst[3], 42, "data incorrect!");
            m_log.CHECK_EQ(rgDst[4], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[5], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[6], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[7], 0, "data incorrect!");

            // Label 2
            m_log.CHECK_EQ(rgDst[8 + 0], 11, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 1], 12, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 2], 31, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 3], 32, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 4], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 5], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 6], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 7], 0, "data incorrect!");

            // Label 3
            m_log.CHECK_EQ(rgDst[16 + 0], 21, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 1], 22, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 2], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 3], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 4], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 5], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 6], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 7], 0, "data incorrect!");

            double[] rgSrcData1 = new double[] { 51, 52, 61, 62, 71, 72, 81, 82, 91, 92 }; // 5 x 2 x 1
            double[] rgSrcLabel1 = new double[] { 1, 1, 3, 2, 1 };

            m_A.SetData(convert(rgSrcData1));
            m_B.SetData(convert(rgSrcLabel1));

            m_cuda.copy_batch(10, 5, 2, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.mutable_gpu_data, m_C.mutable_gpu_diff, 1, 3, nCacheSize, m_hCursors, m_hWork);

            rgDst = convert(m_C.mutable_cpu_data);

            // Label 1
            m_log.CHECK_EQ(rgDst[0], 51, "data incorrect!");
            m_log.CHECK_EQ(rgDst[1], 52, "data incorrect!");
            m_log.CHECK_EQ(rgDst[2], 61, "data incorrect!");
            m_log.CHECK_EQ(rgDst[3], 62, "data incorrect!");
            m_log.CHECK_EQ(rgDst[4], 91, "data incorrect!");
            m_log.CHECK_EQ(rgDst[5], 92, "data incorrect!");
            m_log.CHECK_EQ(rgDst[6], 1, "data incorrect!");
            m_log.CHECK_EQ(rgDst[7], 2, "data incorrect!");

            // Label 2
            m_log.CHECK_EQ(rgDst[8 + 0], 81, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 1], 82, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 2], 11, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 3], 12, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 4], 31, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 5], 32, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 6], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 7], 0, "data incorrect!");

            // Label 3
            m_log.CHECK_EQ(rgDst[16 + 0], 71, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 1], 72, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 2], 21, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 3], 22, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 4], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 5], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 6], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 7], 0, "data incorrect!");

            double[] rgSrcData2 = new double[] { 1, 2, 11, 12, 21, 22, 31, 32, 41, 42 }; // 5 x 2 x 1
            double[] rgSrcLabel2 = new double[] { 1, 2, 3, 2, 1 };

            m_A.SetData(convert(rgSrcData2));
            m_B.SetData(convert(rgSrcLabel2));

            m_cuda.copy_batch(10, 5, 2, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.mutable_gpu_data, m_C.mutable_gpu_diff, 1, 3, nCacheSize, m_hCursors, m_hWork);

            rgDst = convert(m_C.mutable_cpu_data);

            // Label 1
            m_log.CHECK_EQ(rgDst[0], 1, "data incorrect!");
            m_log.CHECK_EQ(rgDst[1], 2, "data incorrect!");
            m_log.CHECK_EQ(rgDst[2], 41, "data incorrect!");
            m_log.CHECK_EQ(rgDst[3], 42, "data incorrect!");
            m_log.CHECK_EQ(rgDst[4], 51, "data incorrect!");
            m_log.CHECK_EQ(rgDst[5], 52, "data incorrect!");
            m_log.CHECK_EQ(rgDst[6], 61, "data incorrect!");
            m_log.CHECK_EQ(rgDst[7], 62, "data incorrect!");

            // Label 2
            m_log.CHECK_EQ(rgDst[8 + 0], 11, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 1], 12, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 2], 31, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 3], 32, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 4], 81, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 5], 82, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 6], 11, "data incorrect!");
            m_log.CHECK_EQ(rgDst[8 + 7], 12, "data incorrect!");

            // Label 3
            m_log.CHECK_EQ(rgDst[16 + 0], 21, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 1], 22, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 2], 71, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 3], 72, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 4], 21, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 5], 22, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 6], 0, "data incorrect!");
            m_log.CHECK_EQ(rgDst[16 + 7], 0, "data incorrect!");
        }

        public void TestCopySequenceK0()
        {
            int nK = 0;
            int nCacheSize = 4;

            TestCopyBatch();

            double[] rgSrcData = new double[] { 1, 2, 11, 12, 21, 22, 31, 32, 41, 42 }; // 5 x 2 x 1
            double[] rgSrcLabel = new double[] { 1, 2, 3, 2, 1 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 2 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 5 });
            m_A.mutable_cpu_data = convert(rgSrcData);
            m_B.mutable_cpu_data = convert(rgSrcLabel);

            BlobCollection<T> colTop = new BlobCollection<T>();
            List<int> rgnCount = new List<int>();
            List<long> rghTop = new List<long>();
            int nTopCount = 2 + nK;

            for (int i = 0; i < nTopCount; i++)
            {
                Blob<T> b = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
                colTop.Add(b);
                rgnCount.Add(b.count());
                rghTop.Add(b.mutable_gpu_data);
            }

            Blob<T> b1 = new Blob<T>(m_cuda, m_log, 5, nTopCount, 1, 1);
            colTop.Add(b1);
            rgnCount.Add(b1.count());
            rghTop.Add(b1.mutable_gpu_data);

            int nNum = 5;
            int nDim = 2;
            m_cuda.copy_sequence(nK, nNum, nDim, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.gpu_data, 1, 3, nCacheSize, m_hCursors, true, rghTop, rgnCount, m_hWork, false, 1704);

            double[] rgTop0 = convert(colTop[0].mutable_cpu_data);
            double[] rgTop1 = convert(colTop[1].mutable_cpu_data);
            double[] rgTop2 = convert(colTop[2].mutable_cpu_data);
            double[] rgAnchor = new double[nDim];
            double[] rgNegative = new double[nDim];

            for (int i = 0; i < nNum; i++)
            {
                int nLabel = (int)rgTop2[i * nDim + 0];
                int nNegative = (int)rgTop2[i * nDim + 1];

                m_log.CHECK_NE(nLabel, nNegative, "The anchor and negative should not be equal!");

                Array.Copy(rgTop0, i * nDim, rgAnchor, 0, nDim);
                Array.Copy(rgTop1, i * nDim, rgNegative, 0, nDim);

                bool bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgNegative[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and negative should have different data!");
            }

            colTop.Dispose();
        }

        public void TestCopySequenceK1()
        {
            int nK = 1;
            int nCacheSize = 4;

            TestCopyBatch();

            double[] rgSrcData = new double[] { 1, 2, 11, 12, 21, 22, 31, 32, 41, 42 }; // 5 x 2 x 1
            double[] rgSrcLabel = new double[] { 1, 2, 3, 2, 1 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 2 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 5 });
            m_A.mutable_cpu_data = convert(rgSrcData);
            m_B.mutable_cpu_data = convert(rgSrcLabel);

            BlobCollection<T> colTop = new BlobCollection<T>();
            List<int> rgnCount = new List<int>();
            List<long> rghTop = new List<long>();
            int nTopCount = 2 + nK;

            for (int i = 0; i < nTopCount; i++)
            {
                Blob<T> b = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
                colTop.Add(b);
                rgnCount.Add(b.count());
                rghTop.Add(b.mutable_gpu_data);
            }

            Blob<T> b1 = new Blob<T>(m_cuda, m_log, 5, nTopCount, 1, 1);
            colTop.Add(b1);
            rgnCount.Add(b1.count());
            rghTop.Add(b1.mutable_gpu_data);

            int nNum = 5;
            int nDim = 2;
            m_cuda.copy_sequence(nK, nNum, nDim, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.gpu_data, 1, 3, nCacheSize, m_hCursors, true, rghTop, rgnCount, m_hWork, false, 1704);

            double[] rgTop0 = convert(colTop[0].mutable_cpu_data);
            double[] rgTop1 = convert(colTop[1].mutable_cpu_data);
            double[] rgTop2 = convert(colTop[2].mutable_cpu_data);
            double[] rgTop3 = convert(colTop[3].mutable_cpu_data);
            double[] rgAnchor = new double[nDim];
            double[] rgPositive = new double[nDim];
            double[] rgNegative = new double[nDim];

            for (int i = 0; i < nNum; i++)
            {
                int nLabel = (int)rgTop3[i * 3 + 0];
                int nPositive = (int)rgTop3[i * 3 + 1];
                int nNegative = (int)rgTop3[i * 3 + 2];

                m_log.CHECK_EQ(nLabel, nPositive, "The anchor and positive should be equal!");
                m_log.CHECK_NE(nLabel, nNegative, "The anchor and negative should not be equal!");

                Array.Copy(rgTop0, i * nDim, rgAnchor, 0, nDim);
                Array.Copy(rgTop1, i * nDim, rgPositive, 0, nDim);
                Array.Copy(rgTop2, i * nDim, rgNegative, 0, nDim);

                bool bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgPositive[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and positive should have different data!");

                bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgNegative[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and negative should have different data!");
            }

            colTop.Dispose();
        }

        public void TestCopySequenceK1Combine()
        {
            int nK = 0;
            int nCacheSize = 4;

            TestCopyBatch();

            double[] rgSrcData = new double[] { 1, 2, 11, 12, 21, 22, 31, 32, 41, 42 }; // 5 x 2 x 1
            double[] rgSrcLabel = new double[] { 1, 2, 3, 2, 1 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 2 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 5 });
            m_A.mutable_cpu_data = convert(rgSrcData);
            m_B.mutable_cpu_data = convert(rgSrcLabel);

            BlobCollection<T> colTop = new BlobCollection<T>();
            List<int> rgnCount = new List<int>();
            List<long> rghTop = new List<long>();
            int nTopCount = 2 + nK;

            for (int i = 0; i < nTopCount; i++)
            {
                Blob<T> b = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
                colTop.Add(b);
                rgnCount.Add(b.count());
                rghTop.Add(b.mutable_gpu_data);
            }

            Blob<T> b1 = new Blob<T>(m_cuda, m_log, 5, nTopCount, 1, 1);
            colTop.Add(b1);
            rgnCount.Add(b1.count());
            rghTop.Add(b1.mutable_gpu_data);

            int nNum = 5;
            int nDim = 2;
            bool bCombinePositiveAndNegative = true;
            m_cuda.copy_sequence(nK, nNum, nDim, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.gpu_data, 1, 3, nCacheSize, m_hCursors, true, rghTop, rgnCount, m_hWork, bCombinePositiveAndNegative, 1704);

            double[] rgTop0 = convert(colTop[0].mutable_cpu_data);
            double[] rgTop1 = convert(colTop[1].mutable_cpu_data);
            double[] rgTop2 = convert(colTop[2].mutable_cpu_data);
            double[] rgAnchor = new double[nDim];
            double[] rgPosNeg = new double[nDim];

            for (int i = 0; i < nNum; i++)
            {
                int nLabel = (int)rgTop2[i * 2 + 0];
                int nPosNeg = (int)rgTop2[i * 2 + 1];

                if (i % 2 == 0)
                    m_log.CHECK_EQ(nLabel, nPosNeg, "The anchor and positive should be equal!");
                else
                    m_log.CHECK_NE(nLabel, nPosNeg, "The anchor and positive should not be equal!");  // actually holds the alternating negative

                Array.Copy(rgTop0, i * nDim, rgAnchor, 0, nDim);
                Array.Copy(rgTop1, i * nDim, rgPosNeg, 0, nDim);

                bool bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgPosNeg[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and positive should have different data!");
            }

            colTop.Dispose();
        }

        public void TestCopySequenceK2()
        {
            int nK = 2;
            int nCacheSize = 4;

            TestCopyBatch();

            double[] rgSrcData = new double[] { 1, 2, 11, 12, 21, 22, 31, 32, 41, 42 }; // 5 x 2 x 1
            double[] rgSrcLabel = new double[] { 1, 2, 3, 2, 1 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 2 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 5 });
            m_A.mutable_cpu_data = convert(rgSrcData);
            m_B.mutable_cpu_data = convert(rgSrcLabel);

            BlobCollection<T> colTop = new BlobCollection<T>();
            List<int> rgnCount = new List<int>();
            List<long> rghTop = new List<long>();
            int nTopCount = 2 + nK;

            for (int i = 0; i < nTopCount; i++)
            {
                Blob<T> b = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
                colTop.Add(b);
                rgnCount.Add(b.count());
                rghTop.Add(b.mutable_gpu_data);
            }

            Blob<T> b1 = new Blob<T>(m_cuda, m_log, 5, nTopCount, 1, 1);
            colTop.Add(b1);
            rgnCount.Add(b1.count());
            rghTop.Add(b1.mutable_gpu_data);

            int nNum = 5;
            int nDim = 2;
            m_cuda.copy_sequence(nK, nNum, nDim, m_A.gpu_data, m_B.gpu_data, m_C.count(), m_C.gpu_data, 1, 3, nCacheSize, m_hCursors, true, rghTop, rgnCount, m_hWork, false, 1704);

            double[] rgTop0 = convert(colTop[0].mutable_cpu_data);
            double[] rgTop1 = convert(colTop[1].mutable_cpu_data);
            double[] rgTop2 = convert(colTop[2].mutable_cpu_data);
            double[] rgTop3 = convert(colTop[3].mutable_cpu_data);
            double[] rgTop4 = convert(colTop[4].mutable_cpu_data);
            double[] rgAnchor = new double[nDim];
            double[] rgPositive = new double[nDim];
            double[] rgNegative1 = new double[nDim];
            double[] rgNegative2 = new double[nDim];

            for (int i = 0; i < nNum; i++)
            {
                int nLabel = (int)rgTop4[i * 4 + 0];
                int nPositive = (int)rgTop4[i * 4 + 1];
                int nNegative1 = (int)rgTop4[i * 4 + 2];
                int nNegative2 = (int)rgTop4[i * 4 + 3];

                m_log.CHECK_EQ(nLabel, nPositive, "The anchor and positive should be equal!");
                m_log.CHECK_NE(nLabel, nNegative1, "The anchor and negative1 should not be equal!");
                m_log.CHECK_NE(nLabel, nNegative2, "The anchor and negative2 should not be equal!");

                Array.Copy(rgTop0, i * nDim, rgAnchor, 0, nDim);
                Array.Copy(rgTop1, i * nDim, rgPositive, 0, nDim);
                Array.Copy(rgTop2, i * nDim, rgNegative1, 0, nDim);
                Array.Copy(rgTop3, i * nDim, rgNegative2, 0, nDim);

                bool bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgPositive[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and positive should have different data!");

                bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgNegative1[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and negative1 should have different data!");

                bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgAnchor[j] != rgNegative2[j])
                        bDifferent = true;
                }

                m_log.CHECK(bDifferent, "The anchor and negative2 should have different data!");

                bDifferent = false;
                for (int j = 0; j < nDim; j++)
                {
                    if (rgNegative1[j] != rgNegative2[j])
                        bDifferent = true;
                }

                if (!bDifferent)
                {
                    Trace.WriteLine("The negative1 and negative2 should be different!");
                    m_log.WriteLine("WARNING! The negative1 and negative2 should have different data!");
                }
            }

            colTop.Dispose();
        }

        public void TestCopySequence2()
        {
            // Using SEQ_MAJOR ordering.
                                                // #. = sequence, max = 5; .# = batch, max = 4
            double[] rgSrcData1 = new double[] { 1.1, 2.1, 3.1, 4.1, 5.1, 1.2, 2.2, 3.2, 4.2, 5.2, 1.3, 2.3, 3.3, 4.3, 5.3, 1.4, 2.4, 3.4, 4.4, 5.4 }; // 5 x 4 x 1 x 1
                                                // #. = sequence, max = 2; .# = batch, max = 4
            double[] rgSrcData2 = new double[] { 10.11, 20.11, 10.22, 20.22, 10.33, 20.33, 10.44, 20.44 }; // 2 x 4 x 1 x 1

            // Copy src data 1 [-1,4,1,1] -> C
            double[] rgExpected1 = new double[] { 5.1, 0.0, 0.0, 5.2, 0.0, 0.0, 5.3, 0.0, 0.0, 5.4, 0.0, 0.0 };
            // Copy src data 2 [0:1,4,1,1] -> C
            double[] rgExpected2 = new double[] { 5.1, 10.11, 20.11, 5.2, 10.22, 20.22, 5.3, 10.33, 20.33, 5.4, 10.44, 20.44 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 4, 1, 1 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 2, 4, 1, 1 });
            m_A.mutable_cpu_data = convert(rgSrcData1);
            m_B.mutable_cpu_data = convert(rgSrcData2);
            m_C = new Blob<T>(m_cuda, m_log, new List<int>() { 3, 4, 1, 1 });
            m_C.SetData(0);

            // Copy the last 1 x 4 x 1 x 1 from A
            int nCopyAxis = 0;
            int nSrcCopyAxisDim = m_A.shape()[nCopyAxis];
            int nSrcStep = 5;   // src copy axis dim
            m_log.CHECK_EQ(nSrcCopyAxisDim, nSrcStep, "The src step is incorrect!");
            int nSrcStartIdx = 4;
            int nCopyCount = 4; // batch
            int nCopyDim = 1;
            int nSrcCopyCount = m_A.shape()[nCopyAxis + 1];
            m_log.CHECK_EQ(nSrcCopyCount, nCopyCount, "The src copy count is incorrect!");
            int nDstCopyAxisDim = m_C.shape()[nCopyAxis];
            int nDstStep = 3;
            m_log.CHECK_EQ(nDstCopyAxisDim, nDstStep, "The dst step is incorrect!");
            int nDstStartIdx = 0;
            int nSpatialDim = 1 * 1;
            m_cuda.copy_sequence(m_A.count(), m_A.gpu_data, nSrcStep, nSrcStartIdx, nCopyCount, nCopyDim, m_C.mutable_gpu_data, nDstStep, nDstStartIdx, nSpatialDim, nSpatialDim);

            double[] rgC = convert(m_C.mutable_cpu_data);

            for (int i = 0; i < rgC.Length; i++)
            {
                double dfExpected = rgExpected1[i];
                double dfActual = rgC[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            // Copy all 2 x 4 x 1 x 1 from B
            nSrcCopyAxisDim = m_B.shape()[nCopyAxis];
            nSrcStep = 2;   // src copy axis dim
            m_log.CHECK_EQ(nSrcCopyAxisDim, nSrcStep, "The src step is incorrect!");
            nSrcStartIdx = 0;
            nCopyDim = 2;
            nSrcCopyCount = m_B.shape()[nCopyAxis + 1];
            m_log.CHECK_EQ(nSrcCopyCount, nCopyCount, "The src copy count is incorrect!");
            nDstCopyAxisDim = m_C.shape()[nCopyAxis];
            nDstStep = 3;
            m_log.CHECK_EQ(nDstCopyAxisDim, nDstStep, "The dst step is incorrect!");
            nDstStartIdx = 1;
            m_cuda.copy_sequence(m_B.count(), m_B.gpu_data, nSrcStep, nSrcStartIdx, nCopyCount, nCopyDim, m_C.mutable_gpu_data, nDstStep, nDstStartIdx, nSpatialDim, nSpatialDim);

            rgC = convert(m_C.mutable_cpu_data);

            for (int i = 0; i < rgC.Length; i++)
            {
                double dfExpected = rgExpected2[i];
                double dfActual = rgC[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }
        }

        public void TestCopySequence2PortionSpatialDomain()
        {
            // Using SEQ_MAJOR ordering.
            // #. = sequence, max = 5; .# = batch, max = 4
            double[] rgSrcData1 = new double[] { 1.1, 1.12, 1.13,  2.1, 2.12, 2.13,  3.1, 3.12, 3.13,  4.1, 4.12, 4.13,  5.1, 5.12, 5.13,                    
                                                 1.2, 1.22, 1.23,  2.2, 2.22, 2.23,  3.2, 3.22, 3.23,  4.2, 4.22, 4.23,  5.2, 5.22, 5.23, 
                                                 1.3, 1.32, 1.33,  2.3, 2.32, 2.33,  3.3, 3.32, 3.33,  4.3, 4.32, 4.33,  5.3, 5.32, 5.33, 
                                                 1.4, 1.42, 1.43,  2.4, 2.42, 2.43,  3.4, 3.42, 3.43,  4.4, 4.42, 4.43,  5.4, 5.42, 5.43 }; // 5 x 4 x 3 x 1
            // #. = sequence, max = 2; .# = batch, max = 4
            double[] rgSrcData2 = new double[] { 10.11, 20.11, 10.22, 20.22, 10.33, 20.33, 10.44, 20.44 }; // 2 x 4 x 1 x 1

            // Copy src data 1 [-1,4,(-1),1] -> C
            double[] rgExpected1 = new double[] { 5.13, 0.0, 0.0, 5.23, 0.0, 0.0, 5.33, 0.0, 0.0, 5.43, 0.0, 0.0 };
            // Copy src data 2 [0:1,4,1,1] -> C
            double[] rgExpected2 = new double[] { 5.13, 10.11, 20.11, 5.23, 10.22, 20.22, 5.33, 10.33, 20.33, 5.43, 10.44, 20.44 };

            m_A = new Blob<T>(m_cuda, m_log, new List<int>() { 5, 4, 3, 1 });
            m_B = new Blob<T>(m_cuda, m_log, new List<int>() { 2, 4, 1, 1 });
            m_A.mutable_cpu_data = convert(rgSrcData1);
            m_B.mutable_cpu_data = convert(rgSrcData2);
            m_C = new Blob<T>(m_cuda, m_log, new List<int>() { 3, 4, 1, 1 });
            m_C.SetData(0);

            // Copy the last 1 x 4 x 1 x 1 from A
            int nCopyAxis = 0;
            int nSrcCopyAxisDim = m_A.shape()[nCopyAxis];
            int nSrcStep = 5;   // src copy axis dim
            m_log.CHECK_EQ(nSrcCopyAxisDim, nSrcStep, "The src step is incorrect!");
            int nSrcStartIdx = 4;
            int nCopyCount = 4; // batch
            int nCopyDim = 1;
            int nSrcCopyCount = m_A.shape()[nCopyAxis + 1];
            m_log.CHECK_EQ(nSrcCopyCount, nCopyCount, "The src copy count is incorrect!");
            int nDstCopyAxisDim = m_C.shape()[nCopyAxis];
            int nDstStep = 3;
            m_log.CHECK_EQ(nDstCopyAxisDim, nDstStep, "The dst step is incorrect!");
            int nDstStartIdx = 0;
            int nSrcSpatialDim = 3 * 1;
            int nDstSpatialDim = 1 * 1;
            int nSrcSpatialDimStartIdx = -1; // index last item in the src spatial dim.
            int nDstSpatialDimStartIdx = -1; // index last item in the dst spatial dim.
            int nSpatialDimCount = 1;
            m_cuda.copy_sequence(m_A.count(), m_A.gpu_data, nSrcStep, nSrcStartIdx, nCopyCount, nCopyDim, m_C.mutable_gpu_data, nDstStep, nDstStartIdx, nSrcSpatialDim, nDstSpatialDim, nSrcSpatialDimStartIdx, nDstSpatialDimStartIdx, nSpatialDimCount);

            double[] rgC = convert(m_C.mutable_cpu_data);

            for (int i = 0; i < rgC.Length; i++)
            {
                double dfExpected = rgExpected1[i];
                double dfActual = rgC[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }

            // Copy all 2 x 4 x 1 x 1 from B
            nSrcCopyAxisDim = m_B.shape()[nCopyAxis];
            nSrcStep = 2;   // src copy axis dim
            m_log.CHECK_EQ(nSrcCopyAxisDim, nSrcStep, "The src step is incorrect!");
            nSrcStartIdx = 0;
            nCopyDim = 2;
            nSrcCopyCount = m_B.shape()[nCopyAxis + 1];
            m_log.CHECK_EQ(nSrcCopyCount, nCopyCount, "The src copy count is incorrect!");
            nDstCopyAxisDim = m_C.shape()[nCopyAxis];
            nDstStep = 3;
            m_log.CHECK_EQ(nDstCopyAxisDim, nDstStep, "The dst step is incorrect!");
            nDstStartIdx = 1;
            nSrcSpatialDim = 1;
            m_cuda.copy_sequence(m_B.count(), m_B.gpu_data, nSrcStep, nSrcStartIdx, nCopyCount, nCopyDim, m_C.mutable_gpu_data, nDstStep, nDstStartIdx, nSrcSpatialDim, nDstSpatialDim);

            rgC = convert(m_C.mutable_cpu_data);

            for (int i = 0; i < rgC.Length; i++)
            {
                double dfExpected = rgExpected2[i];
                double dfActual = rgC[i];
                m_log.EXPECT_NEAR_FLOAT(dfExpected, dfActual, 0.000001, "The values are not as expected!");
            }
        }
    }
}
