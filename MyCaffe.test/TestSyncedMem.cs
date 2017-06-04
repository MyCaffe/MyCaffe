using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSyncedMem
    {
        [TestMethod]
        public void TestInitialization()
        {
            SyncedMemTest test = new SyncedMemTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == common.DataType.DOUBLE)
                        test.TestInitialization<double>((Test<double>)t);
                    else
                        test.TestInitialization<float>((Test<float>)t);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllocationCPUGPU()
        {
            SyncedMemTest test = new SyncedMemTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == common.DataType.DOUBLE)
                        test.TestAllocationCPUGPU<double>((Test<double>)t);
                    else
                        test.TestAllocationCPUGPU<float>((Test<float>)t);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCPUWrite()
        {
            SyncedMemTest test = new SyncedMemTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == common.DataType.DOUBLE)
                        test.TestCPUWrite<double>((Test<double>)t);
                    else
                        test.TestCPUWrite<float>((Test<float>)t);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGPUWrite()
        {
            SyncedMemTest test = new SyncedMemTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == common.DataType.DOUBLE)
                        test.TestGPUWrite<double>((Test<double>)t);
                    else
                        test.TestGPUWrite<float>((Test<float>)t);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGPURead()
        {
            SyncedMemTest test = new SyncedMemTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == common.DataType.DOUBLE)
                        test.TestGPURead<double>((Test<double>)t);
                    else
                        test.TestGPURead<float>((Test<float>)t);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestClone()
        {
            SyncedMemTest test = new SyncedMemTest();

            try
            {
                foreach (ITest t in test.Tests)
                {
                    if (t.DataType == common.DataType.DOUBLE)
                        test.TestClone<double>((Test<double>)t);
                    else
                        test.TestClone<float>((Test<float>)t);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class SyncedMemTest : TestBase
    {
        public SyncedMemTest()
            : base("SyncedMem Test")
        {
        }

        public void TestInitialization<T>(Test<T> test)
        {
            SyncedMemory<T> mem = new SyncedMemory<T>(test.CudaObj, test.Log, 10);

            test.Log.CHECK_EQ(10, mem.Capacity, "The synced mem should have capacity of 10.");
            test.Log.CHECK_EQ(10, mem.Count, "The synced mem should have capacity of 0.");
            test.Log.CHECK_NE(0, mem.gpu_data, "The gpu data should not be NULL.");

            SyncedMemory<T> mem2 = new SyncedMemory<T>(test.CudaObj, test.Log, 20);
            test.Log.CHECK_EQ(20, mem2.Capacity, "The synced mem should have capacity of 10.");
            test.Log.CHECK_EQ(20, mem2.Count, "The synced mem should have capacity of 0.");
            test.Log.CHECK_NE(0, mem2.gpu_data, "The gpu data should not be NULL.");

            mem.Dispose();
            mem2.Dispose();
        }

        public void TestAllocationCPUGPU<T>(Test<T> test)
        {
            SyncedMemory<T> mem = new SyncedMemory<T>(test.CudaObj, test.Log, 10);

            long hGpu = mem.gpu_data;
            test.Log.CHECK_NE(0, hGpu, "Gpu data should have been allocated.");
            long hGpu2 = mem.mutable_gpu_data;
            test.Log.CHECK_NE(0, hGpu2, "Mutable gpu data should have been allocated.");
            T[] rgCpu = mem.cpu_data;
            test.Log.CHECK(rgCpu == null, "Cpu data should be null until updated.");
            T[] rgCpu2 = mem.mutable_cpu_data;
            test.Log.CHECK(rgCpu2 != null, "Mutable cpu data should not be null.");
            rgCpu = mem.cpu_data;
            test.Log.CHECK(rgCpu != null, "Now that the CPU data is updated, it should not be null.");

            mem.Dispose();
        }

        public void TestCPUWrite<T>(Test<T> test)
        {
            SyncedMemory<T> mem = new SyncedMemory<T>(test.CudaObj, test.Log, 10);
            List<T> rg = new List<T>();

            for (int i = 0; i < 5; i++)
            {
                rg.Add((T)Convert.ChangeType((i+1) * 1.1, typeof(T)));
            }

            mem.mutable_cpu_data = rg.ToArray();
            T[] rg1 = mem.mutable_cpu_data;

            test.Log.CHECK_EQ(rg1.Length, rg.Count, "The input and output arrays should have the same size.");

            for (int i=0; i<rg.Count; i++)
            {
                double df1 = (double)Convert.ChangeType(rg[i], typeof(double));
                double df2 = (double)Convert.ChangeType(rg1[i], typeof(double));

                test.Log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " should be the same.");
            }

            T[] rg2 = test.CudaObj.GetMemory(mem.gpu_data);

            for (int i=0; i<rg2.Length; i++)
            {
                double df2 = (double)Convert.ChangeType(rg2[i], typeof(double));

                if (i < rg.Count)
                {
                    double df1 = (double)Convert.ChangeType(rg[i], typeof(double));
                    test.Log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " should be equal.");
                }
                else
                {
                    test.Log.CHECK_EQ(0, df2, "The value at " + i.ToString() + " should be 0.");
                }
            }

            test.Log.CHECK_EQ(mem.gpu_data, mem.mutable_gpu_data, "The gpu handles should be the same.");

            T tVal = (T)Convert.ChangeType(0, typeof(T));
            test.CudaObj.set((int)mem.Count, mem.gpu_data, tVal);

            rg2 = test.CudaObj.GetMemory(mem.gpu_data);
            for (int i=0; i<rg2.Length; i++)
            {
                test.Log.CHECK_EQ(0, (double)Convert.ChangeType(rg2[i], typeof(double)), "The value at " + i.ToString() + " should be 0.");
            }

            mem.Dispose();
        }

        public void TestGPUWrite<T>(Test<T> test)
        {
            SyncedMemory<T> mem1 = new SyncedMemory<T>(test.CudaObj, test.Log, 10);
            SyncedMemory<T> mem2 = new SyncedMemory<T>(test.CudaObj, test.Log, 10);
            List<T> rg = new List<T>();

            for (int i = 0; i < 5; i++)
            {
                rg.Add((T)Convert.ChangeType((i + 1) * 1.1, typeof(T)));
            }

            mem1.mutable_cpu_data = rg.ToArray();
            T[] rg1 = mem1.mutable_cpu_data;

            test.Log.CHECK_EQ(rg1.Length, rg.Count, "The input and output arrays should have the same size.");

            for (int i = 0; i < rg.Count; i++)
            {
                double df1 = (double)Convert.ChangeType(rg[i], typeof(double));
                double df2 = (double)Convert.ChangeType(rg1[i], typeof(double));

                test.Log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " should be the same.");
            }

            mem2.Copy(mem1);
            T[] rg2 = mem2.mutable_cpu_data;

            test.Log.CHECK_EQ(rg2.Length, rg.Count, "The input and output arrays should have the same size.");

            for (int i = 0; i < rg.Count; i++)
            {
                double df1 = (double)Convert.ChangeType(rg[i], typeof(double));
                double df2 = (double)Convert.ChangeType(rg2[i], typeof(double));

                test.Log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " should be the same.");
            }

            mem1.Dispose();
            mem2.Dispose();
        }

        public void TestGPURead<T>(Test<T> test)
        {
            SyncedMemory<T> mem1 = new SyncedMemory<T>(test.CudaObj, test.Log, 10);
            double[] rgData = Utility.ConvertVec<T>(mem1.mutable_cpu_data);

            for (int i = 0; i < rgData.Length; i++)
            {
                rgData[i] = 1.0;
            }

            mem1.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
            long hGpuData = mem1.gpu_data;

            // check that values are the same.
            double[] rgData2 = Utility.ConvertVec<T>(mem1.update_cpu_data());

            for (int i = 0; i < rgData2.Length; i++)
            {
                test.Log.CHECK_EQ(rgData[i], rgData2[i], "The data items at " + i.ToString() + " are not the same!");
            }

            // do another round.
            rgData = Utility.ConvertVec<T>(mem1.mutable_cpu_data);

            for (int i = 0; i < rgData.Length; i++)
            {
                rgData[i] = 2;
            }

            mem1.mutable_cpu_data = Utility.ConvertVec<T>(rgData);

            // check if values are the same.
            rgData2 = Utility.ConvertVec<T>(mem1.update_cpu_data());

            for (int i = 0; i < rgData2.Length; i++)
            {
                test.Log.CHECK_EQ(rgData[i], rgData2[i], "The data items at " + i.ToString() + " are not the same!");
            }

            mem1.Dispose();
        }

        public void TestClone<T>(Test<T> test)
        {
            SyncedMemory<T> mem1 = new SyncedMemory<T>(test.CudaObj, test.Log, 10);
            SyncedMemory<T> mem2 = null;
            List<T> rg = new List<T>();

            for (int i = 0; i < 5; i++)
            {
                rg.Add((T)Convert.ChangeType((i + 1) * 1.1, typeof(T)));
            }

            mem1.mutable_cpu_data = rg.ToArray();
            T[] rg1 = mem1.mutable_cpu_data;

            test.Log.CHECK_EQ(rg1.Length, rg.Count, "The input and output arrays should have the same size.");

            for (int i = 0; i < rg.Count; i++)
            {
                double df1 = (double)Convert.ChangeType(rg[i], typeof(double));
                double df2 = (double)Convert.ChangeType(rg1[i], typeof(double));

                test.Log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " should be the same.");
            }

            mem2 = mem1.Clone();
            T[] rg2 = mem2.mutable_cpu_data;

            test.Log.CHECK_EQ(rg2.Length, rg.Count, "The input and output arrays should have the same size.");

            for (int i = 0; i < rg.Count; i++)
            {
                double df1 = (double)Convert.ChangeType(rg[i], typeof(double));
                double df2 = (double)Convert.ChangeType(rg2[i], typeof(double));

                test.Log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " should be the same.");
            }

            mem1.Dispose();
            mem2.Dispose();
        }
    }
}
