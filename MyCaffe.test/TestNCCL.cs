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
using MyCaffe.imagedb;
using System.Drawing;
using System.Threading.Tasks;
using System.Threading;

namespace MyCaffe.test
{
    [TestClass]
    public class TestNCCL
    {
        [TestMethod]
        public void TestBroadcast()
        {
            NCCLTest test = new NCCLTest();

            try
            {
                foreach (INcclTest t in test.Tests)
                {
                    t.TestBroadcast();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllReduce()
        {
            NCCLTest test = new NCCLTest();

            try
            {
                foreach (INcclTest t in test.Tests)
                {
                    t.TestAllReduce();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface INcclTest : ITest
    {
        void TestBroadcast();
        void TestAllReduce();
    }

    class NCCLTest : TestBase
    {
        public NCCLTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("NCCL Test", 1, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NCCLTest<double>(strName, nDeviceID, engine);
            else
                return new NCCLTest<float>(strName, nDeviceID, engine);
        }
    }

    class NCCLTest<T> : TestEx<T>, INcclTest
    {
        AutoResetEvent m_evtThreadCreated = new AutoResetEvent(false);
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        AutoResetEvent m_evtNcclRootCreated = new AutoResetEvent(false);
        AutoResetEvent m_evtNcclCreated = new AutoResetEvent(false);
        AutoResetEvent m_evtNcclInitialized = new AutoResetEvent(false);
        AutoResetEvent m_evtNcclBroadcasted = new AutoResetEvent(false);
        AutoResetEvent m_evtNcclDone = new AutoResetEvent(false);
        Task<int> m_task1;
        double[] m_rgData;
        Guid m_guid;
        int m_nGpu1 = 3;
        int m_nGpu2 = 4;
        int m_nDataCount = 4000000;

        public NCCLTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 1000, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;

            m_guid = Guid.NewGuid();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private int processTestBroadcast(object arg)
        {
            Tuple<long, long> param = arg as Tuple<long, long>;
            CudaDnn<T> cuda = new CudaDnn<T>(m_nGpu2);

            Blob<T> data = new Blob<T>(cuda, m_log, m_nDataCount, 1, 1, 1);
            Filler<T> filler = Filler<T>.Create(cuda, m_log, new FillerParameter("constant"));
            filler.Fill(data);

            long hNccl2 = cuda.KernelCopyNccl(param.Item1, param.Item2);
            m_evtThreadCreated.Set();

            // Load the broadcasted data.
            cuda.NcclBroadcast(hNccl2, 0, data.mutable_gpu_data, data.count());
            cuda.SynchronizeStream();

            m_rgData = convert(data.update_cpu_data());
            m_evtNcclDone.Set();

            while (!m_evtCancel.WaitOne(50))
            {
            }

            cuda.FreeNCCL(hNccl2);

            data.Dispose();
            cuda.Dispose();

            return 0;
        }

        private bool setGpus()
        {
            CudaDnn<T> cuda = new CudaDnn<T>(0);
            int nDevCount = cuda.GetDeviceCount();

            if (nDevCount < 0)
                return false;

            List<int> rgGpu = new List<int>();
            for (int i = 0; i < nDevCount; i++)
            {
                string strDevInfo = cuda.GetDeviceInfo(i, true);
                string strP2PInfo = cuda.GetDeviceP2PInfo(i);

                if (strP2PInfo.Contains("P2P Capable = YES"))
                    rgGpu.Add(i);
            }

            if (rgGpu.Count < 2)
                return false;

            m_nGpu1 = rgGpu[0];
            m_nGpu2 = rgGpu[1];

            cuda.Dispose();

            return true;
        }

        public void TestBroadcast()
        {
            if (!setGpus())
            {
                m_log.WriteLine("WARNING: You must have 2 P2P capable GPU's that do not have a monitor connected to perform NCCL tests.");
                return;
            }

            CudaDnn<T> cuda = new CudaDnn<T>(m_nGpu1);
            long hNccl1 = cuda.CreateNCCL(m_nGpu1, 2, 0, m_guid);
            long hNccl2 = cuda.CreateNCCL(m_nGpu2, 2, 1, m_guid);

            Blob<T> data = new Blob<T>(cuda, m_log, m_nDataCount, 1, 1, 1);
            Filler<T> filler = Filler<T>.Create(cuda, m_log, new FillerParameter("gaussian"));
            filler.Fill(data);

            cuda.NcclInitializeSingleProcess(hNccl1, hNccl2);
            m_evtNcclInitialized.Set();

            m_task1 = Task.Factory.StartNew(new Func<object, int>(processTestBroadcast), new Tuple<long, long>(cuda.KernelHandle, hNccl2));
            m_evtThreadCreated.WaitOne();

            // Broadcast the random data.
            cuda.NcclBroadcast(hNccl1, 0, data.gpu_data, data.count());
            cuda.SynchronizeStream();

            // At this point the data on the other
            //  thread should contain a copy of
            //  the random data from this thread.

            m_evtNcclDone.WaitOne();
            double[] rgBottom = convert(data.update_cpu_data());

            for (int i = 0; i < data.count(); i++)
            {
                Assert.AreEqual(rgBottom[i], m_rgData[i]);
            }

            m_evtCancel.Set();
            cuda.FreeNCCL(hNccl1);
            cuda.FreeNCCL(hNccl2);

            data.Dispose();
            cuda.Dispose();
        }

        private int processTestReduce(object arg)
        {
            Tuple<long, long> param = arg as Tuple<long, long>;
            CudaDnn<T> cuda = new CudaDnn<T>(m_nGpu2);
            long hStream = cuda.CreateStream();

            Blob<T> data = new Blob<T>(cuda, m_log, m_nDataCount, 1, 1, 1);
            Filler<T> filler = Filler<T>.Create(cuda, m_log, new FillerParameter("constant"));
            filler.Fill(data);

            long hNccl2 = cuda.KernelCopyNccl(param.Item1, param.Item2);
            m_evtThreadCreated.Set();

            // Load the broadcasted data.
            cuda.NcclBroadcast(hNccl2, hStream, data.mutable_gpu_data, data.count());
            cuda.SynchronizeStream(hStream);

            filler = Filler<T>.Create(cuda, m_log, new FillerParameter("constant", 2.0));
            filler.Fill(data);

            // Start the AllReduction using a specific
            //  stream from this thread for synchronization.
            //
            // NOTE: a specific thread is used for the 
            //  reduction - using the default thread causes
            //  a crash. 
            cuda.NcclAllReduce(hNccl2, hStream, data.mutable_gpu_data, data.count(), NCCL_REDUCTION_OP.SUM);
            cuda.SynchronizeStream(hStream);
            m_evtNcclDone.Set();

            while (!m_evtCancel.WaitOne(50))
            {
            }

            cuda.FreeNCCL(hNccl2);
            cuda.FreeStream(hStream);
            data.Dispose();
            cuda.Dispose();

            return 0;
        }

        public void TestAllReduce()
        {
            if (!setGpus())
            {
                m_log.WriteLine("WARNING: You must have 2 P2P capable GPU's that do not have a monitor connected to perform NCCL tests.");
                return;
            }

            CudaDnn<T> cuda = new CudaDnn<T>(m_nGpu1);
            long hNccl1 = cuda.CreateNCCL(m_nGpu1, 2, 0, m_guid);
            long hNccl2 = cuda.CreateNCCL(m_nGpu2, 2, 1, m_guid);
            long hStream = cuda.CreateStream();

            Blob<T> data = new Blob<T>(cuda, m_log, m_nDataCount, 1, 1, 1);
            Filler<T> filler = Filler<T>.Create(cuda, m_log, new FillerParameter("constant", 1.0));
            filler.Fill(data);

            cuda.NcclInitializeSingleProcess(hNccl1, hNccl2);
            m_evtNcclInitialized.Set();

            m_task1 = Task.Factory.StartNew(new Func<object, int>(processTestReduce), new Tuple<long, long>(cuda.KernelHandle, hNccl2));
            m_evtThreadCreated.WaitOne();

            // Broadcast the bottom random data.
            cuda.NcclBroadcast(hNccl1, hStream, data.gpu_data, data.count());
            cuda.SynchronizeStream(hStream);

            // At this point the data from this thread
            //  should have been broadcast to the thread.
            //  Now start the SUM reduction of all data.
            //
            // NOTE: a specific thread is used for the 
            //  reduction - using the default thread causes
            //  a crash. 
            cuda.NcclAllReduce(hNccl1, hStream, data.mutable_gpu_data, data.count(), NCCL_REDUCTION_OP.SUM);
            cuda.SynchronizeStream(hStream);
            m_evtNcclDone.WaitOne();

            // The data received should be a sum of
            //  the data from this thread (e.g. 1's)
            //  and the data in the test thread (e.g. 2's)
            //  with the result being all 3's.
            double[] rgBottom = convert(data.update_cpu_data());

            for (int i = 0; i < data.count(); i++)
            {
                Assert.AreEqual(rgBottom[i], 3.0);
            }

            m_evtCancel.Set();
            cuda.FreeNCCL(hNccl1);
            cuda.FreeNCCL(hNccl2);
            cuda.FreeStream(hStream);
            data.Dispose();
            cuda.Dispose();
        }
    }
}
