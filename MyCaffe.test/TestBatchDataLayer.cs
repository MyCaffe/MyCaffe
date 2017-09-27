using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.imagedb;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using System.Threading;
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBatchDataLayer
    {
        [TestMethod]
        public void TestInitialize()
        {
            BatchDataLayerTest test = new BatchDataLayerTest();
                     
            try
            {
                foreach (IBatchDataLayerTest t in test.Tests)
                {
                    t.TestInitialization(test.SourceName);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup()
        {
            BatchDataLayerTest test = new BatchDataLayerTest();

            try
            {
                foreach (IBatchDataLayerTest t in test.Tests)
                {
                    t.TestSetup(test.SourceName);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward()
        {
            BatchDataLayerTest test = new BatchDataLayerTest();

            try
            {
                foreach (IBatchDataLayerTest t in test.Tests)
                {
                    t.TestForward(test.SourceName);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class BatchDataLayerTest : TestBase
    {
        SettingsCaffe m_settings;
        MyCaffeImageDatabase m_db;
        CancelEvent m_evtCancel = new CancelEvent();
        Guid m_userGuid;

        public BatchDataLayerTest(string strDs = "MNIST")
            : base("Batch Data Layer Test")
        {
            m_userGuid = Guid.NewGuid();
            m_settings = new SettingsCaffe();
            m_db = new MyCaffeImageDatabase();
            m_db.InitializeWithDsName(m_settings, strDs);
        }

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            string strPath = TestBase.CudaPath;

            if (dt == DataType.DOUBLE)
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new BatchDataLayerTest<double>(strName, nDeviceID, this);
            }
            else
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new BatchDataLayerTest<float>(strName, nDeviceID, this);
            }
        }

        protected override void dispose()
        {
            m_db.Dispose();
            m_db = null;

            base.dispose();
        }

        public string SourceName
        {
            get { return "MNIST.training"; }
        }

        public MyCaffeImageDatabase db
        {
            get { return m_db; }
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }
    }

    interface IBatchDataLayerTest 
    {
        DataType Type { get; }
        void TestInitialization(string strSrc);
        void TestSetup(string strSrc);
        void TestForward(string strSrc);
    }

    class BatchDataLayerTest<T> : TestEx<T>, IBatchDataLayerTest
    {
        BatchDataLayerTest m_parent;
        List<int> m_rgInputVerify = null;

        public BatchDataLayerTest(string strName, int nDeviceID, BatchDataLayerTest parent, List<int> rgBottomShape = null)
            : base(strName, rgBottomShape, nDeviceID)
        {
            m_parent = parent;
        }

        public DataType Type
        {
            get { return m_dt; }
        }

        public void TestInitialization(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHDATA);

            m_log.CHECK(p.batch_data_param != null, "The batch_data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            p.batch_data_param.source = strSrc;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db, null);

            layer.Dispose();
        }

        public void TestSetup(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHDATA);

            m_log.CHECK(p.batch_data_param != null, "The batch_data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            p.batch_data_param.source = strSrc;
            p.batch_data_param.iterations = 2;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db, null);

            layer.LayerSetUp(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);

            Thread.Sleep(2000);

            m_log.CHECK_EQ(layer.layer_param.batch_data_param.iterations, 2, "The layer_param.batch_data_param.iterations should be 2.");

            layer.Dispose();

            m_parent.CancelEvent.Reset();
        }

        public void TestForward(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHDATA);

            m_log.CHECK(p.batch_data_param != null, "The batch_data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            int nIterations = 1000;

            p.batch_data_param.source = strSrc;
            p.batch_data_param.iterations = nIterations;
            p.batch_data_param.CompletedEvent = new AutoResetEvent(false);
            p.batch_data_param.batch_size = 5;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db, new TransferInput(null, setInput));
            int nSrcID = m_parent.db.GetSourceID(strSrc);

            layer.LayerSetUp(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);

            double[] rgInput = new double[] { 0, 1, 2, 3, 4, 4, 3, 2, 1, 0 };
            Bottom.Reshape(2, 5, 1, 1);     // batch 0 = 0, 1, 2, 3, 4;  batch 1 = 4, 3, 2, 1, 0
            Bottom.mutable_cpu_data = convert(rgInput);

            Stopwatch sw = new Stopwatch();
            double dfTotalTime = 0;
            int nBatchIdx = 0;
            int nCount = 0;

            while (!p.batch_data_param.CompletedEvent.WaitOne(0))
            {
                sw.Start();

                m_rgInputVerify = new List<int>();

                for (int j = 0; j < 5; j++)
                {
                    int nIdx = nBatchIdx * 5 + j;
                    int nImgIdx = (int)rgInput[nIdx];
                    m_rgInputVerify.Add(nImgIdx);
                }

                layer.Forward(BottomVec, TopVec);
                dfTotalTime += sw.ElapsedMilliseconds;
                sw.Restart();

                m_log.CHECK_EQ(TopVec.Count, 1, "The top vec should have one element.");
                T[] rgData = TopVec[0].update_cpu_data();
                List<byte> rgData2 = new List<byte>();

                for (int j = 0; j < 5; j++)
                {
                    int nIdx = nBatchIdx * 5 + j;
                    int nImgIdx = (int)rgInput[nIdx];
                    SimpleDatum d = m_parent.db.QueryImage(nSrcID, nImgIdx, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    rgData2.AddRange(d.ByteData);
                }

                m_log.CHECK_EQ(rgData2.Count, rgData.Length, "The data from the data forward should have the same length as the first item in the database for the source = " + strSrc);

                for (int j = 0; j < rgData.Length; j++)
                {
                    double dfVal1 = (double)Convert.ChangeType(rgData[j], typeof(double));
                    double dfVal2 = (double)Convert.ChangeType(rgData2[j], typeof(double));

                    m_log.CHECK_EQ(dfVal1, dfVal2, "The values at index " + j.ToString() + " for batch " + nBatchIdx.ToString() + " in source = " + strSrc + " do not match!");
                }

                nBatchIdx++;

                if (nBatchIdx == 2)
                    nBatchIdx = 0;

                nCount++;
            }

            m_log.CHECK_EQ(nCount, nIterations * 2, "The iteration count should equal " + (nIterations * 2).ToString() + " - " + nIterations.ToString() + " iterations over 2 batches.");

            string str = (dfTotalTime / (double)nCount).ToString() + " ms.";
            Trace.WriteLine("Average BatchDataLayer Forward Time = " + str);

            layer.Dispose();

            m_parent.CancelEvent.Reset();
        }

        void setInput(BatchInput bi)
        {
            List<int> rgInput = bi.InputData as List<int>;

            m_log.CHECK(rgInput != null, "The input value should be a List<int> type");
            m_log.CHECK_EQ(rgInput.Count, m_rgInputVerify.Count, "The input and verify should have the same count.");
            m_log.CHECK(Utility.Compare<int>(rgInput, m_rgInputVerify), "The input anf verify should have the same items");
        }
    }
}
