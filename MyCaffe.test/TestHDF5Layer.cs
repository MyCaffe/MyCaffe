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
using MyCaffe.layers.hdf5;
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestHDF5Layer
    {
        [TestMethod]
        public void TestForward()
        {
            HDF5LayerTest test = new HDF5LayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IHDF5LayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestHDF5()
        {
            HDF5LayerTest test = new HDF5LayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IHDF5LayerTest t in test.Tests)
                {
                    t.TestHDF5();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IHDF5LayerTest : ITest
    {
        void TestForward();
        void TestHDF5();
    }

    class HDF5LayerTest : TestBase
    {
        public HDF5LayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("HDF5 Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new HDF5LayerTest<double>(strName, nDeviceID, engine);
            else
                return new HDF5LayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class HDF5LayerTest<T> : TestEx<T>, IHDF5LayerTest
    {
        Blob<T> m_blobCont;
        Blob<T> m_blobInput;
        Blob<T> m_blobTarget;
        Blob<T> m_blobStage;
        Blob<T> m_blobFrameFc7;

        public HDF5LayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blobCont = new Blob<T>(m_cuda, m_log, false);
            m_blobInput = new Blob<T>(m_cuda, m_log, false);
            m_blobTarget = new Blob<T>(m_cuda, m_log, false);
            m_blobStage = new Blob<T>(m_cuda, m_log, false);
            m_blobFrameFc7 = new Blob<T>(m_cuda, m_log, false);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            m_blobCont.Dispose();
            m_blobInput.Dispose();
            m_blobTarget.Dispose();
            m_blobStage.Dispose();
            m_blobFrameFc7.Dispose();

            base.dispose();
        }

        public void TestForward()
        {
            Stopwatch sw = new Stopwatch();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\hdf5\\";
            string strFile = strPath + "file_list.txt";
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.HDF5_DATA);
            p.hdf5_data_param.source = strFile;
            p.hdf5_data_param.batch_size = 80;
            p.top.Add("cont_sentence");
            p.top.Add("input_sentence");
            p.top.Add("target_sentence");
            p.top.Add("stage_indicator");
            p.top.Add("frame_fc7");

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            m_log.CHECK(layer.type == LayerParameter.LayerType.HDF5_DATA, "Incorrect layer type, expected " + LayerParameter.LayerType.HDF5_DATA.ToString());

            BottomVec.Clear();
            TopVec.Clear();

            TopVec.Add(m_blobCont);
            TopVec.Add(m_blobInput);
            TopVec.Add(m_blobTarget);
            TopVec.Add(m_blobStage);
            TopVec.Add(m_blobFrameFc7);

            layer.Setup(BottomVec, TopVec);
            sw.Start();
            layer.Forward(BottomVec, TopVec);
            
            for (int i = 0; i < 100; i++)
            {
                layer.Forward(BottomVec, TopVec);
            }

            sw.Stop();
            double dfTime = sw.Elapsed.TotalMilliseconds;
            int nCount = 101;

            double dfAvePerFwd = dfTime / nCount;
            double dfAvePerItem = dfAvePerFwd / p.hdf5_data_param.batch_size;

            Trace.WriteLine("Average time (ms) per Forward = " + dfAvePerFwd.ToString("N5"));
            Trace.WriteLine("Average time (ms) per Item = " + dfAvePerItem.ToString("N5"));
        }

        public void TestHDF5()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\hdf5\\";
            string strFile = strPath + "batch_0.h5";
            HDF5<T> hdf5 = new HDF5<T>(m_cuda, m_log, strFile);

            hdf5.load_nd_dataset(m_blob_bottom, "cont", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 2, "The 'cont_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'cont_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'cont_sentence' should have shape(1) = 16");

            double[] rgData = convert(m_blob_bottom.mutable_cpu_data);
            int nDim1 = 1000;
            int nDim2 = 16;
            for (int i = 0; i < nDim1; i++)
            {
                int nIdx = i % 80;

                for (int j = 0; j < nDim2; j++)
                {
                    int nDataIdx = i * nDim2 + j;

                    double dfExpected = (nIdx == 0) ? 0 : 1;
                    double dfActual = rgData[nDataIdx];
                    m_log.CHECK_EQ(dfExpected, dfActual, "The data items are not as expected for 'cont'!");
                }
            }

            hdf5.load_nd_dataset(m_blob_bottom, "cont_sentence", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 2, "The 'cont_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'cont_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'cont_sentence' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blob_bottom, "input_sentence", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 2, "The 'input_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'input_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'input_sentence' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blob_bottom, "target_sentence", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 2, "The 'target_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'target_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'target_sentence' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blob_bottom, "stage_indicator", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 2, "The 'stage_indicator' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'stage_indicator' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'stage_indicator' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blob_bottom, "frame_fc7", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 3, "The 'frame_fc7' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'frame_fc7' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'frame_fc7' should have shape(1) = 16");
            m_log.CHECK_EQ(m_blob_bottom.shape(2), 4096, "The 'frame_fc7' should have shape(2) = 4096");
        }
    }
}
