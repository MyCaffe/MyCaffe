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
        public HDF5LayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward()
        {
        }

        public void TestHDF5()
        {
            string strFile = @"C:\Users\winda\Source\Repos\s2vt_data_python\s2vt_data_python\hdf5\buffer_16_s2vt_80\train_batches\batch_0.h5";
            HDF5<T> hdf5 = new HDF5<T>(m_cuda, m_log, strFile);

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
