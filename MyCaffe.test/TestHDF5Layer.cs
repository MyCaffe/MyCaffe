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
using System.IO;
using System.Net;
using System.IO.Compression;
using System.Threading;

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
        public void TestRead()
        {
            HDF5LayerTest test = new HDF5LayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IHDF5LayerTest t in test.Tests)
                {
                    t.TestRead();
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
        void TestRead();
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
        string m_strFileName;
        Blob<T> m_blobCont;
        Blob<T> m_blobInput;
        Blob<T> m_blobTarget;
        Blob<T> m_blobStage;
        Blob<T> m_blobFrameFc7;
        Blob<T> m_blobTopLabel;
        Blob<T> m_blobTopLabel2;
        double m_dfLastProgress = -1;
        Stopwatch m_swUpdateTimer = new Stopwatch();
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public HDF5LayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blobCont = new Blob<T>(m_cuda, m_log, false);
            m_blobInput = new Blob<T>(m_cuda, m_log, false);
            m_blobTarget = new Blob<T>(m_cuda, m_log, false);
            m_blobStage = new Blob<T>(m_cuda, m_log, false);
            m_blobFrameFc7 = new Blob<T>(m_cuda, m_log, false);

            m_blobTopLabel = new Blob<T>(m_cuda, m_log, false);
            m_blobTopLabel2 = new Blob<T>(m_cuda, m_log, false);

            m_strFileName = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            m_strFileName += "\\MyCaffe\\test_data\\data\\hdf5\\sample_data_list.txt";
            m_log.WriteLine("Using sample HDF5 data file '" + m_strFileName + "'");

            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\hdf5\\";
            string strFile = strPath + "batch_0.h5";

            if (!File.Exists(strFile))
                getHdf5Data(strPath);
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
            m_blobTopLabel.Dispose();
            m_blobTopLabel2.Dispose();

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

        public void TestRead()
        {
            int nBatchSize = 5;
            int nNumCols = 8;
            int nHeight = 6;
            int nWidth = 5;

            // Create LayerParameter with the known parameters.
            // The data file we are reading has 10 rows and 8 columns.
            // with values from 0 to 10*8 reshaped in row-major order.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.HDF5_DATA);
            p.top.Add("data");
            p.top.Add("label");
            p.top.Add("label2");
            p.hdf5_data_param.batch_size = (uint)nBatchSize;
            p.hdf5_data_param.source = m_strFileName;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            BottomVec.Clear();
            TopVec.Clear();
            TopVec.Add(m_blob_top);
            TopVec.Add(m_blobTopLabel);
            TopVec.Add(m_blobTopLabel2);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, nBatchSize, "The top 'num' is incorrect.");
            m_log.CHECK_EQ(m_blob_top.channels, nNumCols, "The top 'channels' is incorrect.");
            m_log.CHECK_EQ(m_blob_top.height, nHeight, "The top 'height' is incorrect.");
            m_log.CHECK_EQ(m_blob_top.width, nWidth, "The top 'width' is incorrect.");

            m_log.CHECK_EQ(m_blobTopLabel.num_axes, 2, "The top label 'num_axes' is incorrect.");
            m_log.CHECK_EQ(m_blobTopLabel.shape(0), nBatchSize, "The top label 'shape(0)' is incorrect.");
            m_log.CHECK_EQ(m_blobTopLabel.shape(1), 1, "The top label 'shape(1)' is incorrect.");

            m_log.CHECK_EQ(m_blobTopLabel2.num_axes, 2, "The top label2 'num_axes' is incorrect.");
            m_log.CHECK_EQ(m_blobTopLabel2.shape(0), nBatchSize, "The top label2 'shape(0)' is incorrect.");
            m_log.CHECK_EQ(m_blobTopLabel2.shape(1), 1, "The top label2 'shape(1)' is incorrect.");

            layer.Setup(BottomVec, TopVec);

            // Go through the data 10 times (5 batches)
            int nDataSize = nNumCols * nHeight * nWidth;

            for (int iter = 0; iter < 10; iter++)
            {
                layer.Forward(BottomVec, TopVec);

                // On even iterations, we're reading the first half of the data.
                // On odd iterations, we're reading the second half of the data.
                // NB: label is 1- indexed
                int nLabelOffset = 1 + ((iter % 2 == 0) ? 0 : nBatchSize);
                int nLabel2Offset = 1 + nLabelOffset;
                int nDataOffset = (iter % 2 == 0) ? 0 : nBatchSize * nDataSize;

                // Every two iterations we are reading the second file,
                // which has the same labels, but data is offset by total data size,
                // which is 2400 (see generate_sample_data).
                int nFileOffset = (iter % 4 < 2) ? 0 : 2400;
                double[] rgLabel = convert(m_blobTopLabel.mutable_cpu_data);
                double[] rgLabel2 = convert(m_blobTopLabel2.mutable_cpu_data);

                for (int i=0; i<nBatchSize; i++)
                {
                    m_log.CHECK_EQ(nLabelOffset + i, rgLabel[i], "The label data is incorrect.");
                    m_log.CHECK_EQ(nLabel2Offset + i, rgLabel2[i], "The label2 data is incorrect.");
                }

                double[] rgTopData = convert(m_blob_top.mutable_cpu_data);
                for (int i = 0; i < nBatchSize; i++)
                {
                    for (int j = 0; j < nNumCols; j++)
                    {
                        for (int h = 0; h < nHeight; h++)
                        {
                            for (int w = 0; w < nWidth; w++)
                            {
                                int nIdx = i * nNumCols * nHeight * nWidth +
                                           j * nHeight * nWidth +
                                           h * nWidth +
                                           w;
                                m_log.CHECK_EQ(nFileOffset + nDataOffset + nIdx, rgTopData[nIdx], "debug: i " + i.ToString() + " j " + j.ToString() + " iter " + iter.ToString());
                            }
                        }
                    }
                }
            }
        }

        private void getHdf5Data(string strPath)
        {
            string strUrl = "https://signalpop.blob.core.windows.net/mycaffe/batch_hdf5.zip";
            string strFile1 = "batch_hdf5.zip";
            string strFile = strPath + strFile1;

            try
            {
                m_swUpdateTimer.Restart();
                ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

                if (File.Exists(strFile))
                    File.Delete(strFile);

                using (WebClient webClient = new WebClient())
                {
                    webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                    webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                    webClient.DownloadFileAsync(new Uri(strUrl), strFile, strFile1);
                }

                m_evtDownloadDone.WaitOne(5 * 60 * 1000);
                if (!File.Exists(strFile))
                    throw new Exception("Failed to download '" + strFile1 + "'!");

                ZipFile.ExtractToDirectory(strFile, strPath);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
        }

        private void WebClient_DownloadFileCompleted(object sender, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            bool bTraceEnabled = m_log.EnableTrace;
            m_log.EnableTrace = true;
            m_log.WriteLine("Downloading done.");
            m_log.EnableTrace = bTraceEnabled;

            m_evtDownloadDone.Set();
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_swUpdateTimer.Elapsed.TotalMilliseconds >= 1000)
            {
                if (m_dfLastProgress != e.ProgressPercentage)
                {
                    m_dfLastProgress = e.ProgressPercentage;
                    string strFile = e.UserState.ToString();
                    bool bTraceEnabled = m_log.EnableTrace;
                    m_log.EnableTrace = true;

                    m_log.Progress = e.ProgressPercentage / 100.0;
                    m_log.WriteLine("Downloading '" + strFile + "' at " + m_log.Progress.ToString("P") + "...");
                    m_log.EnableTrace = bTraceEnabled;
                }

                m_swUpdateTimer.Restart();
            }
        }

        public void TestHDF5()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\hdf5\\";
            string strFile = strPath + "batch_0.h5";

            if (!File.Exists(strFile))
                getHdf5Data(strPath);

            HDF5<T> hdf5 = new HDF5<T>(m_cuda, m_log, strFile);

            hdf5.load_nd_dataset(m_blobCont, "cont", true);
            m_log.CHECK_EQ(m_blobCont.num_axes, 2, "The 'cont_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blobCont.shape(0), 1000, "The 'cont_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blobCont.shape(1), 16, "The 'cont_sentence' should have shape(1) = 16");

            double[] rgData = convert(m_blobCont.mutable_cpu_data);
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

            hdf5.load_nd_dataset(m_blobCont, "cont_sentence", true);
            m_log.CHECK_EQ(m_blobCont.num_axes, 2, "The 'cont_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blobCont.shape(0), 1000, "The 'cont_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blobCont.shape(1), 16, "The 'cont_sentence' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blobInput, "input_sentence", true);
            m_log.CHECK_EQ(m_blobInput.num_axes, 2, "The 'input_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blobInput.shape(0), 1000, "The 'input_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blobInput.shape(1), 16, "The 'input_sentence' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blobTarget, "target_sentence", true);
            m_log.CHECK_EQ(m_blobTarget.num_axes, 2, "The 'target_sentence' should have 2 axes.");
            m_log.CHECK_EQ(m_blobTarget.shape(0), 1000, "The 'target_sentence' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blobTarget.shape(1), 16, "The 'target_sentence' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blobStage, "stage_indicator", true);
            m_log.CHECK_EQ(m_blobStage.num_axes, 2, "The 'stage_indicator' should have 2 axes.");
            m_log.CHECK_EQ(m_blobStage.shape(0), 1000, "The 'stage_indicator' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blobStage.shape(1), 16, "The 'stage_indicator' should have shape(1) = 16");

            hdf5.load_nd_dataset(m_blob_bottom, "frame_fc7", true);
            m_log.CHECK_EQ(m_blob_bottom.num_axes, 3, "The 'frame_fc7' should have 2 axes.");
            m_log.CHECK_EQ(m_blob_bottom.shape(0), 1000, "The 'frame_fc7' should have shape(0) = 1000");
            m_log.CHECK_EQ(m_blob_bottom.shape(1), 16, "The 'frame_fc7' should have shape(1) = 16");
            m_log.CHECK_EQ(m_blob_bottom.shape(2), 4096, "The 'frame_fc7' should have shape(2) = 4096");
        }
    }
}
