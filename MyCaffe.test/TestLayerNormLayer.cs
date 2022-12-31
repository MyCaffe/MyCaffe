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
using MyCaffe.param.beta;
using System.IO;
using MyCaffe.param.gpt;
using System.IO.Compression;
using System.Net;
using System.Diagnostics;
using System.Threading;

///
/// WORK IN PROGRESS
///
namespace MyCaffe.test
{
    [TestClass]
    public class TestLayerNormLayer
    {
        [TestMethod]
        public void TestForward()
        {
            LayerNormLayerTest test = new LayerNormLayerTest();

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
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
        public void TestForwardInplace()
        {
            LayerNormLayerTest test = new LayerNormLayerTest();

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestForwardInplace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient()
        {
            LayerNormLayerTest test = new LayerNormLayerTest();

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestForwardPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico3()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestForwardPico(false, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico3()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(false, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico3B()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestForwardPico(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico3B()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(true, 3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPicoBlk()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestBackwardPicoBlk();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardEx()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestForwardEx(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardExCuda()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestForwardEx(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardEx()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestBackwardEx(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardExCuda()
        {
            LayerNormLayerTest test = new LayerNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILayerNormLayerTest t in test.Tests)
                {
                    t.TestBackwardEx(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ILayerNormLayerTest : ITest
    {
        void TestForward();
        void TestForwardInplace();
        void TestGradient();

        void TestForwardPico(bool bBatch, int nHeads);
        void TestBackwardPico(bool bBatch, int nHeads);
        void TestBackwardPicoBlk();

        void TestForwardEx(bool bUseCuda);
        void TestBackwardEx(bool bUseCuda);
    }

    class LayerNormLayerTest : TestBase
    {
        public LayerNormLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("LayerNorm Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LayerNormLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LayerNormLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LayerNormLayerTest<T> : TestEx<T>, ILayerNormLayerTest
    {
        Blob<T> m_blobWork;
        Blob<T> m_blobVal;
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public LayerNormLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 3, 1 }, nDeviceID)
        {
            m_engine = engine;

            m_blobWork = new Blob<T>(m_cuda, m_log);
            m_blobVal = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobVal);
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private double[] calculateLayerNorm(Blob<T> b, LayerNormParameter p)
        {
            double[] rgData = convert(b.update_cpu_data());
            double[] rgNorm = new double[rgData.Length];
            int nSpatialDim = b.height * b.width;

            for (int n = 0; n < b.num; n++)
            {
                for (int c = 0; c < b.channels; c++)
                {
                    double dfTotal = 0;
                    
                    for (int i = 0; i < nSpatialDim; i++)
                    {
                        int nIdx = n * b.channels * nSpatialDim + c * nSpatialDim + i;
                        dfTotal += rgData[nIdx];                        
                    }

                    double dfMean = dfTotal / nSpatialDim;

                    dfTotal = 0;

                    for (int i = 0; i < nSpatialDim; i++)
                    {
                        int nIdx = n * b.channels * nSpatialDim + c * nSpatialDim + i;
                        double dfMeanDiff = rgData[nIdx] - dfMean;
                        double dfMeanDiffSq = dfMeanDiff * dfMeanDiff;
                        dfTotal += dfMeanDiffSq;
                    }

                    double dfVar = dfTotal / nSpatialDim;
                    double dfStd = Math.Sqrt(dfVar + p.epsilon);

                    for (int i = 0; i < nSpatialDim; i++)
                    {
                        int nIdx = n * b.channels * nSpatialDim + c * nSpatialDim + i;
                        double dfNorm = (rgData[nIdx] - dfMean) / dfStd;
                        rgNorm[nIdx] = dfNorm;
                    }
                }
            }

            return rgNorm;
        }

        public void TestForward()
        {
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                m_filler.Fill(m_blob_bottom);
                m_blob_top.SetData(0);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, m_blob_bottom.num, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.channels, m_blob_bottom.channels, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.height, m_blob_bottom.height, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.width, m_blob_bottom.width, "The num does not match!");

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                double[] rgExpected = calculateLayerNorm(m_blob_bottom, p.layer_norm_param);

                for (int i = 0; i < rgTop.Count(); i++)
                {
                    double dfActual = rgTop[i];
                    double dfExpected = rgExpected[i];
                    double dfErr = 1e-5;

                    m_log.EXPECT_NEAR(dfActual, dfExpected, dfErr, "The top data does not match the expected data!");
                }
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestForwardInplace()
        {
            Layer<T> layer = null;
            Blob<T> blobInPlace = null;

            try
            {
                blobInPlace = new Blob<T>(m_cuda, m_log, 2, 3, 3, 1);
                BlobCollection<T> colBottom = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                FillerParameter fp = new FillerParameter("gaussian");
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(blobInPlace);

                m_blob_bottom.CopyFrom(blobInPlace, false, true);

                colBottom.Add(blobInPlace);
                colTop.Add(blobInPlace);

                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                layer.Setup(colBottom, colTop);
                layer.Forward(colBottom, colTop);

                double[] rgTop = convert(blobInPlace.update_cpu_data());
                double[] rgExpected = calculateLayerNorm(m_blob_bottom, p.layer_norm_param);
                
                for (int i = 0; i < rgTop.Count(); i++)
                {
                    double dfActual = rgTop[i];
                    double dfExpected = rgExpected[i];
                    double dfErr = 1e-5;

                    m_log.EXPECT_NEAR(dfActual, dfExpected, dfErr, "The top data does not match the expected data!");
                }
            }
            finally
            {
                if (blobInPlace != null)
                   blobInPlace.Dispose();

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradient()
        {
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);

                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }

        public Tuple<List<int>, float[]> Fill(string strGpt, string strName, Log log, string strPass = "")
        {
            string strFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\gpt\\" + strGpt + "\\";

            if (!string.IsNullOrEmpty(strPass))
                strFile += strPass + "\\";

            strFile += strName + ".txt";

            string[] rgstrLines = File.ReadAllLines(strFile);
            string strSize = rgstrLines[0].Trim('#', ' ', '(', ')', ',');
            string[] rgstrSize = strSize.Split(',');
            List<int> rgnShape = new List<int>() { 1 };

            if (!string.IsNullOrEmpty(strSize))
                rgnShape = rgstrSize.Select(p1 => int.Parse(p1)).ToList();
            List<float> rgfVal = new List<float>();

            while (rgnShape.Count < 4)
            {
                rgnShape.Add(1);
            }

            int nCount = 1;
            foreach (int nDim in rgnShape)
            {
                nCount *= nDim;
            }

            for (int i = 1; i < rgstrLines.Length; i++)
            {
                string[] rgstrVals = rgstrLines[i].Split(' ');

                for (int j = 0; j < rgstrVals.Length; j++)
                {
                    string strVal = rgstrVals[j].Trim();

                    if (!string.IsNullOrEmpty(strVal))
                    {
                        float fVal = float.Parse(strVal);
                        rgfVal.Add(fVal);
                    }
                }
            }

            log.CHECK_EQ(rgfVal.Count, nCount, "The bottom count does not match the number of values read in!");

            float[] rgf = rgfVal.ToArray();

            return new Tuple<List<int>, float[]>(rgnShape, rgf);
        }

        public void TestForwardPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobY = null;

            try
            {
                blobY = new Blob<T>(m_cuda, m_log);

                string strModel = "gpt-pico-ln";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y = Fill(strModel, "y", m_log);
                blobY.Reshape(y.Item1);
                blobY.mutable_cpu_data = convert(y.Item2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Now, check values
                float[] rgExpected = convertF(blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 1e-6f;
                    float fDiff = fActual - fExpected;

                    if (Math.Abs(fDiff) > fErr)
                        m_log.FAIL("The values are not expected!");
                }
            }
            finally
            {
                if (blobY != null)
                    blobY.Dispose();

                layer.Dispose();
            }
        }

        public void TestBackwardPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-ln";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "grad_1a_y", m_log);
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "grad_9_x", m_log);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_blob_top.mutable_cpu_diff = convert(y_grad.Item2);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                // Now, check values
                float[] rgExpected = x_grad.Item2;
                float[] rgActual = convertF(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 1e-6f;
                    float fDiff = fActual - fExpected;

                    if (Math.Abs(fDiff) > fErr)
                        m_log.FAIL("The values are not expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackwardPicoBlk()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-ln-blk";

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "grad_1a_y", m_log, "iter_0");
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "grad_9_x", m_log, "iter_0");
                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log);

                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);
                
                m_blob_top.mutable_cpu_diff = convert(y_grad.Item2);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                // Now, check values
                float[] rgExpected = x_grad.Item2;
                float[] rgActual = convertF(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 1e-6f;
                    float fDiff = fActual - fExpected;

                    if (Math.Abs(fDiff) > fErr)
                        m_log.FAIL("The values are not expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private string loadTestData()
        {
            string strTestDataFile = downloadTestData();
            string strPath = Path.GetDirectoryName(strTestDataFile);

            if (!File.Exists(strPath + "\\test\\ln.1_x.npy"))
                ZipFile.ExtractToDirectory(strTestDataFile, strPath);

            return strPath + "\\test\\";
        }

        private string downloadTestData()
        {
            string strTestDataPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\ln\\";
            if (!Directory.Exists(strTestDataPath))
                Directory.CreateDirectory(strTestDataPath);

            string strTestDataFile = strTestDataPath + "_layernorm_test.zip";
            if (!File.Exists(strTestDataFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/_layernorm_test.zip";
                    string strFile1 = "_layernorm_test.zip";
                    string strFile = strTestDataPath + strFile1;

                    m_swUpdateTimer.Start();
                    m_dfLastProgress = 0;

                    webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                    webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                    webClient.DownloadFileAsync(new Uri(strUrl), strFile, strFile1);

                    m_evtDownloadDone.WaitOne();
                }
            }

            return strTestDataFile;
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

        private void verify(Blob<T> b1, Blob<T> b1exp, bool bCompareDiff, float fTol = 1e-6f)
        {
            float[] rgExpected = (bCompareDiff) ? convertF(b1exp.mutable_cpu_diff) : convertF(b1exp.mutable_cpu_data);
            float[] rgActual = (bCompareDiff) ? convertF(b1.mutable_cpu_diff) : convertF(b1.mutable_cpu_data);

            for (int i = 0; i < rgExpected.Length; i++)
            {
                float fExpected = rgExpected[i];
                float fActual = rgActual[i];
                
                m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fTol, "The values are not as expected!");
            }

            bool bRes = b1.Compare(b1exp, m_blobWork, bCompareDiff, fTol);
            if (!bRes)
                m_log.FAIL("The blobs are not equal!");
        }

        public void TestForwardEx(bool bUseCuda)
        {
            string strTestDataPath = loadTestData();
            Layer<T> layer = null;

            try
            {
                Stopwatch sw = new Stopwatch();

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.layer_norm_param.enable_cuda_impl = bUseCuda;
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                m_blob_bottom.LoadFromNumpy(strTestDataPath + "q0.npy");
                m_blob_top.SetData(0);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, m_blob_bottom.num, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.channels, m_blob_bottom.channels, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.height, m_blob_bottom.height, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.width, m_blob_bottom.width, "The num does not match!");

                m_blobVal.LoadFromNumpy(strTestDataPath + "ln.8_y.npy");

                float fErr = 2e-6f;
                verify(m_blob_top, m_blobVal, false, fErr);

                sw.Start();
                for (int i = 0; i < 100; i++)
                {
                    layer.Forward(BottomVec, TopVec);
                }
                sw.Stop();

                double dfTime = sw.Elapsed.TotalMilliseconds / 100.0;
                Trace.WriteLine("Time Per Forward = " + dfTime.ToString("N6") + " ms");
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestBackwardEx(bool bUseCuda)
        {
            string strTestDataPath = loadTestData();
            Layer<T> layer = null;

            try
            {
                Stopwatch sw = new Stopwatch();

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.layer_norm_param.enable_cuda_impl = bUseCuda;
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.LAYERNORM, "The layer type is incorrect!");

                m_blob_bottom.LoadFromNumpy(strTestDataPath + "q0.npy");
                m_blob_top.SetData(0);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_blob_top.LoadFromNumpy(strTestDataPath + "grad_ln.8_y.npy", true);

                List<bool> rgProp = new List<bool>() { true };
                layer.Backward(TopVec, rgProp, BottomVec);

                m_log.CHECK_EQ(m_blob_top.num, m_blob_bottom.num, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.channels, m_blob_bottom.channels, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.height, m_blob_bottom.height, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.width, m_blob_bottom.width, "The num does not match!");

                m_blobVal.LoadFromNumpy(strTestDataPath + "grad_ln.1_x.npy", true);

                float fErr = 1e-8f;
                verify(m_blob_bottom, m_blobVal, true, fErr);

                sw.Start();
                for (int i = 0; i < 100; i++)
                {
                    layer.Backward(TopVec, rgProp, BottomVec);
                }
                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100.0;
                Trace.WriteLine("Time Per Backward = " + dfTime.ToString("N6") + " ms");
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
