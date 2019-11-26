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
using System.Drawing;
using System.Collections;
using MyCaffe.data;
using System.Diagnostics;
using System.IO;
using System.Drawing.Imaging;
using MyCaffe.extras;
using System.Threading;
using System.Net;

namespace MyCaffe.test
{
    [TestClass]
    public class TestNeuralStyleTransfer
    {
        [TestMethod]
        public void TestNeuralStyleTransfer1()
        {
            NeuralStyleTransferTest test = new NeuralStyleTransferTest();

            try
            {
                foreach (INeuralStyleTransferTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        t.TestNeuralStyleTransfer(null, null, 20, 10, null, "vgg19", "LBFGS", 1.5, 0, 640, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNeuralStyleTransfer1NoIntermediate()
        {
            NeuralStyleTransferTest test = new NeuralStyleTransferTest();

            try
            {
                foreach (INeuralStyleTransferTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        t.TestNeuralStyleTransfer(null, null, 20, 20, null, "vgg19", "LBFGS", 1.5, 0, 640, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNeuralStyleTransferPartial()
        {
            NeuralStyleTransferTest test = new NeuralStyleTransferTest();

            try
            {
                foreach (INeuralStyleTransferTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        t.TestNeuralStyleTransfer(null, null, 20, 10, null, "vgg19", "LBFGS", 1.5, 0, 640, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNeuralStyleTransferPartialNoIntermediate()
        {
            NeuralStyleTransferTest test = new NeuralStyleTransferTest();

            try
            {
                foreach (INeuralStyleTransferTest t in test.Tests)
                {
                    if (t.DataType == DataType.DOUBLE)
                        t.TestNeuralStyleTransfer(null, null, 20, 20, null, "vgg19", "LBFGS", 1.5, 0, 640, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface INeuralStyleTransferTest : ITest
    {
        string TestNeuralStyleTransfer(string strStyleImg, string strContentImg, int nIteration, int nIntermediateOutput, string strResultDir, string strModelName, string strSolverType, double dfLearningRate, double dfTvLoss = 0, int nMaxImageSize = 640, bool bEnablePartial = false);
    }

    class NeuralStyleTransferTest : TestBase
    {
        public NeuralStyleTransferTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Neural Style Transfer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NeuralStyleTransferTest<double>(strName, nDeviceID, engine);
            else
                return new NeuralStyleTransferTest<float>(strName, nDeviceID, engine);
        }
    }

    public class NeuralStyleTransferTest<T> : TestEx<T>, INeuralStyleTransferTest
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();
        MyCaffeControl<T> m_caffe = null;
        string m_strResultDir = null;
        bool m_bProcessing = false;
        Stopwatch m_swDownload = new Stopwatch();
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);
        TestingProgressSet m_testingProgress = new TestingProgressSet();

        public NeuralStyleTransferTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_settings.GpuIds = nDeviceID.ToString();
            m_caffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            m_log.EnableTrace = true;
        }

        protected override void dispose()
        {
            m_testingProgress.Dispose();
            m_caffe.Dispose();
            base.dispose();
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private Tuple<string, string, string> getModel(string strName)
        {
            string strModelFile;
            string strWtsFile;

            switch (strName)
            {
                case "vgg19":
                    strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\vgg\\vgg19\\neuralstyle\\deploy.prototxt");
                    strWtsFile = getTestPath("\\MyCaffe\\test_data\\models\\vgg\\vgg19\\neuralstyle\\", true) + "weights.caffemodel";
                    break;

                case "googlenet":
                    strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\googlenet\\neuralstyle\\deploy.prototxt");
                    strWtsFile = getTestPath("\\MyCaffe\\test_data\\models\\googlenet\\neuralstyle\\weights.mycaffemodel");
                    break;

                default:
                    throw new Exception("Unknown model name '" + strName + "'");
            }

            return new Tuple<string, string, string>(strName, strModelFile, strWtsFile);
        }

        private SolverParameter.SolverType getSolverType(string strType)
        {
            switch (strType)
            {
                case "LBFGS":
                    return SolverParameter.SolverType.LBFGS;

                case "ADAM":
                    return SolverParameter.SolverType.ADAM;

                case "RMSPROP":
                    return SolverParameter.SolverType.RMSPROP;

                case "SGD":
                    return SolverParameter.SolverType.SGD;

                default:
                    throw new Exception("The solver type '" + strType + "' is not supported at this time.");
            }
        }

        private string getFileName(string strFile)
        {
            if (File.Exists(strFile + ".jpg"))
                return strFile + ".jpg";

            return strFile + ".png";
        }

        private Exception downloadFile(string strUrl, string strFile)
        {
            try
            {
                WebClient webClient = new WebClient();

                m_swDownload.Start();

                webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                webClient.DownloadFileAsync(new Uri(strUrl), strFile);

                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.AddRange(m_evtCancel.Handles);
                rgWait.Add(m_evtDownloadDone);

                int nWait = WaitHandle.WaitAny(rgWait.ToArray());

                if (nWait < rgWait.Count - 1)
                {
                    webClient.CancelAsync();

                    if (File.Exists(strFile))
                        File.Delete(strFile);

                    return new Exception("Download Aborted!");
                }
            }
            catch (Exception excpt)
            {
                return excpt;
            }

            return null;
        }

        private void WebClient_DownloadFileCompleted(object sender, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            m_evtDownloadDone.Set();
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_bProcessing)
                return;

            try
            {
                m_bProcessing = true;

                if (m_swDownload.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = (double)e.BytesReceived / (double)e.TotalBytesToReceive;
                    m_testingProgress.SetProgress(dfPct);
                    string strMsg = "(" + dfPct.ToString("P") + ") Downloading weight file...";
                    m_log.WriteLine(strMsg);

                    m_swDownload.Restart();
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_bProcessing = false;
            }
        }


        /// <summary>
        /// The NeuralStyleTransfer test is based on the the open-source Neural Style Transfer algorithm
        /// from https://github.com/ftokarev/caffe-neural-style/blob/master/neural-style.py, but has been
        /// re-written and modified using MyCaffe.  See LICENSE file for details.
        /// </summary>
        /// <param name="strStyleImg">Specifies the path to the style image file, or <i>null</i> which then uses the default at '/programdata/mycaffe/test_data/data/images/style/style.png'.</param>
        /// <param name="strContentImg">Specifies the path to the content image file, or <i>null</i> which then uses the default at '/programdata/mycaffe/test_data/data/images/content/content.png'.</param>
        /// <param name="nIterations">Specifies the number of iterations to learn the style.</param>
        /// <param name="nIntermediateOutput">Optionally, specifies the number of iterations to output an intermediate image, or 0 to ignore.</param>
        /// <param name="strResultDir">Specifies the path where all image results are placed.</param>
        /// <param name="strName">Optionally, specifies the name of the model ('vgg19', 'googlenet') to use (default = 'vgg19').</param>
        /// <param name="strSolverType">Optionally, specifies the solver type ('LBFGS', 'ADAM', 'RMSPROP', 'SGD') to use (default = 'LBFGS').</param>
        /// <param name="dfLearningRate">Optionally, specifies the learning rate to use (default = 1.0).</param>
        /// <param name="dfTvLoss">Optionally, specifies the TVLoss weights which acts as a smoothing factor to use (default = 0, which disables the TVLoss).</param>
        /// <param name="nMaxSize">Optionally, specifies the maximum image size - if you run out of memory when performing neural style, reduce this size.</param>
        /// <param name="bEnablePartial">When enabled, the partial solution functionality is tested.</param>
        /// <returns>The name of the resulting file is returned.</returns>
        public string TestNeuralStyleTransfer(string strStyleImg, string strContentImg, int nIterations, int nIntermediateOutput, string strResultDir, string strName, string strSolverType = "LBFGS", double dfLearningRate = 1.0, double dfTvLoss = 0, int nMaxSize = 640, bool bEnablePartial = false)
        {
            CancelEvent evtCancel = new CancelEvent();
            SolverParameter.SolverType solverType = getSolverType(strSolverType);
            Tuple<string, string, string> info = getModel(strName);
            string strModelName = info.Item1;
            string strModelFile = info.Item2;
            string strWeightFile = info.Item3;
            string strDataDir = getTestPath("\\MyCaffe\\test_data\\data\\images\\", true);
            byte[] rgWeights = null;
            string strModelDesc = "";
            bool bLogTrace = m_log.EnableTrace;
            bool bDownloadNeeded = false;

            if (strModelName == "vgg19")
            {
                // If the weight file exists but has a size of zero (from an aborted download)
                // delete the file and re-download it.
                if (File.Exists(strWeightFile))
                {
                    FileInfo fi = new FileInfo(strWeightFile);
                    if (fi.Length < 574671192L)
                        File.Delete(strWeightFile);
                }

                if (!File.Exists(strWeightFile))
                    bDownloadNeeded = true;
            }

            if (bDownloadNeeded)
            {
                string strWeightsUrl = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel";
                m_log.EnableTrace = true;
                m_log.WriteLine("Downloading weight file for test from '" + strWeightsUrl + "'...");
                downloadFile(strWeightsUrl, strWeightFile);
                m_log.EnableTrace = bLogTrace;
            }

            if (string.IsNullOrEmpty(strStyleImg))
                strStyleImg = getFileName(strDataDir + "style\\starry_night");

            if (string.IsNullOrEmpty(strContentImg))
                strContentImg = getFileName(strDataDir + "content\\johannesburg");

            if (string.IsNullOrEmpty(strResultDir))
                strResultDir = strDataDir + "result\\";

            if (File.Exists(strWeightFile))
            {
                using (FileStream fs = new FileStream(strWeightFile, FileMode.Open, FileAccess.Read))
                {
                    using (BinaryReader br = new BinaryReader(fs))
                    {
                        rgWeights = br.ReadBytes((int)fs.Length);
                    }
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            m_testingProgress.SetProgress(0);

            NeuralStyleTransfer<T> ns = new NeuralStyleTransfer<T>(m_cuda, m_log, m_evtCancel, strModelName, strModelDesc, rgWeights, false, solverType, dfLearningRate);

            if (!bEnablePartial)
                ns.OnIntermediateOutput += Ns_OnIntermediateOutput;

            m_strResultDir = strResultDir;

            if (!Directory.Exists(strResultDir))
                Directory.CreateDirectory(strResultDir);

            Bitmap bmpStyle = new Bitmap(strStyleImg);
            Bitmap bmpContent = new Bitmap(strContentImg);
            Bitmap bmpResult = ns.Process(bmpStyle, bmpContent, nIterations, nIntermediateOutput, dfTvLoss, nMaxSize, bEnablePartial);
            Bitmap bmpIntermediate = null;
            int nIntermediateIdx = 0;
            int nIdx = 0;

            while (bEnablePartial && bmpResult == null && !m_evtCancel.WaitOne(0))
            {
                bmpResult = ns.ProcessNext(out bmpIntermediate, out nIntermediateIdx);
                if (bmpIntermediate != null)
                {
                    string strIntermediateFile = m_strResultDir + "\\tmp_" + nIdx.ToString() + ".png";
                    bmpIntermediate.Save(strIntermediateFile, ImageFormat.Png);
                }

                nIdx++;
            }

            string strContent = Path.GetFileNameWithoutExtension(strContentImg);
            string strStyle = Path.GetFileNameWithoutExtension(strStyleImg);

            string strResultFile = strResultDir + "\\" + nIterations.ToString() + "_" + strContent + "_" + strStyle + ".png";

            if (File.Exists(strResultFile))
                File.Delete(strResultFile);

            bmpResult.Save(strResultFile, ImageFormat.Png);
            bmpResult.Dispose();

            return strResultFile;
        }

        private void Ns_OnIntermediateOutput(object sender, NeuralStyleIntermediateOutputArgs e)
        {
            if (!string.IsNullOrEmpty(m_strResultDir))
            {
                string strIntermediateFile = m_strResultDir + "\\tmp_" + e.Iteration.ToString() + ".png";
                e.Image.Save(strIntermediateFile, ImageFormat.Png);
            }
        }
    }
}
