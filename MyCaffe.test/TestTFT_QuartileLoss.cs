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
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.tft;

/// <summary>
/// Testing the QuartileLoss.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_QuartileLoss
    {
        [TestMethod]
        public void TestForward()
        {
            QuartileLossTest test = new QuartileLossTest();

            try
            {
                foreach (IQuartileLossTest t in test.Tests)
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
        public void TestBackward()
        {
            QuartileLossTest test = new QuartileLossTest();

            try
            {
                foreach (IQuartileLossTest t in test.Tests)
                {
                    t.TestBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IQuartileLossTest : ITest
    {
        void TestForward();
        void TestBackward();
    }

    class QuartileLossTest : TestBase
    {
        public QuartileLossTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT QuartileLoss Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new QuartileLossTest<double>(strName, nDeviceID, engine);
            else
                return new QuartileLossTest<float>(strName, nDeviceID, engine);
        }
    }

    class QuartileLossTest<T> : TestEx<T>, IQuartileLossTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public QuartileLossTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private string getTestDataPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\test\\" + strSubPath + "\\iter_0\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\favorita\\weights\\" + strSubPath + "\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\static_enrichment_grn\\";
        }

        private string buildModel(int nNumSamples, int nNumFuture)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";


            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFuture, 3 }));  // output
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFuture }));     // target
            input.top.Add("output");
            input.top.Add("target");
            p.layer.Add(input);

            //---------------------------------
            //  Quartile Loss
            //---------------------------------
            LayerParameter loss = new LayerParameter(LayerParameter.LayerType.QUANTILE_LOSS, "loss");
            loss.quantile_loss_param.desired_quantiles.Add(0.1f);
            loss.quantile_loss_param.desired_quantiles.Add(0.5f);
            loss.quantile_loss_param.desired_quantiles.Add(0.9f);
            loss.loss_weight.Add(1); // for loss
            loss.loss_weight.Add(1); // for q_risk
            loss.bottom.Add("output");
            loss.bottom.Add("target");
            loss.top.Add("loss");
            loss.top.Add("q_risk");
            p.layer.Add(loss);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test loss forward.
        /// </summary>
        /// <remarks>
        /// To generate the test data run the following:
        /// 
        /// Code: test_10_loss_focused.py
        /// Path: loss
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            string strPath = getTestDataPath("loss");
            string strPathWt = getTestWtsPath("static_enrichment_grn");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 16;
            int nNumFuture = 30;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumFuture);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("output");
                blob1.LoadFromNumpy(strPath + "outputs.npy");
                blob1 = net.FindBlob("target");
                blob1.LoadFromNumpy(strPath + "targets.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "q_loss.npy");
                blob1 = net.FindBlob("loss");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "q_risk.npy");
                blob1 = net.FindBlob("q_risk");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 7e-07), "The blobs are different!");
            }
            catch (Exception ex)
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test loss backward.
        /// </summary>
        /// <remarks>
        /// To generate the test data run the following:
        /// 
        /// Code: test_10_loss_focused.py
        /// Path: loss
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            string strPath = getTestDataPath("loss");
            string strPathWt = getTestWtsPath("static_enrichment_grn");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 16;
            int nNumFuture = 30;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumFuture);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("output");
                blob1.LoadFromNumpy(strPath + "outputs.npy");
                blob1 = net.FindBlob("target");
                blob1.LoadFromNumpy(strPath + "targets.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "q_loss.npy");
                blob1 = net.FindBlob("loss");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "q_risk.npy");
                blob1 = net.FindBlob("q_risk");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 7e-07), "The blobs are different!");

                //*** BACKWARD ***
                blob1 = net.FindBlob("loss");
                blob1.LoadFromNumpy(strPath + "q_loss.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "outputs.grad.npy", true);
                blob1 = net.FindBlob("output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The blobs are different!");
            }
            catch (Exception ex)
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }
    }
}
