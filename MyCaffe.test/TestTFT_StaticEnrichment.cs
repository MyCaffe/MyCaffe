﻿using System;
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
using MyCaffe.param.tft;
using System.IO;

/// <summary>
/// Testing the StaticEnrichment.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_StaticEnrichment
    {
        [TestMethod]
        public void TestForward()
        {
            StaticEnrichmentTest test = new StaticEnrichmentTest();

            try
            {
                foreach (IStaticEnrichmentTest t in test.Tests)
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
            StaticEnrichmentTest test = new StaticEnrichmentTest();

            try
            {
                foreach (IStaticEnrichmentTest t in test.Tests)
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

    interface IStaticEnrichmentTest : ITest
    {
        void TestForward();
        void TestBackward();
    }

    class StaticEnrichmentTest : TestBase
    {
        public StaticEnrichmentTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT StaticEnrichment Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new StaticEnrichmentTest<double>(strName, nDeviceID, engine);
            else
                return new StaticEnrichmentTest<float>(strName, nDeviceID, engine);
        }
    }

    class StaticEnrichmentTest<T> : TestEx<T>, IStaticEnrichmentTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public StaticEnrichmentTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        private void verifyFileDownload(string strSubPath, string strFile)
        {
            string strPath = getTestDataPath(strSubPath);
            if (!File.Exists(strPath + strFile))
                throw new Exception("ERROR: You need to download the TFT test data by running the MyCaffe Test Application and selecting the 'Download Test Data | TFT' menu.");
        }

        private string buildModel(int nNumSamples, int nNumHist, int nNumFuture, float fDropout, int nStateSize)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";


            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHist + nNumFuture, nStateSize }));  // gated_lstm_output
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));                         // c_enrichment
            input.top.Add("gated_lstm_output");
            input.top.Add("c_enrichment");
            p.layer.Add(input);

            //---------------------------------
            //  Static enrichment
            //---------------------------------
            LayerParameter glstmout_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "glstmout_split");
            glstmout_split.bottom.Add("gated_lstm_output");
            glstmout_split.top.Add("glstmout_a");
            glstmout_split.top.Add("glstmout_b");
            p.layer.Add(glstmout_split);

            LayerParameter static_enrich_grn_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_statenr_a");
            static_enrich_grn_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
            static_enrich_grn_reshape_before.bottom.Add("glstmout_a");
            static_enrich_grn_reshape_before.bottom.Add("c_enrichment");
            static_enrich_grn_reshape_before.top.Add("gated_lstm_output1");
            static_enrich_grn_reshape_before.top.Add("c_enrichment1");
            p.layer.Add(static_enrich_grn_reshape_before);

            LayerParameter static_enrich_grn = new LayerParameter(LayerParameter.LayerType.GRN, "static_enrich_gru");
            static_enrich_grn.grn_param.input_dim = nStateSize;
            static_enrich_grn.grn_param.hidden_dim = nStateSize;
            static_enrich_grn.grn_param.output_dim = nStateSize;
            static_enrich_grn.grn_param.context_dim = nStateSize;
            static_enrich_grn.grn_param.dropout_ratio = fDropout;
            static_enrich_grn.bottom.Add("gated_lstm_output1");
            static_enrich_grn.bottom.Add("c_enrichment1");
            static_enrich_grn.top.Add("enriched_sequence1a");
            p.layer.Add(static_enrich_grn);

            LayerParameter static_enrich_grn_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_statenr_b");
            static_enrich_grn_reshape_after.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.AFTER;
            static_enrich_grn_reshape_after.bottom.Add("enriched_sequence1a");
            static_enrich_grn_reshape_after.top.Add("enriched_sequence");
            p.layer.Add(static_enrich_grn_reshape_after);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test static enrichment focused forward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_5_static_enrichment_focused.py
        /// Target Dir: statenr
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            string strPath = getTestDataPath("statenr");
            string strPathWt = getTestWtsPath("static_enrichment_grn");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;

            verifyFileDownload("statenr", "tft.statenr.gated_lstm_output.ase.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, fDropout, nStateSize);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("gated_lstm_output");
                blob1.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.ase.npy");
                blob1 = net.FindBlob("c_enrichment");
                blob1.LoadFromNumpy(strPath + "tft.statenr.static_enrichment_signal.ase.npy");

                net.parameters[0].LoadFromNumpy(strPath + "tft.stateenr.fc1.module.weight.npy");
                net.parameters[1].LoadFromNumpy(strPath + "tft.stateenr.fc1.module.bias.npy");
                net.parameters[2].LoadFromNumpy(strPath + "tft.stateenr.context_projection.module.weight.npy");
                net.parameters[3].LoadFromNumpy(strPath + "tft.stateenr.fc2.module.weight.npy");
                net.parameters[4].LoadFromNumpy(strPath + "tft.stateenr.fc2.module.bias.npy");
                net.parameters[5].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc1.weight.npy");
                net.parameters[6].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc1.bias.npy");
                net.parameters[7].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc2.weight.npy");
                net.parameters[8].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc2.bias.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.statenr.flattened_gated_lstm_output.ase.npy");
                blob1 = net.FindBlob("gated_lstm_output1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.time_distributed_context1.ase.npy");
                blob1 = net.FindBlob("c_enrichment1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence1.ase.npy");
                blob1 = net.FindBlob("enriched_sequence1a");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence.ase.npy");
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 2e-06), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test static enrichment focused backward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_5_static_enrichment_focused.py
        /// Target Dir: statenr
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            string strPath = getTestDataPath("statenr");
            string strPathWt = getTestWtsPath("static_enrichment_grn");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;

            verifyFileDownload("statenr", "tft.statenr.gated_lstm_output.ase.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, fDropout, nStateSize);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("gated_lstm_output");
                blob1.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.ase.npy");
                blob1 = net.FindBlob("c_enrichment");
                blob1.LoadFromNumpy(strPath + "tft.statenr.static_enrichment_signal.ase.npy");

                net.parameters[0].LoadFromNumpy(strPath + "tft.stateenr.fc1.module.weight.npy");
                net.parameters[1].LoadFromNumpy(strPath + "tft.stateenr.fc1.module.bias.npy");
                net.parameters[2].LoadFromNumpy(strPath + "tft.stateenr.context_projection.module.weight.npy");
                net.parameters[3].LoadFromNumpy(strPath + "tft.stateenr.fc2.module.weight.npy");
                net.parameters[4].LoadFromNumpy(strPath + "tft.stateenr.fc2.module.bias.npy");
                net.parameters[5].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc1.weight.npy");
                net.parameters[6].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc1.bias.npy");
                net.parameters[7].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc2.weight.npy");
                net.parameters[8].LoadFromNumpy(strPath + "tft.stateenr.gate.module.fc2.bias.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.statenr.flattened_gated_lstm_output.ase.npy");
                blob1 = net.FindBlob("gated_lstm_output1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.time_distributed_context1.ase.npy");
                blob1 = net.FindBlob("c_enrichment1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence1.ase.npy");
                blob1 = net.FindBlob("enriched_sequence1a");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence.ase.npy");
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 2e-06), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("enriched_sequence");
                blob1.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence.ase.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.val.grad.npy", true);
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.c_enrichment.val.grad.npy", true);
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }
    }
}
