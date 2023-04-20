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

        private string getTestDataPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\iter_0\\";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\static_enrichment_grn\\";
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
            LayerParameter static_enrich_grn_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_statenr_a");
            static_enrich_grn_reshape_before.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            static_enrich_grn_reshape_before.bottom.Add("gated_lstm_output");
            static_enrich_grn_reshape_before.bottom.Add("c_enrichment");
            static_enrich_grn_reshape_before.top.Add("gated_lstm_output1");
            static_enrich_grn_reshape_before.top.Add("c_enrichment1");
            p.layer.Add(static_enrich_grn_reshape_before);

            LayerParameter static_enrich_grn = new LayerParameter(LayerParameter.LayerType.GRN, "static_enrich_gru");
            static_enrich_grn.grn_param.input_dim = nStateSize;
            static_enrich_grn.grn_param.hidden_dim = nStateSize;
            static_enrich_grn.grn_param.output_dim = nStateSize;
            static_enrich_grn.grn_param.context_dim = nStateSize;
            static_enrich_grn.grn_param.dropout = fDropout;
            static_enrich_grn.bottom.Add("gated_lstm_output1");
            static_enrich_grn.bottom.Add("c_enrichment1");
            static_enrich_grn.top.Add("enriched_sequence1");
            p.layer.Add(static_enrich_grn);

            LayerParameter static_enrich_grn_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_statenr_b");
            static_enrich_grn_reshape_after.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            static_enrich_grn_reshape_after.bottom.Add("enriched_sequence1");
            static_enrich_grn_reshape_after.top.Add("enriched_sequence");
            p.layer.Add(static_enrich_grn_reshape_after);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the forward pass for static enrichment
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_5_static_enrichment_focused.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForward()
        {
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, fDropout, nStateSize);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("gated_lstm_output");
                blob1.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.npy");
                blob1 = net.FindBlob("c_enrichment");
                blob1.LoadFromNumpy(strPath + "tft.statenr.static_enrichment_signal.npy");

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

                blobVal.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.npy");
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.static_enrichment_signal.npy");
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.flattened_gated_lstm_output.npy");
                blob1 = net.FindBlob("gated_lstm_output1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.time_distributed_context.npy");
                blob1 = net.FindBlob("c_enrichment1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence1.ase.npy");
                blob1 = net.FindBlob("enriched_sequence1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 8e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence.ase.npy");
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 8e-07), "The blobs are different!");
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
        /// Test the backward pass for static enrichment
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_5_static_enrichment_focused.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestBackward()
        {
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;

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
                blob1.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.npy");
                blob1 = net.FindBlob("c_enrichment");
                blob1.LoadFromNumpy(strPath + "tft.statenr.static_enrichment_signal.npy");

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

                blobVal.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.npy");
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.static_enrichment_signal.npy");
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.flattened_gated_lstm_output.npy");
                blob1 = net.FindBlob("gated_lstm_output1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.time_distributed_context.npy");
                blob1 = net.FindBlob("c_enrichment1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence1.ase.npy");
                blob1 = net.FindBlob("enriched_sequence1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 8e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence.ase.npy");
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 8e-07), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("enriched_sequence");
                blob1.LoadFromNumpy(strPath + "tft.statenr.enriched_sequence.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "tft.statenr.gated_lstm_output.val.grad.npy", true);
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 3e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.statenr.c_enrichment.val.grad.npy", true);
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 3e-05), "The blobs are different!");
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
