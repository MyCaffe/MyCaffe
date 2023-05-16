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
/// Testing the TransformInputs.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_TransformInputs
    {
        [TestMethod]
        public void TestForward()
        {
            TransformInputsTest test = new TransformInputsTest();

            try
            {
                foreach (ITransformInputsTest t in test.Tests)
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
            TransformInputsTest test = new TransformInputsTest();

            try
            {
                foreach (ITransformInputsTest t in test.Tests)
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

    interface ITransformInputsTest : ITest
    {
        void TestForward();
        void TestBackward();
    }

    class TransformInputsTest : TestBase
    {
        public TransformInputsTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT TransformInputs Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TransformInputsTest<double>(strName, nDeviceID, engine);
            else
                return new TransformInputsTest<float>(strName, nDeviceID, engine);
        }
    }

    class TransformInputsTest<T> : TestEx<T>, ITransformInputsTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public TransformInputsTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\hist_ts_transform\\";
        }

        private string buildModel(int nNumSamples, int nStateSize, int nNumHistSteps, int nNumFutureSteps,
            int nNumStaticNumeric, int nNumStaticCategorical, List<int> rgStaticCardinalities,
            int nNumHistNumeric, int nNumHistCategorical, List<int> rgHistCardinalities,
            int nNumFutureNumeric, int nNumFutureCategorical, List<int> rgFutureCardinalities)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticNumeric }));                     // selected_historical
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticCategorical }));                 // selected_historical
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, nNumHistNumeric}));         // selected_future
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, nNumHistCategorical }));    // selected_future
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, nNumFutureNumeric }));    // c_seq_hidden
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, nNumFutureCategorical }));// c_seq_cell
            input.top.Add("x_numeric_static");
            input.top.Add("x_categorical_static");
            input.top.Add("x_numeric_hist");
            input.top.Add("x_categorical_hist");
            input.top.Add("x_numeric_future");
            input.top.Add("x_categorical_future");
            p.layer.Add(input);

            //---------------------------------
            //  Input Transformations
            //---------------------------------
            LayerParameter static_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "static_trfm");
            static_transform.numeric_trans_param.num_input = (uint)nNumStaticNumeric;
            static_transform.numeric_trans_param.state_size = (uint)nStateSize;
            static_transform.categorical_trans_param.num_input = (uint)nNumStaticCategorical;
            static_transform.categorical_trans_param.cardinalities = rgStaticCardinalities;
            static_transform.categorical_trans_param.state_size = (uint)nStateSize;
            static_transform.bottom.Add("x_numeric_static");
            static_transform.bottom.Add("x_categorical_static");
            static_transform.top.Add("static_rep");
            p.layer.Add(static_transform);

            LayerParameter hist_ts_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "hist_ts_trfm");
            hist_ts_transform.numeric_trans_param.num_input = (uint)nNumHistNumeric;
            hist_ts_transform.numeric_trans_param.state_size = (uint)nStateSize;
            hist_ts_transform.categorical_trans_param.num_input = (uint)nNumHistCategorical;
            hist_ts_transform.categorical_trans_param.cardinalities = rgHistCardinalities;
            hist_ts_transform.categorical_trans_param.state_size = (uint)nStateSize;
            hist_ts_transform.bottom.Add("x_numeric_hist");
            hist_ts_transform.bottom.Add("x_categorical_hist");
            hist_ts_transform.top.Add("hist_ts_rep");
            p.layer.Add(hist_ts_transform);

            LayerParameter future_ts_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "future_ts_trfm");
            future_ts_transform.numeric_trans_param.num_input = (uint)nNumFutureNumeric;
            future_ts_transform.numeric_trans_param.state_size = (uint)nStateSize;
            future_ts_transform.categorical_trans_param.num_input = (uint)nNumFutureCategorical;
            future_ts_transform.categorical_trans_param.cardinalities = rgFutureCardinalities;
            future_ts_transform.categorical_trans_param.state_size = (uint)nStateSize;
            future_ts_transform.bottom.Add("x_numeric_future");
            future_ts_transform.bottom.Add("x_categorical_future");
            future_ts_transform.top.Add("future_ts_rep");
            p.layer.Add(future_ts_transform);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the backward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data run the following:
        /// Code: test_1_transforminputs.py
        /// Path: ti
        /// Base: test\iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            string strPath = getTestDataPath("ti");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumHistSteps = 90;
            int nNumFutureSteps = 30;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            List<int> rgStaticCardinalities = new List<int>() { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            List<int> rgHistCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            List<int> rgFutureCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                for (int i=0; i<nNumStaticCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.static.categorical_transform.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumHistNumeric; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.historical.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.historical.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumHistCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.historical.categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumFutureNumeric; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.future.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.future.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumFutureCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.future.categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("x_numeric_static");
                //blob1.LoadFromNumpy(strPath + "tft.ti.static.x_numeric.npy");
                blob1 = net.FindBlob("x_categorical_static");
                blob1.LoadFromNumpy(strPath + "tft.ti.static.x_categorical.npy");

                blob1 = net.FindBlob("x_numeric_hist");
                blob1.LoadFromNumpy(strPath + "tft.ti.historical.x_numeric.npy");
                blob1 = net.FindBlob("x_categorical_hist");
                blob1.LoadFromNumpy(strPath + "tft.ti.historical.x_categorical.npy");

                blob1 = net.FindBlob("x_numeric_future");
                blob1.LoadFromNumpy(strPath + "tft.ti.future.x_numeric.npy");
                blob1 = net.FindBlob("x_categorical_future");
                blob1.LoadFromNumpy(strPath + "tft.ti.future.x_categorical.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.ti.static.merged_transformations.npy");
                blob1 = net.FindBlob("static_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.ti.historical.merged_transformations.npy");
                blob1 = net.FindBlob("hist_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 6e-08), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.ti.future.merged_transformations.npy");
                blob1 = net.FindBlob("future_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The blobs are different!");
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
        /// Test the forward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data run the following:
        /// Code: test_1_transforminputs.py
        /// Path: ti
        /// Base: test\iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            string strPath = getTestDataPath("ti");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumHistSteps = 90;
            int nNumFutureSteps = 30;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            List<int> rgStaticCardinalities = new List<int>() { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            List<int> rgHistCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            List<int> rgFutureCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                for (int i = 0; i < nNumStaticCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.static.categorical_transform.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumHistNumeric; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.historical.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.historical.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumHistCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.historical.categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumFutureNumeric; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.future.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.future.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                    nIdx++;
                }

                for (int i = 0; i < nNumFutureCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.ti.future.categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("x_numeric_static");
                //blob1.LoadFromNumpy(strPath + "tft.static.x_numeric.npy");
                blob1 = net.FindBlob("x_categorical_static");
                blob1.LoadFromNumpy(strPath + "tft.ti.static.x_categorical.npy");

                blob1 = net.FindBlob("x_numeric_hist");
                blob1.LoadFromNumpy(strPath + "tft.ti.historical.x_numeric.npy");
                blob1 = net.FindBlob("x_categorical_hist");
                blob1.LoadFromNumpy(strPath + "tft.ti.historical.x_categorical.npy");

                blob1 = net.FindBlob("x_numeric_future");
                blob1.LoadFromNumpy(strPath + "tft.ti.future.x_numeric.npy");
                blob1 = net.FindBlob("x_categorical_future");
                blob1.LoadFromNumpy(strPath + "tft.ti.future.x_categorical.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.ti.static.merged_transformations.npy");
                blob1 = net.FindBlob("static_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.ti.historical.merged_transformations.npy");
                blob1 = net.FindBlob("hist_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 6e-08), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.ti.future.merged_transformations.npy");
                blob1 = net.FindBlob("future_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("static_rep");
                blob1.LoadFromNumpy(strPath + "tft.ti.static.merged_transformations.grad.npy", true);

                blob1 = net.FindBlob("hist_ts_rep");
                blob1.LoadFromNumpy(strPath + "tft.ti.historical.merged_transformations.grad.npy", true);

                blob1 = net.FindBlob("future_ts_rep");
                blob1.LoadFromNumpy(strPath + "tft.ti.future.merged_transformations.grad.npy", true);

                net.Backward();
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
