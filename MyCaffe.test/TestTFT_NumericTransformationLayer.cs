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
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Tab;

/// <summary>
/// Testing the Numeric Transformation layer.
/// 
/// NumericTransformation Layer - layer converts numerical inputs into embeddings using inner product layers.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_NumericTransformationLayer
    {
        [TestMethod]
        public void TestForward()
        {
            NumericTransformationLayerTest test = new NumericTransformationLayerTest();

            try
            {
                foreach (INumericTransformationLayerTest t in test.Tests)
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
            NumericTransformationLayerTest test = new NumericTransformationLayerTest();

            try
            {
                foreach (INumericTransformationLayerTest t in test.Tests)
                {
                    t.TestBackward();
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
            NumericTransformationLayerTest test = new NumericTransformationLayerTest();

            try
            {
                foreach (INumericTransformationLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface INumericTransformationLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
    }

    class NumericTransformationLayerTest : TestBase
    {
        public NumericTransformationLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("NumericTransformation Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NumericTransformationLayerTest<double>(strName, nDeviceID, engine);
            else
                return new NumericTransformationLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class NumericTransformationLayerTest<T> : TestEx<T>, INumericTransformationLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public NumericTransformationLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\hist_ts_transform\\";
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY0 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobY3 = null;
            Blob<T> blobProcessedNumeric0 = null;
            Blob<T> blobProcessedNumeric1 = null;
            Blob<T> blobProcessedNumeric2 = null;
            Blob<T> blobProcessedNumeric3 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY0 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobY3 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric0 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric1 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric2 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric3 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_numeric.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY0);
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);
                TopVec.Add(blobY3);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessedNumeric0.LoadFromNumpy(strPath + "processed_numeric_0.npy");
                blobProcessedNumeric1.LoadFromNumpy(strPath + "processed_numeric_1.npy");
                blobProcessedNumeric2.LoadFromNumpy(strPath + "processed_numeric_2.npy");
                blobProcessedNumeric3.LoadFromNumpy(strPath + "processed_numeric_3.npy");
                BlobCollection<T> col = new BlobCollection<T>() {  blobProcessedNumeric0, blobProcessedNumeric1, blobProcessedNumeric2, blobProcessedNumeric3 };
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2.5e-07;

                for (int i = 0; i < p.numeric_trans_param.num_input; i++)
                {
                    m_log.CHECK(TopVec[i].Compare(col[i], blobWork, false, dfErr), "The blobs do not match.");
                }
            }
            finally
            {
                dispose(ref blobProcessedNumeric0);
                dispose(ref blobProcessedNumeric1);
                dispose(ref blobProcessedNumeric2); 
                dispose(ref blobProcessedNumeric3);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY0);
                dispose(ref blobY1);
                dispose(ref blobY2);
                dispose(ref blobY3);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX = null;
            Blob<T> blobY0 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobY3 = null;
            Blob<T> blobProcessedNumeric0 = null;
            Blob<T> blobProcessedNumeric1 = null;
            Blob<T> blobProcessedNumeric2 = null;
            Blob<T> blobProcessedNumeric3 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY0 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobY3 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric0 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric1 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric2 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric3 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_numeric.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY0);
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);
                TopVec.Add(blobY3);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessedNumeric0.LoadFromNumpy(strPath + "processed_numeric_0.npy");
                blobProcessedNumeric1.LoadFromNumpy(strPath + "processed_numeric_1.npy");
                blobProcessedNumeric2.LoadFromNumpy(strPath + "processed_numeric_2.npy");
                blobProcessedNumeric3.LoadFromNumpy(strPath + "processed_numeric_3.npy");
                BlobCollection<T> col = new BlobCollection<T>() { blobProcessedNumeric0, blobProcessedNumeric1, blobProcessedNumeric2, blobProcessedNumeric3 };
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2.5e-07;

                for (int i = 0; i < p.numeric_trans_param.num_input; i++)
                {
                    m_log.CHECK(TopVec[i].Compare(col[i], blobWork, false, dfErr), "The blobs do not match.");
                }

                TopVec[0].LoadFromNumpy(strPath + "grad_processed_numeric_0.npy", true);
                TopVec[1].LoadFromNumpy(strPath + "grad_processed_numeric_1.npy", true);
                TopVec[2].LoadFromNumpy(strPath + "grad_processed_numeric_2.npy", true);
                TopVec[3].LoadFromNumpy(strPath + "grad_processed_numeric_3.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobGradExp.LoadFromNumpy(strPath + "grad_x_numeric.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX, blobWork, true, dfErr), "The blobs do not match.");

                if (typeof(T) == typeof(double))
                    dfErr = 0.03;

                blobGradExp.LoadFromNumpy(strPath + "proj_layer.0.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "proj_layer.0.bias_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "proj_layer.1.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "proj_layer.1.bias_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[3], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "proj_layer.2.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[4], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "proj_layer.2.bias_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[5], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "proj_layer.3.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[6], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "proj_layer.3.bias_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[7], blobWork, true, dfErr), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobGradExp);
                dispose(ref blobProcessedNumeric0);
                dispose(ref blobProcessedNumeric1);
                dispose(ref blobProcessedNumeric2);
                dispose(ref blobProcessedNumeric3);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY0);
                dispose(ref blobY1);
                dispose(ref blobY2);
                dispose(ref blobY3);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY0 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobY3 = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY0 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobY3 = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_numeric.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY0);
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);
                TopVec.Add(blobY3);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.bias.npy");

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, 0.5);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY0);
                dispose(ref blobY1);
                dispose(ref blobY2);
                dispose(ref blobY3);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
