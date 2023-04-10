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
/// NumericTransformation Layer - layer converts inputs into embeddings using inner product layers.
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
        public void TestGradient()
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
    }

    interface INumericTransformationLayerTest : ITest
    {
        void TestForward();
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

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobProcessedNumeric0 = null;
            Blob<T> blobProcessedNumeric1 = null;
            Blob<T> blobProcessedNumeric2 = null;
            Blob<T> blobProcessedNumeric3 = null;
            Blob<T> blobCompare = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();


            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric0 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric1 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric2 = new Blob<T>(m_cuda, m_log);
                blobProcessedNumeric3 = new Blob<T>(m_cuda, m_log);
                blobCompare = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_numeric.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                blobProcessedNumeric0.LoadFromNumpy(strPath + "processed_numeric_0");
                blobProcessedNumeric1.LoadFromNumpy(strPath + "processed_numeric_1");
                blobProcessedNumeric2.LoadFromNumpy(strPath + "processed_numeric_2");
                blobProcessedNumeric3.LoadFromNumpy(strPath + "processed_numeric_3");
                BlobCollection<T> col = new BlobCollection<T>() {  blobProcessedNumeric0, blobProcessedNumeric1, blobProcessedNumeric2, blobProcessedNumeric3 };

                blobCompare.ReshapeLike(blobProcessedNumeric0);
                for (int i = 0; i < p.numeric_trans_param.num_input; i++)
                {
                    m_cuda.copy(blobCompare.count(), blobY.gpu_data, blobCompare.mutable_gpu_data, i * blobCompare.count(), 0);
                    m_log.CHECK(blobCompare.Compare(col[i], blobWork), "The blobs do not match.");
                }
            }
            finally
            {
                dispose(ref blobProcessedNumeric0);
                dispose(ref blobProcessedNumeric1);
                dispose(ref blobProcessedNumeric2); 
                dispose(ref blobProcessedNumeric3);
                dispose(ref blobCompare);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS);
            p.numeric_trans_param.num_input = 5;
            p.numeric_trans_param.state_size = 64;
            NumericTransformationLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            string strPath = getTestDataPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as NumericTransformationLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.NUMERIC_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_numeric.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
