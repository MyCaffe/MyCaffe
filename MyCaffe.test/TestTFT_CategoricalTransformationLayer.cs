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
/// Testing the Categorical Transformation layer.
/// 
/// CategoricalTransformation Layer - layer converts categorical inputs into embeddings using embedding layers.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_CategoricalTransformationLayer
    {
        [TestMethod]
        public void TestForward()
        {
            CategoricalTransformationLayerTest test = new CategoricalTransformationLayerTest();

            try
            {
                foreach (ICategoricalTransformationLayerTest t in test.Tests)
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
            CategoricalTransformationLayerTest test = new CategoricalTransformationLayerTest();

            try
            {
                foreach (ICategoricalTransformationLayerTest t in test.Tests)
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
            CategoricalTransformationLayerTest test = new CategoricalTransformationLayerTest();

            try
            {
                foreach (ICategoricalTransformationLayerTest t in test.Tests)
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

    interface ICategoricalTransformationLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
    }

    class CategoricalTransformationLayerTest : TestBase
    {
        public CategoricalTransformationLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("CategoricalTransformation Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new CategoricalTransformationLayerTest<double>(strName, nDeviceID, engine);
            else
                return new CategoricalTransformationLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class CategoricalTransformationLayerTest<T> : TestEx<T>, ICategoricalTransformationLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public CategoricalTransformationLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        /// <summary>
        /// Test categorical transformation forward.
        /// </summary>
        /// <remarks>
        /// To generate the test data run the following:
        /// 
        /// Code: test_1a_categoricalinputtransformation.py
        /// Path: cattrx
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CATEGORICAL_TRANS);
            p.categorical_trans_param.num_input = 7;
            p.categorical_trans_param.cardinalities = new List<int> { 2, 3, 8, 13, 72, 6, 28 };
            p.categorical_trans_param.state_size = 64;
            CategoricalTransformationLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY0 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobY3 = null;
            Blob<T> blobY4 = null;
            Blob<T> blobY5 = null;
            Blob<T> blobY6 = null;
            Blob<T> blobProcessedCategorical0 = null;
            Blob<T> blobProcessedCategorical1 = null;
            Blob<T> blobProcessedCategorical2 = null;
            Blob<T> blobProcessedCategorical3 = null;
            Blob<T> blobProcessedCategorical4 = null;
            Blob<T> blobProcessedCategorical5 = null;
            Blob<T> blobProcessedCategorical6 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("cattrx");
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as CategoricalTransformationLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY0 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobY3 = new Blob<T>(m_cuda, m_log);
                blobY4 = new Blob<T>(m_cuda, m_log);
                blobY5 = new Blob<T>(m_cuda, m_log);
                blobY6 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical0 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical1 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical2 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical3 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical4 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical5 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical6 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CATEGORICAL_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY0);
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);
                TopVec.Add(blobY3);
                TopVec.Add(blobY4);
                TopVec.Add(blobY5);
                TopVec.Add(blobY6);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.1.weight.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.3.weight.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.4.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.5.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.6.weight.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessedCategorical0.LoadFromNumpy(strPath + "processed_categorical_0.npy");
                blobProcessedCategorical1.LoadFromNumpy(strPath + "processed_categorical_1.npy");
                blobProcessedCategorical2.LoadFromNumpy(strPath + "processed_categorical_2.npy");
                blobProcessedCategorical3.LoadFromNumpy(strPath + "processed_categorical_3.npy");
                blobProcessedCategorical4.LoadFromNumpy(strPath + "processed_categorical_4.npy");
                blobProcessedCategorical5.LoadFromNumpy(strPath + "processed_categorical_5.npy");
                blobProcessedCategorical6.LoadFromNumpy(strPath + "processed_categorical_6.npy");
                BlobCollection<T> col = new BlobCollection<T>() {  blobProcessedCategorical0, blobProcessedCategorical1, blobProcessedCategorical2, blobProcessedCategorical3, blobProcessedCategorical4, blobProcessedCategorical5, blobProcessedCategorical6 };
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2.5e-07;

                for (int i = 0; i < p.categorical_trans_param.cardinalities.Count; i++)
                {
                    m_log.CHECK(TopVec[i].Compare(col[i], blobWork, false, dfErr), "The blobs do not match.");
                }
            }
            finally
            {
                dispose(ref blobProcessedCategorical0);
                dispose(ref blobProcessedCategorical1);
                dispose(ref blobProcessedCategorical2); 
                dispose(ref blobProcessedCategorical3);
                dispose(ref blobProcessedCategorical4);
                dispose(ref blobProcessedCategorical5);
                dispose(ref blobProcessedCategorical6);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY0);
                dispose(ref blobY1);
                dispose(ref blobY2);
                dispose(ref blobY3);
                dispose(ref blobY4);
                dispose(ref blobY5);
                dispose(ref blobY6);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test categorical transformation backward.
        /// </summary>
        /// <remarks>
        /// To generate the test data run the following:
        /// 
        /// Code: test_1a_categoricalinputtransformation.py
        /// Path: cattrx
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CATEGORICAL_TRANS);
            p.categorical_trans_param.num_input = 7;
            p.categorical_trans_param.cardinalities = new List<int> { 2, 3, 8, 13, 72, 6, 28 };
            p.categorical_trans_param.state_size = 64;
            CategoricalTransformationLayer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX = null;
            Blob<T> blobY0 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobY3 = null;
            Blob<T> blobY4 = null;
            Blob<T> blobY5 = null;
            Blob<T> blobY6 = null;
            Blob<T> blobProcessedCategorical0 = null;
            Blob<T> blobProcessedCategorical1 = null;
            Blob<T> blobProcessedCategorical2 = null;
            Blob<T> blobProcessedCategorical3 = null;
            Blob<T> blobProcessedCategorical4 = null;
            Blob<T> blobProcessedCategorical5 = null;
            Blob<T> blobProcessedCategorical6 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("cattrx");
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as CategoricalTransformationLayer<T>;
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY0 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobY3 = new Blob<T>(m_cuda, m_log);
                blobY4 = new Blob<T>(m_cuda, m_log);
                blobY5 = new Blob<T>(m_cuda, m_log);
                blobY6 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical0 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical1 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical2 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical3 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical4 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical5 = new Blob<T>(m_cuda, m_log);
                blobProcessedCategorical6 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CATEGORICAL_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY0);
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);
                TopVec.Add(blobY3);
                TopVec.Add(blobY4);
                TopVec.Add(blobY5);
                TopVec.Add(blobY6);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.1.weight.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.3.weight.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.4.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.5.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.6.weight.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessedCategorical0.LoadFromNumpy(strPath + "processed_categorical_0.npy");
                blobProcessedCategorical1.LoadFromNumpy(strPath + "processed_categorical_1.npy");
                blobProcessedCategorical2.LoadFromNumpy(strPath + "processed_categorical_2.npy");
                blobProcessedCategorical3.LoadFromNumpy(strPath + "processed_categorical_3.npy");
                blobProcessedCategorical4.LoadFromNumpy(strPath + "processed_categorical_4.npy");
                blobProcessedCategorical5.LoadFromNumpy(strPath + "processed_categorical_5.npy");
                blobProcessedCategorical6.LoadFromNumpy(strPath + "processed_categorical_6.npy");
                BlobCollection<T> col = new BlobCollection<T>() { blobProcessedCategorical0, blobProcessedCategorical1, blobProcessedCategorical2, blobProcessedCategorical3, blobProcessedCategorical4, blobProcessedCategorical5, blobProcessedCategorical6 };
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2.5e-07;

                for (int i = 0; i < p.categorical_trans_param.num_input; i++)
                {
                    m_log.CHECK(TopVec[i].Compare(col[i], blobWork, false, dfErr), "The blobs do not match.");
                }

                TopVec[0].LoadFromNumpy(strPath + "processed_categorical_0.grad.npy", true);
                TopVec[1].LoadFromNumpy(strPath + "processed_categorical_1.grad.npy", true);
                TopVec[2].LoadFromNumpy(strPath + "processed_categorical_2.grad.npy", true);
                TopVec[3].LoadFromNumpy(strPath + "processed_categorical_3.grad.npy", true);
                TopVec[4].LoadFromNumpy(strPath + "processed_categorical_4.grad.npy", true);
                TopVec[5].LoadFromNumpy(strPath + "processed_categorical_5.grad.npy", true);
                TopVec[6].LoadFromNumpy(strPath + "processed_categorical_6.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                // Data does not have a grad.
                //blobGradExp.LoadFromNumpy(strPath + "grad_x_categorical.npy", true);
                //m_log.CHECK(blobGradExp.Compare(blobX, blobWork, true, dfErr), "The blobs do not match.");

                if (typeof(T) == typeof(double))
                    dfErr = 0.03;

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.0.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.1.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.2.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.3.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[3], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.4.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[4], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.5.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[5], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "emb_layer.6.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[6], blobWork, true, dfErr), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobGradExp);
                dispose(ref blobProcessedCategorical0);
                dispose(ref blobProcessedCategorical1);
                dispose(ref blobProcessedCategorical2);
                dispose(ref blobProcessedCategorical3);
                dispose(ref blobProcessedCategorical4);
                dispose(ref blobProcessedCategorical5);
                dispose(ref blobProcessedCategorical6);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY0);
                dispose(ref blobY1);
                dispose(ref blobY2);
                dispose(ref blobY3);
                dispose(ref blobY4);
                dispose(ref blobY5);
                dispose(ref blobY6);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test categorical transformation gradient check.
        /// </summary>
        /// <remarks>
        /// To generate the test data run the following:
        /// 
        /// Code: test_1a_categoricalinputtransformation.py
        /// Path: cattrx
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CATEGORICAL_TRANS);
            p.categorical_trans_param.num_input = 7;
            p.categorical_trans_param.cardinalities = new List<int> { 2, 3, 8, 13, 72, 6, 28 };
            p.categorical_trans_param.state_size = 64;
            CategoricalTransformationLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY0 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobY3 = null;
            Blob<T> blobY4 = null;
            Blob<T> blobY5 = null;
            Blob<T> blobY6 = null;
            string strPath = getTestDataPath("cattrx");
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as CategoricalTransformationLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY0 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobY3 = new Blob<T>(m_cuda, m_log);
                blobY4 = new Blob<T>(m_cuda, m_log);
                blobY5 = new Blob<T>(m_cuda, m_log);
                blobY6 = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CATEGORICAL_TRANS, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY0);
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);
                TopVec.Add(blobY3);
                TopVec.Add(blobY4);
                TopVec.Add(blobY5);
                TopVec.Add(blobY6);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.0.weight.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.1.weight.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.2.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.3.weight.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.4.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.5.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.6.weight.npy");

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
                dispose(ref blobY4);
                dispose(ref blobY5);
                dispose(ref blobY6);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
