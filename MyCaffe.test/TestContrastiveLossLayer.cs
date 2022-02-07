using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.test;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using System.Drawing;
using MyCaffe.solvers;
using MyCaffe.data;

namespace MyCaffe.test
{
    [TestClass]
    public class TestContrastiveLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
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
        public void TestForwardLabels()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestForwardLabels();
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
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
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
        public void TestForwardLegacy()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestForwardLegacy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientLegacy()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestGradientLegacy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTraining()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestTraining(false, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingTwoIpLayers()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestTraining(false, true, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSameData()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestTraining(true, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingSameNonRandomData()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestTraining(true, false, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IContrastiveLossLayerTest : ITest
    {
        void TestForward();
        void TestForwardLabels();
        void TestGradient();
        void TestForwardLegacy();
        void TestGradientLegacy();
        void TestTraining(bool bSameTrainingTestingData, bool bUseRandomData, bool bTwoIpLayers);
    }

    class ContrastiveLossLayerTest : TestBase
    {
        public ContrastiveLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Contrastive Loss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ContrastiveLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ContrastiveLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ContrastiveLossLayerTest<T> : TestEx<T>, IContrastiveLossLayerTest
    {
        CryptoRandom m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);
        Blob<T> m_blob_bottom_data_i;
        Blob<T> m_blob_bottom_data_j;
        Blob<T> m_blob_bottom_y;
        Blob<T> m_blob_top_loss;

        public ContrastiveLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            BottomVec.Clear();
            TopVec.Clear();

            m_blob_bottom_data_i = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
            m_blob_bottom_data_j = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
            m_blob_bottom_y = new Blob<T>(m_cuda, m_log, 5, 1, 1, 1);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);

            m_cuda.rng_setseed(1701);
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);

            // fill the values
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = -1.0;
            fp.max = 1.0;   // distances ~=1.0 to test both sides of margin
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(m_blob_bottom_data_i);
            BottomVec.Add(m_blob_bottom_data_i);

            filler.Fill(m_blob_bottom_data_j);
            BottomVec.Add(m_blob_bottom_data_j);

            FillYsim();

            BottomVec.Add(m_blob_bottom_y);
            TopVec.Add(m_blob_top_loss);
        }

        protected override void dispose()
        {
            m_blob_bottom_data_i.Dispose();
            m_blob_bottom_data_j.Dispose();
            m_blob_bottom_y.Dispose();
            m_blob_top_loss.Dispose();
            base.dispose();
        }

        private void FillYsim()
        {
            m_blob_bottom_y.Reshape(m_blob_bottom_data_i.num, 1, 1, 1);

            double[] rgdfY = convert(m_blob_bottom_y.mutable_cpu_data);

            for (int i = 0; i < m_blob_bottom_y.count(); i++)
            {
                rgdfY[i] = m_random.Next() % 2; // 0 or 1
            }

            m_blob_bottom_y.mutable_cpu_data = convert(rgdfY);
        }

        private void FillYlabels()
        {
            m_blob_bottom_y.Reshape(m_blob_bottom_data_i.num, 2, 1, 1);

            double[] rgdfY = convert(m_blob_bottom_y.mutable_cpu_data);

            for (int i = 0; i < m_blob_bottom_y.count(); i++)
            {
                rgdfY[i] = m_random.Next() % 2; // 0 or 1
            }

            m_blob_bottom_y.mutable_cpu_data = convert(rgdfY);
        }

        public void TestForward()
        {
            FillYsim();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // manually compute to compare.
                double dfMargin = p.contrastive_loss_param.margin;
                int nNum = m_blob_bottom_data_i.num;
                int nChannels = m_blob_bottom_data_i.channels;
                double dfLoss = 0;

                double[] rgData_i = convert(m_blob_bottom_data_i.update_cpu_data());
                double[] rgData_j = convert(m_blob_bottom_data_j.update_cpu_data());
                double[] rgY = convert(m_blob_bottom_y.update_cpu_data());

                for (int i = 0; i < nNum; i++)
                {
                    double dfDistSq = 0;

                    for (int j = 0; j < nChannels; j++)
                    {
                        int nIdx = i * nChannels + j;
                        double dfDiff = rgData_i[nIdx] - rgData_j[nIdx];
                        dfDistSq += dfDiff * dfDiff;
                    }

                    if (rgY[i] != 0)    // similar pairs
                    {
                        dfLoss += dfDistSq;
                    }
                    else
                    {
                        double dfDist = Math.Max(dfMargin - Math.Sqrt(dfDistSq), 0.0);
                        dfLoss += dfDist * dfDist;
                    }
                }

                dfLoss /= nNum * 2.0;
                double dfTop = convert(m_blob_top_loss.GetData(0));

                m_log.EXPECT_NEAR(dfTop, dfLoss, 1e-6);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardLabels()
        {
            FillYlabels();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // manually compute to compare.
                double dfMargin = p.contrastive_loss_param.margin;
                int nNum = m_blob_bottom_data_i.num;
                int nChannels = m_blob_bottom_data_i.channels;
                double dfLoss = 0;

                double[] rgData_i = convert(m_blob_bottom_data_i.update_cpu_data());
                double[] rgData_j = convert(m_blob_bottom_data_j.update_cpu_data());
                double[] rgY = convert(m_blob_bottom_y.update_cpu_data());

                for (int i = 0; i < nNum; i++)
                {
                    double dfDistSq = 0;

                    for (int j = 0; j < nChannels; j++)
                    {
                        int nIdx = i * nChannels + j;
                        double dfDiff = rgData_i[nIdx] - rgData_j[nIdx];
                        dfDistSq += dfDiff * dfDiff;
                    }

                    if (rgY[i * 2] == rgY[i * 2 + 1])    // similar pairs
                    {
                        dfLoss += dfDistSq;
                    }
                    else
                    {
                        double dfDist = Math.Max(dfMargin - Math.Sqrt(dfDistSq), 0.0);
                        dfLoss += dfDist * dfDist;
                    }
                }

                dfLoss /= nNum * 2.0;
                double dfTop = convert(m_blob_top_loss.GetData(0));

                m_log.EXPECT_NEAR(dfTop, dfLoss, 1e-6);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            FillYsim();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);

                // check the gradient for the first two bottom layers
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 1);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardLegacy()
        {
            FillYsim();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            p.contrastive_loss_param.legacy_version = true;
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // manually compute to compare.
                double dfMargin = p.contrastive_loss_param.margin;
                int nNum = m_blob_bottom_data_i.num;
                int nChannels = m_blob_bottom_data_i.channels;
                double dfLoss = 0;

                double[] rgData_i = convert(m_blob_bottom_data_i.update_cpu_data());
                double[] rgData_j = convert(m_blob_bottom_data_j.update_cpu_data());
                double[] rgY = convert(m_blob_bottom_y.update_cpu_data());

                for (int i = 0; i < nNum; i++)
                {
                    double dfDistSq = 0;

                    for (int j = 0; j < nChannels; j++)
                    {
                        int nIdx = i * nChannels + j;
                        double dfDiff = rgData_i[nIdx] - rgData_j[nIdx];
                        dfDistSq += dfDiff * dfDiff;
                    }

                    if (rgY[i] != 0)    // similar pairs
                        dfLoss += dfDistSq;
                    else
                        dfLoss += Math.Max(dfMargin - dfDistSq, 0.0);
                }

                dfLoss /= (double)(nNum * 2.0);

                double dfTop = convert(m_blob_top_loss.GetData(0));
                m_log.EXPECT_NEAR(dfTop, dfLoss, 1e-6);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientLegacy()
        {
            FillYsim();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            p.contrastive_loss_param.legacy_version = true;
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);

                // check the gradient for the first two bottom layers
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 1);
            }
            finally
            {
                layer.Dispose();
            }
        }

        private byte getVal(int nVal, bool bUseRandomData)
        {
            int nStep = 116;
            byte bVal = (bUseRandomData) ? (byte)m_random.Next(nStep) : (byte)nStep;

            if (nVal > 0)
            {
                bVal += (bUseRandomData) ? (byte)m_random.Next(128 - nStep) : (byte)(128 - nStep);
                bVal += (bUseRandomData) ? (byte)m_random.Next(128) : (byte)128;
            }

            return bVal;
        }

        protected SourceDescriptor createTestData(DatasetFactory factory, string strSrc, bool bUseRandomData)
        {
            int nNum = 12;
            int nChannels = 1;
            int nWidth = 4;
            int nHeight = 4;
            int nCount = nNum * nChannels * nHeight * nWidth;

            SourceDescriptor src = new SourceDescriptor(0, strSrc, nWidth, nHeight, nChannels, false, true);
            src.ID = factory.AddSource(src);

            factory.DeleteSourceData(src.ID);

            factory.Open(src);

            List<int> rgLabels = new List<int>();

            for (int i = 0; i < nNum; i++)
            {
                int nLabel = 0;
                byte[] rgData = new byte[nChannels * nHeight * nWidth];
                for (int j = 0; j < rgData.Length; j++)
                {
                    rgData[j] = getVal(0, bUseRandomData);
                }

                if (i < 4)
                {
                    nLabel = 0;
                    rgData[7] = getVal(1, bUseRandomData);
                    rgData[10] = getVal(1, bUseRandomData);
                    rgData[11] = (byte)((i != 2) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[13] = getVal(1, bUseRandomData);
                    rgData[14] = (byte)((i != 1) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[15] = (byte)((i != 3) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                }
                else if (i < 8)
                {
                    nLabel = 1;
                    rgData[0] = (byte)((i != 7) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[1] = (byte)((i != 6) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[2] = getVal(1, bUseRandomData);
                    rgData[4] = (byte)((i != 5) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[5] = getVal(1, bUseRandomData);
                    rgData[8] = getVal(1, bUseRandomData);
                }
                else
                {
                    nLabel = 2;
                    rgData[3] = (byte)((i != 11) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[6] = (byte)((i != 10) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[9] = (byte)((i != 9) ? getVal(1, bUseRandomData) : getVal(0, bUseRandomData));
                    rgData[12] = getVal(1, bUseRandomData);
                }

                Datum d = new Datum(false, nChannels, nWidth, nHeight, nLabel, DateTime.Today, rgData.ToList(), 0, false, i);

                if (!rgLabels.Contains(nLabel))
                    rgLabels.Add(nLabel);

                factory.PutRawImage(i, d);
            }

            for (int i = 0; i < rgLabels.Count; i++)
            {
                factory.AddLabel(rgLabels[i], rgLabels[i].ToString());
            }

            factory.UpdateLabelCounts();
            factory.Close();

            return src;
        }

        private string getDataLayer(string strSrc, Phase phase)
        {
            string strLayerData = "layer { " +
                                   "name: \"pair_data\" " +
                                   "type: \"Data\" " +
                                   "top: \"pair_data\" " +
                                   "top: \"sim\" " +
                                   "include { phase: " + phase.ToString() + " } " +
                                   "transform_param { scale: 0.00390625 } " +
                                   "data_param " +
                                   "{ " +
                                   "   source: \"" + strSrc + "\" " +
                                   "   batch_size: 1 " +
                                   "   backend: IMAGEDB " +
                                   "   enable_random_selection: True " +
                                   "   images_per_blob: 2 " +           // pack two images per channel
                                   "   output_all_labels: False " +     // only output similarity for label (1 == same class, 0 = different classes)
                                   "   balance_matches: True " +        // alternate between matching and non matching pairs on each query.
                                   "} " +
                                   "} ";
            return strLayerData;
        }

        private string getInnerProductLayer(int nOutputs, bool bUseRandom, string strName, string strParam, string strBottom, string strTop)
        {
            string strLayer = "layer " +
                              "{ " +
                              "name: " + strName + " " +
                              "type: \"InnerProduct\" " +
                              "bottom: " + strBottom + " " +
                              "top: " + strTop + " " +
                              "param { name: \"" + strParam + "_w\" lr_mult: 1 } " +
                              "param { name: \"" + strParam + "_b\" lr_mult: 2 } ";

            if (strTop.Contains("_"))
                strLayer += "exclude { phase: RUN } ";

            strLayer += "inner_product_param " +
                        "{ " +
                        "   num_output: " + nOutputs.ToString() + " " +
                        "   bias_term: True ";

            if (bUseRandom)
            {
                strLayer += "   weight_filler { type: \"xavier\" variance_norm: FAN_IN } " +
                            "   bias_filler { type: \"constant\" value: 0 } ";
            }
            else
            {
                strLayer += "   weight_filler { type: \"constant\" value: 0.5 } " +
                            "   bias_filler { type: \"constant\" value: 0 } ";
            }

            strLayer += "   axis: 1 " +
                        "} " +
                        "} ";
            return strLayer;
        }

        private string getModelDescriptor(string strSrc, bool bUseRandom, bool bTwoIpLayers)
        {
            string strModel = "name: \"SiameseNet\" ";

            strModel += Environment.NewLine;
            strModel += getDataLayer(strSrc + ".train", Phase.TRAIN);
            strModel += Environment.NewLine;
            strModel += getDataLayer(strSrc + ".test", Phase.TEST);
            strModel += Environment.NewLine;

            strModel += "layer " +
                        "{ " +
                        "name: \"slice_pair\" " +
                        "type: \"Slice\" " +
                        "bottom: \"pair_data\" " +
                        "top: \"data\" " +
                        "top: \"data_p\" " +
                        "exclude { phase: RUN } " +
                        "slice_param " +
                        "{ " +
                        "   axis: 1 " +
                        "   slice_point: 1 " +
                        "   slice_dim: 1 " +
                        "} " +
                        "} ";
            strModel += Environment.NewLine;

            string strParam = (bTwoIpLayers) ? "ip" : "feat";
            string strName = (bTwoIpLayers) ? "ip" : "feat";
            string strNameP = (bTwoIpLayers) ? "ip_p" : "feat_p";
            string strBottom = "data";
            string strBottomP = "data_p";
            string strTop = (bTwoIpLayers) ? "ip1" : "feat";
            string strTopP = (bTwoIpLayers) ? "ip1_p" : "feat_p";
            int nNumOutput = (bTwoIpLayers) ? 500 : 2;

            strModel += getInnerProductLayer(nNumOutput, bUseRandom, strName, strParam, strBottom, strTop);
            strModel += Environment.NewLine;

            if (bTwoIpLayers)
            {
                strBottom = strTop;
                strTop = "feat";
                strName = "feat";
                strParam = "feat";

                strModel += getInnerProductLayer(2, bUseRandom, strName, strParam, strBottom, strTop);
                strModel += Environment.NewLine;
                strParam = "ip";
            }

            strModel += getInnerProductLayer(nNumOutput, bUseRandom, strNameP, strParam, strBottomP, strTopP);
            strModel += Environment.NewLine;

            if (bTwoIpLayers)
            {
                strBottom = strTop;
                strBottomP = strTopP;
                strTopP = "feat_p";
                strNameP = "feat_p";
                strParam = "feat";

                strModel += getInnerProductLayer(2, bUseRandom, strNameP, strParam, strBottomP, strTopP);
                strModel += Environment.NewLine;
            }

            strModel += "layer " +
                        "{ " +
                        "name: \"loss\" " +
                        "type: \"ContrastiveLoss\" " +
                        "bottom: \"feat\" " +
                        "bottom: \"feat_p\" " +
                        "bottom: \"sim\" " +
                        "top: \"loss\" " +
                        "top: \"match\" " +
                        "loss_weight: 1 " +
                        "loss_weight: 0 " +
                        "exclude { phase: RUN } " +
                        "loss_param { normalization: VALID } " +
                        "contrastive_loss_param { output_matches: True } " +
                        "} ";
            strModel += Environment.NewLine;

            strModel += "layer " +
                        "{ " +
                        "name: \"accuracy1\" " +
                        "type: \"Accuracy\" " +
                        "bottom: \"match\" " +
                        "bottom: \"sim\" " +
                        "top: \"accuracy1\" " +
                        "include { phase: TEST } " +
                        "accuracy_param { axis: 0 } " +
                        "} ";
            strModel += Environment.NewLine;

            strModel += "layer " +
                        "{ " +
                        "name: \"silence1\" " +
                        "type: \"Silence\" " +
                        "bottom: \"match\" " +
                        "include { phase: TRAIN } " +
                        "} ";
            strModel += Environment.NewLine;

            return strModel;
        }

        private string getSolverDescriptor()
        {
            string strDesc = "test_iter: 100 " +
                             "test_interval: 2000 " + // skip
                             "test_compute_loss: False " +
                             "test_initialization: False " +
                             "base_lr: 0.001 " +
                             "display: 100 " +
                             "average_loss: 1 " +
                             "max_iter: 100 " +
                             "lr_policy: inv " +
                             "gamma: 0.0001 " +
                             "power: 0.75 " +
                             "momentum: 0.9 " +
                             "weight_decay: 0 " +
                             "regularization_type: L2 " +
                             "snapshot: 5000 " + // skip
                             "snapshot_prefix: examples/siamese/mnist_siamese " +
                             "snapshot_format: BINARYPROTO " +
                             "device_id: 1 " +
                             "type: SGD " +
                             "snapshot_include_weights: True " +
                             "snapshot_include_state: True " +
                             "eval_type: classification " +
                             "ap_version: integral " +
                             "show_per_class_result: False";
            return strDesc;
        }

        public void TestTraining(bool bSameTrainingTestingData, bool bUseRandomData, bool bTwoIpLayers)
        {
            TestingProgressSet progress = new TestingProgressSet();
            Solver<T> solver = null;
            IXImageDatabaseBase db = null;
            CancelEvent evtCancel = new CancelEvent();
            DatasetFactory factory = new DatasetFactory();
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);
            SourceDescriptor srcTest = createTestData(factory, "test_cl.test", bUseRandomData);     

            if (bSameTrainingTestingData)
                m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);

            SourceDescriptor srcTrain = createTestData(factory, "test_cl.train", bUseRandomData);   
            DatasetDescriptor ds = new DatasetDescriptor(0, "test_cl", null, null, srcTrain, srcTest, null, null);
            string strModel = getModelDescriptor("test_cl", bUseRandomData, bTwoIpLayers);
            string strSolver = getSolverDescriptor();

            try
            {
                m_log.EnableTrace = true;
                m_log.Enable = false;

                ds.ID = factory.AddDataset(ds);
                factory.UpdateDatasetCounts(ds.ID);
                ds = factory.LoadDataset(ds.ID);

                db = createImageDb(m_log);
                db.InitializeWithDsName1(new SettingsCaffe(), ds.Name);

                if (db is IXImageDatabase2)
                {
                    long lQueryState = ((IXImageDatabase2)db).CreateQueryState(ds.ID, false, false);
                    ((IXImageDatabase2)db).SetDefaultQueryState(ds.ID, lQueryState);
                }

                ProjectEx prj = new ProjectEx("test");
                prj.SolverDescription = strSolver;
                prj.ModelDescription = strModel;
                prj.SetDataset(ds);

                MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(new SettingsCaffe(), m_log, evtCancel);
                double dfAccuracyPreTrain = 0;
                double dfAccuracyPostTrain = 0;
                int nTrainSteps = 0;

                try
                {
                    mycaffe.Load(Phase.TRAIN, prj, null, null, false, db);

                    dfAccuracyPreTrain = calculateAccuracy(mycaffe, ds);
                    m_log.WriteLine("Pre-train accuracy = " + dfAccuracyPreTrain.ToString("P"), true);

                    for (int i = 0; i < 1000; i++)
                    {
                        nTrainSteps++;
                        mycaffe.Train(1);
                        dfAccuracyPostTrain = calculateAccuracy(mycaffe, ds);

                        m_log.WriteLine(i.ToString() + ". Post-train accuracy = " + dfAccuracyPostTrain.ToString("P"), true);

                        if (dfAccuracyPostTrain == 1)
                            break;

                        progress.SetProgress((double)i / 1000);
                    }
                }
                finally
                {
                    mycaffe.Dispose();
                    mycaffe = null;
                }

                m_log.CHECK_GT(dfAccuracyPostTrain, dfAccuracyPreTrain, "The accuracy post train should be greater than the pre-train accurach!");

                if (dfAccuracyPostTrain == 1)
                    m_log.WriteLine("Encoding learned in " + nTrainSteps.ToString(), true);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                progress.Dispose();

                if (solver != null)
                    solver.Dispose();

                if (evtCancel != null)
                    evtCancel.Dispose();
            }
        }

        private double calculateAccuracy(MyCaffeControl<T> mycaffe, DatasetDescriptor ds)
        {
            Dictionary<int, List<double>> rgEncodings = getLabelEncodings(mycaffe, ds);
            int nCorrectCount = 0;

            for (int i = 0; i < ds.TestingSource.ImageCount; i++)
            {
                SimpleDatum sd = mycaffe.ImageDatabase.QueryImage(ds.TestingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                ResultCollection res = mycaffe.Run(sd);

                int nLabel = getLabel(rgEncodings, res);
                if (nLabel == sd.Label)
                    nCorrectCount++;
            }

            double dfAccuracy = (double)nCorrectCount / (double)ds.TestingSource.ImageCount;

            return dfAccuracy;
        }

        private Dictionary<int, List<double>> getLabelEncodings(MyCaffeControl<T> mycaffe, DatasetDescriptor ds)
        {
            Dictionary<int, List<double>> rgEncodings = new Dictionary<int, List<double>>();
            Dictionary<int, int> rgCounts = new Dictionary<int, int>();

            for (int i = 0; i < ds.TestingSource.ImageCount; i++)
            {
                SimpleDatum sd = mycaffe.ImageDatabase.QueryImage(ds.TestingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                ResultCollection res = mycaffe.Run(sd);
                List<double> rgEncoding = res.GetEncoding();
                int nLabel = sd.Label;

                if (!rgEncodings.ContainsKey(nLabel))
                {
                    rgEncodings.Add(nLabel, rgEncoding);
                    rgCounts.Add(nLabel, 1);
                }
                else
                {
                    for (int j = 0; j < rgEncoding.Count; j++)
                    {
                        rgEncodings[nLabel][j] += rgEncoding[j];
                    }

                    rgCounts[nLabel]++;
                }
            }

            // Calculate the centroids.
            foreach (KeyValuePair<int, int> kv in rgCounts)
            {
                int nCount = rgEncodings[kv.Key].Count;

                for (int i = 0; i < nCount; i++)
                {
                    rgEncodings[kv.Key][i] /= rgCounts[kv.Key];
                }
            }

            return rgEncodings;
        }

        private int getLabel(Dictionary<int, List<double>> rgEncodings, ResultCollection res)
        {
            Dictionary<int, double> rgDistances = new Dictionary<int, double>();
            List<double> rgEncoding = res.GetEncoding();

            foreach (KeyValuePair<int, List<double>> kv in rgEncodings)
            {
                double dfDist = calculateDistance(kv.Value, rgEncoding);
                rgDistances.Add(kv.Key, dfDist);
            }

            List<KeyValuePair<int, double>> rgDist = rgDistances.OrderBy(p => p.Value).ToList();
            return rgDist[0].Key;
        }

        private double calculateDistance(List<double> rg1, List<double> rg2)
        {
            double dfDiffSq = 0;

            for (int i = 0; i < rg1.Count; i++)
            {
                dfDiffSq += Math.Pow(rg1[i] - rg2[i], 2);
            }

            return Math.Sqrt(dfDiffSq);
        }
    }
}
