using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.layers.ssd;
using MyCaffe.fillers;
using MyCaffe.param.ssd;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMultiBoxLossLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            MultiBoxLossLayerTest test = new MultiBoxLossLayerTest();

            try
            {
                foreach (IMultiBoxLossLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLocGradient()
        {
            MultiBoxLossLayerTest test = new MultiBoxLossLayerTest();

            try
            {
                foreach (IMultiBoxLossLayerTest t in test.Tests)
                {
                    t.TestLocGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConfGradient()
        {
            MultiBoxLossLayerTest test = new MultiBoxLossLayerTest();

            try
            {
                foreach (IMultiBoxLossLayerTest t in test.Tests)
                {
                    t.TestConfGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IMultiBoxLossLayerTest : ITest
    {
        void TestSetup();
        void TestLocGradient();
        void TestConfGradient();
    }

    class MultiBoxLossLayerTest : TestBase
    {
        public MultiBoxLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MultiBoxLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MultiBoxLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MultiBoxLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MultiBoxLossLayerTest<T> : TestEx<T>, IMultiBoxLossLayerTest
    {
        int m_nNum = 3;
        int m_nNumClasses = 3;
        int m_nWidth = 2;
        int m_nHeight = 2;
        int m_nNumPriorsPerLocation = 4;
        int m_nNumPriors;
        Blob<T> m_blobBottomLoc;
        Blob<T> m_blobBottomConf;
        Blob<T> m_blobBottomPrior;
        Blob<T> m_blobBottomGt;
        Blob<T> m_blobTopLoss;
        List<bool> m_rgbChoices = new List<bool>() { true, false };
        List<MultiBoxLossParameter.LocLossType> m_rgLocLossTypes = new List<MultiBoxLossParameter.LocLossType>() { MultiBoxLossParameter.LocLossType.L2, MultiBoxLossParameter.LocLossType.SMOOTH_L1 };
        List<MultiBoxLossParameter.ConfLossType> m_rgConfLossTypes = new List<MultiBoxLossParameter.ConfLossType>() { MultiBoxLossParameter.ConfLossType.SOFTMAX, MultiBoxLossParameter.ConfLossType.LOGISTIC };
        List<MultiBoxLossParameter.MatchType> m_rgMatchTypes = new List<MultiBoxLossParameter.MatchType>() { MultiBoxLossParameter.MatchType.BIPARTITE, MultiBoxLossParameter.MatchType.PER_PREDICTION };
        List<LossParameter.NormalizationMode> m_rgNormalizationMode = new List<LossParameter.NormalizationMode>() { LossParameter.NormalizationMode.BATCH_SIZE, LossParameter.NormalizationMode.FULL, LossParameter.NormalizationMode.VALID, LossParameter.NormalizationMode.NONE };
        List<MultiBoxLossParameter.MiningType> m_rgMiningType = new List<MultiBoxLossParameter.MiningType>() { MultiBoxLossParameter.MiningType.NONE, MultiBoxLossParameter.MiningType.MAX_NEGATIVE, MultiBoxLossParameter.MiningType.HARD_EXAMPLE };

        public MultiBoxLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_nNumPriors = m_nWidth * m_nHeight * m_nNumPriorsPerLocation;
            m_blobBottomLoc = new Blob<T>(m_cuda, m_log, m_nNum, m_nNumPriors * 4, 1, 1);
            m_blobBottomLoc.Name = "bottom loc";
            m_blobBottomConf = new Blob<T>(m_cuda, m_log, m_nNum, m_nNumPriors * m_nNumClasses, 1, 1);
            m_blobBottomConf.Name = "bottom conf";
            m_blobBottomPrior = new Blob<T>(m_cuda, m_log, m_nNum, 2, m_nNumPriors * 4, 1);
            m_blobBottomPrior.Name = "bottom prior";
            m_blobBottomGt = new Blob<T>(m_cuda, m_log, 1, 1, 4, 8);
            m_blobBottomGt.Name = "bottom gt";
            m_blobTopLoss = new Blob<T>(m_cuda, m_log);
            m_blobTopLoss.Name = "top loss";

            BottomVec.Clear();
            BottomVec.Add(m_blobBottomLoc);
            BottomVec.Add(m_blobBottomConf);
            BottomVec.Add(m_blobBottomPrior);
            BottomVec.Add(m_blobBottomGt);

            TopVec.Clear();
            TopVec.Add(m_blobTopLoss);
        }

        protected override void dispose()
        {
            dispose(ref m_blobBottomLoc);
            dispose(ref m_blobBottomConf);
            dispose(ref m_blobBottomPrior);
            dispose(ref m_blobBottomGt);

            base.dispose();
        }

        public void FillItem(float[] rgf, int nOffset, string strValues)
        {
            // Split values to vector of items.
            string[] rgstr = strValues.Split(',');

            for (int i = 0; i < 8; i++)
            {
                if (i >= 3 && i <= 6)
                    rgf[nOffset + i] = float.Parse(rgstr[i]);
                else
                    rgf[nOffset + i] = int.Parse(rgstr[i]);
            }
        }

        /// <summary>
        /// Fill the bottom blobs.
        /// </summary>
        /// <param name="bShareLocation">Specifies whether or not to share the location.</param>
        public void Fill(bool bShareLocation)
        {
            int nLocClasses = (bShareLocation) ? 1 : m_nNumClasses;
            BlobCollection<T> colFakeBottomVec = new BlobCollection<T>();
            BlobCollection<T> colFakeTopVec = new BlobCollection<T>();

            // Fake input (image) of size 20 x 20
            Blob<T> blobFakeInput = new Blob<T>(m_cuda, m_log, m_nNum, 3, 20, 20);

            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(blobFakeInput);

            List<int> rgGtShape = Utility.Create<int>(4, 1);
            rgGtShape[2] = 4;
            rgGtShape[3] = 8;
            m_blobBottomGt.Reshape(rgGtShape);

            float[] rgfGt = Utility.ConvertVecF<T>(m_blobBottomGt.mutable_cpu_data);
            FillItem(rgfGt, 8 * 0, "0, 1, 0, 0.1, 0.1, 0.3, 0.3, 0");
            FillItem(rgfGt, 8 * 1, "1, 1, 0, 0.1, 0.1, 0.3, 0.3, 0");
            FillItem(rgfGt, 8 * 2, "2, 2, 0, 0.2, 0.2, 0.4, 0.4, 0");
            FillItem(rgfGt, 8 * 3, "2, 2, 1, 0.6, 0.6, 0.8, 0.9, 1");
            m_blobBottomGt.mutable_cpu_data = Utility.ConvertVec<T>(rgfGt);

            // Fake layer
            LayerParameter pooling_param = new LayerParameter(LayerParameter.LayerType.POOLING);
            pooling_param.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
            pooling_param.pooling_param.kernel_size.Add(10);
            pooling_param.pooling_param.stride.Add(10);
            PoolingLayer<T> poolingLayer = Layer<T>.Create(m_cuda, m_log, pooling_param, null) as PoolingLayer<T>;

            Blob<T> blobFakeBlob = new Blob<T>(m_cuda, m_log, m_nNum, 5, m_nHeight, m_nWidth);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakeInput);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(blobFakeBlob);

            poolingLayer.Setup(colFakeBottomVec, colFakeTopVec);
            poolingLayer.Forward(colFakeBottomVec, colFakeTopVec);

            // 2.) Fill bbox location predictions
            LayerParameter convolution_param = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            convolution_param.convolution_param.pad.Add(0);
            convolution_param.convolution_param.kernel_size.Add(1);
            convolution_param.convolution_param.stride.Add(1);
            int nNumOutput = m_nNumPriorsPerLocation * nLocClasses * 4;
            convolution_param.convolution_param.num_output = (uint)nNumOutput;
            convolution_param.convolution_param.weight_filler = new FillerParameter("xavier");
            convolution_param.convolution_param.bias_filler = new FillerParameter("constant", 0.1);
            ConvolutionLayer<T> convLayer = Layer<T>.Create(m_cuda, m_log, convolution_param, null) as ConvolutionLayer<T>;

            Blob<T> blobFakeOutputLoc = new Blob<T>(m_cuda, m_log);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakeBlob);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(blobFakeOutputLoc);

            convLayer.Setup(colFakeBottomVec, colFakeTopVec);
            convLayer.Forward(colFakeBottomVec, colFakeTopVec);

            // Use Permute and Flatten layer to prepare for MultiBoxLoss layer.
            LayerParameter permute_param = new LayerParameter(LayerParameter.LayerType.PERMUTE);
            permute_param.permute_param.order.Add(0);
            permute_param.permute_param.order.Add(2);
            permute_param.permute_param.order.Add(3);
            permute_param.permute_param.order.Add(1);
            PermuteLayer<T> permuteLayer = Layer<T>.Create(m_cuda, m_log, permute_param, null) as PermuteLayer<T>;

            Blob<T> blobFakePermuteLoc = new Blob<T>(m_cuda, m_log);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakeOutputLoc);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(blobFakePermuteLoc);

            permuteLayer.Setup(colFakeBottomVec, colFakeTopVec);
            permuteLayer.Forward(colFakeBottomVec, colFakeTopVec);

            LayerParameter flatten_param = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            flatten_param.flatten_param.axis = 1;
            FlattenLayer<T> flattenLayer = Layer<T>.Create(m_cuda, m_log, flatten_param, null) as FlattenLayer<T>;
            List<int> rgLocShape = Utility.Create<int>(4, 1);
            rgLocShape[0] = m_nNum;
            rgLocShape[1] = nNumOutput * m_nHeight * m_nWidth;
            m_blobBottomLoc.Reshape(rgLocShape);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakePermuteLoc);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(m_blobBottomLoc);

            flattenLayer.Setup(colFakeBottomVec, colFakeTopVec);
            flattenLayer.Forward(colFakeBottomVec, colFakeTopVec);

            // 3.) Fill bbox confidence predictions
            convolution_param.convolution_param.num_output = (uint)(m_nNumPriorsPerLocation * m_nNumClasses);
            ConvolutionLayer<T> convLayerConf = Layer<T>.Create(m_cuda, m_log, convolution_param, null) as ConvolutionLayer<T>;

            Blob<T> blobFakeOutputConf = new Blob<T>(m_cuda, m_log);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakeBlob);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(blobFakeOutputConf);

            convLayerConf.Setup(colFakeBottomVec, colFakeTopVec);
            convLayerConf.Forward(colFakeBottomVec, colFakeTopVec);

            Blob<T> blobFakePermuteConf = new Blob<T>(m_cuda, m_log);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakeOutputConf);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(blobFakePermuteConf);

            permuteLayer.Setup(colFakeBottomVec, colFakeTopVec);
            permuteLayer.Forward(colFakeBottomVec, colFakeTopVec);

            List<int> rgConfShape = Utility.Create<int>(4, 1);
            rgConfShape[0] = m_nNum;
            rgConfShape[1] = nNumOutput * m_nHeight * m_nWidth;
            m_blobBottomConf.Reshape(rgConfShape);

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakePermuteConf);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(m_blobBottomConf);

            flattenLayer.Setup(colFakeBottomVec, colFakeTopVec);
            flattenLayer.Forward(colFakeBottomVec, colFakeTopVec);

            // 4.) Fill prior bboxes
            LayerParameter prior_box_param = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            prior_box_param.prior_box_param.min_size.Add(5);
            prior_box_param.prior_box_param.max_size.Add(10);
            prior_box_param.prior_box_param.aspect_ratio.Add(3.0f);
            prior_box_param.prior_box_param.flip = true;

            PriorBoxLayer<T> priorBoxLayer = Layer<T>.Create(m_cuda, m_log, prior_box_param, null) as PriorBoxLayer<T>;

            colFakeBottomVec.Clear();
            colFakeBottomVec.Add(blobFakeBlob);
            colFakeBottomVec.Add(blobFakeInput);
            colFakeTopVec.Clear();
            colFakeTopVec.Add(m_blobBottomPrior);

            priorBoxLayer.Setup(colFakeBottomVec, colFakeTopVec);
            priorBoxLayer.Forward(colFakeBottomVec, colFakeTopVec);

            poolingLayer.Dispose();
            convLayer.Dispose();
            permuteLayer.Dispose();
            flattenLayer.Dispose();
            convLayerConf.Dispose();
            blobFakeInput.Dispose();
            blobFakeBlob.Dispose();
            blobFakeOutputLoc.Dispose();
            blobFakePermuteLoc.Dispose();
            blobFakeOutputConf.Dispose();
            blobFakePermuteConf.Dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIBOX_LOSS);
            p.multiboxloss_param.num_classes = 3;

            for (int i = 0; i < 2; i++)
            {
                bool bShareLocation = m_rgbChoices[i];

                Fill(bShareLocation);

                for (int j = 0; j < 2; j++)
                {
                    MultiBoxLossParameter.MatchType matchType = m_rgMatchTypes[j];

                    for (int k = 0; k < 2; k++)
                    {
                        bool bUsePrior = m_rgbChoices[k];

                        for (int m = 0; m < 3; m++)
                        {
                            MultiBoxLossParameter.MiningType miningType = m_rgMiningType[m];

                            if (!bShareLocation && miningType != MultiBoxLossParameter.MiningType.NONE)
                                continue;

                            p.multiboxloss_param.share_location = bShareLocation;
                            p.multiboxloss_param.match_type = matchType;
                            p.multiboxloss_param.use_prior_for_matching = bUsePrior;
                            p.multiboxloss_param.mining_type = miningType;

                            MultiBoxLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as MultiBoxLossLayer<T>;

                            layer.Setup(BottomVec, TopVec);

                            layer.Dispose();
                        }
                    }
                }
            }
        }

        public void TestLocGradient()
        {
            TestingProgressSet progress = new TestingProgressSet();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIBOX_LOSS);
            p.propagate_down.Add(true);
            p.propagate_down.Add(false);
            p.multiboxloss_param.num_classes = (uint)m_nNumClasses;

            int nTestTotal = 2 * 2 * 2 * 1 * 4 * 2 * 3;
            int nTestIdx = 0;

            for (int l = 0; l < 2; l++)
            {
                MultiBoxLossParameter.LocLossType locLossType = m_rgLocLossTypes[l];

                for (int i = 0; i < 2; i++)
                {
                    bool bShareLocation = m_rgbChoices[i];

                    Fill(bShareLocation);

                    for (int j = 0; j < 2; j++)
                    {
                        MultiBoxLossParameter.MatchType matchType = m_rgMatchTypes[j];

                        for (int k = 0; k < 1; k++)
                        {
                            bool bUsePrior = m_rgbChoices[k];

                            for (int n = 0; n < 4; n++)
                            {
                                LossParameter.NormalizationMode normalize = m_rgNormalizationMode[n];
                                p.loss_param.normalization = normalize;

                                for (int u = 0; u < 2; u++)
                                {
                                    bool bUseDifficultGt = m_rgbChoices[u];

                                    for (int m = 0; m < 3; m++)
                                    {
                                        MultiBoxLossParameter.MiningType miningType = m_rgMiningType[m];

                                        if (!bShareLocation && miningType != MultiBoxLossParameter.MiningType.NONE)
                                            continue;

                                        p.multiboxloss_param.loc_loss_type = locLossType;
                                        p.multiboxloss_param.share_location = bShareLocation;
                                        p.multiboxloss_param.match_type = matchType;
                                        p.multiboxloss_param.use_prior_for_matching = bUsePrior;
                                        p.multiboxloss_param.use_difficult_gt = bUseDifficultGt;
                                        p.multiboxloss_param.mining_type = miningType;

                                        MultiBoxLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as MultiBoxLossLayer<T>;
                                        GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);

                                        checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);

                                        layer.Dispose();

                                        nTestIdx++;
                                        progress.SetProgress((double)nTestIdx / (double)nTestTotal);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public void TestConfGradient()
        {
            TestingProgressSet progress = new TestingProgressSet();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIBOX_LOSS);
            p.propagate_down.Add(false);
            p.propagate_down.Add(true);
            p.multiboxloss_param.num_classes = (uint)m_nNumClasses;

            int nTestTotal = 2 * 2 * 2 * 1 * 4 * 2 * 3;
            int nTestIdx = 0;

            for (int c = 0; c < 2; c++)
            {
                MultiBoxLossParameter.ConfLossType confLossType = m_rgConfLossTypes[c];

                for (int i = 0; i < 2; i++)
                {
                    bool bShareLocation = m_rgbChoices[i];

                    Fill(bShareLocation);

                    for (int j = 0; j < 2; j++)
                    {
                        MultiBoxLossParameter.MatchType matchType = m_rgMatchTypes[j];

                        for (int k = 0; k < 1; k++)
                        {
                            bool bUsePrior = m_rgbChoices[k];

                            for (int n = 0; n < 4; n++)
                            {
                                LossParameter.NormalizationMode normalize = m_rgNormalizationMode[n];
                                p.loss_param.normalization = normalize;

                                for (int u = 0; u < 2; u++)
                                {
                                    bool bUseDifficultGt = m_rgbChoices[u];

                                    for (int m = 0; m < 3; m++)
                                    {
                                        MultiBoxLossParameter.MiningType miningType = m_rgMiningType[m];

                                        if (!bShareLocation && miningType != MultiBoxLossParameter.MiningType.NONE)
                                            continue;

                                        p.multiboxloss_param.conf_loss_type = confLossType;
                                        p.multiboxloss_param.share_location = bShareLocation;
                                        p.multiboxloss_param.match_type = matchType;
                                        p.multiboxloss_param.use_prior_for_matching = bUsePrior;
                                        p.multiboxloss_param.use_difficult_gt = bUseDifficultGt;
                                        p.multiboxloss_param.background_label_id = 0;
                                        p.multiboxloss_param.mining_type = miningType;

                                        MultiBoxLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as MultiBoxLossLayer<T>;
                                        GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);

                                        checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 1);

                                        layer.Dispose();

                                        nTestIdx++;
                                        progress.SetProgress((double)nTestIdx / (double)nTestTotal);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
