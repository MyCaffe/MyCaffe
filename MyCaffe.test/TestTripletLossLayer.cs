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
using MyCaffe.layers.alpha;

/// <summary>
/// Testing for simple triplet loss layer.
/// 
/// TripletLoss Layer - this is the triplet loss layer used to calculate the triplet loss and gradients using the
/// triplet loss method of learning.  The triplet loss method involves image triplets using the following format:
///     Anchor (A), Positives (P) and Negatives (N),
///     
/// Where Anchors and Positives are from the same class and Negatives are from a different class.  In the basic algorithm,
/// the distance between AP and AN are determined and the learning occurs by shrinking the distance between AP and increasing
/// the distance between AN.
/// 
/// </summary>
/// <remarks>
/// * Initial Python code for TripletDataLayer/TripletSelectionLayer/TripletLossLayer by luhaofang/tripletloss on github. 
/// See https://github.com/luhaofang/tripletloss - for general architecture
/// 
/// * Initial C++ code for TripletLoss layer by eli-oscherovich in 'Triplet loss #3663' pull request on BLVC/caffe github.
/// See https://github.com/BVLC/caffe/pull/3663/commits/c6518fb5752344e1922eaa1b1eb686bae5cc3964 - for triplet loss layer implementation
/// 
/// For an explanation of the gradient calculations,
/// See http://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula/33349475#33349475 - for gradient calculations
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTripletLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            TripletLossLayerTest test = new TripletLossLayerTest();

            try
            {
                foreach (ITripletLossLayerTest t in test.Tests)
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
            TripletLossLayerTest test = new TripletLossLayerTest();

            try
            {
                foreach (ITripletLossLayerTest t in test.Tests)
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

    interface ITripletLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class TripletLossLayerTest : TestBase
    {
        public TripletLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TripletLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TripletLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TripletLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TripletLossLayerTest<T> : TestEx<T>, ITripletLossLayerTest
    {
        Random m_random;
        Blob<T> m_blobBottomAnchor;
        Blob<T> m_blobBottomSame;   // positive.
        Blob<T> m_blobBottomDiff;   // negative.
        Blob<T> m_blobLabel;
        Blob<T> m_blobTopLoss;

        public TripletLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_random = new Random(1701);
            m_engine = engine;

            m_blobBottomAnchor = new Blob<T>(m_cuda, m_log, 10, 4, 5, 2);
            m_blobBottomSame = new Blob<T>(m_cuda, m_log, 10, 4, 5, 2);
            m_blobBottomDiff = new Blob<T>(m_cuda, m_log, 10, 4, 5, 2);
            m_blobLabel = new Blob<T>(m_cuda, m_log, 30, 4, 5, 2);
            m_blobTopLoss = new Blob<T>(m_cuda, m_log);

            BottomVec.Clear();
            TopVec.Clear();

            // fill the values
            FillerParameter fp = getFillerParam();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(m_blobBottomAnchor);
            BottomVec.Add(m_blobBottomAnchor);

            filler.Fill(m_blobBottomSame);
            BottomVec.Add(m_blobBottomSame);

            filler.Fill(m_blobBottomDiff);
            BottomVec.Add(m_blobBottomDiff);

            double[] rgLabel = convert(m_blobLabel.update_cpu_data());
            for (int i = 0; i < 30; i++)
            {
                if (i < 20)
                {
                    rgLabel[i] = 2;
                }
                else
                {
                    int nLabel = m_random.Next(10);
                    while (nLabel == 2)
                    {
                        nLabel = m_random.Next(10);
                    }

                    rgLabel[i] = nLabel;
                }
            }

            m_blobLabel.mutable_cpu_data = convert(rgLabel);
            BottomVec.Add(m_blobLabel);

            TopVec.Add(m_blobTopLoss);
        }

        protected override void dispose()
        {
            m_blobBottomAnchor.Dispose();
            m_blobBottomSame.Dispose();
            m_blobBottomDiff.Dispose();
            m_blobLabel.Dispose();
            m_blobTopLoss.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            // Get the loss without a specified objective weight -- should be
            // equivalent to explicitly specifying a weight of 1.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_LOSS);
            TripletLossLayer<T> layer_weight_1 = new TripletLossLayer<T>(m_cuda, m_log, p);
            layer_weight_1.Setup(BottomVec, TopVec);
            double dfLossWeight1 = layer_weight_1.Forward(BottomVec, TopVec);

            // Make sure the loss is not trivial
            double kNonTrivialAbsThreshold = 1e-1;
            m_log.CHECK_GE(Math.Abs(dfLossWeight1), kNonTrivialAbsThreshold, "The loss is trivial.");

            // Get the loss again with a different objective weight; check that it is
            // scaled appropriately.
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            TripletLossLayer<T> layer_weight_2 = new TripletLossLayer<T>(m_cuda, m_log, p);
            layer_weight_2.Setup(BottomVec, TopVec);
            double dfLossWeight2 = layer_weight_2.Forward(BottomVec, TopVec);

            double kErrorMargin = 1e-5;
            m_log.EXPECT_NEAR(dfLossWeight1 * kLossWeight, dfLossWeight2, kErrorMargin);

            // Get the loss again with a different alpha; check that it is changed
            // appropriately.
            double dfAlpha = 0.314;
            p.triplet_loss_param.alpha = dfAlpha;
            TripletLossLayer<T> layer_weight_2_alpha = new TripletLossLayer<T>(m_cuda, m_log, p);
            layer_weight_2_alpha.Setup(BottomVec, TopVec);
            double dfLossWeight2Alpha = layer_weight_2_alpha.Forward(BottomVec, TopVec);

            m_log.CHECK_GE(dfLossWeight2, dfLossWeight2, "Alpha is not being accounted for.");
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_LOSS);
            double dfLossWeight = 3.7;
            p.loss_weight.Add(dfLossWeight);
            TripletLossLayer<T> layer = new TripletLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
