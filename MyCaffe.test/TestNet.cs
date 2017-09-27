using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.imagedb;
using System.Threading;

namespace MyCaffe.test
{
    [TestClass]
    public class TestNet
    {
        [TestMethod]
        public void TestHasBlob()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestHasBlob();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetBlob()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestGetBlob();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestHasLayer()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestHasLayer();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetLayer()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestGetLayer();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBottomNeedBackward()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestBottomNeedBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBottomNeedBackwardForce()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestBottomNeedBackwardForce();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBottomNeedBackwardEuclideanForce()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestBottomNeedBackwardForce();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBottomNeedBackwardTricky()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestBottomNeedBackwardTricky();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLossWeight()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestLossWeight();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLossWeightMidNet()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestLossWeightMidNet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestComboLossWeight()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestComboLossWeight();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardWithAccuracyLayer()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestBackwardWithAccuracyLayer();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestUnsharedWeightsDataNet()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestUnsharedWeightsDataNet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSharedWeightsDataNet()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestSharedWeightsDataNet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestUnsharedWeightsDiffNet()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestUnsharedWeightsDiffNet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSharedWeightsDiffNet()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestSharedWeightsDiffNet();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSharedWeightsUpdate()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestSharedWeightsUpdate();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSharedWeightsResume()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestSharedWeightsResume();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestParamPropagateDown()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestParamPropagateDown();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFromTo()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFromTo();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNoFilter()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestNoFilter();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterNetTrainTest()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterNetTrainTest();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterOutByStage()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterOutByStage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterOutByStage2()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterOutByStage2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByStage()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByStage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByStage2()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByStage2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterOutByMultipleStage()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterOutByMultipleStage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByMultipleStage()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByMultipleStage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByMultipleStage2()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByMultipleStage2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByNotStage()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByNotStage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterOutByNotStage()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterOutByNotStage();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterOutByMinLevel()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterOutByMinLevel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterOutByMaxLevel()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterOutByMaxLevel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByMinLevel()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByMinLevel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByMinLevel2()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByMinLevel2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByMaxLevel()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByMaxLevel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByMaxLevel2()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByMaxLevel2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInOutByIncludeMultiRule()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInOutByIncludeMultiRule();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInByIncludeMultiRule()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInByIncludeMultiRule();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFilterInOutByExcludeMultiRule()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestFilterInOutByExcludeMultiRule();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReshape()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestReshape();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSkipPropagateDown()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestSkipPropagateDown();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForcePropagateDown()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestForcePropagateDown();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAllInOneNetTrain()
        {
            NetTest test = new NetTest();

            try
            {
                foreach (INetTest t in test.Tests)
                {
                    t.TestAllInOneNetTrain();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface INetTest : ITest
    {
        void TestHasBlob();
        void TestGetBlob();
        void TestHasLayer();
        void TestGetLayer();
        void TestBottomNeedBackward();
        void TestBottomNeedBackwardForce();
        void TestBottomNeedBackwardEuclideanForce();
        void TestBottomNeedBackwardTricky();
        void TestLossWeight();
        void TestLossWeightMidNet();
        void TestComboLossWeight();
        void TestBackwardWithAccuracyLayer();
        void TestUnsharedWeightsDataNet();
        void TestSharedWeightsDataNet();
        void TestUnsharedWeightsDiffNet();
        void TestSharedWeightsDiffNet();
        void TestSharedWeightsUpdate();
        void TestSharedWeightsResume();
        void TestParamPropagateDown();
        void TestFromTo();

        void TestNoFilter();
        void TestFilterNetTrainTest();
        void TestFilterOutByStage();
        void TestFilterOutByStage2();
        void TestFilterInByStage();
        void TestFilterInByStage2();
        void TestFilterOutByMultipleStage();
        void TestFilterInByMultipleStage();
        void TestFilterInByMultipleStage2();
        void TestFilterInByNotStage();
        void TestFilterOutByNotStage();
        void TestFilterOutByMinLevel();
        void TestFilterOutByMaxLevel();
        void TestFilterInByMinLevel();
        void TestFilterInByMinLevel2();
        void TestFilterInByMaxLevel();
        void TestFilterInByMaxLevel2();
        void TestFilterInOutByIncludeMultiRule();
        void TestFilterInByIncludeMultiRule();
        void TestFilterInOutByExcludeMultiRule();

        void TestReshape();
        void TestSkipPropagateDown();
        void TestForcePropagateDown();
        void TestAllInOneNetTrain();
    }

    class NetTest : TestBase
    {
        public NetTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Net Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NetTest<double>(strName, nDeviceID, engine);
            else
                return new NetTest<float>(strName, nDeviceID, engine);
        }
    }

    class NetTest<T> : TestEx<T>, INetTest
    {
        Net<T> m_net;
        long m_lSeed = 1701;
        MyCaffeImageDatabase m_db;
        CancelEvent m_evtCancel;

        public NetTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_evtCancel = new CancelEvent();
            m_db = new MyCaffeImageDatabase();
            m_db.InitializeWithDsName(new SettingsCaffe(), "MNIST");
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public Net<T> Net
        {
            get { return m_net; }
        }

        public virtual void InitNetFromProtoString(string strProto)
        {
            RawProto proto = RawProto.Parse(strProto);
            NetParameter param = NetParameter.FromProto(proto);
            m_net = new Net<T>(m_cuda, m_log, param, m_evtCancel, m_db);
        }

        public virtual void InitNetFromProtoFileWithState(string strProto, Phase phase = Phase.TRAIN, int nLevel = 0, List<string> rgStages = null)
        {
            RawProto proto = RawProto.Parse(strProto);
            NetParameter param = NetParameter.FromProto(proto);

            param.state.level = nLevel;
            param.state.stage = rgStages;

            m_net = new Net<T>(m_cuda, m_log, param, m_evtCancel, m_db, phase);
        }

        public virtual BlobCollection<T> CopyNetBlobs(bool bCopyDiff)
        {
            m_log.CHECK(m_net != null, "The net should not be null!");
            BlobCollection<T> colNetBlobs = m_net.blobs;
            BlobCollection<T> colBlobsCopy = new BlobCollection<T>();
            bool kReshape = true;

            for (int i = 0; i < colNetBlobs.Count; i++)
            {
                Blob<T> blobCopy = new Blob<T>(m_cuda, m_log);
                blobCopy.CopyFrom(colNetBlobs[i], bCopyDiff, kReshape);
                colBlobsCopy.Add(blobCopy);
            }

            return colBlobsCopy;
        }

        public virtual BlobCollection<T> CopyNetParams(bool bCopyDiff)
        {
            m_log.CHECK(m_net != null, "The net should not be null!");
            BlobCollection<T> colNetParams = m_net.parameters;
            BlobCollection<T> colParamCopy = new BlobCollection<T>();
            bool kReshape = true;

            for (int i = 0; i < colNetParams.Count; i++)
            {
                Blob<T> blobCopy = new Blob<T>(m_cuda, m_log);
                blobCopy.CopyFrom(colNetParams[i], bCopyDiff, kReshape);
                colParamCopy.Add(blobCopy);
            }

            return colParamCopy;
        }

        public virtual void InitTinyNet(bool bForceBackward = false, bool bAccuracyLayer = false)
        {
            string proto =
                "name: 'TinyTestNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    shape { " +
                "      dim: 5 " +
                "      dim: 2 " +
                "      dim: 3 " +
                "      dim: 4 " +
                "    } " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    shape { " +
                "      dim: 5 " +
                "    } " +
                "    data_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  top: 'data' " +
                "  top: 'label' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 1000 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  param { " +
                "    lr_mult: 1 " +
                "    decay_mult: 1 " +
                "  } " +
                "  param { " +
                "    lr_mult: 2 " +
                "    decay_mult: 0 " +
                "  } " +
                "  bottom: 'data' " +
                "  top: 'innerproduct' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'SoftmaxWithLoss' " +
                "  bottom: 'innerproduct' " +
                "  bottom: 'label' " +
                "  top: 'top_loss' " +
                "} ";
            if (bAccuracyLayer)
            {
                proto +=
                "layer { " +
                "  name: 'loss' " +
                "  type: 'Accuracy' " +
                "  bottom: 'innerproduct' " +
                "  bottom: 'label' " +
                "  top: 'accuracy' " +
                "} ";
            }
            if (bForceBackward)
                proto += "force_backward: true ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitTinyNetEuclidean(bool bForceBackward = false)
        {
            string proto =
                "name: 'TinyTestEuclidLossNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    num: 5 " +
                "    channels: 2 " +
                "    height: 3 " +
                "    width: 4 " +
                "    num: 5 " +
                "    channels: 1 " +
                "    height: 1 " +
                "    width: 1 " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "  } " +
                "  top: 'data' " +
                "  top: 'label' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 1 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  param { " +
                "    lr_mult: 1 " +
                "    decay_mult: 1 " +
                "  } " +
                "  param { " +
                "    lr_mult: 2 " +
                "    decay_mult: 0 " +
                "  } " +
                "  bottom: 'data' " +
                "  top: 'innerproduct' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'EuclideanLoss' " +
                "  bottom: 'innerproduct' " +
                "  bottom: 'label' " +
                "} ";
            if (bForceBackward)
                proto += "force_backward: true ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitTrickyNet(double? dfLossWeight = null)
        {
            string proto1 =
                "name: 'TrickyTestNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    num: 5 " +
                "    channels: 2 " +
                "    height: 3 " +
                "    width: 4 " +
                "    num: 5 " +
                "    channels: 1 " +
                "    height: 1 " +
                "    width: 1 " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "  } " +
                "  top: 'data' " +
                "  top: 'label' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 1000 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  param { " +
                "    lr_mult: 1 " +
                "    decay_mult: 1 " +
                "  } " +
                "  param { " +
                "    lr_mult: 2 " +
                "    decay_mult: 0 " +
                "  } " +
                "  bottom: 'data' " +
                "  top: 'transformed_data' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 1 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  param { " +
                "    lr_mult: 1 " +
                "    decay_mult: 1 " +
                "  } " +
                "  param { " +
                "    lr_mult: 2 " +
                "    decay_mult: 0 " +
                "  } " +
                "  bottom: 'label' " +
                "  top: 'transformed_label' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'SoftmaxWithLoss' ";
            string proto2 = 
                "  bottom: 'transformed_data' " +
                "  bottom: 'transformed_label' " +
                "} ";
            string proto = proto1;

            if (dfLossWeight.HasValue)
                proto += "  loss_weight: " + dfLossWeight.Value.ToString() + " ";

            proto += proto2;

            InitNetFromProtoString(proto);
        }

        public virtual void InitUnsharedWeightsNet(double? dfLossWeight = null, double? dfMidnetLossWeight = null, bool bForceBackward = false, bool bBiasTerm = false, double blobs_lr_w1 = 1, double blobs_lr_b1 = 2, double blobs_lr_w2 = 1, double blobs_lr_b2 = 2)
        {
            string bias_str = (bBiasTerm) ? "true " : "false ";
            string proto = "name: 'UnsharedWeightsNetwork' ";

            if (bForceBackward)
                proto += "force_backward: true ";

            proto +=
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    num: 5 " +
                "    channels: 2 " +
                "    height: 3 " +
                "    width: 4 " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "  } " +
                "  top: 'data' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct1' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +           
                "    bias_term: " +  bias_str +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 10 " +
                "    } " +
                "  } " +
                "  param { " +
                "    name: 'unsharedweights1' " +
                "    lr_mult: " + blobs_lr_w1.ToString() +
                "  } ";
            if (bBiasTerm)
                proto += "  param { lr_mult: " + blobs_lr_b1.ToString() + " } ";

            proto += 
                "  bottom: 'data' " +
                "  top: 'innerproduct1' ";

            if (dfMidnetLossWeight.HasValue)
                proto += "  loss_weight: " + dfMidnetLossWeight.Value.ToString() + " ";

            proto +=
                "} " +
                "layer { " +
                "  name: 'innerproduct2' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: " + bias_str +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 10 " +
                "    } " +
                "  } " +
                "  param { " +
                "    name: 'unsharedweights2' " +
                "    lr_mult: " + blobs_lr_w2.ToString() +
                "  } ";

            if (bBiasTerm)
                proto += "  param { lr_mult: " + blobs_lr_b2.ToString() + " } ";

            proto +=
                "  bottom: 'data' " +
                "  top: 'innerproduct2' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'EuclideanLoss' ";

            if (dfLossWeight.HasValue)
                proto += "  loss_weight: " + dfLossWeight.Value.ToString() + " ";

            proto += 
                "  bottom: 'innerproduct1' " +
                "  bottom: 'innerproduct2' " +
                "} ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitSharedWeightsNet()
        {
            string proto =
                "name: 'SharedWeightsNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    num: 5 " +
                "    channels: 2 " +
                "    height: 3 " +
                "    width: 4 " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "  } " +
                "  top: 'data' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct1' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: false " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 10 " +
                "    } " +
                "  } " +
                "  param { name: 'sharedweights' } " +
                "  bottom: 'data' " +
                "  top: 'innerproduct1' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct2' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: false " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 10 " +
                "    } " +
                "  } " +
                "  param { name: 'sharedweights' } " +
                "  bottom: 'data' " +
                "  top: 'innerproduct2' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'EuclideanLoss' " +
                "  bottom: 'innerproduct1' " +
                "  bottom: 'innerproduct2' " +
                "} ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitDiffDataUnsharedWeightsNet()
        {
            string proto =
                "name: 'DiffDataUnsharedWeightsNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    num: 10 " +
                "    channels: 10 " +
                "    height: 1 " +
                "    width: 1 " +
                "    num: 10 " +
                "    channels: 10 " +
                "    height: 1 " +
                "    width: 1 " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 10 " +
                "    } " +
                "  } " +
                "  top: 'data1' " +
                "  top: 'data2' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct1' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: false " +
                "    weight_filler { " +
                "      type: 'constant' " +
                "      value: 0.5 " +
                "    } " +
                "  } " +
                "  param { name: 'unsharedweights1' } " +
                "  bottom: 'data1' " +
                "  top: 'innerproduct1' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct2' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: false " +
                "    weight_filler { " +
                "      type: 'constant' " +
                "      value: 0.5 " +
                "    } " +
                "  } " +
                "  param { name: 'unsharedweights2' } " +
                "  bottom: 'innerproduct1' " +
                "  top: 'innerproduct2' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'EuclideanLoss' " +
                "  bottom: 'data2' " +
                "  bottom: 'innerproduct2' " +
                "} ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitDiffDataSharedWeightsNet()
        {
            string proto =
                "name: 'DiffDataSharedWeightsNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    num: 10 " +
                "    channels: 10 " +
                "    height: 1 " +
                "    width: 1 " +
                "    num: 10 " +
                "    channels: 10 " +
                "    height: 1 " +
                "    width: 1 " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 10 " +
                "    } " +
                "  } " +
                "  top: 'data1' " +
                "  top: 'data2' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct1' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: false " +
                "    weight_filler { " +
                "      type: 'constant' " +
                "      value: 0.5 " +
                "    } " +
                "  } " +
                "  param { name: 'sharedweights' } " +
                "  bottom: 'data1' " +
                "  top: 'innerproduct1' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct2' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 10 " +
                "    bias_term: false " +
                "    weight_filler { " +
                "      type: 'constant' " +
                "      value: 0.5 " +
                "    } " +
                "  } " +
                "  param { name: 'sharedweights' } " +
                "  bottom: 'innerproduct1' " +
                "  top: 'innerproduct2' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  type: 'EuclideanLoss' " +
                "  bottom: 'data2' " +
                "  bottom: 'innerproduct2' " +
                "} ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitReshapableNet()
        {
            string proto =
                "name: 'ReshapableNetwork' " +
                "input: 'data' " +
                "input_dim: 1 " +
                "input_dim: 3 " +
                "input_dim: 100 " +
                "input_dim: 100 " +
                "layer { " +
                "  name: 'conv1' " +
                "  type: 'Convolution' " +
                "  bottom: 'data' " +
                "  top: 'conv1' " +
                "  convolution_param { " +
                "    num_output: 5 " +
                "    kernel_size: 3 " +
                "    stride: 2 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0.2 " +
                "    } " +
                "  } " +
                "} " +
                "layer { " +
                "  name: 'relu1' " +
                "  type: 'ReLU' " +
                "  bottom: 'conv1' " +
                "  top: 'conv1' " +
                "} " +
                "layer { " +
                "  name: 'pool1' " +
                "  type: 'Pooling' " +
                "  bottom: 'conv1' " +
                "  top: 'pool1' " +
                "  pooling_param { " +
                "    pool: MAX " +
                "    kernel_size: 2 " +
                "    stride: 2 " +
                "  } " +
                "} " +
                "layer { " +
                "  name: 'norm1' " +
                "  type: 'LRN' " +
                "  bottom: 'pool1' " +
                "  top: 'norm1' " +
                "  lrn_param { " +
                "    local_size: 3 " +
                "  } " +
                "} " +
                "layer { " +
                "  name: 'softmax' " +
                "  type: 'Softmax' " +
                "  bottom: 'norm1' " +
                "  top: 'softmax' " +
                "} ";

            InitNetFromProtoString(proto);
        }

        public virtual void InitSkipPropNet(bool bTestSkipTrue)
        {
            string proto =
                "name: 'SkipPropTestNetwork' " +
                "layer { " +
                "  name: 'data' " +
                "  type: 'DummyData' " +
                "  dummy_data_param { " +
                "    shape { " +
                "      dim: 5 " +
                "      dim: 2 " +
                "      dim: 3 " +
                "      dim: 4 " +
                "    } " +
                "    data_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    shape { " +
                "      dim: 5 " +
                "    } " +
                "    data_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  top: 'data' " +
                "  top: 'label' " +
                "} " +
                "layer { " +
                "  name: 'silence' " +
                "  bottom: 'label' " +
                "  type: 'Silence' " +
                "} " +
                "layer { " +
                "  name: 'innerproduct' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 1 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  param { " +
                "    lr_mult: 1 " +
                "    decay_mult: 1 " +
                "  } " +
                "  param { " +
                "    lr_mult: 2 " +
                "    decay_mult: 0 " +
                "  } " +
                "  bottom: 'data' " +
                "  top: 'innerproduct' " +
                "} " +
                "layer { " +
                "  name: 'ip_fake_labels' " +
                "  type: 'InnerProduct' " +
                "  inner_product_param { " +
                "    num_output: 1 " +
                "    weight_filler { " +
                "      type: 'gaussian' " +
                "      std: 0.01 " +
                "    } " +
                "    bias_filler { " +
                "      type: 'constant' " +
                "      value: 0 " +
                "    } " +
                "  } " +
                "  bottom: 'data' " +
                "  top: 'fake_labels' " +
                "} " +
                "layer { " +
                "  name: 'argmax' " +
                "  bottom: 'fake_labels' " +
                "  top: 'label_argmax' " +
                "  type: 'ArgMax' " +
                "} " +
                "layer { " +
                "  name: 'loss' " +
                "  bottom: 'innerproduct' " +
                "  bottom: 'label_argmax' ";
            if (bTestSkipTrue)
            {
                proto += "  propagate_down: true " +
                         "  propagate_down: false ";
            }
            else
            {
                proto += "  propagate_down: true " +
                         "  propagate_down: true ";
            }

            proto +=
               "  top: 'cross_entropy_loss' " +
               "  type: 'SigmoidCrossEntropyLoss' " +
               "  loss_weight: 0.1 " +
               "} ";                                      

            InitNetFromProtoString(proto);
        }

        public virtual void InitForcePropNet(bool bTestForceTrue)
        {
            string strProto =
               "name: 'ForcePropTestNetwork' " +
               "layer { " +
               "  name: 'data' " +
               "  type: 'DummyData' " +
               "  dummy_data_param { " +
               "    shape { " +
               "      dim: 5 " +
               "      dim: 2 " +
               "      dim: 3 " +
               "      dim: 4 " +
               "    } " +
               "    data_filler { " +
               "      type: 'gaussian' " +
               "      std: 0.01 " +
               "    } " +
               "    shape { " +
               "      dim: 5 " +
               "    } " +
               "    data_filler { " +
               "      type: 'constant' " +
               "      value: 0 " +
               "    } " +
               "  } " +
               "  top: 'data' " +
               "  top: 'label' " +
               "} " +
               "layer { " +
               "  name: 'innerproduct' " +
               "  type: 'InnerProduct' " +
               "  inner_product_param { " +
               "    num_output: 1 " +
               "    weight_filler { " +
               "      type: 'gaussian' " +
               "      std: 0.01 " +
               "    } " +
               "  } " +
               "  bottom: 'data' " +
               "  top: 'innerproduct' ";

            if (bTestForceTrue)
                strProto += " propagate_down: true ";

            strProto +=
               "} " +
               "layer { " +
               "  name: 'loss' " +
               "  bottom: 'innerproduct' " +
               "  bottom: 'label' " +
               "  top: 'cross_entropy_loss' " +
               "  type: 'SigmoidCrossEntropyLoss' " +
               "} ";

            InitNetFromProtoString(strProto);
        }

        public virtual void InitAllInOneNet(Phase phase = Phase.TRAIN, int nLevel = 0, List<string> rgStages = null)
        {
            string strProto =
              "name: 'All-in-one Network'" +
              "layer { " +
              "  name: 'train-data' " +
              "  type: 'DummyData' " +
              "  top: 'data' " +
              "  top: 'label' " +
              "  dummy_data_param { " +
              "    shape { dim: 1 dim: 10 } " +
              "    shape { dim: 1 dim: 1 } " +
              "  } " +
              "  include { phase: TRAIN stage: 'train' } " +
              "} " +
              "layer { " +
              "  name: 'val-data' " +
              "  type: 'DummyData' " +
              "  top: 'data' " +
              "  top: 'label' " +
              "  dummy_data_param { " +
              "    shape { dim: 1 dim: 10 } " +
              "    shape { dim: 1 dim: 1 } " +
              "  } " +
              "  include { phase: TEST stage: 'val' } " +
              "} " +
              "layer { " +
              "  name: 'deploy-data' " +
              "  type: 'Input' " +
              "  top: 'data' " +
              "  input_param { " +
              "    shape { dim: 1 dim: 10 } " +
              "  } " +
              "  include { phase: TEST stage: 'deploy' } " +
              "} " +
              "layer { " +
              "  name: 'ip' " +
              "  type: 'InnerProduct' " +
              "  bottom: 'data' " +
              "  top: 'ip' " +
              "  inner_product_param { " +
              "    num_output: 2 " +
              "  } " +
              "} " +
              "layer { " +
              "  name: 'loss' " +
              "  type: 'SoftmaxWithLoss' " +
              "  bottom: 'ip' " +
              "  bottom: 'label' " +
              "  top: 'loss' " +
              "  include { phase: TRAIN stage: 'train' } " +
              "  include { phase: TEST stage: 'val' } " +
              "} ";

            InitNetFromProtoFileWithState(strProto, phase, nLevel, rgStages);
        }

        public void TestHasBlob()
        {
            InitTinyNet();
            m_log.CHECK(m_net.has_blob("data"), "The net should have the 'data' blob.");
            m_log.CHECK(m_net.has_blob("label"), "The net should have the 'label' blob.");
            m_log.CHECK(m_net.has_blob("innerproduct"), "The net should have the 'innerproduct' blob.");
            m_log.CHECK(!m_net.has_blob("loss"), "The net should NOT have the 'loss' blob.");
            m_log.CHECK(m_net.has_blob("top_loss"), "The net should have the 'top_loss' blob.");
        }

        public void TestGetBlob()
        {
            InitTinyNet();
            m_log.CHECK(m_net.blob_by_name("data") == m_net.blobs[0], "The 'data' blob should be at blob[0].");
            m_log.CHECK(m_net.blob_by_name("label") == m_net.blobs[1], "The 'label' blob should be at blob[1].");
            m_log.CHECK(m_net.blob_by_name("innerproduct") == m_net.blobs[2], "The 'innerproduct' blob should be at blob[2].");
            m_log.CHECK(m_net.blob_by_name("top_loss") == m_net.blobs[3], "The 'top_loss' blob should be at blob[3].");
        }

        public void TestHasLayer()
        {
            InitTinyNet();
            m_log.CHECK(m_net.has_layer("data"), "The net should have the 'data' layer.");
            m_log.CHECK(m_net.has_layer("innerproduct"), "The net should have the 'innerproduct' layer.");
            m_log.CHECK(m_net.has_layer("loss"), "The net should have the 'loss' layer.");
            m_log.CHECK(!m_net.has_layer("label"), "The net should NOT have the 'label' layer.");
        }

        public void TestGetLayer()
        {
            InitTinyNet();
            m_log.CHECK(m_net.layer_by_name("data") == m_net.layers[0], "The 'data' layer should equal layers[0].");
            m_log.CHECK(m_net.layer_by_name("innerproduct") == m_net.layers[1], "The 'innerproduct' layer should equal layers[1].");
            m_log.CHECK(m_net.layer_by_name("loss") == m_net.layers[2], "The 'loss' layer should equal layers[2].");
        }

        public void TestBottomNeedBackward()
        {
            InitTinyNet();
            List<List<bool>> rgrgBottomNeedBackward = m_net.bottom_need_backward;
            m_log.CHECK_EQ(3, rgrgBottomNeedBackward.Count, "The bottom need backward list should have 3 entries.");
            m_log.CHECK_EQ(0, rgrgBottomNeedBackward[0].Count, "The first list in bottom need backward should have 0 entries.");
            m_log.CHECK_EQ(1, rgrgBottomNeedBackward[1].Count, "The second list in the bottom need backward should have 1 entries.");
            m_log.CHECK(false == rgrgBottomNeedBackward[1][0], "The entry in the second list of bottom need backward shoudl be false.");
            m_log.CHECK_EQ(2, rgrgBottomNeedBackward[2].Count, "The third list in the bottom need backward should have 2 entries.");
            m_log.CHECK(true == rgrgBottomNeedBackward[2][0], "The first entry in the third list in the bottom need backward should be true.");
            m_log.CHECK(false == rgrgBottomNeedBackward[2][1], "The second entry in the third list in the bottom need backward should be false.");
        }

        public void TestBottomNeedBackwardForce()
        {
            bool bForceBackward = true;
            InitTinyNet(bForceBackward);
            List<List<bool>> rgrgBottomNeedBackward = m_net.bottom_need_backward;
            m_log.CHECK_EQ(3, rgrgBottomNeedBackward.Count, "The bottom need backward list should have 3 entries.");
            m_log.CHECK_EQ(0, rgrgBottomNeedBackward[0].Count, "The first list in bottom need backward should have 0 entries.");
            m_log.CHECK_EQ(1, rgrgBottomNeedBackward[1].Count, "The second list in the bottom need backward should have 1 entries.");
            m_log.CHECK(true == rgrgBottomNeedBackward[1][0], "The entry in the second list of bottom need backward shoudl be true.");
            m_log.CHECK_EQ(2, rgrgBottomNeedBackward[2].Count, "The third list in the bottom need backward should have 2 entries.");
            m_log.CHECK(true == rgrgBottomNeedBackward[2][0], "The first entry in the third list in the bottom need backward should be true.");
            m_log.CHECK(false == rgrgBottomNeedBackward[2][1], "The second entry in the third list in the bottom need backward should be false.");
        }

        public void TestBottomNeedBackwardEuclideanForce()
        {
            bool bForceBackward = true;
            InitTinyNetEuclidean(bForceBackward);
            List<List<bool>> rgrgBottomNeedBackward = m_net.bottom_need_backward;
            m_log.CHECK_EQ(3, rgrgBottomNeedBackward.Count, "The bottom need backward list should have 3 entries.");
            m_log.CHECK_EQ(0, rgrgBottomNeedBackward[0].Count, "The first list in bottom need backward should have 0 entries.");
            m_log.CHECK_EQ(1, rgrgBottomNeedBackward[1].Count, "The second list in the bottom need backward should have 1 entries.");
            m_log.CHECK(true == rgrgBottomNeedBackward[1][0], "The entry in the second list of bottom need backward shoudl be true.");
            m_log.CHECK_EQ(2, rgrgBottomNeedBackward[2].Count, "The third list in the bottom need backward should have 2 entries.");
            m_log.CHECK(true == rgrgBottomNeedBackward[2][0], "The first entry in the third list in the bottom need backward should be true.");
            m_log.CHECK(true == rgrgBottomNeedBackward[2][1], "The second entry in the third list in the bottom need backward should be true.");
        }

        public void TestBottomNeedBackwardTricky()
        {
            InitTrickyNet();
            List<List<bool>> rgrgBottomNeedBackward = m_net.bottom_need_backward;
            m_log.CHECK_EQ(4, rgrgBottomNeedBackward.Count, "The bottom need backward list should have 4 entries.");
            m_log.CHECK_EQ(0, rgrgBottomNeedBackward[0].Count, "The first list in bottom need backward should have 0 entries.");
            m_log.CHECK_EQ(1, rgrgBottomNeedBackward[1].Count, "The second list in the bottom need backward should have 1 entries.");
            m_log.CHECK(false == rgrgBottomNeedBackward[1][0], "The entry in the second list of bottom need backward shoudl be false.");
            m_log.CHECK_EQ(1, rgrgBottomNeedBackward[2].Count, "The third list in the bottom need backward should have 2 entries.");
            m_log.CHECK(false == rgrgBottomNeedBackward[2][0], "The first entry in the third list in the bottom need backward should be false.");
            m_log.CHECK_EQ(2, rgrgBottomNeedBackward[3].Count, "The fourty list in the bottom need backward should have 2 entries.");
            m_log.CHECK(true == rgrgBottomNeedBackward[3][0], "The first entry in the fourth list in the bottom need backward should be true.");
            // The label input to the SoftmaxLossLayer should say it 'needs backward'
            // since it has weights under it, even though we expect this to cause a crash
            // at training/testing time.
            m_log.CHECK(true == rgrgBottomNeedBackward[3][1], "The second entry in the fourth list in the bottom need backward should be true.");
        }

        public void TestLossWeight()
        {
            // First, compute the loss and gradients with no loss_weight specified.
            // In this case, the loss weight for the 'EuclideanLoss' layer should default
            // to 1.
            BlobCollection<T> colBottom = new BlobCollection<T>();
            m_cuda.rng_setseed(m_lSeed);
            bool kForceBackward = true;
            InitUnsharedWeightsNet(null, null, kForceBackward);
            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
            bool kCopyDiff = true;
            BlobCollection<T> colBlobGrads = CopyNetBlobs(kCopyDiff);
            BlobCollection<T> colParamGrads = CopyNetParams(kCopyDiff);

            // Check that the loss is non-trivial, otherwise the test doesn't prove much.
            double kMinLossAbsValue = 1e-2;
            m_log.CHECK_GE(Math.Abs(dfLoss), kMinLossAbsValue, "The loss (" + dfLoss.ToString() + ") is less than the min loss.");
            double kErrorMargin = 1e-4;
            double[] kLossWeights = new double[] { 2, 0, 1, -1, -2.5, 3.7 };

            for (int i = 0; i < kLossWeights.Length; i++)
            {
                m_cuda.rng_setseed(m_lSeed);
                InitUnsharedWeightsNet(kLossWeights[i], null, kForceBackward);
                double dfWeightedLoss;
                m_net.ForwardBackward(colBottom, out dfWeightedLoss);
                double dfErrorMargin = kErrorMargin * Math.Abs(kLossWeights[i]);
                m_log.EXPECT_NEAR(dfLoss * kLossWeights[i], dfWeightedLoss, dfErrorMargin, "loss_weight = " + kLossWeights[i].ToString());

                BlobCollection<T> colWeightedBlobs = m_net.blobs;
                m_log.CHECK_EQ(colBlobGrads.Count, colWeightedBlobs.Count, "The blob grad count should equal the weighted blob count.");

                for (int j = 0; j < colBlobGrads.Count; j++)
                {
                    m_log.CHECK_EQ(colBlobGrads[j].count(), colWeightedBlobs[j].count(), "The blob count() at " + j.ToString() + " for blob grads and weighted blobs are not the same!");

                    double[] rgdfBlobGrads = convert(colBlobGrads[j].update_cpu_diff());
                    double[] rgdfWeightedBlobs = convert(colWeightedBlobs[j].update_cpu_diff());

                    for (int k = 0; k < colBlobGrads[j].count(); k++)
                    {
                        double dfBlobGrad = rgdfBlobGrads[k];
                        double dfWeightedBlob = rgdfWeightedBlobs[k];

                        m_log.EXPECT_NEAR(dfBlobGrad * kLossWeights[i], dfWeightedBlob, dfErrorMargin);
                    }
                }

                BlobCollection<T> colWeightedParam = m_net.parameters;
                m_log.CHECK_EQ(colParamGrads.Count, colWeightedParam.Count, "The param grad count and weighted param grad count are not the same!");

                for (int j = 0; j < colParamGrads.Count; j++)
                {
                    m_log.CHECK_EQ(colParamGrads[j].count(), colWeightedParam[j].count(), "The blob count at " + j.ToString() + " for the param grads and weighted param grads are not the same!");

                    double[] rgdfParamGrads = convert(colParamGrads[j].update_cpu_diff());
                    double[] rgdfWeightedParams = convert(colWeightedParam[j].update_cpu_diff());

                    for (int k = 0; k < colParamGrads[j].count(); k++)
                    {
                        double dfParamGrad = rgdfParamGrads[k];
                        double dfWeightedParam = rgdfWeightedParams[k];

                        m_log.EXPECT_NEAR(dfParamGrad * kLossWeights[i], dfWeightedParam, dfErrorMargin);
                    }
                }
            }
        }

        public void TestLossWeightMidNet()
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            m_cuda.rng_setseed(m_lSeed);
            bool kForceBackward = true;
            double dfLossWeight = 0;
            double dfMidNetLossWeight = 1;

            InitUnsharedWeightsNet(dfLossWeight, dfMidNetLossWeight, kForceBackward);
            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
            bool kCopyDiff = true;
            bool kReshape = true;
            Blob<T> blobDataGrad = new Blob<T>(m_cuda, m_log);
            blobDataGrad.CopyFrom(m_net.blob_by_name("data"), kCopyDiff, kReshape);
            
            // Check that the loss is non-trivial, otherwise the test doesn't prove much.
            double kMinLossAbsValue = 1e-2;
            m_log.CHECK_GE(Math.Abs(dfLoss), kMinLossAbsValue, "The loss (" + dfLoss.ToString() + ") is less than the min loss.");
            double kErrorMargin = 1e-4;
            double[] kLossWeights = new double[] { 2, 0, 1, -1, -2.5, 3.7 };

            for (int i = 0; i < kLossWeights.Length; i++)
            {
                m_cuda.rng_setseed(m_lSeed);
                InitUnsharedWeightsNet(dfLossWeight, kLossWeights[i], kForceBackward);
                double dfWeightedLoss;
                m_net.ForwardBackward(colBottom, out dfWeightedLoss);
                double dfErrorMargin = kErrorMargin * Math.Abs(kLossWeights[i]);
                m_log.EXPECT_NEAR(dfLoss * kLossWeights[i], dfWeightedLoss, dfErrorMargin, "loss_weight = " + kLossWeights[i].ToString());

                Blob<T> weightedBlob = m_net.blob_by_name("data");
                m_log.CHECK_EQ(blobDataGrad.count(), weightedBlob.count(), "The data blob grad count should equal the weighted blob count.");

                double[] rgdfDataGrad = convert(blobDataGrad.update_cpu_diff());
                double[] rgdfWeightedBlob = convert(weightedBlob.update_cpu_diff());

                for (int j = 0; j < blobDataGrad.count(); j++)
                {
                    double dfExpected = rgdfDataGrad[j] * kLossWeights[i];
                    double dfActual = rgdfWeightedBlob[j];

                    m_log.EXPECT_NEAR(dfExpected, dfActual, dfErrorMargin);
                }
            }
        }

        public void TestComboLossWeight()
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            double dfLossWeight;
            double dfMidNetLossWeight;
            bool kForceBackward = true;
            double kErrorMargin = 1e-4;

            // Get the loss and gradients with 'EuclideanLoss' weight 1,
            // 'InnerProduct' weight 1.
            dfLossWeight = 1;
            dfMidNetLossWeight = 1;
            m_cuda.rng_setseed(m_lSeed);
            InitUnsharedWeightsNet(dfLossWeight, dfMidNetLossWeight, kForceBackward);

            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
            bool kCopyDiff = true;
            BlobCollection<T> colBlobGrads = CopyNetBlobs(kCopyDiff);
            BlobCollection<T> colParamGrads = CopyNetParams(kCopyDiff);

            dfLossWeight = 2;
            dfMidNetLossWeight = 1;
            m_cuda.rng_setseed(m_lSeed);
            InitUnsharedWeightsNet(dfLossWeight, dfMidNetLossWeight, kForceBackward);

            double dfLossMain2;
            m_net.ForwardBackward(colBottom, out dfLossMain2);
            BlobCollection<T> colBlobGradsLoss2 = CopyNetBlobs(kCopyDiff);
            BlobCollection<T> colParamGradsLoss2 = CopyNetParams(kCopyDiff);

            dfLossWeight = 3;
            dfMidNetLossWeight = 1;
            m_cuda.rng_setseed(m_lSeed);
            InitUnsharedWeightsNet(dfLossWeight, dfMidNetLossWeight, kForceBackward);

            double dfLossMain3;
            m_net.ForwardBackward(colBottom, out dfLossMain3);
            BlobCollection<T> colBlobGradsLoss3 = m_net.blobs;

            m_log.CHECK_EQ(colBlobGrads.Count, colBlobGradsLoss3.Count, "The blob grads count and blob grads loss 3 counts should match.");
            m_log.CHECK_EQ(colBlobGradsLoss2.Count, colBlobGradsLoss3.Count, "The blob grads loss 2 and blob grads loss 3 counts should match.");

            for (int j = 0; j < colBlobGrads.Count; j++)
            {
                string blob_name = m_net.blob_names[j];
                bool bGradShouldChange = true;

                if (blob_name == "innerproduct1_innerproduct1_0_split_0")
                    bGradShouldChange = false;

                m_log.CHECK_EQ(colBlobGrads[j].count(), colBlobGradsLoss3[j].count(), "The counts at " + j.ToString() + " for blob grads and blob grads loss 3 should be the same.");
                m_log.CHECK_EQ(colBlobGradsLoss2[j].count(), colBlobGradsLoss3[j].count(), "The counts at " + j.ToString() + " for blob grads loss 2 and blob grads loss 3 should be the same.");

                double[] rgdfBlobGrads = convert(colBlobGrads[j].update_cpu_diff());
                double[] rgdfBlobGrads2 = convert(colBlobGradsLoss2[j].update_cpu_diff());
                double[] rgdfBlobGrads3 = convert(colBlobGradsLoss3[j].update_cpu_diff());

                for (int k = 0; k < colBlobGrads[j].count(); k++)
                {
                    double dfGradDiff2 = rgdfBlobGrads2[k] - rgdfBlobGrads[k];
                    double dfGradDiff3 = rgdfBlobGrads3[k] - rgdfBlobGrads[k];

                    if (bGradShouldChange)
                    {
                        // Test non-triviality
                        double kMinGradDiffAbsValue = 1e-4;
                        m_log.CHECK_GT(Math.Abs(dfGradDiff2), kMinGradDiffAbsValue, "Test non-triviality of '" + blob_name + "'");
                        m_log.EXPECT_NEAR(2 * dfGradDiff2, dfGradDiff3, kErrorMargin, blob_name);
                    }
                    else
                    {
                        m_log.CHECK_EQ(0, dfGradDiff2, blob_name);
                        m_log.CHECK_EQ(0, dfGradDiff3, blob_name);
                    }
                }
            }

            dfLossWeight = 1;
            dfMidNetLossWeight = 2;
            m_cuda.rng_setseed(m_lSeed);
            InitUnsharedWeightsNet(dfLossWeight, dfMidNetLossWeight, kForceBackward);

            double dfLossMidNet2;
            m_net.ForwardBackward(colBottom, out dfLossMidNet2);
            colBlobGradsLoss2 = CopyNetBlobs(kCopyDiff);
            colParamGradsLoss2 = CopyNetParams(kCopyDiff);

            dfLossWeight = 1;
            dfMidNetLossWeight = 3;
            m_cuda.rng_setseed(m_lSeed);
            InitUnsharedWeightsNet(dfLossWeight, dfMidNetLossWeight, kForceBackward);

            double dfLossMidNet3;
            m_net.ForwardBackward(colBottom, out dfLossMidNet3);
            BlobCollection<T> colBlobGradsMidNetLoss3 = m_net.blobs;

            m_log.CHECK_EQ(colBlobGrads.Count, colBlobGradsMidNetLoss3.Count, "The blob grads count and blob grads midnet loss 3 counts should match.");
            m_log.CHECK_EQ(colBlobGradsLoss2.Count, colBlobGradsMidNetLoss3.Count, "The blob grads loss 2 and blob grads midnet loss 3 counts should match.");

            for (int j = 0; j < colBlobGrads.Count; j++)
            {
                string blob_name = m_net.blob_names[j];
                bool bGradShouldChange = false;

                if (blob_name == "innerproduct1" ||
                    blob_name == "innerproduct1_innerproduct1_0_split_0" ||
                    blob_name == "data_data_0_split_0" ||
                    blob_name == "data")
                    bGradShouldChange = true;

                m_log.CHECK_EQ(colBlobGrads[j].count(), colBlobGradsMidNetLoss3[j].count(), "The counts at " + j.ToString() + " for blob grads and blob grads midnet loss 3 should be the same.");
                m_log.CHECK_EQ(colBlobGradsLoss2[j].count(), colBlobGradsMidNetLoss3[j].count(), "The counts at " + j.ToString() + " for blob grads loss 2 and blob grads midnet loss 3 should be the same.");

                double[] rgdfBlobGrads = convert(colBlobGrads[j].update_cpu_diff());
                double[] rgdfBlobGrads2 = convert(colBlobGradsLoss2[j].update_cpu_diff());
                double[] rgdfBlobGrads3 = convert(colBlobGradsMidNetLoss3[j].update_cpu_diff());

                for (int k = 0; k < colBlobGrads[j].count(); k++)
                {
                    double dfGradDiff2 = rgdfBlobGrads2[k] - rgdfBlobGrads[k];
                    double dfGradDiff3 = rgdfBlobGrads3[k] - rgdfBlobGrads[k];

                    if (bGradShouldChange)
                    {
                        // Test non-triviality
                        double kMinGradDiffAbsValue = 1e-4;
                        m_log.CHECK_GT(Math.Abs(dfGradDiff2), kMinGradDiffAbsValue, "Test non-triviality of '" + blob_name + "'");
                        m_log.EXPECT_NEAR(2 * dfGradDiff2, dfGradDiff3, kErrorMargin, blob_name);
                    }
                    else
                    {
                        m_log.CHECK_EQ(0, dfGradDiff2, blob_name);
                        m_log.CHECK_EQ(0, dfGradDiff3, blob_name);
                    }
                }
            }

            double kMinLossDiffAbsValue = 1e-4;

            double dfLossDiff2 = dfLossMain2 - dfLoss;
            // Test non-triviality
            m_log.CHECK_GT(Math.Abs(dfLossDiff2), kMinLossDiffAbsValue, "Loss Diff 2 should be >= 1e-4.");
            double dfLossDiff3 = dfLossMain3 - dfLoss;
            m_log.EXPECT_NEAR(2 * dfLossDiff2, dfLossDiff3, kErrorMargin);

            dfLossDiff2 = dfLossMidNet2 - dfLoss;
            // Test non-triviality.
            m_log.CHECK_GT(Math.Abs(dfLossDiff2), kMinLossDiffAbsValue, "Loss Diff 2 should be >= 1e-4.");
            dfLossDiff3 = dfLossMidNet3 - dfLoss;
            m_log.EXPECT_NEAR(2 * dfLossDiff2, dfLossDiff3, kErrorMargin);
        }

        public void TestBackwardWithAccuracyLayer()
        {
            bool kForceBackward = false;
            bool kAccuracyLayer = true;
            InitTinyNet(kForceBackward, kAccuracyLayer);
            m_log.CHECK(m_net.has_blob("accuracy"), "The net should have the 'accuracy' layer.");
            BlobCollection<T> colBottom = new BlobCollection<T>();
            // Test that we can do Backward even though we have an accuracy layer.
            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
        }

        public void TestUnsharedWeightsDataNet()
        {
            InitUnsharedWeightsNet();
            BlobCollection<T> colBottom = new BlobCollection<T>();
            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
            m_log.CHECK_GT(dfLoss, 0, "The loss should be > 0.");
        }

        public void TestSharedWeightsDataNet()
        {
            InitSharedWeightsNet();
            BlobCollection<T> colBottom = new BlobCollection<T>();
            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
            m_log.CHECK_EQ(dfLoss, 0, "The loss should be == 0.");
        }

        public void TestUnsharedWeightsDiffNet()
        {
            InitUnsharedWeightsNet();
            BlobCollection<T> colBottom = new BlobCollection<T>();
            double dfLoss;
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();
            Layer<T> ip1_layer = m_net.layer_by_name("innerproduct1");
            Layer<T> ip2_layer = m_net.layer_by_name("innerproduct2");
            int nCount = ip1_layer.blobs[0].count();
            double[] rgGrad1 = convert(ip1_layer.blobs[0].update_cpu_diff());
            double[] rgGrad2 = convert(ip2_layer.blobs[0].update_cpu_diff());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_GT(Math.Abs(rgGrad1[i]), 0.0, "The gradient at " + i.ToString() + " should be > 0.");
                m_log.CHECK_EQ(-1 * rgGrad1[i], rgGrad2[i], "The gradients at " + i.ToString() + " should be equal.");
            }
        }

        public void TestSharedWeightsDiffNet()
        {
            InitSharedWeightsNet();
            BlobCollection<T> colBottom = new BlobCollection<T>();
            double dfLoss;
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();

            m_log.CHECK_EQ(dfLoss, 0, "The loss should = 0.");

            Layer<T> ip1_layer = m_net.layer_by_name("innerproduct1");
            Layer<T> ip2_layer = m_net.layer_by_name("innerproduct2");
            int nCount = ip1_layer.blobs[0].count();
            double[] rgGrad1 = convert(ip1_layer.blobs[0].update_cpu_diff());
            double[] rgGrad2 = convert(ip2_layer.blobs[0].update_cpu_diff());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_EQ(0, rgGrad1[i], "The gradient 1 at index " + i.ToString() + " should be 0.");
                m_log.CHECK_EQ(0, rgGrad2[i], "The gradient 1 at index " + i.ToString() + " should be 0.");
            }
        }

        public void TestSharedWeightsUpdate()
        {
            m_cuda.rng_setseed(m_lSeed);
            InitDiffDataSharedWeightsNet();
            BlobCollection<T> colBottom = new BlobCollection<T>();

            m_log.CHECK(m_net.layer_names[1] == "innerproduct1", "The layer 1 should be the 'innerproduct1' layer.");
            m_log.CHECK(m_net.layer_names[2] == "innerproduct2", "The layer 2 should be the 'innerproduct2' layer.");

            Blob<T> ip1_weights = m_net.layers[1].blobs[0];
            Blob<T> ip2_weights = m_net.layers[2].blobs[0];

            // Check that data and diff blobs shared weights share the same memory locations
            m_log.CHECK(ip1_weights.gpu_data == ip2_weights.gpu_data, "The ip1 and ip2 weight data memory is not the same!");
            m_log.CHECK(ip1_weights.gpu_diff == ip2_weights.gpu_diff, "The ip1 and ip2 weight diff memory is not the same!");

            double dfLoss;
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();
            
            // Compute the expected update4 as the data minus the two diffs.
            Blob<T> blobSharedParams = new Blob<T>(m_cuda, m_log);
            bool kReshape = true;
            bool kCopyDiff = false;
            blobSharedParams.CopyFrom(ip1_weights, kCopyDiff, kReshape);
            blobSharedParams.CopyFrom(ip1_weights, !kCopyDiff, kReshape);
            int nCount = ip1_weights.count();

            // Make sure the diffs are non-trivial.
            double[] rgdfIpWeights = convert(ip1_weights.update_cpu_diff());
            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_NE(0.0, rgdfIpWeights[i], "The ip weights at " + i.ToString() + " is zero!");
            }

            m_cuda.axpy(nCount, -1.0, blobSharedParams.gpu_diff, blobSharedParams.mutable_gpu_data);
            double[] rgdfExpectedUpdatedParams = convert(blobSharedParams.mutable_cpu_data);

            m_net.Update();
            double[] rgdfActualUpdatedParams = convert(ip1_weights.mutable_cpu_data);

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_EQ(rgdfExpectedUpdatedParams[i], rgdfActualUpdatedParams[i], "The param values at " + i.ToString() + " are not the same!");
            }

            // Check that data blobs of shared weights SILL point to the same memory
            // locations (because ... who knows).
            m_log.CHECK(ip1_weights.gpu_data == ip2_weights.gpu_data, "The weight data memory is different!");

            m_cuda.rng_setseed(m_lSeed);
            InitDiffDataUnsharedWeightsNet();

            m_log.CHECK(m_net.layer_names[1] == "innerproduct1", "The layer 1 should be the 'innerproduct1' layer.");
            m_log.CHECK(m_net.layer_names[2] == "innerproduct2", "The layer 2 should be the 'innerproduct2' layer.");

            ip1_weights = m_net.layers[1].blobs[0];
            ip2_weights = m_net.layers[2].blobs[0];

            // Check that data and diff blobs shared weights are at different memory locations
            m_log.CHECK(ip1_weights.gpu_data != ip2_weights.gpu_data, "The ip1 and ip2 weight data memory are the same!");
            m_log.CHECK(ip1_weights.gpu_diff != ip2_weights.gpu_diff, "The ip1 and ip2 weight diff memory are the same!");

            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();

            // Compute the expected update.
            Blob<T> blobUnsharedParams1 = new Blob<T>(m_cuda, m_log);
            blobUnsharedParams1.CopyFrom(ip1_weights, kCopyDiff, kReshape);
            blobUnsharedParams1.CopyFrom(ip1_weights, !kCopyDiff, kReshape);
            Blob<T> blobUnsharedParams2 = new Blob<T>(m_cuda, m_log);
            blobUnsharedParams2.CopyFrom(ip2_weights, kCopyDiff, kReshape);
            blobUnsharedParams2.CopyFrom(ip2_weights, !kCopyDiff, kReshape);

            // Make sure the diffs are non-trivial and sum to the diffs in the shared net.
            double[] rgdfIpWeightsNonShared1 = convert(ip1_weights.update_cpu_diff());
            double[] rgdfIpWeightsNonShared2 = convert(ip2_weights.update_cpu_diff());
            double[] rgdfSharedParamsDiff = convert(blobSharedParams.update_cpu_diff());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_NE(0, rgdfIpWeightsNonShared1[i], "The ip non shared weight1 at " + i.ToString() + " is zero.");
                m_log.CHECK_NE(0, rgdfIpWeightsNonShared2[i], "The ip non shared weight2 at " + i.ToString() + " is zero.");
                m_log.CHECK_NE(rgdfIpWeightsNonShared1[i], rgdfIpWeightsNonShared2[i], "The ip non shared weights 1 & 2 should not be equal at " + i.ToString());
                m_log.EXPECT_EQUAL<T>(rgdfIpWeightsNonShared1[i] + rgdfIpWeightsNonShared2[i], rgdfSharedParamsDiff[i], "The shared (expected) params should equal the 1 + 2 non shared params at " + i.ToString());
            }

            m_cuda.axpy(nCount, -1.0, ip1_weights.gpu_diff, blobUnsharedParams1.mutable_gpu_data);
            m_cuda.axpy(nCount, -1.0, ip2_weights.gpu_diff, blobUnsharedParams2.mutable_gpu_data);

            double[] rgdfExpectedUpdatedParam1 = convert(blobUnsharedParams1.update_cpu_data());
            double[] rgdfExpectedUpdatedParam2 = convert(blobUnsharedParams2.update_cpu_data());

            m_net.Update();

            double[] rgdfActualUpdatedParams1 = convert(ip1_weights.update_cpu_data());
            double[] rgdfActualUpdatedParams2 = convert(ip2_weights.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_EQ(rgdfExpectedUpdatedParam1[i], rgdfActualUpdatedParams1[i], "The expected and actual params are not equal at " + i.ToString());
                m_log.CHECK_EQ(rgdfExpectedUpdatedParam2[i], rgdfActualUpdatedParams2[i], "The expected and actual params are not equal at " + i.ToString());
                m_log.CHECK_NE(rgdfActualUpdatedParams1[i], rgdfActualUpdatedParams2[i], "The actual 1 & 2 params at " + i.ToString() + " should not be equal.");
                m_log.CHECK_NE(rgdfExpectedUpdatedParam1[i], rgdfExpectedUpdatedParam2[i], "The actual 1 & 2 params at " + i.ToString() + " should not be equal.");
            }
        }

        public void TestSharedWeightsResume()
        {
            // Create a net with weight sharing; Update it once.
            m_cuda.rng_setseed(m_lSeed);
            InitDiffDataSharedWeightsNet();
            BlobCollection<T> colBottom = new BlobCollection<T>();

            m_log.CHECK(m_net.layer_names[1] == "innerproduct1", "The layer 1 should be the 'innerproduct1' layer.");
            m_log.CHECK(m_net.layer_names[2] == "innerproduct2", "The layer 2 should be the 'innerproduct2' layer.");

            Blob<T> ip1_weights = m_net.layers[1].blobs[0];
            Blob<T> ip2_weights = m_net.layers[2].blobs[0];

            // Check that data and diff blobs shared weights share the same memory locations
            m_log.CHECK(ip1_weights.gpu_data == ip2_weights.gpu_data, "The ip1 and ip2 weight data memory is not the same!");
            m_log.CHECK(ip1_weights.gpu_diff == ip2_weights.gpu_diff, "The ip1 and ip2 weight diff memory is not the same!");

            double dfLoss;
            m_net.ForwardBackward(colBottom, out dfLoss);
            m_net.Update();

            Blob<T> blobSharedParams = new Blob<T>(m_cuda, m_log);
            bool kReshape = true;
            bool kCopyDiff = false;
            blobSharedParams.CopyFrom(ip1_weights, kCopyDiff, kReshape);
            int nCount = ip1_weights.count();

            // Write the net to a NetParameter, as in Solver::Snapshot.
            NetParameter net_param = m_net.ToProto(true);

            // Reinitialize the net and copy parameter from net_param, as in 
            // Solver::Restore.
            m_cuda.rng_setseed(m_lSeed);
            InitDiffDataSharedWeightsNet();
            m_net.CopyTrainedLayersFrom(net_param);

            ip1_weights = m_net.layers[1].blobs[0];
            ip2_weights = m_net.layers[2].blobs[0];

            m_log.CHECK(ip1_weights != null, "The ip1_weights should not be null.");
            m_log.CHECK(ip2_weights != null, "The ip2_weights should not be null.");
            m_log.CHECK(ip1_weights != ip2_weights, "The ip1_weights should != ip2_weights.");

            // Check that data and diff blobs shared weights share the same memory locations
            m_log.CHECK(ip1_weights.gpu_data == ip2_weights.gpu_data, "The ip1 and ip2 weight data memory is not the same!");
            m_log.CHECK(ip1_weights.gpu_diff == ip2_weights.gpu_diff, "The ip1 and ip2 weight diff memory is not the same!");

            double[] rgdfSharedParams = convert(blobSharedParams.update_cpu_data());
            double[] rgdfIp1Weights = convert(ip1_weights.update_cpu_data());

            for (int i = 0; i < nCount; i++)
            {
                m_log.CHECK_EQ(rgdfSharedParams[i], rgdfIp1Weights[i], "The shared weight and ip1 weights at " + i.ToString() + " are not the same!");
            }
        }

        public void TestParamPropagateDown()
        {
            BlobCollection<T> colBottom = new BlobCollection<T>();
            bool kBiasTerm = true;
            bool kForceBackward = false;
            double? kLossWeight1 = null;
            double? kLossWeight2 = null;

            // Run the net with all params learned; check that gradients are nont-zero.
            m_cuda.rng_setseed(m_lSeed);
            double dfBlobsLrW1 = 1;
            double dfBlobsLrW2 = 1;
            double dfBlobsLrB1 = 2;
            double dfBlobsLrB2 = 2;
            double dfLoss;

            InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward, kBiasTerm, dfBlobsLrW1, dfBlobsLrW2, dfBlobsLrB1, dfBlobsLrB2);
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();

            BlobCollection<T> colParams = m_net.parameters;
            int nNumParams = colParams.Count;
            m_log.CHECK_EQ(4, nNumParams, "There should be 4 parameters.");

            double kNonZeroTestMin = 1e-3;
            List<double> rgdfParamAsums = Utility.Create<double>(colParams.Count, 0);

            for (int i = 0; i < nNumParams; i++)
            {
                double dfParamsAsum = convert(colParams[i].asum_diff());
                rgdfParamAsums[i] = dfParamsAsum;
                m_log.CHECK_GT(dfParamsAsum, kNonZeroTestMin, "The param asum at " + i.ToString() + " should be greater than " + kNonZeroTestMin.ToString());
            }

            // Change the learning rates to different non-zero values; should see same
            // gradients.
            m_cuda.rng_setseed(m_lSeed);
            dfBlobsLrW1 *= 2;
            dfBlobsLrW2 *= 2;
            dfBlobsLrB1 *= 2;
            dfBlobsLrB2 *= 2;

            InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward, kBiasTerm, dfBlobsLrW1, dfBlobsLrW2, dfBlobsLrB1, dfBlobsLrB2);
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();

            BlobCollection<T> colParams2 = m_net.parameters;
            m_log.CHECK_EQ(nNumParams, colParams2.Count, "There should be 4 parameters in params2.");

            for (int i = 0; i < colParams2.Count; i++)
            {
                double dfParamsAsum = convert(colParams2[i].asum_diff());
                m_log.EXPECT_EQUAL<T>(dfParamsAsum, rgdfParamAsums[i]);
            }

            // Change a subset of the learning rates to zero; check that we see zero
            // gradients from those.
            m_cuda.rng_setseed(m_lSeed);
            dfBlobsLrW1 = 1;
            dfBlobsLrW2 = 0;
            dfBlobsLrB1 = 0;
            dfBlobsLrB2 = 1;

            InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward, kBiasTerm, dfBlobsLrW1, dfBlobsLrW2, dfBlobsLrB1, dfBlobsLrB2);
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();

            BlobCollection<T> colParams3 = m_net.parameters;
            m_log.CHECK_EQ(nNumParams, colParams3.Count, "There should be 4 parameters in params3.");

            for (int i = 0; i < nNumParams; i++)
            {
                double dfParamsAsum = convert(colParams3[i].asum_diff());

                if (i == 1 || i == 2)
                    m_log.CHECK_EQ(0, dfParamsAsum, "The param asum at " + i.ToString() + " should be zero.");
                else
                    m_log.EXPECT_EQUAL<T>(dfParamsAsum, rgdfParamAsums[i]);
            }

            // Change the opposite subset of the learning rates to zero; check that we see zero
            // gradients from those.
            m_cuda.rng_setseed(m_lSeed);
            dfBlobsLrW1 = 0;
            dfBlobsLrW2 = 1;
            dfBlobsLrB1 = 1;
            dfBlobsLrB2 = 0;

            InitUnsharedWeightsNet(kLossWeight1, kLossWeight2, kForceBackward, kBiasTerm, dfBlobsLrW1, dfBlobsLrW2, dfBlobsLrB1, dfBlobsLrB2);
            m_net.Forward(colBottom, out dfLoss);
            m_net.Backward();

            BlobCollection<T> colParams4 = m_net.parameters;
            m_log.CHECK_EQ(nNumParams, colParams4.Count, "There should be 4 parameters in params3.");

            for (int i = 0; i < nNumParams; i++)
            {
                double dfParamsAsum = convert(colParams4[i].asum_diff());

                if (i == 0 || i == 3)
                    m_log.CHECK_EQ(0, dfParamsAsum, "The param asum at " + i.ToString() + " should be zero.");
                else
                    m_log.EXPECT_EQUAL<T>(dfParamsAsum, rgdfParamAsums[i]);
            }
        }

        public void TestFromTo()
        {
            InitTinyNet();

            // Run Forward and Backward, recording the data diff and loss.
            Blob<T> data = new Blob<T>(m_cuda, m_log);
            data.ReshapeLike(m_net.blob_by_name("data"));
            double dfLoss;
            m_net.Forward(out dfLoss);
            m_net.Backward();

            data.CopyFrom(m_net.blob_by_name("data"), true, true);
            dfLoss = convert(m_net.output_blobs[0].GetData(0));

            // Check that combining partial Forwards give the same loss.
            for (int i = 1; i < m_net.layers.Count; i++)
            {
                // Note that we skip layer zero to keep the same data.
                m_net.ForwardFromTo(1, 1);

                if (i < m_net.layers.Count - 1)
                    m_net.ForwardFromTo(i + 1);

                double dfLoss2 = convert(m_net.output_blobs[0].GetData(0));
                m_log.CHECK_EQ(dfLoss, dfLoss2, "The loss from the partial is not equal to the full loss.");
            }

            // Check that combining partial Backwards gives the same data diff.
            for (int i = 1; i < m_net.layers.Count; i++)
            {
                m_net.Backward(int.MaxValue, i);
                m_net.Backward(i - 1);

                for (int j = 0; j < data.count(); j++)
                {
                    double dfDiff1 = convert(data.GetDiff(j));
                    double dfDiff2 = convert(m_net.blob_by_name("data").GetDiff(j));

                    m_log.CHECK_EQ(dfDiff1, dfDiff2, "The diff values at " + j.ToString() + " should be the same.");
                }
            }
        }


        //=====================================================================
        //  Filter tests.
        //=====================================================================

        public void RunFilterNetTest(string strInputParamString, string strFilteredParamString)
        {
            RawProto proto1 = RawProto.Parse(strInputParamString);
            NetParameter input_param = NetParameter.FromProto(proto1);
            RawProto proto2 = RawProto.Parse(strFilteredParamString);
            NetParameter expected_filtered_param = NetParameter.FromProto(proto2);
            Net<T> net = new Net<T>(m_cuda, m_log, new NetParameter(), m_evtCancel, m_db);

            NetParameter actual_filtered_param = net.FilterNet(input_param);
            m_log.CHECK(expected_filtered_param.Compare(actual_filtered_param), "The expected and actual filtered params are not equal!");

            // Also test idempotence.
            NetParameter double_filtered_param = net.FilterNet(actual_filtered_param);
            m_log.CHECK(actual_filtered_param.Compare(double_filtered_param), "The actual and double filtered params are not equal!");
        }

        public void TestNoFilter()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterNetTrainTest()
        {
            string input_proto = 
                  "name: 'LeNet' " +
                  "layer { " +
                  "  name: 'mnist' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "  data_param { " +
                  "    source: 'mnist-train-leveldb' " +
                  "    batch_size: 64 " +
                  "  } " +
                  "  transform_param { " +
                  "    scale: 0.00390625 " +
                  "  } " +
                  "  include: { phase: TRAIN } " +
                  "} " +
                  "layer { " +
                  "  name: 'mnist' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "  data_param { " +
                  "    source: 'mnist-test-leveldb' " +
                  "    batch_size: 100 " +
                  "  } " +
                  "  transform_param { " +
                  "    scale: 0.00390625 " +
                  "  } " +
                  "  include: { phase: TEST } " +
                  "} " +
                  "layer { " +
                  "  name: 'conv1' " +
                  "  type: 'Convolution' " +
                  "  bottom: 'data' " +
                  "  top: 'conv1' " +
                  "  param { " +
                  "    lr_mult: 1 " +
                  "  } " +
                  "  param { " +
                  "    lr_mult: 2 " +
                  "  } " +
                  "  convolution_param { " +
                  "    num_output: 20 " +
                  "    kernel_size: 5 " +
                  "    stride: 1 " +
                  "    weight_filler { " +
                  "      type: 'xavier' " +
                  "    } " +
                  "    bias_filler { " +
                  "      type: 'constant' " +
                  "    } " +
                  "  } " +
                  "} " +
                  "layer { " +
                  "  name: 'ip1' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'conv1' " +
                  "  top: 'ip1' " +
                  "  param { " +
                  "    lr_mult: 1 " +
                  "  } " +
                  "  param { " +
                  "    lr_mult: 2 " +
                  "  } " +
                  "  inner_product_param { " +
                  "    num_output: 10 " +
                  "    weight_filler { " +
                  "      type: 'xavier' " +
                  "    } " +
                  "    bias_filler { " +
                  "      type: 'constant' " +
                  "    } " +
                  "  } " +
                  "} " +
                  "layer { " +
                  "  name: 'accuracy' " +
                  "  type: 'Accuracy' " +
                  "  bottom: 'ip1' " +
                  "  bottom: 'label' " +
                  "  top: 'accuracy' " +
                  "  include: { phase: TEST } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'ip2' " +
                  "  bottom: 'label' " +
                  "  top: 'loss' " +
                  "} ";
            string input_proto_train = "state: { phase: TRAIN } " + input_proto;
            string input_proto_test = "state: { phase: TEST } " + input_proto;
            string output_proto_train =
                  "name: 'LeNet' " +
                  "layer { " +
                  "  name: 'mnist' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "  data_param { " +
                  "    source: 'mnist-train-leveldb' " +
                  "    batch_size: 64 " +
                  "  } " +
                  "  transform_param { " +
                  "    scale: 0.00390625 " +
                  "  } " +
                  "  include: { phase: TRAIN } " +
                  "} " +
                  "layer { " +
                  "  name: 'conv1' " +
                  "  type: 'Convolution' " +
                  "  bottom: 'data' " +
                  "  top: 'conv1' " +
                  "  param { " +
                  "    lr_mult: 1 " +
                  "  } " +
                  "  param { " +
                  "    lr_mult: 2 " +
                  "  } " +
                  "  convolution_param { " +
                  "    num_output: 20 " +
                  "    kernel_size: 5 " +
                  "    stride: 1 " +
                  "    weight_filler { " +
                  "      type: 'xavier' " +
                  "    } " +
                  "    bias_filler { " +
                  "      type: 'constant' " +
                  "    } " +
                  "  } " +
                  "} " +
                  "layer { " +
                  "  name: 'ip1' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'conv1' " +
                  "  top: 'ip1' " +
                  "  param { " +
                  "    lr_mult: 1 " +
                  "  } " +
                  "  param { " +
                  "    lr_mult: 2 " +
                  "  } " +
                  "  inner_product_param { " +
                  "    num_output: 10 " +
                  "    weight_filler { " +
                  "      type: 'xavier' " +
                  "    } " +
                  "    bias_filler { " +
                  "      type: 'constant' " +
                  "    } " +
                  "  } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'ip2' " +
                  "  bottom: 'label' " +
                  "  top: 'loss' " +
                  "} ";
            string output_proto_test =
                  "name: 'LeNet' " +
                  "layer { " +
                  "  name: 'mnist' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "  data_param { " +
                  "    source: 'mnist-test-leveldb' " +
                  "    batch_size: 100 " +
                  "  } " +
                  "  transform_param { " +
                  "    scale: 0.00390625 " +
                  "  } " +
                  "  include: { phase: TEST } " +
                  "} " +
                  "layer { " +
                  "  name: 'conv1' " +
                  "  type: 'Convolution' " +
                  "  bottom: 'data' " +
                  "  top: 'conv1' " +
                  "  param { " +
                  "    lr_mult: 1 " +
                  "  } " +
                  "  param { " +
                  "    lr_mult: 2 " +
                  "  } " +
                  "  convolution_param { " +
                  "    num_output: 20 " +
                  "    kernel_size: 5 " +
                  "    stride: 1 " +
                  "    weight_filler { " +
                  "      type: 'xavier' " +
                  "    } " +
                  "    bias_filler { " +
                  "      type: 'constant' " +
                  "    } " +
                  "  } " +
                  "} " +
                  "layer { " +
                  "  name: 'ip1' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'conv1' " +
                  "  top: 'ip1' " +
                  "  param { " +
                  "    lr_mult: 1 " +
                  "  } " +
                  "  param { " +
                  "    lr_mult: 2 " +
                  "  } " +
                  "  inner_product_param { " +
                  "    num_output: 10 " +
                  "    weight_filler { " +
                  "      type: 'xavier' " +
                  "    } " +
                  "    bias_filler { " +
                  "      type: 'constant' " +
                  "    } " +
                  "  } " +
                  "} " +
                  "layer { " +
                  "  name: 'accuracy' " +
                  "  type: 'Accuracy' " +
                  "  bottom: 'ip1' " +
                  "  bottom: 'label' " +
                  "  top: 'accuracy' " +
                  "  include: { phase: TEST } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'ip2' " +
                  "  bottom: 'label' " +
                  "  top: 'loss' " +
                  "} ";
            string output_proto_train_explicit = output_proto_train + " state: { phase: TRAIN } ";
            string output_proto_test_explicit = output_proto_test + " state: { phase: TEST } ";

            RunFilterNetTest(input_proto_train, output_proto_train_explicit);
            RunFilterNetTest(input_proto_test, output_proto_test_explicit);
        }

        public void TestFilterOutByStage()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "  include: { stage: 'mystage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";
            string output_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, output_proto);
        }

        public void TestFilterOutByStage2()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { stage: 'mystage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";
            string output_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, output_proto);
        }

        public void TestFilterInByStage()
        {
            string input_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { stage: 'mystage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInByStage2()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  exclude: { stage: 'mystage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterOutByMultipleStage()
        {
            string input_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { stage: 'mystage' stage: 'myotherstage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { stage: 'mystage' } " +
                  "} ";
            string output_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { stage: 'mystage' } " +
                  "} ";

            RunFilterNetTest(input_proto, output_proto);
        }

        public void TestFilterInByMultipleStage()
        {
            string input_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { stage: 'myotherstage' } " +
                  "  include: { stage: 'mystage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { stage: 'mystage' } " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInByMultipleStage2()
        {
            string input_proto =
                  "state: { stage: 'mystage' stage: 'myotherstage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { stage: 'mystage' stage: 'myotherstage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { stage: 'mystage' } " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInByNotStage()
        {
            string input_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { not_stage: 'myotherstage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { not_stage: 'myotherstage' } " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterOutByNotStage()
        {
            string input_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { not_stage: 'mystage' } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { not_stage: 'mystage' } " +
                  "} ";
            string output_proto =
                  "state: { stage: 'mystage' } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, output_proto);
        }

        public void TestFilterOutByMinLevel()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { min_level: 3 } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";
            string output_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, output_proto);
        }

        public void TestFilterOutByMaxLevel()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { max_level: -3 } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";
            string output_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, output_proto);
        }

        public void TestFilterInByMinLevel()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { min_level: 0 } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInByMinLevel2()
        {
            string input_proto =
                  "state: { level: 7 } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { min_level: 3 } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInByMaxLevel()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { max_level: 0 } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInByMaxLevel2()
        {
            string input_proto =
                  "state: { level: -7 } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { max_level: -3 } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "} ";

            RunFilterNetTest(input_proto, input_proto);
        }

        public void TestFilterInOutByIncludeMultiRule()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { min_level: 2  phase: TRAIN } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { min_level: 2  phase: TEST } " +
                  "} ";
            string input_proto_train = "state: { level: 4  phase: TRAIN } " + input_proto;
            string input_proto_test = "state: { level: 4  phase: TEST } " + input_proto;
            string output_proto_train =
                  "state: { level: 4  phase: TRAIN } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { min_level: 2  phase: TRAIN } " +
                  "} ";
            string output_proto_test =
                  "state: { level: 4  phase: TEST } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { min_level: 2  phase: TEST } " +
                  "} ";


            RunFilterNetTest(input_proto_train, output_proto_train);
            RunFilterNetTest(input_proto_test, output_proto_test);
        }

        public void TestFilterInByIncludeMultiRule()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  include: { min_level: 2  phase: TRAIN } " +
                  "  include: { phase: TEST } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  include: { min_level: 2  phase: TEST } " +
                  "  include: { phase: TRAIN } " +
                  "} ";
            string input_proto_train = "state: { level: 2  phase: TRAIN } " + input_proto;
            string input_proto_test = "state: { level: 2  phase: TEST } " + input_proto;

            RunFilterNetTest(input_proto_train, input_proto_train);
            RunFilterNetTest(input_proto_test, input_proto_test);
        }

        public void TestFilterInOutByExcludeMultiRule()
        {
            string input_proto =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  exclude: { min_level: 2  phase: TRAIN } " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  exclude: { min_level: 2  phase: TEST } " +
                  "} ";
            string input_proto_train = "state: { level: 4  phase: TRAIN } " + input_proto;
            string input_proto_test = "state: { level: 4  phase: TEST } " + input_proto;
            string output_proto_train =
                  "state: { level: 4  phase: TRAIN } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss' " +
                  "  type: 'SoftmaxWithLoss' " +
                  "  bottom: 'innerprod' " +
                  "  bottom: 'label' " +
                  "  exclude: { min_level: 2  phase: TEST } " +
                  "} ";
            string output_proto_test =
                  "state: { level: 4  phase: TEST } " +
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod' " +
                  "  exclude: { min_level: 2  phase: TRAIN } " +
                  "} ";

            RunFilterNetTest(input_proto_train, output_proto_train);
            RunFilterNetTest(input_proto_test, output_proto_test);
        }

        public void TestReshape()
        {
            // We set up bottom blobs of two different sizes, switch between
            // them, check that forward and backward both run and the results
            // are the same, and check that the output shapes change.
            m_cuda.rng_setseed(m_lSeed);
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = 1.0;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            // Check smaller shape first as larger first could hide realloc failures.
            Blob<T> blob1 = new Blob<T>(m_cuda, m_log, 2, 3, 12, 10);
            Blob<T> blob2 = new Blob<T>(m_cuda, m_log, 4, 3, 9, 11);

            m_log.CHECK_LT(blob1.count(), blob2.count(), "Blob one count should be < blob2 count.");

            filler.Fill(blob1);
            filler.Fill(blob2);

            InitReshapableNet();

            Blob<T> input_blob = m_net.input_blobs[0];
            Blob<T> output_blob = m_net.output_blobs[0];
            input_blob.Reshape(blob1.num, blob1.channels, blob1.height, blob1.width);
            m_cuda.copy(blob1.count(), blob1.gpu_data, input_blob.mutable_gpu_data);

            double dfLoss;
            m_net.Forward(out dfLoss);
            // call backward just to make sure it runs.
            m_net.Backward();

            Blob<T> output1 = new Blob<T>(m_cuda, m_log, output_blob);
            m_cuda.copy(output1.count(), output_blob.gpu_data, output1.mutable_gpu_data);

            input_blob.ReshapeLike(blob2);
            m_cuda.copy(blob2.count(), blob2.gpu_data, input_blob.mutable_gpu_data);

            m_net.Forward(out dfLoss);
            m_net.Backward();

            Blob<T> output2 = new Blob<T>(m_cuda, m_log, output_blob);
            m_cuda.copy(output2.count(), output_blob.gpu_data, output2.mutable_gpu_data);

            input_blob.ReshapeLike(blob1);
            m_cuda.copy(blob1.count(), blob1.gpu_data, input_blob.mutable_gpu_data);

            m_net.Forward(out dfLoss);
            m_net.Backward();

            double[] rgOutput1 = convert(output1.mutable_cpu_data);
            double[] rgOutput = convert(output_blob.mutable_cpu_data);

            for (int i = 0; i < output1.count(); i++)
            {
                m_log.EXPECT_EQUAL<T>(rgOutput[i], rgOutput1[i]);
            }

            input_blob.ReshapeLike(blob2);
            m_cuda.copy(blob2.count(), blob2.gpu_data, input_blob.mutable_gpu_data);

            m_net.Forward(out dfLoss);
            m_net.Backward();

            double[] rgOutput2 = convert(output2.update_cpu_data());
            rgOutput = convert(output_blob.mutable_cpu_data);

            for (int i = 0; i < output2.count(); i++)
            {
                m_log.EXPECT_NEAR(rgOutput[i], rgOutput2[i], 0.03);
            }

            m_log.CHECK_EQ(output1.num, blob1.num, "Output1 and blob1 should have the same num.");
            m_log.CHECK_EQ(output2.num, blob2.num, "Output2 and blob2 should have the same num.");

            bool bSameSpatialShape = true;
            int kFirstSpatialAxis = 2;

            for (int i = kFirstSpatialAxis; i < output2.num_axes; i++)
            {
                if (output1.shape(i) != output2.shape(i))
                {
                    bSameSpatialShape = false;
                    break;
                }
            }

            m_log.CHECK(!bSameSpatialShape, "Output 1 and output 2 should not have the same spatial shape.");
        }

        public void TestSkipPropagateDown()
        {
            // Check bottom_need_backward if propagate_down is true.
            InitSkipPropNet(false);
            List<bool> rgLayerNeedBackward = m_net.layer_need_backward;

            for (int layer_id = 0; layer_id < m_net.layers.Count; layer_id++)
            {
                string layer_name = m_net.layer_names[layer_id];

                if (layer_name == "loss")
                {
                    // access to bottom_need_backward corresponding to label's blob.
                    bool bNeedBack = m_net.bottom_need_backward[layer_id][1];

                    // if propagate_down is true, the loss layer will try to
                    // backpropagate on labels.
                    m_log.CHECK(bNeedBack, "bottom_need_backward should be true.");
                }

                // layer_need_backward should be true except for data and silence layers
                if (layer_name.ToLower().Contains("data") || layer_name == "silence")
                {
                    m_log.CHECK(!rgLayerNeedBackward[layer_id], "layer_need_backward for " + layer_name + " should be false.");
                }
                else
                {
                    m_log.CHECK(rgLayerNeedBackward[layer_id], "layer_need_backward for " + layer_name + " should be true.");
                }
            }

            // check bottom_need_backward if propagate_down is false.
            InitSkipPropNet(true);
            rgLayerNeedBackward = m_net.layer_need_backward;

            for (int layer_id = 0; layer_id < m_net.layers.Count; layer_id++)
            {
                string layer_name = m_net.layer_names[layer_id];

                if (layer_name == "loss")
                {
                    // access to bottom_need_backward corresponding to label's blob.
                    bool bNeedBack = m_net.bottom_need_backward[layer_id][1];

                    // if propagate_down is true, the loss layer will try to
                    // backpropagate on labels.
                    m_log.CHECK(!bNeedBack, "bottom_need_backward should be false.");
                }

                // layer_need_backward should be False except for innerproduct and loss layers
                if (layer_name == "innerproduct" || layer_name == "loss")
                {
                    m_log.CHECK(rgLayerNeedBackward[layer_id], "layer_need_backward for " + layer_name + " should be true.");
                }
                else
                {
                    m_log.CHECK(!rgLayerNeedBackward[layer_id], "layer_need_backward for " + layer_name + " should be false.");
                }
            }
        }
        public void TestForcePropagateDown()
        {
            InitForcePropNet(false);

            List<bool> rgbLayerNeedBackward = m_net.layer_need_backward;
            for (int layer_id = 0; layer_id < m_net.layers.Count(); layer_id++)
            {
                string strLayerName = m_net.layer_names[layer_id];
                List<bool> rgbNeedBackward = m_net.bottom_need_backward[layer_id];

                if (strLayerName == "data")
                {
                    m_log.CHECK_EQ(rgbNeedBackward.Count, 0, "The layer need backward count should be 0.");
                    m_log.CHECK(rgbLayerNeedBackward[layer_id] == false, "The layer need backward for the data layer should be false.");
                }
                else if (strLayerName == "innerproduct")
                {
                    m_log.CHECK_EQ(rgbNeedBackward.Count, 1, "The layer need backward count should be 1.");
                    m_log.CHECK(rgbNeedBackward[0] == false, "The layer need backward for the data layer should be false.");
                    m_log.CHECK(rgbLayerNeedBackward[layer_id] == true, "The layer need backward for the innerproduct layer should be true.");
                }
                else if (strLayerName == "loss")
                {
                    m_log.CHECK_EQ(rgbNeedBackward.Count, 2, "The layer need backward count should be 2.");
                    m_log.CHECK(rgbNeedBackward[0] == true, "The layer need backward for the inner product layer should be false.");
                    m_log.CHECK(rgbNeedBackward[1] == false, "The layer need backward for the label should be false.");
                    m_log.CHECK(rgbLayerNeedBackward[layer_id] == true, "The layer need backward for the loss layer should be true.");
                }
                else
                {
                    m_log.FAIL("Unknown layer: " + strLayerName);
                }
            }

            InitForcePropNet(true);
            rgbLayerNeedBackward = m_net.layer_need_backward;
            for (int layer_id = 0; layer_id < m_net.layers.Count(); layer_id++)
            {
                string strLayerName = m_net.layer_names[layer_id];
                List<bool> rgbNeedBackward = m_net.bottom_need_backward[layer_id];

                if (strLayerName == "data")
                {
                    m_log.CHECK_EQ(rgbNeedBackward.Count, 0, "The layer need backward count should be 0.");
                    m_log.CHECK(rgbLayerNeedBackward[layer_id] == false, "The layer need backward for the data layer should be false.");
                }
                else if (strLayerName == "innerproduct")
                {
                    m_log.CHECK_EQ(rgbNeedBackward.Count, 1, "The layer need backward count should be 1.");
                    m_log.CHECK(rgbNeedBackward[0] == true, "The layer need backward for the data layer should be false.");
                    m_log.CHECK(rgbLayerNeedBackward[layer_id] == true, "The layer need backward for the innerproduct layer should be true.");
                }
                else if (strLayerName == "loss")
                {
                    m_log.CHECK_EQ(rgbNeedBackward.Count, 2, "The layer need backward count should be 2.");
                    m_log.CHECK(rgbNeedBackward[0] == true, "The layer need backward for the inner product layer should be false.");
                    m_log.CHECK(rgbNeedBackward[1] == false, "The layer need backward for the label should be false.");
                    m_log.CHECK(rgbLayerNeedBackward[layer_id] == true, "The layer need backward for the loss layer should be true.");
                }
                else
                {
                    m_log.FAIL("Unknown layer: " + strLayerName);
                }
            }
        }

        public void TestAllInOneNetTrain()
        {
            List<string> rgStages = new List<string>();

            rgStages.Add("train");
            InitAllInOneNet(Phase.TRAIN, 0, rgStages);
            bool bFoundData = false;
            bool bFoundLoss = false;

            for (int i = 0; i < m_net.layers.Count; i++)
            {
                string strLayerName = m_net.layer_names[i];

                if (strLayerName == "train-data")
                {
                    bFoundData = true;
                }
                else if (strLayerName == "loss")
                {
                    bFoundLoss = true;
                }
                else
                {
                    m_log.CHECK(strLayerName != "val-data", "The layer name is incorrect.");
                    m_log.CHECK(strLayerName != "deploy-data", "The layer name is incorrect.");
                }
            }

            m_log.CHECK(bFoundData, "Did not find the data layer.");
            m_log.CHECK(bFoundLoss, "Did not find the loss layer.");
        }
    }
}
