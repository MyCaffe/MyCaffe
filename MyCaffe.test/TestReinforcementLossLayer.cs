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
using MyCaffe.imagedb;

namespace MyCaffe.test
{
    [TestClass]
    public class TestReinforcementLossLayer
    {
        [TestMethod]
        public void TestGradient()
        {
            ReinforcementLossLayerTest test = new ReinforcementLossLayerTest();

            try
            {
                foreach (IReinforcementLossLayerTest t in test.Tests)
                {
                    t.TestGradient(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientTerminal()
        {
            ReinforcementLossLayerTest test = new ReinforcementLossLayerTest();

            try
            {
                foreach (IReinforcementLossLayerTest t in test.Tests)
                {
                    t.TestGradient(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IReinforcementLossLayerTest : ITest
    {
        void TestGradient(bool bTerminal);
    }

    class ReinforcementLossLayerTest : TestBase
    {
        SettingsCaffe m_settings;
        MyCaffeImageDatabase m_db = null;
        CancelEvent m_evtCancel = new CancelEvent();

        public ReinforcementLossLayerTest(string strDs = "MNIST", EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ReinforcementLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
            m_settings = new SettingsCaffe();
            m_db = new MyCaffeImageDatabase();
            m_db.Initialize(m_settings, strDs, m_evtCancel);
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ReinforcementLossLayerTest<double>(strName, nDeviceID, this);
            else
                return new ReinforcementLossLayerTest<float>(strName, nDeviceID, this);
        }

        protected override void dispose()
        {
            m_db.Dispose();
            m_db = null;

            base.dispose();
        }

        public string SourceName
        {
            get { return "MNIST.training"; }
        }

        public MyCaffeImageDatabase db
        {
            get { return m_db; }
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }
    }

    class ReinforcementLossLayerTest<T> : TestEx<T>, IReinforcementLossLayerTest
    {
        ReinforcementLossLayerTest m_parent;
        BatchInput m_biInput = null;

        public ReinforcementLossLayerTest(string strName, int nDeviceID, ReinforcementLossLayerTest parent)
            : base(strName, null, nDeviceID)
        {
            m_parent = parent;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestGradient(bool bTerminal)
        {
            BatchInformationCollection col = new BatchInformationCollection();

            BatchInformation bi0 = new BatchInformation(0);
            bi0.Add(new BatchItem(5, 5));
            bi0.Add(new BatchItem(0, 0));
            bi0.Add(new BatchItem(4, 4));
            bi0.Add(new BatchItem(1, 1));
            bi0.Add(new BatchItem(9, 9));
            col.Add(bi0);

            BatchInformation bi1 = new BatchInformation(1);
            bi1.Add(new BatchItem(9, 9));
            bi1.Add(new BatchItem(1, 1));
            bi1.Add(new BatchItem(4, 4));
            bi1.Add(new BatchItem(0, 0));
            bi1.Add(new BatchItem(5, 5));
            col.Add(bi1);

            for (int i=0; i<col.Count; i++)
            {
                for (int j=0; j<col[i].Count; j++)
                {
                    col[i][j].Reward = 1.0;
                    col[i][j].QMax1 = 0.6;
                    col[i][j].Terminal = bTerminal;
                }
            }

            LayerParameter pBD = new LayerParameter(LayerParameter.LayerType.BATCHDATA);
            pBD.batch_data_param.source = m_parent.SourceName;
            Layer<T> layerBD = Layer<T>.Create(m_cuda, m_log, pBD, m_parent.CancelEvent, m_parent.db, new TransferInput(null, setInput));

            LayerParameter pRL = new LayerParameter(LayerParameter.LayerType.REINFORCEMENT_LOSS);
            pRL.reinforcement_loss_param.BatchInfoCollection = col;
            Layer<T> layerRL = Layer<T>.Create(m_cuda, m_log, pRL, m_parent.CancelEvent, m_parent.db, new TransferInput(getInput, null));


            //---------------------------------------------
            //  Run forward on first batch (#0).
            //---------------------------------------------
            
            double[] rgInput = new double[] { 0, 1, 2, 3, 4, 4, 3, 2, 1, 0 };
            Bottom.Reshape(2, 5, 1, 1);     // batch 0 = 0, 1, 2, 3, 4;  batch 1 = 4, 3, 2, 1, 0.
            Bottom.mutable_cpu_data = convert(rgInput);

            layerBD.LayerSetUp(BottomVec, TopVec);
            layerBD.Reshape(BottomVec, TopVec);
            layerBD.Forward(BottomVec, TopVec);

            List<double> rgTopData = new List<double>();
            rgTopData.AddRange(new List<double>() { 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 1 in batch 0. MNIST '5'
            rgTopData.AddRange(new List<double>() { 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 2 in batch 0. MNIST '0'
            rgTopData.AddRange(new List<double>() { 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 3 in batch 0. MNIST '4'
            rgTopData.AddRange(new List<double>() { 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 4 in batch 0. MNIST '1'
            rgTopData.AddRange(new List<double>() { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6 });     // outputs for item 5 in batch 0. MNIST '9'
            List<int> rgTargetIdx = new List<int>() { 5, 0, 4, 1, 9 };
            Top.Reshape(5, 10, 1, 1);
            Top.mutable_cpu_data = convert(rgTopData.ToArray());
            Bottom.ReshapeLike(Top);

            layerRL.LayerSetUp(BottomVec, TopVec);
            layerRL.Reshape(BottomVec, TopVec);
            layerRL.Backward(TopVec, new List<bool>() { true }, BottomVec);

            double[] rgOutput = convert(Bottom.update_cpu_diff());

            m_log.CHECK_EQ(rgOutput.Length, rgTopData.Count, "The bottom output count should be equal to the top count.");

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    int nIdx = (i * 10) + j;

                    if (rgTargetIdx[i] == j)
                    {
                        m_log.CHECK_NE(rgOutput[nIdx], 0, "The targeted output should not be zero.");

                        if (bTerminal)
                            m_log.CHECK_EQ(rgOutput[nIdx], 1, "The target output should equal to 1.");
                        else
                            m_log.CHECK_GE(rgOutput[nIdx], 1, "The target output should be greater than 1.");
                    }
                    else
                    {
                        m_log.CHECK_EQ(rgOutput[nIdx], 0, "The non-targeted output should be zero.");
                    }
                }
            }


            //---------------------------------------------
            //  Run forward on second batch (#1).
            //---------------------------------------------

            Bottom.Reshape(2, 5, 1, 1);     // batch 0 = 0, 1, 2, 3, 4;  batch 1 = 4, 3, 2, 1, 0.
            Bottom.mutable_cpu_data = convert(rgInput);
            layerBD.Forward(BottomVec, TopVec);

            rgTopData = new List<double>();
            rgTopData.AddRange(new List<double>() { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6 });     // outputs for item 5 in batch 0. MNIST '9'
            rgTopData.AddRange(new List<double>() { 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 4 in batch 0. MNIST '1'
            rgTopData.AddRange(new List<double>() { 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 3 in batch 0. MNIST '4'
            rgTopData.AddRange(new List<double>() { 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 2 in batch 0. MNIST '0'
            rgTopData.AddRange(new List<double>() { 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1 });     // outputs for item 1 in batch 0. MNIST '5'
            rgTargetIdx = new List<int>() { 9, 1, 4, 0, 5 };
            Top.Reshape(5, 10, 1, 1);
            Top.mutable_cpu_data = convert(rgTopData.ToArray());
            Bottom.ReshapeLike(Top);

            layerRL.LayerSetUp(BottomVec, TopVec);
            layerRL.Reshape(BottomVec, TopVec);
            layerRL.Backward(TopVec, new List<bool>() { true }, BottomVec);

            rgOutput = convert(Bottom.update_cpu_diff());

            m_log.CHECK_EQ(rgOutput.Length, rgTopData.Count, "The bottom output count should be equal to the top count.");

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    int nIdx = (i * 10) + j;

                    if (rgTargetIdx[i] == j)
                    {
                        m_log.CHECK_NE(rgOutput[nIdx], 0, "The targeted output should not be zero.");

                        if (bTerminal)
                            m_log.CHECK_EQ(rgOutput[nIdx], 1, "The target output should equal to 1.");
                        else
                            m_log.CHECK_GE(rgOutput[nIdx], 1, "The target output should be greater than 1.");
                    }
                    else
                    {
                        m_log.CHECK_EQ(rgOutput[nIdx], 0, "The non-targeted output should be zero.");
                    }
                }
            }
        }

        private void setInput(BatchInput bi)
        {
            m_biInput = bi;
        }

        private BatchInput getInput()
        {
            return m_biInput;
        }
    }
}
