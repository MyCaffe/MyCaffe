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
using MyCaffe.layers.beta;
using MyCaffe.layers.gpt;
using MyCaffe.param.gpt;
using MyCaffe.layers.gpt.layers.gpt;

/// <summary>
/// Testing the tokenized data layer.
/// 
/// PreTokenizedDataLayer - layer loads pre-tokenized data for transformer type models.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestGPT_PreTokenizedDataLayer
    {
        [TestMethod]
        public void TestForwardInput()
        {
            PreTokenizedDataLayerTest test = new PreTokenizedDataLayerTest();

            try
            {
                foreach (IPreTokenizedDataLayerTest t in test.Tests)
                {
                    t.TestForward(1, 350);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardInputBatch()
        {
            PreTokenizedDataLayerTest test = new PreTokenizedDataLayerTest();

            try
            {
                foreach (IPreTokenizedDataLayerTest t in test.Tests)
                {
                    t.TestForward(10, 350);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IPreTokenizedDataLayerTest : ITest
    {
        void TestForward(int nBatchSize, int nBlockSize);
    }

    class PreTokenizedDataLayerTest : TestBase
    {
        public PreTokenizedDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PreTokenizedData Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PreTokenizedDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PreTokenizedDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PreTokenizedDataLayerTest<T> : TestEx<T>, IPreTokenizedDataLayerTest
    {
        Blob<T> m_blobData;
        Blob<T> m_blobPos;
        Blob<T> m_blobTarget;

        public PreTokenizedDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_blobData = new Blob<T>(m_cuda, m_log);
            m_blobPos = new Blob<T>(m_cuda, m_log);
            m_blobTarget = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            m_blobData.Dispose();
            m_blobPos.Dispose();
            m_blobTarget.Dispose();

            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward(int nBatchSize, int nBlockSize)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRETOKENIZED_DATA);
            p.pretokenized_data_param.batch_size = (uint)nBatchSize;
            p.pretokenized_data_param.block_size = (uint)nBlockSize;
            p.pretokenized_data_param.source = "C:\\temp\\projects\\llama2\\llama2\\llama2_instruct\\instruct_dataset\\";
            p.pretokenized_data_param.seed = 1701;
            p.pretokenized_data_param.pad_token = -100;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                BottomVec.Clear();
                TopVec.Clear();

                TopVec.Add(m_blobData);
                TopVec.Add(m_blobTarget);

                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(TopVec.Count, 2, "The top vector should contain 2 blobs.");

                // Data
                m_log.CHECK_EQ(TopVec[0].num, nBatchSize, "The top[0].num should equal the batch size of " + nBatchSize.ToString());
                m_log.CHECK_EQ(TopVec[0].channels, nBlockSize, "The top[0].channels should equal the block size of " + nBlockSize.ToString());

                // Target
                m_log.CHECK_EQ(TopVec[1].num, nBatchSize, "The top[1].num should equal the batch size of " + nBatchSize.ToString());
                m_log.CHECK_EQ(TopVec[1].channels, nBlockSize, "The top[1].channels should equal the block size of " + nBlockSize.ToString());

                layer.Forward(BottomVec, TopVec);


                float[] rgData = convertF(TopVec[0].mutable_cpu_data);
                float[] rgTarget = convertF(TopVec[1].mutable_cpu_data);

                for (int i = 0; i < nBatchSize; i++)
                {
                    for (int j = 0; j < nBlockSize - 1; j++)
                    {
                        float fExpected = rgData[i * nBlockSize + j + 1];
                        float fActual = rgTarget[i * nBlockSize + j];

                        if (fActual >= 1)
                            m_log.CHECK_EQ(fActual, fExpected, "The token in batch " + i.ToString() + " at block " + j.ToString() + " is not correct.");
                    }
                }

                rgData = convertF(TopVec[0].mutable_cpu_data);
                rgTarget = convertF(TopVec[1].mutable_cpu_data);
                List<Tuple<string, string>> rgDataOut = new List<Tuple<string, string>>();

                for (int i = 0; i < nBatchSize; i++)
                {
                    string strData = ((PreTokenizedDataLayer<T>)layer).Detokenize(rgData, i * nBlockSize, nBlockSize);
                    string strTarget = ((PreTokenizedDataLayer<T>)layer).Detokenize(rgTarget, i * nBlockSize, nBlockSize);

                    rgDataOut.Add(new Tuple<string, string>(strData, strTarget));
                }

                m_log.EnableTrace = true;
                for (int i = 0; i < rgDataOut.Count; i++)
                {
                    m_log.WriteLine("Data: " + rgDataOut[i].Item1);
                    m_log.WriteLine("Target: " + rgDataOut[i].Item2);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
