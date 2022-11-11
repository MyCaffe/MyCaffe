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

/// <summary>
/// Testing the tokenized data layer.
/// 
/// TokenizedDataLayer - layer converts data into tokens for transformer type models.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTokenizedDataLayer
    {
        [TestMethod]
        public void TestForwardInput()
        {
            TokenizedDataLayerTest test = new TokenizedDataLayerTest();
            string strInputTxt = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\input.txt";

            try
            {
                foreach (ITokenizedDataLayerTest t in test.Tests)
                {
                    t.TestForward(1, 128, strInputTxt);
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
            TokenizedDataLayerTest test = new TokenizedDataLayerTest();
            string strInputTxt = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\input.txt";

            try
            {
                foreach (ITokenizedDataLayerTest t in test.Tests)
                {
                    t.TestForward(10, 128, strInputTxt);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardShakespeare()
        {
            TokenizedDataLayerTest test = new TokenizedDataLayerTest();
            string strInputTxt = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\char-rnn\\shakespeare.txt";

            try
            {
                foreach (ITokenizedDataLayerTest t in test.Tests)
                {
                    t.TestForward(10, 128, strInputTxt);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITokenizedDataLayerTest : ITest
    {
        void TestForward(int nBatchSize, int nBlockSize, string strSrcFile);
    }

    class TokenizedDataLayerTest : TestBase
    {
        public TokenizedDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TokenizedData Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TokenizedDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TokenizedDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TokenizedDataLayerTest<T> : TestEx<T>, ITokenizedDataLayerTest
    {
        Blob<T> m_blobData;
        Blob<T> m_blobPos;
        Blob<T> m_blobTarget;

        public TokenizedDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        public void TestForward(int nBatchSize, int nBlockSize, string strSrcFile)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA);
            p.tokenized_data_param.batch_size = (uint)nBatchSize;
            p.tokenized_data_param.block_size = (uint)nBlockSize;
            p.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            p.tokenized_data_param.source = strSrcFile;
            p.tokenized_data_param.seed = 1701;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                BottomVec.Clear();
                TopVec.Clear();

                TopVec.Add(m_blobData);
                TopVec.Add(m_blobPos);
                TopVec.Add(m_blobTarget);

                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(TopVec.Count, 3, "The top vector should contain 3 blobs.");
                
                // Data
                m_log.CHECK_EQ(TopVec[0].num, nBatchSize, "The top[0].num should equal the batch size of " + nBatchSize.ToString());
                m_log.CHECK_EQ(TopVec[0].channels, nBlockSize, "The top[0].channels should equal the block size of " + nBlockSize.ToString());
//                m_log.CHECK_EQ(TopVec[0].height, 1, "The top[0].height should equal 1.");

                // Pos
                m_log.CHECK_EQ(TopVec[1].num, 1, "The top[1].num should equal 1.");
                m_log.CHECK_EQ(TopVec[1].channels, nBlockSize, "The top[1].channels should equal the block size of " + nBlockSize.ToString());
//                m_log.CHECK_EQ(TopVec[1].height, 1, "The top[1].height should equal 1.");

                // Target
                m_log.CHECK_EQ(TopVec[2].num, nBatchSize, "The top[2].num should equal the batch size of " + nBatchSize.ToString());
                m_log.CHECK_EQ(TopVec[2].channels, nBlockSize, "The top[2].channels should equal the block size of " + nBlockSize.ToString());
//                m_log.CHECK_EQ(TopVec[2].height, 1, "The top[2].height should equal 1.");

                layer.Forward(BottomVec, TopVec);

                float[] rgPos = convertF(TopVec[1].mutable_cpu_data);
                for (int i = 0; i < nBlockSize; i++)
                {
                    m_log.CHECK_EQ((int)rgPos[i], i, "The position at index " + i.ToString() + " should equal " + i.ToString() + ".");
                }
 
                float[] rgData = convertF(TopVec[0].mutable_cpu_data);
                float[] rgTarget = convertF(TopVec[2].mutable_cpu_data);

                for (int i = 0; i < nBatchSize; i++)
                {
                    for (int j = 0; j < nBlockSize - 1; j++)
                    {
                        float fExpected = rgData[i * nBlockSize + j + 1];
                        float fActual = rgTarget[i * nBlockSize + j];

                        m_log.CHECK_EQ(fActual, fExpected, "The token in batch " + i.ToString() + " at block " + j.ToString() + " is not correct.");
                    }
                }

                ((TokenizedDataLayer<T>)layer).Detokenize(TopVec[0], TopVec[0]);
                ((TokenizedDataLayer<T>)layer).Detokenize(TopVec[2], TopVec[2]);

                rgData = convertF(TopVec[0].mutable_cpu_data);
                rgTarget = convertF(TopVec[2].mutable_cpu_data);
                List<Tuple<string, string>> rgDataOut = new List<Tuple<string, string>>();

                for (int i = 0; i < nBatchSize; i++)
                {
                    string strData = "";
                    string strTarget = "";
                    
                    for (int j = 0; j < nBlockSize - 1; j++)
                    {
                        float fExpected = rgData[i * nBlockSize + j + 1];
                        float fActual = rgTarget[i * nBlockSize + j];

                        strData += (char)rgData[i * nBlockSize + j];
                        strTarget += (char)fActual;

                        m_log.CHECK_EQ(fActual, fExpected, "The token in batch " + i.ToString() + " at block " + j.ToString() + " is not correct.");
                    }

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
