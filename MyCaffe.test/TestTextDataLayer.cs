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
using MyCaffe.layers.hdf5;
using System.Diagnostics;
using System.IO;
using MyCaffe.layers.beta;

namespace MyCaffe.test
{
    [TestClass]
    public class TestTextDataLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            TextDataLayerTest test = new TextDataLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITextDataLayerTest t in test.Tests)
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
        public void TestForward()
        {
            TextDataLayerTest test = new TextDataLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITextDataLayerTest t in test.Tests)
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
        public void TestForwardRunPhase()
        {
            TextDataLayerTest test = new TextDataLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITextDataLayerTest t in test.Tests)
                {
                    t.TestForwardRunPhase();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITextDataLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestForwardRunPhase();
    }

    class TextDataLayerTest : TestBase
    {
        public TextDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Text Data Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TextDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TextDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TextDataLayerTest<T> : TestEx<T>, ITextDataLayerTest
    {
        Blob<T> m_blobDecInput;
        Blob<T> m_blobDecClip;
        Blob<T> m_blobDecTarget;
        Blob<T> m_blobEncInput1;
        Blob<T> m_blobEncInput2;
        Blob<T> m_blobEncClip;
        Blob<T> m_blobVocabCount;

        Blob<T> m_blobBtmDecInput;
        Blob<T> m_blobBtmEncInput1;
        Blob<T> m_blobBtmEncInput2;
        Blob<T> m_blobBtmEncClip;

        public TextDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blobDecInput = new Blob<T>(m_cuda, m_log, false);
            m_blobDecInput.Name = "dec_input";

            m_blobDecClip = new Blob<T>(m_cuda, m_log, false);
            m_blobDecClip.Name = "dec_clip";

            m_blobDecTarget = new Blob<T>(m_cuda, m_log, false);
            m_blobDecTarget.Name = "dec_target";

            m_blobEncInput1 = new Blob<T>(m_cuda, m_log, false);
            m_blobEncInput1.Name = "enc_input1";

            m_blobEncInput2 = new Blob<T>(m_cuda, m_log, false);
            m_blobEncInput2.Name = "enc_input2";

            m_blobEncClip = new Blob<T>(m_cuda, m_log, false);
            m_blobEncClip.Name = "enc_clip";

            m_blobVocabCount = new Blob<T>(m_cuda, m_log, false);
            m_blobVocabCount.Name = "vocab_count";

            m_blobBtmDecInput = new Blob<T>(m_cuda, m_log, false);
            m_blobBtmDecInput.Name = "btm_dec_input";

            m_blobBtmEncInput1 = new Blob<T>(m_cuda, m_log, false);
            m_blobBtmEncInput1.Name = "btm_enc_input1";

            m_blobBtmEncInput2 = new Blob<T>(m_cuda, m_log, false);
            m_blobBtmEncInput2.Name = "btm_enc_input2";

            m_blobBtmEncClip = new Blob<T>(m_cuda, m_log, false);
            m_blobBtmEncClip.Name = "btm_enc_clip";
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            m_blobDecInput.Dispose();
            m_blobDecClip.Dispose();
            m_blobDecTarget.Dispose();
            m_blobEncInput1.Dispose();
            m_blobEncInput2.Dispose();
            m_blobEncClip.Dispose();
            m_blobVocabCount.Dispose();

            m_blobBtmDecInput.Dispose();
            m_blobBtmEncInput1.Dispose();
            m_blobBtmEncInput2.Dispose();
            m_blobBtmEncClip.Dispose();

            base.dispose();
        }

        private void verify_shape(Blob<T> b, List<int> rgExpectedShape)
        {
            List<int> rgShape = b.shape();

            m_log.CHECK_EQ(rgShape.Count, rgExpectedShape.Count, "The shapes do not match!");

            for (int i = 0; i < rgShape.Count; i++)
            {
                m_log.CHECK_EQ(rgShape[i], rgExpectedShape[i], "The shapes do not match!");
            }
        }

        public void TestSetup()
        {
            Stopwatch sw = new Stopwatch();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\";
            string strEncSrc = strPath + "human_text.txt";
            string strDecSrc = strPath + "robot_text.txt";

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TEXT_DATA);
            p.text_data_param.batch_size = 1;
            p.text_data_param.time_steps = 80;
            p.text_data_param.shuffle = false;
            p.text_data_param.enable_normal_encoder_output = true;
            p.text_data_param.enable_reverse_encoder_output = true;
            p.text_data_param.encoder_source = strEncSrc;
            p.text_data_param.decoder_source = strDecSrc;
            p.text_data_param.sample_size = 1000;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            m_log.CHECK_EQ((int)layer.type, (int)LayerParameter.LayerType.TEXT_DATA, "The layer type should be TEXT_DATA!");

            TopVec.Clear();
            TopVec.Add(m_blobDecInput);
            TopVec.Add(m_blobDecClip);
            TopVec.Add(m_blobEncInput1);
            TopVec.Add(m_blobEncInput2);
            TopVec.Add(m_blobEncClip);
            TopVec.Add(m_blobVocabCount);
            TopVec.Add(m_blobDecTarget);

            BottomVec.Clear();

            layer.Setup(BottomVec, TopVec);

            int nT = (int)p.text_data_param.time_steps;
            int nN = (int)p.text_data_param.batch_size;

            verify_shape(TopVec[0], new List<int>() { 1, nN, 1 });   // dec input
            verify_shape(TopVec[1], new List<int>() { 1, nN });         // dec clip
            verify_shape(TopVec[2], new List<int>() { nT, nN, 1 });  // enc input1
            verify_shape(TopVec[3], new List<int>() { nT, nN, 1 });  // enc input2
            verify_shape(TopVec[4], new List<int>() { nT, nN });        // enc clip
            verify_shape(TopVec[5], new List<int>() { 1 });    // vocab count
            verify_shape(TopVec[6], new List<int>() { 1, nN, 1 });   // dec target
        }

        private List<string> preprocess(string str, int nMaxLen = 0)
        {
            List<string> rgstr = str.ToLower().Trim().Split(' ').ToList();

            if (nMaxLen > 0)
            {
                rgstr = rgstr.Take(nMaxLen).ToList();
                if (rgstr.Count < nMaxLen)
                    return null;
            }

            return rgstr;
        }

        private List<int> getInput(int nIdx, Vocabulary vocab, string strFile)
        {
            string strLine = "";

            using (StreamReader sr = new StreamReader(strFile))
            {
                for (int i = 0; i <= nIdx; i++)
                {
                    strLine = sr.ReadLine();
                }
            }

            List<string> rgstr = preprocess(strLine);
            List<int> rg = new List<int>();

            foreach (string strWord in rgstr)
            {
                rg.Add(vocab.WordToIndex(strWord));
            }

            return rg;
        }

        private List<int> getInput(Vocabulary vocab, string str, bool bReverse = false)
        {
            List<string> rgstr = preprocess(str);
            List<int> rg = new List<int>();

            foreach (string strWord in rgstr)
            {
                rg.Add(vocab.WordToIndex(strWord));
            }

            if (bReverse)
                rg.Reverse();

            return rg;
        }

        public void TestForward()
        {
            Stopwatch sw = new Stopwatch();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\";
            string strEncSrc = strPath + "human_text.txt";
            string strDecSrc = strPath + "robot_text.txt";

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TEXT_DATA);
            p.text_data_param.batch_size = 1;
            p.text_data_param.time_steps = 80;
            p.text_data_param.shuffle = false;
            p.text_data_param.enable_normal_encoder_output = true;
            p.text_data_param.enable_reverse_encoder_output = true;
            p.text_data_param.encoder_source = strEncSrc;
            p.text_data_param.decoder_source = strDecSrc;
            p.text_data_param.sample_size = 1000;

            int nT = (int)p.text_data_param.time_steps;
            int nN = (int)p.text_data_param.batch_size;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            m_log.CHECK_EQ((int)layer.type, (int)LayerParameter.LayerType.TEXT_DATA, "The layer type should be TEXT_DATA!");

            TopVec.Clear();
            TopVec.Add(m_blobDecInput);
            TopVec.Add(m_blobDecClip);
            TopVec.Add(m_blobEncInput1);
            TopVec.Add(m_blobEncInput2);
            TopVec.Add(m_blobEncClip);
            TopVec.Add(m_blobVocabCount);
            TopVec.Add(m_blobDecTarget);

            BottomVec.Clear();

            layer.Setup(BottomVec, TopVec);

            List<int> rgEncInput = getInput(0, ((TextDataLayer<T>)layer).Vocabulary, strEncSrc);
            List<int> rgEncInputR = new List<int>(rgEncInput);
            rgEncInputR.Reverse();
            List<int> rgDecInput = getInput(0, ((TextDataLayer<T>)layer).Vocabulary, strDecSrc);

            int nVocabCount = ((TextDataLayer<T>)layer).Vocabulary.VocabularCount;
            int nVocabCount1 = (int)convert(m_blobVocabCount.GetData(0));
            m_log.CHECK_EQ(nVocabCount + 2, nVocabCount1, "The vocab count is not as expected!");

            int nSeqIdx = 1;

            for (int i = 0; i < 10; i++)
            {
                layer.Forward(BottomVec, TopVec);

                int nDataIdx = 0;
                bool bRes = verify_top_data(TopVec, nN, nT, nDataIdx, rgEncInput, rgEncInputR, rgDecInput, nVocabCount + 2);

                while (bRes)
                {
                    layer.Forward(BottomVec, TopVec);
                    nDataIdx++;
                    bRes = verify_top_data(TopVec, nN, nT, nDataIdx, rgEncInput, rgEncInputR, rgDecInput, nVocabCount + 2);
                }

                rgEncInput = getInput(nSeqIdx, ((TextDataLayer<T>)layer).Vocabulary, strEncSrc);
                rgEncInputR = new List<int>(rgEncInput);
                rgEncInputR.Reverse();
                rgDecInput = getInput(nSeqIdx, ((TextDataLayer<T>)layer).Vocabulary, strDecSrc);
                nSeqIdx++;
            }
        }

        public void TestForwardRunPhase()
        {
            Stopwatch sw = new Stopwatch();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\";
            string strEncSrc = strPath + "human_text.txt";
            string strDecSrc = strPath + "robot_text.txt";

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TEXT_DATA);
            p.text_data_param.batch_size = 1;
            p.text_data_param.time_steps = 80;
            p.text_data_param.shuffle = false;
            p.text_data_param.enable_normal_encoder_output = true;
            p.text_data_param.enable_reverse_encoder_output = true;
            p.text_data_param.encoder_source = strEncSrc;
            p.text_data_param.decoder_source = strDecSrc;
            p.text_data_param.sample_size = 1000;
            p.phase = Phase.RUN;

            int nT = (int)p.text_data_param.time_steps;
            int nN = (int)p.text_data_param.batch_size;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            m_log.CHECK_EQ((int)layer.type, (int)LayerParameter.LayerType.TEXT_DATA, "The layer type should be TEXT_DATA!");

            TopVec.Clear();
            TopVec.Add(m_blobDecInput);
            TopVec.Add(m_blobDecClip);
            TopVec.Add(m_blobEncInput1);
            TopVec.Add(m_blobEncInput2);
            TopVec.Add(m_blobEncClip);
            TopVec.Add(m_blobVocabCount);

            BottomVec.Clear();
            BottomVec.Add(m_blobBtmDecInput);
            BottomVec.Add(m_blobBtmEncInput1);
            BottomVec.Add(m_blobBtmEncInput2);
            BottomVec.Add(m_blobBtmEncClip);

            layer.Setup(BottomVec, TopVec);


            // Verify the bottom data.
            List<int> rgBtmEncInput = getInput(((TextDataLayer<T>)layer).Vocabulary, "what is your name");
            List<int> rgBtmEncInputR = getInput(((TextDataLayer<T>)layer).Vocabulary, "what is your name", true);
            List<int> rgDecInput = new List<int>() { 1 };

            int nVocabCount = ((TextDataLayer<T>)layer).Vocabulary.VocabularCount;

            int? nDecInput = null;
            for (int i = 0; i < 10; i++)
            {
                ((TextDataLayer<T>)layer).PreProcessInput("what is your name", nDecInput, BottomVec);
                verify_btm_data(BottomVec, nN, nT, rgBtmEncInput, rgBtmEncInputR, nDecInput.GetValueOrDefault(1));

                layer.Forward(BottomVec, TopVec);

                int nDataIdx = 0;
                bool bRes = verify_top_data(TopVec, nN, nT, nDataIdx, rgBtmEncInput, rgBtmEncInputR, rgDecInput, nVocabCount + 2);

                while (bRes)
                {
                    nDecInput = nDecInput.GetValueOrDefault(1) + 1;

                    ((TextDataLayer<T>)layer).PreProcessInput("what is your name", nDecInput, BottomVec);
                    verify_btm_data(BottomVec, nN, nT, rgBtmEncInput, rgBtmEncInputR, nDecInput);

                    layer.Forward(BottomVec, TopVec);

                    rgDecInput[0] = nDecInput.Value;
                    bRes = verify_top_data(TopVec, nN, nT, nDataIdx, rgBtmEncInput, rgBtmEncInputR, rgDecInput, nVocabCount + 2);
                }
            }
        }

        private void verify_btm_data(BlobCollection<T> colBtm, int nN, int nT, List<int> rgEncInput, List<int> rgEncInputR, int? nDecInput)
        {
            m_log.CHECK_EQ(nN, 1, "Currently, only batch = 1 is supported!");

            float[] rgfDecInput = convertF(colBtm[0].mutable_cpu_data);
            float[] rgfEncInput1 = convertF(colBtm[1].mutable_cpu_data);
            float[] rgfEncInput2 = convertF(colBtm[2].mutable_cpu_data);
            float[] rgfEncClip = convertF(colBtm[3].mutable_cpu_data);

            float fDecInput = rgfDecInput[0];

            m_log.CHECK_EQ(rgfDecInput.Length, nN, "The decoder input should = " + nN.ToString());
            m_log.CHECK_EQ(rgfEncInput1.Length, nN * nT, "The encoder input1 should = " + (nN * nT).ToString());
            m_log.CHECK_EQ(rgfEncInput2.Length, nN * nT, "The encoder input2 should = " + (nN * nT).ToString());
            m_log.CHECK_EQ(rgfEncClip.Length, nN * nT, "The encoder clip should = " + (nN * nT).ToString());

            m_log.CHECK_EQ(nDecInput.GetValueOrDefault(1), fDecInput, "The dec input is incorrect!");

            for (int i = 0; i < nT; i++)
            {
                if (i < rgEncInput.Count)
                {
                    m_log.CHECK_EQ(rgfEncInput1[i], rgEncInput[i], "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncInput2[i], rgEncInputR[i], "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncClip[i], (i == 0) ? 0 : 1, "The encoder input1 is incorrect!");
                }
                else
                {
                    m_log.CHECK_EQ(rgfEncInput1[i], 0, "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncInput2[i], 0, "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncClip[i], 0, "The encoder input1 is incorrect!");
                }
            }
        }

        private bool verify_top_data(BlobCollection<T> colTop, int nN, int nT, int nDataIdx, List<int> rgEncInput, List<int> rgEncInputR, List<int> rgDecInput, int nVocabCount)
        {
            m_log.CHECK_EQ(nN, 1, "Currently, only batch = 1 is supported!");

            float[] rgfDecInput = convertF(colTop[0].mutable_cpu_data);
            float[] rgfDecClip = convertF(colTop[1].mutable_cpu_data);
            float[] rgfEncInput1 = convertF(colTop[2].mutable_cpu_data);
            float[] rgfEncInput2 = convertF(colTop[3].mutable_cpu_data);
            float[] rgfEncClip = convertF(colTop[4].mutable_cpu_data);
            float[] rgfVocabCount = convertF(colTop[5].mutable_cpu_data);
            float[] rgfDecTarget = (colTop.Count > 6) ? convertF(colTop[6].mutable_cpu_data) : null;

            float fDecInput = 1;
            if (nDataIdx > 0)
                fDecInput = rgDecInput[nDataIdx - 1];

            float fDecTarget = 0;
            if (nDataIdx < rgDecInput.Count)
                fDecTarget = rgDecInput[nDataIdx];

            float fDecClip = (nDataIdx == 0) ? 0 : 1;

            m_log.CHECK_EQ(rgfVocabCount.Length, 1, "The vocab count length should = 1");
            m_log.CHECK_EQ(rgfVocabCount[0], nVocabCount, "The vocab count is incorrect!");

            m_log.CHECK_EQ(rgfDecInput.Length, nN, "The decoder input should = " + nN.ToString());
            m_log.CHECK_EQ(rgfDecClip.Length, nN, "The decoder clip should = " + nN.ToString());
            if (rgfDecTarget != null)
                m_log.CHECK_EQ(rgfDecTarget.Length, nN, "The decoder target should = " + nN.ToString());

            m_log.CHECK_EQ(rgfEncInput1.Length, nN * nT, "The encoder input1 should = " + (nN * nT).ToString());
            m_log.CHECK_EQ(rgfEncInput2.Length, nN * nT, "The encoder input2 should = " + (nN * nT).ToString());
            m_log.CHECK_EQ(rgfEncClip.Length, nN * nT, "The encoder clip should = " + (nN * nT).ToString());

            m_log.CHECK_EQ(rgfDecInput[0], fDecInput, "The dec input is incorrect!");
            m_log.CHECK_EQ(rgfDecClip[0], fDecClip, "The dec clip is incorrect!");
            if (rgfDecTarget != null)
                m_log.CHECK_EQ(rgfDecTarget[0], fDecTarget, "The dec target is incorrect!");

            for (int i = 0; i < nT; i++)
            {
                if (i < rgEncInput.Count)
                {
                    m_log.CHECK_EQ(rgfEncInput1[i], rgEncInput[i], "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncInput2[i], rgEncInputR[i], "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncClip[i], (i == 0) ? 0 : 1, "The encoder input1 is incorrect!");
                }
                else
                {
                    m_log.CHECK_EQ(rgfEncInput1[i], 0, "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncInput2[i], 0, "The encoder input1 is incorrect!");
                    m_log.CHECK_EQ(rgfEncClip[i], 0, "The encoder input1 is incorrect!");
                }
            }

            if (nDataIdx == rgDecInput.Count - 1)
                return false;
            else
                return true;
        }
    }
}
