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
using MyCaffe.layers.beta.TextData;

namespace MyCaffe.test
{
    [TestClass]
    public class TestModelDataLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            ModelDataLayerTest test = new ModelDataLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IModelDataLayerTest t in test.Tests)
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
            ModelDataLayerTest test = new ModelDataLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IModelDataLayerTest t in test.Tests)
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

    interface IModelDataLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
    }

    class ModelDataLayerTest : TestBase
    {
        SettingsCaffe m_settings;
        IXImageDatabaseBase m_db;
        CancelEvent m_evtCancel = new CancelEvent();

        public ModelDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Text Model Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
            m_settings = new SettingsCaffe();
            m_settings.EnableLabelBalancing = false;
            m_settings.EnableLabelBoosting = false;
            m_settings.EnablePairInputSelection = false;
            m_settings.EnableRandomInputSelection = false;

            m_db = createImageDb(null);
            m_db.InitializeWithDsName1(m_settings, "MNIST");
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ModelDataLayerTest<double>(strName, nDeviceID, engine, this);
            else
                return new ModelDataLayerTest<float>(strName, nDeviceID, engine, this);
        }

        protected override void dispose()
        {
            if (m_db != null)
            {
                ((IDisposable)m_db).Dispose();
                m_db = null;
            }

            base.dispose();
        }

        public IXImageDatabaseBase db
        {
            get { return m_db; }
        }

        public SettingsCaffe Settings
        {
            get { return m_settings; }
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }
    }

    class ModelDataLayerTest<T> : TestEx<T>, IModelDataLayerTest
    {
        ModelDataLayerTest m_parent;
        Blob<T> m_blobDecInput;
        Blob<T> m_blobDecClip;
        Blob<T> m_blobDecTarget;
        Blob<T> m_blobEncInput1;
        Blob<T> m_blobEncClip;
        Blob<T> m_blobVocabCount;

        Blob<T> m_blobBtmDecInput;
        Blob<T> m_blobBtmEncInput1;
        Blob<T> m_blobBtmEncClip;

        public ModelDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine, ModelDataLayerTest parent)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_parent = parent;
            m_engine = engine;

            m_blobDecInput = new Blob<T>(m_cuda, m_log, false);
            m_blobDecInput.Name = "dec_input";

            m_blobDecClip = new Blob<T>(m_cuda, m_log, false);
            m_blobDecClip.Name = "dec_clip";

            m_blobDecTarget = new Blob<T>(m_cuda, m_log, false);
            m_blobDecTarget.Name = "dec_target";

            m_blobEncInput1 = new Blob<T>(m_cuda, m_log, false);
            m_blobEncInput1.Name = "enc_input1";

            m_blobEncClip = new Blob<T>(m_cuda, m_log, false);
            m_blobEncClip.Name = "enc_clip";

            m_blobVocabCount = new Blob<T>(m_cuda, m_log, false);
            m_blobVocabCount.Name = "vocab_count";

            m_blobBtmDecInput = new Blob<T>(m_cuda, m_log, false);
            m_blobBtmDecInput.Name = "btm_dec_input";

            m_blobBtmEncInput1 = new Blob<T>(m_cuda, m_log, false);
            m_blobBtmEncInput1.Name = "btm_enc_input1";

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
            m_blobEncClip.Dispose();
            m_blobVocabCount.Dispose();

            m_blobBtmDecInput.Dispose();
            m_blobBtmEncInput1.Dispose();
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

        public List<SimpleResult> FillDummyData()
        {
            // Create the dummy results for MNIST
            TestMyCaffeImageDatabase2 test = new TestMyCaffeImageDatabase2();
            test.TestGetAllResults();
            return test.Results;
        }

        public void TestSetup()
        {
            Stopwatch sw = new Stopwatch();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MODEL_DATA);
            p.model_data_param.source.Add("MNIST.training");
            p.model_data_param.source.Add("MNIST.testing");
            p.model_data_param.batch_size = 1;
            p.model_data_param.time_steps = 80;
            p.model_data_param.input_dim = 3;
            p.model_data_param.shuffle = false;
            p.model_data_param.sample_size = 1000;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            m_log.CHECK_EQ((int)layer.type, (int)LayerParameter.LayerType.MODEL_DATA, "The layer type should be MODEL_DATA!");

            TopVec.Clear();
            TopVec.Add(m_blobDecInput);
            TopVec.Add(m_blobDecClip);
            TopVec.Add(m_blobEncInput1);
            TopVec.Add(m_blobEncClip);
            TopVec.Add(m_blobVocabCount);
            TopVec.Add(m_blobDecTarget);

            BottomVec.Clear();

            List<SimpleResult> rgRes = FillDummyData();

            layer.Setup(BottomVec, TopVec);

            int nT = (int)p.model_data_param.time_steps;
            int nN = (int)p.model_data_param.batch_size;
            int nI = (int)p.model_data_param.input_dim;

            verify_shape(TopVec[0], new List<int>() { 1, nN, 1 });   // dec input
            verify_shape(TopVec[1], new List<int>() { 1, nN });         // dec clip
            verify_shape(TopVec[2], new List<int>() { nT, nN, nI });  // enc input1
            verify_shape(TopVec[3], new List<int>() { nT, nN });        // enc clip
            verify_shape(TopVec[4], new List<int>() { 1 });    // vocab count
            verify_shape(TopVec[5], new List<int>() { 1, nN, 1 });   // dec target
        }

        public void TestForward()
        {
            Stopwatch sw = new Stopwatch();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MODEL_DATA);
            p.model_data_param.source.Add("MNIST.training");
            p.model_data_param.source.Add("MNIST.testing");
            p.model_data_param.batch_size = 1;
            p.model_data_param.time_steps = 80;
            p.model_data_param.input_dim = 3;
            p.model_data_param.shuffle = false;
            p.model_data_param.sample_size = 1000;

            int nT = (int)p.model_data_param.time_steps;
            int nN = (int)p.model_data_param.batch_size;
            int nI = (int)p.model_data_param.input_dim;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            m_log.CHECK_EQ((int)layer.type, (int)LayerParameter.LayerType.MODEL_DATA, "The layer type should be MODEL_DATA!");

            TopVec.Clear();
            TopVec.Add(m_blobDecInput);
            TopVec.Add(m_blobDecClip);
            TopVec.Add(m_blobEncInput1);
            TopVec.Add(m_blobEncClip);
            TopVec.Add(m_blobVocabCount);
            TopVec.Add(m_blobDecTarget);

            BottomVec.Clear();

            List<SimpleResult> rgRes = FillDummyData();

            layer.Setup(BottomVec, TopVec);

            int nVocabCount = ((ModelDataLayer<T>)layer).DecoderVocabularyCount;
            int nVocabCount1 = (int)convert(m_blobVocabCount.GetData(0));
            m_log.CHECK_EQ(nVocabCount + 2, nVocabCount1, "The vocab count is not as expected!");

            int nSeqIdx = 1;

            for (int i = 0; i < 10; i++)
            {
                layer.Forward(BottomVec, TopVec);

                int nDataIdx = 0;
                bool bRes = verify_top_data(TopVec, nN, nT, nI, nDataIdx, rgRes[i], nVocabCount + 2);

                while (bRes)
                {
                    layer.Forward(BottomVec, TopVec);
                    nDataIdx++;
                    bRes = verify_top_data(TopVec, nN, nT, nI, nDataIdx, rgRes[i], nVocabCount + 2);
                }

                nSeqIdx++;
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

        private bool verify_top_data(BlobCollection<T> colTop, int nN, int nT, int nI, int nDataIdx, SimpleResult res, int nVocabCount, bool bIgnoreTarget = false)
        {
            m_log.CHECK_EQ(nN, 1, "Currently, only batch = 1 is supported!");

            float[] rgfDecInput = convertF(colTop[0].mutable_cpu_data);
            float[] rgfDecClip = convertF(colTop[1].mutable_cpu_data);
            float[] rgfEncInput1 = convertF(colTop[2].mutable_cpu_data);
            float[] rgfEncClip = convertF(colTop[3].mutable_cpu_data);
            float[] rgfVocabCount = convertF(colTop[4].mutable_cpu_data);
            float[] rgfDecTarget = (colTop.Count > 5) ? convertF(colTop[5].mutable_cpu_data) : null;

            float fDecInput = 1;
            if (nDataIdx > 0)
                fDecInput = res.Target[nDataIdx - 1];

            float fDecTarget = 0;
            if (nDataIdx < res.Target.Length)
                fDecTarget = res.Target[nDataIdx];

            float fDecClip = (nDataIdx == 0) ? 0 : 1;

            m_log.CHECK_EQ(rgfVocabCount.Length, 1, "The vocab count length should = 1");
            m_log.CHECK_EQ(rgfVocabCount[0], nVocabCount, "The vocab count is incorrect!");

            m_log.CHECK_EQ(rgfDecInput.Length, nN, "The decoder input should = " + nN.ToString());
            m_log.CHECK_EQ(rgfDecClip.Length, nN, "The decoder clip should = " + nN.ToString());
            if (rgfDecTarget != null)
                m_log.CHECK_EQ(rgfDecTarget.Length, nN, "The decoder target should = " + nN.ToString());

            m_log.CHECK_EQ(rgfEncInput1.Length, nN * nT * nI, "The encoder input1 should = " + (nN * nT * nI).ToString());
            m_log.CHECK_EQ(rgfEncClip.Length, nN * nT, "The encoder clip should = " + (nN * nT).ToString());

            m_log.CHECK_EQ(rgfDecInput[0], fDecInput, "The dec input is incorrect!");
            m_log.CHECK_EQ(rgfDecClip[0], fDecClip, "The dec clip is incorrect!");
            if (rgfDecTarget != null && !bIgnoreTarget)
                m_log.CHECK_EQ(rgfDecTarget[0], fDecTarget, "The dec target is incorrect!");

            for (int i = 0; i < nT; i++)
            {
                if (i < res.BatchCount)
                {
                    for (int j = 0; j < nI; j++)
                    {
                        int nIdx = i * nI + j;
                        m_log.CHECK_EQ(rgfEncInput1[nIdx], res.Result[nIdx], "The encoder input1 is incorrect!");
                    }
                    m_log.CHECK_EQ(rgfEncClip[i], (i == 0) ? 0 : 1, "The encoder input1 is incorrect!");
                }
                else
                {
                    for (int j = 0; j < nI; j++)
                    {
                        int nIdx = i * nI + j;
                        m_log.CHECK_EQ(rgfEncInput1[nIdx], 0, "The encoder input1 is incorrect!");
                    }
                    m_log.CHECK_EQ(rgfEncClip[i], 0, "The encoder input1 is incorrect!");
                }
            }

            if (nDataIdx == res.Target.Length - 1)
                return false;
            else
                return true;
        }
    }
}
