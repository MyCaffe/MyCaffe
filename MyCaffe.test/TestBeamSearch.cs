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
using System.Diagnostics;
using MyCaffe.layers.beta;
using System.IO;
using MyCaffe.layers.beta.TextData;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBeamSearch
    {
        [TestMethod]
        public void Test()
        {
            BeamSearchTest test = new BeamSearchTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IBeamSearchTest t in test.Tests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IBeamSearchTest : ITest
    {
        void Test();
    }

    class BeamSearchTest : TestBase
    {
        public BeamSearchTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Beam Search Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BeamSearchTest2<double>(strName, nDeviceID, engine);
            else
                return new BeamSearchTest2<float>(strName, nDeviceID, engine);
        }
    }

    class BeamSearchTest2<T> : TestEx<T>, IBeamSearchTest
    {
        Net<T> m_net = null;
        List<string> m_rgTestSequences = new List<string>();
        List<List<int>> m_rgrgTestSequenceIndexes = new List<List<int>>();

        public BeamSearchTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;

            NetParameter net_param = new NetParameter();

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { 1, 1, 1 }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { 80, 1, 1 }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { 80, 1, 1 }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { 80, 1 }));
            input.top.Add("dec");
            input.top.Add("enc");
            input.top.Add("encr");
            input.top.Add("encc");
            net_param.layer.Add(input);

            string strModel = net_param.ToProto("root").ToString();
            m_net = new Net<T>(m_cuda, m_log, net_param, new CancelEvent(), null);
            InputLayerEx<T> layer = new InputLayerEx<T>(m_cuda, m_log, m_net.layers[0]);
            layer.OnGetData += Layer_OnGetData;
            m_net.layers[0] = layer;

            m_rgTestSequences.Add("rdany but you can call me dany");
            m_rgTestSequences.Add("rdany call me dany");
            m_rgTestSequences.Add("rdany you can call me dany");
            m_rgTestSequences.Add("my name is dany");
            m_rgTestSequences.Add("call me dany");
            m_rgrgTestSequenceIndexes = new List<List<int>>();

            foreach (string strSequence in m_rgTestSequences)
            {
                string[] rgstrWords = strSequence.Split(' ');
                List<int> rgIdx = new List<int>();

                foreach (string strWord in rgstrWords)
                {
                    int nIdx = layer.Vocabulary.WordToIndex(strWord);
                    rgIdx.Add(nIdx);
                }

                m_rgrgTestSequenceIndexes.Add(rgIdx);
            }
        }

        private void Layer_OnGetData(object sender, TestGetDataArgs<T> e)
        {
            int nIdx;

            e.Data.SetData(0);

            if (e.Input == 1)
            {
                nIdx = e.Vocabulary.WordToIndex("rdany");
                e.Data.SetData(0.3, nIdx);
                nIdx = e.Vocabulary.WordToIndex("my");
                e.Data.SetData(0.25, nIdx);
                nIdx = e.Vocabulary.WordToIndex("call");
                e.Data.SetData(0.2, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("rdany"))
            {
                nIdx = e.Vocabulary.WordToIndex("but");
                e.Data.SetData(0.3, nIdx);
                nIdx = e.Vocabulary.WordToIndex("call");
                e.Data.SetData(0.25, nIdx);
                nIdx = e.Vocabulary.WordToIndex("you");
                e.Data.SetData(0.2, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("my"))
            {
                nIdx = e.Vocabulary.WordToIndex("name");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("but"))
            {
                nIdx = e.Vocabulary.WordToIndex("you");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("can"))
            {
                nIdx = e.Vocabulary.WordToIndex("call");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("call"))
            {
                nIdx = e.Vocabulary.WordToIndex("me");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("you"))
            {
                nIdx = e.Vocabulary.WordToIndex("can");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("me"))
            {
                nIdx = e.Vocabulary.WordToIndex("dany");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("name"))
            {
                nIdx = e.Vocabulary.WordToIndex("is");
                e.Data.SetData(0.3, nIdx);
            }
            else if (e.Input == e.Vocabulary.WordToIndex("is"))
            {
                nIdx = e.Vocabulary.WordToIndex("dany");
                e.Data.SetData(0.3, nIdx);
            }
            else
            {
                e.Data.SetData(2); // EOS
            }
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        private void dispose1(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void Test()
        {
            // Test the beam-search
            BeamSearch<T> search = new BeamSearch<T>(m_net);

            List<Tuple<double, bool, List<Tuple<string, int, double>>>> rgSequences = search.Search(new PropertySet("InputData=what is your name"), 3);

            m_log.EnableTrace = true;

            List<string> rgstrExpected = new List<string>();
            rgstrExpected.Add("rdany call me dany");
            rgstrExpected.Add("my name is dany");
            rgstrExpected.Add("rdany but you can call me dany");

            for (int i = 0; i < rgSequences.Count; i++)
            {
                string strOut = rgSequences[i].Item1.ToString("P");
                string strActual = "";
                strOut += " '";

                for (int j = 0; j < rgSequences[i].Item3.Count; j++)
                {
                    strActual += rgSequences[i].Item3[j].Item1;
                    strActual += " ";
                }

                strActual = strActual.Trim();

                if (strActual != rgstrExpected[i])
                    m_log.FAIL("The actual at index #" + i.ToString() + " should equal '" + rgstrExpected[i] + "', but instead equals '" + strActual + "'.");

                strOut += strActual;
                strOut += "'";

                m_log.WriteLine(strOut);
            }
        }
    }

    class InputLayerEx<T> : InputLayer<T>
    {
        TextDataLayer<T> m_dataLayer;
        int m_nInput = 1;

        public event EventHandler<TestGetDataArgs<T>> OnGetData;

        public InputLayerEx(CudaDnn<T> cuda, Log log, Layer<T> layer) : base(cuda, log, layer.layer_param)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            strPath += "\\mycaffe\\test_data\\data\\text\\";
            LayerParameter text_param = new LayerParameter(LayerParameter.LayerType.TEXT_DATA);
            text_param.text_data_param.batch_size = 1;
            text_param.text_data_param.decoder_source = strPath + "robot_text.txt";
            text_param.text_data_param.encoder_source = strPath + "human_text.txt";
            text_param.text_data_param.enable_normal_encoder_output = true;
            text_param.text_data_param.enable_reverse_encoder_output = true;
            text_param.text_data_param.sample_size = 1000;
            text_param.text_data_param.shuffle = false;
            text_param.text_data_param.time_steps = 80;
            text_param.phase = Phase.TEST;

            m_dataLayer = new TextDataLayer<T>(cuda, log, text_param);
            BlobCollection<T> colBottom = new BlobCollection<T>();
            BlobCollection<T> colTop = new BlobCollection<T>();

            colTop.Add(new Blob<T>(cuda, log));
            colTop.Add(new Blob<T>(cuda, log));
            colTop.Add(new Blob<T>(cuda, log));
            colTop.Add(new Blob<T>(cuda, log));
            colTop.Add(new Blob<T>(cuda, log));
            colTop.Add(new Blob<T>(cuda, log));
            colTop.Add(new Blob<T>(cuda, log));

            m_dataLayer.Setup(colBottom, colTop);

            colTop.Dispose();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public override bool SupportsPostProcessing => true;

        public override bool SupportsPreProcessing => true;

        public override BlobCollection<T> PreProcessInput(PropertySet customInput, out int nSeqLen, BlobCollection<T> colBottom = null)
        {
            return m_dataLayer.PreProcessInput(customInput, out nSeqLen, colBottom);
        }

        public override bool PreProcessInput(string strEncInput, int? nDecInput, BlobCollection<T> colBottom)
        {
            return m_dataLayer.PreProcessInput(strEncInput, nDecInput, colBottom);
        }
        
        public override List<Tuple<string, int, double>> PostProcessOutput(Blob<T> blobSoftmax, int nK = 1)
        {
            return m_dataLayer.PostProcessOutput(blobSoftmax, nK);
        }

        public override string PostProcessOutput(int nIdx)
        {
            return m_dataLayer.PostProcessOutput(nIdx);
        }

        public Vocabulary Vocabulary
        {
            get { return m_dataLayer.Vocabulary; }
        }

        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nInput = (int)convertF(colTop[0].mutable_cpu_data)[0];
            int nCount = m_dataLayer.Vocabulary.VocabularCount;
            // Using top[0] as temporary output, just for testing.
            colTop[0].Reshape(1, 1, nCount + 2, 1);
        }

        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            TestGetDataArgs<T> args = new TestGetDataArgs<T>(m_nInput, m_dataLayer.Vocabulary);
            args.Data = colTop[0];
            OnGetData(this, args);
        }
    }

    class TestGetDataArgs<T>
    {
        Blob<T> m_data;
        int m_nInput;
        Vocabulary m_vocab;

        public TestGetDataArgs(int nInput, Vocabulary v)
        {
            m_nInput = nInput;
            m_vocab = v;
        }

        public Vocabulary Vocabulary
        {
            get { return m_vocab; }
        }

        public int Input
        {
            get { return m_nInput; }
        }

        public Blob<T> Data
        {
            get { return m_data; }
            set { m_data = value; }
        }
    }
}
