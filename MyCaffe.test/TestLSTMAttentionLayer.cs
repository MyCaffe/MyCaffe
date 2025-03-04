﻿using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.basecode;
using System.Diagnostics;
using MyCaffe.solvers;
using System.IO;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestLSTMAttentionLayer
    {
        Log m_log = null;
        CancelEvent m_evtCancel = null;
        string m_strParams;
        bool m_bDisableDouble = false;
        bool m_bDisableFloat = false;
        bool m_bShortModel = true;

        public TestLSTMAttentionLayer()
        {
        }

        public TestLSTMAttentionLayer(Log log, CancelEvent evtCancel, string strParams)
        {
            m_log = log;
            m_evtCancel = evtCancel;
            m_strParams = strParams;

            if (m_strParams != null)
            {
                string[] rgstr = m_strParams.Split();
                if (rgstr.Length > 4 && rgstr[4].Trim(',') != "T")
                    m_bDisableDouble = true;

                if (rgstr.Length > 5 && rgstr[5].Trim(',') != "T")
                    m_bDisableFloat = true;

                if (rgstr.Length > 6 && rgstr[6].Trim(',') != "S")
                    m_bShortModel = false;
            }
        }

        public static string ParameterDescriptions
        {
            get { return LSTMAttentionLayerTest<double>.ParameterDescriptions; }
        }

        public static string ParameterDefaults
        {
            get { return LSTMAttentionLayerTest<double>.ParameterDefaults; }
        }

        [TestMethod]
        public void TestSetup()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.EnabledTests)
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
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.EnabledTests)
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
        public void TestGradientDefault()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.Tests)
                {
                    t.TestGradientDefault();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBatchDefault()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.Tests)
                {
                    t.TestGradientBatchDefault();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientClipMask()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.Tests)
                {
                    t.TestGradientClipMask();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBatchClipMask()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.Tests)
                {
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    t.TestGradientBatchClipMask();
                    sw.Stop();

                    if (m_log != null)
                        m_log.WriteLine("Testing GradientBatchClipMask<" + t.DataType.ToString() + "> - " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAttention()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.Tests)
                {
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    t.TestGradientAttention();
                    sw.Stop();

                    if (m_log != null)
                        m_log.WriteLine("Testing GradientAttention<" + t.DataType.ToString() + "> - " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAttention2()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel);

            try
            {
                foreach (ILSTMAttentionLayerTest t in test.Tests)
                {
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    t.TestGradientAttention2();
                    sw.Stop();

                    if (m_log != null)
                        m_log.WriteLine("Testing GradientAttention<" + t.DataType.ToString() + "> - " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingClockworkGradient()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel, m_strParams);

            try
            {
                if (m_bDisableDouble)
                    test.Tests[0].Enabled = false;

                if (m_bDisableFloat)
                    test.Tests[1].Enabled = false;

                foreach (ILSTMAttentionLayerTest t in test.EnabledTests)
                {
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    t.TestTraining(960, 23, 1, m_bShortModel);
                    sw.Stop();

                    if (m_log != null)
                        m_log.WriteLine("Test Completed - Training<" + t.DataType.ToString() + "> - " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingClockworkGradient2()
        {
            LSTMAttentionLayerTest test = new LSTMAttentionLayerTest(m_log, m_evtCancel, m_strParams);

            try
            {
                if (m_bDisableDouble)
                    test.Tests[0].Enabled = false;

                if (m_bDisableFloat)
                    test.Tests[1].Enabled = false;

                foreach (ILSTMAttentionLayerTest t in test.EnabledTests)
                {
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    t.TestTraining2(960, 23, 320, 1, m_bShortModel);
                    sw.Stop();

                    if (m_log != null)
                        m_log.WriteLine("Test Completed - Training<" + t.DataType.ToString() + "> - " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface ILSTMAttentionLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestGradientDefault();
        void TestGradientBatchDefault();
        void TestGradientClipMask();
        void TestGradientBatchClipMask();
        void TestGradientAttention();
        void TestGradientAttention2();
        void TestTraining(int nTotalDataLength, int nNumOutput, int nBatch, bool bShortModel, int nMaxIter = 10000);
        void TestTraining2(int nTotalDataLength, int nNumOutput, int nTimeSteps, int nBatch, bool bShortModel, int nMaxIter = 10000);
    }

    class LSTMAttentionLayerTest : TestBase
    {
        Log m_log1;
        CancelEvent m_evtCancel1;
        string m_strParams;

        public LSTMAttentionLayerTest(Log log, CancelEvent evtCancel, string strParams = null, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Attention Decoder Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
            m_log1 = log;
            m_evtCancel1 = evtCancel;
            m_strParams = strParams;

            foreach (ITest test in base.Tests)
            {
                if (test.DataType == DataType.DOUBLE)
                    ((LSTMAttentionLayerTest<double>)test).Setup(m_log1, m_evtCancel1, m_strParams);
                else
                    ((LSTMAttentionLayerTest<float>)test).Setup(m_log1, m_evtCancel1, m_strParams);
            }
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LSTMAttentionLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LSTMAttentionLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LSTMAttentionLayerTest<T> : TestEx<T>, ILSTMAttentionLayerTest
    {
        Log m_log1 = null;
        CancelEvent m_evtCancel = new CancelEvent();
        string m_strParams;
        Blob<T> m_blob_bottom2;
        Blob<T> m_blobEncoding;
        Blob<T> m_blobEncodingClip;

        public LSTMAttentionLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            initialize(nDeviceID);
        }

        public void Setup(Log log, CancelEvent evtCancel, string strParams)
        {
            m_log1 = log;
            m_strParams = strParams;

            if (evtCancel != null)
                m_evtCancel = evtCancel;

            if (m_log1 != null)
                m_log.OnWriteLine += new EventHandler<LogArg>(m_log_OnWriteLine);
        }

        private void initialize(int nDeviceID)
        {
            if (m_blob_bottom != null)
                m_blob_bottom.Dispose();

            if (m_blob_bottom2 != null)
                m_blob_bottom2.Dispose();

            if (m_blob_top != null)
                m_blob_top.Dispose();

            if (nDeviceID != m_cuda.GetDeviceID())
            {
                if (m_cuda != null)
                    m_cuda.Dispose();

                m_cuda = GetCuda(nDeviceID);
            }

            m_blob_bottom = new Blob<T>(m_cuda, m_log, 4, 3, 2, 1);

            m_filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            m_filler.Fill(m_blob_bottom);

            m_blob_top = new Blob<T>(m_cuda, m_log);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            TopVec.Clear();
            TopVec.Add(m_blob_top);

            m_blob_bottom2 = new Blob<T>(m_cuda, m_log, 4, 3, 1, 1);
            m_blob_bottom2.SetData(0);

            m_blobEncoding = new Blob<T>(m_cuda, m_log, 3, 3, 1, 1);
            m_blobEncoding.SetData(0);

            m_blobEncodingClip = new Blob<T>(m_cuda, m_log, 3, 3, 1, 1);
            m_blobEncodingClip.SetData(0);
        }

        public static string ParameterDescriptions
        {
            get { return "DeviceID, MaxIter, NumOutput, BatchSize, DoubleTest, FloatTest, S=ShortModel | L=LongModel\n(example: '1, 10000, 23, 3, T, F, S')"; }
        }

        public static string ParameterDefaults
        {
            get { return "1, 10000, 23, 1, T, F, S"; }
        }

        void m_log_OnWriteLine(object sender, LogArg e)
        {
            m_log1.Progress = e.Progress;
            m_log1.WriteLine(e.Message);
        }

        protected override void dispose()
        {
            m_blob_bottom2.Dispose();
            m_blobEncoding.Dispose();
            m_blobEncodingClip.Dispose();
            base.dispose();
        }

        public void Fill(BlobCollection<T> colBottom, Blob<T> blobData, Blob<T> blobClip, int nT, int nB, int nI, bool bAllClip)
        {
            blobData.Reshape(nT, nB, nI, 1);

            float[] rgData = convertF(blobData.mutable_cpu_data);
            for (int t = 0; t < blobData.num; t++)
            {
                for (int b = 0; b < blobData.channels; b++)
                {
                    for (int i = 0; i < nI; i++)
                    {
                        int nIdx = (t * nB * nI) + (b * nI) + i;
                        rgData[nIdx] = (t + 1) + (b * 0.1f) + (i * 0.01f);
                    }
                }
            }

            blobData.mutable_cpu_data = convert(rgData);

            List<int> rgShape = Utility.Clone<int>(blobData.shape());
            while (rgShape.Count > 2)
            {
                rgShape.RemoveAt(rgShape.Count - 1);
            }
            blobClip.Reshape(rgShape);

            blobClip.SetData(1);
            if (!bAllClip)
                blobClip.SetData(0, 0);

            colBottom.Add(blobData);
            colBottom.Add(blobClip);
        }


        public Blob<T> Bottom2
        {
            get { return m_blob_bottom2; }
            set { m_blob_bottom2 = value; }
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            p.min = -0.1;
            p.max = 0.1;
            return p;
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 5;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant");
            p.lstm_attention_param.bias_filler.value = 2;
            p.lstm_attention_param.clipping_threshold = 2.44;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                if (layer.type != LayerParameter.LayerType.LSTM_ATTENTION)
                    m_log.FAIL("Invalid layer type, expected LSTM_ATTENTION.");

                LSTMAttentionLayer<T> lstm = (LSTMAttentionLayer<T>)layer;

                Assert.AreEqual(lstm.layer_param.lstm_attention_param.num_output, (uint)5);
                Assert.AreEqual(lstm.layer_param.lstm_attention_param.weight_filler.type, "uniform");
                Assert.AreEqual(lstm.layer_param.lstm_attention_param.weight_filler.min, -0.01);
                Assert.AreEqual(lstm.layer_param.lstm_attention_param.weight_filler.max, 0.01);
                Assert.AreEqual(lstm.layer_param.lstm_attention_param.bias_filler.type, "constant");
                Assert.AreEqual(lstm.layer_param.lstm_attention_param.bias_filler.value, 2);
                Assert.AreEqual(lstm.layer_param.lstm_attention_param.clipping_threshold, 2.44);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter pLstm = new LayerParameter(LayerParameter.LayerType.LSTM);
            pLstm.recurrent_param.engine = EngineParameter.Engine.CAFFE;
            pLstm.recurrent_param.dropout_ratio = 0;
            pLstm.recurrent_param.bias_filler = new FillerParameter("constant", 0);
            pLstm.recurrent_param.weight_filler = new FillerParameter("constant", 0.1);
            pLstm.recurrent_param.num_layers = 1;
            pLstm.recurrent_param.num_output = 16;

            Layer<T> lstmLayer = Layer<T>.Create(m_cuda, m_log, pLstm, m_evtCancel);
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
                p.lstm_attention_param.num_output = 16;
                p.lstm_attention_param.bias_filler = new FillerParameter("constant", 0.0);
                p.lstm_attention_param.weight_filler = new FillerParameter("constant", 0.1);

                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                Bottom.Reshape(4, 3, 2, 1);
                double[] rgData = convert(Bottom.mutable_cpu_data);
                // t = 0
                rgData[0] = 0.01; rgData[1] = 0.02;  // b0
                rgData[2] = 0.11; rgData[3] = 0.12;  // b1
                rgData[4] = 0.21; rgData[5] = 0.22;  // b2
                                                     // t = 1
                rgData[6] = 1.01; rgData[7] = 1.02;  // b0
                rgData[8] = 1.11; rgData[9] = 1.12;  // b1
                rgData[10] = 1.21; rgData[11] = 1.22; // b2
                                                      // t = 2
                rgData[12] = 2.01; rgData[13] = 2.02; // b0
                rgData[14] = 2.11; rgData[15] = 2.12; // b1
                rgData[16] = 2.21; rgData[17] = 2.22; // b2
                                                      // t = 3
                rgData[18] = 3.01; rgData[19] = 3.02; // b0
                rgData[20] = 3.11; rgData[21] = 3.12; // b1
                rgData[22] = 3.21; rgData[23] = 3.22; // b2
                Bottom.mutable_cpu_data = convert(rgData);

                Bottom2.Reshape(new List<int>() { 4, 3 });
                double[] rgClip = convert(Bottom2.mutable_cpu_data);
                // t = 0
                rgClip[0] = 0;
                rgClip[1] = 0;
                rgClip[2] = 0;
                // t = 1
                rgClip[3] = 1;
                rgClip[4] = 1;
                rgClip[5] = 0;
                // t = 2
                rgClip[6] = 1;
                rgClip[7] = 0;
                rgClip[8] = 1;
                // t = 3
                rgClip[9] = 0;
                rgClip[10] = 1;
                rgClip[11] = 1;
                Bottom2.mutable_cpu_data = convert(rgData);

                BottomVec.Add(m_blob_bottom2);

                lstmLayer.Setup(BottomVec, TopVec);
                lstmLayer.Forward(BottomVec, TopVec);

                double[] rgExpected = convert(m_blob_top.mutable_cpu_data);
                List<int> rgExpectedShape = Utility.Clone<int>(m_blob_top.shape());
                m_blob_top.Reshape(1, 1, 1, 1);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(rgExpectedShape[0], m_blob_top.num, "The num shape is not as expected!");
                m_log.CHECK_EQ(rgExpectedShape[1], m_blob_top.channels, "The channels shape is not as expected!");
                m_log.CHECK_EQ(rgExpectedShape[2], m_blob_top.height, "The height shape is not as expected!");
                if (rgExpectedShape.Count > 3)
                    m_log.CHECK_EQ(rgExpectedShape[3], m_blob_top.width, "The width shape is not as expected!");

                double[] rgActual = convert(m_blob_top.mutable_cpu_data);

                m_log.CHECK_EQ(rgExpected.Length, rgActual.Length, "The data lengths differ!");

                for (int i = 0; i < rgActual.Length; i++)
                {
                    double dfExpected = rgExpected[i];
                    double dfActual = rgActual[i];

                    m_log.EXPECT_NEAR(dfExpected, dfActual, 0.0000001, "The data values at " + i.ToString() + " do not match!");
                }
            }
            finally
            {
                lstmLayer.Dispose();

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradientDefault()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 5;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant");
            p.lstm_attention_param.bias_filler.value = 0;

            BottomVec.Clear();
            BottomVec.Add(Bottom);

            // 12 timesteps, 1 batches, 2 inputs
            Bottom.Reshape(12, 1, 2, 1);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientBatchDefault()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 5;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant");
            p.lstm_attention_param.bias_filler.value = 0;

            // 4 timesteps, 3 batches, 2 inputs
            Bottom.Reshape(4, 3, 2, 1);

            BottomVec.Clear();
            BottomVec.Add(Bottom);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientClipMask()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 4;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant");
            p.lstm_attention_param.bias_filler.value = 0;

            double[] rgData = convert(Bottom2.mutable_cpu_data);
            rgData[0]  = 0;
            rgData[1]  = 1;
            rgData[2]  = 1;
            rgData[3]  = 0;
            rgData[4]  = 1;
            rgData[5]  = 1;
            rgData[6]  = 1;
            rgData[7]  = 0;
            rgData[8]  = 1;          
            rgData[9]  = 1;
            rgData[10] = 0;
            rgData[11] = 1;

            // 12 timesteps, 1 batches, 2 inputs
            Bottom.Reshape(12, 1, 2, 1);
            Bottom2.Reshape(new List<int>() { 12, 1 });
            Bottom2.mutable_cpu_data = convert(rgData);

            BottomVec.Clear();
            BottomVec.Add(Bottom);
            BottomVec.Add(Bottom2);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientBatchClipMask()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 4;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant");
            p.lstm_attention_param.bias_filler.value = 0;

            // 4 timesteps, 3 batches, 2 inputs
            Bottom.Reshape(4, 3, 2, 1);
            Bottom2.Reshape(new List<int>() { 4, 3 });
            double[] rgData = convert(Bottom2.mutable_cpu_data);
            // t = 0
            rgData[0] = 0;
            rgData[1] = 0;
            rgData[2] = 0;
            // t = 1
            rgData[3] = 1;
            rgData[4] = 1;
            rgData[5] = 0;
            // t = 2
            rgData[6] = 1;
            rgData[7] = 0;
            rgData[8] = 1;
            // t = 3
            rgData[9]  = 0;
            rgData[10] = 1;
            rgData[11] = 1;
            Bottom2.mutable_cpu_data = convert(rgData);

            BottomVec.Clear();
            BottomVec.Add(Bottom);
            BottomVec.Add(Bottom2);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientAttention()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 1;
            p.lstm_attention_param.num_output_ip = 0;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant", 0);
            p.lstm_attention_param.enable_attention = true;

            BottomVec.Clear();
            // Decoder blobs
            Fill(BottomVec, m_blob_bottom, m_blob_bottom2, 1, 1, 1, false);
            // Encoder blobs
            Fill(BottomVec, m_blobEncoding, m_blobEncodingClip, 1, 1, 1, true);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientAttention2()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM_ATTENTION);
            p.lstm_attention_param.num_output = 1;
            p.lstm_attention_param.num_output_ip = 0;
            p.lstm_attention_param.weight_filler = new FillerParameter("uniform");
            p.lstm_attention_param.weight_filler.min = -0.01;
            p.lstm_attention_param.weight_filler.max = 0.01;
            p.lstm_attention_param.bias_filler = new FillerParameter("constant", 0);
            p.lstm_attention_param.enable_attention = true;

            BottomVec.Clear();
            // Decoder blobs
            Fill(BottomVec, m_blob_bottom, m_blob_bottom2, 1, 1, 1, false);
            // Encoder blobs
            Fill(BottomVec, m_blobEncoding, m_blobEncodingClip, 3, 1, 1, true);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestTraining(int nTotalDataLength, int nNumOutput, int nBatch, bool bShortModel, int nMaxIter)
        {
            if (m_strParams != null)
            {
                string[] rgstr = m_strParams.Split(',');

                if (rgstr.Length > 1)
                    nMaxIter = int.Parse(rgstr[1].Trim(','));

                if (rgstr.Length > 2)
                    nNumOutput = int.Parse(rgstr[2].Trim(','));

                if (rgstr.Length > 3)
                    nBatch = int.Parse(rgstr[3].Trim(','));
            }

            int nDeviceID = m_cuda.GetDeviceID();

            string strSolver = getSolver(nMaxIter);
            string strModel = (bShortModel) ? getModelShort("LSTMAttention", nNumOutput, 320, nBatch) : getModel("LSTMAttention", nNumOutput, 320, nBatch);
            SolverParameter solver_param = SolverParameter.FromProto(RawProto.Parse(strSolver));
            solver_param.net_param = NetParameter.FromProto(RawProto.Parse(strModel));
            strModel = solver_param.net_param.ToProto("root").ToString();
            solver_param.device_id = nDeviceID;

            // Set device id
            m_log.WriteLine("Running test on " + m_cuda.GetDeviceName(solver_param.device_id) + " with ID = " + solver_param.device_id.ToString());
            m_log.WriteLine("type: " + typeof(T).ToString() + " num_output: " + nNumOutput.ToString() + " batch: " + nBatch.ToString() + " MaxIter: " + nMaxIter.ToString());
            m_cuda.SetDeviceID(solver_param.device_id);

            Solver<T> solver = Solver<T>.Create(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, null);
            Net<T> train_net = solver.net;
            Net<T> test_net = new Net<T>(m_cuda, m_log, solver_param.net_param, m_evtCancel, null, Phase.TEST);

            Assert.AreEqual(true, train_net.has_blob("data"));
            Assert.AreEqual(true, train_net.has_blob("clip"));
            Assert.AreEqual(true, train_net.has_blob("label"));
            Assert.AreEqual(true, test_net.has_blob("data"));
            Assert.AreEqual(true, test_net.has_blob("clip"));

            Blob<T> train_data_blob = train_net.blob_by_name("data");
            Blob<T> train_label_blob = train_net.blob_by_name("label");
            Blob<T> train_clip_blob = train_net.blob_by_name("clip");

            Blob<T> test_data_blob = test_net.blob_by_name("data");
            Blob<T> test_clip_blob = test_net.blob_by_name("clip");

            int seq_length = train_data_blob.shape(0);
            Assert.AreEqual(0, nTotalDataLength % seq_length);

            // Initialize bias for the forget gate to 5 as described in clockwork RNN paper
            for (int i = 0; i < train_net.layers.Count; i++)
            {
                if (train_net.layers[i].type != LayerParameter.LayerType.LSTM_ATTENTION)
                    continue;

                int h = (int)train_net.layers[i].layer_param.lstm_attention_param.num_output;
                Blob<T> bias = train_net.layers[i].blobs[2];

                double[] rgBias = convert(bias.mutable_cpu_data);

                for (int j = 0; j < h; j++)
                {
                    rgBias[h + j] = 5.0;
                }

                bias.mutable_cpu_data = convert(rgBias);

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            List<int> sequence_shape = new List<int>() { nTotalDataLength };
            List<int> data_shape = new List<int>() { seq_length };
            Blob<T> sequence = new Blob<T>(m_cuda, m_log, sequence_shape);

            // Construct the data.
            double dfMean = 0;
            double dfMaxAbs = 0;

            for (int i = 0; i < nTotalDataLength; i++)
            {
                double dfVal = f_x(i * 0.01);
                dfMaxAbs = Math.Max(dfMaxAbs, Math.Abs(dfVal));

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            for (int i = 0; i < nTotalDataLength; i++)
            {
                dfMean += f_x(i * 0.01) / dfMaxAbs;

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            dfMean /= nTotalDataLength;

            double[] rgSequence = convert(sequence.mutable_cpu_data);

            for (int i = 0; i < nTotalDataLength; i++)
            {
                rgSequence[i] = f_x(i * 0.01) / dfMaxAbs - dfMean;

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            sequence.mutable_cpu_data = convert(rgSequence);

            // Training
            train_clip_blob.SetData(1.0);

            int iter = 0;

            Stopwatch swStatus = new Stopwatch();
            Stopwatch sw1 = new Stopwatch();

            swStatus.Start();

            TestingProgressSet progress = new TestingProgressSet();
            double dfTotalTime = 0;

            while (iter < solver_param.max_iter)
            {
                int seq_idx = iter % (nTotalDataLength / seq_length);

                double dfVal = (seq_idx > 0) ? 1 : 0;
                train_clip_blob.SetData(dfVal, 0);

                m_cuda.copy(seq_length, sequence.gpu_data, train_label_blob.mutable_gpu_data, sequence.offset(seq_idx * seq_length));

                sw1.Start();

                solver.Step(1);

                sw1.Stop();
                dfTotalTime += sw1.Elapsed.TotalMilliseconds;
                sw1.Reset();

                iter++;

                if (swStatus.Elapsed.TotalMilliseconds > 2000)
                {
                    m_log.Progress = (double)iter / (double)solver_param.max_iter;
                    m_log.WriteLine("iteration = " + iter.ToString() + "  (" + m_log.Progress.ToString("P") + ")  ave solver time = " + (dfTotalTime / iter).ToString() + " ms.");
                    swStatus.Stop();
                    swStatus.Reset();
                    swStatus.Start();

                    progress.SetProgress(m_log.Progress);
                }

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            m_log.WriteLine("Solving completed.");

            // Output test
            test_net.ShareTrainedLayersWith(train_net);
            List<int> shape = new List<int>() { 1, 1 };
            test_data_blob.Reshape(shape);
            test_clip_blob.Reshape(shape);
            test_net.Reshape();

            string strPath = "c:\\temp\\lstm";
            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strFileName = strPath + "\\lstm_noattendet_noinput_test_results_num_out_" + nNumOutput.ToString() + "_batch_" + nBatch.ToString() + "_maxiter_" + nMaxIter.ToString() + ".csv";

            if (File.Exists(strFileName))
                File.Delete(strFileName);

            swStatus.Stop();
            swStatus.Reset();
            swStatus.Start();

            using (StreamWriter sw = new StreamWriter(strFileName))
            {
                for (int i = 0; i < nTotalDataLength; i++)
                {
                    double dfLoss;

                    test_clip_blob.SetData((i > 0) ? 1 : 0, 0);
                    BlobCollection<T> pred = test_net.Forward(out dfLoss);
                    m_log.CHECK_EQ(pred.Count, 1, "There should only be one result blob.");
                    m_log.CHECK_EQ(pred[0].count(), 1, "The result blob should only have one element.");
                    sw.WriteLine(sequence.GetData(i).ToString() + ", " + pred[0].GetData(0).ToString());

                    if (swStatus.Elapsed.TotalMilliseconds > 2000)
                    {
                        m_log.Progress = (double)i / nTotalDataLength;
                        m_log.WriteLine("Testing iteration " + i.ToString() + " (" + m_log.Progress.ToString("P") + ")");
                        swStatus.Stop();
                        swStatus.Reset();
                        swStatus.Start();
                    }

                    if (m_evtCancel.WaitOne(0))
                    {
                        m_log.WriteLine("Aborted.");
                        return;
                    }
                }
            }
        }

        public void TestTraining2(int nTotalDataLength, int nNumOutput, int nTimeSteps, int nBatch, bool bShortModel, int nMaxIter)
        {
            if (m_strParams != null)
            {
                string[] rgstr = m_strParams.Split(',');

                if (rgstr.Length > 1)
                    nMaxIter = int.Parse(rgstr[1].Trim(','));

                if (rgstr.Length > 2)
                    nNumOutput = int.Parse(rgstr[2].Trim(','));

                if (rgstr.Length > 3)
                    nBatch = int.Parse(rgstr[3].Trim(','));
            }

            int nDeviceID = m_cuda.GetDeviceID();

            string strLstmLayer = "LSTMAttention";
            string strSolver = getSolver(nMaxIter);
            string strModel = (bShortModel) ? getModelShort(strLstmLayer, nNumOutput, nTimeSteps, nBatch) : getModel(strLstmLayer, nNumOutput, nTimeSteps, nBatch);
            SolverParameter solver_param = SolverParameter.FromProto(RawProto.Parse(strSolver));
            solver_param.net_param = NetParameter.FromProto(RawProto.Parse(strModel));
            strModel = solver_param.net_param.ToProto("root").ToString();
            solver_param.device_id = nDeviceID;

            // Set device id
            m_log.WriteLine("Running test on " + m_cuda.GetDeviceName(solver_param.device_id) + " with ID = " + solver_param.device_id.ToString());
            m_log.WriteLine("type: " + typeof(T).ToString() + " num_output: " + nNumOutput.ToString() + " batch: " + nBatch.ToString() + " MaxIter: " + nMaxIter.ToString());
            m_cuda.SetDeviceID(solver_param.device_id);

            Solver<T> solver = Solver<T>.Create(m_cuda, m_log, solver_param, m_evtCancel, null, null, null, null);
            Net<T> train_net = solver.net;
            Net<T> test_net = new Net<T>(m_cuda, m_log, solver_param.net_param, m_evtCancel, null, Phase.TEST);

            Assert.AreEqual(true, train_net.has_blob("data"));
            Assert.AreEqual(true, train_net.has_blob("clip"));
            Assert.AreEqual(true, train_net.has_blob("label"));
            Assert.AreEqual(true, test_net.has_blob("data"));
            Assert.AreEqual(true, test_net.has_blob("clip"));

            Blob<T> train_data_blob = train_net.blob_by_name("data");
            Blob<T> train_label_blob = train_net.blob_by_name("label");
            Blob<T> train_clip_blob = train_net.blob_by_name("clip");

            Blob<T> test_data_blob = test_net.blob_by_name("data");
            Blob<T> test_clip_blob = test_net.blob_by_name("clip");

            int seq_length = train_data_blob.shape(0);
            Assert.AreEqual(0, nTotalDataLength % seq_length);

            // Initialize bias for the forget gate to 5 as described in clockwork RNN paper
            for (int i = 0; i < train_net.layers.Count; i++)
            {
                if (train_net.layers[i].type != LayerParameter.LayerType.LSTM_ATTENTION)
                    continue;

                int h = (int)train_net.layers[i].layer_param.lstm_attention_param.num_output;
                Blob<T> bias = train_net.layers[i].blobs[2];

                double[] rgBias = convert(bias.mutable_cpu_data);

                for (int j = 0; j < h; j++)
                {
                    rgBias[h + j] = 5.0;
                }

                bias.mutable_cpu_data = convert(rgBias);

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            // Construct the data.
            float[] rgSequence = create_sequence(nTotalDataLength);

            Stopwatch swStatus = new Stopwatch();
            Stopwatch sw1 = new Stopwatch();

            swStatus.Start();

            TestingProgressSet progress = new TestingProgressSet();
            double dfTotalTime = 0;
            float fIxInput = 2;

            int iter = 0;
            while (iter < solver_param.max_iter)
            {
                fIxInput = fill_data(fIxInput, rgSequence, iter, train_data_blob, train_label_blob, train_clip_blob);

                sw1.Start();

                solver.Step(1);

                sw1.Stop();
                dfTotalTime += sw1.Elapsed.TotalMilliseconds;
                sw1.Reset();

                iter++;

                if (swStatus.Elapsed.TotalMilliseconds > 2000)
                {
                    m_log.Progress = (double)iter / (double)solver_param.max_iter;
                    m_log.WriteLine("iteration = " + iter.ToString() + "  (" + m_log.Progress.ToString("P") + ")  ave solver time = " + (dfTotalTime / iter).ToString() + " ms.");
                    swStatus.Stop();
                    swStatus.Reset();
                    swStatus.Start();

                    progress.SetProgress(m_log.Progress);
                }

                if (m_evtCancel.WaitOne(0))
                {
                    m_log.WriteLine("Aborted.");
                    return;
                }
            }

            m_log.WriteLine("Solving completed.");

            // Output test
            test_net.ShareTrainedLayersWith(train_net);
            List<int> shape = new List<int>() { 1, 1 };
            test_data_blob.Reshape(shape);
            test_clip_blob.Reshape(shape);
            test_net.Reshape();

            string strPath = "c:\\temp\\lstm";
            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strFileName = strPath + "\\lstm_noattendet_input_test_results_num_out_" + nNumOutput.ToString() + "_batch_" + nBatch.ToString() + "_maxiter_" + nMaxIter.ToString() + ".csv";

            if (File.Exists(strFileName))
                File.Delete(strFileName);

            swStatus.Stop();
            swStatus.Reset();
            swStatus.Start();

            test_data_blob.SetData(2, 0);

            using (StreamWriter sw = new StreamWriter(strFileName))
            {
                for (int i = 0; i < nTotalDataLength; i++)
                {
                    double dfLoss;

                    test_clip_blob.SetData((i > 0) ? 1 : 0, 0);
                    BlobCollection<T> pred = test_net.Forward(out dfLoss);
                    m_log.CHECK_EQ(pred.Count, 1, "There should only be one result blob.");
                    m_log.CHECK_EQ(pred[0].count(), 1, "The result blob should only have one element.");
                    float fPred = convertF(pred[0].GetData(0));
                    sw.WriteLine(rgSequence[i].ToString() + ", " + fPred.ToString());

                    test_data_blob.SetData(fPred, 0);

                    if (swStatus.Elapsed.TotalMilliseconds > 2000)
                    {
                        m_log.Progress = (double)i / nTotalDataLength;
                        m_log.WriteLine("Testing iteration " + i.ToString() + " (" + m_log.Progress.ToString("P") + ")");
                        swStatus.Stop();
                        swStatus.Reset();
                        swStatus.Start();
                    }

                    if (m_evtCancel.WaitOne(0))
                    {
                        m_log.WriteLine("Aborted.");
                        return;
                    }
                }
            }
        }

        private float fill_data(float fIxInput, float[] rgSequence, int iter, Blob<T> data, Blob<T> label, Blob<T> clip)
        {
            int nIdx = iter % rgSequence.Length;
            int nT = data.num;
            int nN = data.channels;

            float[] rgData = new float[data.count()];
            float[] rgLabel = new float[label.count()];
            float[] rgClip = new float[clip.count()];
            float fIxTarget;
            float fIxClip = (nIdx == 0) ? 0 : 1;

            if (nIdx == 0)
                fIxInput = 2;

            for (int i = 0; i < nT; i++)
            {
                rgData[i] = fIxInput;
                fIxTarget = rgSequence[i];
                
                rgLabel[i] = fIxTarget;
                fIxInput = fIxTarget;

                rgClip[i] = fIxClip;
                fIxClip = 1;
            }

            data.mutable_cpu_data = convert(rgData);
            label.mutable_cpu_data = convert(rgLabel);
            clip.mutable_cpu_data = convert(rgClip);

            return rgSequence[1];
        }

        private float[] create_sequence(int nCount)
        {
            float[] rgData = new float[nCount];
            float fMax = -float.MaxValue;
            float fTotal = 0;

            for (int i = 0; i < nCount; i++)
            {
                float fVal = (float)f_x(i * 0.01);
                fMax = Math.Max(fMax, Math.Abs(fVal));
                fTotal += fVal;
                rgData[i] = fVal;
            }

            float fMean = fTotal / nCount;

            return rgData.Select(p => p / fMax - fMean).ToArray();
        }

        private double f_x(double dfT)
        {
            return 0.5 * Math.Sin(2 * dfT) - 0.05 * Math.Cos(17 * dfT + 0.8) + 0.05 * Math.Sin(25 * dfT + 10) - 0.02 * Math.Cos(45 * dfT + 0.3);
        }

        private string getSolver(int nMaxIter = 100000)
        {
            string strSolver = "base_lr: 0.00005                   " +
                               "momentum: 0.95                     " +
                               "lr_policy: \"fixed\"               " +
                               "display: 200                       " +
                               "max_iter: " + nMaxIter.ToString() + " " +
                               "solver_mode: GPU                   " +
                               "average_loss: 200";        

            return strSolver;
        }

        private string getModel(string strLstmLayer, int nNumOutput, int nTimeSteps, int nBatch)
        {
            string strParamName = (strLstmLayer == "LSTMAttention") ? "lstm_attention_param" : "lstm_simple_param";

            string strModel = "name: \"LSTM\"                                     " +
                                "input: \"data\"                                  " +
                                "input_shape { dim: " + nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 } " +
                                "input: \"clip\"                                  " +
                                "input_shape { dim: " + nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 } " +
                                "input: \"label\"                                 " +
                                "input_shape { dim: " + nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 } " +
                                "layer {                                          " +
                                "  name: \"Silence\"                              " +
                                "  type: \"Silence\"                              " +
                                "  bottom: \"label\"                              " +
                                "  include: { phase: TEST }                       " +
                                "}                                                " +
                                "layer {                                          " +
                                "  name: \"lstm1\"                                " +
                                "  type: \"" + strLstmLayer + "\"                 " +
                                "  bottom: \"data\"                               " +
                                "  bottom: \"clip\"                               " +
                                "  top: \"lstm1\"                                 " +
                                "                                                 " +
                                "  " + strParamName + " {                         " +
                                "    num_output: " + nNumOutput.ToString() + " ";

            strModel += "    clipping_threshold: 0.1                      " +
                                "    weight_filler {                              " +
                                "      type: \"gaussian\"                         " +
                                "      std: 0.1                                   " +
                                "    }                                            " +
                                "    bias_filler {                                " +
                                "      type: \"constant\"                         " +
                                "    }                                            " +
                                "  }                                              " +
                                "}                                                " +
                                "layer {                                          " +
                                "  name: \"lstm2\"                                " +
                                "  type: \"" + strLstmLayer + "\"                 " +
                                "  bottom: \"lstm1\"                              " +
                                "  bottom: \"clip\"                               " +
                                "  top: \"lstm2\"                                 " +
                                "                                                 " +
                                "  " + strParamName + " {                         " +
                                "    num_output: " + nNumOutput.ToString() + " ";

            strModel += "    clipping_threshold: 0.1                      " +
                                "    weight_filler {                              " +
                                "      type: \"gaussian\"                         " +
                                "      std: 0.1                                   " +
                                "    }                                            " +
                                "    bias_filler {                                " +
                                "      type: \"constant\"                         " +
                                "    }                                            " +
                                "  }                                              " +
                                "}                                                " +
                                "layer {                                          " +
                                "  name: \"lstm3\"                                " +
                                "  type: \"" + strLstmLayer + "\"                 " +
                                "  bottom: \"lstm2\"                              " +
                                "  bottom: \"clip\"                               " +
                                "  top: \"lstm3\"                                 " +
                                "                                                 " +
                                "  " + strParamName + " {                         " +
                                "    num_output: " + nNumOutput.ToString() + " ";

            strModel +=         "    clipping_threshold: 0.1                      " +
                                "    weight_filler {                              " +
                                "      type: \"gaussian\"                         " +
                                "      std: 0.1                                   " +
                                "    }                                            " +
                                "    bias_filler {                                " +
                                "      type: \"constant\"                         " +
                                "    }                                            " +
                                "  }                                              " +
                                "}                                                " +
                                "layer {                                          " +
                                "  name: \"ip1\"                                  " +
                                "  type: \"InnerProduct\"                         " +
                                "  bottom: \"lstm3\"                              " +
                                "  top: \"ip1\"                                   " +
                                "                                                 " +
                                "  inner_product_param {                          " +
                                "    num_output: 1                                " +
                                "    weight_filler {                              " +
                                "      type: \"gaussian\"                         " +
                                "      std: 0.1                                   " +
                                "    }                                            " +
                                "    bias_filler {                                " +
                                "      type: \"constant\"                         " +
                                "    }                                            " +
                                "  }                                              " +
                                "}                                                " +
                                "layer {                                          " +
                                "  name: \"loss\"                                 " +
                                "  type: \"EuclideanLoss\"                        " +
                                "  bottom: \"ip1\"                                " +
                                "  bottom: \"label\"                              " +
                                "  top: \"loss\"                                  " +
                                "  include: { phase: TRAIN }                      " +
                                "}                                                ";
            return strModel;
        }

        private string getModelShort(string strLstmLayer, int nNumOutput, int nTimesteps, int nBatch)
        {
            string strParamName = (strLstmLayer == "LSTMAttention") ? "lstm_attention_param" : "lstm_simple_param";

            string strModel = "name: \"LSTM\"                                " +
                                "input: \"data\"                             " +
                                "input_shape { dim: " + nTimesteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 }             " +
                                "input: \"clip\"                             " +
                                "input_shape { dim: " + nTimesteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 }             " +
                                "input: \"label\"                            " +
                                "input_shape { dim: " + nTimesteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 }             " +
                                "layer {                                     " +
                                "  name: \"Silence\"                         " +
                                "  type: \"Silence\"                         " +
                                "  bottom: \"label\"                         " +
                                "  include: { phase: TEST }                  " +
                                "}                                           " +
                                "layer {                                     " +
                                "  name: \"lstm1\"                           " +
                                "  type: \"" + strLstmLayer + "\"                " +
                                "  bottom: \"data\"                          " +
                                "  bottom: \"clip\"                          " +
                                "  top: \"lstm1\"                            " +
                                "                                            " +
                                "  " + strParamName + " {                    " +
                                "    num_output: " + nNumOutput.ToString() + " ";

            if (nBatch > 1)
                strModel += "   batch_size: " + nBatch.ToString() + " ";

            strModel +=         "    clipping_threshold: 0.1               " +
                                "    weight_filler {                       " +
                                "      type: \"gaussian\"                    " +
                                "      std: 0.1                            " +
                                "    }                                     " +
                                "    bias_filler {                         " +
                                "      type: \"constant\"                    " +
                                "    }                                     " +
                                "  }                                       " +
                                "}                                         " +
                                "layer {                                   " +
                                "  name: \"ip1\"                             " +
                                "  type: \"InnerProduct\"                    " +
                                "  bottom: \"lstm1\"                         " +
                                "  top: \"ip1\"                              " +
                                "                                          " +
                                "  inner_product_param {                   " +
                                "    num_output: 1                         " +
                                "    weight_filler {                       " +
                                "      type: \"gaussian\"                    " +
                                "      std: 0.1                            " +
                                "    }                                     " +
                                "    bias_filler {                         " +
                                "      type: \"constant\"                    " +
                                "    }                                     " +
                                "  }                                       " +
                                "}                                         " +
                                "layer {                                   " +
                                "  name: \"loss\"                            " +
                                "  type: \"EuclideanLoss\"                   " +
                                "  bottom: \"ip1\"                           " +
                                "  bottom: \"label\"                         " +
                                "  top: \"loss\"                             " +
                                "  include: { phase: TRAIN }               " +
                                "}                                         ";

            return strModel;
        }
    }
}
