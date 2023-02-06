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
using System.IO;
using MyCaffe.param.gpt;
using System.Net;
using System.Threading;
using System.IO.Compression;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPositionalEncodingLayer
    {
        [TestMethod]
        public void TestForward()
        {
            PositionalEncodingLayerTest test = new PositionalEncodingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPositionalEncodingLayerTest t in test.Tests)
                {
                    t.TestForward(3, 8);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward()
        {
            PositionalEncodingLayerTest test = new PositionalEncodingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPositionalEncodingLayerTest t in test.Tests)
                {
                    t.TestBackward(3, 8);
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
            PositionalEncodingLayerTest test = new PositionalEncodingLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPositionalEncodingLayerTest t in test.Tests)
                {
                    t.TestGradient(3, 8);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IPositionalEncodingLayerTest : ITest
    {
        void TestForward(int nBatch, int nHeads);
        void TestBackward(int nBatch, int nHeads);
        void TestGradient(int nBatch, int nHeads);
    }

    class PositionalEncodingLayerTest : TestBase
    {
        public PositionalEncodingLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Positional Encoding Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PositionalEncodingLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new PositionalEncodingLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class PositionalEncodingLayerTest2<T> : TestEx<T>, IPositionalEncodingLayerTest
    {
        Blob<T> m_blobY;
        Blob<T> m_blobQ;

        public PositionalEncodingLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobY = new Blob<T>(m_cuda, m_log);
            m_blobQ = new Blob<T>(m_cuda, m_log);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            dispose(ref m_blobY);
            dispose(ref m_blobQ);
            base.dispose();
        }

        private string loadTestData1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\pos\\";
            string strFileName = "_posenc_test.zip";
            string strTestPath = "test";
            string strTestFile = "pos.output.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }
        
        public void TestForward(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            p.positional_encoder_param.embed = 512;
            p.positional_encoder_param.block_size = 200;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.POSITIONAL_ENCODER, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");
                
                BottomVec.Clear();
                BottomVec.Add(m_blobQ);

                layer.Setup(BottomVec, TopVec);

                // Check the pos embed matrix.
                blobVal.LoadFromNumpy(strTestDataPath + "pos.3_pos_enc.npy");
                for (int i = 0; i < nBatch; i++)
                {
                    verify(layer.internal_blobs[0], blobVal, i, false, 1e-10);
                }

                layer.Forward(BottomVec, TopVec);

                m_blobY.LoadFromNumpy(strTestDataPath + "pos.output.npy");

                // Now, check values
                double dfErr = 1e-5;
                verify(TopVec[0], m_blobY, false, dfErr);
            }
            finally
            {
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        public void TestBackward(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            p.positional_encoder_param.embed = 512;
            p.positional_encoder_param.block_size = 200;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.POSITIONAL_ENCODER, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_pos.output.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                m_blobQ.LoadFromNumpy(strTestDataPath + "grad_pos.1_x.npy", true);

                // Now, check values
                verify(BottomVec[0], m_blobQ, true);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            p.positional_encoder_param.embed = 512;
            p.positional_encoder_param.block_size = 200;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.POSITIONAL_ENCODER, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.0001);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 200);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
