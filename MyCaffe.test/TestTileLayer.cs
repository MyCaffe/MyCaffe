using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestTileLayer
    {
        [TestMethod]
        public void TestTrivialSetup()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
                {
                    t.TestTrivialSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
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
        public void TestForwardNum()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
                {
                    t.TestForwardNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardChannels()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
                {
                    t.TestForwardChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrivialGradient()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
                {
                    t.TestTrivialGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientNum()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
                {
                    t.TestGradientNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientChannels()
        {
            TileLayerTest test = new TileLayerTest();

            try
            {
                foreach (ITileLayerTest t in test.Tests)
                {
                    t.TestGradientChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITileLayerTest : ITest
    {
        void TestTrivialSetup();
        void TestSetup();
        void TestForwardNum();
        void TestForwardChannels();
        void TestTrivialGradient();
        void TestGradientNum();
        void TestGradientChannels();
    }

    class TileLayerTest : TestBase
    {
        public TileLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Tile Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TileLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TileLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TileLayerTest<T> : TestEx<T>, ITileLayerTest
    {
        public TileLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("gaussian");
            p.mean = 0;
            p.std = 1.0;
            return p;
        }

        public void TestTrivialSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kNumTiles = 1;

            p.tile_param.tiles = kNumTiles;

            for (int i = 0; i < Bottom.num_axes; i++)
            {
                p.tile_param.axis = i;
                TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num_axes, Bottom.num_axes, "The top and bottom should have the same number of axes.");

                for (int j = 0; j < Bottom.num_axes; j++)
                {
                    m_log.CHECK_EQ(Top.shape(j), Bottom.shape(j), "The top and bottom should have the same shape at " + j.ToString());
                }

                layer.Dispose();
            }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kNumTiles = 3;

            p.tile_param.tiles = kNumTiles;

            for (int i = 0; i < Bottom.num_axes; i++)
            {
                p.tile_param.axis = i;
                TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num_axes, Bottom.num_axes, "The top and bottom should have the same number of axes.");

                for (int j = 0; j < Bottom.num_axes; j++)
                {
                    int nTopDim = ((i == j) ? kNumTiles : 1) * Bottom.shape(j);
                    m_log.CHECK_EQ(nTopDim, Top.shape(j), "The top.shape(" + j.ToString() + ") should have shape = " + nTopDim.ToString());
                }

                layer.Dispose();
            }
        }

        public void TestForwardNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kTileAxis = 0;
            int kNumTiles = 3;

            p.tile_param.axis = kTileAxis;
            p.tile_param.tiles = kNumTiles;

            TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Top.num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            int bottom_n = n % Bottom.num;
                            double dfBtm = convert(Bottom.data_at(bottom_n, c, h, w));
                            double dfTop = convert(Top.data_at(n, c, h, w));

                            m_log.CHECK_EQ(dfBtm, dfTop, "the top and bottom values should be the same.");
                        }
                    }
                }
            }
        }

        public void TestForwardChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kNumTiles = 3;

            p.tile_param.tiles = kNumTiles;

            TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Top.num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            int bottom_c = c % Bottom.channels;
                            double dfBtm = convert(Bottom.data_at(n, bottom_c, h, w));
                            double dfTop = convert(Top.data_at(n, c, h, w));

                            m_log.CHECK_EQ(dfBtm, dfTop, "the top and bottom values should be the same.");
                        }
                    }
                }
            }
        }

        public void TestTrivialGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kNumTiles = 3;

            p.tile_param.tiles = kNumTiles;

            TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kTileAxis = 0;
            int kNumTiles = 3;

            p.tile_param.axis = kTileAxis;
            p.tile_param.tiles = kNumTiles;

            TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TILE);
            int kTileAxis = 1;
            int kNumTiles = 3;

            p.tile_param.axis = kTileAxis;
            p.tile_param.tiles = kNumTiles;

            TileLayer<T> layer = new TileLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
