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

namespace MyCaffe.test
{
    [TestClass]
    public class TestCropLayer
    {
        [TestMethod]
        public void TestSetupShapeAll()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestSetupShapeAll();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupShapeDefault()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestSetupShapeDefault();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupShapeNegativeIndexing()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TetsSetupShapeNegativeIndexing();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDimensionsCheck()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestDimensionsCheck();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCropAll()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCropAll();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCropAllOffset()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCropAllOffset();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCropHW()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCropHW();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCrop5D()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCrop5D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TetCropAllGradient()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCropAllGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TetCropHWGradient()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCropHWGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TetCrop5DGradient()
        {
            CropLayerTest test = new CropLayerTest();

            try
            {
                foreach (ICropLayerTest t in test.Tests)
                {
                    t.TestCrop5DGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ICropLayerTest : ITest
    {
        void TestSetupShapeAll();
        void TestSetupShapeDefault();
        void TetsSetupShapeNegativeIndexing();
        void TestDimensionsCheck();
        void TestCropAll();
        void TestCropAllOffset();
        void TestCropHW();
        void TestCrop5D();
        void TestCropAllGradient();
        void TestCropHWGradient();
        void TestCrop5DGradient();
    }

    class CropLayerTest : TestBase
    {
        public CropLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("BatchNorm Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new CropLayerTest<double>(strName, nDeviceID, engine);
            else
                return new CropLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class CropLayerTest<T> : TestEx<T>, ICropLayerTest
    {
        Blob<T> m_blobBottom1;

        public CropLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 4, 5, 4 }, nDeviceID)
        {
            m_engine = engine;
            m_blobBottom1 = new Blob<T>(m_cuda, m_log, 2, 3, 4, 2);

            FillerParameter fp = getFillerParam();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blobBottom1);

            BottomVec.Add(m_blobBottom1);
        }

        protected override void dispose()
        {
            m_blobBottom1.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestSetupShapeAll()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Crop all dimensions.
            p.crop_param.axis = 0;
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            for (int i=0; i<Top.num_axes; i++)
            {
                m_log.CHECK_EQ(BottomVec[1].shape(i), Top.shape(i), "The bottom[1].shape at " + i.ToString() + " doesn't match the top shape!");
            }
        }

        public void TestSetupShapeDefault()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Crop last two dimensions, axis is 2 by default.
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            for (int i = 0; i < Top.num_axes; i++)
            {
                if (i < 2)
                    m_log.CHECK_EQ(BottomVec[0].shape(i), Top.shape(i), "The bottom[0].shape at " + i.ToString() + " doesn't match the top shape!");
                else
                    m_log.CHECK_EQ(BottomVec[1].shape(i), Top.shape(i), "The bottom[1].shape at " + i.ToString() + " doesn't match the top shape!");
            }
        }

        public void TetsSetupShapeNegativeIndexing()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Crop last dimension by negative indexing.
            p.crop_param.axis = -1;
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            for (int i = 0; i < Top.num_axes; i++)
            {
                if (i < 3)
                    m_log.CHECK_EQ(BottomVec[0].shape(i), Top.shape(i), "The bottom[0].shape at " + i.ToString() + " doesn't match the top shape!");
                else
                    m_log.CHECK_EQ(BottomVec[1].shape(i), Top.shape(i), "The bottom[1].shape at " + i.ToString() + " doesn't match the top shape!");
            }
        }

        public void TestDimensionsCheck()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Reshape the blob to have incompatible sizes for uncropped dimensions:
            // the size blob has more channels than the data blob, but this is fine
            // since the channels dimension is not corpped in this configuration.
            m_blobBottom1.Reshape(2, 5, 4, 2);
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            for (int i = 0; i < Top.num_axes; i++)
            {
                if (i < 2)
                    m_log.CHECK_EQ(BottomVec[0].shape(i), Top.shape(i), "The bottom[0].shape at " + i.ToString() + " doesn't match the top shape!");
                else
                    m_log.CHECK_EQ(BottomVec[1].shape(i), Top.shape(i), "The bottom[1].shape at " + i.ToString() + " doesn't match the top shape!");
            }
        }


        public void TestCropAll()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Crop all dimensions.
            p.crop_param.axis = 0;
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h=0; h<Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            if (n < Top.shape(0) &&
                                c < Top.shape(1) &&
                                h < Top.shape(2) &&
                                w < Top.shape(3))
                            {
                                double dfTop = convert(Top.data_at(n, c, h, w));
                                double dfBtm = convert(Bottom.data_at(n, c, h, w));
                                m_log.CHECK_EQ(dfTop, dfBtm, "The top and bottom don't match at {" + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString() + "}");
                            }
                        }
                    }
                }
            }
        }

        public void TestCropAllOffset()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Crop all dimensions.
            p.crop_param.axis = 0;
            p.crop_param.offset.Add(0);
            p.crop_param.offset.Add(1);
            p.crop_param.offset.Add(1);
            p.crop_param.offset.Add(2);
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            if (n < Top.shape(0) &&
                                c < Top.shape(1) &&
                                h < Top.shape(2) &&
                                w < Top.shape(3))
                            {
                                double dfTop = convert(Top.data_at(n, c, h, w));
                                double dfBtm = convert(Bottom.data_at(n, c+1, h+1, w+2));
                                m_log.CHECK_EQ(dfTop, dfBtm, "The top and bottom don't match at {" + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString() + "}");
                            }
                        }
                    }
                }
            }
        }

        public void TestCropHW()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Crop HW dimensions.
            p.crop_param.axis = 2;
            p.crop_param.offset.Add(1);
            p.crop_param.offset.Add(2);
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Bottom.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Bottom.height; h++)
                    {
                        for (int w = 0; w < Bottom.width; w++)
                        {
                            if (n < Top.shape(0) &&
                                c < Top.shape(1) &&
                                h < Top.shape(2) &&
                                w < Top.shape(3))
                            {
                                double dfTop = convert(Top.data_at(n, c, h, w));
                                double dfBtm = convert(Bottom.data_at(n, c, h + 1, w + 2));
                                m_log.CHECK_EQ(dfTop, dfBtm, "The top and bottom don't match at {" + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString() + "}");
                            }
                        }
                    }
                }
            }
        }

        public void TestCrop5D()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            // Add dimension to each bottom for >4D check.
            List<int> rgBottom0Shape = Utility.Clone<int>(BottomVec[0].shape());
            List<int> rgBottom1Shape = Utility.Clone<int>(BottomVec[1].shape());
            rgBottom0Shape.Add(2);
            rgBottom1Shape.Add(1);
            BottomVec[0].Reshape(rgBottom0Shape);
            BottomVec[1].Reshape(rgBottom1Shape);

            FillerParameter fp = getFillerParam();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(BottomVec[0]);
            filler.Fill(BottomVec[1]);

            // Make the layer.
            p.crop_param.axis = 2;
            p.crop_param.offset.Add(1);
            p.crop_param.offset.Add(2);
            p.crop_param.offset.Add(0);
            CropLayer<T> layer = new layers.CropLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            List<int> rgBottomIdx = Utility.Create<int>(5, 0);
            List<int> rgTopIdx = Utility.Create<int>(5, 0);

            for (int n = 0; n < Bottom.shape(0); n++)
            {
                for (int c = 0; c < Bottom.shape(1); c++)
                {
                    for (int z = 0; z < Bottom.shape(2); z++)
                    {
                        for (int h = 0; h < Bottom.shape(3); h++)
                        {
                            for (int w = 0; w < Bottom.shape(4); w++)
                            {
                                if (n < Top.shape(0) &&
                                    c < Top.shape(1) &&
                                    z < Top.shape(2) &&
                                    h < Top.shape(3) &&
                                    w < Top.shape(4))
                                {
                                    rgBottomIdx[0] = n;
                                    rgBottomIdx[1] = c;
                                    rgBottomIdx[2] = z;
                                    rgBottomIdx[3] = h;
                                    rgBottomIdx[4] = w;

                                    rgTopIdx[0] = n;
                                    rgTopIdx[1] = c;
                                    rgTopIdx[2] = z + 1;
                                    rgTopIdx[3] = h + 2;
                                    rgTopIdx[4] = w;

                                    double dfTop = convert(Top.data_at(rgBottomIdx));
                                    double dfBtm = convert(Bottom.data_at(rgTopIdx));
                                    m_log.CHECK_EQ(dfTop, dfBtm, "The top and bottom don't match at {" + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString() + "}");
                                }
                            }
                        }
                    }
                }
            }
        }

        public void TestCropAllGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            p.crop_param.axis = 0;
            CropLayer<T> layer = new CropLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestCropHWGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            p.crop_param.axis = 2;
            p.crop_param.offset.Add(1);
            p.crop_param.offset.Add(2);
            CropLayer<T> layer = new CropLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestCrop5DGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CROP);
            p.crop_param.axis = 2;
            p.crop_param.offset.Add(1);
            p.crop_param.offset.Add(2);
            p.crop_param.offset.Add(0);
            CropLayer<T> layer = new CropLayer<T>(m_cuda, m_log, p);

            // Add dimension to each bottom for >4D check.
            List<int> rgBottom0Shape = Utility.Clone<int>(BottomVec[0].shape());
            List<int> rgBottom1Shape = Utility.Clone<int>(BottomVec[1].shape());
            rgBottom0Shape.Add(2);
            rgBottom1Shape.Add(1);
            BottomVec[0].Reshape(rgBottom0Shape);
            BottomVec[1].Reshape(rgBottom1Shape);

            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
