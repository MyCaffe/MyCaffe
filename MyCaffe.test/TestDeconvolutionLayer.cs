using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestDeDeconvolutionLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            DeconvolutionLayerTest test = new DeconvolutionLayerTest();

            try
            {
                foreach (IDeconvolutionLayerTest t in test.Tests)
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
        public void TestSimpleDeconvolution()
        {
            DeconvolutionLayerTest test = new DeconvolutionLayerTest();

            try
            {
                foreach (IDeconvolutionLayerTest t in test.Tests)
                {
                    t.TestSimpleDeconvolution();
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
            DeconvolutionLayerTest test = new DeconvolutionLayerTest();

            try
            {
                foreach (IDeconvolutionLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestNDAgainst2D()
        {
            DeconvolutionLayerTest test = new DeconvolutionLayerTest();

            try
            {
                foreach (IDeconvolutionLayerTest t in test.Tests)
                {
                    t.TestNDAgainst2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestGradient3D()
        {
            DeconvolutionLayerTest test = new DeconvolutionLayerTest();

            try
            {
                foreach (IDeconvolutionLayerTest t in test.Tests)
                {
                    t.TestGradient3D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    class DeconvolutionLayerTest : TestBase
    {
        public DeconvolutionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Deconvolution Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DeconvolutionLayerTest<double>(strName, nDeviceID, engine);
            else
                return new DeconvolutionLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IDeconvolutionLayerTest : ITest
    {
        void TestSetup();
        void TestSimpleDeconvolution();
        void TestGradient();
        void TestNDAgainst2D();
        void TestGradient3D();
    }

    class DeconvolutionLayerTest<T> : TestEx<T>, IDeconvolutionLayerTest
    {
        Blob<T> m_blob_bottom2;
        Blob<T> m_blob_top2;

        public DeconvolutionLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 4 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom2 = new Blob<T>(m_cuda, m_log, Bottom);
            m_blob_top2 = new Blob<T>(m_cuda, m_log);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom2);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public Blob<T> Bottom2
        {
            get { return m_blob_bottom2; }
        }

        public Blob<T> Top2
        {
            get { return m_blob_top2; }
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("gaussian");
            p.value = 1.0;
            return p;
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DECONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 4;
            BottomVec.Add(Bottom2);
            TopVec.Add(Top2);
            DeconvolutionLayer<T> layer = new DeconvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, Top.num, "The top num should equal 2.");
            m_log.CHECK_EQ(4, Top.channels, "The top channels should equal 4.");
            m_log.CHECK_EQ(13, Top.height, "The top height should equal 13.");
            m_log.CHECK_EQ(9, Top.width, "The top width should equal 9.");
            m_log.CHECK_EQ(2, Top2.num, "The top2 num should equal 2.");
            m_log.CHECK_EQ(4, Top2.channels, "The top2 channels should equal 4.");
            m_log.CHECK_EQ(13, Top2.height, "The top2 height should equal 13.");
            m_log.CHECK_EQ(9, Top2.width, "The top2 width should equal 9.");

            // setting group should not change the shape.
            p.convolution_param.num_output = 3;
            p.convolution_param.group = 3;
            layer = new DeconvolutionLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, Top.num, "The top num should equal 2.");
            m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
            m_log.CHECK_EQ(13, Top.height, "The top height should equal 13.");
            m_log.CHECK_EQ(9, Top.width, "The top width should equal 9.");
            m_log.CHECK_EQ(2, Top2.num, "The top2 num should equal 2.");
            m_log.CHECK_EQ(3, Top2.channels, "The top2 channels should equal 3.");
            m_log.CHECK_EQ(13, Top2.height, "The top2 height should equal 13.");
            m_log.CHECK_EQ(9, Top2.width, "The top2 width should equal 9.");
        }

        public void TestSimpleDeconvolution()
        {
            BottomVec.Add(Bottom2);
            TopVec.Add(Top2);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DECONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 4;
            p.convolution_param.weight_filler.type = "constant";
            p.convolution_param.weight_filler.value = 1.0;
            p.convolution_param.bias_filler.type = "constant";
            p.convolution_param.bias_filler.value = 0.1;
            DeconvolutionLayer<T> layer = new DeconvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            // constant-fill the bottom Blob<T>s.
            FillerParameter fp = new FillerParameter("constant");
            fp.value = 1.0;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);
            filler.Fill(Bottom2);

            layer.Forward(BottomVec, TopVec);

            // simple check that accumulation works with overlapping filters
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int n = 0; n < Top.num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            double dfExpected = 3.1;
                            bool h_overlap = h % 2 == 0 && h > 0 && h < Top.height - 1;
                            bool w_overlap = w % 2 == 0 && w > 0 && w < Top.width - 1;

                            if (h_overlap && w_overlap)
                                dfExpected += 9;
                            else if (h_overlap || w_overlap)
                                dfExpected += 3;

                            int nOffset = Top.offset(n, c, h, w);
                            double dfTop = rgTopData[nOffset];

                            m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                        }
                    }
                }
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DECONVOLUTION);
            p.convolution_param.engine = m_engine;
            BottomVec.Add(Bottom2);
            TopVec.Add(Top2);
            p.convolution_param.kernel_size.Add(2);
            p.convolution_param.stride.Add(1);
            p.convolution_param.num_output = 1;
            p.convolution_param.weight_filler.type = "gaussian";
            p.convolution_param.bias_filler.type = "gaussian";
            DeconvolutionLayer<T> layer = new DeconvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        public void TestNDAgainst2D()
        {
            int nKernelH = 11;
            int nKernelW = 13;
            List<int> rgBottomShape = new List<int>() { 15, 12, nKernelH * 2, nKernelW * 2 };
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
                filler.Fill(BottomVec[i]);
            }

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DECONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.num_output = 18;
            p.convolution_param.bias_term = false;
            p.convolution_param.group = 6;
            p.convolution_param.kernel_h = (uint)nKernelH;
            p.convolution_param.kernel_w = (uint)nKernelW;
            p.convolution_param.weight_filler.type = "gaussian";

            Blob<T> weights = new Blob<T>(m_cuda, m_log);
            Blob<T> top_diff = new Blob<T>(m_cuda, m_log);

            // Shape and fill weights and top_diff
            bool bCopyDiff;
            bool bReshape;
            {
                DeconvolutionLayer<T> layer = new DeconvolutionLayer<T>(m_cuda, m_log, p);
                layer.Setup(BottomVec, TopVec);
                top_diff.ReshapeLike(Top);
                filler.Fill(top_diff);
                m_log.CHECK_EQ(1, layer.blobs.Count, "There should only be 1 Blob<T> in layer.");
                bCopyDiff = false;
                bReshape = true;
                weights.CopyFrom(layer.blobs[0], bCopyDiff, bReshape);
            }

            List<bool> rgbPropagateDown = new List<bool>() { true };
            Blob<T> result_2d = new Blob<T>(m_cuda, m_log);
            Blob<T> backward_result_2d = new Blob<T>(m_cuda, m_log);
            Blob<T> backward_weight_result_2d = new Blob<T>(m_cuda, m_log);

            // Test with 2D im2col
            {
                Top.SetData(0);
                Bottom.SetDiff(0);
                weights.SetDiff(0);

                // Do Setup and Forward; save Forward result in result_2d.
                p.convolution_param.force_nd_im2col = false;
                DeconvolutionLayer<T> layer_2d = new DeconvolutionLayer<T>(m_cuda, m_log, p);
                layer_2d.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(1, layer_2d.blobs.Count, "The layer_2d should only have 1 Blob<T>.");

                bCopyDiff = false;
                bReshape = false;
                layer_2d.blobs[0].CopyFrom(weights, bCopyDiff, bReshape);
                layer_2d.Forward(BottomVec, TopVec);

                bCopyDiff = false;
                bReshape = true;
                result_2d.CopyFrom(Top, bCopyDiff, bReshape);

                // Copy pre-generated top diff into actual top diff;
                // do Backward and save result in backward_result_2d.
                m_log.CHECK(Utility.Compare<int>(Top.shape(), top_diff.shape()), "The top and top_diff should have the same shape!");
                m_cuda.copy(top_diff.count(), top_diff.gpu_data, Top.mutable_gpu_diff);
                layer_2d.Backward(TopVec, rgbPropagateDown, BottomVec);

                bCopyDiff = true;
                bReshape = true;
                backward_result_2d.CopyFrom(Bottom, bCopyDiff, bReshape);
                backward_weight_result_2d.CopyFrom(weights, bCopyDiff, bReshape);
            }

            Blob<T> result_nd = new Blob<T>(m_cuda, m_log);
            Blob<T> backward_result_nd = new Blob<T>(m_cuda, m_log);
            Blob<T> backward_weight_result_nd = new Blob<T>(m_cuda, m_log);

            // Test with ND im2col
            {
                Top.SetData(0);
                Bottom.SetDiff(0);
                weights.SetDiff(0);

                // Do Setup and Forward; save Forward result in result_nd.
                p.convolution_param.force_nd_im2col = true;
                DeconvolutionLayer<T> layer_nd = new DeconvolutionLayer<T>(m_cuda, m_log, p);
                layer_nd.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(1, layer_nd.blobs.Count, "The layer_nd should only have 1 Blob<T>.");

                bCopyDiff = false;
                bReshape = false;
                layer_nd.blobs[0].CopyFrom(weights, bCopyDiff, bReshape);
                layer_nd.Forward(BottomVec, TopVec);

                bCopyDiff = false;
                bReshape = true;
                result_nd.CopyFrom(Top, bCopyDiff, bReshape);

                // Copy pre-generated top diff into actual top diff;
                // do Backward and save result in backward_result_nd.
                m_log.CHECK(Utility.Compare<int>(Top.shape(), top_diff.shape()), "The top and top_diff should have the same shape!");
                m_cuda.copy(top_diff.count(), top_diff.gpu_data, Top.mutable_gpu_diff);
                layer_nd.Backward(TopVec, rgbPropagateDown, BottomVec);

                bCopyDiff = true;
                bReshape = true;
                backward_result_nd.CopyFrom(Bottom, bCopyDiff, bReshape);
                backward_weight_result_nd.CopyFrom(weights, bCopyDiff, bReshape);
            }

            m_log.CHECK_EQ(result_nd.count(), result_2d.count(), "The result_2d and result_nd should have the same count().");

            double[] rgResult2D = convert(result_2d.update_cpu_data());
            double[] rgResultND = convert(result_nd.update_cpu_data());

            for (int i = 0; i < result_2d.count(); i++)
            {
                double df2D = rgResult2D[i];
                double dfND = rgResultND[i];

#warning TestDeconvolutionLayer<T>.TestNDAgainst2D test fails.
                m_log.CHECK_EQ(df2D, dfND, "The 2D and ND values at " + i.ToString() + " should be equal!");
            }

            m_log.CHECK_EQ(backward_result_nd.count(), backward_result_2d.count(), "The backward_result_2d and backward_result_nd should have the same count().");

            double[] rgBackwardResult2D = convert(backward_result_2d.update_cpu_diff());
            double[] rgBackwardResultND = convert(backward_result_nd.update_cpu_diff());

            for (int i = 0; i < backward_result_2d.count(); i++)
            {
                double df2D = rgBackwardResult2D[i];
                double dfND = rgBackwardResultND[i];

                m_log.CHECK_EQ(df2D, dfND, "The backward 2D and ND values at " + i.ToString() + " should be equal!");
            }

            m_log.CHECK_EQ(backward_weight_result_nd.count(), backward_weight_result_2d.count(), "The backward_weight_result_2d and backward_result_nd should have the same count().");

            double[] rgBackwardWeightResult2D = convert(backward_weight_result_2d.update_cpu_diff());
            double[] rgBackwardWeightResultND = convert(backward_weight_result_nd.update_cpu_diff());

            for (int i = 0; i < backward_weight_result_2d.count(); i++)
            {
                double df2D = rgBackwardWeightResult2D[i];
                double dfND = rgBackwardWeightResultND[i];

                m_log.CHECK_EQ(df2D, dfND, "The backward weight 2D and ND values at " + i.ToString() + " should be equal!");
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        public void TestGradient3D()
        {
            List<int> rgBottomShape = new List<int>();
            rgBottomShape.Add(BottomVec[0].shape(0));
            rgBottomShape.Add(BottomVec[0].shape(1));
            rgBottomShape.Add(2);
            rgBottomShape.Add(3);
            rgBottomShape.Add(2);
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
                filler.Fill(BottomVec[i]);
            }

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DECONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(2);
            p.convolution_param.stride.Add(2);
            p.convolution_param.pad.Add(1);
            p.convolution_param.num_output = 2;
            p.convolution_param.weight_filler.type = "gaussian";
            p.convolution_param.bias_filler.type = "gaussian";
            DeconvolutionLayer<T> layer = new DeconvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
#warning TestDeconvolutionLayer<T>.TestGradient3D test fails.
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
