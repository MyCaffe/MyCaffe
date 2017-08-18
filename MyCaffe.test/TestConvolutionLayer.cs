using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;
using MyCaffe.layers;
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestConvolutionLayer
    {
        #region CuDNN Tests

        [TestMethod]
        public void TestSetupCuDnn()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
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
        public void TestSimpleConvolutionCuDnn()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSimpleConvolution");
                        t.TestSimpleConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleConvolutionGroupCuDnn()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSimpleConvolutionGroup");
                        t.TestSimpleConvolutionGroup();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSobelConvolutionCuDnn()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSobelConvolution");
                        t.TestSobelConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCuDnn()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
#warning TestConvolutionLayer.TestGradientCuDnn test fails when run as <double> on CUDA 9.0RC + cuDnn 7.0.
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestGradient");
                        t.TestGradient();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientGroupCuDnn()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestGradientGroup");
                        t.TestGradientGroup();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion

        #region CAFFE Only Tests

        [TestMethod]
        public void TestSetup()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
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
        public void TestSimpleConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSimpleConvolution");
                        t.TestSimpleConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimpleConvolutionGroup()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSimpleConvolutionGroup");
                        t.TestSimpleConvolutionGroup();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSobelConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSobelConvolution");
                        t.TestSobelConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
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
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestGradient");
                        t.TestGradient();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDilatedGradient()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestDilatedGradient");
                        t.TestDilatedGradient();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientGroup()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestGradientGroup");
                        t.TestGradientGroup();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDilatedConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestDilatedConvolution");
                        t.TestDilatedConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test0DConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": Test0DConvolution");
                        t.Test0DConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSimple3DConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestSimple3DConvolution");
                        t.TestSimple3DConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDilated3DConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestDialated3DConvolution");
                        t.TestDilated3DConvolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test1x1Convolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": Test1x1Convolution");
                        t.Test1x1Convolution();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
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
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
#warning TestConvolutionLayer.TestNDAgainst2d test fails.
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestNDAgainst2D");
                        t.TestNDAgainst2D();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails
        /// </summary>
        [TestMethod]
        public void TestGradient3D()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            foreach (IConvolutionLayerTest t in test.Tests)
            {
                try
                {
#warning TestConvolutionLayer.TestGradient3D test fails.
                    Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": TestGradient3D");
                    t.TestGradient3D();
                }
                catch (Exception excpt)
                {
                    throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                }
            }
        }

        [TestMethod]
        public void Test1x1Gradient()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionLayerTest t in test.Tests)
                {
                    try
                    {
                        Trace.WriteLine(t.DataType.ToString() + ":" + t.engine.ToString() + ": Test1x1Gradient");
                        t.Test1x1Gradient();
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception(t.DataType.ToString() + ":" + t.engine.ToString() + excpt.Message);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        #endregion
    }

    class ConvolutionLayerTest : TestBase
    {
        public ConvolutionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Convolution Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ConvolutionLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ConvolutionLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IConvolutionLayerTest : ITest
    {
        void TestSetup();
        void TestSimpleConvolution();
        void TestDilatedConvolution();
        void Test0DConvolution();
        void TestSimple3DConvolution();
        void TestDilated3DConvolution();
        void Test1x1Convolution();
        void TestSimpleConvolutionGroup();
        void TestSobelConvolution();
        void TestNDAgainst2D();
        void TestGradient();
        void TestDilatedGradient();
        void TestGradient3D();
        void Test1x1Gradient();
        void TestGradientGroup();
    }

    class ConvolutionLayerTest<T> : TestEx<T>, IConvolutionLayerTest
    {
        Blob<T> m_blob_bottom2;
        Blob<T> m_blob_top2;
        Blob<T> m_ref_blob_top;
        
        public ConvolutionLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 4 }, nDeviceID)
        {
            m_blob_bottom2 = new Blob<T>(m_cuda, m_log);
            m_blob_top2 = new Blob<T>(m_cuda, m_log);

            m_blob_bottom2.ReshapeLike(Bottom);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom2);

            m_engine = engine;

            BottomVec.Add(m_blob_bottom2);
            TopVec.Add(m_blob_top2);
        }

        protected override void dispose()
        {
            m_blob_bottom2.Dispose();
            m_blob_top2.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("gaussian");
            p.value = 1.0;
            return p;
        }

        public Blob<T> MakeReferenceTop(Blob<T> top)
        {
            m_ref_blob_top = new Blob<T>(m_cuda, m_log);
            m_ref_blob_top.ReshapeLike(top);
            return m_ref_blob_top;
        }

        public Blob<T> Bottom2
        {
            get { return m_blob_bottom2; }
        }

        public Blob<T> Top2
        {
            get { return m_blob_top2; }
        }

        public Blob<T> TopRef
        {
            get { return m_ref_blob_top; }
        }

        public void caffe_conv(Blob<T> blobIn, ConvolutionParameter cp, BlobCollection<T> colBlobWeights, Blob<T> blobOut)
        {
            bool bHasDepth = (blobOut.num_axes == 5) ? true : false;
            int nHasDepth = 1;

            if (!bHasDepth)
            {
                nHasDepth = 0;
                m_log.CHECK_EQ(4.0, blobOut.num_axes, "Without depth the blobOut must have 4 axes.");
            }

            // Kernel size
            int nKernelH;
            int nKernelW;

            if (cp.kernel_h.HasValue || cp.kernel_w.HasValue)
            {
                nKernelH = (int)cp.kernel_h.Value;
                nKernelW = (int)cp.kernel_w.Value;
            }
            else
            {
                nKernelH = (int)cp.kernel_size[0];
                nKernelW = (int)cp.kernel_size[0];
            }

            // Kernel pad
            int nPadH;
            int nPadW;

            if (cp.pad_h.HasValue || cp.pad_w.HasValue)
            {
                nPadH = (int)cp.pad_h.Value;
                nPadW = (int)cp.pad_w.Value;
            }
            else
            {
                nPadH = (cp.pad.Count > 0) ? (int)cp.pad[0] : 0;
                nPadW = (cp.pad.Count > 0) ? (int)cp.pad[0] : 0;
            }

            // Kernel stride
            int nStrideH;
            int nStrideW;

            if (cp.stride_h.HasValue || cp.stride_w.HasValue)
            {
                nStrideH = (int)cp.stride_h.Value;
                nStrideW = (int)cp.stride_w.Value;
            }
            else
            {
                nStrideH = (cp.stride.Count > 0) ? (int)cp.stride[0] : 1;
                nStrideW = (cp.stride.Count > 0) ? (int)cp.stride[0] : 1;
            }

            // Dilation
            int nDilationH = (cp.dilation.Count > 0) ? (int)cp.dilation[0] : 1;
            int nDilationW = (cp.dilation.Count > 0) ? (int)cp.dilation[0] : 1;
          
            int nKernelD = 1;
            int nStrideD = 1;
            int nDilationD = 1;
            int nPadD = 0;

            if (bHasDepth)
            {
                nKernelD = nKernelH;
                nStrideD = nStrideH;
                nPadD = nPadH;
                nDilationD = nDilationH;
            }

            // Groups
            int nGroups = (int)cp.group;
            int nOG = blobOut.shape(1) / nGroups;
            int nKG = blobIn.shape(1) / nGroups;
            int nOHead;
            int nKHead;

            // Convolution
            List<int> rgWeightOffset = new List<int>();
            List<int> rgInOffset = new List<int>();
            List<int> rgOutOffset = new List<int>();
            int nCount = 4 + nHasDepth;

            for (int i = 0; i < nCount; i++)
            {
                rgWeightOffset.Add(0);
                rgInOffset.Add(0);
                rgOutOffset.Add(0);
            }

            T[] rgOutData = blobOut.mutable_cpu_data;

            for (int n = 0; n < blobOut.shape(0); n++)
            {
                for (int g = 0; g < nGroups; g++)
                {
                    nOHead = nOG * g;
                    nKHead = nKG * g;

                    for (int o = 0; o < nOG; o++)
                    {
                        for (int k = 0; k < nKG; k++)
                        {
                            int nZCount = (bHasDepth) ? blobOut.shape(2) : 1;

                            for (int z = 0; z < nZCount; z++)
                            {
                                int nYCount = blobOut.shape(2 + nHasDepth);

                                for (int y = 0; y < nYCount; y++)
                                {
                                    int nXCount = blobOut.shape(3 + nHasDepth);

                                    for (int x = 0; x < nXCount; x++)
                                    {
                                        for (int r = 0; r < nKernelD; r++)
                                        {
                                            for (int p = 0; p < nKernelH; p++)
                                            {
                                                for (int q = 0; q < nKernelW; q++)
                                                {
                                                    int in_z = z * nStrideD - nPadD + r * nDilationD;
                                                    int in_y = y * nStrideH - nPadH + p * nDilationH;
                                                    int in_x = x * nStrideW - nPadW + q * nDilationW;

                                                    if (in_z >= 0 && in_z < (bHasDepth ? blobIn.shape(2) : 1) &&
                                                        in_y >= 0 && in_y < blobIn.shape(2 + nHasDepth) &&
                                                        in_x >= 0 && in_x < blobIn.shape(3 + nHasDepth))
                                                    {
                                                        rgWeightOffset[0] = o + nOHead;
                                                        rgWeightOffset[1] = k;

                                                        if (bHasDepth)
                                                            rgWeightOffset[2] = r;

                                                        rgWeightOffset[2 + nHasDepth] = p;
                                                        rgWeightOffset[3 + nHasDepth] = q;

                                                        rgInOffset[0] = n;
                                                        rgInOffset[1] = k + nKHead;

                                                        if (bHasDepth)
                                                            rgInOffset[2] = in_z;

                                                        rgInOffset[2 + nHasDepth] = in_y;
                                                        rgInOffset[3 + nHasDepth] = in_x;

                                                        rgOutOffset[0] = n;
                                                        rgOutOffset[1] = o + nOHead;

                                                        if (bHasDepth)
                                                            rgOutOffset[2] = z;

                                                        rgOutOffset[2 + nHasDepth] = y;
                                                        rgOutOffset[3 + nHasDepth] = x;

                                                        double dfIn = (double)Convert.ChangeType(blobIn.data_at(rgInOffset), typeof(double));
                                                        double dfWts = (double)Convert.ChangeType(colBlobWeights[0].data_at(rgWeightOffset), typeof(double));
                                                        double dfOut = (double)Convert.ChangeType(rgOutData[blobOut.offset(rgOutOffset)], typeof(double));

                                                        dfOut += dfIn * dfWts;

                                                        rgOutData[blobOut.offset(rgOutOffset)] = (T)Convert.ChangeType(dfOut, typeof(T));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Bias
            if (cp.bias_term)
            {
                T[] rgBiasData = colBlobWeights[1].update_cpu_data();

                for (int n = 0; n < blobOut.shape(0); n++)
                {
                    for (int o = 0; o < blobOut.shape(1); o++)
                    {
                        int nZCount = (bHasDepth) ? blobOut.shape(2) : 1;

                        for (int z = 0; z < nZCount; z++)
                        {
                            int nYCount = blobOut.shape(2 + nHasDepth);

                            for (int y = 0; y < nYCount; y++)
                            {
                                int nXCount = blobOut.shape(3 + nHasDepth);

                                for (int x = 0; x < nXCount; x++)
                                {
                                    rgOutOffset[0] = n;
                                    rgOutOffset[1] = o;

                                    if (bHasDepth)
                                        rgOutOffset[2] = z;

                                    rgOutOffset[2 + nHasDepth] = y;
                                    rgOutOffset[3 + nHasDepth] = x;

                                    double dfBias = (double)Convert.ChangeType(rgBiasData[o], typeof(double));
                                    double dfOut = (double)Convert.ChangeType(rgOutData[blobOut.offset(rgOutOffset)], typeof(double));

                                    dfOut += dfBias;

                                    rgOutData[blobOut.offset(rgOutOffset)] = (T)Convert.ChangeType(dfOut, typeof(T));
                                }
                            }
                        }
                    }
                }
            }

            blobOut.mutable_cpu_data = rgOutData;
        }


        #region CuDNN and CAFFE Tests

        public void TestSetup()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 4;
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, Top.num, "The top[0] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(4, Top.channels, "The top[0] Blob<T> should have channels = 4.");
            m_log.CHECK_EQ(2, Top.height, "The top[0] Blob<T> should have height = 2.");
            m_log.CHECK_EQ(1, Top.width, "The top[0] Blob<T> should have width = 1.");
            m_log.CHECK_EQ(2, Top2.num, "The top[1] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(4, Top2.channels, "The top[1] Blob<T> should have channels = 4.");
            m_log.CHECK_EQ(2, Top2.height, "The top[1] Blob<T> should have height = 2.");
            m_log.CHECK_EQ(1, Top2.width, "The top[1] Blob<T> should have width = 1.");

            // setting group should not change the shape
            p.convolution_param.num_output = 3;
            p.convolution_param.group = 3;
            layer = new ConvolutionLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, Top.num, "The top[0] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(3, Top.channels, "The top[0] Blob<T> should have channels = 3.");
            m_log.CHECK_EQ(2, Top.height, "The top[0] Blob<T> should have height = 2.");
            m_log.CHECK_EQ(1, Top.width, "The top[0] Blob<T> should have width = 1.");
            m_log.CHECK_EQ(2, Top2.num, "The top[1] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(3, Top2.channels, "The top[1] Blob<T> should have channels = 3.");
            m_log.CHECK_EQ(2, Top2.height, "The top[1] Blob<T> should have height = 2.");
            m_log.CHECK_EQ(1, Top2.width, "The top[1] Blob<T> should have width = 1.");
        }

        public void TestSimpleConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 4;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("constant");
            p.convolution_param.bias_filler.value = 0.1;
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            caffe_conv(Bottom, p.convolution_param, layer.blobs, MakeReferenceTop(Top));
            T[] rgTopData = Top.update_cpu_data();
            T[] rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }

            caffe_conv(Bottom2, p.convolution_param, layer.blobs, MakeReferenceTop(Top2));
            rgTopData = Top2.update_cpu_data();
            rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }
        }

        public void TestSimpleConvolutionGroup()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 3;
            p.convolution_param.group = 3;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("constant");
            p.convolution_param.bias_filler.value = 0.1;
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            BottomVec.RemoveAt(1);
            TopVec.RemoveAt(1);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            caffe_conv(Bottom, p.convolution_param, layer.blobs, MakeReferenceTop(Top));
            T[] rgTopData = Top.update_cpu_data();
            T[] rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }
        }

        /// <summary>
        /// Test separable convolution by computing the Sobel operator
        /// as a signle filter when comparing the result
        /// as the convolution of two rectangualr filters.
        /// </summary>
        public void TestSobelConvolution()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();

            // Fill bottoms with identical Gaussian noise.
            FillerParameter fp = new FillerParameter("gaussian");
            fp.value = 1;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(Bottom);
            Bottom2.CopyFrom(Bottom);

            // Compute Sobel G_x operator as 3 x 3 convolution.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 1;
            p.convolution_param.bias_term = false;
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            BottomVec.RemoveAt(1);
            TopVec.RemoveAt(1);

            layer.blobs.Clear();
            layer.blobs.Add(new Blob<T>(m_cuda, m_log, 1, 3, 3, 3));
            T[] rgWeights = layer.blobs[0].mutable_cpu_data;

            for (int c = 0; c < 3; c++)
            {
                int i = c * 9; // 3 x 3 filter
                rgWeights[i + 0] = (T)Convert.ChangeType(-1, typeof(T));
                rgWeights[i + 1] = (T)Convert.ChangeType(0, typeof(T));
                rgWeights[i + 2] = (T)Convert.ChangeType(1, typeof(T));
                rgWeights[i + 3] = (T)Convert.ChangeType(-2, typeof(T));
                rgWeights[i + 4] = (T)Convert.ChangeType(0, typeof(T));
                rgWeights[i + 5] = (T)Convert.ChangeType(2, typeof(T));
                rgWeights[i + 6] = (T)Convert.ChangeType(-1, typeof(T));
                rgWeights[i + 7] = (T)Convert.ChangeType(0, typeof(T));
                rgWeights[i + 8] = (T)Convert.ChangeType(1, typeof(T));
            }

            layer.blobs[0].mutable_cpu_data = rgWeights;

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Compute Sobel G_x operator as separable 3x1 and 1x3 convolutions.
            // (1) the [1 2 1] column filter.
            BlobCollection<T> sep_blob_bottom_vec = new BlobCollection<T>();
            BlobCollection<T> sep_blob_top_vec = new BlobCollection<T>();
            sep_blob_bottom_vec.Add(Bottom2);
            sep_blob_top_vec.Add(Top2);
            p.convolution_param.kernel_size = new List<uint>();
            p.convolution_param.kernel_h = 3;
            p.convolution_param.kernel_w = 1;
            p.convolution_param.stride = new List<uint>();
            p.convolution_param.stride_h = 2;
            p.convolution_param.stride_w = 1;
            p.convolution_param.num_output = 1;
            p.convolution_param.bias_term = false;

            layer = new ConvolutionLayer<T>(m_cuda, m_log, p);
            layer.blobs.Clear();
            layer.blobs.Add(new Blob<T>(m_cuda, m_log, 1, 3, 3, 1));
            T[] rgWeights1 = layer.blobs[0].mutable_cpu_data;

            for (int c = 0; c < 3; c++)
            {
                int i = c * 3; // 3 x 1 filter
                rgWeights1[i + 0] = (T)Convert.ChangeType(1, typeof(T));
                rgWeights1[i + 1] = (T)Convert.ChangeType(2, typeof(T));
                rgWeights1[i + 2] = (T)Convert.ChangeType(1, typeof(T));
            }

            layer.blobs[0].mutable_cpu_data = rgWeights1;

            layer.Setup(sep_blob_bottom_vec, sep_blob_top_vec);
            layer.Forward(sep_blob_bottom_vec, sep_blob_top_vec);

            // (2) the [-1 0 1] row filter
            Blob<T> blob_sep = new Blob<T>(m_cuda, m_log);
            blob_sep.CopyFrom(Top2, false, true);
            sep_blob_bottom_vec = new BlobCollection<T>();
            sep_blob_bottom_vec.Add(blob_sep);

            p.convolution_param.kernel_h = 1;
            p.convolution_param.kernel_w = 3;
            p.convolution_param.stride_h = 1;
            p.convolution_param.stride_w = 2;
            p.convolution_param.num_output = 1;
            p.convolution_param.bias_term = false;

            layer = new ConvolutionLayer<T>(m_cuda, m_log, p);
            layer.blobs.Clear();
            layer.blobs.Add(new Blob<T>(m_cuda, m_log, 1, 1, 1, 3));
            T[] rgWeights2 = layer.blobs[0].mutable_cpu_data;

            rgWeights2[0] = (T)Convert.ChangeType(-1, typeof(T));
            rgWeights2[1] = (T)Convert.ChangeType(0, typeof(T));
            rgWeights2[2] = (T)Convert.ChangeType(1, typeof(T));

            layer.blobs[0].mutable_cpu_data = rgWeights2;

            layer.Setup(sep_blob_bottom_vec, sep_blob_top_vec);
            layer.Forward(sep_blob_bottom_vec, sep_blob_top_vec);

            // Test equivalence of full and separate filters.
            T[] rgTop1Data = Top.update_cpu_data();
            T[] rgTop2Data = Top2.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop1 = rgTop1Data[i];
                T dfTop2 = rgTop2Data[i];

                EXPECT_NEAR(dfTop1, dfTop2, 1e-4);
            }
        }

        public void TestGradient()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 2;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestDilatedGradient()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            List<int> rgBottomShape = new List<int>() { 2, 3, 5, 6 };

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
            }

            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.dilation.Add(2);
            p.convolution_param.num_output = 2;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientGroup()
        {
            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            List<int> rgBottomShape = new List<int>();
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 3;
            p.convolution_param.group = 3;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        #endregion

        #region CAFFE Only Tests

        public void TestDilatedConvolution()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            List<int> rgBottomShape = new List<int>() { 2, 3, 8, 7 };

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
            }

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.dilation.Add(2);
            p.convolution_param.num_output = 4;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("constant");
            p.convolution_param.bias_filler.value = 0.1;
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            caffe_conv(Bottom, p.convolution_param, layer.blobs, MakeReferenceTop(Top));
            T[] rgTopData = Top.update_cpu_data();
            T[] rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];
                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }

            caffe_conv(Bottom2, p.convolution_param, layer.blobs, MakeReferenceTop(Top2));
            rgTopData = Top2.update_cpu_data();
            rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }
        }

        public void Test0DConvolution()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            BottomVec.RemoveAt(1);
            TopVec.RemoveAt(1);

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            int nNumOutput = 3;
            p.convolution_param.engine = m_engine;
            p.convolution_param.num_output = (uint)nNumOutput;
            p.convolution_param.axis = 3;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);
            List<int> rgTopShape = Bottom.shape();
            rgTopShape[3] = nNumOutput;

            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(Utility.Compare<int>(rgTopShape, Bottom.shape()), "The bottom shape was not changed as expected!");

            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            List<int> rgWeightOffset = new List<int>() { 0, 0 };
            Blob<T> weight = layer.blobs[0];
            Blob<T> bias = layer.blobs[1];
            int nNum = Top.count(3);
            int nDim = Top.shape(3);
            int nBottomDim = Bottom.shape(3);

            for (int n = 0; n < nNum; n++)
            {
                for (int d = 0; d < nDim; d++)
                {
                    rgWeightOffset[0] = d;
                    T dfVal = bias.GetData(d);

                    for (int nBottomD = 0; nBottomD < nBottomDim; nBottomD++)
                    {
                        rgWeightOffset[1] = nBottomD;

                        double dfWt = (double)Convert.ChangeType(weight.data_at(rgWeightOffset), typeof(double));
                        double dfBtm = (double)Convert.ChangeType(Bottom.GetData(n * nBottomDim + nBottomD), typeof(double));
                        double dfVal1 = (double)Convert.ChangeType(dfVal, typeof(double));
                        dfVal1 += dfWt * dfBtm;

                        dfVal = (T)Convert.ChangeType(dfVal1, typeof(T));
                    }

                    T dfTopData = Top.GetData(n * nDim + d);

                    EXPECT_NEAR(dfVal, dfTopData, 1e-4);
                }
            }
        }

        public void TestSimple3DConvolution()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            List<int> rgBottomShape = new List<int>();
            rgBottomShape.Add(BottomVec[0].shape(0));
            rgBottomShape.Add(BottomVec[0].shape(1));
            rgBottomShape.Add(5);
            rgBottomShape.Add(BottomVec[0].shape(2));
            rgBottomShape.Add(BottomVec[0].shape(3));
            FillerParameter pf = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, pf);

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
                filler.Fill(BottomVec[i]);
            }

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 4;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            caffe_conv(Bottom, p.convolution_param, layer.blobs, MakeReferenceTop(Top));
            T[] rgTopData = Top.update_cpu_data();
            T[] rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }

            caffe_conv(Bottom2, p.convolution_param, layer.blobs, MakeReferenceTop(Top2));
            rgTopData = Top2.update_cpu_data();
            rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }
        }

        public void TestDilated3DConvolution()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            List<int> rgBottomShape = new List<int>();
            rgBottomShape.Add(BottomVec[0].shape(0));
            rgBottomShape.Add(BottomVec[0].shape(1));
            rgBottomShape.Add(6);
            rgBottomShape.Add(7);
            rgBottomShape.Add(8);
            FillerParameter pf = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, pf);

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
                filler.Fill(BottomVec[i]);
            }

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.dilation.Add(2);
            p.convolution_param.num_output = 4;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            caffe_conv(Bottom, p.convolution_param, layer.blobs, MakeReferenceTop(Top));
            T[] rgTopData = Top.update_cpu_data();
            T[] rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }

            caffe_conv(Bottom2, p.convolution_param, layer.blobs, MakeReferenceTop(Top2));
            rgTopData = Top2.update_cpu_data();
            rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }
        }

        public void Test1x1Convolution()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(1);
            p.convolution_param.stride.Add(1);
            p.convolution_param.num_output = 4;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("constant");
            p.convolution_param.bias_filler.value = 0.1;
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            BottomVec.RemoveAt(1);
            TopVec.RemoveAt(1);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Check against reference convolution;
            caffe_conv(Bottom, p.convolution_param, layer.blobs, MakeReferenceTop(Top));
            T[] rgTopData = Top.update_cpu_data();
            T[] rgRefTopData = TopRef.update_cpu_data();

            for (int i = 0; i < Top.count(); i++)
            {
                T dfTop = rgTopData[i];
                T dfTopRef = rgRefTopData[i];

                EXPECT_NEAR(dfTop, dfTopRef, 1e-4);
            }
        }

        /// <summary>
        /// This test fails against CUDNN
        /// </summary>
        public void TestNDAgainst2D()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            BottomVec.RemoveAt(1);
            TopVec.RemoveAt(1);

            int nKernelH = 11;
            int nKernelW = 13;
            List<int> rgBottomShape = new List<int>() { 15, 18, nKernelH * 2, nKernelW * 2 };
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
                filler.Fill(BottomVec[i]);
            }

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            p.convolution_param.engine = m_engine;
            p.convolution_param.num_output = 12;
            p.convolution_param.bias_term = false;
            p.convolution_param.group = 6;
            p.convolution_param.kernel_h = (uint)nKernelH;
            p.convolution_param.kernel_w = (uint)nKernelW;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            Blob<T> weights = new Blob<T>(m_cuda, m_log);
            Blob<T> top_diff = new Blob<T>(m_cuda, m_log);
            bool bCopyDiff = false;
            bool bReshape = false;

            {
                ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);
                layer.Setup(BottomVec, TopVec);
                top_diff.ReshapeLike(Top);
                filler.Fill(top_diff);

                m_log.CHECK_EQ(1, layer.blobs.Count, "There should be 1 Blob<T> in the layer.");
                bCopyDiff = false;
                bReshape = true;
                weights.CopyFrom(layer.blobs[0], bCopyDiff, bReshape);
            }

            List<bool> rgPropagateDown = new List<bool>() { true };
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
                ConvolutionLayer<T> layer_2d = new ConvolutionLayer<T>(m_cuda, m_log, p);
                layer_2d.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(1, layer_2d.blobs.Count, "There must be one Blob<T> in layer_2d.");

                bCopyDiff = false;
                bReshape = false;
                layer_2d.blobs[0].CopyFrom(weights, bCopyDiff, bReshape);
                layer_2d.Forward(BottomVec, TopVec);

                bCopyDiff = false;
                bReshape = true;
                result_2d.CopyFrom(Top, bCopyDiff, bReshape);

                // Copy pre-generated top diff into actual top diff;
                // do Backward and save result in backward_result_2d.
                m_log.CHECK(Utility.Compare<int>(Top.shape(), top_diff.shape()), "Top shape should be equal to top_diff shape.");
                m_cuda.copy(top_diff.count(), top_diff.gpu_data, Top.mutable_gpu_diff);
                layer_2d.Backward(TopVec, rgPropagateDown, BottomVec);

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
                ConvolutionLayer<T> layer_nd = new ConvolutionLayer<T>(m_cuda, m_log, p);
                layer_nd.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(1, layer_nd.blobs.Count, "There must be one Blob<T> in layer_2d.");

                bCopyDiff = false;
                bReshape = false;
                layer_nd.blobs[0].CopyFrom(weights, bCopyDiff, bReshape);
                layer_nd.Forward(BottomVec, TopVec);

                bCopyDiff = false;
                bReshape = true;
                result_nd.CopyFrom(Top, bCopyDiff, bReshape);

                // Copy pre-generated top diff into actual top diff;
                // do Backward and save result in backward_result_nd.
                m_log.CHECK(Utility.Compare<int>(Top.shape(), top_diff.shape()), "Top shape should be equal to top_diff shape.");
                m_cuda.copy(top_diff.count(), top_diff.gpu_data, Top.mutable_gpu_diff);
                layer_nd.Backward(TopVec, rgPropagateDown, BottomVec);

                bCopyDiff = true;
                bReshape = true;
                backward_result_nd.CopyFrom(Bottom, bCopyDiff, bReshape);
                backward_weight_result_nd.CopyFrom(weights, bCopyDiff, bReshape);
            }

            m_log.CHECK_EQ(result_nd.count(), result_2d.count(), "The nd and 2d results must have the same counts.");

            for (int i = 0; i < result_2d.count(); i++)
            {
                T dfNd = result_nd.GetData(i);
                T df2d = result_2d.GetData(i);

                CHECK_EQ(df2d, dfNd, "The 2d and ND results are not the same at " + i.ToString());
            }

            m_log.CHECK_EQ(backward_result_nd.count(), backward_result_2d.count(), "the nd and 2d backward results must have the same count.");

            for (int i = 0; i < backward_result_2d.count(); i++)
            {
                T dfNd = backward_result_nd.GetDiff(i);
                T df2d = backward_result_2d.GetDiff(i);

                m_log.EXPECT_EQUAL<float>(convert(df2d), convert(dfNd), "The 2d and ND backward results are not the same at " + i.ToString());
            }

            m_log.CHECK_EQ(backward_weight_result_nd.count(), backward_weight_result_2d.count(), "the nd and 2d backward weight results must have the same count.");

            for (int i = 0; i < backward_weight_result_2d.count(); i++)
            {
                T dfNd = backward_weight_result_nd.GetDiff(i);
                T df2d = backward_weight_result_2d.GetDiff(i);

                CHECK_EQ(df2d, dfNd, "The 2d and ND backward weight results are not the same at " + i.ToString());
            }
        }

        public void TestGradient3D()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            BottomVec.RemoveAt(1);
            TopVec.RemoveAt(1);

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            List<int> rgBottomShape = new List<int>();
            rgBottomShape.Add(BottomVec[0].shape(0));
            rgBottomShape.Add(BottomVec[0].shape(1));
            rgBottomShape.Add(5);
            rgBottomShape.Add(BottomVec[0].shape(2));
            rgBottomShape.Add(BottomVec[0].shape(3));
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            for (int i = 0; i < BottomVec.Count; i++)
            {
                BottomVec[i].Reshape(rgBottomShape);
                filler.Fill(BottomVec[i]);
            }

            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.num_output = 2;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
#warning Convolution:Caffe:TestGradient3D test fails.
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void Test1x1Gradient()
        {
            if (m_engine != EngineParameter.Engine.CAFFE)
                return;

            ConvolutionLayerTest test = new ConvolutionLayerTest();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
            List<int> rgBottomShape = new List<int>();
            p.convolution_param.engine = m_engine;
            p.convolution_param.kernel_size.Add(1);
            p.convolution_param.stride.Add(1);
            p.convolution_param.num_output = 2;
            p.convolution_param.weight_filler = new FillerParameter("gaussian");
            p.convolution_param.bias_filler = new FillerParameter("gaussian");
            ConvolutionLayer<T> layer = new ConvolutionLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        #endregion
    }
}
