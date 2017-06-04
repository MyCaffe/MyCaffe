using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestIm2Col
    {
        [TestMethod]
        public void Test2D()
        {
            Im2ColTest test = new Im2ColTest();

            try
            {
                foreach (IIm2ColTest t in test.Tests)
                {
                    t.Test2D();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class Im2ColTest : TestBase
    {
        public Im2ColTest()
            : base("Im2Col Test")
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            if (dt == common.DataType.DOUBLE)
                return new Im2ColTest<double>(strName, nDeviceID);
            else
                return new Im2ColTest<float>(strName, nDeviceID);
        }
    }

    interface IIm2ColTest
    {
        void Test2D();
    }

    class Im2ColTest<T> : Test<T>, IIm2ColTest
    {
        Blob<T> m_blobKernelShape;
        Blob<T> m_blobStride;
        Blob<T> m_blobPad;
        Blob<T> m_blobDilation;
        Blob<T> m_blobBottom;
        Blob<T> m_blobTop;
        Blob<T> m_blobTopCpu;
        int m_nHeight;
        int m_nWidth;
        int m_nChannels;
        int m_nPad;
        int m_nStride;
        int m_nDilation;
        int m_nKernelSize;
        int m_nHeightCol;
        int m_nWidthCol;

        public Im2ColTest(string strName, int nDeviceID)
            : base(strName, nDeviceID)
        {
            m_blobBottom = new Blob<T>(m_cuda, m_log, 5, 500, 15, 15);
            m_blobKernelShape = new Blob<T>(m_cuda, m_log);
            m_blobStride = new Blob<T>(m_cuda, m_log);
            m_blobPad = new Blob<T>(m_cuda, m_log);
            m_blobDilation = new Blob<T>(m_cuda, m_log);
            m_blobTop = new Blob<T>(m_cuda, m_log);
            m_blobTopCpu = new Blob<T>(m_cuda, m_log);

            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blobBottom);

            List<int> rgDimBlobShape = new List<int>() { 2 };
            m_blobKernelShape.Reshape(rgDimBlobShape);
            m_blobStride.Reshape(rgDimBlobShape);
            m_blobPad.Reshape(rgDimBlobShape);
            m_blobDilation.Reshape(rgDimBlobShape);

            m_nHeight = m_blobBottom.height;
            m_nWidth = m_blobBottom.width;
            m_nChannels = m_blobBottom.channels;
            m_nPad = 0;
            m_nStride = 2;
            m_nDilation = 3;
            m_nKernelSize = 3;
            m_nHeightCol = (m_nHeight + 2 * m_nPad - (m_nDilation * (m_nKernelSize - 1) + 1)) / m_nStride + 1;
            m_nWidthCol = (m_nWidth + 2 * m_nPad - (m_nDilation * (m_nKernelSize - 1) + 1)) / m_nStride + 1;

            for (int i = 0; i < 2; i++)
            {
                m_blobKernelShape.SetData(m_nKernelSize, i);
                m_blobStride.SetData(m_nStride, i);
                m_blobPad.SetData(m_nPad, i);
                m_blobDilation.SetData(m_nDilation, i);
            }
        }

        protected override void dispose()
        {
            m_blobKernelShape.Dispose();
            m_blobStride.Dispose();
            m_blobPad.Dispose();
            m_blobDilation.Dispose();
            m_blobBottom.Dispose();
            m_blobTop.Dispose();
            m_blobTopCpu.Dispose();
            base.dispose();
        }

        public Blob<T> Bottom
        {
            get { return m_blobBottom; }
        }

        public Blob<T> Top
        {
            get { return m_blobTop; }
        }

        public Blob<T> TopCpu
        {
            get { return m_blobTopCpu; }
        }

        public Blob<T> KernelShape
        {
            get { return m_blobKernelShape; }
        }

        public Blob<T> Stride
        {
            get { return m_blobStride; }
        }

        public Blob<T> Pad
        {
            get { return m_blobPad; }
        }

        public Blob<T> Dilation
        {
            get { return m_blobDilation; }
        }

        public int height
        {
            get { return m_nHeight; }
        }

        public int width
        {
            get { return m_nWidth; }
        }

        public int channels
        {
            get { return m_nChannels; }
        }

        public int pad
        {
            get { return m_nPad; }
        }

        public int stride
        {
            get { return m_nStride; }
        }

        public int dilation
        {
            get { return m_nDilation; }
        }

        public int kernel_size
        {
            get { return m_nKernelSize; }
        }

        public int height_col
        {
            get { return m_nHeightCol; }
        }

        public int width_col
        {
            get { return m_nWidthCol; }
        }

        public void Test2D()
        {
            // Reshape the blobs to correct size for im2col output.
            m_blobTop.Reshape(m_blobBottom.num,
                              m_nChannels * m_nKernelSize * m_nKernelSize,
                              m_nHeightCol,
                              m_nWidthCol);
            m_blobTopCpu.ReshapeLike(m_blobTop);

            long hBottomData = m_blobBottom.gpu_data;
            long hTopData = m_blobTop.mutable_gpu_data;
            long hCpuData = m_blobTopCpu.mutable_gpu_data;

            // CPU version
            for (int n = 0; n < m_blobBottom.num; n++)
            {
                m_cuda.im2col(m_blobBottom.gpu_data,
                              m_blobBottom.offset(n),
                              m_nChannels,
                              m_nHeight,
                              m_nWidth,
                              m_nKernelSize, m_nKernelSize,
                              m_nPad, m_nPad,
                              m_nStride, m_nStride,
                              m_nDilation, m_nDilation,
                              m_blobTopCpu.mutable_gpu_data,
                              m_blobTopCpu.offset(n));
            }

            int nNumKernels = m_nChannels * m_nHeightCol * m_nWidthCol;

            // ND Version
            for (int n = 0; n < m_blobBottom.num; n++)
            {
                m_cuda.im2col_nd(m_blobBottom.gpu_data,
                              m_blobBottom.offset(n),
                              2,
                              nNumKernels,
                              1,
                              m_blobBottom.gpu_shape,
                              m_blobTop.gpu_shape,
                              m_blobKernelShape.gpu_data,
                              m_blobPad.gpu_data,
                              m_blobStride.gpu_data,
                              m_blobDilation.gpu_data,
                              m_blobTop.mutable_gpu_data,
                              m_blobTop.offset(n));                              
            }

            T[] rgTop = m_blobTop.update_cpu_data();
            T[] rgTopCpu = m_blobTopCpu.update_cpu_data();

            m_log.CHECK_EQ(rgTop.Length, rgTopCpu.Length, "The top lengths must be the same.");

            for (int i = 0; i < rgTop.Length; i++)
            {
                double df1 = (double)Convert.ChangeType(rgTop[i], typeof(double));
                double df2 = (double)Convert.ChangeType(rgTopCpu[i], typeof(double));

                m_log.CHECK_EQ(df1, df2, "The values at " + i.ToString() + " are not equal.");
            }
        }

        private int CAFFE_GET_BLOCKS(int n)
        {
            return (n + 512 - 1) / 512;
        }
    }
}
