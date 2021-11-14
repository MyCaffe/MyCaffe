using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.beta;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The InterpLayer changes the spatial resolution by bi-linear interpolation.
    /// </summary>
    /// <remarks>
    /// The target size is specified in terms of pixels where the start and end pixels of the
    /// input are mapped to the start and end pixels of the output.
    /// 
    /// @see [TheLegendAli/DeepLab-Context](https://github.com/TheLegendAli/DeepLab-Context) GitHub
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class InterpLayer<T> : Layer<T>
    {
        int m_nNum;
        int m_nChannels;
        int m_nHeightIn;
        int m_nHeightOut;
        int m_nWidthIn;
        int m_nWidthOut;
        int m_nHeightInEff;
        int m_nWidthInEff;
        int m_nPadBeg;
        int m_nPadEnd;

        /// <summary>
        /// The InterpLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides FlattenParameter flatten_param
        /// </param>
        public InterpLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.INTERP;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: flatten
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumSpecs = 0;

            nNumSpecs += (m_param.interp_param.zoom_factor.HasValue) ? 1 : 0;
            nNumSpecs += (m_param.interp_param.shrink_factor.HasValue) ? 1 : 0;
            nNumSpecs += (m_param.interp_param.height.HasValue && m_param.interp_param.width.HasValue) ? 1 : 0;
            m_log.CHECK_EQ(nNumSpecs, 1, "Output dimension specified by zoom factor OR shrink factor OR explicitly.");

            m_nPadBeg = m_param.interp_param.pad_beg;
            m_nPadEnd = m_param.interp_param.pad_end;
            m_log.CHECK_LE(m_nPadBeg, 0, "Only supports non-positive padding (cropping) for now.");
            m_log.CHECK_LE(m_nPadEnd, 0, "Only supports non-positive padding (cropping) for now.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nNum = colBottom[0].num;
            m_nChannels = colBottom[0].channels;
            m_nHeightIn = colBottom[0].height;
            m_nWidthIn = colBottom[0].width;

            m_nHeightInEff = m_nHeightIn + m_nPadBeg + m_nPadEnd;
            m_nWidthInEff = m_nWidthIn + m_nPadBeg + m_nPadEnd;

            if (m_param.interp_param.zoom_factor.HasValue)
            {
                int nZoomFactor = m_param.interp_param.zoom_factor.Value;
                m_log.CHECK_GE(nZoomFactor, 1, "The zoom factor must be positive.");
                m_nHeightOut = m_nHeightInEff + (m_nHeightInEff /*- 1*/) * (nZoomFactor - 1);
                m_nWidthOut = m_nWidthInEff + (m_nWidthInEff /*- 1*/) * (nZoomFactor - 1);
            }
            else if (m_param.interp_param.shrink_factor.HasValue)
            {
                int nShrinkFactor = m_param.interp_param.shrink_factor.Value;
                m_log.CHECK_GE(nShrinkFactor, 1, "The shrink factor must be positive.");
                m_nHeightOut = (m_nHeightInEff - 1) / nShrinkFactor + 1;
                m_nWidthOut = (m_nWidthInEff - 1) / nShrinkFactor + 1;
            }
            else if (m_param.interp_param.height.HasValue && m_param.interp_param.width.HasValue)
            {
                m_nHeightOut = m_param.interp_param.height.Value;
                m_nWidthOut = m_param.interp_param.width.Value;
            }
            else
            {
                m_log.FAIL("You must specify the zoom factor OR shrink factor OR explicit size.");
            }

            m_log.CHECK_GT(m_nHeightInEff, 0, "The height should be positive.");
            m_log.CHECK_GT(m_nWidthInEff, 0, "The width should be positive.");
            m_log.CHECK_GT(m_nHeightOut, 0, "The height should be positive.");
            m_log.CHECK_GT(m_nWidthOut, 0, "The width should be positive.");

            colTop[0].Reshape(m_nNum, m_nChannels, m_nHeightOut, m_nWidthOut);
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2+)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.
        ///     the inputs.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times CHW \times 1 \times 1) @f$ the outputs -- i.e., the (virtually) copied, flattened inputs
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_cuda.interp2(m_nNum * m_nChannels, colBottom[0].gpu_data, -m_nPadBeg, -m_nPadBeg, m_nHeightInEff, m_nWidthInEff, m_nHeightIn, m_nWidthIn, colTop[0].mutable_gpu_data, 0, 0, m_nHeightOut, m_nWidthOut, m_nHeightOut, m_nWidthOut, false);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the concatenate inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vecotr (length 1), 
        /// providing the error gradient with respect to the outputs.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">input Blob vecotor (length @f$ k @f$), into which the top error
        /// gradient is (virtually) copied.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            colBottom[0].SetDiff(0);
            m_cuda.interp2(m_nNum * m_nChannels, colBottom[0].mutable_gpu_diff, -m_nPadBeg, -m_nPadBeg, m_nHeightInEff, m_nWidthInEff, m_nHeightIn, m_nWidthIn, colTop[0].gpu_diff, 0, 0, m_nHeightOut, m_nWidthOut, m_nHeightOut, m_nWidthOut, true);
        }
    }
}
