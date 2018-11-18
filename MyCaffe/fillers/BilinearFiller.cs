using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;

namespace MyCaffe.fillers
{
    /// <summary>
    /// Fills a Blob with coefficients for bilinear interpolation.
    /// </summary>
    /// <remarks>
    /// A common use case is with the DeconvolutionLayer acting as unsampling.
    /// You can upsample a feature amp with shape of (B, C, H, W) by any integer factor
    /// using the following proto:
    /// 
    /// <code>
    /// layer {
    ///   name: "upsample", type: "Deconvolution"
    ///   bottom: "{{bottom_name}}" top: "{{top_name}}"
    ///   convolution_param {
    ///     kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    ///     num_output: {{C}} group: {{C}}
    ///     pad: {{ceil((factor - 1) / 2.0)}}
    ///     weight_filler: { type: "bilinear" } bias_term: false
    ///   }
    ///   param { lr_mult: 0 decay_mult: 0 }
    /// }
    /// </code>
    /// 
    /// Please use this by replacing '{{}}' with your values.  By specifying
    /// 'num_output: {{C}} group: {{C}}', it behaves as
    /// channel-wise convolution.  The filter shape of this deconvolution layer will be
    /// (C, 1, K, K) where K is 'kernel_size', and this filler will set a (K, K)
    /// interpolation kernel for every channel of the filter identically.  The resulting
    /// shape of the top featur emap will be (B, C, factor * H, factory * W).
    /// 
    /// Note, that the learning rate and the
    /// weight decay are set to 0 in order to keep coefficient values of bilinear
    /// interpolation uncahnged during training.  If you apply this to an image, this
    /// operation is equivalent to the following call in Python with Scikit.Image:
    /// 
    /// <code>
    /// out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
    /// </code>
    /// </remarks>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public class BilinearFiller<T> : Filler<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public BilinearFiller(CudaDnn<T> cuda, Log log, FillerParameter p)
            : base(cuda, log, p)
        {
        }

        /// <summary>
        /// Fill a memory with bilinear values.
        /// </summary>
        /// <param name="nCount">Specifies the number of items to fill.</param>
        /// <param name="hMem">Specifies the handle to GPU memory to fill.</param>
        /// <param name="nNumAxes">Optionally, specifies the number of axes (default = 1).</param>
        /// <param name="nNumOutputs">Optionally, specifies the number of outputs (default = 1).</param>
        /// <param name="nNumChannels">Optionally, specifies the number of channels (default = 1).</param>
        /// <param name="nHeight">Optionally, specifies the height (default = 1).</param>
        /// <param name="nWidth">Optionally, specifies the width (default = 1).</param>
        public override void Fill(int nCount, long hMem, int nNumAxes = 1, int nNumOutputs = 1, int nNumChannels = 1, int nHeight = 1, int nWidth = 1)
        {
            m_log.CHECK_EQ(nNumAxes, 4, "Blob must be 4 dim.");
            m_log.CHECK_EQ(nWidth, nHeight, "Filter must be square.");

            T[] rgData = m_cuda.GetMemory(hMem);
            int nF = (int)Math.Ceiling(nWidth / 2.0);
            double dfC = (nWidth - 1) / (2.0 * nF);

            for (int i = 0; i < nCount; i++)
            {
                double dfX = i % nWidth;
                double dfY = (i / nWidth) % nHeight;
                double dfVal = (1 - Math.Abs(dfX / nF - dfC)) * (1 - Math.Abs(dfY / nF - dfC));

                rgData[i] = (T)Convert.ChangeType(dfVal, typeof(T));
            }

            m_cuda.SetMemory(hMem, rgData);

            m_log.CHECK_EQ(-1, m_param.sparse, "Sparsity not supported by this Filler.");
        }
    }
}
