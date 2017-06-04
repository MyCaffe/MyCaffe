using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The SilenceLayer ignores bottom blobs while producing no top blobs.  (This is useuful
    /// to suppress output during testing.)
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SilenceLayer<T> : Layer<T>
    {
        /// <summary>
        /// The SilenceLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SILENCE.</param>
        public SilenceLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SILENCE;
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns 0 as this layer has no output.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 0; }
        }

        /// @brief Not implemented - no setup needed.
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - no reshape needed.
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - no output.
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// <summary>
        /// The backward computation merely sets the bottom diff to zero.
        /// </summary>
        /// <param name="colTop">Not used.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            for (int i = 0; i < colBottom.Count; i++)
            {
                if (rgbPropagateDown[i])
                {
                    colBottom[i].SetDiff(0);
                }
            }
        }
    }
}
