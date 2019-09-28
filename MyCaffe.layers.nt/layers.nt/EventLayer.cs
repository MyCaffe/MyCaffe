using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

/// <summary>
/// The MyCaffe.layers.beta namespace contains all early release layers that have a fluid and changing code base.
/// </summary>
namespace MyCaffe.layers.nt
{
    /// <summary>
    /// The EventLayer provides an event that fires on the forward pass and another that fires on the backward pass.
    /// </summary>
    /// <remarks>
    /// This layer allows for creating a custom layer.  When either the forward or backward pass event is not implemented,
    /// the layer merely acts as a pass-through.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class EventLayer<T> : Layer<T>
    {
        /// <summary>
        /// Defines the event that fires from within the LayerSetup function.
        /// </summary>
        public event EventHandler<ForwardArgs<T>> OnLayerSetup;
        /// <summary>
        /// Defines the event that fires from within the Reshape function.
        /// </summary>
        public event EventHandler<ForwardArgs<T>> OnReshape;
        /// <summary>
        /// Defines the event that fires from within the forward pass.
        /// </summary>
        public event EventHandler<ForwardArgs<T>> OnForward;
        /// <summary>
        /// Defines the event that fires from within the backward pass.
        /// </summary>
        public event EventHandler<BackwardArgs<T>> OnBackward;

        /// <summary>
        /// The EventLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter.</param>
        public EventLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.EVENT;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_bUseHalfSize = m_param.use_halfsize;

            if (OnLayerSetup != null)
                OnLayerSetup(this, new ForwardArgs<T>(colBottom, colTop));
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (OnReshape != null)
            {
                OnReshape(this, new ForwardArgs<T>(colBottom, colTop));
                return;
            }

            colTop[0].ReshapeLike(colBottom[0], m_bUseHalfSize);
        }

        /// <summary>
        /// Either fires the OnForward event, or acts as a pass-through.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs x</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the outputs y</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (OnForward != null)
            {
                OnForward(this, new ForwardArgs<T>(colBottom, colTop));
                return;
            }

            for (int i = 0; i < colBottom.Count && i < colTop.Count; i++)
            {
                int nCount = colTop[i].count();
                int nCountB = colBottom[i].count();

                m_log.CHECK_EQ(nCount, nCountB, "The top and bottom at " + i.ToString() + " must have the same number of items.");

                long hBottomData = colBottom[i].gpu_data;
                long hTopData = colTop[i].mutable_gpu_data;

                m_cuda.copy(nCount, hBottomData, hTopData);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the absolute value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$ with
        ///     respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with gradients,
        ///     if propagate_down[0] == true.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (OnBackward != null)
            {
                OnBackward(this, new BackwardArgs<T>(colTop, rgbPropagateDown, colBottom));
                return;
            }

            for (int i = 0; i < colBottom.Count && i < colTop.Count; i++)
            {
                if (rgbPropagateDown[i])
                {
                    int nCount = colTop[i].count();
                    int nCountB = colBottom[i].count();

                    m_log.CHECK_EQ(nCount, nCountB, "The top and bottom at " + i.ToString() + " must have the same number of items.");

                    long hBottomDiff = colBottom[i].mutable_gpu_diff;
                    long hTopDiff = colTop[i].gpu_diff;

                    m_cuda.copy(nCount, hTopDiff, hBottomDiff);
                }
            }
        }
    }
}
