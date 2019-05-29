using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The InputLayer provides data to the Net by assigning top Blobs directly.
    /// This layer is initialized with the MyCaffe.param.InputParameter.
    /// </summary>
    /// <remarks>
    /// This data Layer is a container that merely holds the data assigned to it;
    /// forward, backward and Reshape are all no-ops.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class InputLayer<T> : Layer<T>
    {
        /// <summary>
        /// The InputLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter input_param, with options:
        ///   - shape. Defines the shape of each top (output).
        /// </param>
        public InputLayer(CudaDnn<T> cuda, Log log, LayerParameter p) 
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.INPUT;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumTop = colTop.Count;
            InputParameter p = m_param.input_param;
            int nNumShape = p.shape.Count();

            m_log.CHECK(nNumShape == 0 || nNumShape == 1 || nNumShape == nNumTop, "Must specify 'shape' once, once per top blob, or not at all: " + nNumTop.ToString() + " top vs. " + nNumShape.ToString() + " shapes.");

            if (nNumShape > 0)
            {
                for (int i = 0; i < nNumTop; i++)
                {
                    int nShapeIdx = (p.shape.Count() == 1) ? 0 : i;
                    colTop[i].Reshape(p.shape[nShapeIdx]);
                }
            }

            m_bUseHalfSize = m_param.use_halfsize;
            m_bConvertTopOnFwd = m_param.use_halfsize;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// <summary>
        /// The InputLayer has no bottom Blobs and therefore returns 0.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }


        /// <summary>
        /// The forward computation - which is trivial.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# The shape of the data output is specified by the 'shape' parameter value.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - data Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
