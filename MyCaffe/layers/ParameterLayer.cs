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
    /// The ParameterLayer passes its blob[0] data and diff to the top[0].
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ParameterLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The ParameterLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter.</param>
        public ParameterLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.PARAMETER;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: none.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
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
            if (blobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                Blob<T> blob = new Blob<T>(m_cuda, m_log);
                blob.Name = m_param.name + " param";
                blob.Reshape(m_param.parameter_param.shape);
                m_colBlobs.Add(blob);
            }

            colTop[0].Reshape(m_param.parameter_param.shape);
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
        /// Pass the blob[0] through to the top[0].
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs x, which are ignored.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the blob[0] values</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ShareData(m_colBlobs[0]);
            colTop[0].ShareDiff(m_colBlobs[0]);
        }

        /// <summary>
        /// Does nothing.
        /// </summary>
        /// <param name="colTop">top output blob vector, which is ignored.</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward, which is ignored.</param>
        /// <param name="colBottom">bottom input blob vector, which is ignored.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
