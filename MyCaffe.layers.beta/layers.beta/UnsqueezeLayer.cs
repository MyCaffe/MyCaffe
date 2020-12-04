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
    /// The UnsqueezeLayer performs an unsqueeze operation where a single dimension is inserted at each index specified.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class UnsqueezeLayer<T> : Layer<T>
    {
        /// <summary>
        /// The UnsqueezeLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides FlattenParameter flatten_param
        /// </param>
        public UnsqueezeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.UNSQUEEZE;
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
        }

        /// <summary>
        /// Reshape the top (output) blob.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        /// <remarks>
        /// Unsqueeze axes are either specified from the squeeze_param.axes, or
        /// from the colBottom[1] blob, but not both.
        /// </remarks>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!reshapeNeeded(colBottom, colTop))
                return;

            float[] rgIdxF = null;
            List<int> rgDim = new List<int>(colBottom[0].shape());

            if (colBottom.Count > 1)
                rgIdxF = convertF(colBottom[1].mutable_cpu_data);

            List<int> rgIdx = new List<int>();
            if (rgIdxF != null)
            {
                if (m_param.squeeze_param.axes.Count > 0)
                    m_log.WriteLine("WARNING: Squeeze indexes specified in as blob input data, squeeze parameters will be ignored.");

                for (int i = 0; i < rgIdxF.Length; i++)
                {
                    rgIdx.Add((int)rgIdxF[i]);
                }
            }
            else
            {
                rgIdx = new List<int>(m_param.squeeze_param.axes);
            }

            rgIdx = rgIdx.OrderBy(p => p).ToList();

            for (int i = 0; i < rgIdx.Count; i++)
            {
                int nAxis = rgIdx[i];

                if (nAxis >= rgDim.Count)
                    rgDim.Add(1);
                else
                    rgDim.Insert(nAxis, 1);
            }

            colTop[0].Reshape(rgDim);
            colTop[0].ShareData(colBottom[0]);
            colTop[0].ShareDiff(colBottom[0]);
        }

        /// @brief Not implemented - squeeze Layers do not perform forward.
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - squeeze Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
