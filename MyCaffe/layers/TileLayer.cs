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
    /// The TileLayer copies a Blob along specified dimensions.
    /// This layer is initialized with the MyCaffe.param.TileParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TileLayer<T> : Layer<T>
    {
        int m_nAxis;
        int m_nTiles;
        int m_nOuterDim;
        int m_nInnerDim;

        /// <summary>
        /// The TileLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type TILE with parameter tile_param,
        /// with options:
        ///   - axis (\b optional, default 1). The index of the axis to tile.
        ///   
        ///   - tiles. The number of copies (tiles) of the Blob to output.
        /// </param>
        public TileLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TILE;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: tile.
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
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nAxis = colBottom[0].CanonicalAxisIndex(m_param.tile_param.axis);
            m_nTiles = m_param.tile_param.tiles;
            m_log.CHECK_GT(m_param.tile_param.tiles, 0, "Number of tiles must be positive.");

            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape());
            rgTopShape[m_nAxis] = colBottom[0].shape(m_nAxis) * m_nTiles;
            colTop[0].Reshape(rgTopShape);

            m_nOuterDim = colBottom[0].count(0, m_nAxis);
            m_nInnerDim = colBottom[0].count(m_nAxis);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# (Set by the <i>axis</i> and <i>tiles</i> parameters) 
        ///     the computed outputs.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int bottom_tile_axis = colBottom[0].shape(m_nAxis);
            int nCount = colTop[0].count();

            m_cuda.tile_fwd(nCount, hBottomData, m_nInnerDim, m_nTiles, bottom_tile_axis, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int bottom_tile_axis = colBottom[0].shape(m_nAxis);
            int tile_size = m_nInnerDim / bottom_tile_axis;
            int nCount = colBottom[0].count();

            m_cuda.tile_bwd(nCount, hTopDiff, tile_size, m_nTiles, bottom_tile_axis, hBottomDiff);
        }
    }
}
