using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.tft;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The ReshapeTemporalLayer implements the Variable Selection Network
    /// </summary>
    /// <remarks>
    /// When run using the BEFORE mode, this layer is used to reshape static inputs along time while stacking temporal and time distributed contexts along the batch.
    /// When run using the AFTER mode, this layer reshapes the outputs and weight back into their num samples x temporal steps shape.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L1198) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ReshapeTemporalLayer<T> : Layer<T>
    {
        ReshapeTemporalParameter.MODE m_mode;
        int m_nNumInputs;
        int m_nNumSamples;
        int m_nNumTemporalSteps;
        List<int> m_rgShape = new List<int>(4);
        Dictionary<string, List<int>> m_rgShapes = new Dictionary<string, List<int>>();
        Blob<T> m_blobTimeDistributedContext;
        Blob<T> m_blobWork;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public ReshapeTemporalLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.RESHAPE_TEMPORAL;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            if (m_blobTimeDistributedContext != null)
                col.Add(m_blobTimeDistributedContext);
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: temporal_rep
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the max number of required bottom (input) Blobs: temporal_rep, static_selection
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: temporal_selection_output
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: temporal_selection_output, temporal_selection_wts
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// This method converts a static_signal (non-temporal tensor) of shape [num_samples x num_features]
        /// and replicates it along time for 'time_step' number of times to create a new shape
        /// [num_samples x time_steps x num_features]
        /// </summary>
        /// <param name="bSrc">Specifies the source blob</param>
        /// <param name="bDst">Specifies the destination blob.</param>
        /// <param name="nTimeSteps">Specifies the time steps.</param>
        /// <param name="bReshapeOnly">Specifies to reshape the destination only.</param>
        private void replicate_along_time_fwd(Blob<T> bSrc, Blob<T> bDst, int nTimeSteps, bool bReshapeOnly = false)
        {
            m_rgShape.Clear();
            m_rgShape.Add(m_nNumSamples);
            m_rgShape.Add(nTimeSteps);
            m_rgShape.Add(bSrc.shape(1));
            bDst.Reshape(m_rgShape);

            if (!bReshapeOnly)
            {
                int nInnerNum = bSrc.count(1);
                for (int i = 0; i < nTimeSteps; i++)
                {
                    m_cuda.channel_copy(bSrc.count(), m_nNumSamples, 1, nTimeSteps, nInnerNum, i, bDst.mutable_gpu_data, bSrc.gpu_data, DIR.BWD);
                }
            }
        }

        /// <summary>
        /// This method converts a static_signal (non-temporal tensor) of shape [num_samples x num_features]
        /// and replicates it along time for 'time_step' number of times to create a new shape
        /// [num_samples x time_steps x num_features]
        /// </summary>
        /// <param name="bSrc">Specifies the source blob</param>
        /// <param name="bDst">Specifies the destination blob.</param>
        /// <param name="nTimeSteps">Specifies the time steps.</param>
        private void replicate_along_time_bwd(Blob<T> bSrc, Blob<T> bDst, int nTimeSteps)
        {
            int nInnerNum = bSrc.count(1);
            m_cuda.channel_copy(bSrc.count(), m_nNumSamples, 1, nTimeSteps, nInnerNum, 0, bDst.mutable_gpu_diff, bSrc.gpu_diff, DIR.FWD);

            for (int i = 0; i < nTimeSteps; i++)
            {
                m_cuda.channel_copy(bSrc.count(), m_nNumSamples, 1, nTimeSteps, nInnerNum, i, bDst.mutable_gpu_diff, m_blobWork.gpu_diff, DIR.FWD);
                m_cuda.add(bSrc.count(), bSrc.gpu_diff, m_blobWork.gpu_diff, bSrc.mutable_gpu_diff);
            }

            bSrc.scale_diff(1.0 / nTimeSteps);
        }

        private void stack_time_steps_along_batch(Blob<T> bSrc, Blob<T> bDst, bool? bDiff)
        {
            if (bDiff.HasValue)
                bDst.CopyFrom(bSrc, bDiff.Value, true);

            m_rgShape.Clear();
            m_rgShape.Add(bSrc.shape(0) * bSrc.shape(1));
            m_rgShape.Add(bSrc.count(2));
            bDst.Reshape(m_rgShape);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_mode = m_param.reshape_temporal_param.mode;

            if (m_mode == param.tft.ReshapeTemporalParameter.MODE.BEFORE)
            {
                m_nNumSamples = colBottom[0].num;
                m_nNumTemporalSteps = colBottom[0].channels;

                // replicate the selection signal along time
                if (colBottom.Count > 1)
                {
                    m_blobWork = new Blob<T>(m_cuda, m_log);
                    m_blobWork.Name = "work";

                    m_blobTimeDistributedContext = new Blob<T>(m_cuda, m_log);
                    m_blobTimeDistributedContext.Name = "time_distributed_context";

                    replicate_along_time_fwd(colBottom[1], m_blobTimeDistributedContext, m_nNumTemporalSteps, true);
                    stack_time_steps_along_batch(m_blobTimeDistributedContext, colTop[1], null);
                    colTop[1].SetParameter("num_samples", m_nNumSamples);
                    colTop[1].SetParameter("num_temporal_steps", m_nNumTemporalSteps);
                }

                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch(colBottom[0], colTop[0], null);
                colTop[0].SetParameter("num_samples", m_nNumSamples);
                colTop[0].SetParameter("num_temporal_steps", m_nNumTemporalSteps);
            }
            else
            {
                m_nNumSamples = (int)colBottom[0].GetParameter("num_samples");
                m_nNumTemporalSteps = (int)colBottom[0].GetParameter("num_temporal_steps");

                int nCount = colBottom[0].count();
                int nDim = m_nNumSamples * m_nNumTemporalSteps;
                m_rgShape.Clear();
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(m_nNumTemporalSteps);
                m_rgShape.Add(nCount / nDim);
                colTop[0].Reshape(m_rgShape);

                if (colTop.Count > 1)
                {
                    nCount = colBottom[1].count();
                    m_rgShape[2] = nCount / nDim;
                    colTop[1].Reshape(m_rgShape);
                }
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_mode == ReshapeTemporalParameter.MODE.BEFORE)
            {
                // replicate the selection signal along time
                if (colBottom.Count > 1)
                {
                    replicate_along_time_fwd(colBottom[1], m_blobTimeDistributedContext, m_nNumTemporalSteps, true);
                    m_blobWork.ReshapeLike(colBottom[1]);
                    stack_time_steps_along_batch(m_blobTimeDistributedContext, colTop[1], null);
                }

                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch(colBottom[0], colTop[0], null);
            }
            else
            {
                int nCount = colBottom[0].count();
                int nDim = m_nNumSamples * m_nNumTemporalSteps;
                m_rgShape.Clear();
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(m_nNumTemporalSteps);
                m_rgShape.Add(nCount / nDim);
                colTop[0].Reshape(m_rgShape);

                if (colTop.Count > 1)
                {
                    nCount = colBottom[1].count();
                    m_rgShape[2] = nCount / nDim;
                    colTop[1].Reshape(m_rgShape);
                }
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_mode == ReshapeTemporalParameter.MODE.BEFORE)
            {
                if (colBottom.Count > 1)
                {
                    // replicate the selection signal along time
                    replicate_along_time_fwd(colBottom[1], m_blobTimeDistributedContext, m_nNumTemporalSteps);
                    stack_time_steps_along_batch(m_blobTimeDistributedContext, colTop[1], false);
                }

                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch(colBottom[0], colTop[0], false);
            }
            else
            {
                int nCount = colBottom[0].count();
                int nDim = m_nNumSamples * m_nNumTemporalSteps;
                m_rgShape.Clear();
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(m_nNumTemporalSteps);
                m_rgShape.Add(nCount / nDim);
                colTop[0].Reshape(m_rgShape);
                colTop[0].CopyFrom(colBottom[0], false, false, 0, true);

                if (colTop.Count > 1)
                {
                    nCount = colTop[1].count();
                    m_rgShape[2] = nCount / nDim;
                    colTop[1].Reshape(m_rgShape);
                    colTop[1].CopyFrom(colBottom[1], false, false, 0, true);
                }
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the stacked embedding numeric and categorical value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$;  
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (m_mode == ReshapeTemporalParameter.MODE.BEFORE)
            {
                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch(colBottom[0], colTop[0], true);

                // replicate the static selection signal along time
                if (colBottom.Count > 1)
                {
                    stack_time_steps_along_batch(m_blobTimeDistributedContext, colTop[1], true);
                    replicate_along_time_bwd(colBottom[1], m_blobTimeDistributedContext, m_nNumTemporalSteps);
                }
            }
            else
            {
                colBottom[0].CopyFrom(colTop[0], true, false, 0, true);
                if (colBottom.Count > 1)
                    colBottom[1].CopyFrom(colTop[1], true, false, 0, true);
            }
        }
    }
}
