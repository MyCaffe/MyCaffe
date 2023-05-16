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
        int m_nNumSamples;
        int m_nNumRepeatCount;
        int m_nForcedRepeatCount;
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
            dispose(ref m_blobTimeDistributedContext);
            dispose(ref m_blobWork);
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
        /// Returns the exact number of required top (output) Blobs: temporal_selection_output, temporal_selection_output_clip, temporal_selection_wts
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// This method converts a static_signal (non-temporal tensor) of shape [num_samples x num_features]
        /// and replicates it along time for 'time_step' number of times to create a new shape
        /// [num_samples x time_steps x num_features]
        /// </summary>
        /// <param name="bBtm">Specifies the source blob</param>
        /// <param name="bTop">Specifies the destination blob.</param>
        /// <param name="nTimeSteps">Specifies the time steps.</param>
        /// <param name="bTemporalRepeat">Specifies to repeat along temporal axis, otherwise full blob is repeated.</param>
        /// <param name="bReshapeOnly">Specifies to reshape the destination only.</param>
        private void replicate_along_time_fwd(Blob<T> bBtm, Blob<T> bTop, int nTimeSteps, bool bTemporalRepeat, bool bReshapeOnly = false)
        {
            m_rgShape.Clear();

            if (bTemporalRepeat)
            {
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(nTimeSteps);
                m_rgShape.Add(bBtm.shape(1));
                bTop.Reshape(m_rgShape);

                if (!bReshapeOnly)
                {
                    int nInnerNum = bBtm.count(1);
                    for (int i = 0; i < nTimeSteps; i++)
                    {
                        m_cuda.channel_copy(bBtm.count(), m_nNumSamples, 1, nTimeSteps, nInnerNum, i, bTop.mutable_gpu_data, bBtm.gpu_data, DIR.BWD);
                    }
                }
            }
            else
            {
                m_rgShape.Add(nTimeSteps);
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(bBtm.shape(1));
                bTop.Reshape(m_rgShape);

                if (!bReshapeOnly)
                {
                    int nInnerNum = bBtm.count(1);
                    for (int i = 0; i < nTimeSteps; i++)
                    {
                        m_cuda.channel_copy(bBtm.count(), 1, 1, nTimeSteps, m_nNumSamples * nInnerNum, i, bTop.mutable_gpu_data, bBtm.gpu_data, DIR.BWD);
                    }
                }
            }
        }

        /// <summary>
        /// This method converts a static_signal (non-temporal tensor) of shape [num_samples x num_features]
        /// and replicates it along time for 'time_step' number of times to create a new shape
        /// [num_samples x time_steps x num_features]
        /// </summary>
        /// <param name="bBtm">Specifies the source blob</param>
        /// <param name="bTop">Specifies the destination blob.</param>
        /// <param name="nTimeSteps">Specifies the time steps.</param>
        /// <param name="bTemporalRepeat">Specifies to repeat along temporal axis, otherwise full blob is repeated.</param>
        private void replicate_along_time_bwd(Blob<T> bBtm, Blob<T> bTop, int nTimeSteps, bool bTemporalRepeat)
        {
            int nInnerNum = bBtm.count(1);

            if (bTemporalRepeat)    
                m_cuda.channel_sum(bTop.count(), m_nNumSamples, nTimeSteps, nInnerNum, bTop.gpu_diff, bBtm.mutable_gpu_diff, true);
            else
                m_cuda.channel_sum(bTop.count(), 1, nTimeSteps, m_nNumSamples * nInnerNum, bTop.gpu_diff, bBtm.mutable_gpu_diff, true);

            //m_cuda.channel_copy(bBtm.count(), m_nNumSamples, 1, nTimeSteps, nInnerNum, 0, bTop.mutable_gpu_diff, bBtm.gpu_diff, DIR.FWD);

            //for (int i = 1; i < nTimeSteps; i++)
            //{
            //    m_cuda.channel_add(bBtm.count(), m_nNumSamples, 1, nTimeSteps, nInnerNum, i, bTop.mutable_gpu_diff, bBtm.mutable_gpu_diff, DIR.FWD);
            //}
        }

        private void stack_time_steps_along_batch_fwd(Blob<T> bBtm, Blob<T> bTop, bool bResizeOnly = false)
        {
            if (!bResizeOnly)
                bTop.CopyFrom(bBtm, false, true);

            m_rgShape.Clear();
            m_rgShape.Add(bBtm.shape(0) * bBtm.shape(1));
            m_rgShape.Add(bBtm.count(2));
            bTop.Reshape(m_rgShape);
        }

        private void stack_time_steps_along_batch_bwd(Blob<T> bBtm, Blob<T> bTop)
        {
            bBtm.CopyFrom(bTop, true, false, 0, true);
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
                m_nForcedRepeatCount = m_param.reshape_temporal_param.forced_repeat_count;

                if (m_nForcedRepeatCount >= 0)
                    m_nNumRepeatCount = m_nForcedRepeatCount;
                else
                    m_nNumRepeatCount = colBottom[0].shape(1);

                // replicate the selection signal along time
                if (colBottom.Count > 1)
                {
                    m_blobWork = new Blob<T>(m_cuda, m_log);
                    m_blobWork.Name = "work";

                    if (m_nNumRepeatCount > 0)
                    {
                        m_blobTimeDistributedContext = new Blob<T>(m_cuda, m_log);
                        m_blobTimeDistributedContext.Name = m_param.name + ".tdctx";
                        replicate_along_time_fwd(colBottom[1], m_blobTimeDistributedContext, m_nNumRepeatCount, m_nForcedRepeatCount < 0, true);
                        stack_time_steps_along_batch_fwd(m_blobTimeDistributedContext, colTop[1], true);
                    }
                    else
                    {
                        stack_time_steps_along_batch_fwd(colBottom[1], colTop[1], true);
                    }

                    colTop[1].SetParameter("num_samples", m_nNumSamples);
                    colTop[1].SetParameter("num_temporal_steps", m_nNumRepeatCount);
                    colTop[1].SetParameter("forced_temporal_steps", m_nForcedRepeatCount);
                }

                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch_fwd(colBottom[0], colTop[0], true);
                colTop[0].SetParameter("num_samples", m_nNumSamples);
                colTop[0].SetParameter("num_temporal_steps", colBottom[0].shape(1));
            }
            else
            {
                m_nNumSamples = (int)colBottom[0].GetParameter("num_samples");
                int nTemporalSteps = (int)colBottom[0].GetParameter("num_temporal_steps");

                int nCount = colBottom[0].count();
                int nDim = m_nNumSamples * nTemporalSteps;
                m_rgShape.Clear();
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(nTemporalSteps);
                m_rgShape.Add(nCount / nDim);
                colTop[0].Reshape(m_rgShape);

                int nIdx = 1;
                if (m_param.reshape_temporal_param.enable_clip_output)
                {
                    m_log.CHECK_GT(colTop.Count, nIdx, "There must be at least " + (nIdx + 1).ToString() + " tops for the enable clip output!");
                    m_rgShape.Clear();
                    m_rgShape.Add(m_nNumSamples);
                    m_rgShape.Add(nTemporalSteps);
                    colTop[nIdx].Reshape(m_rgShape);
                    nIdx++;
                }

                if (colBottom.Count > 1)
                {
                    m_nNumRepeatCount = (int)colBottom[1].GetParameter("num_temporal_steps");
                    m_nForcedRepeatCount = (int)colBottom[1].GetParameter("forced_temporal_steps");

                    if (m_param.reshape_temporal_param.enable_weight_output)
                    {
                        m_log.CHECK_GT(colTop.Count, nIdx, "There must be at least " + (nIdx + 1).ToString() + " tops for the enable clip output!");
                        nCount = colBottom[1].count();
                        m_rgShape.Clear();

                        if (m_nForcedRepeatCount >= 0)
                        {
                            m_rgShape.Add(m_nNumSamples);
                            m_rgShape.Add(nCount / nDim);
                        }
                        else
                        {
                            m_rgShape.Add(m_nNumSamples);
                            if (m_nNumRepeatCount > 0)
                                m_rgShape.Add(m_nNumRepeatCount);
                            m_rgShape.Add(nCount / nDim);
                        }

                        colTop[nIdx].Reshape(m_rgShape);
                    }
                }
            }

            setup_internal_blobs(m_colInternalBlobs);
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
                    if (m_nNumRepeatCount > 0)
                    {
                        replicate_along_time_fwd(colBottom[1], m_blobTimeDistributedContext, m_nNumRepeatCount, m_nForcedRepeatCount < 0, true);
                        m_blobWork.ReshapeLike(colBottom[1]);
                        stack_time_steps_along_batch_fwd(m_blobTimeDistributedContext, colTop[1], true);
                    }
                    else
                    {
                        stack_time_steps_along_batch_fwd(colBottom[1], colTop[1], true);
                    }
                }

                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch_fwd(colBottom[0], colTop[0], true);
            }
            else
            {
                int nTemporalSteps = (int)colBottom[0].GetParameter("num_temporal_steps");
                int nCount = colBottom[0].count();
                int nDim = m_nNumSamples * nTemporalSteps;
                m_rgShape.Clear();
                m_rgShape.Add(m_nNumSamples);
                m_rgShape.Add(nTemporalSteps);
                m_rgShape.Add(nCount / nDim);
                colTop[0].Reshape(m_rgShape);

                int nIdx = 1;
                if (m_param.reshape_temporal_param.enable_clip_output)
                {
                    m_log.CHECK_GT(colTop.Count, nIdx, "There must be at least " + (nIdx + 1).ToString() + " tops for the enable clip output!");
                    m_rgShape.Clear();
                    m_rgShape.Add(m_nNumSamples);
                    m_rgShape.Add(nTemporalSteps);
                    colTop[nIdx].Reshape(m_rgShape);
                    nIdx++;
                }

                if (colBottom.Count > 1)
                {
                    m_nNumRepeatCount = (int)colBottom[1].GetParameter("num_temporal_steps");
                    m_nForcedRepeatCount = (int)colBottom[1].GetParameter("forced_temporal_steps");

                    if (m_param.reshape_temporal_param.enable_weight_output)
                    {
                        m_log.CHECK_GT(colTop.Count, nIdx, "There must be at least " + (nIdx + 1).ToString() + " tops for the enable clip output!");
                        nCount = colBottom[1].count();
                        m_rgShape.Clear();

                        if (m_nForcedRepeatCount >= 0)
                        {
                            m_rgShape.Add(m_nNumSamples);
                            m_rgShape.Add(nCount / nDim);
                        }
                        else
                        {
                            m_rgShape.Add(m_nNumSamples);
                            if (m_nNumRepeatCount > 0)
                                m_rgShape.Add(m_nNumRepeatCount);
                            m_rgShape.Add(nCount / nDim);
                        }

                        colTop[nIdx].Reshape(m_rgShape);
                    }
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
                    if (m_nNumRepeatCount > 0)
                    {
                        replicate_along_time_fwd(colBottom[1], m_blobTimeDistributedContext, m_nNumRepeatCount, m_nForcedRepeatCount < 0);
                        stack_time_steps_along_batch_fwd(m_blobTimeDistributedContext, colTop[1]);
                    }
                    else
                    {
                        stack_time_steps_along_batch_fwd(colBottom[1], colTop[1]);
                    }
                }

                // Apply the same selection module on all timesteps by stacking the time dimension with the batch dimension
                stack_time_steps_along_batch_fwd(colBottom[0], colTop[0]);
            }
            else
            {
                colTop[0].CopyFrom(colBottom[0], false, false, 0, true);

                int nIdx = 1;
                if (m_param.reshape_temporal_param.enable_clip_output)
                {
                    colTop[nIdx].SetData(1);
                    nIdx++;
                }

                if (m_param.reshape_temporal_param.enable_weight_output)
                {
                    colTop[nIdx].CopyFrom(colBottom[1], false, false, 0, true);
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
                stack_time_steps_along_batch_bwd(colBottom[0], colTop[0]);

                // replicate the static selection signal along time
                if (colBottom.Count > 1)
                {
                    if (m_nNumRepeatCount > 0)
                    {
                        stack_time_steps_along_batch_bwd(m_blobTimeDistributedContext, colTop[1]);
                        replicate_along_time_bwd(colBottom[1], m_blobTimeDistributedContext, m_nNumRepeatCount, m_nForcedRepeatCount < 0);
                    }
                    else
                    {
                        stack_time_steps_along_batch_bwd(colBottom[1], colTop[1]);
                    }
                }
            }
            else
            {
                colBottom[0].CopyFrom(colTop[0], true, false, 0, true);
                int nIdx = 1;

                if (m_param.reshape_temporal_param.enable_clip_output)
                    nIdx++;

                if (colBottom.Count > 1)
                    colBottom[1].CopyFrom(colTop[nIdx], true, false, 0, true);
            }
        }
    }
}
