using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.ts
{
    /// <summary>
    /// The NHitsStackLayer performs the NHiTS Stack layer functionality used by the N-HiTS model.
    /// </summary>
    /// <remarks>
    /// This layer performs the Block processing and prediction accumulation.
    /// 
    /// @see [Understanding N-HiTS Time Series Prediction](https://www.signalpop.com/2024/05/29/n-hits/) by Brown, 2024, SignalPop
    /// @see [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886) by Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, and Artur Dubrawski, 2022, arXiv:2201.12886.
    /// @see [Darts: User-Friendly Modern Machine Learning for Time Series](https://jmlr.org/papers/v23/21-1177.html) by Julien Herzen, Francesco Lässig, Samuele Giuliano Piazzetta, Thomas Neuer, Léo Tafti, Guillaume Raille, Tomas Van Pottelbergh, Marek Pasieka, Andrzej Skrodzki, Nicolas Huguenin, Maxime Dumonal, Jan Kościsz, Dennis Bader, Frédérick Gusset, Mounir Benheddi, Camila Williamson, Michal Kosinski, Matej Petrik, and Gaël Grosch, 2022, JMLR
    /// @see [Github - unit8co/darts](https://github.com/unit8co/darts) by unit8co, 2022, GitHub.
    /// 
    /// WORK IN PROGRESS.
    /// </remarks>
    public class NHitsStackLayer<T> : Layer<T>
    {
        BlobCollection<T> m_colStackRes = new BlobCollection<T>();
        BlobCollection<T> m_colStackFc = new BlobCollection<T>();
        List<Layer<T>> m_rgBlockLayers = new List<Layer<T>>();
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        Blob<T> m_blobWork;
        Blob<T> m_blobBtmRes = null;
        Blob<T> m_blobBtmFc = null;
        int m_nNum;
        int m_nTimeStepsBc;
        int m_nTimeStepsFc;
        int m_nCovariates;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public NHitsStackLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.NHITS_STACK;
            m_nTimeStepsBc = p.nhits_block_param.num_input_chunks;
            m_nTimeStepsFc = p.nhits_block_param.num_output_chunks;

            if (p.nhits_stack_param.data_order != param.ts.NHitsStackParameter.DATA_ORDER.NTC || p.nhits_stack_param.treat_covariates_separately)
            {
                m_blobBtmRes = new Blob<T>(cuda, log);
                m_blobBtmRes.Name = p.name + ".btm.res";
                m_blobBtmFc = new Blob<T>(cuda, log);
                m_blobBtmFc.Name = p.name + ".btm.fc";
            }

            for (int i=0; i< p.nhits_stack_param.num_blocks; i++)
            {
                Blob<T> blob = new Blob<T>(cuda, log);
                blob.Name = "sres" + i.ToString();
                m_colStackRes.Add(blob);

                blob = new Blob<T>(cuda, log);
                blob.Name = "sfc" + i.ToString();
                m_colStackFc.Add(blob);
            }

            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = p.name + ".pred_grad_accum";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobBtmRes);
            dispose(ref m_blobBtmFc);

            m_colStackFc.Dispose();
            m_colStackFc.Clear();
            m_colStackRes.Dispose();
            m_colStackRes.Clear();

            foreach (Layer<T> layer in m_rgBlockLayers)
            {
                layer.Dispose();
            }

            base.dispose();
        }

        private void addBtmTop(Blob<T> btm, Blob<T> top, bool bClear = true)
        {
            if (bClear)
                m_colBtm.Clear();
            if (btm != null)
                m_colBtm.Add(btm);
            if (bClear)
                m_colTop.Clear();
            m_colTop.Add(top);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobBtmRes);
            col.Add(m_blobBtmFc);

            for (int i = 0; i < m_colStackRes.Count; i++)
            {
                col.Add(m_colStackRes[i]);
                col.Add(m_colStackFc[i]);
            }

            col.Add(m_blobWork);
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: x
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the max number of required bottom (input) Blobs: x, y_hat (accumulated predictions)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: y
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        private List<int> calculate_sizes(int nChunks, int nStacks)
        {
            int nMaxV = nChunks / 2;

            if (nStacks == 1)
                return new List<int>() { nMaxV };   

            double dfStart = Math.Log10(1);
            double dfEnd = Math.Log10(nMaxV);
            double dfStep = (dfEnd - dfStart) / (nStacks - 1);

            List<int> rgStep = new List<int>();
            for (int i = 0; i < nStacks; i++)
            {
                double dfLogStep = (dfStart + i * dfStep);
                double dfStep1 = Math.Pow(10, dfLogStep);
                int nVal = (int)(nMaxV / dfStep1);
                rgStep.Add(Math.Max(nVal, 1));
            }

            return rgStep;
        }   

        private void calculate_pooling_downsampling(LayerParameter p, BlobCollection<T> colBtm)
        {
            if (p.nhits_stack_param.auto_pooling_downsample_index >= p.nhits_stack_param.num_stacks)
                throw new Exception("The auto_pooling_downsample_index must be less than the num_stacks.");

            List<int> rgKernelSizes = calculate_sizes(p.nhits_block_param.num_input_chunks, p.nhits_stack_param.num_stacks);
            List<int> rgDownsampleSizes = calculate_sizes(p.nhits_block_param.num_output_chunks, p.nhits_stack_param.num_stacks);

            p.pooling_param.kernel_h = (uint)rgKernelSizes[p.nhits_stack_param.auto_pooling_downsample_index];
            p.pooling_param.kernel_w = 1;
            p.pooling_param.stride_h = (uint)rgKernelSizes[p.nhits_stack_param.auto_pooling_downsample_index];
            p.pooling_param.stride_w = 1;
            p.nhits_block_param.downsample_size = rgDownsampleSizes[p.nhits_stack_param.auto_pooling_downsample_index];
        }

        private Blob<T> arrange_inputs(BlobCollection<T> col, int nIdx, Blob<T> blobBtmT, int nTimeSteps, DIR dir, bool bResize = false)
        {
            if (col.Count <= nIdx)
                return null;

            Blob<T> blobBtm = col[nIdx];

            return arrange_inputs(blobBtm, blobBtmT, nTimeSteps, dir, bResize);
        }

        private Blob<T> arrange_inputs(Blob<T> blobBtm, Blob<T> blobBtmT, int nTimeSteps, DIR dir, bool bResize = false)
        {
            if (layer_param.nhits_stack_param.data_order == param.ts.NHitsStackParameter.DATA_ORDER.NTC)
            {
                if (bResize)
                {
                    m_nNum = blobBtm.num;
                    m_nCovariates = blobBtm.height;
                    m_log.CHECK_EQ(blobBtm.channels, nTimeSteps, "The number of channels in blob '" + blobBtm.Name + "' must be equal to the number of time steps.");

                    if (dir == DIR.FWD)
                    {
                        if (blobBtm.num_axes == 3)
                            blobBtm.UnsqueezeTo(4);
                        else
                            m_log.CHECK_EQ(blobBtm.width, 1, "The width must be 1 when the data order is NTC.");
                    }
                }

                // Treat inputs in similar manner to the original N-HiTS model.
                if (!layer_param.nhits_stack_param.treat_covariates_separately)
                {
                    return blobBtm;
                }
                else
                {
                    if (dir == DIR.FWD)
                    {
                        blobBtmT.Reshape(m_nNum, m_nCovariates, nTimeSteps, 1);
                        m_cuda.transposeHW(m_nNum, 1, nTimeSteps, m_nCovariates, blobBtm.gpu_data, blobBtmT.mutable_gpu_data);
                        return blobBtmT;
                    }
                    else
                    {
                        blobBtmT.Reshape(m_nNum, nTimeSteps, m_nCovariates, 1);
                        m_cuda.transposeHW(m_nNum, 1, m_nCovariates, nTimeSteps, blobBtm.gpu_diff, blobBtmT.mutable_gpu_diff);
                        return blobBtmT;
                    }
                }
            }
            else // NCT data ordering
            {
                if (bResize)
                {
                    m_nNum = blobBtm.num;
                    m_nCovariates = blobBtm.channels;
                    m_log.CHECK_EQ(blobBtm.height, nTimeSteps, "The number of height in blob '" + blobBtm.Name + "' must be equal to the number of time steps.");
                }

                if (!layer_param.nhits_stack_param.treat_covariates_separately)
                {
                    if (dir == DIR.FWD)
                    {
                        blobBtmT.Reshape(m_nNum, nTimeSteps, m_nCovariates, 1);
                        m_cuda.transposeHW(m_nNum, 1, m_nCovariates, nTimeSteps, blobBtm.gpu_data, blobBtmT.mutable_gpu_data);
                        return blobBtmT;
                    }
                    else
                    {
                        blobBtmT.Reshape(m_nNum, m_nCovariates, nTimeSteps, 1);
                        m_cuda.transposeHW(m_nNum, 1, nTimeSteps, m_nCovariates, blobBtm.gpu_diff, blobBtmT.mutable_gpu_diff);
                        return blobBtmT;
                    }
                }
                else
                {
                    return blobBtm;
                }
            }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm1 = arrange_inputs(colBottom, 0, m_blobBtmRes, m_nTimeStepsBc, DIR.FWD, true);

            for (int i = 0; i < layer_param.nhits_stack_param.num_blocks; i++)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.NHITS_BLOCK, "blk" + i.ToString(), m_phase);
                p.nhits_block_param.Copy(layer_param.nhits_block_param);
                p.pooling_param.Copy(layer_param.pooling_param);
                p.fc_param.Copy(layer_param.fc_param);

                if (layer_param.nhits_stack_param.auto_pooling_downsample_index >= 0)
                    calculate_pooling_downsampling(layer_param, colBottom);

                Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(null, m_colStackFc[i], false);
                layer.Setup(m_colBtm, m_colTop);
                m_rgBlockLayers.Add(layer);

                blobBtm1 = m_colStackRes[i];
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm1 = arrange_inputs(colBottom, 0, m_blobBtmRes, m_nTimeStepsBc, DIR.FWD, true);

            for (int i = 0; i < layer_param.nhits_stack_param.num_blocks; i++)
            {
                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(null, m_colStackFc[i], false);
                m_rgBlockLayers[i].Reshape(m_colBtm, m_colTop);

                blobBtm1 = m_colStackRes[i];
            }

            m_blobWork.ReshapeLike(colBottom[1]);

            colTop[0].ReshapeLike(colBottom[0]);

            if (layer_param.nhits_stack_param.data_order == param.ts.NHitsStackParameter.DATA_ORDER.NTC)
                colTop[1].Reshape(m_nNum, m_nTimeStepsFc, m_nCovariates, 1);
            else
                colTop[1].Reshape(m_nNum, m_nCovariates, m_nTimeStepsFc, 1);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times T \times H \times 1) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 2))
        ///  -# @f$ (N \times T \times 1 \times 1 size) @f$
        ///     the computed backcast outputs @f$ x_hat @f$
        ///  -# @f$ (N \times T \times H \times 1) @f$
        ///     the computed forecast outputs @f$ y_hat @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm1 = arrange_inputs(colBottom, 0, m_blobBtmRes, m_nTimeStepsBc, DIR.FWD);
            Blob<T> blobBtm2;

            if (colBottom.Count > 1)
                colTop[1].CopyFrom(colBottom[1]);
            else
                colTop[1].SetData(0);

            for (int i = 0; i < layer_param.nhits_stack_param.num_blocks; i++)
            {
                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(null, m_colStackFc[i], false);
                m_rgBlockLayers[i].Forward(m_colBtm, m_colTop);

                blobBtm1 = m_colStackRes[i];
                blobBtm2 = m_colStackFc[i];

                // Accumulate the prediction values.
                m_cuda.add(colTop[1].count(), colTop[1].gpu_data, blobBtm2.gpu_data, colTop[1].mutable_gpu_data);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 2), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial x_hat} @f$
        ///     with respect to computed outputs @f$ x_hat @f$
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y_hat} @f$
        ///     with respect to computed outputs @f$ y_hat @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     the x inputs @f$ x @f$;  
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     the y prediction accumulator inputs @f$ x @f$;  
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            if (colBottom.Count > 1)
                colBottom[1].CopyFrom(colTop[1], true);

            Blob<T> blobBtm1 = colBottom[0];

            for (int i = layer_param.nhits_stack_param.num_blocks - 1; i>=0; i--)
            {
                // Copy the prediction gradients.
                m_colStackFc[i].CopyFrom(colTop[1], true);

                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(m_blobWork, m_colStackFc[i], false);
                m_rgBlockLayers[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);

                // Accumulate the gradients.
                if (colBottom.Count > 1)
                    m_cuda.add(colBottom[1].count(), colBottom[1].gpu_diff, m_blobWork.gpu_diff, colBottom[1].mutable_gpu_diff);

                blobBtm1 = m_colStackRes[i];
            }

            Blob<T> blobBtm = arrange_inputs(blobBtm1, m_blobBtmRes, m_nTimeStepsBc, DIR.BWD);
            colBottom[0].CopyFrom(blobBtm, true);
        }
    }
}
