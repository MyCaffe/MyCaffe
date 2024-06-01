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

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public NHitsStackLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.NHITS_BLOCK;

            for (int i=0; i< p.nhits_stack_param.num_blocks; i++)
            {
                Blob<T> blob = new Blob<T>(cuda, log);
                blob.Name = "sres" + i.ToString();
                m_colStackRes.Add(blob);

                blob = new Blob<T>(cuda, log);
                blob.Name = "sfc" + i.ToString();
                m_colStackFc.Add(blob);
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
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

            for (int i = 0; i < m_colStackRes.Count; i++)
            {
                col.Add(m_colStackRes[i]);
                col.Add(m_colStackFc[i]);
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x
        /// </summary>
        public override int ExactNumBottomBlobs
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

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm1 = colBottom[0];
            Blob<T> blobBtm2 = colBottom[1];

            for (int i = 0; i < layer_param.nhits_stack_param.num_blocks; i++)
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.NHITS_BLOCK, "blk" + i.ToString(), m_phase);
                p.nhits_block_param.Copy(layer_param.nhits_block_param);
                p.pooling_param.Copy(layer_param.pooling_param);
                p.fc_param.Copy(layer_param.fc_param);

                Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(blobBtm2, m_colStackFc[i]);
                layer.Setup(m_colBtm, m_colTop);
                m_rgBlockLayers.Add(layer);

                blobBtm1 = m_colStackRes[i];
                blobBtm2 = m_colStackFc[i];
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm1 = colBottom[0];
            Blob<T> blobBtm2 = colBottom[1];

            for (int i = 0; i < layer_param.nhits_stack_param.num_blocks; i++)
            {
                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(blobBtm2, m_colStackFc[i]);
                m_rgBlockLayers[i].Reshape(m_colBtm, m_colTop);

                blobBtm1 = m_colStackRes[i];
                blobBtm2 = m_colStackFc[i];
            }
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
            Blob<T> blobBtm1 = colBottom[0];

            for (int i = 0; i < layer_param.nhits_stack_param.num_blocks; i++)
            {
                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(null, m_colStackFc[i]);
                m_rgBlockLayers[i].Forward(m_colBtm, m_colTop);

                blobBtm1 = m_colStackRes[i];

                // Accumulate the prediction values.
                m_cuda.add(colTop[1].count(), colTop[1].gpu_data, m_colStackFc[i].gpu_data, colTop[1].mutable_gpu_data);
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

            Blob<T> blobBtm1 = colBottom[0];

            for (int i = layer_param.nhits_stack_param.num_blocks - 1; i>=0; i--)
            {
                // Copy the prediction gradients.
                m_colStackFc[i].CopyFrom(colTop[1], true);

                addBtmTop(blobBtm1, m_colStackRes[i]);
                addBtmTop(null, m_colStackFc[i]);
                m_rgBlockLayers[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);

                blobBtm1 = m_colStackRes[i];
            }
        }
    }
}
