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
    /// The NHitsBlockLayer performs the NHiTS Block layer functionality used by the N-HiTS model.
    /// </summary>
    /// <remarks>
    /// This layer performs the Pooling, FC processing and Linear backcast and forecast operations.
    /// 
    /// @see [Understanding N-HiTS Time Series Prediction](https://www.signalpop.com/2024/05/29/n-hits/) by Brown, 2024, SignalPop
    /// @see [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886) by Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, and Artur Dubrawski, 2022, arXiv:2201.12886.
    /// @see [Darts: User-Friendly Modern Machine Learning for Time Series](https://jmlr.org/papers/v23/21-1177.html) by Julien Herzen, Francesco Lässig, Samuele Giuliano Piazzetta, Thomas Neuer, Léo Tafti, Guillaume Raille, Tomas Van Pottelbergh, Marek Pasieka, Andrzej Skrodzki, Nicolas Huguenin, Maxime Dumonal, Jan Kościsz, Dennis Bader, Frédérick Gusset, Mounir Benheddi, Camila Williamson, Michal Kosinski, Matej Petrik, and Gaël Grosch, 2022, JMLR
    /// @see [Github - unit8co/darts](https://github.com/unit8co/darts) by unit8co, 2022, GitHub.
    /// 
    /// WORK IN PROGRESS.
    /// </remarks>
    public class NHitsBlockLayer<T> : Layer<T>
    {
        Blob<T> m_blobPool = null;
        Layer<T> m_layerPool = null;
        BlobCollection<T> m_colFc = new BlobCollection<T>();
        List<Layer<T>> m_rgLayerFc = new List<Layer<T>>();
        Blob<T> m_blobFc = null;
        Blob<T> m_blobThetaBc = null;
        Layer<T> m_layerBackcast = null;
        Blob<T> m_blobThetaFc = null;
        Layer<T> m_layerForecast = null;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        int m_nNumCovariates = 1;
        int[] m_rgShape = new int[] { 1, 1, 1, 1 };

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public NHitsBlockLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.NHITS_BLOCK;

            m_blobPool = new Blob<T>(cuda, log);
            m_blobPool.Name = m_param.name + " pool";

            for (int i=0; i<p.nhits_block_param.num_layers; i++)
            {
                Blob<T> blob = new Blob<T>(cuda, log);
                blob.Name = m_param.name + " fc" + i.ToString();
                m_colFc.Add(blob);
            }

            m_blobFc = new Blob<T>(cuda, log);
            m_blobFc.Name = m_param.name + " fc";

            m_blobThetaBc = new Blob<T>(cuda, log);
            m_blobThetaBc.Name = m_param.name + " theta_bc";

            m_blobThetaFc = new Blob<T>(cuda, log);
            m_blobThetaFc.Name = m_param.name + " theta_fc";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobPool);
            dispose(ref m_layerPool);

            m_colFc.Dispose();
            foreach (Layer<T> layer in m_rgLayerFc)
            {
                layer.Dispose();
            }

            m_rgLayerFc.Clear();
            dispose(ref m_blobFc);

            dispose(ref m_blobThetaBc);
            dispose(ref m_layerBackcast);

            dispose(ref m_blobThetaFc);
            dispose(ref m_layerForecast);

            base.dispose();
        }

        private void addBtmTop(Blob<T> btm, Blob<T> top)
        {
            m_colBtm.Clear();
            m_colBtm.Add(btm);
            m_colTop.Clear();
            m_colTop.Add(top);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobPool);
            for (int i = 0; i < m_colFc.Count; i++)
            {
                col.Add(m_colFc[i]);
            }
            col.Add(m_blobFc);
            col.Add(m_blobThetaBc);
            col.Add(m_blobThetaFc);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: y
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        private int[] unsqueeze(Blob<T> blob)
        {
            for (int i = 0; i < blob.shape().Count; i++)
            {
                m_rgShape[i] = blob.shape(i);
            }
            blob.Reshape(blob.shape(0), 1, blob.count(1), 1);
            return m_rgShape;
        }

        private void squeeze(Blob<T> blob)
        {
            blob.Reshape(blob.shape(0), blob.count(1), 1, 1);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nNumCovariates = colBottom[0].count(1) / layer_param.nhits_block_param.num_input_chunks;
            if (m_nNumCovariates < 1)
                m_log.FAIL("The number of covariates must be greater than 0.  The bottom[0] must have the shape (N, T, C, 1) where N = batch, T = time steps, C = covariates.");

            if (colBottom[0].width != 1)
                m_log.FAIL("The width of the bottom[0] must be 1.");

            LayerParameter pool = new LayerParameter(LayerParameter.LayerType.POOLING, "pool", m_phase);
            pool.pooling_param.Copy(layer_param.pooling_param);
            m_layerPool = Layer<T>.Create(m_cuda, m_log, pool, null);

            addBtmTop(colBottom[0], m_blobPool);
            m_layerPool.Setup(m_colBtm, m_colTop);

            Blob<T> blobBtm = m_blobPool;

            for (int i = 0; i < layer_param.nhits_block_param.num_layers; i++)
            {
                LayerParameter fc = new LayerParameter(LayerParameter.LayerType.FC, "fc" + i.ToString(), m_phase);
                fc.fc_param.Copy(layer_param.fc_param);
                Layer<T> fcl = Layer<T>.Create(m_cuda, m_log, fc, null);

                addBtmTop(blobBtm, m_colFc[i]);
                fcl.Setup(m_colBtm, m_colTop);
                blobBtm = m_colFc[i];

                m_rgLayerFc.Add(fcl);
            }

            LayerParameter backcast = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "backcast", m_phase);
            backcast.inner_product_param.axis = 2;
            backcast.inner_product_param.num_output = (uint)(layer_param.nhits_block_param.num_input_chunks / layer_param.nhits_block_param.downsample_size);
            m_layerBackcast = Layer<T>.Create(m_cuda, m_log, backcast, null);

            addBtmTop(blobBtm, m_blobThetaBc);
            m_layerBackcast.Setup(m_colBtm, m_colTop);
            blobBtm = m_blobThetaBc;

            LayerParameter forecast = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "forecast", m_phase);
            forecast.inner_product_param.axis = 2;
            forecast.inner_product_param.num_output = (uint)(layer_param.nhits_block_param.num_output_chunks / layer_param.nhits_block_param.downsample_size);
            m_layerForecast = Layer<T>.Create(m_cuda, m_log, forecast, null);

            addBtmTop(blobBtm, m_blobThetaFc);
            m_layerForecast.Setup(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nNumCovariates = colBottom[0].count(1) / layer_param.nhits_block_param.num_input_chunks;
            if (m_nNumCovariates < 1)
                m_log.FAIL("The number of covariates must be greater than 0.  The bottom[0] must have the shape (N, T, C, 1) where N = batch, T = time steps, C = covariates.");

            addBtmTop(colBottom[0], m_blobPool);
            m_layerPool.Reshape(m_colBtm, m_colTop);

            Blob<T> blobBtm = m_blobPool;

            for (int i = 0; i < layer_param.nhits_block_param.num_layers; i++)
            {
                addBtmTop(blobBtm, m_colFc[i]);
                m_rgLayerFc[i].Reshape(m_colBtm, m_colTop);
                blobBtm = m_colFc[i];
            }
            m_blobFc.ReshapeLike(blobBtm);

            addBtmTop(blobBtm, m_blobThetaBc);
            m_layerBackcast.Reshape(m_colBtm, m_colTop);
            blobBtm = m_blobThetaBc;

            addBtmTop(blobBtm, m_blobThetaFc);
            m_layerForecast.Reshape(m_colBtm, m_colTop);

            colTop[0].ReshapeLike(colBottom[0]);

            if (layer_param.nhits_block_param.data_order == param.ts.NHitsBlockParameter.DATA_ORDER.NTC)
                colTop[1].Reshape(blobBtm.num, layer_param.nhits_block_param.num_output_chunks, m_nNumCovariates, 1);
            else
                colTop[1].Reshape(blobBtm.num, m_nNumCovariates, layer_param.nhits_block_param.num_output_chunks, 1);
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
            // Process the Pooling layer (this creates the frequency domain representation)
            addBtmTop(colBottom[0], m_blobPool);
            m_layerPool.Forward(m_colBtm, m_colTop);

            // Process the FC layers
            Blob<T> blobBtm = m_blobPool;
            for (int i = 0; i < layer_param.nhits_block_param.num_layers; i++)
            {
                addBtmTop(blobBtm, m_colFc[i]);
                m_rgLayerFc[i].Forward(m_colBtm, m_colTop);
                blobBtm = m_colFc[i];
            }
            m_blobFc.CopyFrom(blobBtm);

            // Process the Linear backcast and forecast layers
            addBtmTop(blobBtm, m_blobThetaBc);
            m_layerBackcast.Forward(m_colBtm, m_colTop);
            blobBtm = m_blobThetaBc;

            // Process the Linear forecast layer
            addBtmTop(blobBtm, m_blobThetaFc);
            m_layerForecast.Forward(m_colBtm, m_colTop);

            // Interpolate the backcast and forecast results to the original size
            int nCountBc = m_blobThetaBc.count(1);
            int nCountTop0 = colTop[0].count(1);
            int nCountFc = m_blobThetaFc.count(1);
            int nCountTop1 = colTop[1].count(1);

            m_cuda.channel_interpolate_linear(colTop[0].count(), colTop[0].num, 1, nCountBc, nCountTop0, m_blobThetaBc.gpu_data, colTop[0].mutable_gpu_data, DIR.FWD);
            m_cuda.channel_interpolate_linear(colTop[1].count(), colTop[1].num, 1, nCountFc, nCountTop1, m_blobThetaFc.gpu_data, colTop[1].mutable_gpu_data, DIR.FWD);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     the inputs @f$ x @f$;  
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // Interpolate backward the backcast and forecast results from the original size
            int nCountBc = m_blobThetaBc.count(1);
            int nCountTop0 = colTop[0].count(1);
            int nCountFc = m_blobThetaFc.count(1);
            int nCountTop1 = colTop[1].count(1);

            m_cuda.channel_interpolate_linear(colTop[0].count(), colTop[0].num, 1, nCountBc, nCountTop0, m_blobThetaBc.mutable_gpu_diff, colTop[0].gpu_diff, DIR.BWD);
            m_cuda.channel_interpolate_linear(colTop[1].count(), colTop[1].num, 1, nCountFc, nCountTop1, m_blobThetaFc.mutable_gpu_diff, colTop[1].gpu_diff, DIR.BWD);

            // Process the Linear forecast layer
            addBtmTop(m_blobFc, m_blobThetaFc);
            m_layerForecast.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Process the Linear backcast layer
            addBtmTop(m_colFc[m_colFc.Count-1], m_blobThetaBc);
            m_layerBackcast.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Add the diff from both backcast and forecast to the last FC layer
            m_cuda.add(m_blobFc.count(), m_blobFc.gpu_diff, m_colFc[m_colFc.Count - 1].gpu_diff, m_colFc[m_colFc.Count - 1].mutable_gpu_diff);

            // Process the FC layers
            for (int i = m_colFc.Count - 1; i >= 0; i--)
            {
                Blob<T> blobBtm = (i == 0) ? m_blobPool : m_colFc[i - 1];
                addBtmTop(blobBtm, m_colFc[i]);
                m_rgLayerFc[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            // Process the Pooling layer
            addBtmTop(colBottom[0], m_blobPool);
            m_layerPool.Backward(m_colTop, rgbPropagateDown, m_colBtm);
        }
    }
}
