using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers.beta;
using System.Diagnostics;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The SpatialAttentionLayer provides attention for spatial CNN type models.
    /// </summary>
    /// <remarks>
    /// @see [RFAConv: Innovating Spatial Attention and Standard Convolutional Operation](https://arxiv.org/abs/2304.03198) by Xin Zhang, Chen Liu, Degang Yang, Tingting Song, Yichen Ye, Ke Li, Yingze Song, 2023, arXiv:2304.03198
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SpatialAttentionLayer<T> : Layer<T>
    {
        Blob<T> m_blobAve;
        Blob<T> m_blobMax;
        Blob<T> m_blobFc1Ave;
        Blob<T> m_blobFc1Max;
        Blob<T> m_blobFc2Ave;
        Blob<T> m_blobFc2Max;
        Blob<T> m_blobAttention;
        Blob<T> m_blobBtm;

        Layer<T> m_ave_conv;
        Layer<T> m_max_conv;
        Layer<T> m_fc1;
        Layer<T> m_fc2;
        Layer<T> m_activation;
        Layer<T> m_sigmoid;

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The SpatialAttentionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
        /// </param>
        public SpatialAttentionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SPATIAL_ATTENTION;

            m_blobAve = new Blob<T>(cuda, log);
            m_blobAve.Name = p.name + ".ave";
            m_blobMax = new Blob<T>(cuda, log);
            m_blobMax.Name = p.name + ".max";
            m_blobFc1Ave = new Blob<T>(cuda, log);
            m_blobFc1Ave.Name = p.name + ".ave.fc1";
            m_blobFc1Max = new Blob<T>(cuda, log);
            m_blobFc1Max.Name = p.name + ".max.fc1";
            m_blobFc2Ave = new Blob<T>(cuda, log);
            m_blobFc2Ave.Name = p.name + ".ave.fc2";
            m_blobFc2Max = new Blob<T>(cuda, log);
            m_blobFc2Max.Name = p.name + ".max.fc2";
            m_blobAttention = new Blob<T>(cuda, log);
            m_blobAttention.Name = p.name + ".attention";
            m_blobBtm = new Blob<T>(cuda, log);
            m_blobBtm.Name = p.name + ".btm";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobAve);
            dispose(ref m_blobMax);
            dispose(ref m_blobFc1Ave);
            dispose(ref m_blobFc1Max);
            dispose(ref m_blobFc2Ave);
            dispose(ref m_blobFc2Max);
            dispose(ref m_blobAttention);
            dispose(ref m_blobBtm);

            dispose(ref m_ave_conv);
            dispose(ref m_max_conv);
            dispose(ref m_fc1);
            dispose(ref m_fc2);
            dispose(ref m_activation);
            dispose(ref m_sigmoid);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobAve);
            col.Add(m_blobMax);
            col.Add(m_blobFc1Ave);
            col.Add(m_blobFc1Max);
            col.Add(m_blobFc2Ave);
            col.Add(m_blobFc2Max);
            col.Add(m_blobAttention);
            col.Add(m_blobBtm);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input, state (last ct), clip (1 on each input, 0 otherwise)
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: ip
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Re-initialize the parameters of the layer.
        /// </summary>
        /// <param name="target">Specifies the weights to target (e.g. weights, bias or both).</param>
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public override bool ReInitializeParameters(WEIGHT_TARGET target)
        {
            base.ReInitializeParameters(target);

            return true;
        }

        private void addInternal(Blob<T> bottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();
            m_colInternalBottom.Add(bottom);

            m_colInternalTop.Clear();
            m_colInternalTop.Add(top);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_ave_conv = new ConvolutionLayer<T>(m_cuda, m_log, m_param);
            addInternal(colBottom[0], m_blobAve);
            m_ave_conv.Setup(m_colInternalBottom, m_colInternalTop);
            blobs.Add(m_ave_conv.blobs);

            m_max_conv = new ConvolutionLayer<T>(m_cuda, m_log, m_param);
            addInternal(colBottom[0], m_blobMax);
            m_max_conv.Setup(m_colInternalBottom, m_colInternalTop);
            blobs.Add(m_max_conv.blobs);

            m_fc1 = new ConvolutionLayer<T>(m_cuda, m_log, m_param);
            addInternal(m_blobAve, m_blobFc1Ave);
            m_fc1.Setup(m_colInternalBottom, m_colInternalTop);
            blobs.Add(m_fc1.blobs);

            switch (layer_param.spatial_attention_param.activation)
            {
                case SpatialAttentionParameter.ACTIVATION.RELU:
                    m_activation = new ReLULayer<T>(m_cuda, m_log, m_param);
                    break;

                default:
                    throw new Exception("Unknown activation type.");
            }

            addInternal(m_blobFc1Ave, m_blobFc1Ave);
            m_activation.Setup(m_colInternalBottom, m_colInternalTop);

            m_fc2 = new ConvolutionLayer<T>(m_cuda, m_log, m_param);
            addInternal(m_blobFc1Ave, m_blobFc2Ave);
            m_fc2.Setup(m_colInternalBottom, m_colInternalTop);
            blobs.Add(m_fc2.blobs); 

            m_sigmoid = new SigmoidLayer<T>(m_cuda, m_log, m_param);
            addInternal(m_blobFc2Ave, m_blobAttention);
            m_sigmoid.Setup(m_colInternalBottom, m_colInternalTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!reshapeNeeded(colBottom, colTop))
                return;

            addInternal(colBottom[0], m_blobAve);
            m_ave_conv.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(colBottom[0], m_blobMax);
            m_ave_conv.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAve, m_blobFc1Ave);
            m_fc1.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc1Ave, m_blobFc1Ave);
            m_activation.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc1Ave, m_blobFc2Ave);
            m_fc2.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc2Ave, m_blobAttention);
            m_sigmoid.Reshape(m_colInternalBottom, m_colInternalTop);

            m_blobFc1Max.ReshapeLike(m_blobFc1Ave);
            m_blobFc2Max.ReshapeLike(m_blobFc2Ave);
            m_blobBtm.ReshapeLike(colBottom[0]);
            colTop[0].ReshapeLike(m_blobAttention);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times K \times H \times W) @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobBtm.CopyFrom(colBottom[0], false, false);

            // Average
            addInternal(colBottom[0], m_blobAve);
            m_ave_conv.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAve, m_blobFc1Ave);
            m_fc1.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc1Ave, m_blobFc1Ave);
            m_activation.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc1Ave, m_blobFc2Ave);
            m_fc2.Forward(m_colInternalBottom, m_colInternalTop);

            // Maximum
            addInternal(colBottom[0], m_blobMax);
            m_max_conv.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobMax, m_blobFc1Max);
            m_fc1.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc1Max, m_blobFc1Max);
            m_activation.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobFc1Max, m_blobFc2Max);
            m_fc2.Forward(m_colInternalBottom, m_colInternalTop);

            // Attention
            m_cuda.add(m_blobAttention.count(), m_blobFc2Ave.gpu_data, m_blobFc2Max.gpu_data, m_blobAttention.mutable_gpu_data);
            addInternal(m_blobAttention, m_blobAttention);
            m_sigmoid.Forward(m_colInternalBottom, m_colInternalTop);

            // Multiply
            m_cuda.muladd(colBottom[0].count(), colBottom[0].gpu_data, m_blobAttention.gpu_data, colTop[0].mutable_gpu_data, DIR.FWD);
        }

        /// <summary>
        /// Computes the loss error gradient w.r.t the outputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (N \times K \times H \times W) @f$.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            // Gradient with respect to state then data.
            if (rgbPropagateDown[0])
            {
                // Multiply
                m_cuda.muladd(colBottom[0].count(), colBottom[0].gpu_data, m_blobAttention.gpu_data, colTop[0].gpu_diff, DIR.BWD, colBottom[0].mutable_gpu_diff, m_blobAttention.mutable_gpu_diff);

                // Attention
                addInternal(m_blobAttention, m_blobAttention);
                m_sigmoid.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);
                m_blobFc2Ave.CopyFrom(m_blobAttention, true, false);
                m_blobFc2Max.CopyFrom(m_blobAttention, true, false);

                // Maximum
                addInternal(m_blobFc1Max, m_blobFc2Max);
                m_fc2.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);

                addInternal(m_blobFc1Max, m_blobFc1Max);
                m_activation.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);

                addInternal(m_blobMax, m_blobFc1Max);
                m_fc1.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);

                addInternal(m_blobBtm, m_blobMax);
                m_max_conv.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);
                m_cuda.add(colBottom[0].count(), m_blobBtm.gpu_diff, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);

                // Average
                addInternal(m_blobFc1Ave, m_blobFc2Ave);
                m_fc2.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);

                addInternal(m_blobFc1Ave, m_blobFc1Ave);
                m_activation.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);

                addInternal(m_blobAve, m_blobFc1Ave);
                m_fc1.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);

                addInternal(m_blobBtm, m_blobAve);
                m_ave_conv.Backward(m_colInternalTop, rgbPropagateDown, m_colInternalBottom);
                m_cuda.add(colBottom[0].count(), m_blobBtm.gpu_diff, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
