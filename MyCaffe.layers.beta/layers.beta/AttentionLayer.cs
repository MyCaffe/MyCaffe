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

namespace MyCaffe.layers
{
    /// <summary>
    /// The AttentionLayer provides focus for LSTM based encoder/decoder models.
    /// </summary>
    /// <remarks>
    /// @see [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, 2017, arXiv:1706:03762
    /// @see [Attention is All You Need](https://tzuruey.medium.com/attention-is-all-you-need-98d26aeb3517) by Jenny Ching, 2019, Medium
    /// @see [Attention is All You Need in Speech Separation] by Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, Jianyuan Zhong, 2020, arXiv:2010.13154
    /// @see [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, 2014, arXiv:1409.3215
    /// 
    /// The AttentionLayer implementation was inspired by the C# Seq2SeqLearn implementation by mashmawy for language translation,
    /// @see [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn) distributed under MIT license.
    /// 
    /// And also inspired by the C# ChatBot implementation by HectorPulido which uses Seq2SeqLearn
    /// @see [HectorPulido/Chatbot-seq2seq-C-](https://github.com/HectorPulido/Chatbot-seq2seq-C-) distributed under [MIT license](https://github.com/HectorPulido/Chatbot-seq2seq-C-/blob/master/LICENSE).
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AttentionLayer<T> : Layer<T>
    {
        Layer<T> m_transposeX = null;
        Layer<T> m_transposeClip = null;
        Layer<T> m_ipUa = null;
        Layer<T> m_ipWa = null;
        Layer<T> m_tanh = null;
        Layer<T> m_add1 = null;
        Layer<T> m_ipV = null;

        Blob<T> m_blobX = null;
        Blob<T> m_blobClip = null;
        Blob<T> m_blobX1 = null;
        Blob<T> m_blobState = null;
        Blob<T> m_blobUh = null;
        Blob<T> m_blobWc = null;
        Blob<T> m_blobFullWc = null;
        Blob<T> m_blobAddOutput = null;
        Blob<T> m_blobGG = null;
        Blob<T> m_blobAA = null;
        Blob<T> m_blobScale = null;
        Blob<T> m_blobSoftmax = null;
        Blob<T> m_blobFocusedInput = null;
        Blob<T> m_blobContext = null;
        Blob<T> m_blobWork = null;

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The AttentionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
        /// </param>
        public AttentionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ATTENTION;

            List<int> rgDimClip = new List<int>() { 1, 0 };
            LayerParameter transposeClipparam = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            transposeClipparam.transpose_param.dim = new List<int>(rgDimClip);

            m_transposeClip = new TransposeLayer<T>(cuda, log, transposeClipparam);

            LayerParameter ipUaParam = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ipUaParam.name = "ipUa";
            ipUaParam.inner_product_param.axis = 2;
            ipUaParam.inner_product_param.num_output = m_param.attention_param.dim;
            ipUaParam.inner_product_param.weight_filler = m_param.attention_param.weight_filler;            
            ipUaParam.inner_product_param.bias_filler = m_param.attention_param.bias_filler;

            m_ipUa = new InnerProductLayer<T>(cuda, log, ipUaParam);

            LayerParameter ipWaParam = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ipWaParam.name = "ipWa";
            ipWaParam.inner_product_param.axis = 1;
            ipWaParam.inner_product_param.num_output = m_param.attention_param.dim;
            ipWaParam.inner_product_param.weight_filler = m_param.attention_param.weight_filler;
            ipWaParam.inner_product_param.bias_filler = m_param.attention_param.bias_filler;

            m_ipWa = new InnerProductLayer<T>(cuda, log, ipWaParam);

            LayerParameter addParam = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            addParam.name = "add";
            addParam.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;

            m_add1 = new EltwiseLayer<T>(cuda, log, addParam);

            LayerParameter tanhParam = new LayerParameter(LayerParameter.LayerType.TANH);
            tanhParam.name = "tanh";
            tanhParam.tanh_param.engine = EngineParameter.Engine.CUDNN;

            m_tanh = new TanhLayer<T>(cuda, log, tanhParam);

            LayerParameter ipVParam = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ipVParam.name = "ipV";
            ipVParam.inner_product_param.axis = 2;
            ipVParam.inner_product_param.num_output = 1;
            ipVParam.inner_product_param.bias_term = false;
            ipVParam.inner_product_param.weight_filler = m_param.attention_param.weight_filler;

            m_ipV = new InnerProductLayer<T>(cuda, log, ipVParam);

            m_blobX = new Blob<T>(cuda, log);
            m_blobX.Name = "x";

            m_blobClip = new Blob<T>(cuda, log);
            m_blobClip.Name = "clip";

            m_blobX1 = new Blob<T>(cuda, log);
            m_blobX1.Name = "x1";

            m_blobState = new Blob<T>(cuda, log);
            m_blobState.Name = "state";

            m_blobUh = new Blob<T>(cuda, log);
            m_blobUh.Name = "Uh";

            m_blobWc = new Blob<T>(cuda, log);
            m_blobWc.Name = "Wc";

            m_blobFullWc = new Blob<T>(cuda, log);
            m_blobFullWc.Name = "Full Wc";

            m_blobAddOutput = new Blob<T>(cuda, log);
            m_blobAddOutput.Name = "addOut";

            m_blobGG = new Blob<T>(cuda, log);
            m_blobGG.Name = "gg";

            m_blobAA = new Blob<T>(cuda, log);
            m_blobAA.Name = "aa";

            m_blobScale = new Blob<T>(cuda, log, false);
            m_blobScale.Name = "scale";

            m_blobSoftmax = new Blob<T>(cuda, log);
            m_blobSoftmax.Name = "softmax";

            m_blobFocusedInput = new Blob<T>(cuda, log);
            m_blobFocusedInput.Name = "softmax_full";

            m_blobContext = new Blob<T>(cuda, log);
            m_blobContext.Name = "context";

            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = "work";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_transposeX);
            dispose(ref m_transposeClip);
            dispose(ref m_ipUa);
            dispose(ref m_ipWa);
            dispose(ref m_tanh);
            dispose(ref m_add1);
            dispose(ref m_ipV);

            dispose(ref m_blobState);
            dispose(ref m_blobX);
            dispose(ref m_blobClip);
            dispose(ref m_blobX1);
            dispose(ref m_blobUh);
            dispose(ref m_blobWc);
            dispose(ref m_blobFullWc);
            dispose(ref m_blobAddOutput);
            dispose(ref m_blobGG);
            dispose(ref m_blobAA);
            dispose(ref m_blobScale);
            dispose(ref m_blobSoftmax);
            dispose(ref m_blobFocusedInput);
            dispose(ref m_blobContext);
            dispose(ref m_blobWork);

            base.dispose();
        }

        private void dispose(ref Layer<T> l)
        {
            if (l != null)
            {
                l.Dispose();
                l = null;
            }
        }

        private void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobState);
                col.Add(m_blobUh);
                col.Add(m_blobWc);
                col.Add(m_blobFullWc);
                col.Add(m_blobAddOutput);
                col.Add(m_blobGG);
                col.Add(m_blobAA);
                col.Add(m_blobScale);
                col.Add(m_blobFocusedInput);
                col.Add(m_blobContext);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input, state (last ct), clip (1 on each input, 0 otherwise)
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 3; }
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

            m_ipUa.ReInitializeParameters(target);
            m_ipWa.ReInitializeParameters(target);

            return true;
        }

        private void addInternal(Blob<T> bottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();
            m_colInternalBottom.Add(bottom);

            m_colInternalTop.Clear();
            m_colInternalTop.Add(top);
        }

        private void addInternal(List<Blob<T>> rgBottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();

            for (int i=0; i<rgBottom.Count; i++)
            {
                m_colInternalBottom.Add(rgBottom[i]);
            }

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
            Blob<T> blobX = colBottom[0];
            Blob<T> blobCy = colBottom[1];
            Blob<T> blobClip = colBottom[2];

            m_log.CHECK_EQ(blobX.shape(1), 1, "Currently, only batch size = 1 is supported.");

            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);

            List<int> rgDimX = new List<int>() { 1, 0 };
            while (rgDimX.Count < colBottom[0].num_axes)
            {
                rgDimX.Add(rgDimX.Count);
            }

            LayerParameter transposeXparam = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            transposeXparam.transpose_param.dim = new List<int>(rgDimX);

            m_transposeX = new TransposeLayer<T>(m_cuda, m_log, transposeXparam);

            addInternal(blobX, m_blobX);
            m_transposeX.Setup(m_colInternalBottom, m_colInternalTop);
            m_blobX1.ReshapeLike(m_blobX);

            addInternal(m_blobX, m_blobUh);
            m_ipUa.Setup(m_colInternalBottom, m_colInternalTop);

            addInternal(blobClip, m_blobClip);
            m_transposeClip.Setup(m_colInternalBottom, m_colInternalTop);

            m_blobState.ReshapeLike(blobCy);

            addInternal(m_blobState, m_blobWc);
            m_ipWa.Setup(m_colInternalBottom, m_colInternalTop);

            m_blobFullWc.ReshapeLike(m_blobUh);

            addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
            m_add1.Setup(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAddOutput, m_blobGG);
            m_tanh.Setup(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobGG, m_blobAA);
            m_ipV.Setup(m_colInternalBottom, m_colInternalTop);

            List<int> rgFocusShape = Utility.Clone<int>(blobX.shape());
            rgFocusShape[0] = blobX.shape(1);
            rgFocusShape[1] = blobX.shape(0);
            m_blobFocusedInput.Reshape(rgFocusShape);

            List<int> rgContextShape = Utility.Clone<int>(blobX.shape());
            rgContextShape[0] = rgContextShape[1];
            rgContextShape[1] = 1;
            m_blobContext.Reshape(rgContextShape);

            List<int> rgTopShape = Utility.Clone<int>(m_blobContext.shape());
            rgTopShape[0] = m_blobContext.shape(1);
            rgTopShape[1] = m_blobContext.shape(0);
            colTop[0].Reshape(rgTopShape);

            blobs.Clear();

            foreach (Blob<T> blob in m_ipUa.blobs)
            {
                blobs.Add(blob);
            }

            foreach (Blob<T> blob in m_ipWa.blobs)
            {
                blobs.Add(blob);
            }

            // V
            blobs.Add(m_ipV.blobs[0]);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobX = colBottom[0];
            Blob<T> blobCy = colBottom[1];
            Blob<T> blobClip = colBottom[2];

            m_log.CHECK_EQ(blobClip.count(), blobX.count(0, 2), "The bottom[2] 'clip' must have shape T,B.");

            addInternal(blobX, m_blobX);
            m_transposeX.Reshape(m_colInternalBottom, m_colInternalTop);
            m_blobX1.ReshapeLike(m_blobX);

            addInternal(m_blobX, m_blobUh);
            m_ipUa.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(blobClip, m_blobClip);
            m_transposeClip.Reshape(m_colInternalBottom, m_colInternalTop);

            m_blobState.ReshapeLike(blobCy);

            addInternal(m_blobState, m_blobWc);
            m_ipWa.Reshape(m_colInternalBottom, m_colInternalTop);

            m_blobFullWc.ReshapeLike(m_blobUh);

            addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
            m_add1.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAddOutput, m_blobGG);
            m_tanh.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobGG, m_blobAA);
            m_ipV.Reshape(m_colInternalBottom, m_colInternalTop);

            m_blobSoftmax.ReshapeLike(m_blobAA);
            m_blobScale.ReshapeLike(m_blobSoftmax);

            List<int> rgFocusShape = Utility.Clone<int>(blobX.shape());
            rgFocusShape[0] = blobX.shape(1);
            rgFocusShape[1] = blobX.shape(0);
            m_blobFocusedInput.Reshape(rgFocusShape);

            List<int> rgContextShape = Utility.Clone<int>(blobX.shape());
            rgContextShape[0] = rgContextShape[1];
            rgContextShape[1] = 1;
            m_blobContext.Reshape(rgContextShape);

            List<int> rgTopShape = Utility.Clone<int>(m_blobContext.shape());
            rgTopShape[0] = m_blobContext.shape(1);
            rgTopShape[1] = m_blobContext.shape(0);
            colTop[0].Reshape(rgTopShape);            
        }

        private void apply_clip(Blob<T> blobInput, Blob<T> blobClip, Blob<T> blobOutput, bool bDiff = false)
        {
            long hSrc = (bDiff) ? blobInput.gpu_diff : blobInput.gpu_data;
            long hClip = blobClip.gpu_data;
            long hDst = (bDiff) ? blobOutput.mutable_gpu_diff : blobOutput.gpu_diff;

            m_cuda.channel_scale(blobInput.count(), blobInput.num, blobInput.channels, blobInput.count(2), hSrc, hClip, hDst);
        }

        private void softmax_fwd(Blob<T> blobBottom, Blob<T> blobClip, Blob<T> blobScale, Blob<T> blobTop, int nAxis)
        {
            int nCount = blobBottom.count();
            int nOuterNum = blobBottom.count(0, nAxis);
            int nInnerNum = blobBottom.count(nAxis + 1);
            int nChannels = blobTop.shape(nAxis);
            long hBottomData = blobBottom.gpu_data;
            long hTopData = blobTop.mutable_gpu_data;
            long hScaleData = blobScale.mutable_gpu_data;

            m_cuda.copy(nCount, hBottomData, hTopData);

            // Mask all values pass the valid ones specified in the clip, to -1 billion.
            int nValidValues = (int)convertF(blobClip.asum_data());
            if (nValidValues < nCount)
                blobTop.SetData(-1000000000.0, nValidValues, nCount - nValidValues);

            // We need to subtract the max to avoid numerical issues, compute the exp
            // and then normalize.
            // compute max.
            m_cuda.channel_max(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, hTopData, hScaleData);

            // subtract
            m_cuda.channel_sub(nCount, nOuterNum, nChannels, nInnerNum, hScaleData, hTopData);

            // exponentiate
            m_cuda.exp(nCount, hTopData, hTopData);

            // Sum after exp
            m_cuda.channel_sum(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, hTopData, hScaleData);

            // divide
            m_cuda.channel_div(nCount, nOuterNum, nChannels, nInnerNum, hScaleData, hTopData);

            // Denan for divide by zero.
            m_cuda.denan(nCount, blobTop.mutable_gpu_data, 0);
        }

        private void softmax_bwd(Blob<T> blobTop, Blob<T> blobClip, Blob<T> blobScale, Blob<T> blobBottom, int nAxis)
        {
            int nOuterNum = blobBottom.count(0, nAxis);
            int nInnerNum = blobBottom.count(nAxis + 1);
            long hTopDiff = blobTop.gpu_diff;
            long hTopData = blobTop.gpu_data;
            long hBottomDiff = blobBottom.mutable_gpu_diff;
            long hScaleData = m_blobScale.mutable_gpu_data;
            int nCount = blobTop.count();
            int nChannels = blobTop.shape(nAxis);

            m_cuda.copy(nCount, hTopDiff, hBottomDiff);

            // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
            m_cuda.channel_dot(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, hTopDiff, hTopData, hScaleData);
            m_cuda.channel_sub(nCount, nOuterNum, nChannels, nInnerNum, hScaleData, hBottomDiff);

            // Apply clip.
            m_cuda.channel_scale(nCount, nOuterNum, nChannels, nInnerNum, hTopData, blobClip.gpu_data, hTopData);

            // elementwise multiplication
            m_cuda.mul(nCount, hBottomDiff, hTopData, hBottomDiff);
        }

        // Move to the GPU.
        private float sum_diff(int nCount, Blob<T> b, int nOffset)
        {
            float[] rg = convertF(b.mutable_cpu_diff);
            float fSum = 0;

            for (int i = 0; i < nCount; i++)
            {
                fSum += rg[nOffset + i];
            }

            return fSum;
        }

        private void fill(Blob<T> blob1, Blob<T> blob2, Blob<T> blobFull, bool bUseDiff = false, int nBlob1AxisOffset = 0)
        {
            int nAxis = 1;
            int nM = blob1.shape(nAxis + nBlob1AxisOffset);
            int nN = blobFull.shape(nAxis);
            int nK = blob1.count(nAxis + nBlob1AxisOffset + 1);

            if (blob2 == null)
            {
                List<int> rgShape = new List<int>();
                rgShape.Add(blob1.count(0, nAxis));
                rgShape.Add(nN);
                m_blobWork.Reshape(rgShape);
                m_blobWork.SetData(1.0);
                blob2 = m_blobWork;
            }

            if (bUseDiff)
                m_cuda.gemm(true, true, nM, nN, nK, 1.0, blob1.gpu_diff, blob2.gpu_data, 0.0, blobFull.mutable_gpu_diff);
            else
                m_cuda.gemm(true, false, nM, nN, nK, 1.0, blob1.gpu_data, blob2.gpu_data, 0.0, blobFull.mutable_gpu_data);

            // Transpose result.
            m_blobWork.ReshapeLike(blobFull);

            if (bUseDiff)
            {
                m_cuda.geam(true, false, nM, nN, 1.0, blobFull.gpu_diff, blobFull.gpu_diff, 0.0, m_blobWork.mutable_gpu_data);
                m_cuda.copy(m_blobWork.count(), m_blobWork.gpu_data, blobFull.mutable_gpu_diff);
            }
            else
            {
                m_cuda.geam(true, false, nM, nN, 1.0, blobFull.gpu_data, blobFull.gpu_data, 0.0, m_blobWork.mutable_gpu_data);
                m_cuda.copy(m_blobWork.count(), m_blobWork.gpu_data, blobFull.mutable_gpu_data);
            }
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$
        ///     the computed inner product with the weights, where
        ///     @f$ K @f$ equals <i>num_output</i>.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobX = colBottom[0];
            Blob<T> blobCy = colBottom[1];
            Blob<T> blobClip = colBottom[2];

            // Force values to 1 or 0.
            m_cuda.sign(blobClip.count(), blobClip.gpu_data, blobClip.mutable_gpu_data);

            addInternal(blobX, m_blobX);
            m_transposeX.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(blobClip, m_blobClip);
            m_transposeClip.Forward(m_colInternalBottom, m_colInternalTop);

            // Apply the clip.
            apply_clip(m_blobX, blobClip, m_blobX);

            // No need to transpose for state T = 1.
            m_cuda.copy(blobCy.count(), blobCy.gpu_data, m_blobState.mutable_gpu_data);

            addInternal(m_blobX, m_blobUh);
            m_ipUa.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobState, m_blobWc);
            m_ipWa.Forward(m_colInternalBottom, m_colInternalTop);

            // Duplicate Wc across all T.
            fill(m_blobWc, null, m_blobFullWc);

            addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
            m_add1.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAddOutput, m_blobGG);
            m_tanh.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobGG, m_blobAA);
            m_ipV.Forward(m_colInternalBottom, m_colInternalTop);

            softmax_fwd(m_blobAA, m_blobClip, m_blobScale, m_blobSoftmax, 1);

            // Apply softmax to each channel
            m_blobFocusedInput.CopyFrom(m_blobX);
            m_blobContext.SetData(0);
            int nCount = m_blobFocusedInput.count(2);
            int nOuterNum = m_blobFocusedInput.count(0, 2);
            // Scale by softmax and sum.
            m_cuda.gemv(true, nOuterNum, nCount, 1.0, m_blobFocusedInput.gpu_data, m_blobSoftmax.gpu_data, 0.0, m_blobContext.mutable_gpu_data);

            // Reshape not needed for T = 1 in topT and top(0)
            m_cuda.copy(m_blobContext.count(), m_blobContext.gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Computes the inner product loss error gradient w.r.t the outputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (N \times K \times 1 \times 1) @f$, where @f$ K @f$ is equal to <i>num_output</i>.
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
                Blob<T> blobX = colBottom[0];
                Blob<T> blobCy = colBottom[1];
                Blob<T> blobClip = colBottom[2];

                m_blobGG.SetDiff(0);
                m_blobAA.SetDiff(0);
                m_blobAddOutput.SetDiff(0);
                m_blobUh.SetDiff(0);
                m_blobFullWc.SetDiff(0);
                m_blobWc.SetDiff(0);
                m_blobScale.SetDiff(0);

                List<bool> rgbPropagate = new List<bool>() { true, true };

                // Reshape not needed for T = 1 in topT and top(0)
                m_cuda.copy(colTop[0].count(), colTop[0].gpu_data, m_blobContext.mutable_gpu_data);
                m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, m_blobContext.mutable_gpu_diff);

                // Apply gradient w.r.t input. (x = context x softmax)
                fill(m_blobContext, m_blobSoftmax, m_blobX1, true, 1);

                // Apply gradient w.r.t softmax.
                m_cuda.channel_mulv(m_blobX.count(), m_blobX.num, m_blobX.channels, m_blobX.count(2), m_blobX.gpu_data, m_blobContext.gpu_diff, m_blobFocusedInput.mutable_gpu_diff);
                m_cuda.channel_sum(m_blobX.count(), m_blobX.count(0, 2), m_blobX.shape(2), m_blobX.count(3), m_blobFocusedInput.gpu_diff, m_blobSoftmax.mutable_gpu_diff);

                softmax_bwd(m_blobSoftmax, m_blobClip, m_blobScale, m_blobAA, 1);

                addInternal(m_blobGG, m_blobAA);
                m_ipV.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(m_blobAddOutput, m_blobGG);
                m_tanh.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
                m_add1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Sum weights by channel.
                m_cuda.channel_sum(m_blobFullWc.count(), m_blobFullWc.num, m_blobFullWc.channels, m_blobWc.count(), m_blobFullWc.gpu_diff, m_blobWc.mutable_gpu_diff);

                addInternal(m_blobState, m_blobWc);
                m_ipWa.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(m_blobX, m_blobUh);
                m_ipUa.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                m_cuda.add(m_blobX.count(), m_blobX1.gpu_diff, m_blobX.gpu_diff, m_blobX.mutable_gpu_diff);

                // No need to transpose for state T = 1.
                m_cuda.copy(blobCy.count(), m_blobState.gpu_diff, blobCy.mutable_gpu_diff);

                addInternal(blobX, m_blobX);
                m_transposeX.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
            }
        }
    }
}
