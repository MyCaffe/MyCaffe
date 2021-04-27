using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers.beta;

namespace MyCaffe.layers
{
    /// <summary>
    /// The AttentionLayer provides focus for LSTM based encoder/decoder models.
    /// </summary>
    /// <remarks>
    /// The AttentionLayer implementation was inspired by the C# Seq2SeqLearn implementation by mashmawy for language translation,
    /// @see [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn) distributed under MIT license.
    /// 
    /// And also inspired by the C# ChatBot implementation by HectorPulido which uses Seq2SeqLearn
    /// @see [HectorPulido/Chatbot-seq2seq-C-](https://github.com/HectorPulido/Chatbot-seq2seq-C-) distributed under [MIT license](https://github.com/HectorPulido/Chatbot-seq2seq-C-/blob/master/LICENSE).
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AttentionLayer<T> : Layer<T>
    {
        Layer<T> m_ipUa = null;
        Layer<T> m_ipWa = null;
        Layer<T> m_tanh = null;
        Layer<T> m_add = null;
        Layer<T> m_ipV = null;
        Layer<T> m_softmax = null;
        Layer<T> m_ipWc = null;

        Blob<T> m_blobX = null;
        Blob<T> m_blobX1 = null;
        Blob<T> m_blobState = null;
        Blob<T> m_blobUh = null;
        Blob<T> m_blobWc = null;
        Blob<T> m_blobFullWc = null;
        Blob<T> m_blobAddOutput = null;
        Blob<T> m_blobGG = null;
        Blob<T> m_blobAA = null;
        Blob<T> m_blobSoftmax = null;
        Blob<T> m_blobFocusedInput = null;
        Blob<T> m_blobContext = null;
        Blob<T> m_blobTopT = null;

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();
        Filler<T> m_filler = null;

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

            LayerParameter ipUaParam = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ipUaParam.name = "ipUa";
            ipUaParam.inner_product_param.axis = 2;
            ipUaParam.inner_product_param.num_output = m_param.attention_param.dim;
            ipUaParam.inner_product_param.weight_filler = m_param.attention_param.weight_filler;            
            ipUaParam.inner_product_param.bias_filler = m_param.attention_param.bias_filler;

            m_ipUa = new InnerProductLayer<T>(cuda, log, ipUaParam);

            LayerParameter ipWaParam = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ipWaParam.name = "ipWa";
            ipWaParam.inner_product_param.axis = 2;
            ipWaParam.inner_product_param.num_output = m_param.attention_param.dim;
            ipWaParam.inner_product_param.weight_filler = m_param.attention_param.weight_filler;
            ipWaParam.inner_product_param.bias_filler = m_param.attention_param.bias_filler;

            m_ipWa = new InnerProductLayer<T>(cuda, log, ipWaParam);

            LayerParameter addParam = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            addParam.name = "add";
            addParam.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;

            m_add = new EltwiseLayer<T>(cuda, log, addParam);

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

            LayerParameter softmaxParam = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmaxParam.name = "softmax";
            softmaxParam.softmax_param.axis = 1;

            m_softmax = new SoftmaxLayer<T>(cuda, log, softmaxParam);

            m_filler = Filler<T>.Create(m_cuda, m_log, m_param.attention_param.weight_filler);

            m_blobX = new Blob<T>(cuda, log);
            m_blobX.Name = "x";

            m_blobX1 = new Blob<T>(cuda, log);
            m_blobX1.Name = "x1";

            m_blobState = new Blob<T>(cuda, log);
            m_blobState.Name = "state";

            m_blobUh = new Blob<T>(cuda, log);
            m_blobUh.Name = "Uh";

            m_blobWc = new Blob<T>(cuda, log);
            m_blobWc.Name = "Wc";

            m_blobFullWc = new Blob<T>(cuda, log);
            m_blobFullWc.Name = "full_Wc";

            m_blobAddOutput = new Blob<T>(cuda, log);
            m_blobAddOutput.Name = "addOut";

            m_blobGG = new Blob<T>(cuda, log);
            m_blobGG.Name = "gg";

            m_blobAA = new Blob<T>(cuda, log);
            m_blobAA.Name = "aa";

            m_blobSoftmax = new Blob<T>(cuda, log);
            m_blobSoftmax.Name = "softmax";

            m_blobFocusedInput = new Blob<T>(cuda, log);
            m_blobFocusedInput.Name = "softmax_full";

            m_blobContext = new Blob<T>(cuda, log);
            m_blobContext.Name = "context";

            m_blobTopT = new Blob<T>(cuda, log);
            m_blobTopT.Name = "topT";

            LayerParameter ipWcParam = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ipWcParam.name = "ipWc";
            ipWcParam.inner_product_param.axis = 2;
            ipWcParam.inner_product_param.bias_term = false;
            ipWcParam.inner_product_param.num_output = m_param.attention_param.dim;
            ipWcParam.inner_product_param.weight_filler = new FillerParameter("constant", 1);

            m_ipWc = new InnerProductLayer<T>(cuda, log, ipWcParam);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobX);
            dispose(ref m_blobX1);
            dispose(ref m_blobState);
            dispose(ref m_ipUa);
            dispose(ref m_ipWa);
            dispose(ref m_tanh);
            dispose(ref m_add);
            dispose(ref m_ipV);
            dispose(ref m_softmax);
            dispose(ref m_ipWc);

            dispose(ref m_blobUh);
            dispose(ref m_blobWc);
            dispose(ref m_blobFullWc);
            dispose(ref m_blobAddOutput);
            dispose(ref m_blobGG);
            dispose(ref m_blobAA);
            dispose(ref m_blobSoftmax);
            dispose(ref m_blobFocusedInput);
            dispose(ref m_blobContext);
            dispose(ref m_blobTopT);

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
                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input, state (last ct), clip (1 on each input, 0 otherwise)
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
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
            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);

            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = colBottom[0].shape(1); // batch
            rgShape[1] = colBottom[0].shape(0); // timesteps;
            m_blobX.Reshape(rgShape);
            m_blobX1.Reshape(rgShape);

            addInternal(m_blobX, m_blobUh);
            m_ipUa.Setup(m_colInternalBottom, m_colInternalTop);

            rgShape = Utility.Clone<int>(colBottom[1].shape());
            rgShape[0] = colBottom[1].shape(1); // batch
            rgShape[1] = colBottom[1].shape(0); // timesteps;
            m_blobState.Reshape(rgShape);

            addInternal(m_blobState, m_blobWc);
            m_ipWa.Setup(m_colInternalBottom, m_colInternalTop);

            m_blobFullWc.ReshapeLike(m_blobUh);

            addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
            m_add.Setup(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAddOutput, m_blobGG);
            m_tanh.Setup(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobGG, m_blobAA);
            m_ipV.Setup(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAA, m_blobSoftmax);
            m_softmax.Setup(m_colInternalBottom, m_colInternalTop);

            List<int> rgFocusShape = Utility.Clone<int>(colBottom[0].shape());
            rgFocusShape[0] = colBottom[0].shape(1);
            rgFocusShape[1] = colBottom[0].shape(0);
            m_blobFocusedInput.Reshape(rgFocusShape);

            List<int> rgContextShape = Utility.Clone<int>(colBottom[0].shape());
            rgContextShape[0] = rgContextShape[1];
            rgContextShape[1] = 1;
            m_blobContext.Reshape(rgContextShape);

            addInternal(m_blobContext, m_blobTopT);
            m_ipWc.Setup(m_colInternalBottom, m_colInternalTop);

            List<int> rgTopShape = Utility.Clone<int>(m_blobTopT.shape());
            rgTopShape[0] = m_blobTopT.shape(1);
            rgTopShape[1] = m_blobTopT.shape(0);
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
            // Wc
            blobs.Add(m_ipWa.blobs[0]);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = colBottom[0].shape(1); // batch
            rgShape[1] = colBottom[0].shape(0); // timesteps;
            m_blobX.Reshape(rgShape);
            m_blobX1.Reshape(rgShape);

            addInternal(m_blobX, m_blobUh);
            m_ipUa.Reshape(m_colInternalBottom, m_colInternalTop);

            rgShape = Utility.Clone<int>(colBottom[1].shape());
            rgShape[0] = colBottom[1].shape(1); // batch
            rgShape[1] = colBottom[1].shape(0); // timesteps;
            m_blobState.Reshape(rgShape);

            addInternal(m_blobState, m_blobWc);
            m_ipWa.Reshape(m_colInternalBottom, m_colInternalTop);

            m_blobFullWc.ReshapeLike(m_blobUh);

            addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
            m_add.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAddOutput, m_blobGG);
            m_tanh.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobGG, m_blobAA);
            m_ipV.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAA, m_blobSoftmax);
            m_softmax.Reshape(m_colInternalBottom, m_colInternalTop);

            List<int> rgFocusShape = Utility.Clone<int>(colBottom[0].shape());
            rgFocusShape[0] = colBottom[0].shape(1);
            rgFocusShape[1] = colBottom[0].shape(0);
            m_blobFocusedInput.Reshape(rgFocusShape);

            List<int> rgContextShape = Utility.Clone<int>(colBottom[0].shape());
            rgContextShape[0] = rgContextShape[1];
            rgContextShape[1] = 1;
            m_blobContext.Reshape(rgContextShape);

            addInternal(m_blobContext, m_blobTopT);
            m_ipWc.Reshape(m_colInternalBottom, m_colInternalTop);

            List<int> rgTopShape = Utility.Clone<int>(m_blobTopT.shape());
            rgTopShape[0] = m_blobTopT.shape(1);
            rgTopShape[1] = m_blobTopT.shape(0);
            colTop[0].Reshape(rgTopShape);
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
            // Move this to the GPU.
            float[] rgData = convertF(colBottom[0].mutable_cpu_data);
            rgData = SimpleDatum.Transpose(rgData, colBottom[0].num, colBottom[0].channels, colBottom[0].count(2));
            m_blobX.mutable_cpu_data = convert(rgData);

            // No need to transpose for state T = 1.
            m_cuda.copy(colBottom[1].count(), colBottom[1].gpu_data, m_blobState.mutable_gpu_data);

            addInternal(m_blobX, m_blobUh);
            m_ipUa.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobState, m_blobWc);
            m_ipWa.Forward(m_colInternalBottom, m_colInternalTop);

            // Duplicate Wc across all T.
            // Move this to the GPU
            int nCount = m_blobWc.count() / m_blobWc.num;
            for (int i = 0; i < m_blobFullWc.num; i++)
            {
                int nIdxSrc = (i * nCount);

                for (int j = 0; j < m_blobFullWc.channels; j++)
                {
                    int nIdxDst = (i * m_blobFullWc.channels * nCount) + (j * nCount);
                    m_cuda.copy(nCount, m_blobWc.gpu_data, m_blobFullWc.mutable_gpu_data, nIdxSrc, nIdxDst);
                }
            }

            addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
            m_add.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAddOutput, m_blobGG);
            m_tanh.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobGG, m_blobAA);
            m_ipV.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobAA, m_blobSoftmax);
            m_softmax.Forward(m_colInternalBottom, m_colInternalTop);
            float[] rgSoftmax = convertF(m_blobSoftmax.mutable_cpu_data);

            // Apply softmax to each channel
            // Move this to the GPU
            m_blobFocusedInput.CopyFrom(m_blobX);
            m_blobContext.SetData(0);
            nCount = m_blobFocusedInput.count(2);
            for (int i = 0; i < m_blobFocusedInput.num; i++)
            {
                int nIdxDstContext = (i * m_blobContext.count(2));
                int nIdxSoftmax = (i * m_blobSoftmax.channels);

                for (int j = 0; j < m_blobFocusedInput.channels; j++)
                {
                    int nIdxFI = (i * m_blobFocusedInput.channels * nCount) + (j * nCount);

                    m_cuda.scal(nCount, rgSoftmax[nIdxSoftmax + j], m_blobFocusedInput.mutable_gpu_data, nIdxFI);
                    m_cuda.add(nCount, m_blobFocusedInput.gpu_data, m_blobContext.gpu_data, m_blobContext.mutable_gpu_data, 1.0, 1.0, nIdxFI, nIdxDstContext, nIdxDstContext);
                }
            }

            addInternal(m_blobContext, m_blobTopT);
            m_ipWc.Forward(m_colInternalBottom, m_colInternalTop);

            // Reshape not needed for T = 1 in topT and top(0)
            m_cuda.copy(m_blobTopT.count(), m_blobTopT.gpu_data, colTop[0].mutable_gpu_data);
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
            if (rgbPropagateDown[0] && rgbPropagateDown[1])
            {
                List<bool> rgbPropagate = new List<bool>() { true, true };

                // Reshape not needed for T = 1 in topT and top(0)
                m_cuda.copy(colTop[0].count(), colTop[0].gpu_data, m_blobTopT.mutable_gpu_data);
                m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, m_blobTopT.mutable_gpu_diff);

                addInternal(m_blobContext, m_blobTopT);
                m_ipWc.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Apply gradient w.r.t input.
                // Move this to the GPU.
                int nCount = m_blobContext.count(2);
                float[] rgSoftmaxData = convertF(m_blobSoftmax.mutable_cpu_data);
                for (int i = 0; i < m_blobFocusedInput.num; i++)
                {
                    int nIdxSrc = (i * m_blobContext.count(2));
                    int nIdxSoftmax = i * m_blobFocusedInput.channels;

                    for (int j = 0; j < m_blobFocusedInput.channels; j++)
                    {
                        int nIdxDst = (i * m_blobFocusedInput.channels * nCount) + (j * nCount);
                        float fSoftmax = rgSoftmaxData[nIdxSoftmax + j];
                        m_cuda.scale(nCount, convert(fSoftmax), m_blobContext.gpu_diff, m_blobX1.mutable_gpu_diff, nIdxSrc, nIdxDst);
                    }
                }

                // Apply gradient w.r.t softmax.
                // Move this to the GPU.
                float[] rgSoftmaxDiff = new float[rgSoftmaxData.Length];
                for (int i = 0; i < m_blobX.num; i++)
                {
                    int nIdxSrc = (i * m_blobContext.count(2));
                    int nIdxSoftmax = i * m_blobX.channels;

                    for (int j = 0; j < m_blobFocusedInput.channels; j++)
                    {
                        int nIdxDst = (i * m_blobFocusedInput.channels * nCount) + (j * nCount);
                        m_cuda.mul(nCount, m_blobContext.gpu_diff, m_blobX.gpu_data, m_blobFocusedInput.mutable_gpu_diff, nIdxSrc, nIdxDst, nIdxDst);
                        rgSoftmaxDiff[nIdxSoftmax + j] = m_cuda.asum_float(nCount, m_blobFocusedInput.gpu_diff, nIdxDst);
                    }
                }

                m_blobSoftmax.mutable_cpu_diff = convert(rgSoftmaxDiff);

                addInternal(m_blobAA, m_blobSoftmax);
                m_softmax.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(m_blobGG, m_blobAA);
                m_ipV.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(m_blobAddOutput, m_blobGG);
                m_tanh.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(new List<Blob<T>>() { m_blobUh, m_blobFullWc }, m_blobAddOutput);
                m_add.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Sum weights by channel.
                m_cuda.channel_sum(m_blobFullWc.count(), m_blobFullWc.num, m_blobFullWc.channels, m_blobWc.count(), m_blobFullWc.gpu_diff, m_blobWc.mutable_gpu_diff);

                addInternal(m_blobState, m_blobWc);
                m_ipWa.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                addInternal(m_blobX, m_blobUh);
                m_ipUa.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                m_cuda.add(m_blobX.count(), m_blobX1.gpu_diff, m_blobX.gpu_diff, m_blobX.mutable_gpu_diff);

                // No need to transpose for state T = 1.
                m_cuda.copy(colBottom[1].count(), m_blobState.gpu_diff, colBottom[1].mutable_gpu_diff);

                // Move this to the GPU.
                float[] rgX = convertF(m_blobX.mutable_cpu_diff);
                rgX = SimpleDatum.Transpose(rgX, m_blobX.num, m_blobX.channels, m_blobX.count(2));
                colBottom[0].mutable_cpu_diff = convert(rgX);
            }
        }
    }
}
