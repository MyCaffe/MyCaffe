using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.lnn;

namespace MyCaffe.layers.lnn
{
    /// <summary>
    /// The CfcLayer implements the Closed form Continuous layer. 
    /// </summary>
    /// <remarks>
    /// @see [GitHub:raminmh/CfC](https://github.com/raminmh/CfC) by raminmh, 2021, GitHub (distributed under Apache 2.0).
    /// @see [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl and Daniela Rus, 2022, Nature Machine Intelligence, 4, 992-1003
    /// @see [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, arXiv 2106.13898
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CfcLayer<T> : Layer<T>
    {
        int m_nBatchSize;
        int m_nSeqLen;
        int m_nTrueInFeatures;
        int m_nReshapeCount = 0;
        int m_nMaskCount;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        Layer<T> m_rnn_cell = null;
        Layer<T> m_cat = null;
        Layer<T> m_fc = null;
        Blob<T> m_blobInputs1 = null;
        Blob<T> m_blobInputs = null;
        Blob<T> m_blobHState1 = null;
        Blob<T> m_blobHState = null;
        BlobCollection<T> m_rgBlobHState = new BlobCollection<T>();

        List<BlobCollection<T>> m_rgrgInternalBlobs = new List<BlobCollection<T>>();
       
        Blob<T> m_blobTs = null;
        Blob<T> m_blobTsFull = null;
        Blob<T> m_blobForwardInput = null;
        Blob<T> m_blobForwardInput1 = null;
        Blob<T> m_blobForwardInput2 = null;
        Blob<T> m_blobForwardOutput = null;
        Blob<T> m_blobForwardOutput1 = null;
        Blob<T> m_blobForwardOutput2 = null;
        Blob<T> m_blobTimeSinceUpdate = null;
        Blob<T> m_blobTimeSinceUpdate1 = null;
        Blob<T> m_blobMask = null;
        Blob<T> m_blobMaskInv = null;
        Blob<T> m_blobCurrentMask = null;
        Blob<T> m_blobCurrentMaskFull = null;
        Blob<T> m_blobCurrentOutput = null;
        Blob<T> m_blobOutputSequence = null;
        int[] m_rgShape = new int[] { 1, 1, 1, 1 };
        bool m_bSetup = false;
        int m_nHiddenSize = 0;

        /// <summary>
        /// The CfcLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public CfcLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CFC;

            if (p.cfc_param.cell_type == CfcParameter.CELL_TYPE.LTC)
                m_nHiddenSize = p.ltc_unit_param.hidden_size;
            else
                m_nHiddenSize = p.cfc_unit_param.hidden_size;

            m_blobHState1 = new Blob<T>(m_cuda, m_log);
            m_blobHState = new Blob<T>(m_cuda, m_log);

            m_blobTs = new Blob<T>(m_cuda, m_log);

            m_blobInputs = new Blob<T>(m_cuda, m_log);
            m_blobInputs1 = new Blob<T>(m_cuda, m_log);
            m_blobMask = new Blob<T>(m_cuda, m_log);
            m_blobMaskInv = new Blob<T>(m_cuda, m_log);

            m_blobCurrentMask = new Blob<T>(m_cuda, m_log);
            m_blobCurrentOutput = new Blob<T>(m_cuda, m_log);
            m_blobCurrentMaskFull = new Blob<T>(m_cuda, m_log);

            m_blobForwardInput = new Blob<T>(m_cuda, m_log);
            m_blobForwardInput1 = new Blob<T>(m_cuda, m_log);
            m_blobForwardInput2 = new Blob<T>(m_cuda, m_log);
            m_blobTimeSinceUpdate = new Blob<T>(m_cuda, m_log);
            m_blobTimeSinceUpdate1 = new Blob<T>(m_cuda, m_log);
            m_blobTsFull = new Blob<T>(m_cuda, m_log);

            m_blobForwardOutput = new Blob<T>(m_cuda, m_log);
            m_blobForwardOutput1 = new Blob<T>(m_cuda, m_log);
            m_blobForwardOutput2 = new Blob<T>(m_cuda, m_log);

            m_blobOutputSequence = new Blob<T>(m_cuda, m_log);

            LayerParameter cat = new LayerParameter(LayerParameter.LayerType.CONCAT);
            cat.concat_param.axis = 1;
            m_cat = Layer<T>.Create(m_cuda, m_log, convertLayerParam(cat, p), null);

            LayerParameter rnn = null;

            if (m_param.cfc_param.cell_type == CfcParameter.CELL_TYPE.LTC)
            {
                rnn = new LayerParameter(LayerParameter.LayerType.LTC_UNIT);
                rnn.ltc_unit_param.Copy(m_param.ltc_unit_param);
            }
            else
            {
                rnn = new LayerParameter(LayerParameter.LayerType.CFC_UNIT);
                rnn.cfc_unit_param.Copy(m_param.cfc_unit_param);
            }

            m_rnn_cell = Layer<T>.Create(m_cuda, m_log, convertLayerParam(rnn, p), null);

            LayerParameter fc = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "fc");
            fc.inner_product_param.num_output = (uint)m_param.cfc_param.output_features;
            fc.inner_product_param.bias_term = true;
            fc.inner_product_param.weight_filler = new FillerParameter("xavier");
            fc.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            fc.inner_product_param.axis = 1;
            m_fc = Layer<T>.Create(m_cuda, m_log, convertLayerParam(fc, p), null);
        }

        private void dispose(ref List<BlobCollection<T>> rg)
        {
            if (rg == null)
                return;

            for (int i = 0; i < rg.Count; i++)
            {
                for (int j = 0; j < rg[i].Count; j++)
                {
                    if (rg[i][j] != null)
                        rg[i][j].Dispose();
                }
            }
            rg.Clear();
            rg = null;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            dispose(ref m_rgBlobHState);

            dispose(ref m_rgrgInternalBlobs);

            dispose(ref m_blobHState1);
            dispose(ref m_blobHState);
            dispose(ref m_blobInputs1);
            dispose(ref m_blobInputs);
            dispose(ref m_blobTs);
            dispose(ref m_blobTsFull);
            dispose(ref m_blobForwardInput);
            dispose(ref m_blobForwardInput1);
            dispose(ref m_blobForwardInput2);
            dispose(ref m_blobForwardOutput);
            dispose(ref m_blobForwardOutput1);
            dispose(ref m_blobForwardOutput2);
            dispose(ref m_blobTimeSinceUpdate);
            dispose(ref m_blobTimeSinceUpdate1);
            dispose(ref m_blobMask);
            dispose(ref m_blobMaskInv);
            dispose(ref m_blobCurrentMask);
            dispose(ref m_blobCurrentMaskFull);
            dispose(ref m_blobCurrentOutput);
            dispose(ref m_blobOutputSequence);

            dispose(ref m_rnn_cell);
            dispose(ref m_cat);
            dispose(ref m_fc);
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
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input, hx, ts
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: attn
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

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_bSetup)
                return;

            m_nBatchSize = colBottom[0].num;
            m_nSeqLen = colBottom[0].channels;
            m_nTrueInFeatures = colBottom[0].count(2);
            m_nReshapeCount = 0;

            m_rgShape[0] = m_nBatchSize;
            m_rgShape[1] = m_nHiddenSize;
            m_blobHState1.Reshape(m_rgShape);
            m_blobHState.Reshape(m_rgShape);

            for (int i = 0; i < m_nSeqLen; i++)
            {
                Blob<T> blobHStateT = new Blob<T>(m_cuda, m_log, m_rgShape);
                blobHStateT.Name = "h_state_" + i.ToString();
                m_rgBlobHState.Add(blobHStateT);

                BlobCollection<T> col = ((LnnUnitLayer<T>)m_rnn_cell).CreateInternalSharedBlobs(i, m_cuda, m_log);
                m_rgrgInternalBlobs.Add(col);
            }

            m_rgShape[1] = 1;   
            m_blobTs.Reshape(m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobInputs.Reshape(m_rgShape);
            m_blobMask.ReshapeLike(m_blobInputs);
            m_blobMaskInv.ReshapeLike(m_blobInputs);
            m_nMaskCount = m_blobMask.count(2);

            m_rgShape[1] = 1;
            m_blobCurrentMask.Reshape(m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobForwardInput.Reshape(m_rgShape);
            m_blobForwardInput1.Reshape(m_rgShape);
            m_blobForwardInput2.Reshape(m_rgShape);
            m_blobTimeSinceUpdate.Reshape(m_rgShape);
            m_blobTimeSinceUpdate1.Reshape(m_rgShape);
            m_blobTsFull.Reshape(m_rgShape);

            m_rgShape[1] = m_param.cfc_param.output_features;
            m_blobForwardOutput.Reshape(m_rgShape);
            m_blobForwardOutput1.Reshape(m_rgShape);
            m_blobForwardOutput2.Reshape(m_rgShape);

            addBtmTop(m_blobForwardInput, m_blobInputs1);
            if (m_nTrueInFeatures * 2 < m_param.cfc_param.input_features && m_nMaskCount == m_nTrueInFeatures)
                m_colBtm.Add(m_blobTimeSinceUpdate);
            m_colBtm.Add(m_blobMask);
            m_cat.Setup(m_colBtm, m_colTop);

            ((LnnUnitLayer<T>)m_rnn_cell).SetInternalSharedBlobs(m_rgrgInternalBlobs[0]);

            addBtmTop(m_blobInputs1, m_blobHState);
            m_colBtm.Add(m_blobHState1);
            m_colBtm.Add(m_blobTs);
            m_rnn_cell.Setup(m_colBtm, m_colTop);
            blobs.Add(m_rnn_cell.blobs);

            m_blobHState.Unsqueeze(4);

            addBtmTop(m_blobHState, m_blobCurrentOutput);
            m_fc.Setup(m_colBtm, m_colTop);
            blobs.Add(m_fc.blobs);

            m_blobCurrentMaskFull.ReshapeLike(m_blobCurrentOutput);

            addBtmTop(m_blobHState, colTop[0]);
            m_fc.Reshape(m_colBtm, m_colTop);

            m_bSetup = true;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nTrueInFeatures = (int)colBottom[0].count(2);

            // Only reshape when needed.
            if (m_nReshapeCount > 0 && 
                m_nBatchSize > 0 && m_nBatchSize == colBottom[0].num &&
                m_nSeqLen > 0 && m_nSeqLen == colBottom[0].channels &&
                m_nTrueInFeatures > 0 && m_nTrueInFeatures == nTrueInFeatures)
                return;

            m_nReshapeCount++;
            m_nBatchSize = colBottom[0].num;
            m_nSeqLen = colBottom[0].channels;
            m_nTrueInFeatures = nTrueInFeatures;

            m_rgShape[0] = m_nBatchSize;
            m_rgShape[1] = m_nHiddenSize;

            m_blobHState1.Reshape(m_rgShape);
            m_blobHState.Reshape(m_rgShape);

            for (int i=0; i<m_nSeqLen; i++)
            {
                m_rgBlobHState[i].Reshape(m_rgShape);
            }

            m_rgShape[1] = 1;
            m_blobTs.Reshape(m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobInputs.Reshape(m_rgShape);
            m_blobMask.ReshapeLike(m_blobInputs);
            m_blobMaskInv.ReshapeLike(m_blobInputs);
            m_blobMaskInv.SetDiff(1.0);
            m_nMaskCount = m_blobMask.count(2);

            m_rgShape[1] = 1;
            m_blobCurrentMask.Reshape(m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobForwardInput.Reshape(m_rgShape);
            m_blobForwardInput1.Reshape(m_rgShape);
            m_blobForwardInput2.Reshape(m_rgShape);
            m_blobTimeSinceUpdate.Reshape(m_rgShape);
            m_blobTimeSinceUpdate1.Reshape(m_rgShape);
            m_blobTsFull.ReshapeLike(m_blobTimeSinceUpdate1);

            m_rgShape[1] = m_param.cfc_param.output_features;
            m_blobForwardOutput.Reshape(m_rgShape);
            m_blobForwardOutput1.Reshape(m_rgShape);
            m_blobForwardOutput2.Reshape(m_rgShape);

            addBtmTop(m_blobForwardInput, m_blobInputs1);
            if (m_nTrueInFeatures * 2 < m_param.cfc_param.input_features && m_nMaskCount == m_nTrueInFeatures)
                m_colBtm.Add(m_blobTimeSinceUpdate);
            m_colBtm.Add(m_blobMask);
            m_cat.Reshape(m_colBtm, m_colTop);

            for (int i = 0; i < m_nSeqLen; i++)
            {
                addBtmTop(m_blobInputs1, m_blobHState);
                m_colBtm.Add(m_blobHState1);
                m_colBtm.Add(m_blobTs);

                ((LnnUnitLayer<T>)m_rnn_cell).SetInternalSharedBlobs(m_rgrgInternalBlobs[i]);

                m_rnn_cell.Reshape(m_colBtm, m_colTop);
            }

            m_blobHState.Unsqueeze(4);

            addBtmTop(m_blobHState, m_blobCurrentOutput);
            m_fc.Reshape(m_colBtm, m_colTop);

            m_blobCurrentMaskFull.ReshapeLike(m_blobCurrentOutput);

            addBtmTop(m_blobHState, colTop[0]);
            m_fc.Reshape(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobForwardInput.SetData(0);
            m_blobForwardOutput.SetData(0);
            m_blobTimeSinceUpdate.SetData(0);
            m_blobHState.SetData(0);

            for (int t = 0; t < m_nSeqLen; t++)
            {
                // Copy the t'th time step of the input to the input blob.
                m_cuda.channel_copy(m_blobInputs.count(), m_blobInputs.num, 1, m_nSeqLen, m_nTrueInFeatures, t, colBottom[0].gpu_data, m_blobInputs.mutable_gpu_data, DIR.FWD);
                // Copy the t'th timestep of the time since update to the ts blob.
                m_cuda.channel_copy(m_blobTs.count(), m_blobTs.num, 1, m_nSeqLen, 1, t, colBottom[1].gpu_data, m_blobTs.mutable_gpu_data, DIR.FWD);
                m_cuda.channel_fillfrom(m_blobTsFull.count(), 1, m_blobTs.num, m_blobTsFull.channels, m_blobTs.gpu_data, m_blobTsFull.mutable_gpu_data, DIR.FWD);

                // Apply masking
                if (colBottom.Count() > 2)
                {
                    int nMaskCount = colBottom[2].count(2);
                    if (nMaskCount == m_nTrueInFeatures)
                    {
                        // Copy the t'th mask of the full mask to the mask blob.
                        m_cuda.channel_copy(m_blobMask.count(), m_blobMask.num, 1, m_nSeqLen, nMaskCount, t, colBottom[2].gpu_data, m_blobMask.mutable_gpu_data, DIR.FWD);
                        // Create the mask inverse
                        m_cuda.sub(m_blobMask.count(), m_blobMaskInv.gpu_diff, m_blobMask.gpu_data, m_blobMaskInv.mutable_gpu_data);

                        // Update the forwarded input.
                        m_cuda.mul(m_blobMask.count(), m_blobInputs.gpu_data, m_blobMask.gpu_data, m_blobForwardInput1.mutable_gpu_data);
                        m_cuda.mul(m_blobMask.count(), m_blobForwardInput.gpu_data, m_blobMaskInv.gpu_data, m_blobForwardInput2.mutable_gpu_data);
                        m_cuda.add(m_blobMask.count(), m_blobForwardInput1.gpu_data, m_blobForwardInput2.gpu_data, m_blobForwardInput.mutable_gpu_data);

                        // Update the time since update.
                        m_cuda.add(m_blobTimeSinceUpdate.count(), m_blobTimeSinceUpdate.gpu_data, m_blobTsFull.gpu_data, m_blobTimeSinceUpdate1.mutable_gpu_data);
                        m_cuda.mul(m_blobTimeSinceUpdate1.count(), m_blobTimeSinceUpdate1.gpu_data, m_blobMaskInv.gpu_data, m_blobTimeSinceUpdate.mutable_gpu_data);
                    }
                    else
                    {
                        m_cuda.copy(m_blobForwardInput.count(), m_blobInputs.gpu_data, m_blobForwardInput.mutable_gpu_data);
                    }

                    // Update a 3x in-features mask.
                    if (m_nTrueInFeatures * 2 < m_param.cfc_param.input_features && m_nMaskCount == m_nTrueInFeatures)
                    {
                        addBtmTop(m_blobForwardInput, m_blobInputs1);
                        m_colBtm.Add(m_blobTimeSinceUpdate);
                        m_colBtm.Add(m_blobMask);
                        m_cat.Forward(m_colBtm, m_colTop);
                    }
                    // Update a 2x in-features mask.
                    else
                    {
                        addBtmTop(m_blobForwardInput, m_blobInputs1);
                        m_colBtm.Add(m_blobMask);
                        m_cat.Forward(m_colBtm, m_colTop);
                    }
                }
                else
                {
                    m_blobInputs1.CopyFrom(m_blobInputs);
                }

                // Run the CfcCell forward pass.
                addBtmTop(m_blobInputs1, m_blobHState1);
                m_colBtm.Add(m_blobHState);
                m_colBtm.Add(m_blobTs);

                ((LnnUnitLayer<T>)m_rnn_cell).SetInternalSharedBlobs(m_rgrgInternalBlobs[t]);

                m_rnn_cell.Forward(m_colBtm, m_colTop);

                m_blobHState1.Unsqueeze(4);
                m_rgBlobHState[t].CopyFrom(m_blobHState1);

                // Apply masking
                if (colBottom.Count > 2)
                {
                    m_cuda.channel_max(m_blobMask.count(), m_blobMask.num, m_blobMask.channels, 1, m_blobMask.gpu_data, m_blobCurrentMask.mutable_gpu_data);
                    
                    // Create mask inverse
                    m_blobCurrentMask.SetDiff(1.0);
                    m_cuda.sub(m_blobCurrentMask.count(), m_blobCurrentMask.gpu_diff, m_blobCurrentMask.gpu_data, m_blobCurrentMask.mutable_gpu_diff);

                    m_cuda.channel_fillfrom(m_blobCurrentMaskFull.count(), m_blobCurrentMask.num, m_blobCurrentMask.channels, m_blobCurrentMaskFull.channels, m_blobCurrentMask.gpu_data, m_blobCurrentMaskFull.mutable_gpu_data, DIR.FWD);
                    m_cuda.channel_fillfrom(m_blobCurrentMaskFull.count(), m_blobCurrentMask.num, m_blobCurrentMask.channels, m_blobCurrentMaskFull.channels, m_blobCurrentMask.gpu_diff, m_blobCurrentMaskFull.mutable_gpu_diff, DIR.FWD);

                    addBtmTop(m_blobHState1, m_blobCurrentOutput);
                    m_fc.Forward(m_colBtm, m_colTop);

                    // Update the forwarded output.
                    m_cuda.mul(m_blobCurrentMaskFull.count(), m_blobCurrentOutput.gpu_data, m_blobCurrentMaskFull.gpu_data, m_blobForwardOutput1.mutable_gpu_data);
                    m_cuda.mul(m_blobCurrentMaskFull.count(), m_blobForwardOutput.gpu_data, m_blobCurrentMaskFull.gpu_diff, m_blobForwardOutput2.mutable_gpu_data);
                    m_cuda.add(m_blobCurrentMaskFull.count(), m_blobForwardOutput1.gpu_data, m_blobForwardOutput2.gpu_data, m_blobForwardOutput.mutable_gpu_data);
                }

                m_blobHState.CopyFrom(m_blobHState1);
            }

            if (m_param.cfc_param.return_sequences)
            {
                throw new NotImplementedException("return sequences is not implemented yet.");
            }
            else if (colBottom.Count > 2)
            {
                colTop[0].CopyFrom(m_blobForwardOutput);
            }
            else
            {
                addBtmTop(m_blobHState, colTop[0]);
                m_fc.Forward(m_colBtm, m_colTop);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Cfc value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$ y' @f$
        ///     if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_blobHState.SetDiff(0);
            m_blobInputs.SetDiff(0);
            m_blobForwardInput.SetDiff(0);

            if (m_param.cfc_param.return_sequences)
            {
                throw new NotImplementedException("return sequences is not implemented yet.");
            }
            else if (colBottom.Count > 2)
            {
                m_blobForwardOutput.CopyFrom(colTop[0], true);
            }
            else
            {
                addBtmTop(m_blobHState, colTop[0]);
                m_fc.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            int nMaskCount = colBottom[2].count(2);
            for (int t = m_nSeqLen - 1; t >= 0; t--)
            {
                m_blobHState1.CopyFrom(m_rgBlobHState[t]);

                if (colBottom.Count > 2)
                {
                    // Copy the t'th mask of the full mask to the mask blob.
                    m_cuda.channel_copy(m_blobMask.count(), m_blobMask.num, 1, m_nSeqLen, nMaskCount, t, colBottom[2].gpu_data, m_blobMask.mutable_gpu_data, DIR.FWD);
                    // Create the mask inverse
                    m_cuda.sub(m_blobMask.count(), m_blobMaskInv.gpu_diff, m_blobMask.gpu_data, m_blobMaskInv.mutable_gpu_data);

                    m_cuda.channel_max(m_blobMask.count(), m_blobMask.num, m_blobMask.channels, 1, m_blobMask.gpu_data, m_blobCurrentMask.mutable_gpu_data);

                    // Create mask inverse
                    m_blobCurrentMask.SetDiff(1.0);
                    m_cuda.sub(m_blobCurrentMask.count(), m_blobCurrentMask.gpu_diff, m_blobCurrentMask.gpu_data, m_blobCurrentMask.mutable_gpu_diff);

                    m_cuda.channel_fillfrom(m_blobCurrentMaskFull.count(), m_blobCurrentMask.num, m_blobCurrentMask.channels, m_blobCurrentMaskFull.channels, m_blobCurrentMask.gpu_data, m_blobCurrentMaskFull.mutable_gpu_data, DIR.FWD);
                    m_cuda.channel_fillfrom(m_blobCurrentMaskFull.count(), m_blobCurrentMask.num, m_blobCurrentMask.channels, m_blobCurrentMaskFull.channels, m_blobCurrentMask.gpu_diff, m_blobCurrentMaskFull.mutable_gpu_diff, DIR.FWD);

                    m_blobForwardOutput1.CopyFrom(m_blobForwardOutput, true);
                    m_blobForwardOutput2.CopyFrom(m_blobForwardOutput, true);

                    // Update the forwarded output.
                    m_cuda.mul(m_blobCurrentMaskFull.count(), m_blobForwardOutput1.gpu_diff, m_blobCurrentMaskFull.gpu_data, m_blobCurrentOutput.mutable_gpu_diff);
                    m_cuda.mul(m_blobCurrentMaskFull.count(), m_blobForwardOutput2.gpu_diff, m_blobCurrentMaskFull.gpu_diff, m_blobForwardOutput.mutable_gpu_diff);

                    addBtmTop(m_blobHState1, m_blobCurrentOutput);
                    m_fc.Backward(m_colTop, rgbPropagateDown, m_colBtm);
                }

                if (t < m_nSeqLen - 1)
                    m_cuda.add(m_blobHState1.count(), m_blobHState1.gpu_diff, m_rgBlobHState[t + 1].gpu_diff, m_blobHState1.mutable_gpu_diff);

                // Run the CfcCell backward pass.
                addBtmTop(m_blobInputs1, m_blobHState1);
                m_colBtm.Add(m_blobHState);
                m_colBtm.Add(m_blobTs);

                ((LnnUnitLayer<T>)m_rnn_cell).SetInternalSharedBlobs(m_rgrgInternalBlobs[t]);

                m_rnn_cell.Backward(m_colTop, new List<bool>() { true, true, true }, m_colBtm);
                m_rgBlobHState[t].CopyFrom(m_blobHState, true);

                // Apply masking
                if (colBottom.Count() > 2)
                {
                    if (m_nTrueInFeatures * 2 < m_param.cfc_param.input_features && m_nMaskCount == m_nTrueInFeatures)
                    {
                        addBtmTop(m_blobForwardInput1, m_blobInputs1);
                        m_colBtm.Add(m_blobTimeSinceUpdate);
                        m_colBtm.Add(m_blobMask);
                        m_cat.Backward(m_colTop, new List<bool>() { true, true, true }, m_colBtm);
                    }
                    else
                    {
                        addBtmTop(m_blobForwardInput1, m_blobInputs1);
                        m_colBtm.Add(m_blobMask);
                        m_cat.Backward(m_colTop, new List<bool>() { true, true }, m_colBtm);
                    }

                    // Accumulate grad with previous masked forward input.
                    m_cuda.mul(m_blobForwardInput.count(), m_blobForwardInput.gpu_diff, m_blobForwardInput2.gpu_diff, m_blobForwardInput.mutable_gpu_diff);
                    m_cuda.add(m_blobForwardInput.count(), m_blobForwardInput.gpu_diff, m_blobForwardInput1.gpu_diff, m_blobForwardInput.mutable_gpu_diff);

                    if (nMaskCount == m_nTrueInFeatures)
                    {
                        // Input grad = mask * Inputs1
                        m_cuda.mul(m_blobForwardInput.count(), m_blobForwardInput.gpu_diff, m_blobMask.gpu_data, m_blobInputs.mutable_gpu_diff);
                        // Forwarded input grad = mask_inv * Forwarded input grad
                        //m_cuda.mul(m_blobForwardInput.count(), m_blobForwardInput.gpu_diff, m_blobMaskInv.gpu_data, m_blobForwardInput.mutable_gpu_diff);
                    }
                    else
                    {
                        m_cuda.copy(m_blobForwardInput.count(), m_blobForwardInput.gpu_diff, m_blobInputs.mutable_gpu_diff);
                    }
                }
                else
                {
                    m_blobInputs.CopyFrom(m_blobInputs1, true);
                }

                // Copy the t'th timestep of the time since update to the ts blob.
                //m_cuda.channel_fillfrom(m_blobTsFull.count(), 1, m_blobTs.num, m_blobTsFull.channels, m_blobTs.gpu_diff, m_blobTsFull.mutable_gpu_diff, DIR.BWD);

                //m_cuda.channel_copy(m_blobTs.count(), m_blobTs.num, 1, m_nSeqLen, 1, t, colBottom[1].gpu_diff, m_blobTs.mutable_gpu_diff, DIR.BWD);
                // Copy the t'th time step of the input to the input blob.
                m_cuda.channel_copy(m_blobInputs.count(), m_blobInputs.num, 1, m_nSeqLen, m_nTrueInFeatures, t, colBottom[0].gpu_diff, m_blobInputs.mutable_gpu_diff, DIR.BWD);

                // Save previous mask.
                m_cuda.copy(m_blobForwardInput2.count(), m_blobMaskInv.gpu_data, m_blobForwardInput2.mutable_gpu_diff);
            }
        }
    }
}
