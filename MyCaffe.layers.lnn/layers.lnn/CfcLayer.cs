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
    /// [WORK IN PROGRESS] The CfcLayer implements the Closed form Continuous layer. 
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
        int m_nMaskCount;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        Layer<T> m_rnn_cell = null;
        Layer<T> m_cat = null;
        Layer<T> m_fc = null;
        Blob<T> m_blobInputs = null;
        Blob<T> m_blobInputs1 = null;
        Blob<T> m_blobHState = null;
        Blob<T> m_blobTs = null;
        Blob<T> m_blobHState1 = null;
        Blob<T> m_blobForwardInput = null;
        Blob<T> m_blobForwardInput1 = null;
        Blob<T> m_blobForwardInput2 = null;
        Blob<T> m_blobForwardOutput = null;
        Blob<T> m_blobForwardOutput1 = null;
        Blob<T> m_blobForwardOutput2 = null;
        Blob<T> m_blobTimeSinceUpdate = null;
        Blob<T> m_blobTimeSinceUpdate1 = null;
        Blob<T> m_blobMask = null;
        Blob<T> m_blobCurrentMask = null;
        Blob<T> m_blobCurrentOutput = null;
        Blob<T> m_blobOutputSequence = null;
        int[] m_rgShape = new int[] { 1, 1, 1, 1 };

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
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            dispose(ref m_blobInputs);
            dispose(ref m_blobInputs1);
            dispose(ref m_blobHState);
            dispose(ref m_blobTs);
            dispose(ref m_blobHState1);
            dispose(ref m_blobForwardInput);
            dispose(ref m_blobForwardInput1);
            dispose(ref m_blobForwardInput2);
            dispose(ref m_blobForwardOutput);
            dispose(ref m_blobForwardOutput1);
            dispose(ref m_blobForwardOutput2);
            dispose(ref m_blobTimeSinceUpdate);
            dispose(ref m_blobTimeSinceUpdate1);
            dispose(ref m_blobMask);
            dispose(ref m_blobCurrentMask);
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
            LayerParameter p;

            m_nBatchSize = colBottom[0].num;
            m_nSeqLen = colBottom[0].channels;
            m_nTrueInFeatures = colBottom[0].count(2);

            m_rgShape[0] = m_nBatchSize;

            m_rgShape[1] = m_param.cfc_param.hidden_size;
            m_blobHState = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobHState1 = new Blob<T>(m_cuda, m_log, m_rgShape);

            m_rgShape[1] = 1;   
            m_blobTs = new Blob<T>(m_cuda, m_log, m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobInputs = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobInputs1 = new Blob<T>(m_cuda, m_log);
            m_blobMask = new Blob<T>(m_cuda, m_log);
            m_blobMask.ReshapeLike(m_blobInputs);
            m_nMaskCount = m_blobMask.count(2);

            m_rgShape[1] = 1;
            m_blobCurrentMask = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobCurrentOutput = new Blob<T>(m_cuda, m_log);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobForwardInput = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobForwardInput1 = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobForwardInput2 = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobTimeSinceUpdate = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobTimeSinceUpdate1 = new Blob<T>(m_cuda, m_log, m_rgShape);

            m_rgShape[1] = m_param.cfc_param.output_features;
            m_blobForwardOutput = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobForwardOutput1 = new Blob<T>(m_cuda, m_log, m_rgShape);
            m_blobForwardOutput2 = new Blob<T>(m_cuda, m_log, m_rgShape);

            m_blobOutputSequence = new Blob<T>(m_cuda, m_log);

            p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            p.concat_param.axis = 1;
            m_cat = Layer<T>.Create(m_cuda, m_log, p, null);

            addBtmTop(m_blobForwardInput, m_blobInputs1);
            if (m_nTrueInFeatures * 2 < m_param.cfc_param.input_features && m_nMaskCount == m_nTrueInFeatures)
                m_colBtm.Add(m_blobTimeSinceUpdate);
            m_colBtm.Add(m_blobMask);
            m_cat.Setup(m_colBtm, m_colTop);

            p = new LayerParameter(LayerParameter.LayerType.CFC_UNIT);
            p.cfc_unit_param.Copy(m_param.cfc_unit_param);
            m_rnn_cell = Layer<T>.Create(m_cuda, m_log, p, null);

            addBtmTop(m_blobInputs1, m_blobHState1);
            m_colBtm.Add(m_blobHState);
            m_colBtm.Add(m_blobTs);
            m_rnn_cell.Setup(m_colBtm, m_colTop);
            blobs.Add(m_rnn_cell.blobs);

            p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            p.inner_product_param.num_output = (uint)m_param.cfc_param.output_features;
            p.inner_product_param.bias_term = true;
            p.inner_product_param.weight_filler = new FillerParameter("xavier");
            p.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            p.inner_product_param.axis = 2;
            m_fc = Layer<T>.Create(m_cuda, m_log, p, null);

            addBtmTop(m_blobHState1, m_blobCurrentOutput);
            m_fc.Setup(m_colBtm, m_colTop);
            blobs.Add(m_fc.blobs);

            addBtmTop(m_blobHState1, colTop[0]);
            m_fc.Reshape(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nBatchSize = colBottom[0].num;
            m_nSeqLen = colBottom[0].channels;
            m_nTrueInFeatures = colBottom[0].count(2);

            m_rgShape[0] = m_nBatchSize;

            m_rgShape[1] = m_param.cfc_param.hidden_size;
            m_blobHState.Reshape(m_rgShape);
            m_blobHState1.Reshape(m_rgShape);

            m_rgShape[1] = 1;
            m_blobTs.Reshape(m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobInputs.Reshape(m_rgShape);
            m_blobMask.ReshapeLike(m_blobInputs);
            m_nMaskCount = m_blobMask.count(2);

            m_rgShape[1] = 1;
            m_blobCurrentMask.Reshape(m_rgShape);

            m_rgShape[1] = m_nTrueInFeatures;
            m_blobForwardInput.Reshape(m_rgShape);
            m_blobForwardInput1.Reshape(m_rgShape);
            m_blobForwardInput2.Reshape(m_rgShape);
            m_blobTimeSinceUpdate.Reshape(m_rgShape);
            m_blobTimeSinceUpdate1.Reshape(m_rgShape);

            m_rgShape[1] = m_param.cfc_param.output_features;
            m_blobForwardOutput.Reshape(m_rgShape);
            m_blobForwardOutput1.Reshape(m_rgShape);
            m_blobForwardOutput2.Reshape(m_rgShape);

            addBtmTop(m_blobForwardInput, m_blobInputs1);
            if (m_nTrueInFeatures * 2 < m_param.cfc_param.input_features && m_nMaskCount == m_nTrueInFeatures)
                m_colBtm.Add(m_blobTimeSinceUpdate);
            m_colBtm.Add(m_blobMask);
            m_cat.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobInputs1, m_blobHState1);
            m_colBtm.Add(m_blobHState);
            m_colBtm.Add(m_blobTs);
            m_rnn_cell.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobHState1, m_blobCurrentOutput);
            m_fc.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobHState1, colTop[0]);
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
            string strPath = "C:\\temp\\projects\\LNN\\PythonApplication2\\PythonApplication2\\test\\cfc_no_gate\\iter_0\\";
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);

            m_blobHState.SetData(0);
            m_blobForwardInput.SetData(0);
            m_blobForwardOutput.SetData(0);
            m_blobTimeSinceUpdate.SetData(0);

            for (int t = 0; t < m_nSeqLen; t++)
            {
                // Copy the t'th time step of the input to the input blob.
                m_cuda.channel_copy(m_blobInputs.count(), m_blobInputs.num, 1, m_nSeqLen, m_nTrueInFeatures, t, colBottom[0].gpu_data, m_blobInputs.mutable_gpu_data, DIR.FWD);
                // Copy the t'th timestep of the time since update to the ts blob.
                m_cuda.channel_copy(m_blobTs.count(), m_blobTs.num, 1, m_nSeqLen, 1, t, colBottom[1].gpu_data, m_blobTs.mutable_gpu_data, DIR.FWD);

                blobVal.LoadFromNumpy(strPath + "inputs_a_" + t.ToString() + ".npy");
                Trace.Assert(blobVal.Compare(m_blobInputs, blobWork));
                blobVal.LoadFromNumpy(strPath + "ts_a_" + t.ToString() + ".npy");
                Trace.Assert(blobVal.Compare(m_blobTs, blobWork));

                // Apply masking
                if (colBottom.Count() > 2)
                {
                    int nMaskCount = colBottom[2].count(2);
                    if (nMaskCount == m_nTrueInFeatures)
                    {
                        // Copy the t'th mask of the full mask to the mask blob.
                        m_cuda.channel_copy(m_blobMask.count(), m_blobMask.num, 1, m_nSeqLen, 1, t, colBottom[2].gpu_data, m_blobMask.mutable_gpu_data, DIR.FWD);
                        // Create the mask inverse
                        m_blobMask.SetDiff(1.0);
                        m_cuda.sub(m_blobMask.count(), m_blobMask.gpu_diff, m_blobMask.gpu_data, m_blobMask.mutable_gpu_diff);

                        blobVal.LoadFromNumpy(strPath + "mask1_" + t.ToString() + ".npy");
                        Trace.Assert(blobVal.Compare(m_blobMask, blobWork));
                        blobVal.LoadFromNumpy(strPath + "mask1_inv_" + t.ToString() + ".npy", true);
                        Trace.Assert(blobVal.Compare(m_blobMask, blobWork, true));

                        // Update the forwarded input.
                        m_cuda.mul(m_blobMask.count(), m_blobInputs.gpu_data, m_blobMask.gpu_data, m_blobForwardInput1.mutable_gpu_data);
                        m_cuda.mul(m_blobMask.count(), m_blobForwardInput.gpu_data, m_blobMask.gpu_diff, m_blobForwardInput2.mutable_gpu_data);
                        m_cuda.add(m_blobMask.count(), m_blobForwardInput1.gpu_data, m_blobForwardInput2.gpu_data, m_blobForwardInput.mutable_gpu_data);

                        blobVal.LoadFromNumpy(strPath + "forwarded_input_" + t.ToString() + ".npy");
                        Trace.Assert(blobVal.Compare(m_blobForwardInput, blobWork));

                        // Update the time since update.
                        m_cuda.add(m_blobTimeSinceUpdate.count(), m_blobTimeSinceUpdate.gpu_data, m_blobTs.gpu_data, m_blobTimeSinceUpdate1.mutable_gpu_data);
                        m_cuda.mul(m_blobTimeSinceUpdate1.count(), m_blobTimeSinceUpdate1.gpu_data, m_blobMask.gpu_diff, m_blobTimeSinceUpdate.mutable_gpu_data);

                        blobVal.LoadFromNumpy(strPath + "time_since_update_" + t.ToString() + ".npy");
                        Trace.Assert(blobVal.Compare(m_blobTimeSinceUpdate, blobWork));
                    }
                    else
                    {
                        m_cuda.copy(m_blobForwardInput.count(), m_blobInputs.gpu_data, m_blobForwardInput.mutable_gpu_data);

                        blobVal.LoadFromNumpy(strPath + "forwarded_input_" + t.ToString() + ".npy");
                        Trace.Assert(blobVal.Compare(m_blobForwardInput, blobWork));
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

                blobVal.LoadFromNumpy(strPath + "h_state_" + t.ToString() + ".npy");
                Trace.Assert(blobVal.Compare(m_blobHState, blobWork));
                blobVal.LoadFromNumpy(strPath + "inputs1_" + t.ToString() + ".npy");
                Trace.Assert(blobVal.Compare(m_blobInputs1, blobWork));
                blobVal.LoadFromNumpy(strPath + "ts_" + t.ToString() + ".npy");
                Trace.Assert(blobVal.Compare(m_blobTs, blobWork));

                // Run the CfcCell forward pass.
                addBtmTop(m_blobInputs1, m_blobHState1);
                m_colBtm.Add(m_blobHState);
                m_colBtm.Add(m_blobTs);
                m_rnn_cell.Forward(m_colBtm, m_colTop);

                blobVal.LoadFromNumpy(strPath + "h_state1_" + t.ToString() + ".npy");
                Trace.Assert(blobVal.Compare(m_blobHState1, blobWork));

                m_blobHState.CopyFrom(m_blobHState1);

                // Apply masking
                if (colBottom.Count > 2)
                {
                    m_cuda.channel_max(m_blobMask.count(), m_blobMask.num, m_blobMask.channels, 1, m_blobMask.gpu_data, m_blobCurrentMask.mutable_gpu_data);
                    
                    // Create mask inverse
                    m_blobCurrentMask.SetDiff(1.0);
                    m_cuda.sub(m_blobCurrentMask.count(), m_blobCurrentMask.gpu_diff, m_blobCurrentMask.gpu_data, m_blobCurrentMask.mutable_gpu_diff);

                    blobVal.LoadFromNumpy(strPath + "cur_mask_" + t.ToString() + ".npy");
                    Trace.Assert(blobVal.Compare(m_blobCurrentMask, blobWork));
                    blobVal.LoadFromNumpy(strPath + "cur_mask_inv_" + t.ToString() + ".npy", true);
                    Trace.Assert(blobVal.Compare(m_blobCurrentMask, blobWork, true));

                    addBtmTop(m_blobHState1, m_blobCurrentOutput);
                    m_fc.Setup(m_colBtm, m_colTop);

                    blobVal.LoadFromNumpy(strPath + "current_output_" + t.ToString() + ".npy");
                    Trace.Assert(blobVal.Compare(m_blobCurrentOutput, blobWork));

                    // Update the forwarded output.
                    m_cuda.mul(m_blobCurrentMask.count(), m_blobCurrentOutput.gpu_data, m_blobMask.gpu_data, m_blobForwardOutput1.mutable_gpu_data);
                    m_cuda.mul(m_blobCurrentMask.count(), m_blobForwardOutput.gpu_data, m_blobMask.gpu_diff, m_blobForwardOutput2.mutable_gpu_data);
                    m_cuda.add(m_blobCurrentMask.count(), m_blobForwardOutput1.gpu_data, m_blobForwardOutput2.gpu_data, m_blobForwardOutput.mutable_gpu_data);

                    blobVal.LoadFromNumpy(strPath + "forward_output_" + t.ToString() + ".npy");
                    Trace.Assert(blobVal.Compare(m_blobForwardOutput, blobWork));
                }
            }

            if (m_param.cfc_param.return_sequences)
            {
                throw new NotImplementedException("return sequences is not implemented yet.");
            }
            else if (colBottom.Count > 2)
            {
                colTop[0].CopyFrom(m_blobForwardOutput);

                blobVal.LoadFromNumpy(strPath + "readout.npy");
                Trace.Assert(blobVal.Compare(colTop[0], blobWork));
            }
            else
            {
                addBtmTop(m_blobHState1, colTop[0]);
                m_fc.Setup(m_colBtm, m_colTop);

                blobVal.LoadFromNumpy(strPath + "readout.npy");
                Trace.Assert(blobVal.Compare(colTop[0], blobWork));
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
        }
    }
}
