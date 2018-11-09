using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The LSTMUnitLayer is a helper for LSTMLayer that computes a single timestep of the
    /// non-linearity of the LSTM, producing the updated cell and hidden
    /// states.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LSTMUnitLayer<T> : Layer<T>
    {
        // The hidden output dimension.
        int m_nHiddenDim;
        Blob<T> m_blobXActs;

        /// <summary>
        /// The LSTMUnitLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LSTM_UNIT.</param>
        public LSTMUnitLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LSTM_UNIT;
            m_blobXActs = new Blob<T>(cuda, log);
            m_blobXActs.Name = m_param.name + " xacts";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobXActs != null)
            {
                m_blobXActs.Dispose();
                m_blobXActs = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobXActs);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (intput) Blobs: prevtime, gatein, seqcon
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: cellst, hiddenst
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns <i>true</i> for all but the bottom index = 2 for 
        /// you can't propagate to the sequence comtinuation indicators.
        /// </summary>
        /// <param name="nBottomIdx">Specifies the bottom index.</param>
        /// <returns>Returns whether or not to allow a forced backward.</returns>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            // Can't propagate to sequence contination indicators.
            if (nBottomIdx != 2)
                return true;

            return false;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumInstances = colBottom[0].shape(1);

            for (int i = 0; i < colBottom.Count; i++)
            {
                if (i == 2)
                    m_log.CHECK_EQ(2, colBottom[i].num_axes, "There should be 2 axes at bottom[2]");
                else
                    m_log.CHECK_EQ(3, colBottom[i].num_axes, "There should be 3 axes at bottom[" + i.ToString() + "]");

                m_log.CHECK_EQ(1, colBottom[i].shape(0), "The shape(0) at bottom[" + i.ToString() + "] should be 1.");
                m_log.CHECK_EQ(nNumInstances, colBottom[i].shape(1), "The shape(1) at bottom[" + i.ToString() + "] should equal the number of instances (" + nNumInstances.ToString() + ")");                
            }

            m_nHiddenDim = colBottom[0].shape(2);
            m_log.CHECK_EQ(4 * m_nHiddenDim, colBottom[1].shape(2), "The bottom[1].shape(2) should equal the 4 * the number of hidden dimensions (4 x " + m_nHiddenDim.ToString() + ")");
            colTop[0].ReshapeLike(colBottom[0]);
            colTop[1].ReshapeLike(colBottom[0]);
            m_blobXActs.ReshapeLike(colBottom[1]);
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 3)
        ///  -# @f$ (1 \times N \times D) @f$
        ///     the previous timestep cell state @f$ c_{t-1} @f$
        ///  -# @f$ (1 \times N \times D) @f$
        ///     the 'gate inputs' @f$ [i_t', f_t', o_t', g_t'] @f$
        ///  -# @f$ (1 \times N) @f$
        ///     the sequence continuation indicators @f$ \delta_t @f$
        /// </param>
        /// <param name="colTop">output blob vector (length 2)
        ///  -# @f$ (1 \times N \times d) @f$
        ///     the updated cell state @f$ c_t @f$, computed as 
        ///     <code>
        ///     i_t := sigmoid[i_t']
        ///     f_t := sigmoid[f_t']
        ///     o_t := sigmoid[o_t']
        ///     g_t := tanh[g_t']
        ///     c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
        ///     </code>
        ///  -# @f$ (1 \times N \times D) @f$
        ///     the updated hidden state @f$ h_t @f$, computed as:
        ///         @f$ h_t := o_t .* \tanh[c_t] @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colTop[1].count();
            long hC_prev = colBottom[0].gpu_data;
            long hX = colBottom[1].gpu_data;
            long hCont = colBottom[2].gpu_data;
            long hX_acts = m_blobXActs.mutable_gpu_data;
            long hC = colTop[0].mutable_gpu_data;
            long hH = colTop[1].mutable_gpu_data;
            int nXCount = colBottom[1].count();

            m_cuda.lstm_unit_fwd(nCount, m_nHiddenDim, nXCount, hX, hX_acts, hC_prev, hCont, hC, hH);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the LSTMUnit inputs.
        /// </summary>
        /// <param name="colTop">output Blob vector (length 2), providing the error gradient
        /// w.r.t. the outputs.
        ///  -# @f$ (1 \times N \times D) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial c_t} @f$
        ///     w.r.t. the updated cell state @f$ c_t @f$.
        ///  -# @f$ (1 \times N \times D) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial h_t} @f$
        ///     w.r.t. the updated cell state @f$ h_t @f$.</param>
        /// <param name="rgbPropagateDown">See Layer::Backward.</param>
        /// <param name="colBottom">input Blob vector (length 3), into which the error gradients
        /// w.r.t. the LSTMUnit inputs @f$ c_{t-1} @f$, and the gate inputs are computed.  Computation 
        /// of the error gradients w.r.t. the sequence indicators is not implemented.
        ///  -# @f$ (1 \times N \times D) @f$
        ///     the error gradient w.r.t. the previous timestep cells tate @f$ c_{t-1} @f$
        ///  -# @f$ (1 \times N \times 4D) @f$
        ///     the error gradient w.r.t. the 'gate inputs' @f$
        ///         [
        ///             \frac{\partial E}{\partial 'i_t'}
        ///             \frac{\partial E}{\partial 'f_t'}
        ///             \frac{\partial E}{\partial 'o_t'}
        ///             \frac{\partial E}{\partial 'g_t'}
        ///         ]
        ///     @f$
        ///  -# @f$(1 \times 1 \times N) @f$
        ///     the gradient w.r.t. the sequence continuation indicators @f$ \delta_t @f$ 
        ///     is currently not implemented.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_log.CHECK(!rgbPropagateDown[2], "Cannot backpropagate to sequence indicators.");

            if (!rgbPropagateDown[0] && !rgbPropagateDown[1])
                return;

            int nCount = colTop[1].count();
            long hC_prev = colBottom[0].gpu_data;
            long hX_acts = m_blobXActs.gpu_data;
            long hCont = colBottom[2].gpu_data;
            long hC = colTop[0].gpu_data;
            long hH = colTop[1].gpu_data;
            long hC_diff = colTop[0].gpu_diff;
            long hH_diff = colTop[1].gpu_diff;
            long hC_prev_diff = colBottom[0].mutable_gpu_diff;
            long hX_acts_diff = m_blobXActs.mutable_gpu_diff;
            int nXCount = colBottom[1].count();
            long hX_diff = colBottom[1].mutable_gpu_diff;

            m_cuda.lstm_unit_bwd(nCount, m_nHiddenDim, nXCount, hC_prev, hX_acts, hC, hH, hCont, hC_diff, hH_diff, hC_prev_diff, hX_acts_diff, hX_diff);
        }
    }
}
