using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The LSTMLayer processes sequential inputs using a 'Long Short-Term Memory' (LSTM)
    /// [1] style recurrent neural network (RNN). Implemented by unrolling
    /// the LSTM computation through time.
    /// This layer is initialized with the MyCaffe.param.RecurrentParameter.
    /// </summary>
    /// <remarks>
    /// The specific architecture used in this implementation is a described
    /// in "Learning to Execute" [2], reproduced below: 
    /// <code>
    /// i_t := sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
    /// f_t := sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
    /// o_t := sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
    /// g_t :=    tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
    /// c_t := (f_t .* c_{t-1} + (i_t .* g_t)
    /// h_t := o_t .* tanh[c_t]
    /// </code>
    /// In the implementation, the i, f, o, and g computation are preformed as a
    /// single inner product.
    /// 
    /// Notably, this implementation lacks the 'diagonal' gates, as used in the
    /// LSTM architectures described by Alex Graves [3] and others.
    /// 
    /// [1] Hochreiter, Sepp, and Schmidhuber, Jurgen. [Long short-term memory](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.7752).
    ///     Neural Computation 9, no. 8 (1997): 1735-1780.
    ///     
    /// [2] Zaremba, Wojciech, and Sutskever, Ilya. [Learning to execute](https://arxiv.org/abs/1410.4615).
    ///     arXiv preprint arXiv: 1410.4615 (2014).
    ///     
    /// [3] Graves, Alex. [Generating sequences with recurrent neural networks](https://arxiv.org/abs/1308.0850).
    ///     arXiv preprint arXiv: 1308.0850 (2013).
    ///     
    /// @see [A Clockwork RNN](https://arxiv.org/abs/1402.3511) by Jan Koutnik, Klaus Greff, Faustino Gomez, and Jürgen Schmidhuber, 2014.
    /// @see [Predictive Business Process Monitoring with LSTM Neural Networks](https://arxiv.org/abs/1612.02130) by Niek Tax, Ilya Verenich, Marcello La Rosa, and Marlon Dumas, 2016. 
    /// @see [Using LSTM recurrent neural networks for detecting anomalous behavior of LHC superconducting magnets](https://arxiv.org/abs/1611.06241) by Maciej Wielgosz, Andrzej Skoczeń, and Matej Mertik, 2016.
    /// @see [Spatial, Structural and Temporal Feature Learning for Human Interaction Prediction](https://arxiv.org/abs/1608.05267v2) by Qiuhong Ke, Mohammed Bennamoun, Senjian An, Farid Bossaid, and Ferdous Sohel, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LSTMLayer<T> : RecurrentLayer<T>
    {
        /// <summary>
        /// The LSTMLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LSTM with parameter recurrent_param,
        /// with options:
        ///   - num_output.  The dimension of the output (and ususally hidden state) representation -- must be explicitly set to non-zero.
        ///   
        ///   - weight_filler (\b optional, default = "gaussian"). The weight filler used to initialize the weights.
        ///   
        ///   - bias_filler (\b optional, default = "constant, 1.0"). The bias filler used to initialize the bias values.
        ///   
        ///   - debug_info (\b optional, default = false). Whether or not to output extra debug information.
        ///   
        ///   - expose_hidden (\b optional, default = false).  Whether @f$ t @f$ add as additional bottom (inputs) the initial hidden state
        ///     Blob%s, and add a additional top (output) the final timestep hidden state Blob%s.  The LSTM architecture adds
        ///     2 additional Blob%s.
        /// </param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel training operations.</param>
        public LSTMLayer(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel) 
            : base(cuda, log, p, evtCancel)
        {
            m_type = LayerParameter.LayerType.LSTM;
        }

        /// <summary>
        /// Fills the <i>rgNames</i> array with the names of the 0th timestep recurrent input Blobs.
        /// </summary>
        /// <param name="rgNames">Specifies the list of names to fill.</param>
        protected override void RecurrentInputBlobNames(List<string> rgNames)
        {
            rgNames.Clear();
            rgNames.Add("h_0");
            rgNames.Add("c_0");
        }

        /// <summary>
        /// Fills the <i>rgNames</i> array with names of the Tth timestep recurrent output Blobs.
        /// </summary>
        /// <param name="rgNames">Specifies the list of names to fill.</param>
        protected override void RecurrentOutputBlobNames(List<string> rgNames)
        {
            rgNames.Clear();
            rgNames.Add("h_" + m_nT.ToString());
            rgNames.Add("c_T");
        }

        /// <summary>
        /// Fill the <i>rgShapes</i> array with the shapes of the recurrent input Blobs.
        /// </summary>
        /// <param name="rgShapes">Specifies the array of BlobShape to fill.</param>
        protected override void RecurrentInputShapes(List<BlobShape> rgShapes)
        {
            int nNumBlobs = 2;

            rgShapes.Clear();

            for (int i = 0; i < nNumBlobs; i++)
            {
                BlobShape s = new param.BlobShape();
                s.dim.Add(1);   // a single timestep
                s.dim.Add(m_nN);
                s.dim.Add((int)m_param.recurrent_param.num_output);
                rgShapes.Add(s);
            }
        }

        /// <summary>
        /// Fills the <i>rgNames</i> array with  the names of the output
        /// Blobs, concatenated across all timesteps.
        /// </summary>
        /// <param name="rgNames">Specifies the array of names to fill.</param>
        protected override void OutputBlobNames(List<string> rgNames)
        {
            rgNames.Clear();
            rgNames.Add("h");
        }

        /// <summary>
        /// Fills the NetParameter  with the LSTM network architecture.
        /// </summary>
        /// <param name="net_param"></param>
        protected override void FillUnrolledNet(NetParameter net_param)
        {
            uint nNumOutput = m_param.recurrent_param.num_output;
            m_log.CHECK_GT(nNumOutput, 0, "num_output must be positive.");
            FillerParameter weight_filler = m_param.recurrent_param.weight_filler;
            FillerParameter bias_filler = m_param.recurrent_param.bias_filler;

            // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
            // use to save redundant code.
            LayerParameter hidden_param = new param.LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            hidden_param.inner_product_param.num_output = nNumOutput * 4;
            hidden_param.inner_product_param.bias_term = false;
            hidden_param.inner_product_param.axis = 2;
            hidden_param.inner_product_param.weight_filler = weight_filler.Clone();

            LayerParameter biased_hidden_param = hidden_param.Clone(false);
            biased_hidden_param.inner_product_param.bias_term = true;
            biased_hidden_param.inner_product_param.bias_filler = bias_filler.Clone();

            LayerParameter sum_param = new param.LayerParameter(LayerParameter.LayerType.ELTWISE);
            sum_param.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;

            LayerParameter scale_param = new LayerParameter(LayerParameter.LayerType.SCALE);
            scale_param.scale_param.axis = 0;

            LayerParameter slice_param = new LayerParameter(LayerParameter.LayerType.SLICE);
            slice_param.slice_param.axis = 0;

            LayerParameter split_param = new LayerParameter(LayerParameter.LayerType.SPLIT);

            List<BlobShape> rgInputShapes = new List<BlobShape>();
            RecurrentInputShapes(rgInputShapes);
            m_log.CHECK_EQ(2, rgInputShapes.Count, "There should be 2 input shapes.");


            //--- Add the layers ---

            LayerParameter input_layer_param = new LayerParameter(LayerParameter.LayerType.INPUT);
            input_layer_param.top.Add("c_0");
            input_layer_param.input_param.shape.Add(rgInputShapes[0].Clone());
            input_layer_param.top.Add("h_0");
            input_layer_param.input_param.shape.Add(rgInputShapes[1].Clone());
            net_param.layer.Add(input_layer_param);

            LayerParameter cont_slice_param = slice_param.Clone(false);
            cont_slice_param.name = "cont_slice";
            cont_slice_param.bottom.Add("cont");
            cont_slice_param.slice_param.axis = 0;
            net_param.layer.Add(cont_slice_param);

            // Add layer to transform all timesteps of x to the hidden state dimension.
            //  W_xc_x = W_xc * x + b_c
            {
                LayerParameter x_transform_param = biased_hidden_param.Clone(false);
                x_transform_param.name = "x_transform";
                x_transform_param.parameters.Add(new ParamSpec("W_xc"));
                x_transform_param.parameters.Add(new ParamSpec("b_c"));
                x_transform_param.bottom.Add("x");
                x_transform_param.top.Add("W_xc_x");
                x_transform_param.propagate_down.Add(true);
                net_param.layer.Add(x_transform_param);
            }

            if (m_bStaticInput)
            {
                // Add layer to transform x_static to the hidden state dimension.
                //  W_xc_x_static = W_xc_static * x_static
                LayerParameter x_static_transform_param = hidden_param.Clone(false);
                x_static_transform_param.inner_product_param.axis = 1;
                x_static_transform_param.name = "W_xc_x_static";
                x_static_transform_param.parameters.Add(new ParamSpec("W_xc_static"));
                x_static_transform_param.bottom.Add("x_static");
                x_static_transform_param.top.Add("W_xc_x_static_preshape");
                x_static_transform_param.propagate_down.Add(true);
                net_param.layer.Add(x_static_transform_param);

                LayerParameter reshape_param = new LayerParameter(LayerParameter.LayerType.RESHAPE);
                BlobShape new_shape = reshape_param.reshape_param.shape;
                new_shape.dim.Add(1);   // One timestep.
                new_shape.dim.Add(-1);  // Should infer m_nN as the dimension so we can reshape on batch size.
                new_shape.dim.Add((int)x_static_transform_param.inner_product_param.num_output);
                reshape_param.name = "W_xc_x_static_reshape";
                reshape_param.bottom.Add("W_xc_x_static_preshape");
                reshape_param.top.Add("W_xc_x_static");
                net_param.layer.Add(reshape_param);
            }

            LayerParameter x_slice_param = slice_param.Clone(false);
            x_slice_param.name = "W_xc_x_slice";
            x_slice_param.bottom.Add("W_xc_x");
            net_param.layer.Add(x_slice_param);

            LayerParameter output_concat_layer = new LayerParameter(LayerParameter.LayerType.CONCAT);
            output_concat_layer.name = "h_concat";
            output_concat_layer.top.Add("h");
            output_concat_layer.concat_param.axis = 0;

            for (int t = 1; t <= m_nT; t++)
            {
                string tm1s = (t - 1).ToString();
                string ts = t.ToString();

                cont_slice_param.top.Add("cont_" + ts);
                x_slice_param.top.Add("W_xc_x_" + ts);


                // Add layer to flush the hidden state when beginning a new sequence,
                //  as indicated by cont_t.
                //      h_conted_{t-1} := cont_t * h_{t-1}
                //
                //  Normally, cont_t is binary (i.e., 0 or 1), so:
                //      h_conted_{t-1} := h_{t-1} if cont_t == 1
                //                        0 otherwise.
                {
                    LayerParameter cont_h_param = scale_param.Clone(false);
                    cont_h_param.group_start = true;
                    cont_h_param.name = "h_conted_" + tm1s;
                    cont_h_param.bottom.Add("h_" + tm1s);
                    cont_h_param.bottom.Add("cont_" + ts);
                    cont_h_param.top.Add("h_conted_" + tm1s);
                    net_param.layer.Add(cont_h_param);
                }

                // Add layer to compute
                //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
                {
                    LayerParameter w_param = hidden_param.Clone(false);
                    w_param.name = "transform_" + ts;
                    w_param.parameters.Add(new ParamSpec("W_hc"));
                    w_param.bottom.Add("h_conted_" + tm1s);
                    w_param.top.Add("W_hc_h_" + tm1s);
                    w_param.inner_product_param.axis = 2;
                    net_param.layer.Add(w_param);
                }

                // Add the outputs of the linear transformations to compute the gate input.
                //  get_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
                //               = W_hc_h_{t-1} + W_xc_x_t + b_c
                {
                    LayerParameter input_sum_layer = sum_param.Clone(false);
                    input_sum_layer.name = "gate_input_" + ts;
                    input_sum_layer.bottom.Add("W_hc_h_" + tm1s);
                    input_sum_layer.bottom.Add("W_xc_x_" + ts);
                    if (m_bStaticInput)
                        input_sum_layer.bottom.Add("W_xc_x_static");
                    input_sum_layer.top.Add("gate_input_" + ts);
                    net_param.layer.Add(input_sum_layer);
                }

                // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
                //  Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
                //  Outputs: c_t, h_t
                //      [ i_t' ] 
                //      [ f_t' ] := gate_input_t
                //      [ o_t' ]
                //      [ g_t' ]
                //          i_t := \sigmoid[i_t']
                //          f_t := \sigmoid[f_t']
                //          o_t := \sigmoid[o_t']
                //          g_t := \tanh[g_t']
                //          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
                //          h_t := o_t .* \tanh[c_t]
                {
                    LayerParameter lstm_unit_param = new LayerParameter(LayerParameter.LayerType.LSTM_UNIT);
                    lstm_unit_param.bottom.Add("c_" + tm1s);
                    lstm_unit_param.bottom.Add("gate_input_" + ts);
                    lstm_unit_param.bottom.Add("cont_" + ts);
                    lstm_unit_param.top.Add("c_" + ts);
                    lstm_unit_param.top.Add("h_" + ts);
                    lstm_unit_param.name = "unit_" + ts;
                    net_param.layer.Add(lstm_unit_param);
                }

                output_concat_layer.bottom.Add("h_" + ts);
            }

            {
                LayerParameter c_T_copy_param = split_param.Clone(false);
                c_T_copy_param.bottom.Add("c_" + m_nT.ToString());
                c_T_copy_param.top.Add("c_T");
                net_param.layer.Add(c_T_copy_param); 
            }

            net_param.layer.Add(output_concat_layer.Clone(false));
        }
    }
}
