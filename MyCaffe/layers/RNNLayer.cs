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
    /// The RNNLayer processes time-varying inputs using a simple recurrent neural network (RNN).  Implemented
    /// as a network unrolling the RNN computation in time.
    /// This layer is initialized with the MyCaffe.param.RecurrentParameter.
    /// </summary>
    /// <remarks>
    /// Given time-varying inputs @f$ x_t @f$, computes hidden state @f$
    ///     h_t := \tanh[ W_{hh} h_{t_1} + W_{xh} x_t + b_h ]
    /// @f$, and outputs @f$
    ///     o_t := \tanh[ W_{ho} h_t + b_o ]
    /// @f$.
    /// 
    /// @see [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) by Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu, , 2016.
    /// @see [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798) by Meire Fotunato, Charles Blundell, and Oriol Vinyals, 2017. 
    /// @see [Higher Order Recurrent Neural Networks](https://arxiv.org/abs/1605.00064) by Rohollah Soltani and Hui Jiang, 2016.
    /// @see [Hierarchical Multiscale Recurrent Neural Networks](https://arxiv.org/abs/1609.01704) by Junyoung Chung, Sungjin Ahn, and Yoshua Bengio, 2016. 
    /// @see [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148) by George Toderici, Damien Vincent, Nick Johnston, Sung Jin Hwang, David Minnen, Joel Shor, and Michele Covell, 2016.
    /// @see [ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](https://arxiv.org/abs/1505.00393v3) by Francesco Visin, Kyle Kastner, Kyunghyun Cho, Matteo Matteucci, Aaron Courville, and Yoshua Bengio, 2015.
    /// @see [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144v2) by Wu, et al., 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class RNNLayer<T> : RecurrentLayer<T>
    {
        /// <summary>
        /// The RNNLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type RNN with parameter recurrent_param,
        /// with options:
        ///   - num_output.  The dimension of the output (and ususally hidden state) representation -- must be explicitly set to non-zero.
        ///   
        ///   - weight_filler (/b optional, default = "gaussian"). The weight filler used to initialize the weights.
        ///   
        ///   - bias_filler (/b optional, default = "constant, 1.0"). The bias filler used to initialize the bias values.
        ///   
        ///   - debug_info (/b optional, default = false). Whether or not to output extra debug information.
        ///   
        ///   - expose_hidden (/b optional, default = false).  Whether t add as additional bottom (inputs) the initial hidden state
        ///     Blobs, and add a additional top (output) the final timestep hidden state Blobs.  The RNN architecture adds
        ///     1 additional Blobs.
        /// </param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel training operations.</param>
        public RNNLayer(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel) 
            : base(cuda, log, p, evtCancel)
        {
            m_type = LayerParameter.LayerType.RNN;
        }

        /// <summary>
        /// Fills the <i>rgNames</i> array with the names of the 0th timestep recurrent input Blobs.
        /// </summary>
        /// <param name="rgNames">Specifies the list of names to fill.</param>
        protected override void RecurrentInputBlobNames(List<string> rgNames)
        {
            rgNames.Clear();
            rgNames.Add("h_0");
        }

        /// <summary>
        /// Fills the <i>rgNames</i> array with names of the Tth timestep recurrent output Blobs.
        /// </summary>
        /// <param name="rgNames">Specifies the list of names to fill.</param>
        protected override void RecurrentOutputBlobNames(List<string> rgNames)
        {
            rgNames.Clear();
            rgNames.Add("h_" + m_nT.ToString());
        }

        /// <summary>
        /// Fill the <i>rgShapes</i> array with the shapes of the recurrent input Blobs.
        /// </summary>
        /// <param name="rgShapes">Specifies the array of BlobShape to fill.</param>
        protected override void RecurrentInputShapes(List<BlobShape> rgShapes)
        {
            rgShapes.Clear();

            BlobShape s = new param.BlobShape();
            s.dim.Add(1);   // a single timestep
            s.dim.Add(m_nN);
            s.dim.Add((int)m_param.recurrent_param.num_output);
            rgShapes.Add(s);
        }

        /// <summary>
        /// Fills the <i>rgNames</i> array with  the names of the output
        /// Blobs, concatenated across all timesteps.
        /// </summary>
        /// <param name="rgNames">Specifies the array of names to fill.</param>
        protected override void OutputBlobNames(List<string> rgNames)
        {
            rgNames.Clear();
            rgNames.Add("o");
        }

        /// <summary>
        /// Fills the NetParameter  with the RNN network architecture.
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
            hidden_param.inner_product_param.num_output = nNumOutput;
            hidden_param.inner_product_param.bias_term = false;
            hidden_param.inner_product_param.axis = 2;
            hidden_param.inner_product_param.weight_filler = weight_filler.Clone();

            LayerParameter biased_hidden_param = hidden_param.Clone(false);
            biased_hidden_param.inner_product_param.bias_term = true;
            biased_hidden_param.inner_product_param.bias_filler = bias_filler.Clone();

            LayerParameter sum_param = new param.LayerParameter(LayerParameter.LayerType.ELTWISE);
            sum_param.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;

            LayerParameter tanh_param = new LayerParameter(LayerParameter.LayerType.TANH);

            LayerParameter scale_param = new LayerParameter(LayerParameter.LayerType.SCALE);
            scale_param.scale_param.axis = 0;

            LayerParameter slice_param = new LayerParameter(LayerParameter.LayerType.SLICE);
            slice_param.slice_param.axis = 0;

            List<BlobShape> rgInputShapes = new List<BlobShape>();
            RecurrentInputShapes(rgInputShapes);
            m_log.CHECK_EQ(1, rgInputShapes.Count, "There should only be one input shape.");


            //--- Add the layers ---

            LayerParameter input_layer_param = new LayerParameter(LayerParameter.LayerType.INPUT);
            input_layer_param.top.Add("h_0");
            input_layer_param.input_param.shape.Add(rgInputShapes[0]);
            net_param.layer.Add(input_layer_param);

            LayerParameter cont_slice_param = slice_param.Clone(false);
            cont_slice_param.name = "cont_slice";
            cont_slice_param.bottom.Add("cont");
            cont_slice_param.slice_param.axis = 0;
            net_param.layer.Add(cont_slice_param);

            // Add layer to transform all timesteps of x to the hidden state dimension.
            //  W_xh_x = W_xh * x + b_h
            {
                LayerParameter x_transform_param = biased_hidden_param.Clone(false);
                x_transform_param.name = "x_transform";
                x_transform_param.parameters.Add(new ParamSpec("W_xh"));
                x_transform_param.parameters.Add(new ParamSpec("b_h"));
                x_transform_param.bottom.Add("x");
                x_transform_param.top.Add("W_xh_x");
                x_transform_param.propagate_down.Add(true);
                net_param.layer.Add(x_transform_param);
            }

            if (m_bStaticInput)
            {
                // Add layer to transform x_static to the hidden state dimension.
                //  W_xh_x_static = W_xh_static * x_static
                LayerParameter x_static_transform_param = hidden_param.Clone(false);
                x_static_transform_param.inner_product_param.axis = 1;
                x_static_transform_param.name = "W_xh_x_static";
                x_static_transform_param.parameters.Add(new ParamSpec("W_xh_static"));
                x_static_transform_param.bottom.Add("x_static");
                x_static_transform_param.top.Add("W_xh_x_static_preshape");
                x_static_transform_param.propagate_down.Add(true);
                net_param.layer.Add(x_static_transform_param);

                LayerParameter reshape_param = new LayerParameter(LayerParameter.LayerType.RESHAPE);
                BlobShape new_shape = reshape_param.reshape_param.shape;
                new_shape.dim.Add(1);   // One timestep.
                new_shape.dim.Add(-1);  // Should infer m_nN as the dimension so we can reshape on batch size.
                new_shape.dim.Add((int)x_static_transform_param.inner_product_param.num_output);
                reshape_param.name = "W_xh_x_static_reshape";
                reshape_param.bottom.Add("W_xh_x_static_preshape");
                reshape_param.top.Add("W_xh_x_static");
                net_param.layer.Add(reshape_param);
            }

            LayerParameter x_slice_param = slice_param.Clone(false);
            x_slice_param.name = "W_xh_x_slice";
            x_slice_param.bottom.Add("W_xh_x");
            net_param.layer.Add(x_slice_param);

            LayerParameter output_concat_layer = new LayerParameter(LayerParameter.LayerType.CONCAT);
            output_concat_layer.name = "o_concat";
            output_concat_layer.top.Add("o");
            output_concat_layer.concat_param.axis = 0;

            for (int t = 1; t <= m_nT; t++)
            {
                string tm1s = (t - 1).ToString();
                string ts = t.ToString();

                cont_slice_param.top.Add("cont_" + ts);
                x_slice_param.top.Add("W_xh_x_" + ts);


                // Add layer to flush the hidden state when beginning a new sequence,
                //  as indicated by cont_t.
                //      h_conted_{t-1} := cont_t * h_{t-1}
                //
                //  Normally, cont_t is binary (i.e., 0 or 1), so:
                //      h_conted_{t-1} := h_{t-1} if cont_t == 1
                //                        0 otherwise.
                {
                    LayerParameter cont_h_param = scale_param.Clone(false);
                    cont_h_param.name = "h_conted_" + tm1s;
                    cont_h_param.bottom.Add("h_" + tm1s);
                    cont_h_param.bottom.Add("cont_" + ts);
                    cont_h_param.top.Add("h_conted_" + tm1s);
                    net_param.layer.Add(cont_h_param);
                }

                // Add layer to compute
                //     W_hh_h_{t-1} := W_hh * h_conted_{t-1}
                {
                    LayerParameter w_param = hidden_param.Clone(false);
                    w_param.name = "W_hh_h_" + tm1s;
                    w_param.parameters.Add(new ParamSpec("W_hh"));
                    w_param.bottom.Add("h_conted_" + tm1s);
                    w_param.top.Add("W_hh_h_" + tm1s);
                    w_param.inner_product_param.axis = 2;
                    net_param.layer.Add(w_param);
                }

                // Add layers to compute
                //      h_t := \tanh( W_hh * h_conted_t{t-1} + W_xh * x_t + b_h )
                //           = \tanh( W_hh_h_{t-1} + W_xh_t )
                {
                    LayerParameter h_input_sum_param = sum_param.Clone(false);
                    h_input_sum_param.name = "h_input_sum_" + ts;
                    h_input_sum_param.bottom.Add("W_hh_h_" + tm1s);
                    h_input_sum_param.bottom.Add("W_xh_x_" + ts);

                    if (m_bStaticInput)
                        h_input_sum_param.bottom.Add("W_xh_x_static");

                    h_input_sum_param.top.Add("h_neuron_input_" + ts);
                    net_param.layer.Add(h_input_sum_param);
                }
                {
                    LayerParameter h_neuron_param = tanh_param.Clone(false);
                    h_neuron_param.name = "h_neuron_input_" + ts;
                    h_neuron_param.bottom.Add("h_neuron_input_" + ts);
                    h_neuron_param.top.Add("h_" + ts);
                    net_param.layer.Add(h_neuron_param);
                }

                // Add layer to compute
                //      W_ho_h_t := W_ho * h_t + b_o
                {
                    LayerParameter w_param = biased_hidden_param.Clone(false);
                    w_param.name = "W_ho_h_" + ts;
                    w_param.parameters.Add(new ParamSpec("W_ho"));
                    w_param.parameters.Add(new ParamSpec("b_o"));
                    w_param.bottom.Add("h_" + ts);
                    w_param.top.Add("W_ho_h_" + ts);
                    w_param.inner_product_param.axis = 2;
                    net_param.layer.Add(w_param);
                }

                // Add layer to compute
                //      o_t := \tanh( W_ho * h_t + b_o
                //           = \tanh( W_ho_h_t )
                {
                    LayerParameter o_neuron_param = tanh_param.Clone(false);
                    o_neuron_param.name = "o_neuron_" + ts;
                    o_neuron_param.bottom.Add("W_ho_h_" + ts);
                    o_neuron_param.top.Add("o_" + ts);
                    net_param.layer.Add(o_neuron_param);
                }

                output_concat_layer.bottom.Add("o_" + ts);
            }

            net_param.layer.Add(output_concat_layer.Clone(false));
        }
    }
}
