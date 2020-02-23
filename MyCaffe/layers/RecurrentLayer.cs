using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.layers
{
    /// <summary>
    /// The RecurrentLayer is an abstract class for implementing recurrent behavior inside of an
    /// unrolled newtork.  This layer type cannot be instantiated -- instead,
    /// you should use one of teh implementations which defines the recurrent
    /// architecture, such as RNNLayer or LSTMLayer.
    /// This layer is initialized with the MyCaffe.param.RecurrentParameter.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class RecurrentLayer<T> : Layer<T>
    {
        /// <summary>
        /// A Net to implement the Recurrent functionality.
        /// </summary>
        Net<T> m_unrolledNet = null;

        /// <summary>
        /// The number of independent streams to process simultaneously.
        /// </summary>
        protected int m_nN;

        /// <summary>
        /// The number of timesteps in the layer's input, and the number of 
        /// timesteps over which to backpropagate through time.
        /// </summary>
        protected int m_nT;

        /// <summary>
        /// Whether the layer has a 'static' input copies across all timesteps.
        /// </summary>
        protected bool m_bStaticInput;

        /// <summary>
        /// The last layer to run in the network.  (Any later layers are losses
        /// added to force the recurrent net to do backprop.)
        /// </summary>
        int m_nLastLayerIndex;

        /// <summary>
        /// Whether the layer's hidden state at the first and last timesteps
        /// are layer inputs and outputs, respectively.
        /// </summary>
        bool m_bExposeHidden;

        BlobCollection<T> m_colRecurInputBlobs = new BlobCollection<T>();
        BlobCollection<T> m_colRecurOutputBlobs = new BlobCollection<T>();
        BlobCollection<T> m_colOutputBlobs = new BlobCollection<T>();
        Blob<T> m_blobXInputBlob;       
        Blob<T> m_blobXStaticInputBlob; 
        Blob<T> m_blobContInputBlob;
        CancelEvent m_evtCancel;

        // cuDNN Specific Members
        long m_hCuDnn;
        int m_nInputSize = 1;
        int m_nHiddenSize;
        int m_nNumLayers;
        Blob<T> m_blobX;
        Blob<T> m_blobHx;
        Blob<T> m_blobCx;
        Blob<T> m_blobY;
        Blob<T> m_blobHy;
        Blob<T> m_blobCy;
        Blob<T> m_blobWts;
        long m_hXDesc;
        long m_hYDesc;
        long m_hHxDesc;
        long m_hCxDesc;
        long m_hHyDesc;
        long m_hCyDesc;
        long m_hDropoutDesc;
        long m_hDropoutStates;
        long m_hWeightDesc;
        long m_hRnnDesc;
        long m_hWorkspace;
        int m_nWorkspaceCount;
        long m_hReserved;
        int m_nReservedCount;
        RNN_MODE m_rnnMode;
        bool m_bUseTensors = false;

        /// <summary>
        /// The RecurrentLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LSTM or RNN.
        /// </param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel training operations.</param>
        public RecurrentLayer(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel)
            : base(cuda, log, p)
        {
            m_evtCancel = evtCancel;

            if (p.type == LayerParameter.LayerType.LSTM)
                m_rnnMode = RNN_MODE.LSTM;
            else
                m_rnnMode = RNN_MODE.RNN_RELU;
        }

        private void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        private void free_tensor(ref long h)
        {
            if (h != 0)
            {
                m_cuda.FreeTensorDesc(h);
                h = 0;
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_unrolledNet != null)
            {
                m_unrolledNet.Dispose();
                m_unrolledNet = null;
            }

            dispose(ref m_blobHx);
            dispose(ref m_blobCx);
            dispose(ref m_blobHy);
            dispose(ref m_blobCy);
            dispose(ref m_blobWts);

            free_tensor(ref m_hHxDesc);
            free_tensor(ref m_hCxDesc);
            free_tensor(ref m_hHyDesc);
            free_tensor(ref m_hCyDesc);

            if (m_hWeightDesc != 0)
            {
                m_cuda.FreeFilterDesc(m_hWeightDesc);
                m_hWeightDesc = 0;
            }

            if (m_hRnnDesc != 0)
            {
                m_cuda.FreeRnnDesc(m_hRnnDesc);
                m_hRnnDesc = 0;
            }

            if (m_hDropoutDesc != 0)
            {
                m_cuda.FreeDropoutDesc(m_hDropoutDesc);
                m_hDropoutDesc = 0;
            }

            if (m_hDropoutStates != 0)
            {
                m_cuda.FreeMemory(m_hDropoutStates);
                m_hDropoutStates = 0;
            }

            if (m_hXDesc != 0)
            {
                m_cuda.FreeRnnDataDesc(m_hXDesc);
                m_hXDesc = 0;
            }

            if (m_hYDesc != 0)
            {
                m_cuda.FreeRnnDataDesc(m_hYDesc);
                m_hYDesc = 0;
            }

            if (m_hWorkspace != 0)
            {
                m_cuda.FreeMemory(m_hWorkspace);
                m_hWorkspace = 0;
            }

            if (m_hReserved != 0)
            {
                m_cuda.FreeMemory(m_hReserved);
                m_hReserved = 0;
            }

            if (m_hCuDnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCuDnn);
                m_hCuDnn = 0;
            }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GE(colBottom[0].num_axes, 2, "Bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)");
            m_nT = colBottom[0].shape(0);
            m_nN = colBottom[0].shape(1);

            if (colBottom[0].num_axes > 2)
                m_nInputSize = colBottom[0].count(2);

            m_log.WriteLine("Initializing recurrent layer: assuming input batch contains " + m_nT.ToString() + " timesteps of " + m_nN.ToString() + " independent streams.");

            m_log.CHECK_EQ(colBottom[1].num_axes, 2, "Bottom[1] must have exactly 2 axes -- (#timesteps, #streams)");
            m_log.CHECK_EQ(m_nT, colBottom[1].shape(0), "The bottom[1].shape(0) must equal T = " + m_nT.ToString());
            m_log.CHECK_EQ(m_nN, colBottom[1].shape(1), "The bottom[1].shape(1) must equal N = " + m_nN.ToString());

            // If expose_hidden is set, we take as input and produce as output
            // the hidden state blobs at the first and last timesteps.
            m_bExposeHidden = m_param.recurrent_param.expose_hidden;

            if (m_param.recurrent_param.useCudnn())
                layerSetUpCuDnn(colBottom, colTop);
            else
                layerSetUpCaffe(colBottom, colTop);
        }

        private void layerSetUpCuDnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            try
            {
                m_bUseTensors = m_param.recurrent_param.cudnn_enable_tensor_cores;
                m_nHiddenSize = (int)m_param.recurrent_param.num_output;
                m_nNumLayers = (int)m_param.recurrent_param.num_layers;

                m_hCuDnn = m_cuda.CreateCuDNN();

                m_blobX = new Blob<T>(m_cuda, m_log);
                m_blobX.Name = m_param.name + " x";
                m_blobY = new Blob<T>(m_cuda, m_log);
                m_blobY.Name = m_param.name + " y";

                m_blobHx = new Blob<T>(m_cuda, m_log);
                m_blobHx.Name = m_param.name + " hx";
                m_blobCx = new Blob<T>(m_cuda, m_log);
                m_blobCx.Name = m_param.name + " cx";
                m_blobHy = new Blob<T>(m_cuda, m_log);
                m_blobHy.Name = m_param.name + " hy";
                m_blobCy = new Blob<T>(m_cuda, m_log);
                m_blobCy.Name = m_param.name + " cy";
                m_blobWts = new Blob<T>(m_cuda, m_log);
                m_blobWts.Name = m_param.name + " weights";

                blobs.Clear();
                blobs.Add(m_blobWts);

                m_hXDesc = m_cuda.CreateRnnDataDesc();
                m_hYDesc = m_cuda.CreateRnnDataDesc();

                m_hHxDesc = m_cuda.CreateTensorDesc();
                m_hCxDesc = m_cuda.CreateTensorDesc();
                m_hHyDesc = m_cuda.CreateTensorDesc();
                m_hCyDesc = m_cuda.CreateTensorDesc();

                // Setup Rnn Descriptor
                m_hRnnDesc = m_cuda.CreateRnnDesc();
                m_hWeightDesc = m_cuda.CreateFilterDesc();
                m_hDropoutDesc = m_cuda.CreateDropoutDesc();


                //------------------------------------
                //  Start reshape here.
                //------------------------------------

                m_blobX.ReshapeLike(colBottom[0]);
                m_blobX.ShareData(colBottom[0]);
                m_blobX.ShareDiff(colBottom[0]);
                m_log.CHECK_EQ(m_blobX.count(), m_nT * m_nN * m_nInputSize, "The input should be Sequence * Batch * InputSize in length.");

                m_blobHx.Reshape(m_nNumLayers, m_nN, m_nHiddenSize, 1);
                m_blobCx.Reshape(m_nNumLayers, m_nN, m_nHiddenSize, 1);

                m_blobY.Reshape(m_nT, m_nN, m_nHiddenSize, 1);
                m_blobHy.Reshape(m_nNumLayers, m_nN, m_nHiddenSize, 1);
                m_blobCy.Reshape(m_nNumLayers, m_nN, m_nHiddenSize, 1);

                m_blobHx.SetData(0);
                m_blobCx.SetData(0);
                m_blobHy.SetData(0);
                m_blobCy.SetData(0);

                // Set the input/output data descriptors
                m_cuda.SetRnnDataDesc(m_hXDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, m_nT, m_nN, m_nInputSize);
                m_cuda.SetRnnDataDesc(m_hYDesc, RNN_DATALAYOUT.RNN_SEQ_MAJOR, m_nT, m_nN, m_nHiddenSize);

                int[] rgDimA = new int[3];
                int[] rgStrideA = new int[3];

                rgDimA[0] = m_nNumLayers; // Currently, only unidirectional.
                rgDimA[1] = m_nN; // mini batch.
                rgDimA[2] = m_nHiddenSize;

                rgStrideA[0] = rgDimA[2] * rgDimA[1];
                rgStrideA[1] = rgDimA[2];
                rgStrideA[2] = 1;

                m_cuda.SetTensorNdDesc(m_hHxDesc, rgDimA, rgStrideA);
                m_cuda.SetTensorNdDesc(m_hCxDesc, rgDimA, rgStrideA);
                m_cuda.SetTensorNdDesc(m_hHyDesc, rgDimA, rgStrideA);
                m_cuda.SetTensorNdDesc(m_hCyDesc, rgDimA, rgStrideA);

                // Setup the dropout descriptor.
                ulong ulStateCount;
                ulong ulReservedCount;
                m_cuda.GetDropoutInfo(m_hCuDnn, 0, out ulStateCount, out ulReservedCount);
                m_hDropoutStates = m_cuda.AllocMemory((long)ulStateCount);
                m_cuda.SetDropoutDesc(m_hCuDnn, m_hDropoutDesc, m_param.recurrent_param.dropout_ratio, m_hDropoutStates, m_param.recurrent_param.dropout_seed);

                // Setup the RNN descriptor.
                m_cuda.SetRnnDesc(m_hCuDnn, m_hRnnDesc, m_nHiddenSize, m_nNumLayers, m_hDropoutDesc, m_rnnMode, m_bUseTensors);

                // Setup parameters - do this after the rnn descriptor is set
                // otherwise we will not know how many parameters we have to allocate.
                int nCount = m_cuda.GetRnnParamCount(m_hCuDnn, m_hRnnDesc, m_hXDesc);
                List<int> rgWtShape = new List<int>() { nCount, 1, 1 };
                m_blobWts.Reshape(rgWtShape);

                int[] rgDimW = new int[3];
                rgDimW[0] = nCount;
                rgDimW[1] = 1;
                rgDimW[2] = 1;

                m_cuda.SetFilterNdDesc(m_hWeightDesc, rgDimW);

                // Setup the workspace and reserved memory.
                m_nWorkspaceCount = m_cuda.GetRnnWorkspaceCount(m_hCuDnn, m_hRnnDesc, m_hXDesc, out m_nReservedCount);
                m_hWorkspace = m_cuda.AllocMemory(m_nWorkspaceCount);
                m_hReserved = m_cuda.AllocMemory(m_nReservedCount);

                // Fill the weights.
                if (!shareParameter(m_blobWts, rgWtShape))
                {
                    int nNumLinearLayers = (m_rnnMode == RNN_MODE.LSTM) ? 8 : 2;
                    Filler<T> fillerWt = Filler<T>.Create(m_cuda, m_log, m_param.recurrent_param.weight_filler);
                    Filler<T> fillerBias = Filler<T>.Create(m_cuda, m_log, m_param.recurrent_param.bias_filler);
                    int nWtCount;
                    long hWt;
                    int nBiasCount;
                    long hBias;

                    for (int i = 0; i < m_nNumLayers; i++)
                    {
                        for (int j = 0; j < nNumLinearLayers; j++)
                        {
                            m_cuda.GetRnnLinLayerParams(m_hCuDnn, m_hRnnDesc, i, m_hXDesc, m_hWeightDesc, m_blobWts.gpu_data, j, out nWtCount, out hWt, out nBiasCount, out hBias);

                            if (nWtCount % 2 != 0)
                            {
                                // Since, some fillers (gaussian) require an even number of items,
                                // we can temporarily use the all weight diff area and then copy 
                                // the non-even number of items into the layer weights.
                                fillerWt.Fill(nWtCount + 1, m_blobWts.mutable_gpu_diff);
                                m_cuda.copy(nWtCount, m_blobWts.mutable_gpu_diff, hWt);
                            }
                            else
                            {
                                fillerWt.Fill(nWtCount, hWt);
                            }

                            if (nBiasCount % 2 != 0)
                            {
                                // Since, some fillers (gaussian) require an even number of items,
                                // we can temporarily use the all weight diff area and then copy 
                                // the non-even number of items into the layer bias.
                                fillerBias.Fill(nBiasCount + 1, m_blobWts.mutable_gpu_diff);
                                m_cuda.copy(nBiasCount, m_blobWts.mutable_gpu_diff, hBias);
                            }
                            else
                            {
                                fillerBias.Fill(nBiasCount, hBias);
                            }

                            m_cuda.FreeMemoryPointer(hWt);
                            m_cuda.FreeMemoryPointer(hBias);
                        }
                    }
                }

                m_blobWts.SetDiff(0);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
            }
        }

        private void layerSetUpCaffe(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Get (recurrent) input/output names.
            List<string> rgOutputNames = new List<string>();
            OutputBlobNames(rgOutputNames);

            List<string> rgRecurInputNames = new List<string>();
            RecurrentInputBlobNames(rgRecurInputNames);

            List<string> rgRecurOutputNames = new List<string>();
            RecurrentOutputBlobNames(rgRecurOutputNames);

            int nNumRecurBlobs = rgRecurInputNames.Count;
            m_log.CHECK_EQ(nNumRecurBlobs, rgRecurOutputNames.Count, "The number of recurrent input names must equal the number of recurrent output names.");

            // If provided, bottom[2] is a static input to the recurrent net.
            int nNumHiddenExposed = (m_bExposeHidden) ? nNumRecurBlobs : 0;
            m_bStaticInput = (colBottom.Count > 2 + nNumHiddenExposed) ? true : false;

            if (m_bStaticInput)
            {
                m_log.CHECK_GE(colBottom[2].num_axes, 1, "When static input is present, the bottom[2].num_axes must be >= 1");
                m_log.CHECK_EQ(m_nN, colBottom[2].shape(0), "When static input is present, the bottom[2].shape(0) must = N which is " + m_nN.ToString());
            }

            // Create a NetParameter; setup the inputs that aren't unique to particular
            // recurrent architectures.
            NetParameter net_param = new NetParameter();

            LayerParameter input_layer = new LayerParameter(LayerParameter.LayerType.INPUT);
            input_layer.top.Add("x");
            BlobShape input_shape1 = new param.BlobShape();
            for (int i = 0; i < colBottom[0].num_axes; i++)
            {
                input_shape1.dim.Add(colBottom[0].shape(i));
            }
            input_layer.input_param.shape.Add(input_shape1);

            input_layer.top.Add("cont");
            BlobShape input_shape2 = new param.BlobShape();
            for (int i = 0; i < colBottom[1].num_axes; i++)
            {
                input_shape2.dim.Add(colBottom[1].shape(i));
            }
            input_layer.input_param.shape.Add(input_shape2);

            if (m_bStaticInput)
            {
                input_layer.top.Add("x_static");
                BlobShape input_shape3 = new BlobShape();
                for (int i = 0; i < colBottom[2].num_axes; i++)
                {
                    input_shape3.dim.Add(colBottom[2].shape(i));
                }
                input_layer.input_param.shape.Add(input_shape3);
            }

            net_param.layer.Add(input_layer);

            // Call the child's FillUnrolledNet implementation to specify the unrolled
            // recurrent architecture.
            FillUnrolledNet(net_param);

            // Prepend this layer's name to the names of each layer in the unrolled net.
            string strLayerName = m_param.name;
            if (strLayerName.Length > 0)
            {
                for (int i = 0; i < net_param.layer.Count; i++)
                {
                    LayerParameter layer = net_param.layer[i];
                    layer.name = strLayerName + "_" + layer.name;
                }
            }

            // Add 'pseudo-losses' to all outputs to force backpropagation.
            // (Setting force_backward is too agressive as we may not need to backprop to 
            // all inputs, e.g., the sequence continuation indicators.)
            List<string> rgPseudoLosses = new List<string>();
            for (int i = 0; i < rgOutputNames.Count; i++)
            {
                rgPseudoLosses.Add(rgOutputNames[i] + "_pseudoloss");
                LayerParameter layer = new LayerParameter(LayerParameter.LayerType.REDUCTION, rgPseudoLosses[i]);
                layer.bottom.Add(rgOutputNames[i]);
                layer.top.Add(rgPseudoLosses[i]);
                layer.loss_weight.Add(1.0);
                net_param.layer.Add(layer);
            }

            // Create the unrolled net.
            Net<T> sharedNet = null;
            if (m_param is LayerParameterEx<T>)
            {
                RecurrentLayer<T> sharedLayer = ((LayerParameterEx<T>)m_param).SharedLayer as RecurrentLayer<T>;
                if (sharedLayer != null)
                    sharedNet = sharedLayer.m_unrolledNet;
            }

            m_unrolledNet = new Net<T>(m_cuda, m_log, net_param, m_evtCancel, null, m_phase, null, sharedNet);
            m_unrolledNet.set_debug_info(m_param.recurrent_param.debug_info);

            // Setup pointers to the inputs.
            m_blobXInputBlob = m_unrolledNet.blob_by_name("x");
            m_blobContInputBlob = m_unrolledNet.blob_by_name("cont");

            if (m_bStaticInput)
                m_blobXStaticInputBlob = m_unrolledNet.blob_by_name("x_static");

            // Setup pointers to paired recurrent inputs/outputs.
            m_colRecurInputBlobs = new common.BlobCollection<T>();
            m_colRecurOutputBlobs = new common.BlobCollection<T>();

            for (int i = 0; i < nNumRecurBlobs; i++)
            {
                m_colRecurInputBlobs.Add(m_unrolledNet.blob_by_name(rgRecurInputNames[i]));
                m_colRecurOutputBlobs.Add(m_unrolledNet.blob_by_name(rgRecurOutputNames[i]));
            }

            // Setup pointers to outputs.
            m_log.CHECK_EQ(colTop.Count() - nNumHiddenExposed, rgOutputNames.Count, "OutputBlobNames must provide output blob name for each top.");
            m_colOutputBlobs = new common.BlobCollection<T>();
            for (int i = 0; i < rgOutputNames.Count; i++)
            {
                m_colOutputBlobs.Add(m_unrolledNet.blob_by_name(rgOutputNames[i]));
            }

            // We should have 2 inputs (x and cont), plus a number of recurrent inputs,
            // plus maybe a static input.
            int nStaticInput = (m_bStaticInput) ? 1 : 0;
            m_log.CHECK_EQ(2 + nNumRecurBlobs + nStaticInput, m_unrolledNet.input_blobs.Count, "The unrolled net input count should equal 2 + number of recurrent blobs (" + nNumRecurBlobs.ToString() + ") + static inputs (" + nStaticInput.ToString() + ")");

            // This layer's parameters are any parameters in the layers of the unrolled 
            // net.  We only want one copy of each parameter, so check that the parameter
            // is 'owned' by the layer, rather than shared with another.
            blobs.Clear();
            for (int i = 0; i < m_unrolledNet.parameters.Count; i++)
            {
                if (m_unrolledNet.param_owners[i] == -1)
                {
                    m_log.WriteLine("Adding parameter " + i.ToString() + ": " + m_unrolledNet.param_display_names[i]);
                    blobs.Add(m_unrolledNet.parameters[i]);
                }
            }

            // Check that param_propagate_down is set for all of the parameters in the 
            // unrolled net; set param_propagate_down to true in this layer.
            for (int i = 0; i < m_unrolledNet.layers.Count; i++)
            {
                for (int j = 0; j < m_unrolledNet.layers[i].blobs.Count; j++)
                {
                    m_log.CHECK(m_unrolledNet.layers[i].param_propagate_down(j), "param_propagate_down not set for layer " + i.ToString() + ", param " + j.ToString());
                }
            }
            m_rgbParamPropagateDown = new DictionaryMap<bool>(blobs.Count, true);

            // Set the diffs of recurrent outputs to 0 -- we can't backpropagate across
            // batches.
            for (int i = 0; i < m_colRecurOutputBlobs.Count; i++)
            {
                m_colRecurOutputBlobs[i].SetDiff(0);
            }

            // Check that the last output_names.count layers are the pseudo-losses;
            // set last_layer_index so that we don't actually run these layers.
            List<string> rgLayerNames = m_unrolledNet.layer_names;
            m_nLastLayerIndex = rgLayerNames.Count - 1 - rgPseudoLosses.Count;
            for (int i = m_nLastLayerIndex + 1, j = 0; i < rgLayerNames.Count; i++, j++)
            {
                m_log.CHECK(rgLayerNames[i] == rgPseudoLosses[j], "The last layer at idx " + i.ToString() + " should be the pseudo layer named " + rgPseudoLosses[j]);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GE(colBottom[0].num_axes, 2, "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)");
            m_log.CHECK_EQ(m_nT, colBottom[0].shape(0), "input number of timesteps changed.");
            m_nN = colBottom[0].shape(1);
            m_log.CHECK_EQ(colBottom[1].num_axes, 2, "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)");
            m_log.CHECK_EQ(m_nT, colBottom[1].shape(0), "bottom[1].shape(0) should equal the timesteps T (" + m_nT.ToString() + ")");
            m_log.CHECK_EQ(m_nN, colBottom[1].shape(1), "bottom[1].shape(1) should equal the streams N (" + m_nN + ")");

            if (m_param.recurrent_param.useCudnn())
                reshapeCuDnn(colBottom, colTop);
            else
                reshapeCaffe(colBottom, colTop);
        }

        private void reshapeCuDnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(m_blobY);
            colTop[0].ShareData(m_blobY);
            colTop[0].ShareDiff(m_blobY);
        }

        private void reshapeCaffe(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobXInputBlob.ReshapeLike(colBottom[0]);
            List<int> rgContShape = colBottom[1].shape();
            m_blobContInputBlob.Reshape(rgContShape);

            if (m_bStaticInput)
                m_blobXStaticInputBlob.ReshapeLike(colBottom[2]);

            List<BlobShape> rgRecurInputShapes = new List<BlobShape>();
            RecurrentInputShapes(rgRecurInputShapes);
            m_log.CHECK_EQ(rgRecurInputShapes.Count, m_colRecurInputBlobs.Count, "The number of recurrent input shapes must equal the number of recurrent input blobs!");

            for (int i = 0; i < rgRecurInputShapes.Count; i++)
            {
                m_colRecurInputBlobs[i].Reshape(rgRecurInputShapes[i]);
            }

            m_unrolledNet.Reshape();

            m_blobXInputBlob.ShareData(colBottom[0]);
            m_blobXInputBlob.ShareDiff(colBottom[0]);
            m_blobContInputBlob.ShareData(colBottom[1]);

            int nStaticInput = 0;

            if (m_bStaticInput)
            {
                nStaticInput = 1;
                m_blobXStaticInputBlob.ShareData(colBottom[2]);
                m_blobXStaticInputBlob.ShareDiff(colBottom[2]);
            }

            if (m_bExposeHidden)
            {
                int nBottomOffset = 2 + nStaticInput;
                for (int i = nBottomOffset, j = 0; i < colBottom.Count; i++, j++)
                {
                    m_log.CHECK(Utility.Compare<int>(m_colRecurInputBlobs[j].shape(), colBottom[i].shape()), "Shape mismatch - recur_input_blobs_[" + j.ToString() + "]: '" + m_colRecurInputBlobs[j].shape_string + "' vs. bottom[" + i.ToString() + "]: '" + colBottom[i].shape_string + "'");
                    m_colRecurInputBlobs[j].ShareData(colBottom[i]);
                }
            }

            for (int i = 0; i < m_colOutputBlobs.Count; i++)
            {
                colTop[i].ReshapeLike(m_colOutputBlobs[i]);
                colTop[i].ShareData(m_colOutputBlobs[i]);
                colTop[i].ShareDiff(m_colOutputBlobs[i]);
            }

            if (m_bExposeHidden)
            {
                int nTopOffset = m_colOutputBlobs.Count;
                for (int i = nTopOffset, j = 0; i < colTop.Count; i++, j++)
                {
                    colTop[i].ReshapeLike(m_colRecurOutputBlobs[j]);
                }
            }
        }

        /// <summary>
        /// Reset the hidden state of the net by zeroing out all recurrent outputs.
        /// </summary>
        public virtual void Reset()
        {
            for (int i = 0; i < m_colRecurOutputBlobs.Count; i++)
            {
                m_colRecurOutputBlobs[i].SetData(0);
            }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs.
        /// </summary>
        public override int MinBottomBlobs
        {
            get
            {
                int nMinBottoms = 2;

                if (m_param.recurrent_param.expose_hidden)
                {
                    List<string> rgInputs = new List<string>();
                    RecurrentInputBlobNames(rgInputs);
                    nMinBottoms += rgInputs.Count;
                }

                return nMinBottoms;
            }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: min+1
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return MinBottomBlobs + 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get
            {
                int nNumTops = 1;

                if (m_param.recurrent_param.expose_hidden)
                {
                    List<string> rgOutputs = new List<string>();
                    RecurrentOutputBlobNames(rgOutputs);
                    nNumTops += rgOutputs.Count;
                }

                return nNumTops;
            }
        }

        /// <summary>
        /// Returns <i>true</i> for all but the bottom index = 1, for you can't propagate to the sequence continuation indicators.
        /// </summary>
        /// <param name="nBottomIdx">Specifies the bottom index.</param>
        /// <returns>Returns whether or not to allow forced backward.</returns>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            // Can't propagate to sequence continuation indicators.
            return (nBottomIdx != 1) ? true : false;
        }

        /// <summary>
        /// Fills net_param with the recurrent network architecture.  Subclasses
        /// should define this -- see RNNLayer and LSTMLayer for examples.
        /// </summary>
        /// <param name="net_param">Specifies the net_param to be filled.</param>
        protected abstract void FillUnrolledNet(NetParameter net_param);

        /// <summary>
        /// Fills names with the names of the 0th timestep recurrent input
        /// Blob's.  Subclasses should define this -- see RNNlayer and LSTMLayer
        /// for examples.
        /// </summary>
        /// <param name="rgNames">Specifies the input names.</param>
        protected abstract void RecurrentInputBlobNames(List<string> rgNames);

        /// <summary>
        /// Fills shapes with the shapes of the recurrent input Blob's.
        /// Subclassses should define this -- see RNNLayer and LSTMLayer
        /// for examples.
        /// </summary>
        /// <param name="rgShapes">Specifies the shapes to be filled.</param>
        protected abstract void RecurrentInputShapes(List<BlobShape> rgShapes);

        /// <summary>
        /// Fills names with the names of the Tth timestep recurrent output
        /// Blob's.  Subclassses should define this -- see RNNLayer and LSTMLayer
        /// for examples.
        /// </summary>
        /// <param name="rgNames">Specifies the output names.</param>
        protected abstract void RecurrentOutputBlobNames(List<string> rgNames);

        /// <summary>
        /// Fills names with the names of the output blobs, concatenated across
        /// all timesteps. Should return a name for each top Blob.
        /// Subclassses should define this -- see RNNLayer and LSTMLayer
        /// for examples.
        /// </summary>
        /// <param name="rgNames">Specifies the output names.</param>
        protected abstract void OutputBlobNames(List<string> rgNames);

        /// <summary>
        /// Peforms the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (length 2-3)
        /// -# @f$ (T \times N \times ...) @f$
        ///     the time-varying input @f$ x @f$. After the first two axes, whose
        ///     dimensions must correspond to the number of timesteps @f$ T @f$ and
        ///     the number of independent streams @f$ N @f$, respectively, its
        ///     dimensions may be arbitrary.  Note that the ordering of dimensions --
        ///     @f$ (T \times N \times ...) @f$, rather than
        ///     @f$ (N \times T \times ...) @f$ -- means that the @f$ N @f$ independent input
        ///     streams must be 'interleaved'.
        ///     
        /// -# @f$ (T \times N) @f$
        ///     the sequence continuation indicators @f$ \delta @f$. 
        ///     These inputs should be binary (0 or 1) indicators, where
        ///     @f$ \delta{t,n} = 0 @f$ means that timestep @f$ t @f$ of stream
        ///     @f$ n @f$ is the beginning of a new sequence, and hence the previous
        ///     hidden state @f$ h_{t-1} @f$ is mulitplied by @f$ \delta_t = 0 @f$ and
        ///     has no effect on the cell's output at timestep 't', and 
        ///     a value of @f$ \delta_{t,n} = 1 @f$ means that timestep @f$ t @f$ of
        ///     stream @f$ n @f$ is a continuation from the previous timestep
        ///     @f$ t-1 @f$, and the previous hidden state @f$ h_{t-1} @f$ affects the
        ///     updated hidden state and output.
        ///     
        /// -# @f$ (N \times ...) @f$ (optional)
        ///     the static (non-time-varying) input @f$ x_{static} @f$.
        ///     After the first axis, whose dimensions must be the number of
        ///     independent streams, its dimensions must be the number of
        ///     independent streams, its dimensions may be arbitrary.
        ///     This is mathematically equivalent to using a time-varying input of
        ///     @f$ x'_t = [x_t; x_{static}] @f$ -- i.e., tiling the static input
        ///     across the 'T' timesteps and concatenating with the time-varying
        ///     input.  Note that if this input is used, all timesteps in a single
        ///     batch within a particular one of the @f$ N @f$ streams must share the
        ///     same static input, even if the sequence continuation indicators
        ///     suggest that difference sequences are ending and beginning within a 
        ///     single batch.  This may require padding and/or truncation for uniform
        ///     length.
        /// </param>
        /// <param name="colTop">top output Blob (length 1)
        /// -# @f$ (T \times N \times D) @f$
        ///     the time-varying output @f$ y @f$, where @f$ d @f$ is
        ///     <code>recurrent_param.num_output</code>.
        ///     Refer to documentation for particular RecurrentLayer implementations
        ///     (such as RNNLayer or LSTMLayer) for the definition of @f$ y @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.recurrent_param.useCudnn())
                forward_cudnn(colBottom, colTop);
            else
                forward_cuda(colBottom, colTop);
        }

        private void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            double dfClip = Utility.ConvertVal<T>(colBottom[1].GetData(0));

            if (dfClip > 0)
            {
                m_blobCx.CopyFrom(m_blobCy);
                m_blobHx.CopyFrom(m_blobHy);
            }

            m_cuda.RnnForward(m_hCuDnn,
                              m_hRnnDesc,
                              m_hXDesc,
                              m_blobX.gpu_data,
                              m_hHxDesc,
                              m_blobHx.gpu_data,
                              m_hCxDesc,
                              m_blobCx.gpu_data,
                              m_hWeightDesc,
                              m_blobWts.gpu_data,
                              m_hYDesc,
                              m_blobY.gpu_data,
                              m_hHyDesc,
                              m_blobHy.gpu_data,
                              m_hCyDesc,
                              m_blobCy.gpu_data,
                              m_hWorkspace,
                              m_nWorkspaceCount,
                              m_hReserved,
                              m_nReservedCount,
                              (m_phase == Phase.TRAIN) ? true : false);
        }

        private void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Hacky fix for test time... reshare all the shared blobs.
            // TODO: somehow make this work non-hackily.
            //if (m_phase == Phase.TEST || m_phase == Phase.RUN)
            //    m_unrolledNet.ShareWeights();

            m_log.CHECK_EQ(m_colRecurInputBlobs.Count, m_colRecurOutputBlobs.Count, "The recurrent input and output blobs must have the same count.");

            if (!m_bExposeHidden)
            {
                for (int i = 0; i < m_colRecurInputBlobs.Count; i++)
                {
                    int nCount = m_colRecurInputBlobs[i].count();
                    m_log.CHECK_EQ(nCount, m_colRecurOutputBlobs[i].count(), "The input and output blob at " + i.ToString() + " must have the same count.");
                    long hTimestep_T_Data = m_colRecurOutputBlobs[i].gpu_data;
                    long hTimestep_0_Data = m_colRecurInputBlobs[i].mutable_gpu_data;
                    m_cuda.copy(nCount, hTimestep_T_Data, hTimestep_0_Data);
                }
            }

            m_unrolledNet.ForwardFromTo(0, m_nLastLayerIndex);

            if (m_bExposeHidden)
            {
                int nTopOffset = m_colOutputBlobs.Count;

                for (int i = nTopOffset, j = 0; i < colTop.Count; i++, j++)
                {
                    colTop[i].ShareData(m_colRecurOutputBlobs[j]);
                }
            }
        }

        /// <summary>
        /// Backward computation.
        /// </summary>
        /// <param name="colTop">See 'foward' documetation.</param>
        /// <param name="rgbPropagateDown">Specifies whether or not to propagate down.</param>
        /// <param name="colBottom">See 'forward' documentation.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (m_param.recurrent_param.useCudnn())
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
            else
                backward_cuda(colTop, rgbPropagateDown, colBottom);
        }

        private void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_log.CHECK(!rgbPropagateDown[1], "Cannot backpropagate to sequence indicators.");

            m_cuda.RnnBackwardData(m_hCuDnn,
                              m_hRnnDesc,
                              m_hYDesc,
                              m_blobY.gpu_data,
                              m_blobY.gpu_diff,
                              m_hHyDesc,
                              m_blobHy.gpu_diff,
                              m_hCyDesc,
                              m_blobCy.gpu_diff,
                              m_hWeightDesc,
                              m_blobWts.gpu_data,
                              m_hHxDesc,
                              m_blobHx.gpu_data,
                              m_hCxDesc,
                              m_blobCx.gpu_data,
                              m_hXDesc,
                              m_blobX.mutable_gpu_diff,
                              m_hHxDesc,
                              m_blobHx.mutable_gpu_diff,
                              m_hCxDesc,
                              m_blobCx.mutable_gpu_diff,
                              m_hWorkspace,
                              m_nWorkspaceCount,
                              m_hReserved,
                              m_nReservedCount);
            // cudnnBackwardWeights adds to the data in weight diff.
            m_blobWts.SetDiff(0);

            m_cuda.RnnBackwardWeights(m_hCuDnn,
                              m_hRnnDesc,
                              m_hXDesc,
                              m_blobX.gpu_data,
                              m_hHxDesc,
                              m_blobHx.gpu_data,
                              m_hYDesc,
                              m_blobY.gpu_data,
                              m_hWorkspace,
                              m_nWorkspaceCount,
                              m_hWeightDesc,
                              m_blobWts.mutable_gpu_diff,
                              m_hReserved,
                              m_nReservedCount);
        }

        private void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_log.CHECK(!rgbPropagateDown[1], "Cannot backpropagate to sequence indicators.");

            // TODO: skip backpropagation to inputs and parameters inside the unrolled
            // net according to propagate_down[0] and propagate_down[2].  For now just
            // backprop to inputs and parameters unconditionally, as either the inputs or
            // the parameters do need backward (or Net would have set
            // layer_needs_backward[i] = false for this layer).
            m_unrolledNet.Backward(m_nLastLayerIndex);
        }
    }
}
