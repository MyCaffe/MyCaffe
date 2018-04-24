using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using MyCaffe.param;
using MyCaffe.data;
using MyCaffe.layers;
using MyCaffe.layers.alpha;

namespace MyCaffe.common
{
    /// <summary>
    /// Connects Layer's together into a direct acrylic graph (DAG)
    /// specified by a NetParameter
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Net<T> : IDisposable 
    {       
        NetParameter m_param;
        CudaDnn<T> m_cuda;
        Log m_log;

        // The network name.
        string m_strName;

        // The phase: TRAIN or TEST
        Phase m_phase = Phase.NONE;

        // Individual layers in the net.
        List<Layer<T>> m_rgLayers = new List<Layer<T>>();
        List<string> m_rgstrLayerNames = new List<string>();
        DictionaryEx<string, int> m_rgLayerNamesIndex = new DictionaryEx<string, int>(0);
        List<bool> m_rgbLayerNeedBackward = new List<bool>();

        // The blobs storing intermediate results between the layer.
        BlobCollection<T> m_colBlobs = new BlobCollection<T>();
        List<string> m_rgstrBlobNames = new List<string>();
        DictionaryEx<string, int> m_rgBlobNamesIndex = new DictionaryEx<string, int>(0);
        List<bool> m_rgbBlobNeedBackward = new List<bool>();

        // The bottom vecs storing the vectors containing the input for each layer.
        // They don't actually host the blobs (blobs_ does), so we simply store
        // the reference.
        List<BlobCollection<T>> m_rgcolBottomVecs = new List<BlobCollection<T>>();
        List<List<int>> m_rgrgnBottomIdVecs;
        List<List<bool>> m_rgrgbBottomNeedBackward = new List<List<bool>>();
        
        // The top vecs stores the vecotrs containing the output of each layer.
        List<BlobCollection<T>> m_rgcolTopVecs = new List<BlobCollection<T>>();
        List<List<int>> m_rgrgnTopIdVecs = new List<List<int>>();

        // Vector of weight in the loss (or objective) function of each net blob,
        // indexed by blob_id.
        List<double> m_rgdfBlobLossWeights = new List<double>();
        List<List<int>> m_rgrgnParamIdVecs = new List<List<int>>();
        List<int> m_rgnParamOwners = new List<int>();
        List<string> m_rgstrParamDisplayNames = new List<string>();
        List<KeyValuePair<int, int>> m_rgParamLayerIndices = new List<KeyValuePair<int, int>>();
        DictionaryEx<string, int> m_rgParamNamesIndex = new DictionaryEx<string, int>(0);

        // blob indices for the input and the output of the net.
        List<int> m_rgnNetInputBlobIndices = new List<int>();
        List<int> m_rgnNetOutputBlobIndices = new List<int>();
        BlobCollection<T> m_colNetInputBlobs = new BlobCollection<T>();
        BlobCollection<T> m_colNetOutputBlobs = new BlobCollection<T>();

        // The parameters in the network.
        BlobCollection<T> m_colParams = new BlobCollection<T>();
        BlobCollection<T> m_colLearnableParams = new BlobCollection<T>();

        // The mapping from params -> learnable_params : we have
        // learnable_param_ids.Count == params.Count,
        // and learnable_params[learnable_param_ids[i]] == params[i]
        // if and only if params[i] is an 'owner'; otherwise params[i] is a sharer
        // and learnable_params[learnable_param_ids[i]] gives its owner.
        List<int> m_rgnLearnableParamIds = new List<int>();
        
        // The learning rate multipliers from learnable params.
        List<double?> m_rgdfParamsLr = new List<double?>();

        // The weight decay multipliers for learnable params.
        List<double?> m_rgdfParamsWeightDecay = new List<double?>();

        // The bytes of memory used by this net
        long m_lMemoryUsed = 0;

        // Whether to compute and display debug info for the net.
        bool m_bDebugInfo = false;

        // The image database passed through to the data layer(s).
        IXImageDatabase m_db = null;

        // Cancel event used to cancel training and testing.
        CancelEvent m_evtCancel;

        // Store the last forward input to the data layer, if any (currently only supported by BatchDataInput).
        BatchInput m_lastBatchInput = null;

        // When enabled, the best result mask zero's out all data items except for those that have the greatest value
        string m_strBestResultTargetNodeToMask = null;
        int m_nBestResultCount = 5;
        BEST_RESULT_TYPE m_nBestResultType = BEST_RESULT_TYPE.BY_CHANNEL;

        long m_hWorkspaceData = 0;  // shared among the layers, only grows in size.
        long m_lWorkspaceSize = 0;

        Net<T> m_sharedNet = null;
        bool m_bBreakOnFirstNan = false;
        bool m_bDetectDetailedNans = false;
        Blob<T> m_blobWork = null;

        /// Specifies the OnGetIteration event that fires when a layer needs to get the current iteration from the solver.
        /// </summary>
        public event EventHandler<GetIterationArgs> OnGetIteration;

        public enum BEST_RESULT_TYPE /** @private */
        {
            BY_CHANNEL,
            BY_WEIGHT
        }

        /// <summary>
        /// The Net constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the NetParameter used to initialize the Net.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel operations run by the Net.</param>
        /// <param name="imgDb">Specifies the CaffeImageDatabase.</param>
        /// <param name="phaseOverride">Optionally, specifies an override of the Phase for which the Net is used.</param>
        /// <param name="evtTrainingCompleted">Optionally, specifies an auto reset event that is set after training has completed.</param>
        /// <param name="sharedNet">Specifies another Net that shares the GPU memory created by this Net.</param>
        public Net(CudaDnn<T> cuda, Log log, NetParameter p, CancelEvent evtCancel, IXImageDatabase imgDb, Phase phaseOverride = Phase.NONE, AutoResetEvent evtTrainingCompleted = null, Net<T> sharedNet = null)
        {
            m_sharedNet = sharedNet;
            m_db = imgDb;
            m_cuda = cuda;
            m_log = log;

            m_evtCancel = evtCancel;

            Init(p, phaseOverride, evtTrainingCompleted);
        }

        /// <summary>
        /// Releases all resources (GPU and Host) used by the Net.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called from Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            foreach (Layer<T> l in m_rgLayers)
            {
                l.Dispose();
            }

            m_rgLayers.Clear();

            if (m_colBlobs != null)
            {
                m_colBlobs.Dispose();
                m_colBlobs = null;
            }

            foreach (BlobCollection<T> b in m_rgcolBottomVecs)
            {
                b.Dispose();
            }

            m_rgcolBottomVecs.Clear();

            foreach (BlobCollection<T> b in m_rgcolTopVecs)
            {
                b.Dispose();
            }

            m_rgcolTopVecs.Clear();

            if (m_colNetOutputBlobs != null)
            {
                m_colNetOutputBlobs.Dispose();
                m_colNetOutputBlobs = null;
            }

            if (m_colParams != null)
            {
                m_colParams.Dispose();
                m_colParams = null;
            }

            if (m_colLearnableParams != null)
            {
                m_colLearnableParams.Dispose();
                m_colLearnableParams = null;
            }

            if (m_hWorkspaceData != 0)
            {
                m_cuda.DisableGhostMemory();
                m_cuda.FreeMemory(m_hWorkspaceData);
                m_cuda.ResetGhostMemory();
                m_hWorkspaceData = 0;
            }

            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }
        }

        /// <summary>
        /// Releases all resources (GPU and Host) used by the Net.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Initialize a network with a NetParameter.
        /// </summary>
        /// <param name="p">Specifies the NetParameter.</param>
        /// <param name="phaseOverride">Optionally, specifies a Phase override for which the Net will run under.</param>
        /// <param name="evtTrainingCompleted">Optionally, specifies an auto reset event that is set upon the completion of training.</param>
        public void Init(NetParameter p, Phase phaseOverride = Phase.NONE, AutoResetEvent evtTrainingCompleted = null)
        {
            try
            {
                m_param = p;

                // Set phase from the state.
                if (phaseOverride != Phase.NONE)
                {
                    m_phase = phaseOverride;
                    p.state.phase = m_phase;
                }
                else
                {
                    m_phase = p.state.phase;
                }

                // Filter layser based on their include/exclude rules and
                // the current NetState.
                NetParameter filtered_param = FilterNet(p);
                m_log.WriteLine("Initializing net from parameters: " + filtered_param.DebugString());

                // Create a copy of filtered_param with splits added where necessary;
                NetParameter param = InsertSplits(filtered_param);

                // Basically, build all the layers and set up their connections.
                m_strName = param.name;

                DictionaryEx<string, int> blob_name_to_idx = new DictionaryEx<string, int>(0);
                List<string> available_blobs = new List<string>();

                m_log.CHECK(param.input_dim.Count == 0 || param.input_shape.Count == 0, "Must specify either input_shape OR depreciated input_dim, not both.");

                if (param.input_dim.Count > 0)
                {
                    // Depreciated 4D dimensions.
                    m_log.CHECK_EQ(param.input.Count * 4, param.input_dim.Count, "Incorrect inpub blob dimension specification.");
                }
                else
                {
                    m_log.CHECK_EQ(param.input.Count, param.input_shape.Count, "Exactly one input_shape must be specified per input.");
                }

                m_lMemoryUsed = 0;

                // Set the input blobs
                for (int input_id = 0; input_id < param.input.Count; input_id++)
                {
                    int layer_id = -1; // inputs have fake layer ID = -1
                    AppendTop(param, layer_id, input_id, available_blobs, blob_name_to_idx);
                }

                // For each layer, set up its input and output
                m_rgcolBottomVecs = new List<BlobCollection<T>>();
                m_rgcolTopVecs = new List<BlobCollection<T>>();
                m_rgrgnBottomIdVecs = new List<List<int>>();
                m_rgrgnParamIdVecs = new List<List<int>>();
                m_rgrgnTopIdVecs = new List<List<int>>();
                m_rgrgbBottomNeedBackward = new List<List<bool>>();

                for (int layer_id = 0; layer_id < param.layer.Count; layer_id++)
                {
                    m_rgcolBottomVecs.Add(new BlobCollection<T>());
                    m_rgcolTopVecs.Add(new BlobCollection<T>());
                    m_rgrgnBottomIdVecs.Add(new List<int>());
                    m_rgrgnTopIdVecs.Add(new List<int>());
                    m_rgrgnParamIdVecs.Add(new List<int>());
                    m_rgrgbBottomNeedBackward.Add(new List<bool>());

                    // Inherit phase from net if unset.
                    if (param.layer[layer_id].phase == Phase.NONE)
                        param.layer[layer_id].phase = m_phase;

                    // Setup layer.
                    LayerParameter layer_param = param.layer[layer_id];
                    if (layer_param.propagate_down.Count > 0)
                        m_log.CHECK_EQ(layer_param.propagate_down.Count, layer_param.bottom.Count, "propagate_down param must be specified either 0 or bottom.Count times.");


                    //-------------------------------------------
                    //  Set the training completed event for any
                    //  layers that use it, such as the BATCHDATA
                    //  layer.
                    //-------------------------------------------
                    switch (layer_param.type)
                    {
                        case LayerParameter.LayerType.BATCHDATA:
                            if (layer_param.batch_data_param.CompletedEvent == null)
                                layer_param.batch_data_param.CompletedEvent = evtTrainingCompleted;
                            break;
                    }

                    //-------------------------------------------
                    //  When sharing the blobs of another net
                    //  (e.g. The run net does this when also
                    //  training, to save memory)
                    //  pass the parameters and internal blobs
                    //  into the layer_parameter thus allowing
                    //  each layer to share blobs as appropriate.
                    //-------------------------------------------
                    LayerParameter layer_paramEx = layer_param;
                    if (m_sharedNet != null)
                        layer_paramEx = new LayerParameterEx<T>(layer_param, m_sharedNet.parameters, m_sharedNet.layer_blobs(layer_param.name));

                    layer_paramEx.solver_count = m_param.solver_count;
                    layer_paramEx.solver_rank = m_param.solver_rank;

                    // Setup layer cont.
                    Layer<T> layer1 = Layer<T>.Create(m_cuda, m_log, layer_paramEx, m_evtCancel, m_db, new TransferInput(getInput, setInput));
                    layer1.OnGetWorkspace += layer_OnGetWorkspace;
                    layer1.OnSetWorkspace += layer_OnSetWorkspace;
                    layer1.OnGetIteration += layer_OnGetIteration;
                    m_rgLayers.Add(layer1);

                    m_rgstrLayerNames.Add(layer_param.name);
                    m_log.WriteLine("Creating layer " + layer_param.name);

                    bool need_backward = false;

                    //-------------------------------------------
                    // Add input bottom blob for batch data.
                    //-------------------------------------------
                    if (layer_param.type == LayerParameter.LayerType.BATCHDATA)
                    {
                        Blob<T> blobInput = new Blob<T>(m_cuda, m_log);
                        blobInput.Name = "_batchdata_input_";
                        m_colBlobs.Add(blobInput);
                        m_rgstrBlobNames.Add(blobInput.Name);
                        m_rgbBlobNeedBackward.Add(false);
                        int blob_id = blobs.Count - 1;

                        m_rgBlobNamesIndex.Add(blobInput.Name, blob_id);
                        m_log.WriteLine(m_rgstrLayerNames[layer_id] + " <- " + blobInput.Name);

                        m_rgcolBottomVecs[layer_id].Add(m_colBlobs[blob_id]);
                        m_rgrgnBottomIdVecs[layer_id].Add(blob_id);
                        m_rgrgbBottomNeedBackward[layer_id].Add(false);

                        m_rgnNetInputBlobIndices.Add(blob_id);
                        m_colNetInputBlobs.Add(blobs[blob_id]);
                    }

                    // Figure out this layer's input and output
                    for (int bottom_id = 0; bottom_id < layer_param.bottom.Count; bottom_id++)
                    {
                        int blob_id = AppendBottom(param, layer_id, bottom_id, available_blobs, blob_name_to_idx);

                        // If a blob needs backward, this layer should provide it.
                        need_backward |= m_rgbBlobNeedBackward[blob_id];
                    }

                    int num_top = layer_param.top.Count;
                    for (int top_id = 0; top_id < num_top; top_id++)
                    {
                        // Ignore top's named 'null'
                        if (param.layer[layer_id] != null && param.layer[layer_id].top[top_id] == "null")
                            continue;

                        AppendTop(param, layer_id, top_id, available_blobs, blob_name_to_idx);

                        // Collect Input layer tops as Net inputs.
                        if (layer_param.type == LayerParameter.LayerType.INPUT)
                        {
                            int nBlobID = blobs.Count - 1;
                            m_rgnNetInputBlobIndices.Add(nBlobID);
                            m_colNetInputBlobs.Add(blobs[nBlobID]);
                        }
                    }

                    // If the layer specifies that AutoTopBlobs() == true and the LayerParameter
                    // specified fewer than the required number (as specified by
                    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
                    Layer<T> layer = m_rgLayers[layer_id];
                    if (layer.AutoTopBlobs)
                    {
                        int needed_num_top = Math.Max(layer.MinTopBlobs, layer.ExactNumTopBlobs);

                        while (num_top < needed_num_top)
                        {
                            // Add 'anonymous' top blobs -- do not modify available_blobs or
                            // blob_name_to_idx as we don't want these blbos to be usable as input
                            // to other layers.
                            AppendTop(param, layer_id, num_top, null, null);
                            num_top++;
                        }
                    }

                    // After this layer is connected, set it up.
                    m_rgLayers[layer_id].SetNetParameterUsed(param); // used for label mapping
                    m_rgLayers[layer_id].Setup(m_rgcolBottomVecs[layer_id], m_rgcolTopVecs[layer_id]);

                    m_log.WriteLine("Setting up " + m_rgstrLayerNames[layer_id]);

                    for (int top_id = 0; top_id < m_rgcolTopVecs[layer_id].Count; top_id++)
                    {
                        int nIdx = m_rgrgnTopIdVecs[layer_id][top_id];

                        if (m_rgdfBlobLossWeights.Count <= nIdx)
                            Utility.Resize<double>(ref m_rgdfBlobLossWeights, nIdx + 1, 0.0);

                        double dfLoss = layer.loss(top_id);

                        m_rgdfBlobLossWeights[nIdx] = dfLoss;

                        if (m_log.IsEnabled)
                        {
                            string strOut = "Top shape: " + m_rgcolTopVecs[layer_id][top_id].shape_string;

                            if (dfLoss != 0)
                                strOut += "  with loss weight " + dfLoss.ToString();

                            m_log.WriteLine(strOut);
                        }

                        m_lMemoryUsed += m_rgcolTopVecs[layer_id][top_id].count();
                    }

                    m_log.WriteLine("Memory required for data: " + (m_lMemoryUsed * Utility.BaseTypeSize<T>()).ToString());

                    int param_size = layer_param.GetParameterCount();
                    int num_param_blobs = m_rgLayers[layer_id].blobs.Count();
                    m_log.CHECK_LE(param_size, num_param_blobs, "Too many params specified for layer " + layer_param.name);

                    ParamSpec default_param_spec = new ParamSpec();

                    for (int param_id = 0; param_id < num_param_blobs; param_id++)
                    {
                        ParamSpec param_spec = (param_id < param_size) ? layer_param.parameters[param_id] : default_param_spec;
                        bool param_need_backward = (param_spec.lr_mult != 0.0) ? true : false;

                        need_backward |= param_need_backward;
                        m_rgLayers[layer_id].set_param_propagate_down(param_id, param_need_backward);
                    }

                    for (int param_id = 0; param_id < num_param_blobs; param_id++)
                    {
                        AppendParam(param, layer_id, param_id);
                    }

                    // Finally, set the backward flag.
                    m_rgbLayerNeedBackward.Add(need_backward);

                    if (need_backward)
                    {
                        for (int top_id = 0; top_id < m_rgrgnTopIdVecs[layer_id].Count; top_id++)
                        {
                            int nIdx = m_rgrgnTopIdVecs[layer_id][top_id];
                            m_rgbBlobNeedBackward[nIdx] = true;
                        }
                    }
                }

                // Go through the net backwards to determine which blobs contribute to the
                // loss.  We can skip backward computation for blobs that don't contribute
                // to the loss.
                // Also checks if all bottom blobs don't need backward computation (possible
                // because the skip_propagate_down param) and so we can skip backward
                // computation for the entire layer.
                List<string> blobs_under_loss = new List<string>();
                List<string> blobs_skip_backp = new List<string>();

                for (int layer_id = m_rgLayers.Count - 1; layer_id >= 0; layer_id--)
                {
                    bool layer_contributes_loss = false;
                    bool layer_skip_propagate_down = true;

                    for (int top_id = 0; top_id < m_rgcolTopVecs[layer_id].Count; top_id++)
                    {
                        int nIdx = m_rgrgnTopIdVecs[layer_id][top_id];
                        string blob_name = m_rgstrBlobNames[nIdx];

                        if (m_rgLayers[layer_id].loss(top_id) != 0 || blobs_under_loss.Contains(blob_name))
                            layer_contributes_loss = true;

                        if (!blobs_skip_backp.Contains(blob_name))
                            layer_skip_propagate_down = false;

                        if (layer_contributes_loss && !layer_skip_propagate_down)
                            break;
                    }

                    // If this layer can skip backward computation, also all of its bottom blobs
                    // don't need backpropagation
                    if (m_rgbLayerNeedBackward[layer_id] && layer_skip_propagate_down)
                    {
                        m_rgbLayerNeedBackward[layer_id] = false;

                        for (int bottom_id = 0; bottom_id < m_rgcolBottomVecs[layer_id].Count; bottom_id++)
                        {
                            m_rgrgbBottomNeedBackward[layer_id][bottom_id] = false;
                        }
                    }

                    if (!layer_contributes_loss)
                        m_rgbLayerNeedBackward[layer_id] = false;

                    if (m_log.IsEnabled)
                    {
                        if (m_rgbLayerNeedBackward[layer_id])
                            m_log.WriteLine(m_rgstrLayerNames[layer_id] + " needs backward computation.");
                        else
                            m_log.WriteLine(m_rgstrLayerNames[layer_id] + " does not need backward computation.");
                    }

                    for (int bottom_id = 0; bottom_id < m_rgcolBottomVecs[layer_id].Count; bottom_id++)
                    {
                        if (layer_contributes_loss)
                        {
                            int nIdx = m_rgrgnBottomIdVecs[layer_id][bottom_id];
                            string blob_name = m_rgstrBlobNames[nIdx];

                            blobs_under_loss.Add(blob_name);
                        }
                        else
                        {
                            m_rgrgbBottomNeedBackward[layer_id][bottom_id] = false;
                        }

                        if (!m_rgrgbBottomNeedBackward[layer_id][bottom_id])
                        {
                            int nIdx = m_rgrgnBottomIdVecs[layer_id][bottom_id];
                            string blob_name = m_rgstrBlobNames[nIdx];

                            blobs_skip_backp.Add(blob_name);
                        }
                    }
                }

                // Handle force_backward if needed.
                if (param.force_backward)
                {
                    for (int layer_id = 0; layer_id < m_rgLayers.Count; layer_id++)
                    {
                        m_rgbLayerNeedBackward[layer_id] = true;

                        for (int bottom_id = 0; bottom_id < m_rgrgbBottomNeedBackward[layer_id].Count; bottom_id++)
                        {
                            m_rgrgbBottomNeedBackward[layer_id][bottom_id] = m_rgrgbBottomNeedBackward[layer_id][bottom_id] || m_rgLayers[layer_id].AllowForceBackward(bottom_id);

                            int nIdx = m_rgrgnBottomIdVecs[layer_id][bottom_id];
                            m_rgbBlobNeedBackward[nIdx] = m_rgbBlobNeedBackward[nIdx] || m_rgrgbBottomNeedBackward[layer_id][bottom_id];
                        }

                        for (int param_id = 0; param_id < m_rgLayers[layer_id].blobs.Count; param_id++)
                        {
                            m_rgLayers[layer_id].set_param_propagate_down(param_id, true);
                        }
                    }
                }

                // In the end, all remaining blobs are considered output blobs.
                foreach (string blob_name in available_blobs)
                {
                    m_log.WriteLine("This network produces output " + blob_name);
                    int nIdx = blob_name_to_idx[blob_name];
                    Blob<T> blob = m_colBlobs[nIdx];

                    m_colNetOutputBlobs.Add(blob);
                    m_rgnNetOutputBlobIndices.Add(nIdx);
                }

                for (int blob_id = 0; blob_id < m_rgstrBlobNames.Count; blob_id++)
                {
                    string blob_name = m_rgstrBlobNames[blob_id];
                    m_rgBlobNamesIndex[blob_name] = blob_id;
                }

                for (int layer_id = 0; layer_id < m_rgstrLayerNames.Count; layer_id++)
                {
                    string layer_name = m_rgstrLayerNames[layer_id];
                    m_rgLayerNamesIndex[layer_name] = layer_id;
                }

                ShareWeights();
                m_bDebugInfo = param.debug_info;
                m_log.WriteLine("Network initialization done.");
            }
            catch (Exception excpt)
            {
                foreach (Layer<T> layer in m_rgLayers)
                {
                    layer.Dispose();
                }

                m_rgLayers.Clear();
                m_rgstrLayerNames.Clear();
                m_rgLayerNamesIndex.Clear();
                m_rgbBlobNeedBackward.Clear();
                m_colBlobs.Dispose();
                m_rgBlobNamesIndex.Clear();
                m_rgbBlobNeedBackward.Clear();
                m_rgcolBottomVecs.Clear();
                m_rgrgbBottomNeedBackward.Clear();
                m_rgcolTopVecs.Clear();
                m_rgrgnTopIdVecs.Clear();
                m_rgdfBlobLossWeights.Clear();
                m_rgrgnParamIdVecs.Clear();
                m_rgnParamOwners.Clear();
                m_rgstrParamDisplayNames.Clear();
                m_rgParamLayerIndices.Clear();
                m_rgParamNamesIndex.Clear();
                m_rgnNetInputBlobIndices.Clear();
                m_rgnNetOutputBlobIndices.Clear();
                m_colNetInputBlobs.Clear();
                m_colNetOutputBlobs.Clear();
                m_colParams.Clear();
                m_colLearnableParams.Clear();
                m_rgnLearnableParamIds.Clear();
                m_rgdfParamsLr.Clear();
                m_rgdfParamsWeightDecay.Clear();
                m_lMemoryUsed = 0;
                m_bDebugInfo = false;
                m_db = null;
                throw excpt;
            }
        }

        private void layer_OnGetIteration(object sender, GetIterationArgs e)
        {
            if (OnGetIteration != null)
                OnGetIteration(sender, e);
        }

        private void layer_OnSetWorkspace(object sender, WorkspaceArgs e)
        {
            if (e.Size < m_lWorkspaceSize)
                return;

            m_lWorkspaceSize = e.Size;
            m_cuda.DisableGhostMemory();

            if (m_hWorkspaceData != 0)
                m_cuda.FreeMemory(m_hWorkspaceData);

            m_hWorkspaceData = m_cuda.AllocMemory(m_lWorkspaceSize);
            m_cuda.ResetGhostMemory();
        }

        private void layer_OnGetWorkspace(object sender, WorkspaceArgs e)
        {
            e.Data = m_hWorkspaceData;
            e.Size = m_lWorkspaceSize;
        }

        /// <summary>
        /// Enable/disable break the first NaN functionality where training stops immediately upon detecting a NaN in one of the Layer blobs.
        /// </summary>
        public bool EnableBreakOnFirstNaN
        {
            get { return m_bBreakOnFirstNan; }
            set { m_bBreakOnFirstNan = value; }
        }

        /// <summary>
        /// Enable/disable whether or not detailed nans are detected - this will make debugging slower and is only recommended
        /// when running on a TCC enabled driver (as opposed to an WDM driver one used with the monitor).
        /// </summary>
        public bool EnableDetailedNanDetection
        {
            get { return m_bDetectDetailedNans; }
            set { m_bDetectDetailedNans = value; }
        }

        public void EnableBestResultMask(string strTargetNode, int nBestResultCount = 5, BEST_RESULT_TYPE resultType = BEST_RESULT_TYPE.BY_CHANNEL) /** @private */
        {
            m_strBestResultTargetNodeToMask = strTargetNode;
            m_nBestResultCount = nBestResultCount;
            m_nBestResultType = resultType;
        }

        public void DisableBestResultMask() /** @private */
        {
            m_strBestResultTargetNodeToMask = null;
            m_nBestResultCount = 50;
        }

        /// <summary>
        /// Returns the active label counts observed during training.
        /// </summary>
        public string ActiveLabelCounts
        {
            get
            {
                string strSrc = null;

                foreach (Layer<T> layer in m_rgLayers)
                {
                    if (layer.type == LayerParameter.LayerType.DATA ||
                        layer.type == LayerParameter.LayerType.TRIPLET_DATA)
                        strSrc = layer.layer_param.data_param.source;

                    else if (layer.type == LayerParameter.LayerType.BATCHDATA)
                        strSrc = layer.layer_param.batch_data_param.source;

                    else if (layer.type == LayerParameter.LayerType.LABELMAPPING)
                        return ((LabelMappingLayer<T>)layer).GetActualLabelCounts(strSrc);
                }

                return m_db.GetLabelCountsAsTextFromSourceName(strSrc);
            }
        }

        /// <summary>
        /// Enables/disables passthrough on each layer of the net.
        /// </summary>
        /// <remarks>
        /// If enabled by a given layer, the Bottom inputs are copied directly to the Top outputs during the forward pass and the forward pass returns.  This is used by the BatchDataLayer.
        /// </remarks>
        /// <param name="bEnable">Specifies whether or not to enable passthrough.</param>
        public void SetEnablePassthrough(bool bEnable)
        {
            foreach (Layer<T> layer in m_rgLayers)
            {
                layer.SetEnablePassthrough(bEnable);
            }   
        }

        /// <summary>
        /// Set the reinforcement information on each ReinforcementLossLayer - this is used during reinforcement training.
        /// </summary>
        /// <param name="col"></param>
        public void SetReinforcementInformation(BatchInformationCollection col)
        {
            for (int i = m_rgLayers.Count - 1; i >= 0; i--)
            {
                if (m_rgLayers[i] is ReinforcementLossLayer<T>)
                    m_rgLayers[i].layer_param.reinforcement_loss_param.BatchInfoCollection = col;
            }
        }

        private BatchInput getInput()
        {
            return m_lastBatchInput;
        }

        private void setInput(BatchInput biInput)
        {
            m_lastBatchInput = biInput;
        }

        /// <summary>
        /// Removes layers that the user specified should be excluded given the current
        /// phase, level and stage.
        /// </summary>
        /// <param name="param">Specifies the NetParameter to filter.</param>
        /// <returns>The newly filtered NetParmeter is returned.</returns>
        public NetParameter FilterNet(NetParameter param)
        {
            NetState net_state = param.state;
            NetParameter param_filtered = param.Clone(false);

            for (int i = 0; i < param.layer.Count; i++)
            {
                LayerParameter layer_param = param.layer[i];
                string layer_name = layer_param.name;

                m_log.CHECK(layer_param.include.Count == 0 || layer_param.exclude.Count == 0, "Specify either include rules or exclude rules; not both.");

                // If no include rules are specified, the layer is included by default and
                // only excluded if it meets one of the exclude rules.
                bool layer_included = (layer_param.include.Count == 0) ? true : false;

                for (int j = 0; layer_included && j < layer_param.exclude.Count; j++)
                {
                    if (StateMeetsRule(net_state, layer_param.exclude[j], layer_name))
                        layer_included = false;
                }

                for (int j = 0; !layer_included & j < layer_param.include.Count; j++)
                {
                    if (StateMeetsRule(net_state, layer_param.include[j], layer_name))
                        layer_included = true;
                }

                if (layer_included)
                    param_filtered.layer.Add(layer_param.Clone(true));
            }

            return param_filtered;
        }

        /// <summary>
        /// Returns whether NetState state meets NetStateRule rule.
        /// </summary>
        /// <param name="state">Specifies the NetState to test.</param>
        /// <param name="rule">Specifies the NetStateRul to test against the NetState.</param>
        /// <param name="strLayer">Specifies the name of the Layer for which the test is taking place.</param>
        /// <returns>If the NetState of the Layer meets the NetStateRule, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool StateMeetsRule(NetState state, NetStateRule rule, string strLayer)
        {
            if (rule.phase == Phase.ALL)
                return true;

            // Check whether the rule is broken due to phase.
            if (rule.phase != Phase.NONE)
            {
                if (rule.phase != state.phase)
                {
                    m_log.WriteLine("The NetState phase (" + state.phase.ToString() + ") differed from the phase (" + rule.phase.ToString() + ") specified by a rule in layer " + strLayer);
                    return false;
                }
            }

            // Check whether the rule is broken due to min level.
            if (rule.min_level.HasValue)
            {
                if (state.level < rule.min_level.Value)
                {
                    m_log.WriteLine("The NetState level (" + state.level.ToString() + ") is below the min_level ( " + rule.min_level.Value.ToString() + ") specified by a rule in layer " + strLayer);
                    return false;
                }
            }

            // Check whether the rule is broken due to max level.
            if (rule.max_level.HasValue)
            {
                if (state.level > rule.max_level.Value)
                {
                    m_log.WriteLine("The NetState level (" + state.level.ToString() + ") is above the max_level ( " + rule.max_level.Value.ToString() + ") specified by a rule in layer " + strLayer);
                    return false;
                }
            }

            // Check whether the rule is broken due to stage.  The NetState must
            // contain ALL of the rule's stages to meet it.
            for (int i = 0; i < rule.stage.Count; i++)
            {
                // Check that the NetState contains the rule's ith stage.
                bool has_stage = false;

                for (int j = 0; !has_stage && j < state.stage.Count; j++)
                {
                    if (rule.stage[i] == state.stage[j])
                    {
                        has_stage = true;
                        break;
                    }
                }

                if (!has_stage)
                {
                    m_log.WriteLine("The NetState did not contain stage '" + rule.stage[i] + "' specified by a rule in layer " + strLayer);
                    return false;
                }
            }

            // Check whether the rule is broken due to not_stage.  The NetState must
            // contain NONE of the rule's not_stages to meet it.
            for (int i = 0; i < rule.not_stage.Count; i++)
            {
                // Check that the NetState contains the rule's ith not_stage.
                bool has_stage = false;

                for (int j = 0; !has_stage && j < state.stage.Count; j++)
                {
                    if (rule.not_stage[i] == state.stage[j])
                    {
                        has_stage = true;
                        break;
                    }
                }

                if (has_stage)
                {
                    m_log.WriteLine("The NetState contained a not_stage '" + rule.not_stage[i] + "' specified by a rule in layer " + strLayer);
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Append a new input or top blob to the net.
        /// </summary>
        /// <param name="param">Specifies the NetParameter used.</param>
        /// <param name="layer_id">Specifies the Layer index associated with the Blob.</param>
        /// <param name="top_id">Specifies the Blob index of the (top) Blob.</param>
        /// <param name="available_blobs">Specifies the list of available Blob names.</param>
        /// <param name="blob_name_to_idx">Specifies the Blob name to index dictionary lookup.</param>
        protected void AppendTop(NetParameter param, int layer_id, int top_id, List<string> available_blobs, DictionaryEx<string, int> blob_name_to_idx)
        {
            LayerParameter layer_param = null;
            string blob_name;

            if (layer_id >= 0)
                layer_param = param.layer[layer_id].Clone(false);

            if (layer_param != null)
            {
                if (layer_param.top.Count > top_id)
                    blob_name = layer_param.top[top_id];
                else
                    blob_name = "(automatic)";
            }
            else
            {
                blob_name = param.input[top_id];
            }

            // Check if we are doing in-place computation
            if (blob_name_to_idx != null && layer_param != null && layer_param.bottom.Count > top_id && blob_name == layer_param.bottom[top_id])
            {
                // In-place computation
                m_log.WriteLine(layer_param.name + " -> " + blob_name + " (in-place)");
                int nIdx = blob_name_to_idx[blob_name];
                m_rgcolTopVecs[layer_id].Add(m_colBlobs[nIdx]);
                m_rgrgnTopIdVecs[layer_id].Add(nIdx);
            }
            else if (blob_name_to_idx != null && blob_name_to_idx.ContainsKey(blob_name))
            {
                // If we are not doing in-place computation but have duplicated blobs,
                // raise an error.
                m_log.FAIL("Top blob '" + blob_name + "' produced by multiple sources.");
            }
            else
            {
                // Normal output.
                if (m_log.IsEnabled)
                {
                    if (layer_param != null)
                        m_log.WriteLine(layer_param.name + " -> " + blob_name);
                    else
                        m_log.WriteLine("Input " + top_id.ToString() + " -> " + blob_name);
                }

                Blob<T> blob_pointer = new Blob<T>(m_cuda, m_log);
                blob_pointer.Name = blob_name;


                //---------------------------------------------------
                //  When sharing this net with another (e.g. the run
                //  net shares the blobs of the training net to 
                //  help conserve gpu memory.) do not share the input 
                //  blob or the output blob for sharing the input
                //  blob will change the batch size, and sharing the
                //  output blob will cause the training layer's loss
                //  layer to be overwritten, which we do not want.
                //
                //  NOTE: The blob sharing only works when the network
                //  using the shared nodes uses sizes that are less
                //  than or equal to those of the shared node.  In
                //  the case of the run network this is not a problem
                //  for its batch size is 1 whereas the training net
                //  has a batch size of 1 or greater.
                //
                //  When sharing the training net with the testing
                //  net, blobs are only shared when the training
                //  net batch size is >= to the testing nets.
                //----------------------------------------------------
                if (m_sharedNet != null && layer_id >= 0 && layer_id < param.layer.Count() - 1)
                    m_sharedNet.blobs.Share(blob_pointer, null, false);

                int blob_id = m_colBlobs.Count;
                m_colBlobs.Add(blob_pointer);
                m_rgstrBlobNames.Add(blob_name);
                m_rgbBlobNeedBackward.Add(false);

                if (blob_name_to_idx != null)
                    blob_name_to_idx[blob_name] = blob_id;

                if (layer_id == -1)
                {
                    // Set the (explicitly specified) dimensions of the input blob.
                    if (param.input_dim.Count > 0)
                    {
                        blob_pointer.Reshape(param.input_dim[top_id * 4 + 0],
                                             param.input_dim[top_id * 4 + 1],
                                             param.input_dim[top_id * 4 + 2],
                                             param.input_dim[top_id * 4 + 3]);
                    }
                    else
                    {
                        blob_pointer.Reshape(param.input_shape[top_id]);
                    }

                    m_rgnNetInputBlobIndices.Add(blob_id);
                    m_colNetInputBlobs.Add(blob_pointer);
                }
                else
                {
                    m_rgrgnTopIdVecs[layer_id].Add(blob_id);
                    m_rgcolTopVecs[layer_id].Add(blob_pointer);
                }
            }

            if (available_blobs != null)
                available_blobs.Add(blob_name);
        }

        /// <summary>
        /// Append a new bottom blob to the net.
        /// </summary>
        /// <param name="param">Specifies the NetParameter used.</param>
        /// <param name="layer_id">Specifies the Layer index associated with the Blob.</param>
        /// <param name="bottom_id">Specifies the Blob index of the (bottom) Blob.</param>
        /// <param name="available_blobs">Specifies the list of available Blob names.</param>
        /// <param name="blob_name_to_idx">Specifies the Blob name to index dictionary lookup.</param>
        protected int AppendBottom(NetParameter param, int layer_id, int bottom_id, List<string> available_blobs, DictionaryEx<string, int> blob_name_to_idx)
        {
            LayerParameter layer_param = param.layer[layer_id];
            string blob_name = layer_param.bottom[bottom_id];

            if (!available_blobs.Contains(blob_name))
                m_log.FAIL("Unknown bottom blob '" + blob_name + "' (layer '" + layer_param.name + "', bottom index " + bottom_id.ToString() + ")");

            int blob_id = blob_name_to_idx[blob_name];
            m_log.WriteLine(m_rgstrLayerNames[layer_id] + " <- " + blob_name);

            m_rgcolBottomVecs[layer_id].Add(m_colBlobs[blob_id]);
            m_rgrgnBottomIdVecs[layer_id].Add(blob_id);
            available_blobs.Remove(blob_name);

            bool need_backward = m_rgbBlobNeedBackward[blob_id];
            // Check if the backpropagation on bottom_id should be skipped
            if (layer_param.propagate_down.Count > 0)
                need_backward = layer_param.propagate_down[bottom_id];

            m_rgrgbBottomNeedBackward[layer_id].Add(need_backward);

            return blob_id;
        }

        /// <summary>
        /// Append a new parameter blob to the net.
        /// </summary>
        /// <param name="param">Specifies the NetParameter used.</param>
        /// <param name="layer_id">Specifies the Layer index associated with the Blob.</param>
        /// <param name="param_id">Specifies the Blob index of the (parameter) Blob.</param>
        protected void AppendParam(NetParameter param, int layer_id, int param_id)
        {
            LayerParameter layer_param = m_rgLayers[layer_id].layer_param;
            int param_size = layer_param.parameters.Count;
            string param_name = (param_size > param_id) ? layer_param.parameters[param_id].name : "";

            if (param_name.Length > 0)
                m_rgstrParamDisplayNames.Add(param_name);
            else
                m_rgstrParamDisplayNames.Add(param_id.ToString());

            int net_param_id = m_colParams.Count;

            m_colParams.Add(m_rgLayers[layer_id].blobs[param_id]);
            m_rgrgnParamIdVecs[layer_id].Add(net_param_id);
            m_rgParamLayerIndices.Add(new KeyValuePair<int ,int>(layer_id, param_id));

            ParamSpec default_param_spec = new ParamSpec();
            ParamSpec param_spec = (layer_param.parameters.Count > param_id) ? layer_param.parameters[param_id] : default_param_spec;

            if (param_size == 0 || param_name.Length == 0 || (param_name.Length > 0 && !m_rgParamNamesIndex.ContainsKey(param_name)))
            {
                // This layer 'owns' this parameter blob -- it is either anonymous
                // (i.e., not given a param_name) or explicitly given a name that we
                // haven't already seen.
                m_rgnParamOwners.Add(-1);

                if (param_name.Length > 0)
                    m_rgParamNamesIndex[param_name] = net_param_id;

                int learnable_param_id = m_rgnLearnableParamIds.Count;
                m_colLearnableParams.Add(m_colParams[net_param_id]);
                m_rgnLearnableParamIds.Add(learnable_param_id);
                m_rgdfParamsLr.Add(param_spec.lr_mult);
                m_rgdfParamsWeightDecay.Add(param_spec.decay_mult);
            }
            else
            {
                // Named param blob with name we've seen before: share params
                int owner_net_param_id = m_rgParamNamesIndex[param_name];
                m_rgnParamOwners.Add(owner_net_param_id);

                KeyValuePair<int,int> owner_index = m_rgParamLayerIndices[owner_net_param_id];
                int owner_layer_id = owner_index.Key;
                int owner_param_id = owner_index.Value;

                m_log.WriteLine("Sharing parameters '" + param_name + "' owned by layer '" + m_rgstrLayerNames[owner_layer_id] + "', param index " + owner_param_id.ToString());
                
                Blob<T> this_blob = m_rgLayers[layer_id].blobs[param_id];
                Blob<T> owner_blob = m_rgLayers[owner_layer_id].blobs[owner_param_id];
                int param_size2 = layer_param.parameters.Count;

                if (param_size2 > param_id && (layer_param.parameters[param_id].share_mode == ParamSpec.DimCheckMode.PERMISSIVE))
                {
                    // Permissive dimension checking  -- only check counts are thet same.
                    m_log.CHECK_EQ(this_blob.count(), owner_blob.count(), "Cannot share param '" + param_name + "' owned by layer '" + m_rgstrLayerNames[owner_layer_id] + "' with layer '" + m_rgstrLayerNames[layer_id] + "'; count mismatch.  Owner layer param shape is " + owner_blob.shape_string + "; sharing layer shape is " + this_blob.shape_string);
                }
                else
                {
                    // Strict dimension checking -- all dems must be the same.
                    m_log.CHECK(Utility.Compare<int>(this_blob.shape(), owner_blob.shape()), "Cannot share param '" + param_name + "' owned by layer '" + m_rgstrLayerNames[owner_layer_id] + "' with layer '" + m_rgstrLayerNames[layer_id] + "'; shape mismatch.  Owner layer param shape is " + owner_blob.shape_string + "; sharing layer expects shape " + this_blob.shape_string);
                }

                int learnable_param_id = m_rgnLearnableParamIds[owner_net_param_id];
                m_rgnLearnableParamIds.Add(learnable_param_id);

                if (param_spec.lr_mult != 1.0)
                {
                    if (m_rgdfParamsLr[learnable_param_id].HasValue)
                        m_log.CHECK_EQ(param_spec.lr_mult, m_rgdfParamsLr[learnable_param_id].Value, "Shared param '" + param_name + "' has mismatched lr_mult.");
                    else
                        m_rgdfParamsLr[learnable_param_id] = param_spec.lr_mult;
                }

                if (param_spec.decay_mult != 1.0)
                {
                    if (m_rgdfParamsWeightDecay[learnable_param_id].HasValue)
                        m_log.CHECK_EQ(param_spec.decay_mult, m_rgdfParamsWeightDecay[learnable_param_id].Value, "Shared param '" + param_name + "' has mismatched decay_mult.");
                    else
                        m_rgdfParamsWeightDecay[learnable_param_id] = param_spec.decay_mult;
                }
            }
        }

        /// <summary>
        /// The FromTo variant of forward and backwarde operate on the
        /// (topological) ordering by which the net is specified.  For general DAG
        /// netowrks, note that (1) computing from one layer to another might entail
        /// extra computation on unrelated branches, and (2) computation starting in
        /// the middle may be incorrect if all the layers of a fan-in are not
        /// included.
        /// </summary>
        /// <param name="nStart">Optionally, specifies the index of the first Layer where the Forward operation is to start.</param>
        /// <param name="nEnd">Optionally, specifies the index of the last Layer to run the Forward operation on.</param>
        /// <returns>The loss is returned.</returns>
        public double ForwardFromTo(int nStart = 0, int nEnd = int.MaxValue)
        {
            if (nEnd == int.MaxValue)
                nEnd = m_rgLayers.Count - 1;

            m_log.CHECK_GE(nStart, 0, "Start must be >= 0.");
            m_log.CHECK_LT(nEnd, m_rgLayers.Count, "End must be < the layer count of " + m_rgLayers.Count.ToString());
            double dfLoss = 0;

            for (int i = nStart; i <= nEnd; i++)
            {
                double dfLayerLoss = m_rgLayers[i].Forward(m_rgcolBottomVecs[i], m_rgcolTopVecs[i]);
                dfLoss += dfLayerLoss;

                if (m_bDebugInfo)
                    ForwardDebugInfo(i);

                //-----------------------------------------------
                // Used when debugging.
                //-----------------------------------------------
                if (m_strBestResultTargetNodeToMask != null && m_rgLayers[i].layer_param.name == m_strBestResultTargetNodeToMask)
                {
                    Blob<T> blob = blob_by_name(m_strBestResultTargetNodeToMask);
                    if (blob == null)
                        m_log.FAIL("Could not find the Best Result Target Node '" + m_strBestResultTargetNodeToMask + "'!");

                    if (m_nBestResultType == BEST_RESULT_TYPE.BY_CHANNEL)
                        blob.KeepBestResultsByChannel(m_nBestResultCount);
                    else
                        blob.KeepBestResultsByWeight(m_nBestResultCount);
                }
            }

            return dfLoss;
        }

        /// <summary>
        /// Run forward with the input Blob's already fed separately.
        /// </summary>
        /// <remarks>
        /// You can get the input blobs using input_blobs().
        /// </remarks>
        /// <param name="dfLoss">Returns the loss of the operation.</param>
        /// <returns>The collection of output Blobs is returned.</returns>
        public BlobCollection<T> Forward(out double dfLoss)
        {
            dfLoss = ForwardFromTo();
            return m_colNetOutputBlobs;
        }

        /// <summary>
        /// Run forward using a set of bottom blobs and return the result.
        /// </summary>
        /// <param name="colBottom"></param>
        /// <param name="dfLoss"></param>
        /// <returns></returns>
        public BlobCollection<T> Forward(BlobCollection<T> colBottom, out double dfLoss)
        {
            // Copy bottom to internal bottom
            for (int i = 0; i < colBottom.Count; i++)
            {
                bool bReshape = false;

                if (m_rgLayers.Count > 0 && m_rgLayers[0].type == LayerParameter.LayerType.BATCHDATA)
                    bReshape = true;

                m_colNetInputBlobs[i].CopyFrom(colBottom[i], false, bReshape);
            }

            return Forward(out dfLoss);
        }

        /// <summary>
        /// The network backward should take no input and output, since it solely computes the 
        /// gradient w.r.t. the parameters, and the data has already been provided during the
        /// forward pass.
        /// </summary>
        /// <param name="nStart">Specifies the Layer index where the Backward operation is to start.</param>
        /// <param name="nEnd">Specifies the Layer index of the last Layer that the Backward operation is run.</param>
        public void Backward(int nStart = int.MaxValue, int nEnd = 0)
        {
            if (nStart == int.MaxValue)
                nStart = m_rgLayers.Count - 1;

            m_log.CHECK_GE(nEnd, 0, "End must be greater than 0.");
            m_log.CHECK_LT(nStart, m_rgLayers.Count, "Start must be less than the number of layers (" + m_rgLayers.Count.ToString() + ")");

            for (int i = nStart; i >= nEnd; i--)
            {
                if (m_rgbLayerNeedBackward[i])
                {
                    m_rgLayers[i].Backward(m_rgcolTopVecs[i], m_rgrgbBottomNeedBackward[i], m_rgcolBottomVecs[i]);

                    if (m_bDebugInfo)
                        BackwardDebugInfo(i);
                }
            }

            if (m_bDebugInfo)
            {
                double dfAsumData = 0;
                double dfAsumDiff = 0;
                double dfSumsqData = 0;
                double dfSumsqDiff = 0;

                for (int i = 0; i < m_colLearnableParams.Count; i++)
                {
                    dfAsumData += Utility.ConvertVal<T>(m_colLearnableParams[i].asum_data());
                    dfAsumDiff += Utility.ConvertVal<T>(m_colLearnableParams[i].asum_diff());
                    dfSumsqData += Utility.ConvertVal<T>(m_colLearnableParams[i].sumsq_data());
                    dfSumsqDiff += Utility.ConvertVal<T>(m_colLearnableParams[i].sumsq_diff());
                }

                double dfL2NormData = Math.Sqrt(dfSumsqData);
                double dfL2NormDiff = Math.Sqrt(dfSumsqDiff);

                m_log.WriteLine("  [Backward] All net params (data, diff): L1 norm = (" + dfAsumData.ToString() + ", " + dfAsumDiff.ToString() + "; L2 norm = (" + dfL2NormData.ToString() + ", " + dfL2NormDiff.ToString() + ")");
            }
        }

        /// <summary>
        /// Helper for displaying debug info in Forward about input blobs.
        /// </summary>
        /// <param name="input_id">Specifies the index of the input Blob within input_blobs().</param>
        protected void InputDebugInfo(int input_id)
        {
            Blob<T> blob = m_colNetInputBlobs[input_id];
            int nIdx = m_rgnNetInputBlobIndices[input_id];
            string blob_name = m_rgstrBlobNames[nIdx];
            double data_abs_val_mean = Utility.ConvertVal<T>(blob.asum_data()) / blob.count();

            m_log.WriteLine("  [Forward] Input " + blob_name + " data: " + data_abs_val_mean.ToString());
        }

        /// <summary>
        /// Helper for displaying debug info in Forward.
        /// </summary>
        /// <param name="layer_id">Specifies the Layer index associated with the Blob.</param>
        protected void ForwardDebugInfo(int layer_id)
        {
            for (int top_id = 0; top_id < m_rgcolTopVecs[layer_id].Count; top_id++)
            {
                Blob<T> blob = m_rgcolTopVecs[layer_id][top_id];
                int nIdx = m_rgrgnTopIdVecs[layer_id][top_id];
                string blob_name = m_rgstrBlobNames[nIdx];
                double data_abs_val_mean = Utility.ConvertVal<T>(blob.asum_data()) / blob.count();

                m_log.WriteLine("  [Forward] Layer " + m_rgstrLayerNames[layer_id] + ", top blob " + blob_name + " data: " + data_abs_val_mean.ToString());
            }

            for (int param_id = 0; param_id < m_rgLayers[layer_id].blobs.Count; param_id++)
            {
                Blob<T> blob = m_rgLayers[layer_id].blobs[param_id];
                int net_param_id = m_rgrgnParamIdVecs[layer_id][param_id];
                string blob_name = m_rgstrParamDisplayNames[net_param_id];
                double data_abs_val_mean = Utility.ConvertVal<T>(blob.asum_data()) / blob.count();

                m_log.WriteLine("  [Forward] Layer " + m_rgstrLayerNames[layer_id] + ", param blob " + blob_name + " data: " + data_abs_val_mean.ToString());
            }
        }

        /// <summary>
        /// Helper for displaying debug info in Backward.
        /// </summary>
        /// <param name="layer_id">Specifies the Layer index associated with the Blob.</param>
        protected void BackwardDebugInfo(int layer_id)
        {
            BlobCollection<T> bottom_vec = m_rgcolBottomVecs[layer_id];

            for (int bottom_id = 0; bottom_id < bottom_vec.Count; bottom_id++)
            {
                if (!m_rgrgbBottomNeedBackward[layer_id][bottom_id])
                    continue;

                Blob<T> blob = bottom_vec[bottom_id];
                int nIdx = m_rgrgnBottomIdVecs[layer_id][bottom_id];
                string blob_name = m_rgstrBlobNames[nIdx];
                double diff_abs_val_mean = Utility.ConvertVal<T>(blob.asum_diff()) / blob.count();

                m_log.WriteLine("  [Backward] Layer " + m_rgstrLayerNames[layer_id] + ", bottom blob " + blob_name + " diff: " + diff_abs_val_mean.ToString());
            }

            for (int param_id = 0; param_id < m_rgLayers[layer_id].blobs.Count; param_id++)
            {
                if (!m_rgLayers[layer_id].param_propagate_down(param_id))
                    continue;

                Blob<T> blob = m_rgLayers[layer_id].blobs[param_id];
                double diff_abs_val_mean = Utility.ConvertVal<T>(blob.asum_diff()) / blob.count();

                m_log.WriteLine("  [Backward] Layer " + m_rgstrLayerNames[layer_id] + ", param blob " + param_id.ToString() + " diff: " + diff_abs_val_mean.ToString());
            }
        }

        /// <summary>
        /// Helper for displaying debug info in Update.
        /// </summary>
        /// <param name="param_id">Specifies the parameter index associated with the Blob.</param>
        protected void UpdateDebugInfo(int param_id)
        {
            Blob<T> blob = m_colBlobs[param_id];
            int param_owner = m_rgnParamOwners[param_id];
            int nIdx = m_rgParamLayerIndices[param_id].Key;
            string layer_name = m_rgstrLayerNames[nIdx];
            string param_display_name = m_rgstrParamDisplayNames[param_id];
            double diff_abs_val_mean = Utility.ConvertVal<T>(blob.asum_diff()) / blob.count();

            if (param_owner < 0)
            {
                double data_abs_val_mean = Utility.ConvertVal<T>(blob.asum_data()) / blob.count();
                m_log.WriteLine("  [Update] Layer " + layer_name + ", param " + param_display_name + " data: " + data_abs_val_mean.ToString() + "; diff: " + diff_abs_val_mean.ToString());
            }
            else
            {
                int nIdx2 = m_rgParamLayerIndices[param_owner].Key;
                string owner_layer_name = m_rgstrLayerNames[nIdx2];
                int nIdx3 = m_rgnParamOwners[param_id];
                string param_display_name_owner = m_rgstrParamDisplayNames[nIdx3];
                m_log.WriteLine("  [Update] Layer " + layer_name + ", param blob " + param_display_name + " (owned by layer " + owner_layer_name + ", param " + param_display_name_owner + ") diff: " + diff_abs_val_mean.ToString());
            }
        }

        /// <summary>
        /// For an already initialized net, impliciitly compies (i.e., using no
        /// additional memory) the pre-trained layers from another Net.
        /// </summary>
        /// <param name="srcNet">Specifies the source Net whos blobs are shared with the calling Net.</param>
        public void ShareTrainedLayersWith(Net<T> srcNet)
        {
            if (srcNet == this)
                return;

            int num_source_layers = srcNet.layers.Count();

            for (int i = 0; i < num_source_layers; i++)
            {
                Layer<T> source_layer = srcNet.layers[i];
                string source_layer_name = srcNet.layer_names[i];
                int target_layer_id = 0;

                while (target_layer_id != m_rgstrLayerNames.Count && m_rgstrLayerNames[target_layer_id] != source_layer_name)
                {
                    target_layer_id++;
                }

                if (target_layer_id == m_rgstrLayerNames.Count)
                {
                    m_log.WriteLine("Ignoring source layer " + source_layer_name, true);
                    continue;
                }

                m_log.WriteLine("Copying source layer " + source_layer_name);
                BlobCollection<T> target_blobs = m_rgLayers[target_layer_id].blobs;
                m_log.CHECK_EQ(target_blobs.Count, source_layer.blobs.Count, "Incompatible number of blobs for layer " + source_layer_name);

                for (int j = 0; j < target_blobs.Count; j++)
                {
                    Blob<T> source_blob = source_layer.blobs[j];
                    m_log.CHECK(Utility.Compare<int>(target_blobs[j].shape(), source_blob.shape()), "Cannot share param " + j.ToString() + " weights from layer '" + source_layer_name + "'; shape mismatch.  Source param shape is " + source_blob.shape_string + "; target param shape is " + target_blobs[j].shape_string);
                    target_blobs[j].ShareData(source_blob);
                }
            }
        }

        /// <summary>
        /// Copies the trained layer of this Net to another Net.
        /// </summary>
        /// <param name="dstNet">Specifies the Net where the trained layers are to be copied.</param>
        public void CopyTrainedLayersTo(Net<T> dstNet)
        {
            int num_source_layers = layers.Count();

            for (int i = 0; i < num_source_layers; i++)
            {
                Layer<T> source_layer = layers[i];
                string source_layer_name = layer_names[i];
                int target_layer_id = 0;

                while (target_layer_id != dstNet.m_rgstrLayerNames.Count && dstNet.m_rgstrLayerNames[target_layer_id] != source_layer_name)
                {
                    target_layer_id++;
                }

                if (target_layer_id == dstNet.m_rgstrLayerNames.Count)
                {
                    m_log.WriteLine("Ignoring source layer " + source_layer_name, true);
                    continue;
                }

                m_log.WriteLine("Copying source layer " + source_layer_name);
                BlobCollection<T> target_blobs = dstNet.m_rgLayers[target_layer_id].blobs;
                m_log.CHECK_EQ(target_blobs.Count, source_layer.blobs.Count, "Incompatible number of blobs for layer " + source_layer_name);

                for (int j = 0; j < target_blobs.Count; j++)
                {
                    Blob<T> source_blob = source_layer.blobs[j];
                    m_log.CHECK(Utility.Compare<int>(target_blobs[j].shape(), source_blob.shape()), "Cannot copy param " + j.ToString() + " weights from layer '" + source_layer_name + "'; shape mismatch.  Source param shape is " + source_blob.shape_string + "; target param shape is " + target_blobs[j].shape_string);
                    target_blobs[j].CopyFrom(source_blob);
                }
            }
        }

        /// <summary>
        /// Copies the trained layers of this Net to another Net.
        /// </summary>
        /// <param name="dstNet">Specifies the destination Net where the trained layers are to be copied.</param>
        /// <param name="rgLayerNames">Specifies the layer name dictionary lookup, where only the weights of layer names within the Dictionary lookup are copied.</param>
        /// <param name="bTranspose">Specifies whether or not to copy the weights and transpose the copy.</param>
        public void CopyTrainedLayersTo(Net<T> dstNet, DictionaryEx<string, string> rgLayerNames, bool bTranspose)
        {
            foreach (Layer<T> sourceLayer in m_rgLayers)
            {
                string source_layer_name = sourceLayer.layer_param.name;

                if (rgLayerNames.ContainsKey(source_layer_name))
                {
                    string strTargetLayer = rgLayerNames[source_layer_name];

                    if (strTargetLayer != null && strTargetLayer.Length > 0)
                    {
                        foreach (Layer<T> targetLayer in dstNet.m_rgLayers)
                        {
                            if (targetLayer.layer_param.name == strTargetLayer)
                            {
                                m_log.WriteLine("Copying source layer " + source_layer_name);
                                BlobCollection<T> target_blobs = targetLayer.blobs;
                                m_log.CHECK_EQ(target_blobs.Count, sourceLayer.blobs.Count, "Incompatible number of blobs for layer " + source_layer_name);
                                int nCount = 1;  // currently the bias is ignored.

                                for (int i = 0; i < nCount; i++)
                                {
                                    Blob<T> source_blob = sourceLayer.blobs[i];
                                    m_log.CHECK(Utility.Compare<int>(target_blobs[i].shape(), source_blob.shape()), "Cannot copy param " + i.ToString() + " weights from layer '" + source_layer_name + "'; shape mismatch.  Source param shape is " + source_blob.shape_string + "; target param shape is " + target_blobs[i].shape_string);

                                    if (bTranspose)
                                        target_blobs[i].CopyFromAndTransposeHeightWidth(source_blob, false);
                                    else
                                        target_blobs[i].CopyFrom(source_blob, false, false);
                                }
                            }
                        }
                    }
                }
            }
        }
        

        /// <summary>
        /// Reshape all layers from the bottom to the top.
        /// </summary>
        /// <remarks>
        /// This is useful to propagate changes to layer sizes without running
        /// a forward pass, e.g. to compute output feature size.
        /// </remarks>
        public void Reshape()
        {
            for (int i = 0; i < m_rgLayers.Count; i++)
            {
                m_rgLayers[i].Reshape(m_rgcolBottomVecs[i], m_rgcolTopVecs[i]);
            }
        }

        /// <summary>
        /// For an already initialized net, CopyTrainedLayersFrom copies the already
        /// trained layers from another net parameter instance.
        /// </summary>
        /// <param name="param">Specifies the NetParameter to use.</param>
        public void CopyTrainedLayersFrom(NetParameter param)
        {
            int num_source_layers = param.layer.Count();

            for (int i = 0; i < num_source_layers; i++)
            {
                LayerParameter source_layer = param.layer[i];
                string source_layer_name = source_layer.name;
                int target_layer_id = 0;

                while (target_layer_id != m_rgstrLayerNames.Count && m_rgstrLayerNames[target_layer_id] != source_layer_name)
                {
                    target_layer_id++;
                }

                if (target_layer_id == m_rgstrLayerNames.Count)
                {
                    m_log.WriteLine("Ignoring source layer " + source_layer_name, true);
                    continue;
                }

                m_log.WriteLine("Copying source layer " + source_layer_name);
                BlobCollection<T> target_blobs = m_rgLayers[target_layer_id].blobs;
                m_log.CHECK_EQ(target_blobs.Count, source_layer.blobs.Count, "Incompatible number of blobs for layer " + source_layer_name);

                for (int j = 0; j < target_blobs.Count; j++)
                {
                    if (!target_blobs[j].ShapeEquals(source_layer.blobs[j]))
                    {
                        Blob<T> source_blob = new Blob<T>(m_cuda, m_log);
                        source_blob.FromProto(source_layer.blobs[j], true);
                        m_log.FAIL("Cannot copy param " + j.ToString() + " weights from layer " + source_layer_name + "; shape mismatch. Source param shape is " + source_blob.shape_string + "; target param shape is " + target_blobs[j].shape_string + ". To learn this layer's arameters from scratch rather than copying from the saved net, rename the layer.");
                    }

                    target_blobs[j].FromProto(source_layer.blobs[j], false);
                }
            }
        }

        /// <summary>
        /// Writes the net to a proto.
        /// </summary>
        /// <returns>A new NetParameter is returned.</returns>
        public NetParameter ToProto(bool bIncludeBlobs)
        {
            NetParameter p = m_param.Clone(true);

            if (bIncludeBlobs)
            {
                foreach (Layer<T> layer in m_rgLayers)
                {
                    if (layer.blobs.Count > 0)
                    {
                        foreach (LayerParameter lp in p.layer)
                        {
                            if (lp.type == layer.layer_param.type &&
                                lp.name == layer.layer_param.name)
                            {
                                foreach (Blob<T> blob in layer.blobs)
                                {
                                    lp.blobs.Add(blob.ToProto());
                                }
                            }
                        }
                    }
                }
            }

            return p;
        }

        /// <summary>
        /// Updates the network weights based on the diff values computed.
        /// </summary>
        public void Update()
        {
            for (int i = 0; i < m_colLearnableParams.Count; i++)
            {
                m_colLearnableParams[i].Update();
            }
        }

        /// <summary>
        /// Zero out the diffs of all netw parameters.  This should be run before Backward.
        /// </summary>
        public void ClearParamDiffs()
        {
            for (int i = 0; i < m_colLearnableParams.Count; i++)
            {
                Blob<T> blob = m_colLearnableParams[i];
                blob.SetDiff(0.0);
            }
        }

        /// <summary>
        /// Shares weight data of owner blobs with shared blobs.
        /// </summary>
        /// <remarks>
        /// Note: this is called by Net::Init, and thus should normally not be called
        /// manually.
        /// </remarks>
        public void ShareWeights()
        {
            for (int i = 0; i < m_colParams.Count; i++)
            {
                if (m_rgnParamOwners[i] < 0)
                    continue;

                int nIdx = m_rgnParamOwners[i];
                m_colParams[i].ShareData(m_colParams[nIdx]);
                m_colParams[i].ShareDiff(m_colParams[nIdx]);
            }
        }

        /// <summary>
        /// Runs a Forward pass followed by a Backward pass.
        /// </summary>
        /// <param name="colBottom">Optionally, specifies input data passed to the Forward pass.</param>
        /// <param name="dfLocalLoss">Returns the local loss of the Forward pass.</param>
        /// <returns>If EnableBreakOnFirstNaN == <i>true</i> and a NaN is detected, this function returns <i>false</i>, otherwise <i>true</i> is returned.</returns>
        public bool ForwardBackward(BlobCollection<T> colBottom, out double dfLocalLoss)
        {
            Forward(colBottom, out dfLocalLoss);

            if (m_bBreakOnFirstNan)
            {
                DebugInformation<T> dbgInfo = GetDebugInformation(m_bDetectDetailedNans);
                string strType;
                string strFirstNan = dbgInfo.DetectFirstNaN(out strType);
                if (strFirstNan != null)
                    return false;
            }

            Backward();

            return true;
        }

        /// <summary>
        /// Returns the network name.
        /// </summary>
        public string name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the layer names.
        /// </summary>
        public List<string> layer_names
        {
            get { return m_rgstrLayerNames; }
        }

        /// <summary>
        /// Returns the blob names.
        /// </summary>
        public List<string> blob_names
        {
            get { return m_rgstrBlobNames; }
        }

        /// <summary>
        /// Returns the blobs.
        /// </summary>
        public BlobCollection<T> blobs
        {
            get { return m_colBlobs; }
        }

        /// <summary>
        /// Returns the layers.
        /// </summary>
        public List<Layer<T>> layers
        {
            get { return m_rgLayers; }
        }

        /// <summary>
        /// Returns the network phase: TRAIN or TEST
        /// </summary>
        public Phase phase
        {
            get { return m_phase; }
        }

        /// <summary>
        /// Returns the bottom vecs for each layer -- usually you won't
        /// need this unless you do per-layer checks such as gradients.
        /// </summary>
        public List<BlobCollection<T>> bottom_vecs
        {
            get { return m_rgcolBottomVecs; }
        }

        /// <summary>
        /// Returns the top vecs for each layer -- usually y ou won't
        /// need this unless you do per-layer checks such as gradients.
        /// </summary>
        public List<BlobCollection<T>> top_vecs
        {
            get { return m_rgcolTopVecs; }
        }

        /// <summary>
        /// Returns the ids of the top blobs of layer i.
        /// </summary>
        /// <param name="layer_id">Specifies the index of the Layer.</param>
        /// <returns></returns>
        public List<int> top_ids(int layer_id)
        {
            m_log.CHECK_GE(layer_id, 0, "Invalid layer id.");
            m_log.CHECK_LT(layer_id, m_rgrgnTopIdVecs.Count, "Invalid layer id.");
            return m_rgrgnTopIdVecs[layer_id];
        }

        /// <summary>
        /// Returns the ids of the bottom blobs of layer i.
        /// </summary>
        /// <param name="layer_id">Specifies the index of the Layer.</param>
        /// <returns></returns>
        public List<int> bottom_ids(int layer_id)
        {
            m_log.CHECK_GE(layer_id, 0, "Invalid layer id.");
            m_log.CHECK_LT(layer_id, m_rgrgnBottomIdVecs.Count, "Invalid layer id.");
            return m_rgrgnBottomIdVecs[layer_id];
        }

        /// <summary>
        /// Returns the collection of lists that tell whether or not the bottom of each layer needs a backward pass or not.
        /// </summary>
        public List<List<bool>> bottom_need_backward
        {
            get { return m_rgrgbBottomNeedBackward; }
        }

        /// <summary>
        /// Returns the collection of blob loss weights.
        /// </summary>
        public List<double> blob_loss_weights
        {
            get { return m_rgdfBlobLossWeights; }
        }

        /// <summary>
        /// Returns a collection of items that tell whether each layer nees a backward pass or not.
        /// </summary>
        public List<bool> layer_need_backward
        {
            get { return m_rgbLayerNeedBackward; }
        }

        /// <summary>
        /// Returns the parameters.
        /// </summary>
        public BlobCollection<T> parameters
        {
            get { return m_colParams; }
        }

        /// <summary>
        /// Returns the collection of Blobs internal to a Layer.
        /// </summary>
        /// <param name="strLayerName">Specifies the name of the Layer.</param>
        /// <returns>The Layer's internal Blobs are returned in a collection.</returns>
        public BlobCollection<T> layer_blobs(string strLayerName)
        {
            if (!has_layer(strLayerName))
                return null;

            Layer<T> layer = layer_by_name(strLayerName);

            return layer.internal_blobs;
        }

        /// <summary>
        /// Sets the learned parameters.
        /// </summary>
        /// <param name="col">Specifies a collection of Blobs containing the learned parameters.</param>
        public void SetLearnedParameters(BlobCollection<T> col)
        {
            m_colLearnableParams = col;
        }

        /// <summary>
        /// Returns the learnable parameters.
        /// </summary>
        public BlobCollection<T> learnable_parameters
        {
            get { return m_colLearnableParams; }
        }

        /// <summary>
        /// Returns the learnable parameter learning rate multipliers.
        /// </summary>
        public List<double?> params_lr
        {
            get { return m_rgdfParamsLr; }
        }

        /// <summary>
        /// Returns the learnable parameter decay multipliers.
        /// </summary>
        public List<double?> params_weight_decay
        {
            get { return m_rgdfParamsWeightDecay; }
        }

        /// <summary>
        /// Returns the dictionary look for parameter names to their indexes.
        /// </summary>
        public DictionaryEx<string, int> param_names_index
        {
            get { return m_rgParamNamesIndex; }
        }

        /// <summary>
        /// Returns the list of parameter owner indexes.
        /// </summary>
        public List<int> param_owners
        {
            get { return m_rgnParamOwners; }
        }

        /// <summary>
        /// Returns the list of parameter display names.
        /// </summary>
        public List<string> param_display_names
        {
            get { return m_rgstrParamDisplayNames; }
        }

        /// <summary>
        /// Returns the number of inputs.
        /// </summary>
        public int num_inputs
        {
            get { return m_colNetInputBlobs.Count; }
        }

        /// <summary>
        /// Returns the number of outputs.
        /// </summary>
        public int num_outputs
        {
            get { return m_colNetOutputBlobs.Count; }
        }

        /// <summary>
        /// Returns the collection of input Blobs.
        /// </summary>
        public BlobCollection<T> input_blobs
        {
            get { return m_colNetInputBlobs; }
        }

        /// <summary>
        /// Returns the collection of output Blobs.
        /// </summary>
        public BlobCollection<T> output_blobs
        {
            get { return m_colNetOutputBlobs; }
        }

        /// <summary>
        /// Returns a list of the output Blob indexes.
        /// </summary>
        public List<int> output_blob_indices
        {
            get { return m_rgnNetOutputBlobIndices; }
        }

        /// <summary>
        /// Returns a list of the input Blob indexes.
        /// </summary>
        public List<int> input_blob_indices
        {
            get { return m_rgnNetInputBlobIndices; }
        }

        /// <summary>
        /// Returns whether or not the Net contains a given Blob.
        /// </summary>
        /// <param name="strBlobName">Specifies the Blob name.</param>
        /// <returns>If the Net has the Blob, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool has_blob(string strBlobName)
        {
            return m_rgBlobNamesIndex.ContainsKey(strBlobName);
        }

        /// <summary>
        /// Returns a blob given its name.
        /// </summary>
        /// <param name="strName">Specifies the Blob's name.</param>
        /// <returns>The Blob with the given name is returned, or <i>null</i> if not found.</returns>
        public Blob<T> blob_by_name(string strName)
        {
            Blob<T> blob_ptr = null;

            if (has_blob(strName))
            {
                int nIdx = m_rgBlobNamesIndex[strName];
                blob_ptr = m_colBlobs[nIdx];
            }
            else
            {
                m_log.FAIL("Unknown blob name " + strName);
            }

            return blob_ptr;
        }

        /// <summary>
        /// Returns the index of a blob given its name.
        /// </summary>
        /// <param name="strName">Specifies the name of the blob to look for.</param>
        /// <returns>The index of the blob within the 'blobs' is returned.</returns>
        public int blob_index_by_name(string strName)
        {
            if (!has_blob(strName))
                return -1;

            return m_rgBlobNamesIndex[strName];
        }

        /// <summary>
        /// Returns whether or not the Net has a given Layer by its name.
        /// </summary>
        /// <param name="strLayer">Specifies the Layer name.</param>
        /// <returns>If the Net contains the Layer, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool has_layer(string strLayer)
        {
            return m_rgLayerNamesIndex.ContainsKey(strLayer);
        }

        /// <summary>
        /// Returns a Layer given its name.
        /// </summary>
        /// <param name="strLayer">Specifies the Layer name.</param>
        /// <returns>The Layer with the given name is returned, or <i>null</i> if not found.</returns>
        public Layer<T> layer_by_name(string strLayer)
        {
            Layer<T> layer_ptr = null;

            if (has_layer(strLayer))
            {
                int nIdx = m_rgLayerNamesIndex[strLayer];
                layer_ptr = m_rgLayers[nIdx];
            }
            else
            {
                m_log.FAIL("Unknown layer name " + strLayer);
            }

            return layer_ptr;
        }

        /// <summary>
        /// Returns a Layer's index given its name.
        /// </summary>
        /// <param name="strLayer">Specifies the Layer name.</param>
        /// <returns>The index of the Layer with the given name is returned, or <i>-1</i> if not found.</returns>
        public int layer_index_by_name(string strLayer)
        {
            if (!has_layer(strLayer))
                return -1;

            return m_rgLayerNamesIndex[strLayer];
        }

        /// <summary>
        /// Sets the debug information flag.
        /// </summary>
        /// <remarks>
        /// When set, extra debug information is output during each Forward and Backward pass, which will slow down training.
        /// </remarks>
        /// <param name="bVal">Specifies whether to enable/disable debug information.</param>
        public void set_debug_info(bool bVal)
        {
            m_bDebugInfo = bVal;
        }

        /// <summary>
        /// Create a new NetParameter and insert splits into it based on a given NetParameter.
        /// </summary>
        /// <remarks>
        /// Splits are used when a given input (top) is used by more than one Layer.  For example a DataLayer 'label' top may
        /// used by both an AccuracyLayer and SoftmaxLossLayer.  In such a case a split is created that allows the 'label' top
        /// to be sent to both.
        /// </remarks>
        /// <param name="param">Specifies the original NetParameter.</param>
        /// <returns>A new NetParameter containing split layers is returned.</returns>
        public NetParameter InsertSplits(NetParameter param)
        {
            // Initialize by copying from the input NetParameter.
            NetParameter param_split = param.Clone(false);

            DictionaryEx<string, KeyValuePair<int, int>> blob_name_to_last_top_idx = new DictionaryEx<string,KeyValuePair<int,int>>(new KeyValuePair<int,int>(-1, -1));
            DictionaryEx<KeyValuePair<int, int>, KeyValuePair<int, int>> bottom_idx_to_source_top_idx = new DictionaryEx<KeyValuePair<int,int>,KeyValuePair<int,int>>(new KeyValuePair<int,int>(-1, -1));
            DictionaryEx<KeyValuePair<int, int>, int> top_idx_to_bottom_count = new DictionaryEx<KeyValuePair<int,int>,int>(0);
            DictionaryEx<KeyValuePair<int, int>, double> top_idx_to_loss_weight = new DictionaryEx<KeyValuePair<int,int>,double>(0);
            DictionaryEx<KeyValuePair<int, int>, int> top_idx_to_bottom_split_idx = new DictionaryEx<KeyValuePair<int,int>,int>(0);
            DictionaryEx<int, string> layer_idx_to_layer_name = new DictionaryEx<int,string>("");

            layer_idx_to_layer_name[-1] = "input";

            // Determine the number of times each blob is used as an input (bottom) blob.

            for (int i = 0; i < param.input.Count; i++)
            {
                string blob_name = param.input[i];
                blob_name_to_last_top_idx[blob_name] = new KeyValuePair<int, int>(-1, i);
            }

            for (int i = 0; i < param.layer.Count; i++)
            {
                LayerParameter layer_param = param.layer[i];
                layer_idx_to_layer_name[i] = layer_param.name;

                for (int j = 0; j < layer_param.bottom.Count; j++)
                {
                    string blob_name = layer_param.bottom[j];

                    if (!blob_name_to_last_top_idx.ContainsKey(blob_name))
                        m_log.FAIL("Unknown bottom blob '" + blob_name + "' (layer '" + layer_param.name + "', bottom index " + j.ToString() + ")");

                    KeyValuePair<int, int> bottom_idx = new KeyValuePair<int, int>(i, j);
                    KeyValuePair<int, int> top_idx = blob_name_to_last_top_idx[blob_name];
                    bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
                    top_idx_to_bottom_count[top_idx]++;
                }

                for (int j = 0; j < layer_param.top.Count; j++)
                {
                    string blob_name = layer_param.top[j];
                    blob_name_to_last_top_idx[blob_name] = new KeyValuePair<int, int>(i, j);
                }

                // A use of a top blob as a loss should be handled similarly to the use of
                // a top blob as an input (bottom) blob to another layer.
                int last_loss = Math.Min(layer_param.loss_weight.Count, layer_param.top.Count);

                for (int j = 0; j < last_loss; j++)
                {
                    string blob_name = layer_param.top[j];
                    KeyValuePair<int, int> top_idx = blob_name_to_last_top_idx[blob_name];
                    top_idx_to_loss_weight[top_idx] = layer_param.loss_weight[j];

                    if (top_idx_to_loss_weight[top_idx] != 0)
                        top_idx_to_bottom_count[top_idx]++;
                }
            }

            // Create split layer for any input blobs used by other layer as bottom
            // blobs more than once.
            for (int i = 0; i < param.input.Count; i++)
            {
                int split_count = top_idx_to_bottom_count[new KeyValuePair<int, int>(-1, i)];

                if (split_count > 1)
                {
                    string layer_name = layer_idx_to_layer_name[-1];
                    string blob_name = param.input[i];
                    double kZeroLossWeight = 0;
                    LayerParameter split_layer_param = CreateSplitLayer(layer_name, blob_name, i, split_count, kZeroLossWeight);
                    param_split.layer.Add(split_layer_param);
                }
            }

            for (int i = 0; i < param.layer.Count; i++)
            {
                LayerParameter layer_param = param.layer[i].Clone(true);
                param_split.layer.Add(layer_param);

                // Replace any shared bottom blobs with split layer outputs.
                for (int j = 0; j < layer_param.bottom.Count; j++)
                {
                    KeyValuePair<int, int> top_idx = bottom_idx_to_source_top_idx[new KeyValuePair<int, int>(i, j)];
                    int split_count = top_idx_to_bottom_count[top_idx];

                    if (split_count > 1)
                    {
                        string layer_name = layer_idx_to_layer_name[top_idx.Key];
                        string blob_name = layer_param.bottom[j];

                        layer_param.bottom[j] = SplitBlobName(layer_name, blob_name, top_idx.Value, top_idx_to_bottom_split_idx[top_idx]++);
                    }
                }

                // Create split layer for any top blobs used by other layer as bottom
                // blobs more than once.
                for (int j = 0; j < layer_param.top.Count; j++)
                {
                    KeyValuePair<int, int> top_idx = new KeyValuePair<int, int>(i, j);
                    int split_count = top_idx_to_bottom_count[top_idx];

                    if (split_count > 1)
                    {
                        string layer_name = layer_idx_to_layer_name[i];
                        string blob_name = layer_param.top[j];
                        double loss_weight = top_idx_to_loss_weight[top_idx];
                        LayerParameter split_layer_param = CreateSplitLayer(layer_name, blob_name, j, split_count, loss_weight);
                        param_split.layer.Add(split_layer_param);

                        if (loss_weight != 0)
                        {
                            layer_param.loss_weight.Clear();
                            top_idx_to_bottom_split_idx[top_idx]++;
                        }
                    }
                }
            }

            return param_split;
        }

        private LayerParameter CreateSplitLayer(string layer_name, string blob_name, int blob_idx, int split_count, double loss_weight)
        {
            LayerParameter split_layer_param = new LayerParameter(LayerParameter.LayerType.SPLIT, SplitLayerName(layer_name, blob_name, blob_idx));
            split_layer_param.bottom.Add(blob_name);

            for (int k = 0; k < split_count; k++)
            {
                split_layer_param.top.Add(SplitBlobName(layer_name, blob_name, blob_idx, k));

                if (loss_weight != 0)
                {
                    if (k == 0)
                        split_layer_param.loss_weight.Add(loss_weight);
                    else
                        split_layer_param.loss_weight.Add(0);
                }
            }

            return split_layer_param;
        }

        private string SplitLayerName(string layer_name, string blob_name, int blob_idx)
        {
            return blob_name + "_" + layer_name + "_" + blob_idx.ToString() + "_split";
        }

        private string SplitBlobName(string layer_name, string blob_name, int blob_idx, int split_idx)
        {
            return blob_name + "_" + layer_name + "_" + blob_idx.ToString() + "_split_" + split_idx.ToString();
        }

        /// <summary>
        /// Loads new weights into the Net.
        /// </summary>
        /// <param name="rgWeights">Specifies the weights themselves.</param>
        /// <param name="persist">Specifies an interface to the persistance object used to load the weights.</param>
        /// <param name="inputWtInfo">Optionally, specifies the input blobs to import.  Note, when set, the <i>targetWtInfo</i> must also be specified.  When <i>null</i>, this parameter is ignored.</param>
        /// <param name="targetWtInfo">Optionally, specifies the target blobs to import into.  Note, when set, the <i>inputWtInfo</i> must also be specified.  When <i>null</i>, this parameter is ignored.</param>
        /// <param name="strSkipBlobType">Optionally, specifies a blob type where weights are NOT loaded.  See Blob.BLOB_TYPE for the types of Blobs.</param>
        public void LoadWeights(byte[] rgWeights, IXPersist<T> persist, List<string> inputWtInfo = null, List<string> targetWtInfo = null, string strSkipBlobType = null)
        {
            if (rgWeights == null)
                return;

            List<string> rgExpectedShapes = new List<string>();
            bool bLoadedDiffs;

            foreach (Blob<T> b in m_colLearnableParams)
            {
                rgExpectedShapes.Add(b.shape_string);
            }

            if (inputWtInfo != null && inputWtInfo.Count == 0)
                inputWtInfo = null;

            if (targetWtInfo != null && targetWtInfo.Count == 0)
                targetWtInfo = null;

            bool bSizeToFit = (inputWtInfo != null && targetWtInfo != null) ? true : false;

            persist.LoadWeights(rgWeights, rgExpectedShapes, m_colLearnableParams, bSizeToFit, out bLoadedDiffs, inputWtInfo, targetWtInfo, strSkipBlobType);
        }

        /// <summary>
        /// Save the weights to a byte array.
        /// </summary>
        /// <param name="bSaveDiff">Specifies whether or not to save the diff values in addition to the data values.</param>
        /// <param name="persist">Specifies an interface to the persistance object used to save the weights.</param>
        /// <returns>The byte array containing the weights is returned.</returns>
        public byte[] SaveWeights(bool bSaveDiff, IXPersist<T> persist)
        {
            foreach (Blob<T> blob in m_colLearnableParams)
            {
                foreach (Layer<T> layer in m_rgLayers)
                {
                    if (layer.blobs.Contains(blob))
                        blob.Tag = layer.layer_param.name;
                }
            }

            return persist.SaveWeights(m_colLearnableParams, bSaveDiff);
        }

        /// <summary>
        /// Finds the Layer that owns a given Blob.
        /// </summary>
        /// <param name="b">Specifies the Blob to search for.</param>
        /// <returns>If found, the Layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<T> FindLayerOwningBlob(Blob<T> b)
        {
            foreach (Layer<T> layer in layers)
            {
                if (layer.blobs.Contains(b))
                    return layer;
            }

            return null;
        }

        /// <summary>
        /// Finds the index of the Layer that owns a given Blob.
        /// </summary>
        /// <param name="b">Specifies the Blob to search for.</param>
        /// <returns>If found, the Layer index is returned, otherwise -1 is returned.</returns>
        public int FindLayerIndexOwningBlob(Blob<T> b)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (layers[i].blobs.Contains(b))
                    return i;
            }

            return -1;
        }

        /// <summary>
        /// Returns the DebugInformation for the Net.
        /// </summary>
        /// <param name="bDetectNans">Specifies whether or not to detect Nan's in the data.</param>
        /// <returns>The DebugInformation for the Net is returned.</returns>
        public DebugInformation<T> GetDebugInformation(bool bDetectNans)
        {
            if (m_blobWork == null)
                m_blobWork = new common.Blob<T>(m_cuda, m_log);
 
            DebugInformation<T> debugInfo = new DebugInformation<T>(name, m_blobWork, bDetectNans);

            for (int i = 0; i < m_rgLayers.Count; i++)
            {
                debugInfo.Add(m_rgLayers[i], m_rgcolBottomVecs[i], m_rgcolTopVecs[i]);
            }

            return debugInfo;
        }

        /// <summary>
        /// Finds a Blob in the Net by name.
        /// </summary>
        /// <param name="strName">Specifies the Blob name.</param>
        /// <returns>If found, the Blob is returned, otherwise <i>null</i> is returned.</returns>
        public Blob<T> FindBlob(string strName)
        {
            foreach (Blob<T> blob in blobs)
            {
                if (blob.Name == strName)
                    return blob;
            }

            foreach (Layer<T> layer in m_rgLayers)
            {
                foreach (Blob<T> blob in layer.blobs)
                {
                    if (blob.Name == strName)
                        return blob;
                }

                foreach (Blob<T> blob in layer.internal_blobs)
                {
                    if (blob.Name == strName)
                        return blob;
                }
            }

            return null;
        }

        /// <summary>
        /// Returns the data source used by the network.
        /// </summary>
        /// <returns>The data source name is returned.</returns>
        public string GetDataSource()
        {
            foreach (LayerParameter lp in m_param.layer)
            {
                if (lp.type == LayerParameter.LayerType.DATA ||
                    lp.type == LayerParameter.LayerType.TRIPLET_DATA)
                    return lp.data_param.source;

                if (lp.type == LayerParameter.LayerType.BATCHDATA)
                    return lp.batch_data_param.source;
            }

            return null;
        }
    }
}
