﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters use to create a Net
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class NetParameter : BaseParameter 
    {
        string m_strName = "";
        List<string> m_rgstrInput = new List<string>();
        List<BlobShape> m_rgInputShape = new List<BlobShape>();
        List<int> m_rgInputDim = new List<int>();
        bool m_bForceBackward = false;
        NetState m_state = new NetState();
        bool m_bDebugInfo = false;
        List<LayerParameter> m_rgLayers = new List<LayerParameter>();
        int m_nProjectID = 0;
        int m_nSolverCount = 1;
        int m_nSolverRank = 0;
        bool m_bEnableMemoryStats = false;
        bool m_bEnableLoraOnlyLoad = false;
        bool m_bEnableLora = false;

        /** @copydoc BaseParameter */
        public NetParameter()
            : base()
        {
        }

        /// <summary>
        /// Save the parameter to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_strName);
            Utility.Save<string>(bw, m_rgstrInput);
            Utility.Save<BlobShape>(bw, m_rgInputShape);
            Utility.Save<int>(bw, m_rgInputDim);
            bw.Write(m_bForceBackward);
            m_state.Save(bw);
            bw.Write(m_bDebugInfo);
            Utility.Save<LayerParameter>(bw, m_rgLayers);
        }

        /// <summary>
        /// Load a new instance of the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The new instance is returned.</returns>
        public static NetParameter Load(BinaryReader br)
        {
            NetParameter p = new NetParameter();

            p.m_strName = br.ReadString();
            p.m_rgstrInput = Utility.Load<string>(br);
            p.m_rgInputShape = Utility.Load<BlobShape>(br);
            p.m_rgInputDim = Utility.Load<int>(br);
            p.m_bForceBackward = br.ReadBoolean();
            p.m_state = NetState.Load(br);
            p.m_bDebugInfo = br.ReadBoolean();
            p.m_rgLayers = Utility.Load<LayerParameter>(br);

            return p;
        }

        /// <summary>
        /// Specifies the ID of the project that created this net param (if any).
        /// </summary>
        [ReadOnly(true)]
        [Description("Specifies the ID of the project that created this net param (if any).")]
        public int ProjectID
        {
            get { return m_nProjectID; }
            set { m_nProjectID = value; }
        }

        /// <summary>
        /// The name of the network.
        /// </summary>
        [Description("The name of the network.")]
        public string name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        /// <summary>
        /// When using enabled, the network will only load the Lora model and not the base model into the learnable parameters that are updated saving memory.
        /// </summary>
        [Description("When using enabled, the network will only load the Lora model and not the base model into the learnable parameters that are updated saving memory.")]
        public bool enable_lora_only_load
        {
            get { return m_bEnableLoraOnlyLoad; }
            set { m_bEnableLoraOnlyLoad = value; }
        }

        /// <summary>
        /// When using enabled, all enabled output adapters are run on each layer using them.
        /// </summary>
        [Description("When using enabled, all enabled output adapters are run on each layer using them.")]
        public bool enable_lora
        {
            get { return m_bEnableLora; }
            set { m_bEnableLora = value; }
        }

        /// <summary>
        /// The input blobs to the network.
        /// </summary>
        [Browsable(false)]
        public List<string> input
        {
            get { return m_rgstrInput; }
            set { m_rgstrInput = value; }
        }

        /// <summary>
        /// The shape of the input blobs.
        /// </summary>
        [Browsable(false)]
        public List<BlobShape> input_shape
        {
            get { return m_rgInputShape; }
            set { m_rgInputShape = value; }
        }

        /// <summary>
        /// <b>DEPRECIATED</b> - 4D input dimensions - use 'input_shape' instead.
        /// If specified, for each input blob there should be four
        /// values specifying the num, channels, height and width of the input blob.
        /// Thus, there should be a total of (4 * #input) numbers.
        /// </summary>
        [Browsable(false)]
        public List<int> input_dim
        {
            get { return m_rgInputDim; }
            set { m_rgInputDim = value; }
        }

        /// <summary>
        /// Whether the network will force every layer to carry out backward operation.
        /// If set False, then whether to carry out backward is determined
        /// automatically according to the net structure and learning rates.
        /// </summary>
        [Description("Specifies whether or not the network will force every layer to carry out the backward operation.")]
        public bool force_backward
        {
            get { return m_bForceBackward; }
            set { m_bForceBackward = value; }
        }

        /// <summary>
        /// The current 'state' of the network, including the phase, level and stage.
        /// Some layers may be included/excluded depending on this state and the states
        /// specified in the layers' include and exclude fields.
        /// </summary>
        [Browsable(false)]
        public NetState state
        {
            get { return m_state; }
            set { m_state = value; }
        }

        /// <summary>
        /// Print debugging information about results while running Net::Forward,
        /// Net::Backward and Net::Update.
        /// </summary>
        [Description("Specifies whether or not to output debugging information to the output window.")]
        public bool debug_info
        {
            get { return m_bDebugInfo; }
            set { m_bDebugInfo = value; }
        }

        /// <summary>
        /// The layers that make up the net.  Each of their configurations, including
        /// connectivity and behavior, is specified as a LayerParameter.
        /// </summary>
        [Browsable(false)]
        public List<LayerParameter> layer
        {
            get { return m_rgLayers; }
            set { m_rgLayers = value; }
        }

        /// <summary>
        /// When enabled, memory use is output to the debug window (only recommended for debugging).
        /// </summary>
        [Browsable(false)]
        public bool enable_memory_stats
        {
            get { return m_bEnableMemoryStats; }
            set { m_bEnableMemoryStats = value; }
        }

        /// <summary>
        /// Specifies the number of solvers used in a multi-gpu training session.
        /// </summary>
        [Browsable(false)]
        public int solver_count
        {
            get { return m_nSolverCount; }
            set { m_nSolverCount = value; }
        }

        /// <summary>
        /// Specifies the rank of the solver using this network.
        /// </summary>
        [Browsable(false)]
        public int solver_rank
        {
            get { return m_nSolverRank; }
            set { m_nSolverRank = value; }
        }

        /** @copydoc BaseParameter */
        public override RawProto ToProto(string strName)
        {
            return ToProto(strName, false);
        }

        /// <summary>
        /// Save the parameter settings to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name, typically 'root'.</param>
        /// <param name="bIncludeState">Specifies whether or not to also include the state when saving.</param>
        /// <returns>The RawProto representing the settings is returned.</returns>
        public RawProto ToProto(string strName, bool bIncludeState)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("name", name, RawProto.TYPE.STRING);
            rgChildren.Add<string>("input", input);

            foreach (BlobShape bs in input_shape)
            {
                rgChildren.Add(bs.ToProto("input_shape"));
            }

            rgChildren.Add<int>("input_dim", input_dim);

            if (force_backward != false)
                rgChildren.Add("force_backward", force_backward.ToString());

            if (bIncludeState)
                rgChildren.Add(state.ToProto("state"));

            if (debug_info != false)
                rgChildren.Add("debug_info", debug_info.ToString());

            foreach (LayerParameter lp in layer)
            {
                rgChildren.Add(lp.ToProto("layer"));
            }

            rgChildren.Add("enable_memory_stats", enable_memory_stats.ToString());
            rgChildren.Add("enable_lora", enable_lora.ToString());
            rgChildren.Add("enable_lora_only_load", enable_lora_only_load.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parse a RawProto into a new instance of the parameter.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static NetParameter FromProto(RawProto rp)
        {
            string strVal;
            NetParameter p = new NetParameter();

            if ((strVal = rp.FindValue("name")) != null)
                p.name = strVal;

            p.input = rp.FindArray<string>("input");
            
            RawProtoCollection rgp = rp.FindChildren("input_shape");
            foreach (RawProto rpChild in rgp)
            {
                p.input_shape.Add(BlobShape.FromProto(rpChild));
            }

            p.input_dim = rp.FindArray<int>("input_dim");

            if ((strVal = rp.FindValue("force_backward")) != null)
                p.force_backward = bool.Parse(strVal);

            RawProto rpState = rp.FindChild("state");
            if (rpState != null)
                p.state = NetState.FromProto(rpState);

            if ((strVal = rp.FindValue("debug_info")) != null)
                p.debug_info = bool.Parse(strVal);

            rgp = rp.FindChildren("layer", "layers");
            foreach (RawProto rpChild in rgp)
            {
                p.layer.Add(LayerParameter.FromProto(rpChild));
            }

            if ((strVal = rp.FindValue("enable_memory_stats")) != null)
                p.enable_memory_stats = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_lora")) != null)
                p.enable_lora = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_lora_only_load")) != null)
                p.enable_lora_only_load = bool.Parse(strVal);

            return p;
        }

        /// <summary>
        /// Collect the inputs from the RawProto.
        /// </summary>
        /// <param name="rp">Specifies the raw proto.</param>
        /// <returns>A dictionary of the inputs and their shapes is returned.</returns>
        public static Dictionary<string, BlobShape> InputFromProto(RawProto rp)
        {
            List<string> rgstrInput = rp.FindArray<string>("input");
            List<BlobShape> rgShape = new List<BlobShape>();

            RawProtoCollection rgp = rp.FindChildren("input_shape");
            foreach (RawProto rpChild in rgp)
            {
                rgShape.Add(BlobShape.FromProto(rpChild));
            }

            if (rgstrInput.Count != rgShape.Count)
                throw new Exception("The input array and shape array must have the same count!");

            Dictionary<string, BlobShape> rgInput = new Dictionary<string, BlobShape>();
            for (int i = 0; i < rgstrInput.Count; i++)
            {
                rgInput.Add(rgstrInput[i], rgShape[i]);
            }

            return rgInput;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <param name="bCloneLayers">When <i>true</i>, each layer is cloned as well.</param>
        /// <param name="nSolverCount">Optionally, specifies a solver count for the clone.</param>
        /// <param name="nSolverRank">Optionally, specifies a solver rank for the clone.</param>
        /// <returns>A new instance of this parameter is returned.</returns>
        public NetParameter Clone(bool bCloneLayers = true, int? nSolverCount = null, int? nSolverRank = null)
        {
            NetParameter p = new NetParameter();

            p.m_nProjectID = m_nProjectID;
            p.m_strName = m_strName;
            p.m_rgstrInput = Utility.Clone<string>(m_rgstrInput);
            p.m_rgInputShape = Utility.Clone<BlobShape>(m_rgInputShape);
            p.m_rgInputDim = Utility.Clone<int>(m_rgInputDim);
            p.m_bForceBackward = m_bForceBackward;
            p.m_state = (m_state != null) ? m_state.Clone() : null;
            p.m_bDebugInfo = m_bDebugInfo;

            if (bCloneLayers)
                p.m_rgLayers = Utility.Clone<LayerParameter>(m_rgLayers);

            if (nSolverCount == null)
                nSolverCount = m_nSolverCount;

            if (nSolverRank == null)
                nSolverRank = m_nSolverRank;

            p.m_nSolverCount = nSolverCount.Value;
            p.m_nSolverRank = nSolverRank.Value;
            p.m_bEnableMemoryStats = m_bEnableMemoryStats;
            p.m_bEnableLora = m_bEnableLora;
            p.m_bEnableLoraOnlyLoad = m_bEnableLoraOnlyLoad;

            return p;
        }

        /// <summary>
        /// Returns a debug string for the network.
        /// </summary>
        /// <returns>The debug string is returned.</returns>
        public string DebugString()
        {
            string str = m_strName + " -> layers(" + m_rgLayers.Count.ToString() + ") {";

            foreach (LayerParameter p in m_rgLayers)
            {
                str += p.name;
                str += ", ";
            }

            str = str.TrimEnd(' ', ',');
            str += "}";

            return str;
        }

        /// <summary>
        /// Locates a layer based on a layer type and phase.
        /// </summary>
        /// <param name="type">Specifies the LayerParameter.LayerType to look for.</param>
        /// <param name="phase">Optionally, specifies a phase to look for.</param>
        /// <returns></returns>
        public LayerParameter FindLayer(LayerParameter.LayerType type, Phase phase = Phase.NONE)
        {
            foreach (LayerParameter p in m_rgLayers)
            {
                if (p.type == type)
                {
                    if (p.MeetsPhase(phase))
                        return p;
                }
            }

            return null;
        }

        /// <summary>
        /// Locates the index of a layer based on a given layer name.
        /// </summary>
        /// <param name="strName">Specifies the layer name to look for.</param>
        /// <returns>The index of the layer is returned.</returns>
        public int FindLayerIndex(string strName)
        {
            for (int i = 0; i < m_rgLayers.Count; i++)
            {
                if (m_rgLayers[i].name == strName)
                    return i;
            }

            return -1;
        }

        /// <summary>
        /// Sets all pooling layers to use the specified reshape algorithm.
        /// </summary>
        /// <param name="alg">Specifies the reshape algorithm to use (DEFAULT, CAFFE or ONNX).</param>
        public void SetPoolingReshapeAlgorithm(PoolingParameter.PoolingReshapeAlgorithm alg)
        {
            foreach (LayerParameter layer in m_rgLayers)
            {
                if (layer.type == LayerParameter.LayerType.POOLING)
                    layer.pooling_param.reshape_algorithm = alg;
            }
        }
    }
}
