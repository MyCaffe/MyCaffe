using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ConvolutionLayer.  The default weight filler is set to the XavierFiller, and the default bias filler is set to ConstantFiller with a value of 0.1.
    /// </summary>
    /// <remarks>
    /// @see [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998.
    /// @see [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by Vincent Dumoulin, and Francesco Visin, 2016.
    /// @see [Joint Semantic and Motion Segmentation for dynamic scenes using Deep Convolutional Networks](https://arxiv.org/abs/1704.08331) by Nazrul Haque, N. Dinesh Reddy, and K. Madhava Krishna, 2017. 
    /// @see [A New Convolutional Network-in-Network Structure and Its Applications in Skin Detection, Semantic Segmentation, and Artifact Reduction](https://arxiv.org/abs/1701.06190v1) by Yoonsik Kim, Insung Hwang, and Nam Ik Cho, 2017.
    /// @see [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, and Trevor Darrell, 2014.
    /// @see [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by Fisher Yu, and Vladlen Koltun, 2015.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ConvolutionParameter : KernelParameter 
    {
        uint m_nNumOutput = 0;
        bool m_bBiasTerm = true;
        uint m_nGroup = 1;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        int m_nAxis = 1;
        bool m_bForceNDIm2Col = false;
        int m_nCudnnWorkspaceLimit = 1024 * 1024;   // Used with cuDnn only.
        bool m_bCudnnWorkspaceAllowOnGroups = false;
        bool m_bCudnnEnableTensorCores = false;

        /** @copydoc KernelParameter */
        public ConvolutionParameter()
        {            
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public string useCaffeReason(int nNumSpatialAxes = 2)
        {
            if (engine == Engine.CAFFE)
                return "The engine setting is set on CAFFE.";

            if (nNumSpatialAxes != 2)
                return "Currently only 2 spatial axes (ht x wd) are supported by cuDnn.";

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <param name="nNumSpatialAxes">Specifies the number of spatial axes used.  For example typically four spatial axes are used: N, C, H, W.</param>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn(int nNumSpatialAxes = 2)
        {
            if (engine == EngineParameter.Engine.CAFFE)
                return false;

            if (nNumSpatialAxes != 2)
                return false;

            return true;
        }


        /// <summary>
        /// When true, allows workspace usage on groups > 1 (default = false).
        /// </summary>
        /// <remarks>
        /// Currently using workspaces on groups > 1 can cause cuDnn memory errors and for this reason
        /// the default to this setting is false.
        /// </remarks>
        [Description("Specifies whether or not (default) worspaces are used when the group > 1.  Currently using workspaces on groups > 1 can cause cuDnn errors and for this reason defaults to false.")]
        public bool cudnn_workspace_allow_on_groups
        {
            get { return m_bCudnnWorkspaceAllowOnGroups; }
            set { m_bCudnnWorkspaceAllowOnGroups = value; }
        }

        /// <summary>
        /// Specifies the workspace limit used by cuDnn.  A value of 0 directs cuDNN to use the fastest algorithm possible.
        /// </summary>
        [Description("Specifies the cuDnn workspace limit to use.\n\n - A positive value, greater than zero, directs cuDnn to use the most efficient algorithm within the memory limit specified.\n - A value equal to zero directs cuDnn to use the most memory efficient algorithm.\n\n  The default for this value is 1,048,576 (i.e 1024 * 1024 which is then multiplied by 8 internally). This value is only used by the CUDNN and DEFAULT engines.")]
        [ReadOnly(true)]
        public int cudnn_workspace_limit
        {
            get { return m_nCudnnWorkspaceLimit; }
            set { m_nCudnnWorkspaceLimit = value; }
        }

        /// <summary>
        /// Specifies to enable the CUDA tensor cores when performing the convolution which is faster but not supported by all GPU's.
        /// </summary>
        /// <remarks>
        /// When run on GPU's that do not support Tensor cores, the default math (non-tensor core) is used.
        /// </remarks>
        [Description("Specifies to enable CUDA tensor cores when performing the convolution which is faster but not supported by all GPU's.  When not supported, the default math is used.")]
        public bool cudnn_enable_tensor_cores
        {
            get { return m_bCudnnEnableTensorCores; }
            set { m_bCudnnEnableTensorCores = value; }
        }

        /// <summary>
        /// The number of outputs for the layer.
        /// </summary>
        [Description("Specifies the number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Whether to have bias terms or not.
        /// </summary>
        [Description("Specifies the whether to have bias terms or not.")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
        }

        /// <summary>
        /// The group size for group convolution.
        /// </summary>
        [Description("Specifies the group size for group convolution.")]
        public uint group
        {
            get { return m_nGroup; }
            set { m_nGroup = value; }
        }

        /// <summary>
        /// The filler for the weight.  The default is set to use the 'xavier' filler.
        /// </summary>
        [Category("Fillers")]
        [Description("Specifies the filler used to initialize the weights.")]
        public FillerParameter weight_filler
        {
            get { return m_fillerParam_weights; }
            set { m_fillerParam_weights = value; }
        }

        /// <summary>
        /// The filler for the bias.  The default is set to use the 'constant = 0.1' filler.
        /// </summary>
        [Category("Fillers")]
        [Description("Specifies the filler used to initialize the bias.")]
        public FillerParameter bias_filler
        {
            get { return m_fillerParam_bias; }
            set { m_fillerParam_bias = value; }
        }

        /// <summary>
        /// The axis to interpret as 'channels' when performing convolution.
        /// Preceding dimensions are treated as independent inputs;
        /// succeeding dimensions are treated as 'spatial'.
        /// With @f$ (N \times C \times H \times W) @f$ inputs, and axis == 1 (the default), we perform
        /// N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
        /// groups g>1) filters across the spatial axes (H, W) of the input.
        /// With @f$ (N \times C \times D \times H \times W) @f$ inputs, and axis == 1, we perform
        /// N independent 3D convolutions, sliding (C/g)-channels
        /// filters across teh spatial axes @f$ (D \times H \times W) @f$ of the input.
        /// </summary>
        [Description("Specifies the axis to interpret as 'channels' when performing convolution  The preceding dimensions are treated as independent inputs; succeeding dimensions are treated as 'spatial'.  With (N,C,H,W) inputs and axis == 1 (the default), we perform N independent 2D convolutions, sliding C-channel (or C/g-channels, for groups > 1) filters across the spatial axes (H,W) of the input.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Whether to force use of the general ND convolution, even if a specific
        /// implementation for blobs of the appropriate number of spatial dimensions
        /// is available.  (Currently, there is only a 2D-specific convolution
        /// implementation; for input blobs with num_axes != 2, this option
        /// is ignored and the ND implementation will be used.)
        /// </summary>
        [Description("Specifies whether to force use of the general ND convolution, even if a specific implementation for blobs of the appopriate number of spatial dimensions is available.  (Currently, there is only a 2D-specific convolution implementation; for input blobs with 'num_axes' != 2, this option is ignored and the ND implementation is used.)")]
        public bool force_nd_im2col
        {
            get { return m_bForceNDIm2Col; }
            set { m_bForceNDIm2Col = value; }
        }

        /** @copydoc KernelParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ConvolutionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc KernelParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is ConvolutionParameter)
            {
                ConvolutionParameter p = (ConvolutionParameter)src;
                m_nNumOutput = p.m_nNumOutput;
                m_bBiasTerm = p.m_bBiasTerm;
                m_nGroup = p.m_nGroup;

                if (p.m_fillerParam_bias != null)
                    m_fillerParam_bias = p.m_fillerParam_bias.Clone();

                if (p.m_fillerParam_weights != null)
                    m_fillerParam_weights = p.m_fillerParam_weights.Clone();

                m_nAxis = p.m_nAxis;
                m_bForceNDIm2Col = p.m_bForceNDIm2Col;
                m_nCudnnWorkspaceLimit = p.m_nCudnnWorkspaceLimit;
                m_bCudnnWorkspaceAllowOnGroups = p.m_bCudnnWorkspaceAllowOnGroups;
                m_bCudnnEnableTensorCores = p.m_bCudnnEnableTensorCores;
            }
        }

        /** @copydoc KernelParameter::Clone */
        public override LayerParameterBase Clone()
        {
            ConvolutionParameter p = new ConvolutionParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc KernelParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("kernel");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("num_output", num_output.ToString());

            if (bias_term != true)
                rgChildren.Add("bias_term", bias_term.ToString());

            if (group != 1)
                rgChildren.Add("group", group.ToString());

            if (m_fillerParam_weights != null)
                rgChildren.Add(m_fillerParam_weights.ToProto("weight_filler"));

            if (bias_term && m_fillerParam_bias != null)
                rgChildren.Add(m_fillerParam_bias.ToProto("bias_filler"));

            if (axis != 1)
                rgChildren.Add("axis", axis.ToString());

            if (force_nd_im2col != false)
                rgChildren.Add("force_nd_im2col", force_nd_im2col.ToString());

            if (cudnn_workspace_limit != (1024 * 1024))
                rgChildren.Add("cudnn_workspace_limit", cudnn_workspace_limit.ToString());

            if (cudnn_workspace_allow_on_groups)
                rgChildren.Add("cudnn_worspace_allow_on_groups", cudnn_workspace_allow_on_groups.ToString());

            if (cudnn_enable_tensor_cores)
                rgChildren.Add("cudnn_enable_tensor_cores", cudnn_enable_tensor_cores.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /** @copydoc KernelParameter::FromProto */
        public static new ConvolutionParameter FromProto(RawProto rp)
        {
            string strVal;
            ConvolutionParameter p = new ConvolutionParameter();

            ((KernelParameter)p).Copy(KernelParameter.FromProto(rp));

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

            if ((strVal = rp.FindValue("group")) != null)
                p.group = uint.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("force_nd_im2col")) != null)
                p.force_nd_im2col = bool.Parse(strVal);

            if ((strVal = rp.FindValue("cudnn_workspace_limit")) != null)
                p.cudnn_workspace_limit = int.Parse(strVal);

            if ((strVal = rp.FindValue("cudnn_worspace_allow_on_groups")) != null)
                p.cudnn_workspace_allow_on_groups = bool.Parse(strVal);

            if ((strVal = rp.FindValue("cudnn_enable_tensor_cores")) != null)
                p.cudnn_enable_tensor_cores = bool.Parse(strVal);

            return p;
        }
    }
}
