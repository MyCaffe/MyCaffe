using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores the parameters used by loss layers.
    /// </summary>
    public class LossParameter : LayerParameterBase 
    {
        int? m_nIgnoreLabel = null;
        NormalizationMode m_normalization = NormalizationMode.VALID;
        bool m_bNormalize = false;

        /// <summary>
        /// How to normalize the loss for loss layers that aggregate across batches,
        /// spatial dimensions, or other dimensions.  Currenly only implemented in
        /// SoftmaxWithLoss layer.
        /// </summary>
        public enum NormalizationMode
        {
            /// <summary>
            /// Divide by the number of examples in the batch times spatial dimensions.
            /// Outputs that receive the ignore label will NOT be ignored in computing
            /// the normalization factor.
            /// </summary>
            FULL = 0,

            /// <summary>
            /// Divide by the total number of output locations that do not take the
            /// ignore label.  If ignore label is not set, this behaves like FULL.
            /// </summary>
            VALID = 1,

            /// <summary>
            /// Divide by the batch size.
            /// </summary>
            BATCH_SIZE = 2,

            /// <summary>
            /// Do not normalize the loss.
            /// </summary>
            NONE = 3
        }

        /** @copydoc LayerParameterBase */
        public LossParameter()
        {
        }

        /// <summary>
        /// If specified, the ignore instances with the given label.
        /// </summary>
        [Description("Ignore instances with the given label, when specified.")]
        public int? ignore_label
        {
            get { return m_nIgnoreLabel; }
            set { m_nIgnoreLabel = value; }
        }

        /// <summary>
        /// Specifies the normalization mode (default = VALID).
        /// </summary>
        [Description("Specifies the normalization mode to use (default = VALID).")]
        public NormalizationMode normalization
        {
            get { return m_normalization; }
            set { m_normalization = value; }
        }

        /// <summary>
        /// <b>DEPRECIATED</b>.  Ignore if normalization is specified.  If normalization
        /// is not specified, then setting this to false will be equivalent to 
        /// normalization = BATCH_SIZE to be consistent with previous behavior.
        /// </summary>
        [Description("DEPRECIATED - use 'normalization == BATCH_SIZE' instead.")]
        [Browsable(false)]
        public bool normalize
        {
            get { return m_bNormalize; }
            set { m_bNormalize = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            LossParameter p = (LossParameter)src;
            
            m_nIgnoreLabel = p.m_nIgnoreLabel;
            m_normalization = p.m_normalization;
            m_bNormalize = p.m_bNormalize;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LossParameter p = new LossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (ignore_label.HasValue)
                rgChildren.Add("ignore_label", ignore_label);

            if (normalization != NormalizationMode.NONE)
                rgChildren.Add("normalization", normalization.ToString());

            if (normalization == NormalizationMode.NONE && normalize != false)
                rgChildren.Add("normalize", normalize.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LossParameter FromProto(RawProto rp)
        {
            string strVal;
            LossParameter p = new LossParameter();

            p.ignore_label = (int?)rp.FindValue("ignore_label", typeof(int));

            if ((strVal = rp.FindValue("normalization")) != null)
            {
                switch (strVal)
                {
                    case "FULL":
                        p.normalization = NormalizationMode.FULL;
                        break;

                    case "VALID":
                        p.normalization = NormalizationMode.VALID;
                        break;

                    case "BATCH_SIZE":
                        p.normalization = NormalizationMode.BATCH_SIZE;
                        break;

                    case "NONE":
                        p.normalization = NormalizationMode.NONE;
                        break;

                    default:
                        throw new Exception("Unknown 'normalization' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("normalize")) != null)
                p.normalize = bool.Parse(strVal);

            return p;
        }
    }
}
