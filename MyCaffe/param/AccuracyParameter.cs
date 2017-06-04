using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the AccuracyLayer.
    /// </summary>
    /// <remarks>
    /// @see [Convolutional Architecture Exploration for Action Recognition and Image Classification](https://arxiv.org/abs/1512.07502v1) by J. T. Turner, David Aha, Leslie Smith, and Kalyan Moy Gupta, 2015.
    /// </remarks>
    public class AccuracyParameter : LayerParameterBase 
    {
        uint m_nTopK = 1;
        int m_nAxis = 1;
        int? m_nIgnoreLabel = null;

        /** @copydoc LayerParameterBase */
        public AccuracyParameter()
        {
        }

        /// <summary>
        /// When computing accuracy, count as correct by comparing the true label to
        /// the top_k scoring classes.  By default, only compare the top scoring
        /// class (i.e. argmax).
        /// </summary>
        [Description("When computing accuracy, count as correct by comparing the true label to the 'top_k' scoring classes.  By default, only compare the top scoring class (i.e. argmax).")]
        public uint top_k
        {
            get { return m_nTopK; }
            set { m_nTopK = value; }
        }

        /// <summary>
        /// The 'label' axis of the prediction blob, whos argmax corresponds to the
        /// predicted label -- may be negative to index from the end (e.g., -1 for the
        /// last axis).  For example, if axis == 1 and the predictions are
        /// @f$ (N \times C \times H \times W) @f$, the label blob is expected to 
        /// contain N*H*W ground truth labels with integer values in {0, 1, ..., C-1}.
        /// </summary>
        [Description("Specifies the 'label' axis of the prediction blob, whos argmax corresponds to the predicted label -- may be negative to index from the end (e.g., -1 for the last axis).  For example, if axis == 1 and the predictions are (NxCxHxW), the label blob is expected to contain N*H*W ground truth labels with integer values in {0, 1,..., C-1}.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// If specified, ignore instances with the given label.
        /// </summary>
        [Description("If specified, ignore instances with the given label.")]
        public int? ignore_label
        {
            get { return m_nIgnoreLabel; }
            set { m_nIgnoreLabel = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            AccuracyParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            AccuracyParameter p = (AccuracyParameter)src;
            m_nTopK = p.m_nTopK;
            m_nAxis = p.m_nAxis;
            m_nIgnoreLabel = p.m_nIgnoreLabel;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            AccuracyParameter p = new AccuracyParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (top_k != 1)
                rgChildren.Add("top_k", top_k.ToString());

            if (axis != 1)
                rgChildren.Add("axis", axis.ToString());

            if (m_nIgnoreLabel.HasValue)
                rgChildren.Add("ignore_label", m_nIgnoreLabel.Value);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static AccuracyParameter FromProto(RawProto rp)
        {
            string strVal;
            AccuracyParameter p = new AccuracyParameter();

            if ((strVal = rp.FindValue("top_k")) != null)
                p.top_k = uint.Parse(strVal);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("ignore_label")) != null)
                p.ignore_label = int.Parse(strVal);

            return p;
        }
    }
}
