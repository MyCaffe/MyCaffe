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
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AccuracyParameter : LayerParameterBase 
    {
        uint m_nTopK = 1;
        int m_nAxis = 1;
        List<int> m_rgnIgnoreLabel = new List<int>();
        bool m_bEnableSimpleAccuracy = false;
        bool m_bEnableLastElementOnly = false;

        /** @copydoc LayerParameterBase */
        public AccuracyParameter()
        {
        }

        /// <summary>
        /// Enables a simple accuracy calculation where the argmax is compared with the actual.
        /// </summary>
        [Description("Enables a simple argmax based accuracy calculation where the argmax is compared with the actual.")]
        public bool enable_simple_accuracy
        {
            get { return m_bEnableSimpleAccuracy; }
            set { m_bEnableSimpleAccuracy = value; }
        }

        /// <summary>
        /// When computing accuracy, only count the last element of the prediction blob.
        /// </summary>
        [Description("When computing accuracy, only count the last element of the prediction blob.")]
        public bool enable_last_element_only
        {
            get { return m_bEnableLastElementOnly; }
            set { m_bEnableLastElementOnly = value; }
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
        /// If specified, ignore instances with the given label(s).
        /// </summary>
        [Description("If specified, ignore instances with the given label(s).")]
        public List<int> ignore_labels
        {
            get { return m_rgnIgnoreLabel; }
            set
            {
                if (value == null)
                    m_rgnIgnoreLabel.Clear();
                else
                    m_rgnIgnoreLabel = value;
            }
        }

        /// <summary>
        /// Returns 'true' if the label is to be ignored.
        /// </summary>
        /// <param name="nLabel">Specifies the label to test.</param>
        /// <returns>Returns 'true' if the label is to be ignored.</returns>
        public bool IgnoreLabel(int nLabel)
        {
            return m_rgnIgnoreLabel.Contains(nLabel);
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
            m_bEnableSimpleAccuracy = p.m_bEnableSimpleAccuracy;
            m_bEnableLastElementOnly = p.m_bEnableLastElementOnly;
            m_rgnIgnoreLabel = Utility.Clone<int>(p.m_rgnIgnoreLabel);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            AccuracyParameter p = new AccuracyParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (top_k != 1)
                rgChildren.Add("top_k", top_k.ToString());
            
            if (axis != 1)
                rgChildren.Add("axis", axis.ToString());

            if (enable_simple_accuracy)
                rgChildren.Add("enable_simple_accuracy", enable_simple_accuracy.ToString());

            if (enable_last_element_only)
                rgChildren.Add("enable_last_element_only", enable_last_element_only.ToString());

            if (m_rgnIgnoreLabel.Count > 0)
            {
                if (m_rgnIgnoreLabel.Count == 1)
                    rgChildren.Add("ignore_label", m_rgnIgnoreLabel[0]);
                else
                    rgChildren.Add("ignore_labels", Utility.ToString<int>(m_rgnIgnoreLabel));
            }

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

            if ((strVal = rp.FindValue("enable_simple_accuracy")) != null)
                p.enable_simple_accuracy = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_last_element_only")) != null)
                p.enable_last_element_only = bool.Parse(strVal);

            if ((strVal = rp.FindValue("ignore_label")) != null)
            {
                p.ignore_labels.Clear();
                p.ignore_labels.Add(int.Parse(strVal));
            }

            if ((strVal = rp.FindValue("ignore_labels")) != null)
            {
                p.ignore_labels.Clear();
                
                string[] rgstr = strVal.Trim(' ', '{', '}').Split(',');

                foreach (string str in rgstr)
                {
                    if (!string.IsNullOrEmpty(str))
                        p.ignore_labels.Add(int.Parse(str));
                }
            }

            return p;
        }
    }
}
