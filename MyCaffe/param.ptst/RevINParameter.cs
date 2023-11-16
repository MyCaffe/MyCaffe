using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the RevINLayer (Reversible Instance Normalization).  
    /// </summary>
    /// <remarks>
    /// This layer performs the reversible instance normalization that normalizes the inputs by centering the data then brining
    /// it to the unit variance and then applying a learnable affine weight and bias.  This layer performs both normalization and
    /// denormalization functions.  The output of the layer is a (B x T x Ch) in size.
    /// 
    /// @see [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p) by Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo, 2022, ICLR 2022
    /// @see [Github - ts-kim/RevIN](https://github.com/ts-kim/RevIN) by ts-kim, 2022, GitHub.
    /// @see [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam, International conference on machine learning, 2022, arXiv:2211.14730
    /// @see [Github - yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) by yuqinie98, 2022, GitHub.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class RevINParameter : LayerParameterBase
    {
        int m_nNumFeatures = 0;
        float m_fEps = 1e-05f;
        bool m_bAffine = false;
        bool m_bSubtractLast = false;
        MODE m_mode = MODE.NORMALIZE;

        /// <summary>
        /// Defines the mode of operation.
        /// </summary>
        public enum MODE
        {
            /// <summary>
            /// Specifies to normalize the data.
            /// </summary>
            NORMALIZE = 0,
            /// <summary>
            /// Specifies to denormalize the data.
            /// </summary>
            DENORMALIZE = 1
        }

        /** @copydoc LayerParameterBase */
        public RevINParameter()
        {
        }

        /// <summary>
        /// Specifies the mode of operation - NORMALIZE or DENORMALIZE.
        /// </summary>
        [Description("Specifies the mode of operation - NORMALIZE or DENORMALIZE.")]
        public MODE mode
        {
            get { return m_mode; }
            set { m_mode = value; }
        }

        /// <summary>
        /// Specifies the number of features in the channel.
        /// </summary>
        [Description("Specifies the number of features in the channel.")]
        public int num_features
        {
            get { return m_nNumFeatures; }
            set { m_nNumFeatures = value; }
        }

        /// <summary>
        /// Specifies the epsilon used to prevent division by zero when normalizing.
        /// </summary>
        [Description("Specifies the epsilon used to prevent division by zero when normalizing.")]
        public float eps
        {
            get { return m_fEps; }
            set { m_fEps = value; }
        }

        /// <summary>
        /// Specifies whether or not to apply an affine weight and bias to the normalized data.
        /// </summary>
        [Description("Specifies whether or not to apply an affine weight and bias to the normalized data.")]
        public bool affine
        {
            get { return m_bAffine; }
            set { m_bAffine = value; }
        }

        /// <summary>
        /// Specifies whether or not to subtract the last value from the input before normalizing.
        /// </summary>
        [Description("Specifies whether or not to subtract the last value from the input before normalizing.")]
        public bool subtract_last
        {
            get { return m_bSubtractLast; }
            set { m_bSubtractLast = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            RevINParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            RevINParameter p = (RevINParameter)src;

            m_mode = p.m_mode;
            m_nNumFeatures = p.m_nNumFeatures;
            m_fEps = p.m_fEps;
            m_bAffine = p.m_bAffine;
            m_bSubtractLast = p.m_bSubtractLast;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            RevINParameter p = new RevINParameter();
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

            rgChildren.Add("mode", mode.ToString());
            rgChildren.Add("num_features", num_features.ToString());
            rgChildren.Add("eps", eps.ToString());
            rgChildren.Add("affine", affine.ToString());
            rgChildren.Add("subtract_last", subtract_last.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static RevINParameter FromProto(RawProto rp)
        {
            string strVal;
            RevINParameter p = new RevINParameter();

            if ((strVal = rp.FindValue("mode")) != null)
            {
                if (strVal == "NORMALIZE")
                    p.mode = MODE.NORMALIZE;
                else if (strVal == "DENORMALIZE")
                    p.mode = MODE.DENORMALIZE;
                else
                    throw new Exception("Unknown mode '" + strVal + "'!");
            }

            if ((strVal = rp.FindValue("num_features")) != null)
                p.num_features = int.Parse(strVal);

            if ((strVal = rp.FindValue("eps")) != null)
                p.eps = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("affine")) != null)
                p.affine = bool.Parse(strVal);

            if ((strVal = rp.FindValue("subtract_last")) != null)
                p.subtract_last = bool.Parse(strVal);

            return p;
        }
    }
}
