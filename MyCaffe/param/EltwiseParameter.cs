using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the EltwiseLayer.
    /// </summary>
    /// <remarks>
    /// @see [DeMeshNet: Blind Face Inpainting for Deep MeshFace Verification](https://arxiv.org/abs/1611.05271v1) by Shu Zhang, Ran He, and Tieniu Tan, 2016. 
    /// @see [Mixed context networks for semantic segmentation](https://arxiv.org/abs/1610.05854v1) by Haiming Sun, Di Xie, and Shiliang Pu, 2016. 
    /// @see [Why M Heads are Better than One: Training a Diverse Ensemble of Deep Networks](https://arxiv.org/abs/1511.06314v1) by Stefan Lee, Senthil Purushwalkam, Michael Cogswell, David Crandall, and Dhruv Batra, 2015.
    /// </remarks>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class EltwiseParameter : LayerParameterBase
    {
        /// <summary>
        /// Defines the operation to perform.
        /// </summary>
        public enum EltwiseOp
        {
            /// <summary>
            /// Perform an eltwise product.
            /// </summary>
            PROD = 0,
            /// <summary>
            /// Perform an eltwise summation.
            /// </summary>
            SUM = 1,
            /// <summary>
            /// Find the eltwise maximum.
            /// </summary>
            MAX = 2,
            /// <summary>
            /// Find the eltwise minimum.
            /// </summary>
            MIN = 3,
            /// <summary>
            /// Perform an eltwise division.
            /// </summary>
            DIV = 4,
            /// <summary>
            /// Perform the eltwise subtraction
            /// </summary>
            SUB = 5
        }

        EltwiseOp m_operation = EltwiseOp.SUM;
        List<double> m_rgdfCoeff = new List<double>();
        bool m_bStableProdGrad = true;
        bool m_bCoeffBlob = false;

        /** @copydoc LayerParameterBase */
        public EltwiseParameter()
        {
        }

        /// <summary>
        /// Specifies the element-wise operation.
        /// </summary>
        [Description("Specifies the element-wise operation to perform.")]
        public EltwiseOp operation
        {
            get { return m_operation; }
            set { m_operation = value; }
        }

        /// <summary>
        /// Specifies the blob-wise coefficient for SUM operation.
        /// </summary>
        [Description("Specifies the blob-wise coefficient for SUM operation.")]
        public List<double> coeff
        {
            get { return m_rgdfCoeff; }
            set { m_rgdfCoeff = value; }
        }

        /// <summary>
        /// Specifies whether or not to use an asymptotically slower (for > 2 inputs) but stabler method
        /// of computing the gradient for PROD operation.  (No effect for SUM op.)
        /// </summary>
        [Description("Specifies whether or not to use an asymtotically slower (for > 2 inputs) but stabler method for computing the gradient for PROD operation (No effect for SUM operation).")]
        public bool stable_prod_grad
        {
            get { return m_bStableProdGrad; }
            set { m_bStableProdGrad = value; }
        }

        /// <summary>
        /// If true and the EltwiseOp is SUM, the last bottom blob is a singleton
        /// coefficient for the first N-1 bottom blobs, with shape @f$ (N-1 \times 1 \times 1 \times 1) @f$.
        /// </summary>
        [Description("If true and EltwiseOp is SUM, the last bottom blob is a singleton coefficient for the first N-1 bottom blobs, with shape (N-1, 1, 1, 1).")]
        public bool coeff_blob
        {
            get { return m_bCoeffBlob; }
            set { m_bCoeffBlob = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            EltwiseParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            EltwiseParameter p = (EltwiseParameter)src;
            m_operation = p.m_operation;
            m_rgdfCoeff = Utility.Clone<double>(p.m_rgdfCoeff);
            m_bStableProdGrad = p.m_bStableProdGrad;
            m_bCoeffBlob = p.m_bCoeffBlob;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            EltwiseParameter p = new EltwiseParameter();
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

            rgChildren.Add("operation", operation.ToString());
            rgChildren.Add<double>("coeff", coeff);

            if (stable_prod_grad != true)
                rgChildren.Add("stable_prod_grad", stable_prod_grad.ToString());

            if (coeff_blob != false)
                rgChildren.Add("coeff_blob", coeff_blob.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static EltwiseParameter FromProto(RawProto rp)
        {
            string strVal;
            EltwiseParameter p = new EltwiseParameter();

            if ((strVal = rp.FindValue("operation")) != null)
            {
                switch (strVal)
                {
                    case "PROD":
                        p.operation = EltwiseOp.PROD;
                        break;

                    case "SUM":
                        p.operation = EltwiseOp.SUM;
                        break;

                    case "MAX":
                        p.operation = EltwiseOp.MAX;
                        break;

                    case "MIN":
                        p.operation = EltwiseOp.MIN;
                        break;

                    case "DIV":
                        p.operation = EltwiseOp.DIV;
                        break;

                    case "SUB":
                        p.operation = EltwiseOp.SUB;
                        break;

                    default:
                        throw new Exception("Unknown 'operation' value: " + strVal);
                }
            }

            p.coeff = rp.FindArray<double>("coeff");

            if ((strVal = rp.FindValue("stable_prod_grad")) != null)
                p.stable_prod_grad = bool.Parse(strVal);

            if ((strVal = rp.FindValue("coeff_blob")) != null)
                p.coeff_blob = bool.Parse(strVal);

            return p;
        }
    }
}
