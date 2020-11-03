using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by ReductionLayer.
    /// </summary>
    public class ReductionParameter : LayerParameterBase
    {
        /// <summary>
        /// Defines the reduction operation.
        /// </summary>
        public enum ReductionOp
        {
            /// <summary>
            /// Sum the values.
            /// </summary>
            SUM = 1,
            /// <summary>
            /// Sum the absolute values.
            /// </summary>
            ASUM = 2,
            /// <summary>
            /// Sum the squared values.
            /// </summary>
            SUMSQ = 3,
            /// <summary>
            /// Calculate the mean value.
            /// </summary>
            MEAN = 4
        }

        ReductionOp m_operation = ReductionOp.SUM;
        int m_nAxis = 0;
        double m_dfCoeff = 1.0;

        /** @copydoc LayerParameterBase */
        public ReductionParameter()
        {
        }

        /// <summary>
        /// Specifies the reduction operation.
        /// </summary>
        [Description("Specifies the reduction operation.")]
        public ReductionOp operation
        {
            get { return m_operation; }
            set { m_operation = value; }
        }

        /// <summary>
        /// The first axis to reduce to scalar -- may be negative index from the
        /// end (e.g., -1 for the last axis).
        /// (Currently, only reduction along ALL 'tail' axes is supported; reduction
        /// on axis M through N, where N less than num_axis - 1, is unsupported.)
        /// Suppose we have an n-axis bottom Blob with shape:
        ///   (d0, d1, d2, ..., d(m-1), dm, d(m+1), ..., d(n-1)).
        ///  
        /// if (axis == m, the output Blob will have shape
        ///   (d0, d1, d2, ..., d(m-1)),
        ///   
        /// and the ReductionOp operation is performed (d0 * d1 * d2 * ... * d(m-1))
        /// times, each including (dm * d(m+1) * ... * d(n-1)) individual data.
        /// 
        /// if axis == 0 (the default), the output Blob always has the empty shape
        /// (count 1), performing reduction across the entire input --
        /// often useful for creating new loss functions.
        /// </summary>
        [Description("Specifies the first axis to reduce to scalar -- may be negative to index from the end (e.g., -1 for the last axis)..")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the coefficient used to scale the output.
        /// </summary>
        [Description("Specifies the coefficient for output.")]
        public double coeff
        {
            get { return m_dfCoeff; }
            set { m_dfCoeff = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ReductionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ReductionParameter p = (ReductionParameter)src;
            m_operation = p.m_operation;
            m_nAxis = p.m_nAxis;
            m_dfCoeff = p.m_dfCoeff;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ReductionParameter p = new ReductionParameter();
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

            if (axis != 0)
                rgChildren.Add("axis", axis.ToString());

            if (coeff != 1.0)
                rgChildren.Add("coeff", coeff.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ReductionParameter FromProto(RawProto rp)
        {
            string strVal;
            ReductionParameter p = new ReductionParameter();

            if ((strVal = rp.FindValue("operation")) != null)
            {
                switch (strVal)
                {
                    case "ASUM":
                        p.operation = ReductionOp.ASUM;
                        break;

                    case "MEAN":
                        p.operation = ReductionOp.MEAN;
                        break;

                    case "SUM":
                        p.operation = ReductionOp.SUM;
                        break;

                    case "SUMSQ":
                        p.operation = ReductionOp.SUMSQ;
                        break;

                    default:
                        throw new Exception("Uknown 'operation' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("coeff")) != null)
                p.coeff = parseDouble(strVal);

            return p;
        }
    }
}
