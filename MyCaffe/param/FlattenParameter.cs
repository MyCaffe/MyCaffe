using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the FlattenLayer.
    /// </summary>
    /// <remarks>
    /// @see [Representation Learning and Pairwise Ranking for Implicit and Explicit Feedback in Recommendation Systems](https://arxiv.org/abs/1705.00105v1) by Mikhail Trofimov, Sumit Sidana, Oleh Horodnitskii, Charlotte Laclau, Yury Maximov, and Massih-Reza Amini, 2017. 
    /// @see [Deep Neural Networks to Enable Real-time Multimessenger Astrophysics](https://arxiv.org/abs/1701.00008v2) by Daniel George, and E. A. Huerta, 2016.
    /// </remarks>
    public class FlattenParameter : LayerParameterBase
    {
        int m_nAxis = 1;
        int m_nEndAxis = -1;

        /** @copydoc LayerParameterBase */
        public FlattenParameter()
        {
        }

        /// <summary>
        /// Specifies the first axis to flatten: all preceding axes are retained in the output.
        /// May be negative to index from the end (e.g., -1 for the last axis).
        /// </summary>
        [Description("Specifies the first axis to flatten: all preceding axes are retained in the output.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the last axis to flatten: all following axes are retained in the output.
        /// May be negative to index from the end (e.g., -1 for the last axis).
        /// </summary>
        [Description("Specifies the last axis to flatten: all following axes are retained in the output.")]
        public int end_axis
        {
            get { return m_nEndAxis; }
            set { m_nEndAxis = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            FlattenParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            FlattenParameter p = (FlattenParameter)src;
            m_nAxis = p.m_nAxis;
            m_nEndAxis = p.m_nEndAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            FlattenParameter p = new FlattenParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("end_axis", end_axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static FlattenParameter FromProto(RawProto rp)
        {
            string strVal;
            FlattenParameter p = new FlattenParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("end_axis")) != null)
                p.end_axis = int.Parse(strVal);

            return p;
        }

        /// <summary>
        /// Calculate the reshape array given the parameters.
        /// </summary>
        /// <param name="nParamAxis">Specifies the parameter start axis.</param>
        /// <param name="nParamEndAxis">Specifies the parameter end axis.</param>
        /// <param name="rgShape">Specifies the shape parameter.</param>
        /// <param name="nStartAxis">Specifies the already initialized canonical start axis, or -1 if not initialized (default = -1).</param>
        /// <param name="nEndAxis">Specifies the already initialized canonical end axis or -1 if not initialized (default = -1).</param>
        /// <returns></returns>
        public static List<int> Reshape(int nParamAxis, int nParamEndAxis, List<int> rgShape, int nStartAxis = -1, int nEndAxis = -1)
        {
            if (nStartAxis < 0)
                nStartAxis = Utility.CanonicalAxisIndex(nParamAxis, rgShape.Count);

            if (nEndAxis < 0)
                nEndAxis = Utility.CanonicalAxisIndex(nParamEndAxis, rgShape.Count);

            List<int> rgTopShape = new List<int>();
            for (int i = 0; i < nStartAxis; i++)
            {
                rgTopShape.Add(rgShape[i]);
            }

            int nFlattenDim = Utility.Count(rgShape, nStartAxis, nEndAxis + 1);
            rgTopShape.Add(nFlattenDim);

            for (int i = nEndAxis + 1; i < rgShape.Count; i++)
            {
                rgTopShape.Add(rgShape[i]);
            }

            return rgTopShape;
        }
    }
}
