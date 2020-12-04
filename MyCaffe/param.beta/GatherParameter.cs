using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the GatherLayer.
    /// </summary>
    public class GatherParameter : LayerParameterBase
    {
        int m_nAxis = 0;

        /** @copydoc LayerParameterBase */
        public GatherParameter()
        {
        }

        /// <summary>
        /// Specifies the first axis to gather: all preceding axes are retained in the output.
        /// May be negative to index from the end (e.g., -1 for the last axis).
        /// </summary>
        [Description("Specifies the first axis to gather: all preceding axes are retained in the output.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            GatherParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            GatherParameter p = (GatherParameter)src;
            m_nAxis = p.m_nAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            GatherParameter p = new GatherParameter();
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

            rgChildren.Add("axis", axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GatherParameter FromProto(RawProto rp)
        {
            string strVal;
            GatherParameter p = new GatherParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }

        /// <summary>
        /// Calculate the reshape array given the parameters.
        /// </summary>
        /// <param name="nAxis">Specifies the axis.</param>
        /// <param name="rgBtmShape">Specifies the shape parameter of the input data.</param>
        /// <param name="rgIdxShape">Specifies the shape of the indexes.</param>
        /// <param name="rgIdxF">Specifies the list of indices for verification, or null to ignore.</param>
        /// <param name="nDim">Specifies the dimension of data after the axis.</param>
        /// <param name="nDimAtAxis">Specifies the dimension at the axis.</param>
        /// <param name="nM">Specifies the count of items up to the axis.</param>
        /// <param name="nN">Specifies the count of items in the indexes.</param>
        /// <param name="strErr">Specifies the error string if any.</param>
        /// <returns></returns>
        public static List<int> Reshape(int nAxis, List<int> rgBtmShape, List<int> rgIdxShape, float[] rgIdxF, out int nDim, out int nDimAtAxis, out int nM, out int nN, out string strErr)
        {
            nAxis = Utility.CanonicalAxisIndex(nAxis, rgBtmShape.Count);
            nDim = Utility.Count(rgBtmShape, nAxis + 1);
            nDimAtAxis = rgBtmShape[nAxis];
            nM = Utility.Count(rgBtmShape, 0, nAxis);
            nN = Utility.Count(rgIdxShape);

            strErr = null;

            if (rgIdxF != null)
            {
                if (nN != rgIdxF.Length)
                {
                    strErr = "N should equal the number of indices.";
                    return null;
                }

                for (int i = 0; i < nN; i++)
                {
                    int nIdx = (int)rgIdxF[i];
                    if (nIdx < -nDimAtAxis || nIdx > nDimAtAxis)
                    {
                        strErr = "The index at idx=" + i.ToString() + " is out of range!  Must be within range [-" + nDimAtAxis.ToString() + "," + nDimAtAxis.ToString() + "]";
                        return null;
                    }
                }
            }

            List<int> rgTopShape = new List<int>(rgIdxShape);
            int nLen = rgTopShape.Count;

            while (rgTopShape.Count > 0 && rgTopShape[rgTopShape.Count - 1] == 1)
            {
                rgTopShape.RemoveAt(rgTopShape.Count - 1);
            }

            if (nAxis == 0)
                rgTopShape.Add(nDim);
            else if (nAxis == 1)
                rgTopShape.Insert(0, nM);

            for (int i = rgTopShape.Count; i < nLen; i++)
            {
                rgTopShape.Add(1);
            }

            return rgTopShape;
        }
    }
}
