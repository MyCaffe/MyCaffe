using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;
using System.Configuration;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ConstantLayer.
    /// </summary>
    public class ConstantParameter : LayerParameterBase 
    {
        BlobShape m_outputShape = new BlobShape();
        List<float> m_rgF = new List<float>();
        string m_strBinaryDataFile = null;

        /** @copydoc LayerParameterBase */
        public ConstantParameter()
        {
        }

        /// <summary>
        /// Specifies the output shape.
        /// </summary>
        [Description("Specifies the output shape.")]
        public BlobShape output_shape
        {
            get { return m_outputShape; }
            set { m_outputShape = value; }
        }

        /// <summary>
        /// Specifies a binary data file containing the values to load.
        /// </summary>
        /// <remarks>
        /// The binary data file is in the format:
        /// int nCount
        /// float f0
        /// float f1
        /// :
        /// float fn
        /// </remarks>
        [Description("Specifies a binary data file containing the values to load.")]
        public string binary_data_file
        {
            get { return m_strBinaryDataFile; }
            set { m_strBinaryDataFile = value; }
        }

        /// <summary>
        /// Specifies a set of float values used to fill the output.  When only one item is specified, all outputs are set to that value.
        /// </summary>
        [Description("Specifies a set of float values used to fill the output.  When only one item is specified, all outputs are set to that value.")]
        public List<float> values_f
        {
            get { return m_rgF; }
            set { m_rgF = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ConstantParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            ConstantParameter p = (ConstantParameter)src;
            m_outputShape = p.output_shape.Clone();
            m_rgF = new List<float>(p.m_rgF);
            m_strBinaryDataFile = p.binary_data_file;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            ConstantParameter p = new ConstantParameter();
            p.Copy(this);
            return p;
        }

        private static string replace(string str, char chTarget, char chReplace)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (ch == chTarget)
                    strOut += chReplace;
                else
                    strOut += ch;
            }

            return strOut;
        }


        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(m_outputShape.ToProto("output_shape"));
            rgChildren.Add<float>("valuef", m_rgF);

            if (!string.IsNullOrEmpty(m_strBinaryDataFile))
                rgChildren.Add("binary_data_file", replace(m_strBinaryDataFile, ' ', '~'));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ConstantParameter FromProto(RawProto rp)
        {
            string strVal;
            ConstantParameter p = new ConstantParameter();

            RawProto shape = rp.FindChild("output_shape");
            if (shape != null)
                p.m_outputShape = BlobShape.FromProto(shape);

            p.m_rgF = rp.FindArray<float>("valuef");

            strVal = rp.FindValue("binary_data_file");
            if (strVal != null)
                p.m_strBinaryDataFile = replace(strVal, '~', ' ');

            return p;
        }
    }
}
