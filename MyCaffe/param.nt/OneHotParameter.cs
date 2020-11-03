using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param.nt
{
    /// <summary>
    /// Specifies the parameters used by the OneHotLayer
    /// </summary>
    /// <remarks>
    /// The OneHotLayer converts each single input into 'num_output' vector items each containing a 1 or 0 
    /// if the input falls within the range of the given vector item.
    /// 
    /// For example, when using a min/max range of -1,1 spread across 8 vector items (num_output), inputs
    /// less than or equal to -1 go in the first bucket, inputs greater than or equal to 1 go in the last
    /// bucket and values in between -1 and 1 go into their repsective buckets (e.g input -0.12 goes into bucket
    /// index 3 and input 0.12 goes into bucket 4)
    /// 
    /// 8 inputs span across -1 to 1 range creates the following buckets:
    /// 
    /// index:        0            1            2            3           4           5           6           7 
    /// bucket: [-1.00,-0.75][-0.75,-0.50][-0.50,-0.25][-0.25, 0.00][0.00, 0.25][0.25, 0.50][0.50, 0.75][0.75, 1.00]
    /// 
    /// input: -0.75 or less set bucket #0 = 1
    /// input:  0.75 or greater set bucket #7 = 1
    /// 
    /// Except for end buckets, inputs are placed in bucket where:  bucket min &lt;= input &lt; bucket max.
    /// </remarks>
    public class OneHotParameter : LayerParameterBase
    {
        int m_nAxis = 2;
        uint m_nNumOutput = 16;
        double m_dfMin = -1.0;
        double m_dfMax = 1.0;
        int m_nMinAxes = 4;

        /// <summary>
        /// The constructor.
        /// </summary>
        public OneHotParameter()
        {
        }

        /// <summary>
        /// Specifies the axis over which to apply the one-hot vectoring.
        /// </summary>
        [Description("Specifies the axis over which to apply the one-hot vectoring.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the minimum number of axes.  Axes of size 1 are added to the current axis count up to the minimum.
        /// </summary>
        [Description("Specifies the minimum number of axes.  Axes of size 1 are added to the current axis count up to the minimum.")]
        public int min_axes
        {
            get { return m_nMinAxes; }
            set { m_nMinAxes = value; }
        }

        /// <summary>
        /// Specifies the number of items within the one-hot vector output.
        /// </summary>
        [Description("Specifies the number of items within the one-hot vector output.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Specifies the minimum data range over which to bucketize for the one-hot vector input.
        /// </summary>
        [Description("Specifies the minimum data range over which to bucketize for the one-hot vector input.")]
        public double min
        {
            get { return m_dfMin; }
            set { m_dfMin = value; }
        }

        /// <summary>
        /// Specifies the maximum data range over which to bucketize for the one-hot vector input.
        /// </summary>
        [Description("Specifies the maximum data range over which to bucketize for the one-hot vector input.")]
        public double max
        {
            get { return m_dfMax; }
            set { m_dfMax = value; }
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
            OneHotParameter p = FromProto(proto);

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
            OneHotParameter p = (OneHotParameter)src;

            m_nAxis = p.m_nAxis;
            m_nNumOutput = p.m_nNumOutput;
            m_dfMin = p.m_dfMin;
            m_dfMax = p.m_dfMax;
            m_nMinAxes = p.m_nMinAxes;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            OneHotParameter p = new OneHotParameter();
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
            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("min", min.ToString());
            rgChildren.Add("max", max.ToString());
            rgChildren.Add("min_axes", min_axes.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static OneHotParameter FromProto(RawProto rp)
        {
            string strVal;
            OneHotParameter p = new OneHotParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("min")) != null)
                p.min = ParseDouble(strVal);

            if ((strVal = rp.FindValue("max")) != null)
                p.max = ParseDouble(strVal);

            if ((strVal = rp.FindValue("min_axes")) != null)
                p.min_axes = int.Parse(strVal);

            return p;
        }
    }
}
