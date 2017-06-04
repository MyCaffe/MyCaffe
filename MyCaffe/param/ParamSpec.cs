using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies training parameters (multipliers on global learning constants,
    /// and the name of other settings used for weight sharing).
    /// </summary>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ParamSpec : BaseParameter, ICloneable, IBinaryPersist 
    {
        /// <summary>
        /// The names of the parameter blobs -- useful for sharing parameters among
        /// layers, but never requires otherwise.  To share a parameter between two
        /// layers, give it a (non-empty) name.
        /// </summary>
        string m_strName = "";

        /// <summary>
        /// Whether to require shared weights to have the same shape, or just the same
        /// count -- defaults to STRICT if unspecified.
        /// </summary>
        DimCheckMode m_shareMode = DimCheckMode.STRICT;

        /// <summary>
        /// Defines the dimension check mode.
        /// </summary>
        public enum DimCheckMode
        {
            /// <summary>
            /// STRICT (default) requires that num, channels, height, width each match.
            /// </summary>
            STRICT = 0,

            /// <summary>
            /// PERMISSIVE requires only the count (num*channels*height*width) to match.
            /// </summary>
            PERMISSIVE = 1
        }

        /// <summary>
        /// The multiplier on the global learning rate for this parameter.
        /// </summary>
        double m_dfLrMult = 1.0;

        /// <summary>
        /// The multiplier on the global weight decay for this parameter.
        /// </summary>
        double m_dfDecayMult = 1.0;

        /// <summary>
        /// The ParamSpec constructor.
        /// </summary>
        public ParamSpec()
        {
        }

        /// <summary>
        /// The ParamSpec constructor.
        /// </summary>
        /// <param name="strName">Specifies a name given to the ParamSpec.</param>
        public ParamSpec(string strName)
        {
            m_strName = strName;
        }

        /// <summary>
        /// The ParamSpec constructor.
        /// </summary>
        /// <param name="dfLrMult">Specifies the learning rate multiplier given to the ParamSpec.</param>
        /// <param name="dfDecayMult">Specifies the decay rate multiplier given to the ParamSpec.</param>
        /// <param name="strName">Specifies the name given to the ParamSpec.</param>
        public ParamSpec(double dfLrMult, double dfDecayMult, string strName = null)
        {
            m_dfLrMult = dfLrMult;
            m_dfDecayMult = dfDecayMult;

            if (strName != null)
                m_strName = strName;
        }

        /// <summary>
        /// Saves the ParamSpec to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_strName);
            bw.Write((int)m_shareMode);
            bw.Write(m_dfLrMult);
            bw.Write(m_dfDecayMult);
        }

        /// <summary>
        /// Loads a ParamSpec from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader to use.</param>
        /// <param name="bNewInstance">When <i>true</i>, a new ParamSpec instance is created and loaded, otherwise this instance is loaded.</param>
        /// <returns>The ParamSpec instance loaded is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            ParamSpec p = this;

            if (bNewInstance)
                p = new ParamSpec();

            p.m_strName = br.ReadString();
            p.m_shareMode = (DimCheckMode)br.ReadInt32();
            p.m_dfLrMult = br.ReadDouble();
            p.m_dfDecayMult = br.ReadDouble();

            return p;
        }

        [Browsable(false)]
        public string Name /** @private */
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Specifies the name of this parameter.
        /// </summary>
        [Description("Specifies the name of this parameter.")]
        public string name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        /// <summary>
        /// Specifies whether to require shared weights to have the same shape, or just the same count - defaults to STICT (same shape).
        /// </summary>
        [Description("Specifies whether to require shared weights to have the same shape, or just the same count - defaults to STICT (same shape).")]
        public DimCheckMode share_mode
        {
            get { return m_shareMode; }
            set { m_shareMode = value; }
        }

        /// <summary>
        /// Specifies the multiplier used on the global learning rate for this parameter.
        /// </summary>
        [Description("Specifies the multiplier used on the global learning rate for this parameter.")]
        public double lr_mult
        {
            get { return m_dfLrMult; }
            set { m_dfLrMult = value; }
        }

        /// <summary>
        /// Specifies the multiplier used on the global weight decay for this parameter.
        /// </summary>
        [Description("Specifies the multiplier used on the global weight decay for this parameter.")]
        public double decay_mult
        {
            get { return m_dfDecayMult; }
            set { m_dfDecayMult = value; }
        }

        /// <summary>
        /// Creates a new copy of the ParamSpec.
        /// </summary>
        /// <returns>A new instance of the ParamSpec is returned.</returns>
        public object Clone()
        {
            ParamSpec p = new ParamSpec();

            p.m_strName = m_strName;
            p.m_shareMode = m_shareMode;
            p.m_dfDecayMult = m_dfDecayMult;
            p.m_dfLrMult = m_dfLrMult;

            return p;
        }

        /// <summary>
        /// Converts the ParamSpec into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name given to the RawProto.</param>
        /// <returns>The new RawProto representing the ParamSpec is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (name.Length > 0)
                rgChildren.Add("name", "\"" + name + "\"");

            if (share_mode != DimCheckMode.STRICT)
                rgChildren.Add("share_mode", share_mode.ToString());

            rgChildren.Add("lr_mult", lr_mult.ToString());

            if (decay_mult != 1)
                rgChildren.Add("decay_mult", decay_mult.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses a new ParamSpec from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto representing the ParamSpec.</param>
        /// <returns>The new ParamSpec instance is returned.</returns>
        public static ParamSpec FromProto(RawProto rp)
        {
            string strVal;
            ParamSpec p = new ParamSpec();

            if ((strVal = rp.FindValue("name")) != null)
                p.name = strVal;

            if ((strVal = rp.FindValue("share_mode")) != null)
            {
                switch (strVal)
                {
                    case "STRICT":
                        p.share_mode = DimCheckMode.STRICT;
                        break;

                    case "PERMISSIVE":
                        p.share_mode = DimCheckMode.PERMISSIVE;
                        break;

                    default:
                        throw new Exception("Unknown 'share_mode' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("lr_mult")) != null)
                p.lr_mult = double.Parse(strVal);

            if ((strVal = rp.FindValue("decay_mult")) != null)
                p.decay_mult = double.Parse(strVal);

            return p;
        }
    }
}
