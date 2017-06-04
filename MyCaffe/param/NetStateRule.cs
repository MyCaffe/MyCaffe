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
    /// Specifies a NetStateRule used to determine whether a Net falls within a given <i>include</i> or <i>exclude</i> pattern.
    /// </summary>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class NetStateRule : BaseParameter, ICloneable, IComparable, IBinaryPersist
    {
        Phase m_phase = Phase.NONE;
        int? m_nMinLevel = null;
        int? m_nMaxLevel = null;
        List<string> m_rgStage = new List<string>();
        List<string> m_rgNotStage = new List<string>();

        /// <summary>
        /// Specifies the NetStateRule constructor.
        /// </summary>
        public NetStateRule()
        {
        }

        /// <summary>
        /// Specifies the NetStateRule constructor.
        /// </summary>
        /// <param name="p">Specifies the phase to assign to the NetStateRule</param>
        public NetStateRule(Phase p)
        {
            m_phase = p;
        }

        /// <summary>
        /// Saves the NetStateRule to a given binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write((int)m_phase);

            bw.Write(m_nMinLevel.HasValue);
            if (m_nMinLevel.HasValue)
                bw.Write(m_nMinLevel.Value);

            bw.Write(m_nMaxLevel.HasValue);
            if (m_nMaxLevel.HasValue)
                bw.Write(m_nMaxLevel.Value);

            Utility.Save<string>(bw, m_rgStage);
            Utility.Save<string>(bw, m_rgNotStage);
        }

        /// <summary>
        /// Loads a NetStateRule from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader to use.</param>
        /// <param name="bNewInstance">When <i>true</i>, a new NetStateRule instance is created and loaded, otherwise this instance is loaded.</param>
        /// <returns>The instance of the NetStateRule is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            NetStateRule p = this;

            if (bNewInstance)
                p = new NetStateRule();

            p.m_phase = (Phase)br.ReadInt32();

            if (br.ReadBoolean())
                p.m_nMinLevel = br.ReadInt32();

            if (br.ReadBoolean())
                p.m_nMaxLevel = br.ReadInt32();

            p.m_rgStage = Utility.Load<string>(br);
            p.m_rgNotStage = Utility.Load<string>(br);

            return p;
        }

        /// <summary>
        /// Set phase to require the NetState to have a particular phase (TRAIN or TEST)
        /// to meet this rule.
        /// </summary>
        /// <remarks>
        /// Note when the phase is set to NONE, the rule applies to all phases.
        /// </remarks>
        [Description("Specifies the phase required to meet this rule.")]
        public Phase phase
        {
            get { return m_phase; }
            set { m_phase = value; }
        }

        /// <summary>
        /// Set the minimum levels in which the layer should be used.
        /// Leave undefined to meet the rule regardless of level.
        /// </summary>
        [Description("Specifies the minimum level in which the layer should be used.")]
        public int? min_level
        {
            get { return m_nMinLevel; }
            set { m_nMinLevel = value; }
        }

        /// <summary>
        /// Set the maximum levels in which the layer should be used.
        /// Leave undefined to meet the rule regardless of level.
        /// </summary>
        [Description("Specifies the maximum level in which the layer should be used.")]
        public int? max_level
        {
            get { return m_nMaxLevel; }
            set { m_nMaxLevel = value; }
        }

        /// <summary>
        /// Customizable sets of stages to include.
        /// The net must have ALL of the specified stages and NONE of the specified
        /// 'not_stage's to meet the rule.
        /// (Use mutiple NetStateRules to specify conjunctions of stages.)
        /// </summary>
        [Description("Specifies the stage required to meet this rule.")]
        public List<string> stage
        {
            get { return m_rgStage; }
            set { m_rgStage = value; }
        }

        /// <summary>
        /// Customizable sets of stages to exclude.
        /// The net must have ALL of the specified stages and NONE of the specified
        /// 'not_stage's to meet the rule.
        /// (Use mutiple NetStateRules to specify conjunctions of stages.)
        /// </summary>
        [Description("Specifies the 'not_stage' that cannot be specified to meet this rule.")]
        public List<string> not_stage
        {
            get { return m_rgNotStage; }
            set { m_rgNotStage = value; }
        }

        /// <summary>
        /// Creates a new copy of a NetStateRule instance.
        /// </summary>
        /// <returns>The new instance is returned.</returns>
        public object Clone()
        {
            NetStateRule ns = new NetStateRule();

            ns.m_nMaxLevel = m_nMaxLevel;
            ns.m_nMinLevel = m_nMinLevel;
            ns.m_phase = m_phase;
            ns.m_rgNotStage = Utility.Clone<string>(m_rgNotStage);
            ns.m_rgStage = Utility.Clone<string>(m_rgStage);

            return ns;
        }

        /// <summary>
        /// Converts a NetStateRule into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name given to the RawProto.</param>
        /// <returns>The new RawProto representing the NetStateRule is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("phase", phase.ToString());

            if (min_level.HasValue)
                rgChildren.Add("min_level", min_level);

            if (max_level.HasValue)
                rgChildren.Add("max_level", max_level);

            if (stage.Count > 0)
                rgChildren.Add<string>("stage", stage);

            if (not_stage.Count > 0)
                rgChildren.Add<string>("not_stage", not_stage);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses a RawProto representing a NetStateRule and creates a new instance of a NetStateRule from it.
        /// </summary>
        /// <param name="rp">Specifies the RawProto used.</param>
        /// <returns>The new NeteStateRule instance is returned.</returns>
        public static NetStateRule FromProto(RawProto rp)
        {
            string strVal;
            NetStateRule p = new NetStateRule();

            if ((strVal = rp.FindValue("phase")) != null)
            {
                switch (strVal)
                {
                    case "TEST":
                        p.phase = Phase.TEST;
                        break;

                    case "TRAIN":
                        p.phase = Phase.TRAIN;
                        break;

                    case "RUN":
                        p.phase = Phase.RUN;
                        break;

                    case "RUN_NODB":
                        p.phase = Phase.RUN_NODB;
                        break;

                    case "NONE":
                        p.phase = Phase.NONE;
                        break;

                    case "ALL":
                        p.phase = Phase.ALL;
                        break;

                    default:
                        throw new Exception("Unknown 'phase' value: " + strVal);
                }
            }

            p.min_level = (int?)rp.FindValue("min_level", typeof(int));
            p.max_level = (int?)rp.FindValue("max_level", typeof(int));
            p.stage = rp.FindArray<string>("stage");
            p.not_stage = rp.FindArray<string>("not_stage");

            return p;
        }

        /// <summary>
        /// Compares this NetStateRule to another one.
        /// </summary>
        /// <param name="obj">Specifies the other NetStateRule to compare.</param>
        /// <returns>0 is returned if the NetStateRule instances match, otherwise 1 is returned.</returns>
        public int CompareTo(object obj)
        {
            NetStateRule nsr = obj as NetStateRule;

            if (nsr == null)
                return 1;

            if (!Compare(nsr))
                return 1;

            return 0;
        }
    }
}
