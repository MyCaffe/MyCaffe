using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the NetState which includes the phase, level and stage
    /// for which a given Net is to run under.
    /// </summary>
    public class NetState : BaseParameter, ICloneable, IComparable 
    {
        Phase m_phase = Phase.TEST;
        int m_nLevel = 0;
        List<string> m_rgStage = new List<string>();

        /// <summary>
        /// The NetState constructor.
        /// </summary>
        public NetState()
        {
        }

        /// <summary>
        /// Saves the NetState to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write((int)m_phase);
            bw.Write(m_nLevel);
            Utility.Save<string>(bw, m_rgStage);
        }

        /// <summary>
        /// Loads a new NetState instance from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader to use.</param>
        /// <returns>The new NetState instance is returned.</returns>
        public static NetState Load(BinaryReader br)
        {
            NetState ns = new NetState();

            ns.m_phase = (Phase)br.ReadInt32();
            ns.m_nLevel = br.ReadInt32();
            ns.m_rgStage = Utility.Load<string>(br);

            return ns;
        }

        /// <summary>
        /// Specifies the Phase of the NetState.
        /// </summary>
        public Phase phase
        {
            get { return m_phase; }
            set { m_phase = value; }
        }

        /// <summary>
        /// Specifies the level of the NetState.
        /// </summary>
        public int level
        {
            get { return m_nLevel; }
            set { m_nLevel = value; }
        }

        /// <summary>
        /// Specifies the stages of the NetState.
        /// </summary>
        public List<string> stage
        {
            get { return m_rgStage; }
            set
            {
                if (value == null)
                    m_rgStage = new List<string>();
                else
                    m_rgStage = value;
            }
        }

        /// <summary>
        /// Merges another NetState with this instance.
        /// </summary>
        /// <param name="ns"></param>
        public void MergeFrom(NetState ns)
        {
            if (ns == null)
                return;

            if (m_phase == Phase.NONE)
                m_phase = ns.phase;

            m_nLevel = ns.level;

            foreach (string strStage in ns.m_rgStage)
            {
                if (!m_rgStage.Contains(strStage))
                    m_rgStage.Add(strStage);
            }
        }

        /// <summary>
        /// Creates a new copy of this NetState instance.
        /// </summary>
        /// <returns>The new NetState instance is returned.</returns>
        public NetState Clone()
        {
            NetState ns = new NetState();

            ns.m_phase = m_phase;
            ns.m_nLevel = m_nLevel;
            ns.m_rgStage = Utility.Clone<string>(m_rgStage);

            return ns;
        }

        /// <summary>
        /// Creates a new copy of this NetState instance.
        /// </summary>
        /// <returns>The new NetState instance is returned.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        /// Compares this NetState to another one.
        /// </summary>
        /// <param name="obj">Specifies the other NetState to compare with this one.</param>
        /// <returns>If the NetStates are the same 0 is returned, otherwise 1 is returned.</returns>
        public int CompareTo(object obj)
        {
            NetState ns = obj as NetState;

            if (obj == null)
                return 1;

            if (!Compare(ns))
                return 1;

            return 0;
        }

        /// <summary>
        /// Converts this NetState to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name given to the RawProto.</param>
        /// <returns>The new RawProto representing the NetState is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("phase", phase.ToString());
            rgChildren.Add("level", level.ToString());
            rgChildren.Add<string>("stage", stage);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses a RawProto representing a NetState into a NetState instance.
        /// </summary>
        /// <param name="rp">Specifies the RawProto representing the NetState.</param>
        /// <returns>The new instance of the NetState is returned.</returns>
        public static NetState FromProto(RawProto rp)
        {
            string strVal;
            NetState p = new NetState();

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

                    case "NONE":
                        p.phase = Phase.NONE;
                        break;

                    default:
                        throw new Exception("Unknown 'phase' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("level")) != null)
                p.level = int.Parse(strVal);

            p.stage = rp.FindArray<string>("stage");

            return p;
        }
    }
}
