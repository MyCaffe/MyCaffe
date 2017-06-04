using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the DebugLayer
    /// </summary>
    public class DebugParameter : LayerParameterBase 
    {
        int m_nMaxBatchesToStore = 1000;

        /** @copydoc LayerParameterBase */
        public DebugParameter()
        {
        }

        /// <summary>
        /// Specifies the maximum number of batches to store and search for neighbors.  Each batch input is stored until the maximum count is reached at which time, the oldest batch is released.  A larger max value = more GPU memory used.
        /// </summary>
        [Description("Specifies the maximum number of batches to store and search for neighbors.  Each batch input is stored until the maximum count is reached at which time, the oldest batch is released.  A larger max value = more GPU memory used.")]
        public int max_stored_batches
        {
            get { return m_nMaxBatchesToStore; }
            set { m_nMaxBatchesToStore = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DebugParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            DebugParameter p = (DebugParameter)src;
            m_nMaxBatchesToStore = p.m_nMaxBatchesToStore;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            DebugParameter p = new DebugParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("max_stored_batches", max_stored_batches.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DebugParameter FromProto(RawProto rp)
        {
            string strVal;
            DebugParameter p = new DebugParameter();

            if ((strVal = rp.FindValue("max_stored_batches")) != null)
                p.max_stored_batches = int.Parse(strVal);

            return p;
        }
    }
}
