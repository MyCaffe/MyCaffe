using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the KnnLayer.
    /// </summary>
    public class KnnParameter : LayerParameterBase 
    {
        int m_nMaxBatchesToStore = 1000;
        int m_nOutput = 10;
        int m_nK = 100;

        /** @copydoc LayerParameterBase */
        public KnnParameter()
        {
        }

        /// <summary>
        /// Specifies the number of output items (e.g. classes)
        /// </summary>
        [Description("Specifies the number of output items (e.g. classes).")]
        public int num_output
        {
            get { return m_nOutput; }
            set { m_nOutput = value; }
        }

        /// <summary>
        /// Specifies the 'k' number of nearest neighbors to compare (per class).
        /// </summary>
        [Description("Specifies the number of nearest neighbors to compare per class, selected from items with the shortest distance.  The default = 100, which selects the class with the highest count from the 10 shortest distances.  It is recommended that K > ((2 * num_output) + 1) or convergence may not occur.")]
        public int k
        {
            get { return m_nK; }
            set { m_nK = value; }
        }

        /// <summary>
        /// Specifies the maximum number of batches to store before releasing batches.
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
            KnnParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            KnnParameter p = (KnnParameter)src;
            m_nOutput = p.m_nOutput;
            m_nK = p.m_nK;
            m_nMaxBatchesToStore = p.m_nMaxBatchesToStore;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            KnnParameter p = new KnnParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("k", k.ToString());
            rgChildren.Add("max_stored_batches", max_stored_batches.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static KnnParameter FromProto(RawProto rp)
        {
            string strVal;
            KnnParameter p = new KnnParameter();

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = int.Parse(strVal);

            if ((strVal = rp.FindValue("k")) != null)
                p.k = int.Parse(strVal);

            if ((strVal = rp.FindValue("max_stored_batches")) != null)
                p.max_stored_batches = int.Parse(strVal);

            return p;
        }
    }
}
