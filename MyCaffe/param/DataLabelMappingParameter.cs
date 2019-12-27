using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param
{
    /// <summary>
    /// The DataLabelMappingParameter is used by the DataParameter when the 'enable_labelmapping' = True.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DataLabelMappingParameter
    {
        LabelMappingCollection m_rgMapping = new LabelMappingCollection();

        /// <summary>
        /// The constructor.
        /// </summary>
        public DataLabelMappingParameter()
        {
        }

        /// <summary>
        /// Specifies the label mapping where the original label is mapped to the new label specified.
        /// </summary>
        [Description("Specifies the label mapping where the original label is mapped to the new label specified.")]
        public List<LabelMapping> mapping
        {
            get { return m_rgMapping.Mappings; }
            set { m_rgMapping.Mappings = value; }
        }

        /// <summary>
        /// Queries the mapped label for a given label.
        /// </summary>
        /// <param name="nLabel">Specifies the label to query the mapped label from.</param>
        /// <returns>The mapped label is returned.</returns>
        public int MapLabel(int nLabel)
        {
            return m_rgMapping.MapLabel(nLabel);
        }

        /// <summary>
        /// Copies the specified source data noise parameter to this one.
        /// </summary>
        /// <param name="src">Specifies the source data noise parameter.</param>
        public void Copy(DataLabelMappingParameter src)
        {
            if (src == null)
                return;

            m_rgMapping = src.m_rgMapping.Clone();
        }

        /// <summary>
        /// Convert the DataLabelMappingParameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the RawProto name.</param>
        /// <returns>The RawProto containing the settings is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add<string>("mapping", m_rgMapping.ToStringList());

            return new RawProto(strName, "", rgChildren);
        }


        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DataLabelMappingParameter FromProto(RawProto rp, DataLabelMappingParameter p = null)
        {
            if (p == null)
                p = new DataLabelMappingParameter();

            p.m_rgMapping = LabelMappingCollection.Parse(rp.FindArray<string>("mapping"));

            return p;
        }
    }
}
