using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the DataLabelMappingParameter used to map labels by the DataTransformer.TransformLabel when active.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DataLabelMappingParameter : OptionalParameter
    {
        LabelMappingCollection m_rgMapping = new LabelMappingCollection();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public DataLabelMappingParameter(bool bActive) : base(bActive)
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
        /// <param name="nBoost">Specifies the boost condition that must be met if specified.</param>
        /// <returns>The mapped label is returned.</returns>
        public int MapLabel(int nLabel, int nBoost)
        {
            return m_rgMapping.MapLabel(nLabel, nBoost);
        }

        /// <summary>
        /// Load the and return a new DataLabelMappingParameter. 
        /// </summary>
        /// <param name="br"></param>
        /// <param name="bNewInstance"></param>
        /// <returns>The new object is returned.</returns>
        public DataLabelMappingParameter Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DataLabelMappingParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copies the specified source data label mapping parameter to this one.
        /// </summary>
        /// <param name="src">Specifies the source data label mapping parameter.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is DataLabelMappingParameter)
            {
                DataLabelMappingParameter p = (DataLabelMappingParameter)src;
                m_rgMapping = p.m_rgMapping.Clone();
            }
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public DataLabelMappingParameter Clone()
        {
            DataLabelMappingParameter p = new DataLabelMappingParameter(Active);
            p.Copy(this);
            return p;
        }


        /// <summary>
        /// Convert the DataLabelMappingParameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the RawProto name.</param>
        /// <returns>The RawProto containing the settings is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("option");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase);
            rgChildren.Add<string>("mapping", m_rgMapping.ToStringList());

            return new RawProto(strName, "", rgChildren);
        }


        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new DataLabelMappingParameter FromProto(RawProto rp)
        {
            DataLabelMappingParameter p = new DataLabelMappingParameter(true);

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            p.m_rgMapping = LabelMappingCollection.Parse(rp.FindArray<string>("mapping"));

            return p;
        }
    }
}
