using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.db.image;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// /b DEPRECIATED (use DataLayer DataLabelMappingParameter instead) Specifies the parameters for the LabelMappingLayer.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class LabelMappingParameter : LayerParameterBase 
    {
        LabelMappingCollection m_rgMapping = new LabelMappingCollection();
        bool m_bUpdateDatabase = false;
        bool m_bResetDatabaseLabels = false;
        string m_strLabelBoosts = "";

        /** @copydoc LayerParameterBase */
        public LabelMappingParameter()
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
        /// Specifies whether or not to directly update the database with the label mapping for the data source used - when 'false' only the 'in-memory' labels are updated.  WARNING: Updating the database sets the label mapping globally and will impact all other projects using this data source.
        /// </summary>
        [Description("Specifies whether or not to directly update the database with the label mapping for the data source used - when 'false' only the 'in-memory' labels are updated.  WARNING: Updating the database sets the label mapping globally and will impact all other projects using this data source.")]
        public bool update_database
        {
            get { return m_bUpdateDatabase; }
            set { m_bUpdateDatabase = value; }
        }

        /// <summary>
        /// Specifies whether or not to reset the database labels to the original label values for the data source used.  <b>WARNING:</b> This resets the labels globally to their original setting and will impact all other projects using this data source.
        /// </summary>
        [Description("Specifies whether or not to reset the database labels to the original label values for the data source used.  WARNING: This resets the labels globally to their original setting and will impact all other projects using this data source.")]
        public bool reset_database_labels
        {
            get { return m_bResetDatabaseLabels; }
            set { m_bResetDatabaseLabels = value; }
        }

        /// <summary>
        /// DEPRECIATED: Specifies the labels for which the label boost is to be set.  When set, all labels specified are given a boost such that images are selected with equal probability between all labels specified.
        /// </summary>
        [Description("DEPRECIATED: Specifies the labels for which the label boost is to be set.  When set, all labels specified are given a boost such that images are selected with equal probability between all labels specified.")]
        public string label_boosts
        {
            get { return m_strLabelBoosts; }
            set { m_strLabelBoosts = value; }
        }

        /// <summary>
        /// Queries the mapped label for a given label.
        /// </summary>
        /// <param name="nLabel">Specifies the label to query the mapped label from.</param>
        /// <returns>The mapped label is returned.</returns>
        public int MapLabel(int nLabel)
        {
            return m_rgMapping.MapLabelWithoutBoost(nLabel);
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LabelMappingParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            LabelMappingParameter p = (LabelMappingParameter)src;

            m_rgMapping = p.m_rgMapping.Clone();
            m_bUpdateDatabase = p.m_bUpdateDatabase;
            m_bResetDatabaseLabels = p.m_bResetDatabaseLabels;
            m_strLabelBoosts = p.m_strLabelBoosts;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LabelMappingParameter p = new LabelMappingParameter();
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

            rgChildren.Add<string>("mapping", m_rgMapping.ToStringList());

            if (m_bUpdateDatabase)
                rgChildren.Add(new RawProto("update_database", (m_bUpdateDatabase) ? "true" : "false"));

            if (m_bResetDatabaseLabels)
                rgChildren.Add(new RawProto("reset_database_labels", (m_bResetDatabaseLabels) ? "true" : "false"));

            if (m_strLabelBoosts != null && m_strLabelBoosts.Length > 0)
                rgChildren.Add(new RawProto("label_boosts", m_strLabelBoosts));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LabelMappingParameter FromProto(RawProto rp)
        {
            LabelMappingParameter p = new LabelMappingParameter();

            p.m_rgMapping = LabelMappingCollection.Parse(rp.FindArray<string>("mapping"));

            string strUpdateDb = rp.FindValue("update_database");
            if (strUpdateDb != null && strUpdateDb.ToLower() == "true")
                p.m_bUpdateDatabase = true;

            string strReset = rp.FindValue("reset_database_labels");
            if (strReset != null && strReset.ToLower() == "true")
                p.m_bResetDatabaseLabels = true;

            p.m_strLabelBoosts = rp.FindValue("label_boosts");
            if (p.m_strLabelBoosts == null)
                p.m_strLabelBoosts = "";

            return p;
        }
    }
}
