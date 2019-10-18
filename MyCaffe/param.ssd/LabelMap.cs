using MyCaffe.basecode;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the LabelMap used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class LabelMap
    {
        List<LabelMapItem> m_rgItems = new List<LabelMapItem>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public LabelMap()
        {
        }

        /// <summary>
        /// Find a label with its label id.
        /// </summary>
        /// <param name="nLabel">Specifies the label id.</param>
        /// <returns>If a label item with a matching id is found, it is returned, otherwise null is returned.</returns>
        public LabelMapItem FindByLabel(int nLabel)
        {
            foreach (LabelMapItem li in m_rgItems)
            {
                if (li.label == nLabel)
                    return li;
            }

            return null;
        }

        /// <summary>
        /// Find a label with a given name.
        /// </summary>
        /// <param name="strName">Specifies the label name.</param>
        /// <returns>If a label item with a matching name is found, it is returned, otherwise null is returned.</returns>
        public LabelMapItem FindByName(string strName)
        {
            foreach (LabelMapItem li in m_rgItems)
            {
                if (li.name == strName)
                    return li;
            }

            return null;
        }

        /// <summary>
        /// Map the labels into a dictionary.
        /// </summary>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="bStrict">Specifies whether or not to allow duplicates, when allowed, the duplicate overwrites previous labels with the same ID.</param>
        /// <param name="bDisplayName">Specifies whether or not to use the display name (true) or name (false)</param>
        /// <returns>The label to name dictionary is returned.</returns>
        public Dictionary<int, string> MapToName(Log log, bool bStrict, bool bDisplayName)
        {
            Dictionary<int, string> rgLabelToName = new Dictionary<int, string>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                string strName = (bDisplayName) ? m_rgItems[i].display : m_rgItems[i].name;
                int nLabel = m_rgItems[i].label;

                if (bStrict)
                {
                    if (rgLabelToName.ContainsKey(nLabel))
                        log.FAIL("There are duplicates of the label: " + nLabel.ToString());

                    rgLabelToName.Add(nLabel, strName);
                }
                else
                {
                    if (rgLabelToName.ContainsKey(nLabel))
                        rgLabelToName[nLabel] = strName;
                    else
                        rgLabelToName.Add(nLabel, strName);
                }
            }

            return rgLabelToName;
        }

        /// <summary>
        /// Map the names to their labels.
        /// </summary>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="bStrict">Specifies whether or not to allow duplicates, when allowed, the duplicate overwrites previous labels with the same ID.</param>
        /// <returns>The name to label mapping is returned.</returns>
        public Dictionary<string, int> MapToLabel(Log log, bool bStrict)
        {
            Dictionary<string, int> rgNameToLabel = new Dictionary<string, int>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                string strName = m_rgItems[i].name;
                int nLabel = m_rgItems[i].label;

                if (bStrict)
                {
                    if (rgNameToLabel.ContainsKey(strName))
                        log.FAIL("There are duplicates of the name: " + strName.ToLower());

                    rgNameToLabel.Add(strName, nLabel);
                }
                else
                {
                    if (rgNameToLabel.ContainsKey(strName))
                        rgNameToLabel[strName] = nLabel;
                    else
                        rgNameToLabel.Add(strName, nLabel);
                }
            }

            return rgNameToLabel;
        }

        /// <summary>
        /// Specifies the list of label items.
        /// </summary>
        public List<LabelMapItem> item
        {
            get { return m_rgItems; }
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public void Copy(LabelMap src)
        {
            m_rgItems = new List<LabelMapItem>();

            foreach (LabelMapItem item in src.item)
            {
                m_rgItems.Add(item.Clone());
            }
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public LabelMap Clone()
        {
            LabelMap p = new LabelMap();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert this object to a raw proto.
        /// </summary>
        /// <param name="strName">Specifies the name of the proto.</param>
        /// <returns>The new proto is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            RawProtoCollection col = new RawProtoCollection();
            foreach (LabelMapItem item in m_rgItems)
            {
                col.Add(item.ToProto("item"));
            }

            rgChildren.Add(col);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LabelMap FromProto(RawProto rp)
        {
            LabelMap p = new LabelMap();

            RawProtoCollection col = rp.FindChildren("item");
            foreach (RawProto child in col)
            {
                LabelMapItem item = LabelMapItem.FromProto(child);
                p.item.Add(item);
            }

            return p;
        }
    }

    /// <summary>
    /// The LabelMapItem class stores the information for a single label.
    /// </summary>
    public class LabelMapItem
    {
        string m_strName;
        string m_strDisplay;
        int m_nLabel;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nLabel">Optionally, specify the label id (default = 0).</param>
        /// <param name="strName">Optionally, specify the label name (default = null).</param>
        /// <param name="strDisplay">Optionally, specify the label display name (default = null).</param>
        public LabelMapItem(int nLabel = 0, string strName = null, string strDisplay = null)
        {
            m_strName = strName;
            m_nLabel = nLabel;
            m_strDisplay = strDisplay;
        }

        /// <summary>
        /// Get/set the label name.
        /// </summary>
        public string name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        /// <summary>
        /// Optionally, get/set the display name for the label.
        /// </summary>
        public string display
        {
            get { return m_strDisplay; }
            set { m_strDisplay = value; }
        }

        /// <summary>
        /// Get/set the label id.
        /// </summary>
        public int label
        {
            get { return m_nLabel; }
            set { m_nLabel = value; }
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public void Copy(LabelMapItem src)
        {
            m_strName = src.m_strName;
            m_strDisplay = src.m_strDisplay;
            m_nLabel = src.m_nLabel;
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public LabelMapItem Clone()
        {
            return new LabelMapItem(m_nLabel, m_strName, m_strDisplay);
        }

        /// <summary>
        /// Convert this object to a raw proto.
        /// </summary>
        /// <param name="strName">Specifies the name of the proto.</param>
        /// <returns>The new proto is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("name", m_strName));
            rgChildren.Add(new RawProto("label", m_nLabel.ToString()));

            if (!string.IsNullOrEmpty(m_strDisplay))
                rgChildren.Add(new RawProto("display", m_strDisplay));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LabelMapItem FromProto(RawProto rp)
        {
            LabelMapItem item = new LabelMapItem();
            string strVal;

            if ((strVal = rp.FindValue("name")) != null)
                item.name = strVal;

            if ((strVal = rp.FindValue("label")) != null)
                item.label = int.Parse(strVal);

            if ((strVal = rp.FindValue("display")) != null)
                item.display = strVal;

            return item;
        }
    }
}
