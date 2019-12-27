using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The LabelMappingCollection manages a collection of LabelMapping's.
    /// </summary>
    [Serializable]
    public class LabelMappingCollection : IEnumerable<LabelMapping>
    {
        List<LabelMapping> m_rgMappings = new List<LabelMapping>();

        /// <summary>
        /// The LabelMappingCollection constructor.
        /// </summary>
        public LabelMappingCollection()
        {
        }

        /// <summary>
        /// Returns the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgMappings.Count; }
        }

        /// <summary>
        /// Returns the label mapping list.
        /// </summary>
        public List<LabelMapping> Mappings
        {
            get { return m_rgMappings; }
            set { m_rgMappings = value.OrderBy(p => p.OriginalLabel).ToList(); }
        }

        /// <summary>
        /// Get/set an item at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The item at the index is returned.</returns>
        public LabelMapping this[int nIdx]
        {
            get { return m_rgMappings[nIdx]; }
            set { m_rgMappings[nIdx] = value; }
        }

        /// <summary>
        /// Adds a new label mapping.
        /// </summary>
        /// <param name="map">Specifies the label mapping.</param>
        public void Add(LabelMapping map)
        {
            foreach (LabelMapping m in m_rgMappings)
            {
                if (m.OriginalLabel == map.OriginalLabel && m.ConditionBoostEquals == map.ConditionBoostEquals)
                    throw new Exception("You already have a mapping for the original label '" + map.OriginalLabel.ToString() + "' with the conditional boost = '" + map.ConditionBoostEquals.ToString() + "'.");
            }

            m_rgMappings.Add(map);
            m_rgMappings = m_rgMappings.OrderBy(p => p.OriginalLabel).ToList();
        }

        /// <summary>
        /// Removes a label mapping.
        /// </summary>
        /// <param name="map">Specifies the label mapping.</param>
        /// <returns>If found and removed, returns <i>true</i>, otherwise returns <i>false</i>.</returns>
        public bool Remove(LabelMapping map)
        {
            return m_rgMappings.Remove(map);
        }

        /// <summary>
        /// Returns a copy of the label mapping collection.
        /// </summary>
        /// <returns></returns>
        public LabelMappingCollection Clone()
        {
            LabelMappingCollection col = new LabelMappingCollection();

            foreach (LabelMapping m in m_rgMappings)
            {
                col.Add(m.Clone());
            }

            return col;
        }

        /// <summary>
        /// Returns the mapped label associated with a given label and boost (if a boost condition is used).
        /// </summary>
        /// <param name="nLabel">Specifies the label to map.</param>
        /// <param name="nBoost">Specifies the boost condition that must be met if specified.</param>
        /// <returns>The mapped label is returned.</returns>
        public int MapLabel(int nLabel, int nBoost)
        {
            foreach (LabelMapping m in m_rgMappings)
            {
                if (m.OriginalLabel == nLabel && (!m.ConditionBoostEquals.HasValue || m.ConditionBoostEquals == nBoost))
                    return m.NewLabel;
            }

            return nLabel;
        }

        /// <summary>
        /// Returns the mapped label associated with a given label.
        /// </summary>
        /// <param name="nLabel">Specifies the label to map.</param>
        /// <returns>The mapped label is returned.</returns>
        public int MapLabelWithoutBoost(int nLabel)
        {
            foreach (LabelMapping m in m_rgMappings)
            {
                if (m.OriginalLabel == nLabel )
                    return m.NewLabel;
            }

            return nLabel;
        }

        /// <summary>
        /// Returns the enumerator of the collection.
        /// </summary>
        /// <returns>The collection enumerator is returned.</returns>
        public IEnumerator<LabelMapping> GetEnumerator()
        {
            return m_rgMappings.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator of the collection.
        /// </summary>
        /// <returns>The collection enumerator is returned.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgMappings.GetEnumerator();
        }

        /// <summary>
        /// Sorts the label mappings.
        /// </summary>
        public void Sort()
        {
            m_rgMappings.Sort(new Comparison<LabelMapping>(compare));
        }

        private int compare(LabelMapping a, LabelMapping b)
        {
            if (a.OriginalLabel < b.OriginalLabel)
                return -1;

            if (a.OriginalLabel > b.OriginalLabel)
                return 1;

            if (a.NewLabel < b.NewLabel)
                return -1;

            if (a.NewLabel > b.NewLabel)
                return 1;

            return 0;
        }

        /// <summary>
        /// Compares one label mapping collection to another.
        /// </summary>
        /// <param name="col">Specifies the other collection to compare.</param>
        /// <returns>If the two collections are the same, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(LabelMappingCollection col)
        {
            Sort();
            col.Sort();

            string strA = ToString();
            string strB = col.ToString();

            if (strA == strB)
                return true;

            return false;
        }

        /// <summary>
        /// Returns a string representation of the label mapping collection.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            List<string> rgstr = ToStringList();
            string str = "";

            foreach (string strMapping in rgstr)
            {
                str += strMapping;
                str += ";";
            }

            return str.TrimEnd(';');
        }

        /// <summary>
        /// Returns a list of strings where each entry represents a mapping.
        /// </summary>
        /// <returns>Returns the label mappings as a list of strings.</returns>
        public List<string> ToStringList()
        {
            List<string> rgstrMappings = new List<string>();

            foreach (LabelMapping map in m_rgMappings)
            {
                rgstrMappings.Add(map.ToString());
            }

            return rgstrMappings;
        }

        /// <summary>
        /// Parses a label mapping string into a collection of label mappings.
        /// </summary>
        /// <param name="strMappings">Specifies the list of ';' separated label mappings.</param>
        /// <returns>The new LabelMappingCollection is returned.</returns>
        public static LabelMappingCollection Parse(string strMappings)
        {
            string[] rgstrMappings = strMappings.Split(';');
            return Parse(new List<string>(rgstrMappings));
        }

        /// <summary>
        /// Parses a list of strings where each string is a label mapping.
        /// </summary>
        /// <param name="rgstrMappings">Specifies the list of label mapping strings.</param>
        /// <returns>The new LabelMappingCollection is returned.</returns>
        public static LabelMappingCollection Parse(List<string> rgstrMappings)
        {
            LabelMappingCollection col = new LabelMappingCollection();
            List<string> rgstrSeparators = new List<string>() { "->" };

            foreach (string strMapping in rgstrMappings)
            {
                string[] rgstr = strMapping.Split(rgstrSeparators.ToArray(), StringSplitOptions.RemoveEmptyEntries);

                if (rgstr.Length != 2)
                    throw new Exception("Each label mapping should have the format 'original->new_label'.");

                if (rgstr[0].Length == 0 || rgstr[1].Length == 0)
                    throw new Exception("Each label mapping should have the format 'original->new_label'.");

                string str1 = rgstr[0].Trim('\"');
                string str2 = rgstr[1].Trim('\"');
                int? nConditionBoostEquals = null;
                int? nConditionFalse = null;

                rgstr = str2.Split('?');
                if (rgstr.Length > 1)
                {
                    str2 = rgstr[0].Trim();

                    string strCondition = "";
                    string strRightSide = rgstr[1].Trim();
                    rgstr = strRightSide.Split(',');

                    if (rgstr.Length == 2)
                    {
                        string str3 = rgstr[0].Trim();
                        if (str3.Length > 0)
                            nConditionFalse = int.Parse(str3);

                        strCondition = rgstr[1].Trim();
                    }
                    else if (rgstr.Length == 1)
                    {
                        strCondition = rgstr[0].Trim();
                    }
                    else
                    {
                        throw new Exception("Invalid mapping format! Expected format <true_int>?<false_int>,boost=<int>");
                    }

                    rgstr = strCondition.Split('=');

                    if (rgstr.Length != 2 || rgstr[0].Trim().ToLower() != "boost")
                        throw new Exception("Invalid boost condition!  Expected format = ?boost=<int>");

                    nConditionBoostEquals = int.Parse(rgstr[1].Trim());
                }

                col.Add(new LabelMapping(int.Parse(str1), int.Parse(str2), nConditionBoostEquals, nConditionFalse));
            }

            return col;
        }
    }

    /// <summary>
    /// The LabelMapping class represents a single label mapping.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class LabelMapping
    {
        int m_nOrignalLabel = 0;
        int m_nNewLabelConditionTrue = 0;
        int? m_nNewLabelConditionFalse = null;
        int? m_nConditionBoostEquals = null;


        /// <summary>
        /// The LabelMapping constructor.
        /// </summary>
        /// <param name="nOriginalLabel">Specifies the original label.</param>
        /// <param name="nNewLabel">Specifies the new label.</param>
        /// <param name="nConditionBoostEquals">Specifies a conditional boost value.</param>
        /// <param name="nNewLabelConditionFalse">Specifies the label to use if the conditional fails.</param>
        public LabelMapping(int nOriginalLabel, int nNewLabel, int? nConditionBoostEquals, int? nNewLabelConditionFalse)
        {
            m_nOrignalLabel = nOriginalLabel;
            m_nNewLabelConditionTrue = nNewLabel;
            m_nNewLabelConditionFalse = nNewLabelConditionFalse;
            m_nConditionBoostEquals = nConditionBoostEquals;
        }

        /// <summary>
        /// The LabelMapping constructor.
        /// </summary>
        public LabelMapping()
        {
        }

        /// <summary>
        /// Get/set the original label.
        /// </summary>
        [Description("Specifies the original label.")]
        public int OriginalLabel
        {
            get { return m_nOrignalLabel; }
            set { m_nOrignalLabel = value; }
        }

        /// <summary>
        /// Get/set the new label.
        /// </summary>
        [Description("Specifies the new label replacement.")]
        public int NewLabel
        {
            get { return m_nNewLabelConditionTrue; }
            set { m_nNewLabelConditionTrue = value; }
        }

        /// <summary>
        /// Get/set the label to use if the boost condition fails.
        /// </summary>
        [Description("Specifies the label to use if the boost condition fails.")]
        public int? NewLabelConditionFalse
        {
            get { return m_nNewLabelConditionFalse; }
            set { m_nNewLabelConditionFalse = value; }
        }

        /// <summary>
        /// Get/set the boost condition to test which if met, the new label is set, otherwise it is not.
        /// </summary>
        [Description("Specifies the boost condition to test.")]
        public int? ConditionBoostEquals
        {
            get { return m_nConditionBoostEquals; }
            set { m_nConditionBoostEquals = value; }
        }

        /// <summary>
        /// Return a copy of the LabelMapping.
        /// </summary>
        /// <returns>The copy is returned.</returns>
        public LabelMapping Clone()
        {
            return new LabelMapping(m_nOrignalLabel, m_nNewLabelConditionTrue, m_nConditionBoostEquals, m_nNewLabelConditionFalse);
        }

        /// <summary>
        /// Return a string representation of the label mapping.
        /// </summary>
        /// <returns>The string representatio is returned.</returns>
        public override string ToString()
        {
            string strMapping = m_nOrignalLabel.ToString() + "->" + m_nNewLabelConditionTrue.ToString();

            if (m_nConditionBoostEquals.HasValue)
            {
                strMapping += "?";

                if (m_nNewLabelConditionFalse.HasValue)
                {
                    strMapping += m_nNewLabelConditionFalse.Value.ToString();
                    strMapping += ",";
                }

                strMapping += "boost=";
                strMapping += m_nConditionBoostEquals.Value.ToString();
            }

            return strMapping;
        }
    }
}
