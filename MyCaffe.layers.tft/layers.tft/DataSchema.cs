using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The DataSchema class is used by the DataTemporalLayer to load the data schema describing the NPY files containing the dataset.
    /// </summary>
    public class DataSchema
    {
        Data m_data = new Data();
        LookupCollection m_lookups = new LookupCollection();

        /// <summary>
        /// The constructor.
        /// </summary>
        public DataSchema()
        {
        }

        /// <summary>
        /// Loads the data schema from the XML schema file produced by the AiDesigner.TFT data creator.
        /// </summary>
        /// <param name="strFile">Specifies the XML schema file.</param>
        /// <returns>The DataSchema describing the NPY data is returned.</returns>
        public static DataSchema Load(string strFile)
        {
            DataSchema schema = new DataSchema();

            XmlDocument xmlDoc = new XmlDocument();
            xmlDoc.Load(strFile);

            XmlNode node = xmlDoc.SelectSingleNode("Schema");
            if (node == null)
                return null;

            XmlNode dataNode = node.SelectSingleNode("Data");

            XmlNode syncNode = dataNode.SelectSingleNode("Sync");
            schema.m_data.LoadSync(syncNode);

            XmlNode observedNode = dataNode.SelectSingleNode("Observed");
            XmlNode observedNumNode = observedNode.SelectSingleNode("Numeric");
            schema.m_data.LoadObservedNum(observedNumNode);
            XmlNode observedCatNode = observedNode.SelectSingleNode("Categorical");
            schema.m_data.LoadObservedCat(observedCatNode);

            XmlNode knownNode = dataNode.SelectSingleNode("//Known");
            XmlNode knownNumNode = knownNode.SelectSingleNode("Numeric");
            schema.m_data.LoadKnownNum(knownNumNode);
            XmlNode knownCatNode = knownNode.SelectSingleNode("Categorical");
            schema.m_data.LoadKnownCat(knownCatNode);

            XmlNode staticNode = dataNode.SelectSingleNode("Static");
            XmlNode staticNumNode = staticNode.SelectSingleNode("Numeric");
            schema.m_data.LoadStaticNum(staticNumNode);
            XmlNode staticCatNode = staticNode.SelectSingleNode("Categorical");
            schema.m_data.LoadStaticCat(staticCatNode);

            XmlNode lookupsNode = node.SelectSingleNode("Lookups");
            schema.m_lookups = LookupCollection.Load(lookupsNode);

            return schema;
        }

        /// <summary>
        /// Returns the data portion of the schema.
        /// </summary>
        public Data Data
        {
            get { return m_data; }
        }

        /// <summary>
        /// Returns the lookups portion of the schema.
        /// </summary>
        public LookupCollection Lookups
        {
            get { return m_lookups; }
        }
    }

    /// <summary>
    /// The Data class describes the data portions of the schema, including the known, observed and static data.
    /// </summary>
    public class Data
    {
        FieldCollection m_sync = new FieldCollection();
        FieldCollection m_numKnown = new FieldCollection();
        FieldCollection m_catKnown = new FieldCollection();
        FieldCollection m_numObserved = new FieldCollection();
        FieldCollection m_catObserved = new FieldCollection();
        FieldCollection m_numStatic = new FieldCollection();
        FieldCollection m_catStatic = new FieldCollection();

        /// <summary>
        /// The constructor.
        /// </summary>
        public Data()
        {
        }

        /// <summary>
        /// Loads the Sync fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Sync fields</param>
        public void LoadSync(XmlNode node)
        {
            m_sync = FieldCollection.Load(node);
        }

        /// <summary>
        /// Loads the Known numeric fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Known numeric fields</param>
        public void LoadKnownNum(XmlNode node)
        {
            m_numKnown = FieldCollection.Load(node);
        }

        /// <summary>
        /// Loads the Known categorical fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Known categorical fields</param>
        public void LoadKnownCat(XmlNode node)
        {
            m_catKnown = FieldCollection.Load(node);
        }

        /// <summary>
        /// Loads the Observed numeric fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Observed numeric fields</param>
        public void LoadObservedNum(XmlNode node)
        {
            m_numObserved = FieldCollection.Load(node);
        }

        /// <summary>
        /// Loads the Observed categorical fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Observed categorical fields</param>
        public void LoadObservedCat(XmlNode node)
        {
            m_catObserved = FieldCollection.Load(node);
        }

        /// <summary>
        /// Loads the Static numeric fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Static numeric fields</param>
        public void LoadStaticNum(XmlNode node)
        {
            m_numStatic = FieldCollection.Load(node);
        }

        /// <summary>
        /// Loads the Static categorical fields.
        /// </summary>
        /// <param name="node">Specifies the XML portion of the schema containing the Static categorical fields</param>
        public void LoadStaticCat(XmlNode node)
        {
            m_catStatic = FieldCollection.Load(node);
        }

        /// <summary>
        /// Returns the field collection for synchronization that includes time and category id fields.
        /// </summary>
        public FieldCollection Sync
        {
            get { return m_sync; }
        }

        /// <summary>
        /// Returns the Known numeric fields.
        /// </summary>
        public FieldCollection KnownNum
        {
            get { return m_numKnown; }
        }

        /// <summary>
        /// Returns the Known categorical fields.
        /// </summary>
        public FieldCollection KnownCat
        {
            get { return m_catKnown; }
        }

        /// <summary>
        /// Returns the Observed numeric fields.
        /// </summary>
        public FieldCollection ObservedNum
        {
            get { return m_numObserved; }
        }

        /// <summary>
        /// Returns the Observed categorical fields.
        /// </summary>
        public FieldCollection ObservedCat
        {
            get { return m_catObserved; }
        }

        /// <summary>
        /// Returns the Static numeric fields.
        /// </summary>
        public FieldCollection StaticNum
        {
            get { return m_numStatic; }
        }

        /// <summary>
        /// Returns the Static categorical fields.
        /// </summary>
        public FieldCollection StaticCat
        {
            get { return m_catStatic; }
        }
    }

    /// <summary>
    /// The LookupCollection class contains a collection of Lookup objects.
    /// </summary>
    public class LookupCollection
    {
        Dictionary<string, Lookup> m_rgLookups = new Dictionary<string, Lookup>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public LookupCollection()
        {
        }

        /// <summary>
        /// Loads the LookupCollection from a node.
        /// </summary>
        /// <param name="node">Specifies the XML node containing the Lookups.</param>
        /// <returns>The LookupCollection is returned.</returns>
        public static LookupCollection Load(XmlNode node)
        {
            LookupCollection lookups = new LookupCollection();

            XmlNodeList lookupNodes = node.SelectNodes("Lookup");
            foreach (XmlNode lookupNode in lookupNodes)
            {
                Lookup lookup = Lookup.Load(lookupNode);
                lookups.Add(lookup);
            }

            return lookups;
        }

        /// <summary>
        /// Adds a new lookup to the collection.
        /// </summary>
        /// <param name="lookup">Specifies the lookup to add.</param>
        public void Add(Lookup lookup)
        {
            m_rgLookups.Add(lookup.Name, lookup);
        }

        /// <summary>
        /// Locates a lookup by name.
        /// </summary>
        /// <param name="strName">Specifies the lookup to find.</param>
        /// <returns>If found the lookup is returned, otherwise null is returned.</returns>
        public Lookup Find(string strName)
        {
            if (!m_rgLookups.ContainsKey(strName))
                return null;

            return m_rgLookups[strName];
        }

        /// <summary>
        /// Get a specific lookup by index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the lookup.</param>
        /// <returns>The lookup is returned.</returns>
        public Lookup this[int nIdx]
        {
            get { return m_rgLookups.ElementAt(nIdx).Value; }
        }

        /// <summary>
        /// Specifies the number of lookups in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgLookups.Count; }
        }
    }

    /// <summary>
    /// The Lookup class is used to manage a single lookup table.
    /// </summary>
    public class Lookup
    {
        string m_strName;
        Dictionary<string, LookupItem> m_rgLookupNameToId = new Dictionary<string, LookupItem>();
        Dictionary<int, LookupItem> m_rgLookupIdToName = new Dictionary<int, LookupItem>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public Lookup()
        {
        }

        /// <summary>
        /// Load a lookup table from an XML node.
        /// </summary>
        /// <param name="node">Specifies the XML node containing the Lookup/</param>
        /// <returns>The Lookup object is returned.</returns>
        public static Lookup Load(XmlNode node)
        {
            Lookup lookup = new Lookup();

            lookup.m_strName = node.Attributes["Name"].Value;

            XmlNodeList itemNodes = node.SelectNodes("Item");
            foreach (XmlNode itemNode in itemNodes)
            {
                lookup.Add(LookupItem.Load(itemNode));
            }

            return lookup;
        }

        /// <summary>
        /// Add a new item to the lookup table.
        /// </summary>
        /// <param name="item">Specifies the lookup item to add.</param>
        public void Add(LookupItem item)
        {
            m_rgLookupIdToName.Add(item.ID, item);
            m_rgLookupNameToId.Add(item.Name, item);
        }

        /// <summary>
        /// Specifies the lookup item name.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Find a given lookup ID by name.
        /// </summary>
        /// <param name="strName">Specifies the lookup name.</param>
        /// <returns>Returns the lookup ID or -1 if not found.</returns>
        public int FindID(string strName)
        {
            if (!m_rgLookupNameToId.ContainsKey(strName))
                return -1;

            return m_rgLookupNameToId[strName].ID;
        }

        /// <summary>
        /// Find a given lookup name by ID.
        /// </summary>
        /// <param name="nID">Specifies the lookup ID.</param>
        /// <returns>Returns the lookup name or null if not found.</returns>
        public string FindName(int nID)
        {
            if (!m_rgLookupIdToName.ContainsKey(nID))
                return null;

            return m_rgLookupIdToName[nID].Name;
        }

        /// <summary>
        /// Returns the number of lookup items in the table.
        /// </summary>
        public int Count
        {
            get { return m_rgLookupIdToName.Count; }
        }

        /// <summary>
        /// Returns the lookup item at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index to retrieve.</param>
        /// <returns>The lookup item is returned.</returns>
        public LookupItem this[int nIdx]
        {
            get { return m_rgLookupIdToName.ElementAt(nIdx).Value; }
        }
    }

    /// <summary>
    /// The lookup item manages a single lookup item used to map a name to an ID.
    /// </summary>
    public class LookupItem
    {
        string m_strName;
        int m_nIndex;
        int m_nValidRangeStartIndex = -1;
        int m_nValidRangeEndIndex = -1;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strName">Specifies the lookup name.</param>
        /// <param name="nIndex">Specifies the lookup ID.</param>
        public LookupItem(string strName, int nIndex)
        {
            m_strName = strName;
            m_nIndex = nIndex;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strName">Specifies the lookup name.</param>
        /// <param name="nIndex">Specifies the lookup ID.</param>
        /// <param name="nValidRangeStart">Specifies the valid data range start index (or -1 to ignore).</param>
        /// <param name="nValidRangeEnd">Specifies the valid data range end index (or -1 to ignore).</param>
        public LookupItem(string strName, int nIndex, int nValidRangeStart, int nValidRangeEnd)
        {
            m_strName = strName;
            m_nIndex = nIndex;
            m_nValidRangeStartIndex = nValidRangeStart;
            m_nValidRangeEndIndex = nValidRangeEnd;
        }

        /// <summary>
        /// Load a LookupItem from an XML node.
        /// </summary>
        /// <param name="node">Specifies the XML node containing the lookup item.</param>
        /// <returns>The new lookup item is returned.</returns>
        public static LookupItem Load(XmlNode node)
        {
            string strName = node.FirstChild.Value;
            int nIndex = int.Parse(node.Attributes["Index"].Value);
            int nValidRangeStart = -1;
            int nValidRangeEnd = -1;

            if (node.Attributes["ValidRangeStartIdx"] != null)
                nValidRangeStart = int.Parse(node.Attributes["ValidRangeStartIdx"].Value);

            if (node.Attributes["ValidRangeEndIdx"] != null)
                nValidRangeEnd = int.Parse(node.Attributes["ValidRangeEndIdx"].Value);

            return new LookupItem(strName, nIndex, nValidRangeStart, nValidRangeEnd);
        }

        /// <summary>
        /// Specifies the lookup name.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Specifies the lookup ID.
        /// </summary>
        public int ID
        {
            get { return m_nIndex; }
        }

        /// <summary>
        /// Specifies the valid data range start index or -1 to ignore.
        /// </summary>
        public int ValidRangeStartIndex
        {
            get { return m_nValidRangeStartIndex; }
        }

        /// <summary>
        /// Specifies the valid data range end index or -1 to ignore.
        /// </summary>
        public int ValidRangeEndIndex
        {
            get { return m_nValidRangeEndIndex; }
        }

        /// <summary>
        /// Returns a string representation of the lookup item.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return m_strName + " @" + m_nIndex.ToString() + " (" + m_nValidRangeStartIndex.ToString() + "," + m_nValidRangeEndIndex.ToString() + ")";
        }
    }

    /// <summary>
    /// The FieldCollection manages a collection of fields.
    /// </summary>
    public class FieldCollection
    {
        string m_strFile;
        List<Field> m_rgFields = new List<Field>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public FieldCollection()
        {
        }

        /// <summary>
        /// Loads a collection of fields from an XML node.
        /// </summary>
        /// <param name="node">Specifies the XML node containing the fields.</param>
        /// <returns>The new field collection is returned.</returns>
        public static FieldCollection Load(XmlNode node)
        {
            FieldCollection col = new FieldCollection();

            XmlNode nodeFile = node.SelectSingleNode("File");
            if (nodeFile != null)
                col.m_strFile = nodeFile.FirstChild.Value;

            XmlNodeList nodes = node.SelectNodes("Field");
            foreach (XmlNode node1 in nodes)
            {
                Field field = Field.Load(node1);
                col.Add(field);
            }

            return col;
        }

        /// <summary>
        /// Locates the index of the field by its type.
        /// </summary>
        /// <param name="type">The field type to look for.</param>
        /// <returns>The index of the first field of the specified type is returned.</returns>
        public int FindFieldIndex(Field.INPUT_TYPE type)
        {
            for (int i = 0; i < m_rgFields.Count; i++)
            {
                if ((m_rgFields[i].InputType & type) == type)
                    return i;
            }

            return -1;
        }

        /// <summary>
        /// Add a new field to the collection.
        /// </summary>
        /// <param name="field">Specifies the field to add.</param>
        public void Add(Field field)
        {
            m_rgFields.Add(field);
        }

        /// <summary>
        /// Get a field at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The field at the index is returned.</returns>
        public Field this[int nIdx]
        {
            get { return m_rgFields[nIdx]; }
        }

        /// <summary>
        /// Get a field by name.
        /// </summary>
        /// <param name="strName">Specifies the field name of the field to get.</param>
        /// <returns>The field is returned if found, othwerwise null is returned.</returns>
        public Field this[string strName]
        {
            get
            {
                foreach (Field field in m_rgFields)
                {
                    if (field.Name == strName)
                        return field;
                }

                return null;
            }
        }

        /// <summary>
        /// Returns the number of fields in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgFields.Count; }
        }

        /// <summary>
        /// Returns the NPY file for which the fields are associated with.
        /// </summary>
        public string File
        {
            get { return m_strFile; }
        }
    }

    /// <summary>
    /// The Field class manages a single field.
    /// </summary>
    public class Field
    {
        string m_strName;
        int m_nIdx = 0;
        DATA_TYPE m_dataType = DATA_TYPE.REAL;
        INPUT_TYPE m_inputType = INPUT_TYPE.TIME;

        /// <summary>
        /// Defines the Data Type of the field.
        /// </summary>
        public enum DATA_TYPE
        {
            /// <summary>
            /// Specifies the field contains a real value.
            /// </summary>
            REAL,
            /// <summary>
            /// Specifies the field contains a categorical value.
            /// </summary>
            CATEGORICAL
        }

        /// <summary>
        /// Defines the input type of the field.
        /// </summary>
        public enum INPUT_TYPE
        {
            /// <summary>
            /// Specifies an empty input type.
            /// </summary>
            NONE = 0x0000,
            /// <summary>
            /// Specifies a time input type used for reference.
            /// </summary>
            TIME = 0x0001,
            /// <summary>
            /// Specifies an ID input type used for reference.
            /// </summary>
            ID = 0x0002,
            /// <summary>
            /// Specifies a known input type used in historical and future data.
            /// </summary>
            KNOWN = 0x0004,
            /// <summary>
            /// Specifies an observed input type used in historical data.
            /// </summary>
            OBSERVED = 0x0008,
            /// <summary>
            /// Specifies a static input type used in static data.
            /// </summary>
            STATIC = 0x0010,
            /// <summary>
            /// Specifies a target input type used in future data.  Note an OBSERVED type can be OBSERVED and TARGET.
            /// </summary>
            TARGET = 0x1000,
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strName">Specifies the field name.</param>
        /// <param name="nIdx">Specifies the index of the field within the NPY file.</param>
        /// <param name="dataType">Specifies the field data type.</param>
        /// <param name="inputType">Specifies the field input type.</param>
        public Field(string strName, int nIdx, DATA_TYPE dataType, INPUT_TYPE inputType)
        {
            m_strName = strName;
            m_nIdx = nIdx;
            m_dataType = dataType;
            m_inputType = inputType;
        }

        /// <summary>
        /// Loads a new field from an XML node.
        /// </summary>
        /// <param name="node">Specifies the XML node containing the field.</param>
        /// <returns><The new Field is returned./returns>
        public static Field Load(XmlNode node)
        {
            string strName = node.FirstChild.Value;
            int nIdx = int.Parse(node.Attributes["Index"].Value);
            string strDataType = node.Attributes["DataType"].Value;
            string strInputType = node.Attributes["InputType"].Value;

            DATA_TYPE dataType = DATA_TYPE.REAL;
            INPUT_TYPE inputType = INPUT_TYPE.NONE;

            if (strDataType == "REAL")
                dataType = DATA_TYPE.REAL;
            else if (strDataType == "CATEGORICAL")
                dataType = DATA_TYPE.CATEGORICAL;

            if (strInputType == "TIME")
                inputType = INPUT_TYPE.TIME;
            else if (strInputType == "ID")
                inputType = INPUT_TYPE.ID;
            else if (strInputType == "KNOWN")
                inputType = INPUT_TYPE.KNOWN;
            else if (strInputType == "STATIC")
                inputType = INPUT_TYPE.STATIC;
            else
            {
                if (strInputType.Contains("OBSERVED"))
                    inputType |= INPUT_TYPE.OBSERVED;
                if (strInputType.Contains("TARGET"))
                    inputType |= INPUT_TYPE.TARGET;
            }

            return new Field(strName, nIdx, dataType, inputType);
        }

        /// <summary>
        /// Returns the name of the field.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the Index of the field within the numpy file.
        /// </summary>
        public int Index
        {
            get { return m_nIdx; }
        }

        /// <summary>
        /// Returns the data type of the field.
        /// </summary>
        public DATA_TYPE DataType
        {
            get { return m_dataType; }
        }

        /// <summary>
        /// Returns the input type of the field.
        /// </summary>
        public INPUT_TYPE InputType
        {
            get { return m_inputType; }
        }

        /// <summary>
        /// Returns a string representation of the field.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        override public string ToString()
        {
            return m_strName + " @" + m_nIdx.ToString() + " (" + m_dataType.ToString() + "," + m_inputType.ToString() + ")";
        }
    }
}
