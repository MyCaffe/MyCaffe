using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.descriptors.ValueStreamDescriptor;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The TemporalDescriptor is used to describe a temporal aspects of the data source.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TemporalDescriptor
    {
        List<ValueStreamDescriptor> m_rgValStrmDesc = new List<ValueStreamDescriptor>();
        List<ValueItemDescriptor> m_rgValItemDesc = new List<ValueItemDescriptor>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public TemporalDescriptor()
        {
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="td">Specifies a TemporalDescriptor to copy.</param>
        public TemporalDescriptor(TemporalDescriptor td)
        {
            foreach (ValueStreamDescriptor vsd in td.m_rgValStrmDesc)
            {
                m_rgValStrmDesc.Add(new ValueStreamDescriptor(vsd));
            }

            foreach (ValueItemDescriptor vid in td.m_rgValItemDesc)
            {
                m_rgValItemDesc.Add(new ValueItemDescriptor(vid));
            }
        }

        /// <summary>
        /// Returns the value stream descriptor.
        /// </summary>
        [ReadOnly(true)]
        public List<ValueStreamDescriptor> ValueStreamDescriptors
        {
            get { return m_rgValStrmDesc; }
            set { m_rgValStrmDesc = value; }
        }

        /// <summary>
        /// Retunrs the ordered set of stream descriptors.
        /// </summary>
        [Browsable(false)]
        public OrderedValueStreamDescriptorSet OrderedValueStreamDescriptors
        {
            get { return new OrderedValueStreamDescriptorSet(m_rgValStrmDesc); }
        }

        /// <summary>
        /// Returns the value item descriptor.
        /// </summary>
        [Browsable(false)]
        public List<ValueItemDescriptor> ValueItemDescriptors
        {
            get { return m_rgValItemDesc; }
            set { m_rgValItemDesc = value; }
        }

        /// <summary>
        /// Returns the value item descriptors as a read-only array.
        /// </summary>
        public ReadOnlyCollection<ValueItemDescriptor> ValueItemDescriptorItems
        {
            get { return m_rgValItemDesc.AsReadOnly(); }
        }

        /// <summary>
        /// Returns the total number of temporal steps.
        /// </summary>
        public int TotalSteps
        {
            get
            {
                int nTotal = 0;

                foreach (ValueStreamDescriptor vsd in m_rgValStrmDesc)
                {
                    nTotal = Math.Max(nTotal, vsd.Steps);
                }

                return nTotal;
            }
        }

        /// <summary>
        /// Return the start date.
        /// </summary>
        public DateTime StartDate
        {
            get { return m_rgValStrmDesc.Min(p => p.Start.GetValueOrDefault(DateTime.MaxValue)); }
        }

        /// <summary>
        /// Return the end date.
        /// </summary>
        public DateTime EndDate
        {
            get { return m_rgValStrmDesc.Max(p => p.End.GetValueOrDefault(DateTime.MinValue)); }
        }

        /// <summary>
        /// Returns a new TemporalDescriptor from a byte array. 
        /// </summary>
        /// <param name="rgb">Specifies the byte array.</param>
        /// <returns>The temporal descriptor is returned, or null.</returns>
        public static TemporalDescriptor FromBytes(byte[] rgb)
        {
            if (rgb == null)
                return null;

            TemporalDescriptor desc = new TemporalDescriptor();

            using (MemoryStream ms = new MemoryStream(rgb))
            using (BinaryReader br = new BinaryReader(ms))
            {
                int nCount = br.ReadInt32();
                for (int i = 0; i < nCount; i++)
                {
                    desc.m_rgValStrmDesc.Add(ValueStreamDescriptor.FromBytes(br));
                }

                nCount = br.ReadInt32();
                for (int i = 0; i < nCount; i++)
                {
                    desc.m_rgValItemDesc.Add(ValueItemDescriptor.FromBytes(br));
                }
            }

            return desc;
        }

        /// <summary>
        /// Returns the temporal descriptor as a byte array.
        /// </summary>
        /// <returns>The byte array is returned.</returns>
        public byte[] ToBytes()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write(m_rgValItemDesc.Count);
                foreach (ValueStreamDescriptor vid in m_rgValStrmDesc)
                {
                    vid.ToBytes(bw);
                }

                bw.Write(m_rgValItemDesc.Count);
                foreach (ValueItemDescriptor vid in m_rgValItemDesc)
                {
                    vid.ToBytes(bw);
                }

                return ms.ToArray();
            }
        }
    }

    /// <summary>
    /// The ordered value stream descriptor set is used to order the value stream descriptors by class and value type.
    /// </summary>
    public class OrderedValueStreamDescriptorSet
    {
        Dictionary<STREAM_CLASS_TYPE, Dictionary<STREAM_VALUE_TYPE, List<ValueStreamDescriptor>>> m_rgOrdered = new Dictionary<STREAM_CLASS_TYPE, Dictionary<STREAM_VALUE_TYPE, List<ValueStreamDescriptor>>>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rg">Specifies the list of stream descriptors.</param>
        public OrderedValueStreamDescriptorSet(List<ValueStreamDescriptor> rg)
        {
            foreach (ValueStreamDescriptor vsd in rg)
            {
                if (!m_rgOrdered.ContainsKey(vsd.ClassType))
                    m_rgOrdered.Add(vsd.ClassType, new Dictionary<STREAM_VALUE_TYPE, List<ValueStreamDescriptor>>());

                if (!m_rgOrdered[vsd.ClassType].ContainsKey(vsd.ValueType))
                    m_rgOrdered[vsd.ClassType].Add(vsd.ValueType, new List<ValueStreamDescriptor>());

                m_rgOrdered[vsd.ClassType][vsd.ValueType].Add(vsd);
            }
        }

        /// <summary>
        /// Retrieves the set of stream descriptors with the given class and value type.
        /// </summary>
        /// <param name="classType">Specifies the class type.</param>
        /// <param name="valueType">Specifies the value type.</param>
        /// <returns>The list of value stream descriptors is returned.</returns>
        public List<ValueStreamDescriptor> GetStreamDescriptors(STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valueType)
        {
            if (!m_rgOrdered.ContainsKey(classType))
                return null;

            if (!m_rgOrdered[classType].ContainsKey(valueType))
                return null;

            return m_rgOrdered[classType][valueType];
        }
    }

    /// <summary>
    /// The ValueItemDescriptor describes each value item (e.g., customer, station or stock)
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ValueItemDescriptor
    {
        int m_nID;
        string m_strName;   

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nID">Specifies the value item ID.</param>
        /// <param name="strName">Specifies the value item name.</param>
        public ValueItemDescriptor(int nID, string strName)
        {
            m_nID = nID;
            m_strName = strName;
        }

        /// <summary>
        /// The copy constructor.
        /// </summary>
        /// <param name="vid">Specifies the value item descriptor to copy.</param>
        public ValueItemDescriptor(ValueItemDescriptor vid)
        {
            m_nID = vid.m_nID;
            m_strName = vid.m_strName;
        }

        /// <summary>
        /// Returns a new ValueItemDescriptor from a binary reader. 
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The value item descriptor is returned, or null.</returns>
        public static ValueItemDescriptor FromBytes(BinaryReader br)
        {
            int nID = br.ReadInt32();
            string strName = br.ReadString();
            ValueItemDescriptor desc = new ValueItemDescriptor(nID, strName);
            return desc;
        }

        /// <summary>
        /// Returns the value item descriptor as a byte array.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <returns>The byte array is returned.</returns>
        public void ToBytes(BinaryWriter bw)
        {
            bw.Write(m_nID);
            bw.Write(m_strName);
        }

        /// <summary>
        /// Returns the value item ID.
        /// </summary>
        [ReadOnly(true)]
        public int ID
        {
            get { return m_nID; }
        }

        /// <summary>
        /// Returns the value item name.
        /// </summary>
        [ReadOnly(true)]
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns a string representation of the value item descriptor.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return m_strName;
        }
    }

    /// <summary>
    /// The value stream descriptor describes a single value stream within a value item.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ValueStreamDescriptor
    {
        STREAM_CLASS_TYPE m_classType = STREAM_CLASS_TYPE.STATIC;
        STREAM_VALUE_TYPE m_valueType = STREAM_VALUE_TYPE.NUMERIC;
        int m_nID;
        string m_strName;
        int m_nOrdering;
        DateTime? m_dtStart = null;
        DateTime? m_dtEnd = null;
        int? m_nSecondsPerStep = null;
        int m_nSteps = 0;

        /// <summary>
        /// Defines the stream value type.
        /// </summary>
        public enum STREAM_VALUE_TYPE
        {
            /// <summary>
            /// Specifies that the value stream hold numeric data.
            /// </summary>
            NUMERIC = 0x01,
            /// <summary>
            /// Specifies that the value stream holds categorical data.
            /// </summary>
            CATEGORICAL = 0x02
        }

        /// <summary>
        /// Defines the stream class type.
        /// </summary>
        public enum STREAM_CLASS_TYPE
        {
            /// <summary>
            /// Specifies static values that are not time related.  The DateTime in each static value is set to NULL.
            /// </summary>
            STATIC = 0x01,
            /// <summary>
            /// Specifies raw values that are only known up to the present time.
            /// </summary>
            OBSERVED = 0x02,
            /// <summary>
            /// Specifies raw values that are known in both the past and future.
            /// </summary>
            KNOWN = 0x04
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nID">Specifies the value stream ID.</param>
        /// <param name="strName">Specifies the value stream name.</param>
        /// <param name="nOrdering">Specifies the value stream ordering.</param>
        /// <param name="classType">Specifies the value stream class type.</param>
        /// <param name="valueType">Specifies the value stream value type.</param>
        /// <param name="dtStart">Specifies the start time of the value stream.</param>
        /// <param name="dtEnd">Specifies the end time of the value stream.</param>
        /// <param name="nSecPerStep">Specifies the number of seconds in each step.</param>
        public ValueStreamDescriptor(int nID, string strName, int nOrdering, STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valueType, DateTime? dtStart = null, DateTime? dtEnd = null, int? nSecPerStep = null, int nSteps = 1)
        {
            m_nID = nID;
            m_strName = strName;
            m_nOrdering = nOrdering;
            m_classType = classType;
            m_valueType = valueType;
            m_dtStart = dtStart;
            m_dtEnd = dtEnd;
            m_nSecondsPerStep = nSecPerStep;
            m_nSteps = nSteps;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="vsd">Specifies the ValueStreamDescriptor to copy.</param>
        public ValueStreamDescriptor(ValueStreamDescriptor vsd)
        {
            m_nID = vsd.m_nID;
            m_strName = vsd.m_strName;
            m_nOrdering = vsd.m_nOrdering;
            m_classType = vsd.m_classType;
            m_valueType = vsd.m_valueType;
            m_dtStart = vsd.m_dtStart;
            m_dtEnd = vsd.m_dtEnd;
            m_nSecondsPerStep = vsd.m_nSecondsPerStep;
            m_nSteps = vsd.m_nSteps;
        }

        /// <summary>
        /// Returns a new ValueStreamDescriptor from a byte array. 
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The value stream descriptor is returned.</returns>
        public static ValueStreamDescriptor FromBytes(BinaryReader br)
        {
            int nID = br.ReadInt32();
            string strName = br.ReadString();
            int nOrdering = br.ReadInt32();
            STREAM_CLASS_TYPE classType = (STREAM_CLASS_TYPE)br.ReadInt32();
            STREAM_VALUE_TYPE valueType = (STREAM_VALUE_TYPE)br.ReadInt32();
            DateTime? dtStart = null;
            DateTime? dtEnd = null;
            int? nSecondsPerStep = null;

            bool bHasStart = br.ReadBoolean();
            if (bHasStart)
                dtStart = new DateTime(br.ReadInt64());

            bool bHasEnd = br.ReadBoolean();
            if (bHasEnd)
                dtEnd = new DateTime(br.ReadInt64());

            bool bHasSecondsPerStep = br.ReadBoolean();
            if (bHasSecondsPerStep)
                nSecondsPerStep = br.ReadInt32();

            int nSteps = br.ReadInt32();

            return new ValueStreamDescriptor(nID, strName, nOrdering, classType, valueType, dtStart, dtEnd, nSecondsPerStep, nSteps);
        }

        /// <summary>
        /// Returns the value stream descriptor as a byte array.
        /// </summary>
        /// <returns>The byte array is returned.</returns>
        public void ToBytes(BinaryWriter bw)
        {
            bw.Write(m_nID);
            bw.Write(m_strName);
            bw.Write(m_nOrdering);
            bw.Write((int)m_classType);
            bw.Write((int)m_valueType);

            bw.Write(m_dtStart.HasValue);
            if (m_dtStart.HasValue)
                bw.Write(m_dtStart.Value.Ticks);

            bw.Write(m_dtEnd.HasValue);
            if (m_dtEnd.HasValue)
                bw.Write(m_dtEnd.Value.Ticks);

            bw.Write(m_nSecondsPerStep.HasValue);
            if (m_nSecondsPerStep.HasValue)
                bw.Write(m_nSecondsPerStep.Value);

            bw.Write(m_nSteps);
        }

        /// <summary>
        /// Returns the value stream ID.
        /// </summary>
        [ReadOnly(true)]
        public int ID
        {
            get { return m_nID; }
        }

        /// <summary>
        /// Return the value stream name.
        /// </summary>
        [ReadOnly(true)]
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the value stream ordering.
        /// </summary>
        [ReadOnly(true)]
        public int Ordering
        {
            get { return m_nOrdering; }
        }

        /// <summary>
        /// Returns the value stream class type.
        /// </summary>
        [ReadOnly(true)]
        public STREAM_CLASS_TYPE ClassType
        {
            get { return m_classType; }
        }

        /// <summary>
        /// Returns the value stream value type.
        /// </summary>
        [ReadOnly(true)]
        public STREAM_VALUE_TYPE ValueType
        {
            get { return m_valueType; }
        }

        /// <summary>
        /// Returns the value stream start time (null with STATIC class).
        /// </summary>
        [ReadOnly(true)]
        public DateTime? Start
        {
            get { return m_dtStart; }
        }

        /// <summary>
        /// Returns the value stream end time (null with STATIC class).
        /// </summary>
        [ReadOnly(true)]
        public DateTime? End
        {
            get { return m_dtEnd; }
        }

        /// <summary>
        /// Returns the value stream seconds per step (null with STATIC class).
        /// </summary>
        [ReadOnly(true)]
        public int? SecondsPerStep
        {
            get { return m_nSecondsPerStep; }
        }

        /// <summary>
        /// Returns the number of items in the value stream.
        /// </summary>
        [ReadOnly(true)]
        public int Steps
        {
            get { return m_nSteps; }
        }

        /// <summary>
        /// Returns the string rendering of the value stream descriptor.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = m_nOrdering.ToString() + ". " + m_strName + " (" + m_classType.ToString() + ", " + m_valueType.ToString() + ")";

            if (m_dtStart.HasValue)
                strOut += " start = " + m_dtStart.ToString();

            if (m_dtEnd.HasValue)
                strOut += " end = " + m_dtEnd.ToString();

            if (m_nSecondsPerStep.HasValue)
                strOut += " seconds_per_step = " + m_nSecondsPerStep.ToString();

            return strOut;
        }
    }
}
