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

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The TemporalDescriptor is used to describe a temporal aspects of the data source.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TemporalDescriptor
    {
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
            foreach (ValueItemDescriptor vid in td.m_rgValItemDesc)
            {
                m_rgValItemDesc.Add(new ValueItemDescriptor(vid));
            }
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
                foreach (ValueItemDescriptor vid in m_rgValItemDesc)
                {
                    vid.ToBytes(bw);
                }

                return ms.ToArray();
            }
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
        List<ValueStreamDescriptor> m_rgValStreamDesc = new List<ValueStreamDescriptor>();

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

            foreach (ValueStreamDescriptor vsd in vid.m_rgValStreamDesc)
            {
                m_rgValStreamDesc.Add(new ValueStreamDescriptor(vsd));
            }
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

            int nCount = br.ReadInt32();
            for (int i = 0; i < nCount; i++)
            {
                desc.m_rgValStreamDesc.Add(ValueStreamDescriptor.FromBytes(br));
            }

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
            bw.Write(m_rgValStreamDesc.Count);
            foreach (ValueStreamDescriptor vsd in m_rgValStreamDesc)
            {
                vsd.ToBytes(bw);
            }
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
        /// Get/set the value stream descriptors.
        /// </summary>
        [Browsable(false)]
        public List<ValueStreamDescriptor> ValueStreamDescriptors
        {
            get { return m_rgValStreamDesc; }
            set { m_rgValStreamDesc = value; }
        }

        /// <summary>
        /// Returns the value stream descriptors as a read-only array.
        /// </summary>
        public ReadOnlyCollection<ValueStreamDescriptor> ValueStreamDescriptorItems
        {
            get { return m_rgValStreamDesc.AsReadOnly(); }
        }

        /// <summary>
        /// Returns a string representation of the value item descriptor.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return m_strName + " (" + m_rgValStreamDesc.Count.ToString() + " streams)";
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
        int m_nItemCount = 0;

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
        /// <param name="dtStart">Specifies the value stream start date (null with STATIC class).</param>
        /// <param name="dtEnd">Specifies the value stream end date (null with STATIC class).</param>
        /// <param name="nSecondsPerStep">Specifies the value stream seconds per time step (null with STATIC class)</param>
        /// <param name="nItemCount">Specifies the value stream item count.</param>
        public ValueStreamDescriptor(int nID, string strName, int nOrdering, STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valueType, DateTime? dtStart, DateTime? dtEnd, int? nSecondsPerStep, int nItemCount)
        {
            m_nID = nID;
            m_strName = strName;
            m_nOrdering = nOrdering;
            m_classType = classType;
            m_valueType = valueType;
            m_dtStart = dtStart;
            m_dtEnd = dtEnd;
            m_nSecondsPerStep = nSecondsPerStep;
            m_nItemCount = nItemCount;
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
            m_nItemCount = vsd.m_nItemCount;
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

            int nItemCount = br.ReadInt32();

            return new ValueStreamDescriptor(nID, strName, nOrdering, classType, valueType, dtStart, dtEnd, nSecondsPerStep, nItemCount);
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

            bw.Write(m_nItemCount);
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
        public int ItemCount
        {
            get { return m_nItemCount; }
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
