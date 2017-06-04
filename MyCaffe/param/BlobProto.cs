using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// The BlobProto contains the descripion of a blob.
    /// </summary>
    public class BlobProto : BaseParameter, ICloneable, IComparable, IBinaryPersist  
    {
        BlobShape m_rgShape = null;
        List<double> m_rgdfData = new List<double>();
        List<double> m_rgdfDiff = new List<double>();
        List<float> m_rgfData = new List<float>();
        List<float> m_rgfDiff = new List<float>();

        // 4D dimensions -- depreciated.  Use 'shape' instead.
        int? m_nNum = null;
        int? m_nChannels = null;
        int? m_nHeight = null;
        int? m_nWidth = null;

        /// <summary>
        /// Constructor for the BlobProto.
        /// </summary>
        public BlobProto()
        {
        }

        /// <summary>
        /// Constructor for the BlobProto
        /// </summary>
        /// <param name="rgShape">Specifies the shape of the blob.</param>
        public BlobProto(List<int> rgShape)
        {
            m_rgShape = new BlobShape(rgShape);
        }

        /// <summary>
        /// Saves the BlobProto to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        public void Save(BinaryWriter bw)
        {
            bool bHasShape = (m_rgShape != null) ? true : false;

            bw.Write(bHasShape);
            if (bHasShape)
                m_rgShape.Save(bw);

            Utility.Save<double>(bw, m_rgdfData);
            Utility.Save<double>(bw, m_rgdfDiff);
            Utility.Save<float>(bw, m_rgfData);
            Utility.Save<float>(bw, m_rgfDiff);

            bw.Write(m_nNum.HasValue);
            if (m_nNum.HasValue)
                bw.Write(m_nNum.Value);

            bw.Write(m_nChannels.HasValue);
            if (m_nChannels.HasValue)
                bw.Write(m_nChannels.Value);

            bw.Write(m_nHeight.HasValue);
            if (m_nHeight.HasValue)
                bw.Write(m_nHeight.Value);

            bw.Write(m_nWidth.HasValue);
            if (m_nWidth.HasValue)
                bw.Write(m_nWidth.Value);
        }

        /// <summary>
        /// Loads a BlobProto from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created, otherwise the data is read into the existing BlobProto.</param>
        /// <returns>The instance of the BlobProto is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            BlobProto p = this;

            if (bNewInstance)
                p = new BlobProto();

            if (br.ReadBoolean())
                m_rgShape = BlobShape.Load(br);

            m_rgdfData = Utility.Load<double>(br);
            m_rgdfDiff = Utility.Load<double>(br);
            m_rgfData = Utility.Load<float>(br);
            m_rgfDiff = Utility.Load<float>(br);

            if (br.ReadBoolean())
                m_nNum = br.ReadInt32();

            if (br.ReadBoolean())
                m_nChannels = br.ReadInt32();

            if (br.ReadBoolean())
                m_nHeight = br.ReadInt32();

            if (br.ReadBoolean())
                m_nWidth = br.ReadInt32();

            return p;
        }

        /// <summary>
        /// Specifies the shape of the Blob.
        /// </summary>
        public BlobShape shape
        {
            get { return m_rgShape; }
            set { m_rgShape = value; }
        }

        /// <summary>
        /// Specifies the number of inputs (such as images) in the Blob.
        /// </summary>
        public int? num
        {
            get { return m_nNum; }
            set { m_nNum = value; }
        }

        /// <summary>
        /// Specifies the number of images per input.
        /// </summary>
        public int? channels
        {
            get { return m_nChannels; }
            set { m_nChannels = value; }
        }

        /// <summary>
        /// Specifies the height of each input.
        /// </summary>
        public int? height
        {
            get { return m_nHeight; }
            set { m_nHeight = value; }
        }

        /// <summary>
        /// Specifies the width of each input.
        /// </summary>
        public int? width
        {
            get { return m_nWidth; }
            set { m_nWidth = value; }
        }

        /// <summary>
        /// Get/set the data as a List of <i>double</i>.
        /// </summary>
        public List<double> double_data
        {
            get { return m_rgdfData; }
            set { m_rgdfData = value; }
        }

        /// <summary>
        /// Get/set the diff as a List of <i>double</i>.
        /// </summary>
        public List<double> double_diff
        {
            get { return m_rgdfDiff; }
            set { m_rgdfDiff = value; }
        }

        /// <summary>
        /// Get/set the data as a List of <i>float</i>.
        /// </summary>
        public List<float> data
        {
            get { return m_rgfData; }
            set { m_rgfData = value; }
        }

        /// <summary>
        /// Get/set the diff as a List of <i>float</i>.
        /// </summary>
        public List<float> diff
        {
            get { return m_rgfDiff; }
            set { m_rgfDiff = value; }
        }

        /// <summary>
        /// Copies the BlobProto and returns a new instance.
        /// </summary>
        /// <returns>The new instance is returned.</returns>
        public object Clone()
        {
            BlobProto bp = new BlobProto(m_rgShape.dim);

            bp.m_rgdfData = Utility.Clone<double>(m_rgdfData);
            bp.m_rgdfDiff = Utility.Clone<double>(m_rgdfDiff);
            bp.m_rgfData = Utility.Clone<float>(m_rgfData);
            bp.m_rgfDiff = Utility.Clone<float>(m_rgfDiff);
            bp.num = num;
            bp.channels = channels;
            bp.height = height;
            bp.width = width;

            return bp;
        }

        /// <summary>
        /// Converts the BlobProto to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name for the RawProto.</param>
        /// <returns>The RawProto representing the BlobProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (shape != null)
                rgChildren.Add(shape.ToProto("shape"));

            rgChildren.Add("num", num);
            rgChildren.Add("channels", channels);
            rgChildren.Add("height", height);
            rgChildren.Add("width", width);
            rgChildren.Add<double>("double_data", double_data);
            rgChildren.Add<double>("double_diff", double_diff);
            rgChildren.Add<float>("data", data);
            rgChildren.Add<float>("diff", diff);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses a new BlobProto from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the BlobProto is returned.</returns>
        public static BlobProto FromProto(RawProto rp)
        {
            BlobProto p = new BlobProto();

            RawProto rpShape = rp.FindChild("shape");
            if (rpShape != null)
                p.shape = BlobShape.FromProto(rpShape);

            p.num = (int?)rp.FindValue("num", typeof(int));
            p.channels = (int?)rp.FindValue("channels", typeof(int));
            p.height = (int?)rp.FindValue("height", typeof(int));
            p.width = (int?)rp.FindValue("width", typeof(int));
            p.double_data = rp.FindArray<double>("double_data");
            p.double_diff = rp.FindArray<double>("double_diff");
            p.data = rp.FindArray<float>("data");
            p.diff = rp.FindArray<float>("diff");

            return p;
        }

        /// <summary>
        /// Compares the BlobProto to another BlobProto.
        /// </summary>
        /// <param name="obj">Specifies the other BlobProto to compare to.</param>
        /// <returns>If the two BlobProto's are the same <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public int CompareTo(object obj)
        {
            BlobProto bp = obj as BlobProto;

            if (bp == null)
                return 1;

            if (!Compare(bp))
                return 1;

            return 0;
        }

        /// <summary>
        /// Load a new BlobProto from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>A new instance of the BlobProto is returned.</returns>
        public static BlobProto Load(BinaryReader br)
        {
            BlobProto b = new BlobProto();
            return (BlobProto)b.Load(br, true);
        }
    }
}
