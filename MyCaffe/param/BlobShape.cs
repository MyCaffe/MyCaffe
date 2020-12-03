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
    /// Specifies the shape of a Blob.
    /// </summary>
    public class BlobShape : BaseParameter, ICloneable, IComparable, IBinaryPersist  
    {
        List<int> m_rgDim = new List<int>();

        /// <summary>
        /// The BlobShape constructor.
        /// </summary>
        public BlobShape()
        {
        }

        /// <summary>
        /// The BlobShape constructor.
        /// </summary>
        /// <param name="nNum">Specifies the number of data items.</param>
        /// <param name="nC">Specifies the data channels.</param>
        /// <param name="nH">Specifies the data height.</param>
        /// <param name="nW">Specifies the data width.</param>
        public BlobShape(int nNum, int nC, int nH, int nW)
        {
            m_rgDim = new List<int>() { nNum, nC, nH, nW };
        }

        /// <summary>
        /// The BlobShape constructor.
        /// </summary>
        /// <param name="rgShape">Specifies the shape of a blob.</param>
        public BlobShape(List<int> rgShape)
        {
            m_rgDim = new List<int>();

            for (int i = 0; i < rgShape.Count; i++)
            {
                m_rgDim.Add(rgShape[i]);
            }
        }

        /// <summary>
        /// Save the BlobShape to a binary writer.
        /// </summary>
        /// <param name="bw">The binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            Utility.Save<int>(bw, m_rgDim);
        }

        /// <summary>
        /// Load the BlobShape from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <param name="bNewInstance">When <i>true</i>, a the BlobShape is read into a new instance, otherwise it is read into the current instance.</param>
        /// <returns>The BlobShape instance is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            BlobShape b = this;
            
            if (bNewInstance)
                b = new BlobShape();

            b.m_rgDim = Utility.Load<int>(br);

            return b;
        }

        /// <summary>
        /// Load the BlobShape from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <returns>A new BlobShape instance is returned.</returns>
        public static BlobShape Load(BinaryReader br)
        {
            BlobShape b = new BlobShape();
            return (BlobShape)b.Load(br, true);
        }

        /// <summary>
        /// The blob shape dimensions.
        /// </summary>
        public List<int> dim
        {
            get { return m_rgDim; }
            set { m_rgDim = new List<int>(value); }
        }

        /// <summary>
        /// Creates a copy of the BlobShape.
        /// </summary>
        /// <returns>A new instance of the BlobShape is returned.</returns>
        public BlobShape Clone()
        {
            BlobShape bs = new BlobShape();

            bs.m_rgDim = Utility.Clone<int>(m_rgDim);

            return bs;
        }

        /// <summary>
        /// Creates a copy of the BlobShape.
        /// </summary>
        /// <returns>A new instance of the BlobShape is returned.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        /// Compares this BlobShape to another.
        /// </summary>
        /// <param name="bs">Specifies the other BlobShape to compare this one to.</param>
        /// <returns>If the two BlobShape's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(BlobShape bs)
        {
            return Utility.Compare<int>(m_rgDim, bs.m_rgDim);
        }

        /// <summary>
        /// Compares this BlobShape to another.
        /// </summary>
        /// <param name="obj">Specifies the other BlobShape to compare this one to.</param>
        /// <returns>If the two BlobShape's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public int CompareTo(object obj)
        {
            BlobShape bs = obj as BlobShape;

            if (bs == null)
                return 1;

            if (!Compare(bs))
                return 1;

            return 0;
        }

        /// <summary>
        /// Converts the BlobShape to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name for the RawProto.</param>
        /// <returns>A new RawProto representing the BlobShape is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add<int>("dim", m_rgDim);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parse a new BlobShape from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto containing a representation of the BlobShape.</param>
        /// <returns>A new instance of the BlobShape is returned.</returns>
        public static BlobShape FromProto(RawProto rp)
        {
            return new BlobShape(rp.FindArray<int>("dim"));
        }

        /// <summary>
        /// Return the string representation of the shape.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "{";

            for (int i = 0; i < m_rgDim.Count; i++)
            {
                strOut += m_rgDim[i].ToString();
                strOut += ",";
            }

            strOut = strOut.TrimEnd(',');
            strOut += "}";

            return strOut;
        }
    }
}
