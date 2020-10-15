using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The NormalizedBBox manages a bounding box used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    public class NormalizedBBox
    {
        float m_fxmin = 0;  // [0]
        float m_fymin = 0;  // [1]
        float m_fxmax = 0;  // [2]
        float m_fymax = 0;  // [3]
        int m_nLabel = -1;
        bool m_bDifficult = false;
        float m_fScore = 0;
        float m_fSize = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fxmin">Specifies the bounding box x minimum.</param>
        /// <param name="fymin">Specifies the bounding box y minimum.</param>
        /// <param name="fxmax">Specifies the bounding box x maximum.</param>
        /// <param name="fymax">Specifies the bounding box y maximum.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="bDifficult">Specifies the difficulty.</param>
        /// <param name="fScore">Specifies the score.</param>
        /// <param name="fSize">Specifies the size.</param>
        public NormalizedBBox(float fxmin, float fymin, float fxmax, float fymax, int nLabel = 0, bool bDifficult = false, float fScore = 0, float fSize = 0)
        {
            Set(fxmin, fymin, fxmax, fymax, nLabel, bDifficult, fScore, fSize);
        }

        /// <summary>
        /// Set the values of the NormalizedBbox.
        /// </summary>
        /// <param name="fxmin">Specifies the bounding box x minimum.</param>
        /// <param name="fymin">Specifies the bounding box y minimum.</param>
        /// <param name="fxmax">Specifies the bounding box x maximum.</param>
        /// <param name="fymax">Specifies the bounding box y maximum.</param>
        /// <param name="nLabel">Optionally, specifies the label (default = null, which is ignored).</param>
        /// <param name="bDifficult">Optionally, specifies the difficulty (default = null, which is ignored).</param>
        /// <param name="fScore">Optionally, specifies the score (default = null, which is ignored).</param>
        /// <param name="fSize">Optionally, specifies the size (default = null, which is ignored).</param>
        public void Set(float fxmin, float fymin, float fxmax, float fymax, int? nLabel = null, bool? bDifficult = null, float? fScore = null, float? fSize = null)
        {
            m_fxmin = fxmin;
            m_fxmax = fxmax;
            m_fymin = fymin;
            m_fymax = fymax;

            if (nLabel.HasValue)
                m_nLabel = nLabel.Value;

            if (bDifficult.HasValue)
                m_bDifficult = bDifficult.Value;

            if (fScore.HasValue)
                m_fScore = fScore.Value;

            if (fSize.HasValue)
                m_fSize = fSize.Value;
        }

        /// <summary>
        /// Return a copy of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public NormalizedBBox Clone()
        {
            return new NormalizedBBox(m_fxmin, m_fymin, m_fxmax, m_fymax, m_nLabel, m_bDifficult, m_fScore, m_fSize);
        }

        /// <summary>
        /// Get/set the x minimum.
        /// </summary>
        public float xmin
        {
            get { return m_fxmin; }
            set { m_fxmin = value; }
        }

        /// <summary>
        /// Get/set the x maximum.
        /// </summary>
        public float xmax
        {
            get { return m_fxmax; }
            set { m_fxmax = value; }
        }

        /// <summary>
        /// Get/set the y minimum.
        /// </summary>
        public float ymin
        {
            get { return m_fymin; }
            set { m_fymin = value; }
        }

        /// <summary>
        /// Get/set the y maximum.
        /// </summary>
        public float ymax
        {
            get { return m_fymax; }
            set { m_fymax = value; }
        }

        /// <summary>
        /// Get/set the label.
        /// </summary>
        public int label
        {
            get { return m_nLabel; }
            set { m_nLabel = value; }
        }

        /// <summary>
        /// Get/set the difficulty.
        /// </summary>
        public bool difficult
        {
            get { return m_bDifficult; }
            set { m_bDifficult = value; }
        }

        /// <summary>
        /// Get/set the score.
        /// </summary>
        public float score
        {
            get { return m_fScore; }
            set { m_fScore = value; }
        }

        /// <summary>
        /// Get/set the size.
        /// </summary>
        public float size
        {
            get { return m_fSize; }
            set { m_fSize = value; }
        }

        /// <summary>
        /// Save the NormalizedBbox using the binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer used to save the data.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_fxmin);
            bw.Write(m_fxmax);
            bw.Write(m_fymin);
            bw.Write(m_fymax);
            bw.Write(m_nLabel);
            bw.Write(m_bDifficult);
            bw.Write(m_fScore);
            bw.Write(m_fSize);
        }

        /// <summary>
        /// Load and return a new NormalizedBbox.
        /// </summary>
        /// <param name="br">Specifies the binary reader used to load the data.</param>
        /// <returns>The newly loaded NormalizedBbox is returned.</returns>
        public static NormalizedBBox Load(BinaryReader br)
        {
            float fXmin = br.ReadSingle();
            float fXmax = br.ReadSingle();
            float fYmin = br.ReadSingle();
            float fYmax = br.ReadSingle();
            int nLabel = br.ReadInt32();
            bool bDifficult = br.ReadBoolean();
            float fScore = br.ReadSingle();
            float fSize = br.ReadSingle();

            return new NormalizedBBox(fXmin, fYmin, fXmax, fYmax, nLabel, bDifficult, fScore, fSize);
        }

        /// <summary>
        /// Returns a string representation of the NormalizedBBox.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "(" + m_nLabel.ToString() + ") { ";

            strOut += m_fxmin.ToString() + ", ";
            strOut += m_fymin.ToString() + ", ";
            strOut += m_fxmax.ToString() + ", ";
            strOut += m_fymax.ToString() + " } -> ";
            strOut += m_fScore.ToString();
            strOut += " size = " + m_fSize.ToString();
            strOut += " difficult = " + m_bDifficult.ToString();

            return strOut;
        }

        /// <summary>
        /// Calculates and returns the non-normalized bounding rectangle based in the specified width and height.
        /// </summary>
        /// <param name="nWidth">Specifies the non-normalized width.</param>
        /// <param name="nHeight">Specifies the non-normalized height.</param>
        /// <returns>The non-normalized bounding rectangle is returned.</returns>
        public RectangleF GetBounds(int nWidth, int nHeight)
        {
            float fX1 = m_fxmin * nWidth;
            float fX2 = m_fxmax * nWidth;
            float fY1 = m_fymin * nHeight;
            float fY2 = m_fymax * nHeight;

            return new RectangleF(fX1, fY1, fX2 - fX1, fY2 - fY1);
        }
    }
}
