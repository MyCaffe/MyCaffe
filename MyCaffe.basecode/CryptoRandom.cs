using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Security.Cryptography;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The CryptoRandom is a random number generator that can use either the standard .Net Random objec or the more precise 
    /// RandomNumberGenerator defined within the System.Security.Cryptograph.
    /// </summary>
    public class CryptoRandom : RandomNumberGenerator
    {
        private static readonly RNGCryptoServiceProvider m_rand = new RNGCryptoServiceProvider();
        private static Random m_rand1 = new Random();
        private IndexCollection m_rgIdx = null;

        METHOD m_method = METHOD.DEFAULT;

        /// <summary>
        /// Defines the random number generation method to use.
        /// </summary>
        public enum METHOD
        {
            /// <summary>
            /// Uses the default Random object.
            /// </summary>
            SYSTEM,
            /// <summary>
            /// Uses the RandomNumberGenerator object.
            /// </summary>
            /// <remarks>
            /// @see https://stackoverflow.com/questions/4892588/rngcryptoserviceprovider-random-number-review
            /// </remarks>
            CRYPTO,
            /// <summary>
            /// Only used with Next(nMaxVal) and returns exactly evenly selected numbers from the range [0, nMaxVal).
            /// </summary>
            UNIFORM_EXACT,
            /// <summary>
            /// Specifies to use the default CRYPTO.
            /// </summary>
            DEFAULT = CRYPTO
        }

        /// <summary>
        /// The CryptoRandom constructor.
        /// </summary>
        /// <param name="method">Specifies the random number generation method (default = CRYPTO1).</param>
        /// <param name="nSeed">Specifies the seed used to initialize the random number generator, only used when <i>bUseCrypto</i> = <i>false</i> (default = 0, which is ignored).</param>
        public CryptoRandom(METHOD method = METHOD.DEFAULT, int nSeed = 0)
        {
            m_method = method;

            if (nSeed != 0)
            {
                if (m_method == METHOD.CRYPTO)
                    m_method = METHOD.SYSTEM;

                m_rand1 = new Random(nSeed);
            }
        }


        public override void GetBytes(byte[] data) /** @private */
        {
            m_rand.GetBytes(data);
        }

        public override void GetNonZeroBytes(byte[] data) /** @private */
        {
            m_rand.GetNonZeroBytes(data);
        }

        /// <summary>
        /// Returns a random double within the range @f$ [0, 1] @f$.
        /// </summary>
        /// <returns>The random double is returned.</returns>
        public double NextDouble()
        {
            if (m_method == METHOD.CRYPTO)
            {
                byte[] rgb = new byte[sizeof(UInt64)];
                m_rand.GetBytes(rgb);
                return (double)BitConverter.ToUInt64(rgb, 0) / (double)UInt64.MaxValue;
            }

            return m_rand1.NextDouble();
        }

        /// <summary>
        /// Returns a random <i>double</i> within the range @f$ [dfMin, dfMax] @f$
        /// </summary>
        /// <param name="dfMin">Specifies the range minimum.</param>
        /// <param name="dfMax">Specifies the range maximum.</param>
        /// <returns>The random <i>double</i> is returned.</returns>
        public double NextDouble(double dfMin, double dfMax)
        {
            return (NextDouble() * (dfMax - dfMin)) + dfMin;
        }

        /// <summary>
        /// Returns a random <i>int</i> within the range @f$ [nMinVal, nMaxVal] @f$
        /// </summary>
        /// <param name="nMinVal">Specifies the range minimum.</param>
        /// <param name="nMaxVal">Specifies the range maximum.</param>
        /// <returns>The random <i>int</i> is returned.</returns>
        public int Next(int nMinVal, int nMaxVal)
        {
            int nVal = (int)Math.Floor((NextDouble() * ((double)nMaxVal - nMinVal)) + nMinVal);

            if (nVal == nMaxVal)
                nVal--;

            return nVal;
        }

        /// <summary>
        /// Returns a random <i>int</i> within the range @f$ [0, int.MaxValue] @f$
        /// </summary>
        /// <returns>The random <i>int</i> is returned.</returns>
        public int Next()
        {
            return Next(int.MaxValue);
        }

        /// <summary>
        /// Returns a random <i>int</i> within the range @f$ [0, nMaxVal) @f$, where the random number is less than <i>nMaxVal</i>.
        /// </summary>
        /// <param name="nMaxVal">Specifies the non-inclusive maximum of the range.</param>
        /// <returns>The random <i>int</i> is returned.</returns>
        public int Next(int nMaxVal)
        {
            if (m_method == METHOD.UNIFORM_EXACT)
            {
                if (m_rgIdx == null)
                    m_rgIdx = new IndexCollection(nMaxVal);
                else if (nMaxVal != m_rgIdx.Count)
                    throw new Exception("CryptoRandom: The maximum count has changed!");

                IndexCollection rgMin = m_rgIdx.GetMinumums();
                int nIdx = Next(0, rgMin.Count);

                nIdx = rgMin.Index(nIdx);
                return nIdx;
            }

            return Next(0, nMaxVal);
        }
    }

    class IndexCollection
    {
        Item[] m_rgItems;

        public IndexCollection(int nCount)
        {
            m_rgItems = new Item[nCount];

            for (int i = 0; i < nCount; i++)
            {
                m_rgItems[i] = new Item(i);
            }
        }

        public IndexCollection(Item[] rg)
        {
            m_rgItems = rg;
        }

        public int Count
        {
            get { return m_rgItems.Length; }
        }

        public int Add(int nIdx)
        {
            m_rgItems[nIdx].Count++;
            return m_rgItems[nIdx].Count;
        }

        public int Index(int nIdx)
        {
            m_rgItems[nIdx].Count++;
            return m_rgItems[nIdx].Index;
        }

        public IndexCollection GetMinumums()
        {
            int nMinCount = int.MaxValue;

            for (int i = 0; i < m_rgItems.Length; i++)
            {
                if (m_rgItems[i].Count < nMinCount)
                    nMinCount = m_rgItems[i].Count;
            }

            Item[] rgMin = m_rgItems.Where(p => p.Count == nMinCount).ToArray();
            return new IndexCollection(rgMin);
        }
    }

    class Item
    {
        int m_nIdx;
        int m_nCount;

        public Item(int nIdx)
        {
            m_nIdx = nIdx;
            m_nCount = 0;
        }

        public int Index
        {
            get { return m_nIdx; }
        }

        public int Count
        {
            get { return m_nCount; }
            set { m_nCount = value; }
        }

        public override string ToString()
        {
            return m_nIdx.ToString() + " -> " + m_nCount.ToString();
        }
    }
}
