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
        private static RandomNumberGenerator m_rand = RandomNumberGenerator.Create();
        private static Random m_rand1 = new Random();
        bool m_bUseCrypto = true;

        /// <summary>
        /// The CryptoRandom constructor.
        /// </summary>
        /// <param name="bUseCrypto">Specifies whether to use RandomNumberGenerator (<i>true</i>) or Random (<i>false</i>) as the random number generator.</param>
        /// <param name="nSeed">Specifies the seed used to initialize the random number generator, only used when <i>bUseCrypto</i> = <i>false</i> (default = 0, which is ignored).</param>
        public CryptoRandom(bool bUseCrypto = true, int nSeed = 0)
        {
            m_bUseCrypto = bUseCrypto;

            if (nSeed != 0)
                m_rand1 = new Random(nSeed);
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
            if (m_bUseCrypto)
            {
                byte[] rgb = new byte[4];
                m_rand.GetBytes(rgb);
                return (double)BitConverter.ToUInt32(rgb, 0) / (double)UInt32.MaxValue;
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
            int nVal = (int)(NextDouble() * (nMaxVal - nMinVal)) + nMinVal;

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
            return Next(0, nMaxVal);
        }
    }
}
