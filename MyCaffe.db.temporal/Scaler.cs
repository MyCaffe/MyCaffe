using SimpleGraphing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The Scaler abstract class is the base class for all scalers.
    /// </summary>
    public abstract class Scaler
    {
        /// <summary>
        /// Specifies the type of scaler.
        /// </summary>
        protected SCALER m_type = SCALER.IDENTITY;
        /// <summary>
        /// Optionally specifies the minimum value used by some scalers.
        /// </summary>
        protected float? m_fMin = null;
        /// <summary>
        /// Optionally specifies the maximum value used by some scalers.
        /// </summary>
        protected float? m_fMax = null;
        /// <summary>
        /// Optionally specifies the final scale value applied to the scaled value.
        /// </summary>
        protected float m_fFinalScale = 1.0f;
        /// <summary>
        /// The calculation array is used to calculate different statistics.
        /// </summary>
        protected CalculationArray m_ca;
        /// <summary>
        /// Specifies the length of the calculation array used to calculate certain statistics.
        /// </summary>
        protected int m_nLength = 0;

        /// <summary>
        /// Defines the type of scaler to use.
        /// </summary>
        public enum SCALER
        {
            /// <summary>
            /// Creates a ScalerIdentity scaler.
            /// </summary>
            IDENTITY = 0,
            /// <summary>
            /// Creates a ScalerCenter scaler.
            /// </summary>
            CENTER = 1,
            /// <summary>
            /// Creates a ScalerCenter scaler that only subtracts the mean with a scale of 1.0.
            /// </summary>
            CENTER1 = 2,
            /// <summary>
            /// Creates a ScalerMinMax scaler.
            /// </summary>
            MINMAX = 3,
            /// <summary>
            /// Creates a ScalerPctChange scaler.
            /// </summary>
            PCTCHG = 4,
            /// <summary>
            /// Creates a ScalerPctChange scaler.
            /// </summary>
            PCTCHG_CENTER = 5
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nLength">Specifies the length of the calculation array used to calculate certain statistics.</param>
        /// <param name="fFinalScale">Optionally, specifies the final scale applied to the scaled value (default = 1.0).</param>
        /// <param name="fMin">Optionally, specifies the min used by some scalers (default = null).</param>
        /// <param name="fMax">Optionally, specifies the max used by some scalers (default = null).</param>
        public Scaler(int nLength, float fFinalScale = 1.0f, float? fMin = null, float? fMax = null)
        {
            m_fFinalScale = fFinalScale;
            m_fMin = fMin;
            m_fMax = fMax;
            m_nLength = nLength;
            m_ca = new CalculationArray(nLength);
        }

        /// <summary>
        /// Returns the type of the scaler.
        /// </summary>
        public SCALER ID
        {
            get { return m_type; }
        }

        /// <summary>
        /// Returns the length of the scaler.
        /// </summary>
        public int Length
        {
            get { return m_nLength; }
        }

        /// <summary>
        /// Returns the minimum value if any.
        /// </summary>
        public float? Minimum
        {
            get { return m_fMin; }
        }

        /// <summary>
        /// Returns the maximum value if any.
        /// </summary>
        public float? Maximum
        {
            get { return m_fMax; }
        }

        /// <summary>
        /// Returns the final scale value.
        /// </summary>
        public float FinalScale
        {
            get { return m_fFinalScale; }
        }

        /// <summary>
        /// Add a value to the scaler.
        /// </summary>
        /// <param name="fVal">Specifies the value to be added.</param>
        /// <returns>If enough values are in the scaler to calculate the scale, True is returned, otherwise False.</returns>
        public virtual bool Add(float fVal)
        {
            return m_ca.Add(fVal, null, false);
        }

        /// <summary>
        /// Calculate the scaled value.
        /// </summary>
        /// <param name="fVal">Specifies the value to scale.</param>
        /// <returns>The scaled value is returned.</returns>
        public virtual float? Scale(float fVal)
        {
            if (m_fFinalScale == 1.0f)
                return fVal;

            return fVal * m_fFinalScale;
        }

        /// <summary>
        /// Calculate the unscaled value.
        /// </summary>
        /// <param name="fVal">Specifies the scaled value to unscale.</param>
        /// <returns>The unscaled value is returned.</returns>
        public virtual float? UnScale(float fVal)
        {
            if (m_fFinalScale == 1.0f)
                return fVal;

            return fVal / m_fFinalScale;
        }
    }

    /// <summary>
    /// The ScalerCenter class scales the data by centering it around the mean and optionally dividing by the standard deviation.
    /// </summary>
    public class ScalerCenter : Scaler
    {
        bool m_bDivByStdDev = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nLength">Specifies the length over which the mean and stdev are calculated.</param>
        /// <param name="bDivByStdDev">Optional, when true the (val - mean) is divided by the stddev (default = True).</param>
        /// <param name="fFinalScale">Optionally, specifies the final scale applied to the calculated scale value (default = 1.0).</param>
        public ScalerCenter(int nLength, bool bDivByStdDev = true, float fFinalScale = 1.0f) : base(nLength, fFinalScale)
        {
            m_type = (bDivByStdDev) ? SCALER.CENTER : SCALER.CENTER1;
            m_bDivByStdDev = bDivByStdDev;
        }

        /// <summary>
        /// Calculate the centered scaled value.
        /// </summary>
        /// <param name="fVal">Specifies the value to scale.</param>
        /// <returns>The centered scaled value is returned.</returns>
        public override float? Scale(float fVal)
        {
            float fMean = (float)m_ca.Average;

            fVal -= fMean;

            if (m_bDivByStdDev)
            {
                float fStdDev = (float)m_ca.StdDev;
                if (fStdDev != 0)
                    fVal /= fStdDev;
            }

            return base.Scale(fVal);
        }

        /// <summary>
        /// Unscale the centered scaled value.
        /// </summary>
        /// <param name="fVal">Specifies the scaled value</param>
        /// <returns>The unscaled value is returned.</returns>
        public override float? UnScale(float fVal)
        {
            float? fVal1 = base.UnScale(fVal);
            if (!fVal1.HasValue)
                return null;

            if (m_bDivByStdDev)
            {
                float fStdDev = (float)m_ca.StdDev;
                if (fStdDev != 0)
                    fVal1 *= fStdDev;
            }

            float fMean = (float)m_ca.Average;
            fVal1 += fMean;

            return fVal1;
        }
    }

    /// <summary>
    /// The ScalerMinMax class scales the data by subtracting the min and dividing by the max - min.
    /// </summary>
    public class ScalerMinMax : Scaler
    {
        float m_fRange = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fMin">Specifies the min value.</param>
        /// <param name="fMax">Specifies the max value.</param>
        /// <param name="fFinalScale">Optionally, specifies the final scale applied to the calculated scale value (default = 1.0).</param>
        public ScalerMinMax(float fMin, float fMax, float fFinalScale = 1.0f) : base(1, fFinalScale, fMin, fMax)
        {
            m_fRange = fMax - fMin;
            m_type = SCALER.MINMAX;
        }

        /// <summary>
        /// Adds a new value to the scaler.
        /// </summary>
        /// <param name="fVal">Specifies the value to add.</param>
        /// <returns>This method always returns true for the MinMax scaler does not require a length of items.</returns>
        public override bool Add(float fVal)
        {
            return true;
        }

        /// <summary>
        /// Calculate the minmax scaled value.
        /// </summary>
        /// <param name="fVal">Specifies the value to scale.</param>
        /// <returns>The minmax scaled value is returned.</returns>
        public override float? Scale(float fVal)
        {
            float fScaled = (m_fRange == 0) ? 0 : (fVal - m_fMin.Value) / m_fRange;
            return base.Scale(fScaled);
        }

        /// <summary>
        /// Calculate the unscaled value.
        /// </summary>
        /// <param name="fVal">Specifies the scaled value to unscale.</param>
        /// <returns>The unscaled value is returned.</returns>
        public override float? UnScale(float fVal)
        {
            float? fVal1 = base.UnScale(fVal);
            if (!fVal1.HasValue)
                return null;

            float fUnscaled = fVal1.Value * m_fRange + m_fMin.Value;
            return fUnscaled;
        }
    }

    /// <summary>
    /// The ScalerPctChange class scales the data by calculating the percent change from the previous value and accumulating the results.
    /// </summary>
    public class ScalerPctChange : Scaler
    {
        bool m_bCenter = false;
        float? m_fLast = null;
        List<float> m_rgfPctAccum = new List<float>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bCenter">Specifies to center after creating pct change accumulator.</param>
        /// <param name="nLength">Optionally, specifies the lenght (value greater than 5 required for centering)</param>
        /// <param name="fFinalScale">Optionally, specifies the final scale applied to the calculated scale value (default = 1.0).</param>
        public ScalerPctChange(bool bCenter, int nLength = 2, float fFinalScale = 1.0f) : base(nLength, fFinalScale)
        {
            m_bCenter = bCenter;
            m_type = SCALER.PCTCHG_CENTER;
        }

        /// <summary>
        /// Add a new value to the scaler.
        /// </summary>
        /// <param name="fVal">Specifies the value to add.</param>
        /// <returns>This scaler requires at least 2 items and returns True after that number of items has been added and False otherwise.</returns>
        public override bool Add(float fVal)
        {
            if (!m_fLast.HasValue || fVal == 0)
            {
                m_rgfPctAccum.Add(0);
                m_fLast = fVal;
                return false;
            }

            float fPct = (fVal - m_fLast.Value) / m_fLast.Value;

            if (m_ca.Count == 0 && fPct == 0)
                return false;

            float fLastPct = m_rgfPctAccum[m_rgfPctAccum.Count - 1];
            float fPctAccum = fLastPct + fPct;

            m_rgfPctAccum.Add(fPctAccum);

            m_fLast = fVal;

            if (!base.Add(fPctAccum))
                return false;

            return true;
        }

        /// <summary>
        /// Calculate the pct change scaled value.
        /// </summary>
        /// <param name="fVal">Specifies the value to scale.</param>
        /// <returns>The pct change scaled value is returned.</returns>
        public override float? Scale(float fVal)
        {
            float fScaled = m_rgfPctAccum[m_rgfPctAccum.Count - 1];

            if (m_bCenter)
                fScaled -= (float)m_ca.Average;

            return base.Scale(fScaled);
        }

        /// <summary>
        /// Calculate the unscaled value.
        /// </summary>
        /// <param name="fVal">Specifies the scaled value.</param>
        /// <returns>The unscaled value is returned.</returns>
        public override float? UnScale(float fVal)
        {
            float? fVal1 = base.UnScale(fVal);
            if (!fVal1.HasValue)
                return null;

            if (m_bCenter)
                fVal1 += (float)m_ca.Average;

            return fVal1.Value;
        }
    }

    /// <summary>
    /// The ScalerIdentity class scales the data by returning the value unchanged.
    /// </summary>
    public class ScalerIdentity : Scaler
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nLength">Specifies the length of items which is unused.</param>
        public ScalerIdentity(int nLength) : base(nLength)
        {
            m_type = SCALER.IDENTITY;
        }

        /// <summary>
        /// Add a new value to the scaler.
        /// </summary>
        /// <param name="fVal">Specifies the value to add.</param>
        /// <returns>This method always returns True.</returns>
        public override bool Add(float fVal)
        {
            return true;
        }

        /// <summary>
        /// Calculate the identity scaled value.
        /// </summary>
        /// <param name="fVal">Specifies the value to be scaled.</param>
        /// <returns>Since this scaler does no scaling, the fVal is returned unchanged.</returns>
        public override float? Scale(float fVal)
        {
            return fVal;
        }

        /// <summary>
        /// Calculate the unscaled value.
        /// </summary>
        /// <param name="fVal">Specifies the scaled value.</param>
        /// <returns>The unscaled value is returned.</returns>
        public override float? UnScale(float fVal)
        {
            return fVal;
        }
    }

    /// <summary>
    /// The ScalerCollection class manages a collection of scalers.
    /// </summary>
    public class ScalerCollection
    {
        Dictionary<int, Scaler> m_rgScaler = new Dictionary<int, Scaler>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public ScalerCollection()
        {
        }

        /// <summary>
        /// Create a new scaler.
        /// </summary>
        /// <param name="scaler">Specifies the scaler type to create.</param>
        /// <param name="nLength">Specifies the required number of items needed to calculate the scaler.</param>
        /// <param name="dfMin">Specifies the minimum value.</param>
        /// <param name="dfMax">Specifies the maximum value.</param>
        /// <param name="fScale">Optionally, specifies a scaler multiplier.</param>
        /// <returns>The created Scaler is returned.</returns>
        public static Scaler CreateScaler(Scaler.SCALER scaler, int nLength, float dfMin, float dfMax, float fScale = 1.0f)
        {
            switch (scaler)
            {
                case Scaler.SCALER.CENTER:
                    return new ScalerCenter(nLength);

                case Scaler.SCALER.CENTER1:
                    return new ScalerCenter(nLength, false, fScale);

                case Scaler.SCALER.MINMAX:
                    return new ScalerMinMax(dfMin, dfMax);

                case Scaler.SCALER.PCTCHG:
                    return new ScalerPctChange(false, 2, fScale);

                case Scaler.SCALER.PCTCHG_CENTER:
                    return new ScalerPctChange(true, nLength, fScale);

                case Scaler.SCALER.IDENTITY:
                default:
                    return new ScalerIdentity(nLength);
            }
        }

        /// <summary>
        /// Returns the number of scalers in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgScaler.Count; }
        }

        /// <summary>
        /// Returns whether or not the collection contains a given scaler type..
        /// </summary>
        /// <param name="nScalerType">Specifies the scaler type.</param>
        /// <returns>If the scaler type exists, True is returned.</returns>
        public bool ContainsField(int nScalerType)
        {
            return m_rgScaler.ContainsKey(nScalerType);
        }

        /// <summary>
        /// Add a new scaler to the collection.
        /// </summary>
        /// <param name="nScalerType">Specifies the scaler type of SCALER.</param>
        /// <param name="scaler">Specifies the new scaler.</param>
        public void Add(int nScalerType, Scaler scaler)
        {
            m_rgScaler.Add(nScalerType, scaler);
        }

        /// <summary>
        /// Get a scaler from the collection.
        /// </summary>
        /// <param name="nScalerType">Specifies the scaler SCALER type to return.</param>
        /// <returns>If found, the Scaler is returned, otherwise null.</returns>
        public Scaler Get(int nScalerType)
        {
            if (!m_rgScaler.ContainsKey(nScalerType))
                return null;

            return m_rgScaler[nScalerType];
        }

        /// <summary>
        /// Get a scaler from the collection.
        /// </summary>
        /// <param name="nScalerType">Specifies the scaler SCALER type to return.</param>
        /// <returns>If found, the Scaler is returned, otherwise null.</returns>
        public Scaler Get(Scaler.SCALER nScalerType)
        {
            if (!m_rgScaler.ContainsKey((int)nScalerType))
                return null;

            return m_rgScaler[(int)nScalerType];
        }
    }
}
