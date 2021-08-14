using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ColorMapper maps a value within a number range, to a Color within a color scheme.
    /// </summary>
    public class ColorMapper
    {
        double m_dfMin = 0;
        double m_dfMax = 0;
        double m_dfMid = 0;
        int m_nMidIdx = 0;
        List<Tuple<double, int>> m_rgQuads0 = new List<Tuple<double, int>>();
        List<Tuple<double, int>> m_rgQuads1 = new List<Tuple<double, int>>();
        int m_nResolution;
        List<KeyValuePair<Color, SizeF>> m_rgColorMappings = new List<KeyValuePair<Color, SizeF>>();
        List<List<double>> m_rgrgColors = new List<List<double>>();
        Color m_clrDefault;
        Color m_clrError;
        Color m_clrNoMinMax = Color.HotPink;


        /// <summary>
        /// Defines the color scheme to use.
        /// </summary>
        public enum COLORSCHEME
        {
            /// <summary>
            /// Use normal coloring where Red is a high value, Blue is a low value, and Green is in the middle.
            /// </summary>
            NORMAL,
            /// <summary>
            /// Use coloring where Green is a high value, Red is a low value and Blue is in the middle.
            /// </summary>
            GBR
        }

        /// <summary>
        /// The ColorMapper constructor.
        /// </summary>
        /// <param name="dfMin">Specifies the minimum value in the number range.</param>
        /// <param name="dfMax">Specifies the maximum value in the number range.</param>
        /// <param name="clrDefault">Specifies the default color to use.</param>
        /// <param name="clrError">Specifies the color to use when an error is detected.</param>
        /// <param name="clrScheme">Specifies the color scheme to use (default = COLORSCHEME.NORMAL).</param>
        /// <param name="nResolution">Specifies the number of colors to generate (default = 160).</param>
        public ColorMapper(double dfMin, double dfMax, Color clrDefault, Color clrError, COLORSCHEME clrScheme = COLORSCHEME.NORMAL, int nResolution = 160)
        {
            m_clrDefault = clrDefault;
            m_clrError = clrError;
            m_nResolution = nResolution;
            m_dfMin = dfMin;
            m_dfMax = dfMax;

            if (clrScheme == COLORSCHEME.GBR)
            {
                m_rgrgColors.Add(new List<double>() { 1, 0, 0 });   // red
                m_rgrgColors.Add(new List<double>() { 0, 0, 0.5 });   // blue
                m_rgrgColors.Add(new List<double>() { 0, 1, 0 });   // green
            }
            else
            {
                m_rgrgColors.Add(new List<double>() { 0, 0, 0 });   // black
                m_rgrgColors.Add(new List<double>() { 0, 0, 1 });   // blue
                m_rgrgColors.Add(new List<double>() { 0, 1, 0 });   // green
                m_rgrgColors.Add(new List<double>() { 1, 0, 0 });   // red
                m_rgrgColors.Add(new List<double>() { 1, 1, 0 });   // yellow
            }


            double dfRange = dfMax - dfMin;
            double dfInc = dfRange / m_nResolution;

            dfMax = dfMin + dfInc;

            while (dfMax < m_dfMax)
            {
                Color clr = calculate(dfMin);
                m_rgColorMappings.Add(new KeyValuePair<Color, SizeF>(clr, new SizeF((float)dfMin, (float)dfMax)));
                dfMin = dfMax;
                dfMax += dfInc;
            }

            m_dfMid = dfRange / 2;
            m_nMidIdx = m_rgColorMappings.Count / 2;
            m_rgQuads0 = createSearchQuads(0, 0, m_rgColorMappings.Count, dfRange);
            m_rgQuads1 = createSearchQuads(m_dfMid, m_nMidIdx, m_rgColorMappings.Count, dfRange);
        }

        /// <summary>
        /// The ColorMapper constructor.
        /// </summary>
        /// <param name="dfMin">Specifies the minimum value in the number range.</param>
        /// <param name="dfMax">Specifies the maximum value in the number range.</param>
        /// <param name="clrDefault">Specifies the default color to use.</param>
        /// <param name="clrError">Specifies the color to use when an error is detected.</param>
        /// <param name="rgClrStart">Specifies the RGB three color starting color with values of 0 to 1.</param>
        /// <param name="rgClrEnd">Specifies the RGB three color ending color with values of 0 to 1.</param>
        /// <param name="nResolution">Specifies the number of colors to generate (default = 160).</param>
        public ColorMapper(double dfMin, double dfMax, Color clrDefault, Color clrError, List<double> rgClrStart, List<double> rgClrEnd, int nResolution = 160)
        {
            m_clrDefault = clrDefault;
            m_clrError = clrError;
            m_nResolution = nResolution;
            m_dfMin = dfMin;
            m_dfMax = dfMax;

            m_rgrgColors.Add(rgClrStart);   
            m_rgrgColors.Add(rgClrEnd);

            double dfRange = dfMax - dfMin;
            double dfInc = dfRange / m_nResolution;

            dfMax = dfMin + dfInc;

            while (dfMax < m_dfMax)
            {
                Color clr = calculate(dfMin);
                m_rgColorMappings.Add(new KeyValuePair<Color, SizeF>(clr, new SizeF((float)dfMin, (float)dfMax)));
                dfMin = dfMax;
                dfMax += dfInc;
            }

            m_dfMid = dfRange / 2;
            m_nMidIdx = m_rgColorMappings.Count / 2;
            m_rgQuads0 = createSearchQuads(0, 0, m_rgColorMappings.Count, dfRange);
            m_rgQuads1 = createSearchQuads(m_dfMid, m_nMidIdx, m_rgColorMappings.Count, dfRange);
        }

        private List<Tuple<double, int>> createSearchQuads(double dfStart, int nIdxStart, int nItemCount, double dfRange)
        {
            List<Tuple<double, int>> rgQuads = new List<Tuple<double, int>>();
            int nCount = (nItemCount < 160) ? 2 : 4;
            double dfStep = 0.5 / nCount;
            double dfPortion = dfStep;

            for (int i = 1; i < nCount; i++)
            {
                double dfQuad = dfStart + dfRange * dfPortion;
                int nIdx = nIdxStart + (int)(nItemCount * dfPortion);
                rgQuads.Add(new Tuple<double, int>(dfQuad, nIdx));

                dfPortion += dfStep;
            }

            return rgQuads;
        }

        /// <summary>
        /// Returns the color resolution used.
        /// </summary>
        public int Resolution
        {
            get { return m_nResolution; }
        }

        /// <summary>
        /// Get/set the color used when the Min and Max both equal 0.
        /// </summary>
        public Color NoMinMaxColor
        {
            get { return m_clrNoMinMax; }
            set { m_clrNoMinMax = value; }
        }

        /// <summary>
        /// Returns the color associated with the value.
        /// </summary>
        /// <param name="dfVal">Specifies the value.</param>
        /// <returns>The color associated with the value is returned.</returns>
        public Color GetColorOld(double dfVal)
        {
            if (double.IsNaN(dfVal) || double.IsInfinity(dfVal))
                return m_clrError;

            if (dfVal >= m_dfMin || dfVal <= m_dfMax)
            {
                if (m_rgColorMappings.Count > 0)
                {
                    int nStart = 0;
                    int nEnd = m_rgColorMappings.Count;
                    List<Tuple<double, int>> rgQuads;

                    if (dfVal < m_dfMid)
                    {
                        nEnd = m_nMidIdx + 1;
                        rgQuads = m_rgQuads0;
                    }
                    else
                    {
                        nStart = m_nMidIdx - 1;
                        rgQuads = m_rgQuads1;
                    }

                    for (int i = 0; i < rgQuads.Count; i++)
                    {
                        if (dfVal < rgQuads[i].Item1)
                        {
                            if (i > 0)
                                nStart = rgQuads[i].Item2 - 1;

                            if (i < rgQuads.Count - 1)
                                nEnd = rgQuads[i].Item2 + 1;

                            break;
                        }
                    }

                    if (nStart < 0)
                        nStart = 0;

                    if (nEnd > m_rgColorMappings.Count)
                        nEnd = m_rgColorMappings.Count;

                    for (int i = nStart; i < nEnd; i++)
                    {
                        if (dfVal < m_rgColorMappings[i].Value.Height && dfVal >= m_rgColorMappings[i].Value.Width)
                            return m_rgColorMappings[i].Key;
                    }

                    if (dfVal == m_rgColorMappings[m_rgColorMappings.Count - 1].Value.Height)
                        return m_rgColorMappings[m_rgColorMappings.Count - 1].Key;
                }
                else if (m_dfMin == m_dfMax && m_dfMin == 0)
                {
                    return m_clrNoMinMax;
                }
            }

            return m_clrDefault;
        }

        /// <summary>
        /// Returns the value associated with the color.
        /// </summary>
        /// <param name="clr"></param>
        /// <returns>The value associated with the color is returned.</returns>
        public double GetValue(Color clr)
        {
            List<KeyValuePair<Color, SizeF>> rgItems = m_rgColorMappings.ToList();
            Color clr0 = Color.Black;

            for (int i = 0; i < rgItems.Count; i++)
            {
                Color clr1 = rgItems[i].Key;
                double dfMin = rgItems[i].Value.Width;
                double dfMax = rgItems[i].Value.Height;

                if (i == 0)
                {
                    if (clr.R <= clr1.R && clr.G <= clr1.G && clr.B <= clr1.B)
                        return dfMin + (dfMax - dfMin) / 2;
                }
                else
                {
                    if (((clr1.R >= clr0.R && clr.R <= clr1.R) ||
                         (clr1.R < clr0.R && clr.R >= clr1.R)) &&
                        ((clr1.G >= clr0.G && clr.G <= clr1.G) ||
                         (clr1.G < clr0.G && clr.G >= clr1.G)) &&
                        ((clr1.B >= clr0.B && clr.B <= clr1.B) ||
                         (clr1.B < clr0.B && clr.B >= clr1.B)))
                        return dfMin + (dfMax - dfMin) / 2;
                }

                clr0 = clr1;
            }

            int nCount = rgItems.Count;
            double dfMin1 = rgItems[nCount - 1].Value.Width;
            double dfMax1 = rgItems[nCount - 1].Value.Height;

            return dfMin1 + (dfMax1 - dfMin1) / 2;
        }

        /// <summary>
        /// Calculate the gradient color.
        /// </summary>
        /// <remarks>
        /// Algorithm used from http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients, 
        /// used under the WTFPL (do what you want)! license.
        /// </remarks>
        /// <param name="dfVal">Specifies the value converted into the color range.</param>
        /// <returns>The color associated with the value is returned.</returns>
        private Color calculate(double dfVal)
        {
            double dfMin = m_dfMin;
            double dfMax = m_dfMax;
            double dfRange = dfMax - dfMin;

            // Convert into [0,1] range.
            dfVal = (dfVal - dfMin) / dfRange;

            int nIdx1 = 0;  // |-- desired color will fall between these two indexes.
            int nIdx2 = 0;  // |
            double dfFraction = 0;  // Fraction between two indexes where the value is located.

            if (dfVal < -1)
            {
                nIdx1 = 0;
                nIdx2 = 0;
            }
            else if (dfVal > 1)
            {
                nIdx1 = m_rgrgColors.Count - 1;
                nIdx2 = m_rgrgColors.Count - 1;
            }
            else
            {
                dfVal *= (m_rgrgColors.Count - 1);
                nIdx1 = (int)Math.Floor(dfVal);     // desired color will be after this index.
                nIdx2 = nIdx1 + 1;                  // .. and this index (inclusive).
                dfFraction = dfVal - (double)nIdx1; // distance between two indexes (0-1).
            }

            double dfR = (m_rgrgColors[nIdx2][0] - m_rgrgColors[nIdx1][0]) * dfFraction + m_rgrgColors[nIdx1][0];
            double dfG = (m_rgrgColors[nIdx2][1] - m_rgrgColors[nIdx1][1]) * dfFraction + m_rgrgColors[nIdx1][1];
            double dfB = (m_rgrgColors[nIdx2][2] - m_rgrgColors[nIdx1][2]) * dfFraction + m_rgrgColors[nIdx1][2];

            int nR = (int)(dfR * 255);
            int nG = (int)(dfG * 255);
            int nB = (int)(dfB * 255);

            return Color.FromArgb(nR, nG, nB);
        }

        /// <summary>
        /// Find the color using a binary search algorithm.
        /// </summary>
        /// <remarks>
        /// @see[Binary Search Implementation Using C#](https://www.c-sharpcorner.com/blogs/binary-search-implementation-using-c-sharp1)
        /// </remarks>
        /// <param name="dfVal">Specifies the value to search for.</param>
        /// <returns>The matching color is returned.</returns>
        public Color GetColor(double dfVal)
        {
            if (double.IsNaN(dfVal) || double.IsInfinity(dfVal))
                return m_clrError;

            else if (m_dfMin == m_dfMax && m_dfMin == 0)
                return m_clrNoMinMax;

            else if (dfVal >= m_rgColorMappings[m_rgColorMappings.Count - 1].Value.Width)
                return m_rgColorMappings[m_rgColorMappings.Count - 1].Key;

            else if (dfVal < m_rgColorMappings[0].Value.Height)
                return m_rgColorMappings[0].Key;

            int nMinNum = 0;
            int nMaxNum = m_rgColorMappings.Count - 1;

            while (nMinNum <= nMaxNum)
            {
                int nMid = (nMinNum + nMaxNum) / 2;

                if (dfVal < m_rgColorMappings[nMid].Value.Height && dfVal >= m_rgColorMappings[nMid].Value.Width)
                    return m_rgColorMappings[nMid].Key;

                else if (dfVal == m_rgColorMappings[nMid].Value.Height)
                    return m_rgColorMappings[nMid + 1].Key;

                else if (dfVal < m_rgColorMappings[nMid].Value.Width)
                    nMaxNum = nMid - 1;

                else if (dfVal >= m_rgColorMappings[nMid].Value.Height)
                    nMinNum = nMid + 1;

                else
                    break;
            }

            return m_clrDefault;
        }
    }
}
