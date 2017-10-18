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
        int m_nResolution;
        List<KeyValuePair<Color, SizeF>> m_rgColorMappings = new List<KeyValuePair<Color, SizeF>>();
        List<List<double>> m_rgrgColors = new List<List<double>>();
        Color m_clrDefault;
        Color m_clrError;

        /// <summary>
        /// Defines the color scheme to use.
        /// </summary>
        public enum COLORSCHEME
        {
            /// <summary>
            /// Use normal coloring where Red is a high value, Blue is a low value, and Green is in the middle.
            /// </summary>
            NORMAL
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

            m_rgrgColors.Add(new List<double>() { 0, 0, 0 });   // black
            m_rgrgColors.Add(new List<double>() { 0, 0, 1 });   // blue
            m_rgrgColors.Add(new List<double>() { 0, 1, 0 });   // green

            if (clrScheme == COLORSCHEME.NORMAL)
                m_rgrgColors.Add(new List<double>() { 1, 0, 0 });   // red

            m_rgrgColors.Add(new List<double>() { 1, 1, 0 });   // yellow

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
        }

        /// <summary>
        /// Returns the color resolution used.
        /// </summary>
        public int Resolution
        {
            get { return m_nResolution; }
        }

        /// <summary>
        /// Returns the color associated with the value.
        /// </summary>
        /// <param name="dfVal">Specifies the value.</param>
        /// <returns>The color associated with the value is returned.</returns>
        public Color GetColor(double dfVal)
        {
            if (double.IsNaN(dfVal) || double.IsInfinity(dfVal))
                return m_clrError;

            if (dfVal >= m_dfMin || dfVal <= m_dfMax)
            {
                if (m_rgColorMappings.Count > 0)
                {
                    for (int i = 0; i < m_rgColorMappings.Count; i++)
                    {
                        if (dfVal < m_rgColorMappings[i].Value.Height && dfVal >= m_rgColorMappings[i].Value.Width)
                            return m_rgColorMappings[i].Key;
                    }

                    if (dfVal == m_rgColorMappings[m_rgColorMappings.Count - 1].Value.Height)
                        return m_rgColorMappings[m_rgColorMappings.Count - 1].Key;
                }
                else if (m_dfMin == m_dfMax && m_dfMin == 0)
                {
                    return Color.Black;
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
            KeyValuePair<Color, SizeF> kvLast = new KeyValuePair<Color, SizeF>(Color.Black, new SizeF(0, 0));

            foreach (KeyValuePair<Color, SizeF> kv in m_rgColorMappings)
            {
                if (kv.Key.R >= clr.R && kv.Key.G >= clr.G && kv.Key.B >= clr.B)
                {
                    if (kv.Key.R == clr.R && kv.Key.G == clr.G && kv.Key.B == clr.B)
                        return kv.Value.Width + ((kv.Value.Height - kv.Value.Width) / 2);
                    else
                        return kvLast.Value.Width + ((kvLast.Value.Height + kvLast.Value.Width) / 2);
                }

                kvLast = kv;
            }

            return 0;
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
    }
}
