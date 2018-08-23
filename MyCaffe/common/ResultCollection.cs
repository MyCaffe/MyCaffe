using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.imagedb;

namespace MyCaffe.common
{
    /// <summary>
    /// The ResultCollection contains the result of a given CaffeControl::Run.
    /// </summary>
    public class ResultCollection
    {
        Dictionary<int, string> m_rgLabels = new Dictionary<int, string>();
        List<KeyValuePair<int, double>> m_rgResultsOriginal = new List<KeyValuePair<int, double>>();
        List<KeyValuePair<int, double>> m_rgResultsSorted = new List<KeyValuePair<int, double>>();

        /// <summary>
        /// The ResultCollection constructor.
        /// </summary>
        /// <param name="rgResults">Specifies the results listed in pairs of label/result.</param>
        public ResultCollection(List<KeyValuePair<int, double>> rgResults)
        {
            m_rgResultsOriginal = rgResults;

            foreach (KeyValuePair<int, double> kv in rgResults)
            {
                m_rgResultsSorted.Add(kv);
            }

            m_rgResultsSorted.Sort(new Comparison<KeyValuePair<int, double>>(sortResults));
        }

        /// <summary>
        /// Sort the results by label.
        /// </summary>
        /// <param name="a">Specifies item A.</param>
        /// <param name="b">Specifies item B.</param>
        /// <returns>Returns 1 when A &lt B, -1 when B &lt A, and 0 when they are equal.</returns>
        protected int sortResults(KeyValuePair<int, double> a, KeyValuePair<int, double> b)
        {
            if (a.Value < b.Value)
                return 1;

            if (b.Value < a.Value)
                return -1;

            return 0;
        }

        /// <summary>
        /// Returns the original results.
        /// </summary>
        public List<KeyValuePair<int, double>> ResultsOriginal
        {
            get { return m_rgResultsOriginal; }
        }

        /// <summary>
        /// Returns the original results in sorted order.
        /// </summary>
        public List<KeyValuePair<int, double>> ResultsSorted
        {
            get { return m_rgResultsSorted; }
        }

        /// <summary>
        /// Returns the detected label.
        /// </summary>
        public int DetectedLabel
        {
            get { return m_rgResultsSorted[0].Key; }
        }

        /// <summary>
        /// Returns the detected label output.
        /// </summary>
        public double DetectedLabelOutput
        {
            get { return m_rgResultsSorted[0].Value; }
        }

        /// <summary>
        /// Returns the dictionary lookup of the labels and their names.
        /// </summary>
        public Dictionary<int, string> Labels
        {
            get { return m_rgLabels; }
        }

        /// <summary>
        /// Sets the label names in the label dictionary lookup.
        /// </summary>
        /// <param name="rgLabels"></param>
        public void SetLabels(List<LabelDescriptor> rgLabels)
        {
            m_rgLabels = new Dictionary<int, string>();

            foreach (LabelDescriptor l in rgLabels)
            {
                if (!m_rgLabels.ContainsKey(l.Label))
                    m_rgLabels.Add(l.Label, l.Name);
            }
        }

        /// <summary>
        /// Returns a string representation of the results.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string strOut = "";

            for (int i = 0; i < m_rgResultsOriginal.Count; i++)
            {
                int nLabel = m_rgResultsOriginal[i].Key;
                double dfVal = m_rgResultsOriginal[i].Value;
                string strName = null;

                if (m_rgLabels.ContainsKey(nLabel))
                    strName = m_rgLabels[nLabel];

                if (nLabel == DetectedLabel)
                    strOut += "[";

                if (strName != null)
                    strOut += strName;
                else
                    strOut += nLabel.ToString();

                strOut += "->";
                strOut += dfVal.ToString("N4");

                if (nLabel == DetectedLabel)
                    strOut += "]";

                if (i < m_rgResultsOriginal.Count - 1)
                    strOut += ", ";
            }

            return strOut;
        }

        /// <summary>
        /// Converts the result collection into an image.
        /// </summary>
        /// <param name="clrMap">Optionally, specifies a colormap to use.</param>
        /// <returns>The image respresentation of the result collection is returned.</returns>
        public Image ToImage(ColorMapper clrMap)
        {
            int nW = (int)Math.Ceiling(Math.Sqrt(m_rgResultsOriginal.Count));
            int nH = nW;
            Size sz = new Size(nW, nH);
            return ImageData.GetImage(m_rgResultsOriginal, sz, clrMap);
        }
    }
}
