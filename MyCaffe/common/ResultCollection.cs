using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using MyCaffe.param;

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
        RESULT_TYPE m_resultType = RESULT_TYPE.NONE;

        /// <summary>
        /// Defines the type of result.
        /// </summary>
        public enum RESULT_TYPE
        {
            /// <summary>
            /// Specifies that no result type for the data.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies that the results represent probabilities.
            /// </summary>
            PROBABILITIES,
            /// <summary>
            /// Specifies that the results represent distances.
            /// </summary>
            DISTANCES
        }

        /// <summary>
        /// The ResultCollection constructor.
        /// </summary>
        /// <param name="rgResults">Specifies the results listed in pairs of label/result.</param>
        /// <param name="outputLayerType">Specifies the output layer type.</param>
        public ResultCollection(List<KeyValuePair<int, double>> rgResults, LayerParameter.LayerType outputLayerType)
        {
            m_resultType = getResultType(outputLayerType);
            m_rgResultsOriginal = rgResults;

            foreach (KeyValuePair<int, double> kv in rgResults)
            {
                m_rgResultsSorted.Add(kv);
            }

            m_rgResultsSorted.Sort(new Comparison<KeyValuePair<int, double>>(sortResults));
        }

        private RESULT_TYPE getResultType(LayerParameter.LayerType type)
        {
            switch (type)
            {
                case LayerParameter.LayerType.SOFTMAX:
                    return RESULT_TYPE.PROBABILITIES;

                case LayerParameter.LayerType.DECODE:
                    return RESULT_TYPE.DISTANCES;

                default:
                    return RESULT_TYPE.NONE;
            }
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

        public RESULT_TYPE ResultType
        {
            get { return m_resultType; }
        }

        /// <summary>
        /// Returns the data encoding values.
        /// </summary>
        public List<double> GetEncoding()
        {
            List<double> rg = new List<double>();

            foreach (KeyValuePair<int, double> kv in m_rgResultsOriginal)
            {
                rg.Add(kv.Value);
            }

            return rg;
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
        /// Returns the detected label with the maximum signal.
        /// </summary>
        /// <remarks>
        /// The maximum signal label is used to detect the output from a SoftMax where each 
        /// label is given a probability and the label with the highest probability is the
        /// detected label.
        /// </remarks>
        public int DetectedLabelMaxSignal
        {
            get { return m_rgResultsSorted[0].Key; }
        }

        /// <summary>
        /// Returns the detected label output with the maximum signal.
        /// </summary>
        /// <remarks>
        /// The maximum signal label is used to detect the output from a SoftMax where each 
        /// label is given a probability and the label with the highest probability is the
        /// detected label.
        /// </remarks>
        public double DetectedLabelOutputMaxSignal
        {
            get { return m_rgResultsSorted[0].Value; }
        }

        /// <summary>
        /// Returns the detected label with the minimum signal.
        /// </summary>
        /// <remarks>
        /// The minimum signal label is used to detect the output from a Decode alyer where each 
        /// label is given the distance from which the data's encoding is from the centroid of the
        /// label - the encoding with the minimum distance signifies the detected label.
        /// </remarks>
        public int DetectedLabelMinSignal
        {
            get { return m_rgResultsSorted[m_rgResultsSorted.Count-1].Key; }
        }

        /// <summary>
        /// Returns the detected label output of the label with the minimum signal.
        /// </summary>
        /// <remarks>
        /// The minimum signal label is used to detect the output from a Decode alyer where each 
        /// label is given the distance from which the data's encoding is from the centroid of the
        /// label - the encoding with the minimum distance signifies the detected label.
        /// </remarks>
        public double DetectedLabelOutputMinSignal
        {
            get { return m_rgResultsSorted[m_rgResultsSorted.Count-1].Value; }
        }

        /// <summary>
        /// Returns the detected label depending on the result type (distance or probability) with a default type of probability (max label signal) used.
        /// </summary>
        public int DetectedLabel
        {
            get
            {
                if (m_resultType == RESULT_TYPE.DISTANCES)
                    return DetectedLabelMinSignal;
                else
                    return DetectedLabelMaxSignal;
            }
        }

        /// <summary>
        /// Returns the detected label output depending on the result type (distance or probability) with a default type of probability (max label signal) used.
        /// </summary>
        public double DetectedLabelOutput
        {
            get
            {
                if (m_resultType == RESULT_TYPE.DISTANCES)
                    return DetectedLabelOutputMinSignal;
                else
                    return DetectedLabelOutputMaxSignal;
            }
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

                if (nLabel == DetectedLabelMaxSignal)
                    strOut += "[";

                if (strName != null)
                    strOut += strName;
                else
                    strOut += nLabel.ToString();

                strOut += "->";
                strOut += dfVal.ToString("N4");

                if (nLabel == DetectedLabelMaxSignal)
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
