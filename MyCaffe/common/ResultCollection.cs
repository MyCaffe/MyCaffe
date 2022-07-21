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
        List<Result> m_rgResultsOriginal = new List<Result>();
        List<Result> m_rgResultsSorted = new List<Result>();
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
            DISTANCES,
            /// <summary>
            /// Specifies that the results represent multibox results.
            /// </summary>
            MULTIBOX
        }

        /// <summary>
        /// The ResultCollection constructor.
        /// </summary>
        /// <param name="rgResults">Specifies the results.</param>
        /// <param name="outputLayerType">Specifies the output layer type.</param>
        public ResultCollection(List<Result> rgResults, LayerParameter.LayerType outputLayerType)
        {
            m_resultType = GetResultType(outputLayerType);
            m_rgResultsOriginal = rgResults;

            foreach (Result item in rgResults)
            {
                m_rgResultsSorted.Add(item);
            }

            m_rgResultsSorted = m_rgResultsSorted.OrderByDescending(p => p.Score).ToList();
        }

        public static RESULT_TYPE GetResultType(LayerParameter.LayerType type)
        {
            switch (type)
            {
                case LayerParameter.LayerType.SOFTMAX:
                    return RESULT_TYPE.PROBABILITIES;

                case LayerParameter.LayerType.DECODE:
                    return RESULT_TYPE.DISTANCES;

                case LayerParameter.LayerType.DETECTION_OUTPUT:
                    return RESULT_TYPE.MULTIBOX;

                default:
                    return RESULT_TYPE.NONE;
            }
        }

        /// <summary>
        /// Returns the result type of the result data: PROBABILITIES (Sigmoid), DISTANCES (Decode), or NONE (Unknown).
        /// </summary>
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

            foreach (Result item in m_rgResultsOriginal)
            {
                rg.Add(item.Score);
            }

            return rg;
        }

        /// <summary>
        /// Returns the original results.
        /// </summary>
        public List<Result> ResultsOriginal
        {
            get { return m_rgResultsOriginal; }
        }

        /// <summary>
        /// Returns the original results in sorted order.
        /// </summary>
        public List<Result> ResultsSorted
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
            get { return m_rgResultsSorted[0].Label; }
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
            get { return m_rgResultsSorted[0].Score; }
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
            get { return m_rgResultsSorted[m_rgResultsSorted.Count-1].Label; }
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
            get { return m_rgResultsSorted[m_rgResultsSorted.Count-1].Score; }
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
                int nLabel = m_rgResultsOriginal[i].Label;
                double dfVal = m_rgResultsOriginal[i].Score;
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
