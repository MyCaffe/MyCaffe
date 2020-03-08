using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{

    /// <summary>
    /// The Result class contains a single result.
    /// </summary>
    public class Result
    {
        int m_nLabel;
        double m_dfScore;
        double[] m_rgExtra = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nLabel">The label detected.</param>
        /// <param name="dfScore">The score of the run.</param>
        /// <param name="rgExtra">Extra data related to the result.</param>
        public Result(int nLabel, double dfScore, double[] rgExtra = null)
        {
            m_nLabel = nLabel;
            m_dfScore = dfScore;
            m_rgExtra = rgExtra;
        }

        /// <summary>
        /// Returns the label.
        /// </summary>
        public int Label
        {
            get { return m_nLabel; }
        }

        /// <summary>
        /// Returns the score of the run.
        /// </summary>
        public double Score
        {
            get { return m_dfScore; }
        }

        /// <summary>
        /// Returns the extra data.
        /// </summary>
        public double[] Extra
        {
            get { return m_rgExtra; }
        }

        /// <summary>
        /// Returns a string representation of the result.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return "Label " + m_nLabel.ToString() + " -> " + m_dfScore.ToString("N5");
        }
    }
}
