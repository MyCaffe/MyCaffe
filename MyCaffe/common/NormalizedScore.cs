using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.param;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The NormalizedScore manages score normalization.
    /// </summary>
    public class NormalizedScore
    {
        bool m_bEnabled = false;
        Log m_log;
        LayerParameterBase.LABEL_TYPE m_labelType = LayerParameterBase.LABEL_TYPE.NONE;
        SCORE_AS_LABEL_NORMALIZATION m_method = SCORE_AS_LABEL_NORMALIZATION.NONE;
        double m_dfPositiveShiftMultiplier = 100.0;
        int m_nSourceID;
        float? m_fScoreMean;
        float? m_fScoreStdev;
        float? m_fNegScoreMean;
        float? m_fNegScoreStdev;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the normalization parameter.</param>
        /// <param name="bEnableOverride">Specifies an enable override.</param>
        public NormalizedScore(Log log, NormalizationScoreParameter p, bool? bEnableOverride = null)
        {
            m_bEnabled = bEnableOverride.GetValueOrDefault(p.enabled);
            m_log = log;
            m_method = p.method;
            m_dfPositiveShiftMultiplier = p.pos_shift_mult;
        }

        /// <summary>
        /// Specifies whether or not the normalization is enabled.
        /// </summary>
        public bool enabled
        {
            get { return m_bEnabled; }
            set { m_bEnabled = value; }
        }

        /// <summary>
        /// If available, returns the score mean set when 'enabled' = true.
        /// </summary>
        public float? score_mean
        {
            get { return m_fScoreMean; }
        }

        /// <summary>
        /// If available, returns the score stdev set when 'enabled' = true.
        /// </summary>
        public float? score_stdev
        {
            get { return m_fScoreStdev; }
        }

        /// <summary>
        /// If available, returns the pos score mean set when 'enabled' = true.
        /// </summary>
        public float? pos_score_mean
        {
            get { return m_fScoreMean; }
        }

        /// <summary>
        /// If available, returns the pos score stdev set when 'enabled' = true.
        /// </summary>
        public float? pos_score_stdev
        {
            get { return m_fScoreStdev; }
        }

        /// <summary>
        /// If available, returns the neg score mean set when 'enabled' = true.
        /// </summary>
        public float? neg_score_mean
        {
            get { return m_fNegScoreMean; }
        }

        /// <summary>
        /// If available, returns the neg score stdev set when 'enabled' = true.
        /// </summary>
        public float? neg_score_stdev
        {
            get { return m_fNegScoreStdev; }
        }


        /// <summary>
        /// Specifies the Setup method used to setup the normalization.
        /// </summary>
        /// <param name="db">Specifies the database used to query mean and other dataset properties used during normalization.</param>
        /// <param name="strSourceName">Specifies the source data to use.</param>
        /// <param name="lblType">Specifies the label type that indicates whether to use SCORE1 or SCORE2 as the primary score.</param>
        public void Setup(IXDatabaseBase db, string strSourceName, LayerParameterBase.LABEL_TYPE lblType)
        {
            if (lblType != LayerParameterBase.LABEL_TYPE.SCORE1 && lblType != LayerParameterBase.LABEL_TYPE.SCORE2)
            {
                m_log.WriteLine("WARNING: Score based normalization disabled as label type '" + m_labelType.ToString() + "' is not supported.");
                m_bEnabled = false;
            }

            m_labelType = lblType;
            m_nSourceID = db.GetSourceID(strSourceName);

            string strActiveScore = (lblType == LayerParameterBase.LABEL_TYPE.SCORE2) ? "2" : "";
            string strMean = "Mean" + strActiveScore;
            string strStdDev = "StdDev" + strActiveScore;
            string strNegMean = "MeanNeg" + strActiveScore;
            string strNegStdDev = "StdDevNeg" + strActiveScore;
            SimpleDatum sdMean = db.GetItemMean(m_nSourceID, strMean, strStdDev, strNegMean, strNegStdDev);

            if (sdMean != null)
            {
                m_fScoreMean = sdMean.GetParameter(strMean);
                m_fScoreStdev = sdMean.GetParameter(strStdDev);
                m_fNegScoreMean = sdMean.GetParameter(strNegMean);
                m_fNegScoreStdev = sdMean.GetParameter(strNegStdDev);
            }

            if (m_method == SCORE_AS_LABEL_NORMALIZATION.Z_SCORE ||
                m_method == SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG)
            {
                if (!m_fScoreMean.HasValue)
                    m_log.FAIL("The score mean '" + strMean + "' was not found!");

                if (!m_fScoreStdev.HasValue)
                    m_log.FAIL("The score stdev '" + strStdDev + "' was not found!");

                if (m_method == SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG)
                {
                    if (!m_fNegScoreMean.HasValue)
                        m_log.FAIL("The score mean '" + strNegMean + "' was not found!");

                    if (!m_fNegScoreStdev.HasValue)
                        m_log.FAIL("The score stdev '" + strNegStdDev + "' was not found!");
                }
            }
        }

        /// <summary>
        /// Unnormalize the score based label - only applies when 'enabled' = true.
        /// </summary>
        /// <param name="fVal">Specifies the normalized score based label.</param>
        /// <returns>The un-normalized score is returned.</returns>
        public float UnNormalize(float fVal)
        {
            switch (m_method)
            {
                case SCORE_AS_LABEL_NORMALIZATION.Z_SCORE:
                    fVal *= m_fScoreStdev.Value;
                    fVal += m_fScoreMean.Value;
                    break;

                case SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG:
                    if (fVal >= 0)
                    {
                        fVal *= m_fScoreStdev.Value;
                        fVal += m_fScoreMean.Value;
                    }
                    else
                    {
                        fVal *= m_fNegScoreStdev.Value;
                        fVal += m_fNegScoreMean.Value;
                    }
                    break;

                case SCORE_AS_LABEL_NORMALIZATION.POS_SHIFT:
                    fVal /= (float)m_dfPositiveShiftMultiplier;
                    fVal -= 1.0f;
                    break;

                default:
                    return fVal;
            }

            return fVal;
        }

        /// <summary>
        /// Normalize the score and return the normalized returns based on the normalization method used.
        /// </summary>
        /// <param name="datum">Specifies the datum containing the scores.</param>
        /// <returns>The normalized score is returned.</returns>
        public float Normalize(SimpleDatum datum)
        {
            decimal? dScore = (m_labelType == DataParameter.LABEL_TYPE.SCORE1) ? datum.Score : datum.Score2;
            if (!dScore.HasValue)
                m_log.FAIL("The score value '" + m_labelType.ToString() + "' is not set!");

            if (!m_bEnabled)
                return (float)dScore.Value;

            float fVal = (float)Convert.ChangeType(dScore.Value, typeof(float));

            return Normalize(fVal);
        }

        /// <summary>
        /// Normalize the score and return the normalized returns based on the normalization method used.
        /// </summary>
        /// <param name="fVal">Specifies the raw value to normalize.</param>
        /// <returns>The normalized score is returned.</returns>
        public float Normalize(float fVal)
        {
            if (!m_bEnabled)
                return fVal;

            switch (m_method)
            {
                case SCORE_AS_LABEL_NORMALIZATION.Z_SCORE:
                    if (!m_fScoreMean.HasValue || m_fScoreStdev.HasValue)
                        m_log.WriteLine("WARNING: missing 'Mean' and/or 'StdDev' needed for Z_SCORE normalization - no normalization taking place.");
                    else
                    {
                        fVal -= m_fScoreMean.Value;
                        fVal /= m_fScoreStdev.Value;
                    }
                    break;

                case SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG:
                    if (!m_fScoreMean.HasValue || m_fScoreStdev.HasValue)
                        m_log.WriteLine("WARNING: missing 'Mean' and/or 'StdDev' needed for Z_SCORE_POSNEG normalization - no normalization taking place.");
                    else if (!m_fNegScoreMean.HasValue || !m_fNegScoreStdev.HasValue)
                        m_log.WriteLine("WARNING: missing 'NegMean' and/or 'NegStdDev' needed for Z_SCORE_POSNEG normalization - no normalization taking place.");
                    else
                    {
                        if (fVal > 0 && m_fScoreMean.HasValue && m_fScoreStdev.HasValue && m_fScoreStdev.Value != 0)
                        {
                            fVal -= m_fScoreMean.Value;
                            fVal /= m_fScoreStdev.Value;
                            // Ensure all positive values.
                            if (fVal < 0)
                                fVal = 0;
                        }
                        if (fVal < 0 && m_fNegScoreMean.HasValue && m_fNegScoreStdev.HasValue && m_fNegScoreStdev.Value != 0)
                        {
                            fVal -= m_fNegScoreMean.Value;
                            fVal /= m_fNegScoreStdev.Value;
                            // Ensure all negative values.
                            if (fVal > 0)
                                fVal = 0;
                        }
                    }
                    break;

                case SCORE_AS_LABEL_NORMALIZATION.POS_SHIFT:
                    fVal = (fVal + 1) * (float)m_dfPositiveShiftMultiplier;
                    break;
            }

            return fVal;
        }
    }
}
