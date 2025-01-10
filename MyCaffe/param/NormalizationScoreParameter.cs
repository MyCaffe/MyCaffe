using MyCaffe.basecode;
using MyCaffe.db.image;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.param.LayerParameterBase;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the normalization object responsible for normalizing score values.
    /// </summary>
    public class NormalizationScoreParameter
    {
        bool m_bEnabled = false;
        SCORE_AS_LABEL_NORMALIZATION m_method = SCORE_AS_LABEL_NORMALIZATION.NONE;
        double m_dfPositiveShiftMultiplier = 100.0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bEnabled">Specifies whether the normalization is enabled or not.</param>
        /// <param name="method">Specifies the normalization method.</param>
        /// <param name="dfPosShift">Specifies the positive shift multiplier, only used in POS_SHIFT normalization.</param>
        public NormalizationScoreParameter(bool bEnabled = false, SCORE_AS_LABEL_NORMALIZATION method = SCORE_AS_LABEL_NORMALIZATION.NONE, double dfPosShift = 100.0)
        {
            m_bEnabled = bEnabled;
            m_method = method;
            m_dfPositiveShiftMultiplier = dfPosShift;
        }

        /// <summary>
        /// Copy the parameters and return the new parameter object.
        /// </summary>
        /// <returns>The new normalization parameter is returned.</returns>
        public NormalizationScoreParameter Clone()
        {
            return new NormalizationScoreParameter(enabled, method, pos_shift_mult);
        }

        /// <summary>
        /// Specifies whether or not the normalization is enabled.
        /// </summary>
        [Category("Label"), Description("Specifies whether or not the normalization is enabled.")]
        public bool enabled
        {
            get { return m_bEnabled; }
            set { m_bEnabled = value; }
        }

        /// <summary>
        /// When enabled, score as label normalization is attempted (default = false).
        /// </summary>
        /// <remarks>
        /// Score as label normalization requires that a Mean image exist in the dataset with the following
        /// image parameters set in the database.
        /// Z_SCORE
        ///     'Mean' - specifies the mean score.
        ///     'StdDev' - specifies the standard deviation of the score.
        /// Z_SCORE_POSNEG
        ///     'PosMean' - specifies the mean score used with positive numbers.
        ///     'PosStdDev' - specifies the stddev score used with positive numbers.
        ///     'NegMean' - specifies the mean score used with negative numbers.
        ///     'NegStdDev' - specifies the stddev score used with negative numbers.
        /// During normalization, these values are used to perform Z-score normalization where the mean score is
        /// subtracted from each score then divided by the score standard deviation.
        /// 
        /// If these parameters or the mean image do not exist, a warning is produced and no normalization
        /// takes place.
        /// </remarks>
        [Category("Labels"), Description("When enabled, score as label normalization is run using z-score normalization method specified (default = NONE).")]
        public SCORE_AS_LABEL_NORMALIZATION method
        {
            get { return m_method; }
            set { m_method = value; }
        }

        /// <summary>
        /// Specifies the positive shift multiplier used when running the SCORE_AS_LABEL_NORMALIZATION.POS_SHIFT
        /// </summary>
        /// <remarks>
        /// Normalization function:
        /// @f$ y = beta * (x + 1.0) @f$, beta default = 100
        /// </remarks>
        [Category("Labels"), Description("When using the POS_SHIFT label normalization, the mutlipier defines the Beta values in the normalization function x1 = Beta * (x + 1)")]
        public double pos_shift_mult
        {
            get { return m_dfPositiveShiftMultiplier; }
            set { m_dfPositiveShiftMultiplier = value; }
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("enabled", enabled.ToString());
            rgChildren.Add("method", method.ToString());
            rgChildren.Add("pos_shift_mult", pos_shift_mult.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static NormalizationScoreParameter FromProto(RawProto rp, NormalizationScoreParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new NormalizationScoreParameter();

            if ((strVal = rp.FindValue("enabled")) != null)
                p.enabled = bool.Parse(strVal);

            if ((strVal = rp.FindValue("method")) != null)
            {
                switch (strVal)
                {
                    case "NONE":
                        p.method = SCORE_AS_LABEL_NORMALIZATION.NONE;
                        break;

                    case "Z_SCORE":
                        p.method = SCORE_AS_LABEL_NORMALIZATION.Z_SCORE;
                        break;

                    case "Z_SCORE_POSNEG":
                        p.method = SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG;
                        break;

                    case "POS_SHIFT":
                        p.method = SCORE_AS_LABEL_NORMALIZATION.POS_SHIFT;
                        break;

                    default:
                        throw new Exception("Unknown 'score_as_label_normalization' value " + strVal);
                }
            }

            if ((strVal = rp.FindValue("pos_shift_mult")) != null)
                p.pos_shift_mult = double.Parse(strVal);

            return p;
        }
    }
}
