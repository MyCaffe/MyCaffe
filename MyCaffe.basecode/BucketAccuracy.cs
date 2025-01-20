using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.ConfusionMatrixStats;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The BucketAccuracy calculates the accuracy using two parallel rolling bucket collections, one for ground truth and the other for correct counts.
    /// </summary>
    public class BucketAccuracy
    {
        float m_fIgnoreMin = -float.MaxValue;
        float m_fIgnoreMax = float.MaxValue;
        RollingBucketCollection m_buckets = null;
        RollingBucketCollection m_bucketsCorrect = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="dfMin">Specifies the minimum value covered by the bucket range.</param>
        /// <param name="dfMax">Specifies the maximim value covered by the bucket range.</param>
        /// <param name="nBucketCount">Specifies the bucker count.</param>
        /// <param name="nHistoryCount">Optionally, specifies the history of rolling buckets used to monitor the most recent accuracies in the history (default = 100).</param>
        /// <param name="colOverride">Optionally, specifies a bucker range override.</param>
        /// <param name="fIgnoreMin">Optionally, specifies a value to ignore all values below (default = null).</param>
        /// <param name="fIgnoreMax">Optionally, specifies a value to ignore all values above (default = null).</param>
        public BucketAccuracy(double dfMin, double dfMax, int nBucketCount, int nHistoryCount = 100, BucketCollection colOverride = null, float? fIgnoreMin = null, float? fIgnoreMax = null)
        {
            if (fIgnoreMin.HasValue)
                m_fIgnoreMin = fIgnoreMin.Value;
            if (fIgnoreMax.HasValue)
                m_fIgnoreMax = fIgnoreMax.Value;

            m_buckets = new RollingBucketCollection(dfMin, dfMax, nBucketCount, nHistoryCount, colOverride);
            m_bucketsCorrect = new RollingBucketCollection(dfMin, dfMax, nBucketCount, nHistoryCount, colOverride);
        }

        /// <summary>
        /// Returns the most recent GroundTruth counts in the history.
        /// </summary>
        public BucketCollection GroundTruth
        {
            get { return m_buckets.Current; }
        }

        /// <summary>
        /// Returns the most recent Correct counts in the history.
        /// </summary>
        public BucketCollection Correct
        {
            get { return m_bucketsCorrect.Current; }
        }

        /// <summary>
        /// Adds a set of predicted and ground truth values all with optional returns.
        /// </summary>
        /// <param name="fPred">Specifies the predicted value.</param>
        /// <param name="fTarget">Specifies the target ground truth value.</param>
        /// <param name="rgReturns">Optionally, specifies returns associated with the ground truth value.</param>
        public void Add(float fPred, float fTarget, List<float> rgReturns = null)
        {
            if (fPred < m_fIgnoreMin || fPred > m_fIgnoreMax)
                return;

            m_buckets.AddCollection();
            m_bucketsCorrect.AddCollection();

            int nIdxTgt = m_buckets.Add(fTarget, false, rgReturns);
            int nIdxPred = m_bucketsCorrect.Add(fPred, true);

            if (nIdxTgt == nIdxPred)
                m_bucketsCorrect.Add(fPred, false, rgReturns);
        }

        /// <summary>
        /// Adds a set of predicted and ground truth values all with optional returns.
        /// </summary>
        /// <param name="rgPred">Specifies the predicted values.</param>
        /// <param name="rgTarget">Specifies the target ground truth values.</param>
        /// <param name="rgrgReturns">Optionally, specifies returns associated with the ground truth values.</param>
        public void Add(float[] rgPred, float[] rgTarget, List<List<float>> rgrgReturns = null)
        {
            int nCount = rgTarget.Length;

            m_buckets.AddCollection();
            m_bucketsCorrect.AddCollection();

            // Add values to buckets
            for (int i = 0; i < nCount; i++)
            {
                float fPred = rgPred[i];

                if (fPred < m_fIgnoreMin || fPred > m_fIgnoreMax)
                    continue;

                float fTarget = rgTarget[i];
                List<float> rgReturns = null;
                
                if (rgrgReturns != null)
                    rgReturns = rgrgReturns[i];

                int nIdxTgt = m_buckets.Add(fTarget, false, rgReturns);
                int nIdxPred = m_bucketsCorrect.Add(fPred, true);

                if (nIdxTgt == nIdxPred)
                    m_bucketsCorrect.Add(fPred, false, rgReturns);
            }
        }

        /// <summary>
        /// Calculate the accuracy and optionally specify the details such as the confusion matrix.
        /// </summary>
        /// <param name="bGetDetails">Specifies to return details including the confusion matrix.</param>
        /// <param name="strDetails">When 'bGetDetails' is true, the details are returned here.</param>
        /// <param name="rgstrReturnNames">Optionally, specifies a list of names for each of the returns.</param>
        /// <returns>The accuracy value is returned.</returns>
        public float CalculateAccuracy(bool bGetDetails, out string strDetails, List<string> rgstrReturnNames = null)
        {
            strDetails = "";
            BucketCollection colTotals = m_buckets.Current;
            BucketCollection colCorrect = m_bucketsCorrect.Current;

            if (bGetDetails)
            {
                StringBuilder sb = new StringBuilder();

                for (int i = 0; i < colTotals.Count; i++)
                {
                    Bucket bTotal = colTotals[i];
                    Bucket bCorrect = colCorrect[i];
                    float fAcc = (bTotal.Count == 0) ? 0 : (float)bCorrect.Count / bTotal.Count;
                    
                    sb.Append("Bucket: ");
                    sb.Append(bTotal.ToString());
                    sb.Append(" Accuracy: ");
                    sb.Append(fAcc.ToString("P2"));
                    sb.AppendLine();
                }

                ConfusionMatrix cmtx = new ConfusionMatrix(this);
                sb.Append(cmtx.CreateConfusionMatrix(rgstrReturnNames));
                strDetails = sb.ToString();
            }

            int nTotalCount = colTotals.TotalCount;
            int nCorrectCount = colCorrect.TotalCount;

            return (float)nCorrectCount / (float)nTotalCount;
        }
    }
}
