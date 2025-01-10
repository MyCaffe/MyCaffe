using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.ConfusionMatrixStats;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The RollingBucketAccuracy provides rolling accuracy.
    /// </summary>
    public class RollingBucketAccuracy
    {
        double m_dfMin;
        double m_dfMax;
        int m_nCount;
        int m_nIteration = 0;
        int m_nMinIterations;
        int m_nMaxIterations;
        double? m_dfIgnoreMax;
        double? m_dfIgnoreMin;
        List<BucketAccuracy> m_rgItems = new List<BucketAccuracy>();
        BucketCollection m_colOverride = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colOverride">Optional bucket collection override that when specified is used instead of the other fixed buckets.</param>
        /// <param name="dfMin">Specifies the minimum value.</param>
        /// <param name="dfMax">Specifies the maximum value.</param>
        /// <param name="nCount">Specfies the number of buckets.</param>
        /// <param name="nMinIterations">Specifies the minimum number of iterations.</param>
        /// <param name="nMaxIterations">Specifies the maximum number of iterations.</param>
        /// <param name="dfIgnoreMin">Specifies the minimum ignore value.</param>
        /// <param name="dfIgnoreMax">Specifies the maximum ignore value.</param>
        public RollingBucketAccuracy(BucketCollection colOverride, double dfMin, double dfMax, int nCount, int nMinIterations, int nMaxIterations, double? dfIgnoreMin, double? dfIgnoreMax)
        {
            m_colOverride = colOverride;
            m_dfMin = dfMin;
            m_dfMax = dfMax;
            m_nCount = nCount;
            m_nIteration = 0;
            m_nMinIterations = nMinIterations;
            m_nMaxIterations = nMaxIterations;
            m_dfIgnoreMax = dfIgnoreMax;
            m_dfIgnoreMin = dfIgnoreMin;

            m_rgItems.Add(new BucketAccuracy(colOverride, dfMin, dfMax, nCount, dfIgnoreMin, dfIgnoreMax));
        }

        /// <summary>
        /// Add a set of predictions and target values.
        /// </summary>
        /// <param name="rgPred">Specifies the predictions.</param>
        /// <param name="rgTgt">Specifies the targets.</param>
        public void Add(float[] rgPred, float[] rgTgt)
        {
            m_nIteration++;

            BucketAccuracy b = new BucketAccuracy(m_colOverride, m_dfMin, m_dfMax, m_nCount, m_dfIgnoreMax, m_dfIgnoreMin);
            m_rgItems.Add(b);

            foreach (BucketAccuracy b1 in m_rgItems)
            {
                b1.Add(rgPred, rgTgt, null);
            }

            if (m_nIteration >= m_nMaxIterations)
                m_rgItems.RemoveAt(0);
        }

        /// <summary>
        /// Calculates the accuracy.
        /// </summary>
        /// <returns>The accuracy value is returned.</returns>
        public double CalculateAccuracy()
        {
            if (m_rgItems.Count == 0 || m_nIteration < m_nMinIterations)
                return 0;

            string strTmp;
            return m_rgItems[0].CalculateAccuracy(false, out strTmp);
        }
    }

    /// <summary>
    /// The BucketAccuracy layer tracks the accuracy across both positive and negative bucket collections between the target and predicted values.
    /// </summary>
    public class BucketAccuracy
    {
        BucketCollection m_colPredPos = null;
        BucketCollection m_colPredNeg = null;
        BucketCollection m_colTgtPos = null;
        BucketCollection m_colTgtNeg = null;
        BucketCollection m_colCorrectPredPos = null;
        BucketCollection m_colCorrectPredNeg = null;
        double? m_dfIgnoreMax = null;
        double? m_dfIgnoreMin = null;
        BucketCollection m_colOverride = null;
        float[] m_rgPred = new float[1];
        float[] m_rgTgt = new float[1];
        float[] m_rgReturn = new float[1];

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colOverride">Optional bucket collection override that when specified is used instead of the other fixed buckets.</param>
        /// <param name="dfMin">Specifies the minimum of all values.</param>
        /// <param name="dfMax">Specifies the maximum of all values.</param>
        /// <param name="nCount">Specifies the number of buckets.</param>
        /// <param name="dfIgnoreMin">Specifies to the minimum ignore range (default = -double.MaxValue).</param>
        /// <param name="dfIgnoreMax">Specifies to the maximum ignore range (default = double.MaxValue).</param>
        public BucketAccuracy(BucketCollection colOverride, double dfMin, double dfMax, int nCount, double? dfIgnoreMin, double? dfIgnoreMax)
        {
            m_colOverride = colOverride;
            m_dfIgnoreMax = dfIgnoreMax;
            m_dfIgnoreMin = dfIgnoreMin;

            if (m_colOverride != null)
            {
                string strCfg = m_colOverride.ToConfigString();
                string[] rgstr = strCfg.Split(';');
                List<string> rgstrNeg = new List<string>();
                List<string> rgstrPos = new List<string>();

                for (int i = 1; i < rgstr.Length; i++)
                {
                    if (string.IsNullOrEmpty(rgstr[i]))
                        continue;

                    string str = rgstr[i].Trim('[', ']');
                    string[] rgstr1 = str.Split(',');
                    double dfMin1 = double.Parse(rgstr1[0]);
                    double dfMax1 = double.Parse(rgstr1[1]);

                    if (dfMin1 < 0)
                        rgstrNeg.Add(rgstr[i]);
                    else
                        rgstrPos.Add(rgstr[i]);
                }

                string strNegConfig = "Count=" + rgstrNeg.Count.ToString() + ";";
                for (int i = 0; i < rgstrNeg.Count; i++)
                {
                    strNegConfig += rgstrNeg[i];
                    if (i < rgstrNeg.Count - 1)
                        strNegConfig += ";";
                }

                string strPosConfig = "Count=" + rgstrPos.Count.ToString() + ";";
                for (int i = 0; i < rgstrPos.Count; i++)
                {
                    strPosConfig += rgstrPos[i];
                    if (i < rgstrPos.Count - 1)
                        strPosConfig += ";";
                }

                m_colPredNeg = new BucketCollection(strNegConfig);
                m_colCorrectPredNeg = new BucketCollection(strNegConfig);
                m_colTgtNeg = new BucketCollection(strNegConfig);
                m_colPredPos = new BucketCollection(strPosConfig);
                m_colCorrectPredPos = new BucketCollection(strPosConfig);
                m_colTgtPos = new BucketCollection(strPosConfig);
            }
            else
            {
                int nBucketCount = nCount;
                if (dfMin < 0)
                {
                    nBucketCount /= 2;
                    m_colPredNeg = new BucketCollection(dfMin, 0, nBucketCount);
                    m_colCorrectPredNeg = new BucketCollection(dfMin, 0, nBucketCount);
                    m_colTgtNeg = new BucketCollection(dfMin, 0, nBucketCount);
                }

                m_colPredPos = new BucketCollection(0, dfMax, nBucketCount);
                m_colCorrectPredPos = new BucketCollection(0, dfMax, nBucketCount);
                m_colTgtPos = new BucketCollection(0, dfMax, nBucketCount);
            }
        }

        /// <summary>
        /// Add an array of predicted and target values.
        /// </summary>
        /// <param name="rgPred">Specifies the predicted values.</param>
        /// <param name="rgTgt">Specifies the target values.</param>
        /// <param name="rgrgReturn">Optionally, specifies the return values.</param>
        public int Add(float[] rgPred, float[] rgTgt, List<List<float>> rgrgReturn = null)
        {
            int nPredIdxNeg = -1;
            int nPredIdxPos = -1;

            for (int i = 0; i < rgPred.Length; i++)
            {
                int nTgtIdxNeg = -1;
                int nTgtIdxPos = -1;
                nPredIdxNeg = -1;
                nPredIdxPos = -1;

                List<float> rgReturns = null;
                if (rgrgReturn != null && rgrgReturn.Count > i)
                    rgReturns = rgrgReturn[i];

                if (m_dfIgnoreMin.HasValue && m_dfIgnoreMax.HasValue)
                {
                    if (rgPred[i] > m_dfIgnoreMin.Value && rgPred[i] < m_dfIgnoreMax.Value)
                        continue;
                }

                if (rgTgt[i] < 0 && m_colTgtNeg != null)
                    nTgtIdxNeg = m_colTgtNeg.Add(rgTgt[i], false, rgReturns);
                else
                    nTgtIdxPos = m_colTgtPos.Add(rgTgt[i], false, rgReturns);

                if (rgPred[i] < 0 && m_colPredNeg != null)
                    nPredIdxNeg = m_colPredNeg.Add(rgPred[i], false, rgReturns);
                else
                    nPredIdxPos = m_colPredPos.Add(rgPred[i], false, rgReturns);

                if (m_colCorrectPredNeg != null && nTgtIdxNeg >= 0 && nPredIdxNeg == nTgtIdxNeg)
                    m_colCorrectPredNeg.Add(rgPred[i], false, rgReturns);

                if (m_colCorrectPredPos != null && nTgtIdxPos >= 0 && nPredIdxPos == nTgtIdxPos)
                    m_colCorrectPredPos.Add(rgPred[i], false, rgReturns);
            }

            if (nPredIdxNeg >= 0)
                return -nPredIdxNeg;
            else if (nPredIdxPos >= 0)
                return nPredIdxPos;
            else
                return 0;
        }

        /// <summary>
        /// Add a single predicted and target value.
        /// </summary>
        /// <param name="fPred">Specifies the predicted value.</param>
        /// <param name="fTgt">Specifies the target value.</param>
        /// <param name="rgfReturn">Optionally, specifies a list of return values.</param>
        public int Add(float fPred, float fTgt, List<float> rgfReturn)
        {
            m_rgPred[0] = fPred;
            m_rgTgt[0] = fTgt;
            List<List<float>> rgrgReturns = null;

            if (rgfReturn != null && rgfReturn.Count > 0)
            {
                rgrgReturns = new List<List<float>>();
                rgrgReturns.Add(rgfReturn);
            }

            return Add(m_rgPred, m_rgTgt, rgrgReturns);
        }

        /// <summary>
        /// Calculates the overall accuracy.
        /// </summary>
        /// <param name="bGetDetails">Specifies to fill out the details string.</param>
        /// <param name="strDetails">Specifies the string to receive the details, when specified.</param>
        /// <param name="rgstrReturnNames">Optionally, specifies names for the returns.</param>
        /// <returns>The accuracy is returned.</returns>
        public double CalculateAccuracy(bool bGetDetails, out string strDetails, List<string> rgstrReturnNames = null)
        {
            strDetails = (bGetDetails) ? "" : null;

            int nTotalCorrectPos = (m_colCorrectPredPos != null) ? m_colCorrectPredPos.BucketCountSum : 0;
            int nTotalCorrectNeg = (m_colCorrectPredNeg != null) ? m_colCorrectPredNeg.BucketCountSum : 0;
            int nTotalPos = (m_colTgtPos != null) ? m_colTgtPos.BucketCountSum : 0;
            int nTotalNeg = (m_colTgtNeg != null) ? m_colTgtNeg.BucketCountSum : 0;

            int nTotalCorrect = nTotalCorrectPos + nTotalCorrectNeg;
            int nTotalPredictions = nTotalPos + nTotalNeg;

            if (bGetDetails)
            {
                if (m_colTgtNeg != null)
                {
                    for (int i = 0; i < m_colTgtNeg.Count; i++)
                    {
                        Bucket bTgt = m_colTgtNeg[i];
                        Bucket bPred = m_colCorrectPredNeg[i];
                        double dfAcc = (double)bPred.Count / (double)bTgt.Count;

                        strDetails += "Bucket: " + m_colTgtNeg[i].ToString() + " Accuracy: " + dfAcc.ToString("P2") + Environment.NewLine;
                    }
                }

                if (m_colTgtPos != null)
                {
                    for (int i = 0; i < m_colTgtPos.Count; i++)
                    {
                        Bucket bTgt = m_colTgtPos[i];
                        Bucket bPred = m_colCorrectPredPos[i];
                        double dfAcc = (double)bPred.Count / (double)bTgt.Count;

                        strDetails += "Bucket: " + m_colTgtPos[i].ToString() + " Accuracy: " + dfAcc.ToString("P2") + Environment.NewLine;
                    }
                }

                strDetails += CreateConfusionMatrix(rgstrReturnNames);
            }

            // Calculate accuracy as a percentage
            double accuracy = nTotalPredictions > 0 ? (nTotalCorrect / (double)nTotalPredictions) : 0;
            return accuracy;
        }

        private int[,] createConfusionMatrix(List<string> rgstrTargetLabels, List<string> rgstrPredLabels, List<Bucket> rgTargets, List<Bucket> rgPredicted)
        {
            int[,] confusionMatrix = new int[rgstrTargetLabels.Count, rgstrPredLabels.Count];

            for (int i = 0; i < rgstrTargetLabels.Count; i++)
            {
                Bucket bTarget = rgTargets[i]; // Current target bucket

                for (int j = 0; j < rgstrPredLabels.Count; j++)
                {
                    int nVal = 0;
                    Bucket bPredicted = rgPredicted[i];

                    if (i == j)
                    {
                        nVal = bPredicted.Count;
                    }
                    else
                    {
                        int nTargetCount = bTarget.Count;
                        int nPredictedCount = bPredicted.Count;
                        nVal = nTargetCount - nPredictedCount;
                    }

                    confusionMatrix[i, j] = nVal;
                }
            }

            return confusionMatrix;
        }

        private double[,] createReturnConfusionMatrix(List<string> rgstrTargetLabels, List<string> rgstrPredLabels, List<Bucket> rgTgt, List<Bucket> rgPred, int nReturnIdx)
        {
            double[,] confusionMatrix = new double[rgstrTargetLabels.Count, rgstrPredLabels.Count];

            for (int i = 0; i < rgstrTargetLabels.Count; i++)
            {
                Bucket bTarget = rgTgt[i]; // Current target bucket
                double? dfAveTgtReturn = bTarget.AveReturns(nReturnIdx);

                for (int j = 0; j < rgstrPredLabels.Count; j++)
                {
                    Bucket bPred = rgPred[j]; // Current predicted bucket
                    double? dfAvePredReturn = bPred.AveReturns(nReturnIdx);

                    if (i == j)
                        confusionMatrix[i, j] = dfAvePredReturn.GetValueOrDefault(0);
                    else
                        confusionMatrix[i, j] = dfAveTgtReturn.GetValueOrDefault(0) - dfAvePredReturn.GetValueOrDefault(0);
                }
            }

            return confusionMatrix;
        }

        private double[,] createAccuracyPercentMatrix(List<string> rgstrTargetLabels, List<string> rgstrPredLabels, List<Bucket> rgTarget, int[,] confusionMatrix)
        {
            double[,] percentageMatrix = new double[rgstrTargetLabels.Count, rgstrPredLabels.Count];

            // Calculate percentages for the confusion matrix
            for (int i = 0; i < rgstrTargetLabels.Count; i++)
            {
                int total = 0;
                for (int j = 0; j < rgstrPredLabels.Count; j++)
                {
                    total += confusionMatrix[i, j];
                }

                for (int j = 0; j < rgstrPredLabels.Count; j++)
                {
                    percentageMatrix[i, j] = total > 0 ? (double)confusionMatrix[i, j] / total * 100 : 0;
                }
            }

            return percentageMatrix;
        }

        private double[,] createPrecisionPercentMatrix(List<string> rgstrTargetLabels, List<string> rgstrPredLabels, List<Bucket> rgTarget, int[,] confusionMatrix)
        {
            double[,] percentageMatrix = new double[rgstrTargetLabels.Count, rgstrPredLabels.Count];

            // Calculate percentages for the confusion matrix
            for (int j = 0; j < rgstrTargetLabels.Count; j++)
            {
                int total = 0;

                for (int i = 0; i < rgstrPredLabels.Count; i++)
                {
                    total += confusionMatrix[i, j];
                }

                for (int i = 0; i < rgstrPredLabels.Count; i++)
                {
                    percentageMatrix[i, j] = total > 0 ? (double)confusionMatrix[i, j] / total * 100 : 0;
                }
            }

            return percentageMatrix;
        }

        /// <summary>
        /// Prints a matrix with labels and values, formatted as counts or percentages.
        /// </summary>
        private void printMatrix(string strName, StringBuilder sb, string[] labels, dynamic matrix, int maxLabelWidth, int maxCellWidth, bool isPercentage)
        {
            string strActual = "actuals ";

            // Print the matrix title
            sb.Append("Confusion Matrix [rows = actuals, cols = predicted]");
            if (isPercentage)
                sb.Append(" (Percentages)");
            sb.AppendLine(":");

            // Print the predicted row with labels
            string headerBar = new string('▄', maxLabelWidth + strActual.Length);
            sb.Append(headerBar);
            foreach (string label in labels)
            {
                sb.Append($" | {"predicted".PadRight(maxCellWidth + 2)}");
            }
            sb.AppendLine();

            // Print the header row with name and underscores
            int nameStart = (maxLabelWidth + strActual.Length - strName.Length) / 2; // Center the name
            sb.Append(new string('_', nameStart))
              .Append(strName)
              .Append(new string('_', maxLabelWidth + strActual.Length - nameStart - strName.Length));
            foreach (string label in labels)
            {
                sb.Append($" | {label.PadRight(maxCellWidth + 2)}");
            }
            sb.AppendLine();

            // Print each row of the matrix
            for (int i = 0; i < labels.Length; i++)
            {
                string rowLabel = $"{strActual}{labels[i]}";
                sb.Append(rowLabel.PadRight(maxLabelWidth + strActual.Length));

                for (int j = 0; j < labels.Length; j++)
                {
                    string value = isPercentage ? $"{matrix[i, j]:F2}%" : $"{matrix[i, j]}";
                    sb.Append($" | {value.PadRight(maxCellWidth + 2)}");
                }
                sb.AppendLine();
            }
        }


        /// <summary>
        /// Create a confusion matrix of the values.
        /// </summary>
        /// <param name="rgstrReturnNames">Optionally, specifies the names of the returns.</param>
        /// <returns>The confusion matrix is returned as a pretty-print string.</returns>
        public string CreateConfusionMatrix(List<string> rgstrReturnNames)
        {
            List<string> rgstrPredLabels = new List<string>();
            List<string> rgstrTargetLabels = new List<string>();
            List<Bucket> rgTargetLabels = new List<Bucket>();
            List<Bucket> rgPredLabels = new List<Bucket>();

            if (m_colTgtNeg != null)
            {
                foreach (Bucket b in m_colTgtNeg)
                {
                    rgstrTargetLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                    rgTargetLabels.Add(b);
                }
            }

            if (m_colPredNeg != null)
            {
                foreach (Bucket b in m_colPredNeg)
                {
                    rgstrPredLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                }
            }

            if (m_colCorrectPredNeg != null)
            {
                foreach (Bucket b in m_colCorrectPredNeg)
                {
                    rgPredLabels.Add(b);
                }
            }

            if (m_colTgtPos != null)
            {
                foreach (Bucket b in m_colTgtPos)
                {
                    rgstrTargetLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                    rgTargetLabels.Add(b);
                }
            }

            if (m_colPredPos != null)
            {
                foreach (Bucket b in m_colPredPos)
                {
                    rgstrPredLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                }
            }

            if (m_colCorrectPredPos != null)
            {
                foreach (Bucket b in m_colCorrectPredPos)
                {
                    rgPredLabels.Add(b);
                }
            }

            // Create the confusion matrix
            int[,] confusionMatrix = createConfusionMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, rgPredLabels);
            double[,] accuracyPctMatrix = createAccuracyPercentMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, confusionMatrix);
            double[,] precisionPctMatrix = createPrecisionPercentMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, confusionMatrix);
            StringBuilder sb = new StringBuilder();

            string[] labels = rgstrTargetLabels.ToArray();
            int maxLabelWidth = labels.Max(label => label.Length);
            int maxCellWidth = 10;

            sb.AppendLine("=================================");
            sb.AppendLine("CONFUSION MATRIX");
            printMatrix("Counts", sb, labels, confusionMatrix, maxLabelWidth, maxCellWidth, false);
            sb.AppendLine();
            printMatrix("Accuracy", sb, labels, accuracyPctMatrix, maxLabelWidth, maxCellWidth, true);
            sb.AppendLine();
            printMatrix("Precision", sb, labels, precisionPctMatrix, maxLabelWidth, maxCellWidth, true);

            if (rgstrReturnNames != null)
            {
                for (int i = 0; i < rgstrReturnNames.Count; i++)
                {
                    double[,] returnMatrix = createReturnConfusionMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, rgPredLabels, i);
                    sb.AppendLine();
                    printMatrix(rgstrReturnNames[i], sb, labels, returnMatrix, maxLabelWidth, maxCellWidth, true);
                }
            }

            int nTotalPos = 0;
            int nTotalNeg = 0;

            if (m_colTgtPos != null && m_colTgtPos.Count > 0)
                nTotalPos = m_colTgtPos.Sum(p => p.Count);

            if (m_colTgtNeg != null && m_colTgtNeg.Count > 0)
                nTotalNeg = m_colTgtNeg.Sum(p => p.Count);

            int nTotal = nTotalPos + nTotalNeg;
            double dfGtPercentPos = (double)nTotalPos / nTotal;
            double dfGtPercentNeg = (double)nTotalNeg / nTotal;
            sb.AppendLine();
            sb.AppendLine("Ground Truth Sample Size = " + nTotal.ToString("N0"));
            sb.AppendLine("Ground Truth % Positive = " + dfGtPercentPos.ToString("P2"));
            sb.AppendLine("Ground Truth % Negative = " + dfGtPercentNeg.ToString("P2"));

            StatisticalResults stats = ConfusionMatrixStats.Analyze(confusionMatrix);
            sb.AppendLine("---------------------------------");
            sb.AppendLine("Confusion Matrix Statistics:");
            sb.AppendLine(stats.ToString());

            return sb.ToString();
        }
    }
}
