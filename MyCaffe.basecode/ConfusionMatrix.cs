using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.ConfusionMatrixStats;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ConfusionMatrix creates a confusion matrix for counts, accuracy and precision and returns the results as a string.
    /// </summary>
    public class ConfusionMatrix
    {
        BucketAccuracy m_acc;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="col">Specifies the BucketAccruacy to calculate the confusion matrix for.</param>
        public ConfusionMatrix(BucketAccuracy col)
        {
            m_acc = col;
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

            foreach (Bucket b in m_acc.GroundTruth)
            {
                rgstrTargetLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                rgTargetLabels.Add(b);
            }

            foreach (Bucket b in m_acc.Correct)
            {
                rgstrPredLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                rgPredLabels.Add(b);
            }

            // Create the confusion matrix
            int[,] confusionMatrix = createConfusionMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, rgPredLabels);
            double[,] accuracyPctMatrix = createAccuracyPercentMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, confusionMatrix);
            double[,] precisionPctMatrix = createPrecisionPercentMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, confusionMatrix);

            string[] labels = rgstrTargetLabels.ToArray();
            int maxLabelWidth = labels.Max(label => label.Length);
            int maxCellWidth = 10;

            StringBuilder sb = new StringBuilder();
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
                    printMatrix(rgstrReturnNames[i], sb, labels, returnMatrix, maxLabelWidth, maxCellWidth, true, 3);
                }
            }

            sb.AppendLine();
            sb.AppendLine("Ground Truth Sample Size = " + m_acc.GroundTruth.TotalCount.ToString("N0"));

            StatisticalResults stats = ConfusionMatrixStats.Analyze(confusionMatrix);
            sb.AppendLine("---------------------------------");
            sb.AppendLine("Confusion Matrix Statistics:");
            sb.AppendLine(stats.ToString());

            return sb.ToString();
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
        private void printMatrix(string strName, StringBuilder sb, string[] labels, dynamic matrix, int maxLabelWidth, int maxCellWidth, bool isPercentage, int nPrecision = 2)
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
                    string value = isPercentage ? (nPrecision == 3) ? $"{matrix[i, j]:F3}%" : $"{matrix[i, j]:F2}%" : $"{matrix[i, j]}";
                    sb.Append($" | {value.PadRight(maxCellWidth + 2)}");
                }
                sb.AppendLine();
            }
        }
    }
}
