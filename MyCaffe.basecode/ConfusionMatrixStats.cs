using System;
using System.Linq;
using System.Collections.Generic;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ConfusionMatrixStats class contains methods to analyze a confusion matrix and calculate statistical significance
    /// </summary>
    public class ConfusionMatrixStats
    {
        /// <summary>
        /// The StatisticalResults class contains the results of the statistical analysis
        /// </summary>
        public class StatisticalResults
        {
            /// <summary>
            /// Get or set the chi-square statistic
            /// </summary>
            public double ChiSquareStatistic { get; set; }
            /// <summary>
            /// Get or set the p-value
            /// </summary>
            public double PValue { get; set; }
            /// <summary>
            /// Get or set the degrees of freedom
            /// </summary>
            public int DegreesOfFreedom { get; set; }
            /// <summary>
            /// Get or set whether the result is significant
            /// </summary>
            public bool IsSignificant { get; set; }
            /// <summary>
            /// Get or set the expected frequencies
            /// </summary>
            public double[,] ExpectedFrequencies { get; set; }
            /// <summary>
            /// Get or set the odds ratio
            /// </summary>
            public double? OddsRatio { get; set; }
            /// <summary>
            /// Get or set the lower confidence interval of the odds ratio
            /// </summary>
            public double? OddsRatioLowerCI { get; set; }
            /// <summary>
            /// Get or set the upper confidence interval of the odds ratio
            /// </summary>
            public double? OddsRatioUpperCI { get; set; }
            /// <summary>
            /// Get or set the test used
            /// </summary>
            public string TestUsed { get; set; }
            /// <summary>
            /// Get or set whether assumptions are met
            /// </summary>
            public bool AssumptionsMet { get; set; }
            /// <summary>
            /// Get or set the warnings
            /// </summary>
            public List<string> Warnings { get; set; }
            /// <summary>
            /// Get or set the Cramer's V value
            /// </summary>
            public double CramersV { get; set; }

            /// <summary>
            /// Output the results as a string
            /// </summary>
            /// <returns>The string is returned.</returns>
            public override string ToString()
            {
                var result = $"{TestUsed}:\n" +
                            $"Chi-square({DegreesOfFreedom}) = {ChiSquareStatistic:F4}\n" +
                            $"p-value = {PValue:G4}\n" +
                            $"Statistically Significant: {IsSignificant}\n" +
                            $"Cramer's V = {CramersV:F4}";

                if (OddsRatio.HasValue)
                {
                    result += $"\nOdds Ratio = {OddsRatio:F4} (95% CI: {OddsRatioLowerCI:F4} - {OddsRatioUpperCI:F4})";
                }

                if (Warnings.Any())
                {
                    result += "\nWarnings:\n" + string.Join("\n", Warnings);
                }

                return result;
            }
        }

        /// <summary>
        /// Calculates the statistical results for a confusion matrix
        /// </summary>
        /// <param name="matrix">Specifies the confusion matrix.</param>
        /// <param name="alpha">Specifies the alpha (default = 0.05)</param>
        /// <returns>The StatisticalResults calculated are returned.</returns>
        public static StatisticalResults Analyze(int[,] matrix, double alpha = 0.05)
        {
            ValidateInput(matrix, alpha);

            var results = new StatisticalResults
            {
                Warnings = new List<string>(),
                TestUsed = "Chi-square test"
            };

            // Calculate basic metrics
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            results.DegreesOfFreedom = (rows - 1) * (cols - 1);

            // Calculate totals and expected frequencies
            var (rowTotals, colTotals, total) = CalculateTotals(matrix);
            results.ExpectedFrequencies = CalculateExpectedFrequencies(rowTotals, colTotals, total, rows, cols);

            // Check assumptions
            results.AssumptionsMet = CheckAssumptions(matrix, results.ExpectedFrequencies, total, results.Warnings);

            // For 2x2 matrices with assumption violations, use Fisher's exact test
            if (rows == 2 && cols == 2 && !results.AssumptionsMet)
            {
                return CalculateFishersExact(matrix, alpha, results);
            }

            // Calculate chi-square statistic
            results.ChiSquareStatistic = CalculateChiSquare(matrix, results.ExpectedFrequencies);

            // Calculate p-value
            results.PValue = CalculateChiSquarePValue(results.ChiSquareStatistic, results.DegreesOfFreedom);
            results.IsSignificant = results.PValue < alpha;

            // Calculate Cramer's V
            results.CramersV = Math.Sqrt(results.ChiSquareStatistic / (total * Math.Min(rows - 1, cols - 1)));

            // Calculate odds ratio for 2x2 matrices
            if (rows == 2 && cols == 2)
            {
                (results.OddsRatio, results.OddsRatioLowerCI, results.OddsRatioUpperCI) =
                    CalculateOddsRatio(matrix);
            }

            return results;
        }

        private static void ValidateInput(int[,] matrix, double alpha)
        {
            if (matrix == null)
                throw new ArgumentNullException(nameof(matrix));

            if (matrix.GetLength(0) < 2 || matrix.GetLength(1) < 2)
                throw new ArgumentException("Matrix must be at least 2x2");

            if (matrix.Cast<int>().Any(x => x < 0))
                throw new ArgumentException("Matrix cannot contain negative values");

            if (alpha <= 0 || alpha >= 1)
                throw new ArgumentException("Alpha must be between 0 and 1");

            if (matrix.Cast<int>().Sum() == 0)
                throw new ArgumentException("Matrix cannot be empty (all zeros)");
        }

        private static (double[] rowTotals, double[] colTotals, double total)
            CalculateTotals(int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            double[] rowTotals = new double[rows];
            double[] colTotals = new double[cols];
            double total = 0;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    rowTotals[i] += matrix[i, j];
                    colTotals[j] += matrix[i, j];
                    total += matrix[i, j];
                }
            }

            return (rowTotals, colTotals, total);
        }

        private static double[,] CalculateExpectedFrequencies(
            double[] rowTotals, double[] colTotals, double total, int rows, int cols)
        {
            var expected = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    expected[i, j] = (rowTotals[i] * colTotals[j]) / total;
                }
            }

            return expected;
        }

        private static bool CheckAssumptions(
            int[,] observed, double[,] expected, double total, List<string> warnings)
        {
            bool assumptionsMet = true;

            if (total < 20)
            {
                warnings.Add("Sample size is small (N < 20)");
                assumptionsMet = false;
            }

            // Check expected frequencies
            int smallExpected = expected.Cast<double>().Count(e => e < 5);
            double percentSmall = (smallExpected * 100.0) / expected.Length;

            if (observed.GetLength(0) == 2 && observed.GetLength(1) == 2)
            {
                if (expected.Cast<double>().Any(e => e < 5))
                {
                    warnings.Add("Some expected frequencies are less than 5");
                    assumptionsMet = false;
                }
            }
            else if (percentSmall > 20)
            {
                warnings.Add($"{percentSmall:F1}% of expected frequencies are less than 5 (should be ≤ 20%)");
                assumptionsMet = false;
            }

            if (observed.Cast<int>().Any(o => o == 0))
            {
                warnings.Add("Matrix contains zero cells");
                assumptionsMet = false;
            }

            return assumptionsMet;
        }

        private static double CalculateChiSquare(int[,] observed, double[,] expected)
        {
            double chiSquare = 0;
            int rows = observed.GetLength(0);
            int cols = observed.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (expected[i, j] > 0)
                    {
                        chiSquare += Math.Pow(observed[i, j] - expected[i, j], 2) / expected[i, j];
                    }
                }
            }

            return chiSquare;
        }

        private static double CalculateChiSquarePValue(double chiSquare, int df)
        {
            if (chiSquare <= 0 || df < 1) return 1.0;
            return 1 - GammaP(df / 2.0, chiSquare / 2.0);
        }

        private static (double? oddsRatio, double? lowerCI, double? upperCI)
            CalculateOddsRatio(int[,] matrix)
        {
            // Apply Haldane-Anscombe correction
            double a = matrix[0, 0] + 0.5;
            double b = matrix[0, 1] + 0.5;
            double c = matrix[1, 0] + 0.5;
            double d = matrix[1, 1] + 0.5;

            if (b * c == 0) return (null, null, null);

            double or = (a * d) / (b * c);
            double logOr = Math.Log(or);
            double seLogOr = Math.Sqrt(1 / a + 1 / b + 1 / c + 1 / d);

            return (
                or,
                Math.Exp(logOr - 1.96 * seLogOr),
                Math.Exp(logOr + 1.96 * seLogOr)
            );
        }

        private static StatisticalResults CalculateFishersExact(
            int[,] matrix, double alpha, StatisticalResults results)
        {
            int a = matrix[0, 0];
            int b = matrix[0, 1];
            int c = matrix[1, 0];
            int d = matrix[1, 1];

            int r1 = a + b;
            int r2 = c + d;
            int c1 = a + c;
            int c2 = b + d;
            int n = r1 + r2;

            double logP = LogFactorial(r1) + LogFactorial(r2) + LogFactorial(c1) +
                         LogFactorial(c2) - LogFactorial(n) - LogFactorial(a) -
                         LogFactorial(b) - LogFactorial(c) - LogFactorial(d);

            results.PValue = Math.Exp(logP);
            results.IsSignificant = results.PValue < alpha;
            results.TestUsed = "Fisher's exact test";
            results.ChiSquareStatistic = double.NaN;

            var (oddsRatio, lowerCI, upperCI) = CalculateOddsRatio(matrix);
            results.OddsRatio = oddsRatio;
            results.OddsRatioLowerCI = lowerCI;
            results.OddsRatioUpperCI = upperCI;

            return results;
        }

        private static double LogFactorial(int n)
        {
            if (n <= 1) return 0;

            if (n > 20)
            {
                return n * Math.Log(n) - n + 0.5 * Math.Log(2 * Math.PI * n);
            }

            return Enumerable.Range(2, n - 1).Sum(i => Math.Log(i));
        }

        private static double GammaP(double a, double x)
        {
            if (x <= 0 || a <= 0) return 0;
            if (x >= a + 1) return 1 - GammaQ(a, x);

            double sum = 1.0 / a;
            double term = sum;
            for (int i = 1; i < 100; i++)
            {
                term *= x / (a + i);
                sum += term;
                if (Math.Abs(term) < sum * 1e-15) break;
            }

            return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
        }

        private static double GammaQ(double a, double x)
        {
            if (x <= 0 || a <= 0) return 1;
            if (x < a + 1) return 1 - GammaP(a, x);

            double f = 1;
            double c = 1;
            double pn0 = 1;
            double pn1 = x;
            double pn2 = x + 1;
            double rn = x;

            for (int n = 1; n < 100; n++)
            {
                double an = n;
                double ana = an - a;
                pn2 = (x * pn1 + ana * pn0) / an;
                pn0 = pn1;
                pn1 = pn2;
                c = 1 + ana / an;
                f *= c;
                rn = pn2 / (pn0 * f);
                if (Math.Abs(rn / x - 1) < 1e-15) break;
            }

            return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * rn;
        }

        private static double LogGamma(double x)
        {
            if (x <= 0) throw new ArgumentException("Input must be positive");

            double[] c = {
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5
        };

            double sum = 0.999999999999997;
            for (int i = 0; i < c.Length; i++)
                sum += c[i] / (x + i + 1);

            return (x + 0.5) * Math.Log(x + 5.5) - (x + 5.5) +
                   Math.Log(2.506628274631 * sum / x);
        }
    }
}