using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe;
using System.Drawing;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestCryptoRandom
    {
        [TestMethod]
        public void TestRandom10()
        {
            GC.Collect();
            Log log = new Log("Test CryptoRandom - 10 items.");
            log.EnableTrace = true;

            int nCount = 10;
            testRandom(log, CryptoRandom.METHOD.SYSTEM, nCount);
            testRandom(log, CryptoRandom.METHOD.CRYPTO, nCount);
            testRandomIdx(log, CryptoRandom.METHOD.UNIFORM_EXACT, nCount);
        }

        [TestMethod]
        public void TestRandom2()
        {
            GC.Collect();
            Log log = new Log("Test CryptoRandom - 2 items");
            log.EnableTrace = true;

            int nCount = 2;
            testRandom(log, CryptoRandom.METHOD.SYSTEM, nCount);
            testRandom(log, CryptoRandom.METHOD.CRYPTO, nCount);
            testRandomIdx(log, CryptoRandom.METHOD.UNIFORM_EXACT, nCount);
        }

        private void testRandom(Log log, CryptoRandom.METHOD method, int nBuckets)
        {
            GC.Collect();
            BucketCollection col = new BucketCollection(0, 1, nBuckets);
            CryptoRandom rand = new CryptoRandom(method, Guid.NewGuid().GetHashCode());
            int nTotal = 1000000;

            log.WriteLine("Testing (" + nBuckets.ToString() + ") "  + method.ToString());

            for (int i = 0; i < nTotal; i++)
            {
                double df = rand.NextDouble();
                col.Add(df);
            }

            string str = "";
            List<double> rgdf = new List<double>();

            for (int i = 0; i < nBuckets; i++)
            {
                double dfPct = col[i].Count / (double)nTotal;
                str += dfPct.ToString("P");
                str += ", ";
                rgdf.Add(dfPct);
            }

            str = str.TrimEnd(',', ' ');

            log.WriteLine(method.ToString() + " =>> " + str);

            double dfStdev = stdDev(rgdf, false);
            log.WriteLine(method.ToString() + " stdev = " + dfStdev.ToString());
        }

        private void testRandomIdx(Log log, CryptoRandom.METHOD method, int nBuckets)
        {
            GC.Collect();
            BucketCollection col = new BucketCollection(0.0, 1.0, nBuckets);
            CryptoRandom rand = new CryptoRandom(method, Guid.NewGuid().GetHashCode());
            int nTotal = 100000;

            log.WriteLine("Testing (" + nBuckets.ToString() + ") " + method.ToString());

            List<int> rgIdx1 = new List<int>();
            List<List<int>> rgrgPermutations = new List<List<int>>();

            for (int i = 0; i < nTotal/nBuckets; i++)
            {
                List<int> rgPermutation = new List<int>();

                for (int j = 0; j < nBuckets; j++)
                {
                    int nIdx = rand.Next(nBuckets);
                    double dfPct = (double)nIdx / (double)nBuckets;

                    rgPermutation.Add(nIdx);
 
                    col.Add(dfPct);
                }

                rgrgPermutations.Add(rgPermutation);
            }

            string str = "";
            List<double> rgdf = new List<double>();

            for (int i = 0; i < nBuckets; i++)
            {
                double dfPct = col[i].Count / (double)nTotal;
                str += dfPct.ToString("P");
                str += ", ";
                rgdf.Add(dfPct);
            }

            str = str.TrimEnd(',', ' ');

            log.WriteLine(method.ToString() + " =>> " + str);

            double dfStdev = stdDev(rgdf, false);
            log.WriteLine(method.ToString() + " stdev = " + dfStdev.ToString());


            // Verify permuation uniqueness
            int nDuplicateCount = 0;
            int nPermutationCount = rgrgPermutations.Count;
            Stopwatch sw = new Stopwatch();

            sw.Start();
            int nProgressIdx = 0;

            while (rgrgPermutations.Count > 1)
            {
                List<int> rgPermutation1 = rgrgPermutations[0];
                rgrgPermutations.RemoveAt(0);

                List<int> rgRemove = new List<int>();

                for (int j = 0; j < rgrgPermutations.Count; j++)
                {
                    if (compareLists(rgPermutation1, rgrgPermutations[j]))
                    {
                        nDuplicateCount++;
                        rgRemove.Add(j);
                    }
                }

                for (int j = rgRemove.Count - 1; j >= 0; j--)
                {
                    rgrgPermutations.RemoveAt(rgRemove[j]);
                }

                if (sw.Elapsed.TotalMilliseconds > 2000)
                {
                    log.Progress = (double)nProgressIdx / (double)nPermutationCount;
                    log.WriteLine("Permutation checking at " + log.Progress.ToString("P") + "...");
                    sw.Restart();
                }

                nProgressIdx++;
            }

            log.WriteLine("Out of " + nPermutationCount.ToString("N0") + " permutations, " + nDuplicateCount.ToString("N0") + " duplicates were found (" + ((double)nDuplicateCount / nPermutationCount).ToString("P") + ").");
        }

        private bool compareLists(List<int> rg1, List<int> rg2)
        {
            if (rg1.Count != rg2.Count)
                throw new Exception("Both lists should have the same count!");

            for (int i = 0; i < rg1.Count; i++)
            {
                if (rg1[i] != rg2[i])
                    return false;
            }

            return true;
        }

        private double stdDev(IEnumerable<double> values, bool as_sample)
        {
            // Get the mean.
            double mean = values.Sum() / values.Count();

            // Get the sum of the squares of the differences
            // between the values and the mean.
            var squares_query =
                from double value in values
                select (value - mean) * (value - mean);
            double sum_of_squares = squares_query.Sum();

            if (as_sample)
            {
                return Math.Sqrt(sum_of_squares / (values.Count() - 1));
            }
            else
            {
                return Math.Sqrt(sum_of_squares / values.Count());
            }
        }
    }
}
