using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.test;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestRandomNumberGenerator
    {
        [TestMethod]
        public void TestRngGaussian()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngGaussian();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngGaussian2()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngGaussian2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngUniform()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngUniform();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngUniform2()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngUniform2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngBernoulli()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngBernoulli();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngBernoulli2()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngBernoulli2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngGaussianTimesGaussian()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngGaussianTimesGaussian();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngUniformTimesUniform()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngUniformTimesUniform();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngGaussianTimesBernoulli()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngGaussianTimesBernoulli();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngUniformTimesBernoulli()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngUniformTimesBernoulli();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRngBernoulliTimesBernoulli()
        {
            RandomNumberGeneratorTest test = new RandomNumberGeneratorTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRandomNumberGeneratorTest t in test.Tests)
                {
                    t.TestRngBernoulliTimesBernoulli();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IRandomNumberGeneratorTest : ITest
    {
        void TestRngGaussian();
        void TestRngGaussian2();
        void TestRngUniform();
        void TestRngUniform2();
        void TestRngBernoulli();
        void TestRngBernoulli2();
        void TestRngGaussianTimesGaussian();
        void TestRngUniformTimesUniform();
        void TestRngGaussianTimesBernoulli();
        void TestRngUniformTimesBernoulli();
        void TestRngBernoulliTimesBernoulli();
    }

    class RandomNumberGeneratorTest : TestBase
    {
        public RandomNumberGeneratorTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Random Number Generator Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new RandomNumberGeneratorTest<double>(strName, nDeviceID, engine);
            else
                return new RandomNumberGeneratorTest<float>(strName, nDeviceID, engine);
        }
    }

    class RandomNumberGeneratorTest<T> : TestEx<T>, IRandomNumberGeneratorTest
    {
        double m_dfMeanBoundMultiplier = 3.8;   // ~99.99% confidence for test failure.
        int m_nSampleSize = 10000;
        int m_nSeed = 1701;
        Blob<T> m_blobResult;

        public RandomNumberGeneratorTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            Bottom.Reshape(m_nSampleSize, 1, 1, 1);
            Top.Reshape(m_nSampleSize, 1, 1, 1);
            m_blobResult = new Blob<T>(m_cuda, m_log, m_nSampleSize, 1, 1, 1);
            m_cuda.rng_setseed(m_nSeed);
        }

        protected override void dispose()
        {
            m_blobResult.Dispose();
            base.dispose();
        }

        private double sample_mean(List<double> rg)
        {
            double dfTotal = 0;

            for (int i = 0; i < rg.Count; i++)
            {
                dfTotal += rg[i];
            }

            return dfTotal / rg.Count;
        }

        public double mean_bound(double dfStd)
        {
            return m_dfMeanBoundMultiplier * dfStd / Math.Sqrt(m_nSampleSize);
        }

        public void RngGaussianFill(double dfMu, double dfSigma, Blob<T> b)
        {
            m_cuda.rng_gaussian(b.count(), dfMu, dfSigma, b.mutable_gpu_data);
        }

        public void RngGaussianChecks(double dfMu, double dfSigma, Blob<T> b, double dfSparceP = 0)
        {
            double[] rgData = convert(b.update_cpu_data());
            double dfTrueMean = dfMu;
            double dfTrueStd = dfSigma;

            // Check that sample mean roughly matches true mean.
            double dfBound = mean_bound(dfTrueStd);
            double dfSampleMean = sample_mean(rgData.ToList());
            m_log.EXPECT_NEAR(dfSampleMean, dfTrueMean, dfBound);

            // Check that roughly half the samples are above the true mean.
            int num_above_mean = 0;
            int num_below_mean = 0;
            int num_mean = 0;
            int num_nan = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                if (rgData[i] > dfTrueMean)
                    num_above_mean++;
                else if (rgData[i] < dfTrueMean)
                    num_below_mean++;
                else if (rgData[i] == dfTrueMean)
                    num_mean++;
                else
                    num_nan++;
            }

            m_log.CHECK_EQ(0, num_nan, "There should be no nans!");

            if (dfSparceP == 0)
                m_log.CHECK_EQ(0, num_mean, "There should be no values at the exact mean.");

            double dfSamplePAboveMean = (double)num_above_mean / rgData.Length;
            double dfBernoulli_p = (1 - dfSparceP) * 0.5;
            double dfBernoulli_std = Math.Sqrt(dfBernoulli_p * (1 - dfBernoulli_p));
            double dfBernoulli_bound = mean_bound(dfBernoulli_std);

            m_log.EXPECT_NEAR(dfBernoulli_p, dfSamplePAboveMean, dfBernoulli_bound);
        }

        public void RngUniformFill(double dfLower, double dfUpper, Blob<T> b)
        {
            m_cuda.rng_uniform(b.count(), dfLower, dfUpper, b.mutable_gpu_data);
        }

        public void RngUniformChecks(double dfLower, double dfUpper, Blob<T> b, double dfSparceP = 0)
        {
            double[] rgData = convert(b.update_cpu_data());
            double dfTrueMean = (dfLower + dfUpper) / 2;
            double dfTrueStd = (dfUpper - dfLower) / Math.Sqrt(12);

            // Check that sample mean roughly matches true mean.
            double dfBound = mean_bound(dfTrueStd);
            double dfSampleMean = sample_mean(rgData.ToList());
            m_log.EXPECT_NEAR(dfSampleMean, dfTrueMean, dfBound);

            // Check that roughly half the samples are above the true mean.
            int num_above_mean = 0;
            int num_below_mean = 0;
            int num_mean = 0;
            int num_nan = 0;
            int num_above_upper = 0;
            int num_below_lower = 0;

            for (int i = 0; i < rgData.Length; i++)
            {
                if (rgData[i] > dfTrueMean)
                    num_above_mean++;
                else if (rgData[i] < dfTrueMean)
                    num_below_mean++;
                else if (rgData[i] == dfTrueMean)
                    num_mean++;
                else
                    num_nan++;

                if (rgData[i] > dfUpper)
                    num_above_upper++;
                else if (rgData[i] < dfLower)
                    num_below_lower++;
            }

            m_log.CHECK_EQ(0, num_nan, "There should be no nans!");
            m_log.CHECK_EQ(0, num_above_upper, "There should be no values above the upper value.");
            m_log.CHECK_EQ(0, num_below_lower, "There should be no values below the lower value.");

            if (dfSparceP == 0)
                m_log.CHECK_EQ(0, num_mean, "There should be no values at the exact mean.");

            double dfSamplePAboveMean = (double)num_above_mean / rgData.Length;
            double dfBernoulli_p = (1 - dfSparceP) * 0.5;
            double dfBernoulli_std = Math.Sqrt(dfBernoulli_p * (1 - dfBernoulli_p));
            double dfBernoulli_bound = mean_bound(dfBernoulli_std);

            m_log.EXPECT_NEAR(dfBernoulli_p, dfSamplePAboveMean, dfBernoulli_bound);
        }

        public void RngBernoulliFill(double dfP, Blob<T> b)
        {
            m_cuda.rng_bernoulli(b.count(), dfP, b.mutable_gpu_data);
        }
 
        public void RngBernoulliChecks(double dfP, Blob<T> b, double dfSparceP = 0)
        {
            double[] rgData = convert(b.update_cpu_data());
            double dfTrueMean = dfP;
            double dfTrueStd = Math.Sqrt(dfP * (1 - dfP));

            // Check that sample mean roughly matches true mean.
            double dfBound = mean_bound(dfTrueStd);
            double dfSampleMean = sample_mean(rgData.ToList());
            m_log.EXPECT_NEAR(dfSampleMean, dfTrueMean, dfBound);
        }

        public void TestRngGaussian()
        {
            double dfMu = 0;
            double dfSigma = 1;
            RngGaussianFill(dfMu, dfSigma, Bottom);
            RngGaussianChecks(dfMu, dfSigma, Bottom);
        }

        public void TestRngGaussian2()
        {
            double dfMu = -1;
            double dfSigma = 2;
            RngGaussianFill(dfMu, dfSigma, Bottom);
            RngGaussianChecks(dfMu, dfSigma, Bottom);
        }

        public void TestRngUniform()
        {
            double dfLower = 0;
            double dfUpper = 1;
            RngUniformFill(dfLower, dfUpper, Bottom);
            RngUniformChecks(dfLower, dfUpper, Bottom);
        }

        public void TestRngUniform2()
        {
            double dfLower = -7.3;
            double dfUpper = -2.3;
            RngUniformFill(dfLower, dfUpper, Bottom);
            RngUniformChecks(dfLower, dfUpper, Bottom);
        }

        public void TestRngBernoulli()
        {
            double dfP = 0.3;
            RngBernoulliFill(dfP, Bottom);
            RngBernoulliChecks(dfP, Bottom);
        }

        public void TestRngBernoulli2()
        {
            double dfP = 0.9;
            RngBernoulliFill(dfP, Bottom);
            RngBernoulliChecks(dfP, Bottom);
        }

        public void TestRngGaussianTimesGaussian()
        {
            double dfMu = 0;
            double dfSigma = 1;

            // Sample from 0 mean Gaussian
            RngGaussianFill(dfMu, dfSigma, Bottom);

            // Sample from 0 mean Gaussian again.
            RngGaussianFill(dfMu, dfSigma, Top);

            // Multiply Gaussians.
            m_cuda.mul(Bottom.count(), Bottom.gpu_data, Top.gpu_data, m_blobResult.mutable_gpu_data);

            // Check that result has mean 0.
            double dfMuProduct = Math.Pow(dfMu, 2.0);
            double dfSigmaProduct = Math.Sqrt(Math.Pow(dfSigma, 2.0) / 2.0);
            RngGaussianChecks(dfMuProduct, dfSigmaProduct, m_blobResult);
        }

        public void TestRngUniformTimesUniform()
        {
            // Sample from Uniform on [-2, 2]
            double dfLower1 = -2;
            double dfUpper1 = 2;
            RngUniformFill(dfLower1, dfUpper1, Bottom);

            // Sample from Uniform on [-3, 3]
            double dfLower2 = -3;
            double dfUpper2 = 3;
            RngUniformFill(dfLower2, dfUpper2, Top);

            // Multiply Uniforms.
            m_cuda.mul(Bottom.count(), Bottom.gpu_data, Top.gpu_data, m_blobResult.mutable_gpu_data);

            // Check that result does not violate checked properties of Uniform on [-6,6]
            // (though it is not actually uniformly distributed).
            double dfLowerProd = dfLower1 * dfUpper2;
            double dfUpperProd = -dfLowerProd;
            RngUniformChecks(dfLowerProd, dfUpperProd, m_blobResult);
        }

        public void TestRngGaussianTimesBernoulli()
        {
            // Sample from 0 mean Gaussian
            double dfMu = 0;
            double dfSigma = 1;
            RngGaussianFill(dfMu, dfSigma, Bottom);

            // Sample from Bernoulli with p = 0.3
            double dfBernoulli_p = 0.3;
            RngBernoulliFill(dfBernoulli_p, Top);

            // Multiply Gaussians by Bernoulli.
            m_cuda.mul(Bottom.count(), Bottom.gpu_data, Top.gpu_data, m_blobResult.mutable_gpu_data);

            // Check that result does not violate checked properties of sparsified
            // Gaussian (though it is not actually a Gaussian).
            RngGaussianChecks(dfMu, dfSigma, m_blobResult, 1 - dfBernoulli_p);
        }

        public void TestRngUniformTimesBernoulli()
        {
            // Sample from Uniform on [-1, 1]
            double dfLower = -1;
            double dfUpper = 1;
            RngUniformFill(dfLower, dfUpper, Bottom);

            // Sample from Bernoulli with p = 0.3
            double dfBernoulli_p = 0.3;
            RngBernoulliFill(dfBernoulli_p, Top);

            // Multiply Uniforms.
            m_cuda.mul(Bottom.count(), Bottom.gpu_data, Top.gpu_data, m_blobResult.mutable_gpu_data);

            // Check that result does not violate checked properties of sparsified
            // Uniform on [-1, 1] (though it is not actually uniformly distributed).
            RngUniformChecks(dfLower, dfUpper, m_blobResult, 1 - dfBernoulli_p);
        }

        public void TestRngBernoulliTimesBernoulli()
        {
            // Sample from Bernoulli with p = 0.5;
            double dfPa = 0.5;
            RngBernoulliFill(dfPa, Bottom);

            // Sample from Bernoulli with p = 0.3
            double dfPb = 0.3;
            RngBernoulliFill(dfPb, Top);

            // Multiply Uniforms.
            m_cuda.mul(Bottom.count(), Bottom.gpu_data, Top.gpu_data, m_blobResult.mutable_gpu_data);

            double[] rgData = convert(m_blobResult.update_cpu_data());
            int num_ones = 0;
            for (int i = 0; i < rgData.Length; i++)
            {
                if (rgData[i] != 0)
                {
                    m_log.CHECK_EQ(1.0, rgData[i], "The data item at " + i.ToString() + " should be 1.0");
                    num_ones++;
                }
            }

            // Check that the reuslting product has roughly p_a * p_b ones.
            double dfSampleP = sample_mean(convert(m_blobResult.update_cpu_data()).ToList());
            double dfTrueMean = dfPa * dfPb;
            double dfTrueStd = Math.Sqrt(dfTrueMean * (1 - dfTrueMean));
            double dfBound = mean_bound(dfTrueStd);

            m_log.EXPECT_NEAR(dfTrueMean, dfSampleP, dfBound);
        }
    }
}
