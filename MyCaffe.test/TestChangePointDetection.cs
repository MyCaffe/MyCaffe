using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using System.Diagnostics;
using MyCaffe.param.beta;
using MyCaffe.solvers;
using MyCaffe.extras;
using System.Windows.Forms;
using System.Data.Entity.Migrations.Model;
using SimpleGraphing;
using System.Drawing;

namespace MyCaffe.test
{
    [TestClass]
    public class TestChangePointDetection
    {
        [TestMethod]
        public void TestCPDPrimitives()
        {
            ChangePointDetectionPrimitivesTest test = new ChangePointDetectionPrimitivesTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IChangePointDetectionTest t in test.Tests)
                {
                    t.TestCPDPrimitives();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCPD()
        {
            ChangePointDetectionPrimitivesTest test = new ChangePointDetectionPrimitivesTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IChangePointDetectionTest t in test.Tests)
                {
                    t.TestCPD();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IChangePointDetectionTest : ITest
    {
        void TestCPDPrimitives();
        void TestCPD();
    }

    class ChangePointDetectionPrimitivesTest : TestBase
    {
        public ChangePointDetectionPrimitivesTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Change Point Detection Primitives Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ChangePointDetectionTest2<double>(strName, nDeviceID, engine);
            else
                return new ChangePointDetectionTest2<float>(strName, nDeviceID, engine);
        }
    }

    class ChangePointDetectionTest2<T> : TestEx<T>, IChangePointDetectionTest
    {
        Blob<T> m_blobZ;
        Blob<T> m_blobTval;
        Random m_rand = new Random(1);

        public ChangePointDetectionTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobZ = new Blob<T>(m_cuda, m_log);
            m_blobZ.Name = "Z";
            m_blobTval = new Blob<T>(m_cuda, m_log);
            m_blobTval.Name = "Tval";
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            dispose(ref m_blobZ);
            dispose(ref m_blobTval);
            base.dispose();
        }

        // A method to generate a random float from a normal distribution
        public float Randn()
        {
            // Use the Box-Muller transform to generate two independent standard normal random variables
            // See https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
            double u1 = 1.0 - m_rand.NextDouble(); // Uniform(0,1] random doubles
            double u2 = 1.0 - m_rand.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius
            double theta = 2.0 * Math.PI * u2; // Angle
                                               // Use one of the normal random variables
            return (float)(r * Math.Cos(theta));
        }

        // A method to generate an array of random floats from a normal distribution
        public float[] Randn(int nTau, double dfMu, double dfSigma, params int[] shape)
        {
            // Check if the shape is valid
            if (shape == null || shape.Length == 0)
            {
                throw new ArgumentException("Shape must be a non-empty array of positive integers.");
            }
            if (shape.Any(x => x <= 0))
            {
                throw new ArgumentException("Shape must be a non-empty array of positive integers.");
            }
            // Compute the total size of the array
            int size = shape.Aggregate((x, y) => x * y);
            // Create an array of random floats
            float[] array = new float[size];
            for (int i = 0; i < size; i++)
            {
                array[i] = (float)dfSigma * Randn();

                if (i >= nTau)
                    array[i] += (float)dfMu;
            }
            return array;
        }

        private PlotCollection createPlots(Blob<T> blob)
        {
            PlotCollection plots = new PlotCollection(blob.Name);

            for (int i = 0; i < blob.count(); i++)
            {
                double dfVal = (double)Convert.ChangeType(blob.GetData(i), typeof(double));
                plots.Add(i, dfVal);
            }

            return plots;
        }

        public void TestCPD()
        {
            Blob<T> blobX = null;
            Blob<T> blobS = null;
            Blob<T> blobScumsum = null;
            ChangePointDetectorNN<T> cpd = null;
            ChangePointDetectorCUMSUM<T> cpdCumsum = null;
            int nN = 150;           // number of observations.
            int nTau = 75;          // true change point location.
            double dfMu = 0.2;      // shift size.
            double dfSigma = 0.1;   // Standard deviation (noise level).
            int nB = 10;
            int nEpochs = 10;
            int nOutMin = 10;
            int nTMin = 10;
            Stopwatch sw = new Stopwatch();
            Random random = new Random(1);
            string strResultFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\cpd\\result.png";

            try
            {
                blobX = new Blob<T>(m_cuda, m_log);
                blobX.Name = "X";
                blobX.Reshape(nN, 1, 1, 1);
                blobX.SetData(0);

                // Generate a gaussian signal with a change point at tau.
                float[] rgX = Randn(nTau, dfMu, dfSigma, nN);
                blobX.mutable_cpu_data = convert(rgX);

                cpd = new ChangePointDetectorNN<T>(m_cuda, m_log, "1");
                cpdCumsum = new ChangePointDetectorCUMSUM<T>();

                m_log.WriteLine("Initializing CPD...");
                sw.Start();
                cpd.Initialize(nN, blobX, nOutMin, nEpochs, nB);
                double dfTime = sw.Elapsed.TotalMilliseconds;
                m_log.WriteLine("CPD Initialization timing = " + dfTime.ToString("N2") + " ms");

                sw.Restart();
                m_log.WriteLine("Computing CPD...");
                blobS = cpd.ComputeSvalues(nTMin, false);
                dfTime = sw.Elapsed.TotalMilliseconds;
                m_log.WriteLine("CPD Compute timing = " + dfTime.ToString("N2") + " ms");

                sw.Restart();
                m_log.WriteLine("Computing CUMSUM CPD...");
                blobScumsum = cpdCumsum.ComputeSvalues(blobX);

                PlotCollection plotsX = createPlots(blobX);
                PlotCollection plotsS = createPlots(blobS);
                PlotCollection plotsScumsum = createPlots(blobScumsum);
                PlotCollectionSet set = new PlotCollectionSet() {  plotsX, plotsS, plotsScumsum };

                Image img = SimpleGraphingControl.QuickRender(set, 1000, 800);
                img.Save(strResultFile);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobS);
                dispose(ref blobScumsum);

                if (cpd != null)
                {
                    cpd.Dispose();
                    cpd = null;
                }
            }
        }

        public void TestCPDPrimitives()
        {
            string strDataPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\cpd\\";
            List<Tuple<string, string>> rgFiles = new List<Tuple<string, string>>()
            {
                new Tuple<string, string>("Z_10.npy", "tval_10.npy"),
                new Tuple<string, string>("Z_11.npy", "tval_11.npy")
            };
            ChangePointDetectionPrimitive<T> cpd = null;
            long hCpd = 0;
            int nN = 40;
            int nT = 22;
            int nTau = 10;
            int nB = 10;

            try
            {
                hCpd = m_cuda.CreateCpd();
                m_cuda.SetCpd(hCpd, nN, nB);

                cpd = new ChangePointDetectionPrimitive<T>(m_cuda, m_log);
                cpd.SetT(nN);

                foreach (Tuple<string, string> files in rgFiles)
                {

                    m_blobZ.LoadFromNumpy(strDataPath + files.Item1);
                    m_blobTval.LoadFromNumpy(strDataPath + files.Item2);

                    double fVal = cpd.ComputeTValueAt(nT, nTau, nB, m_blobZ.mutable_cpu_data);
                    double dfTVal = m_cuda.ComputeCpdTvalueAt(hCpd, nT, nTau, m_blobZ.count(), m_blobZ.gpu_data);

                    double fDiff = Math.Abs(fVal - dfTVal);
                    double fErr = 1e-07f;
                    m_log.EXPECT_NEAR(fDiff, 0, fErr);

                    nT++;
                }
            }
            finally
            {
                if (hCpd != 0)
                {
                    m_cuda.FreeCpd(hCpd);
                    hCpd = 0;
                }

                if (cpd != null)
                {
                    cpd.Dispose();
                    cpd = null;
                }
            }
        }
    }

    /// <summary>
    /// Change point detection primitives.
    /// </summary>
    /// <remarks>
    /// @see [A Contrastive Approach to Online Change Point Detection](https://arxiv.org/abs/2206.10143) by Artur Goldman, Nikita Puchkin, Valeriia Shcherbakova, and Uliana Vinogradova, 2022, arXiv
    /// @see [Numerical experiments on the WISDM data set described in the paper "A Contrastive Approach to Online Change Point Detection"](https://github.com/npuchkin/contrastive_change_point_detection/blob/main/WISDM_experiments.ipynb) by npuchkin, GitHub 2023
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class ChangePointDetectionPrimitive<T> : IDisposable
    {
        CudaDnn<T> m_cuda;
        Log m_log;
        Blob<T> m_blobT;
        Blob<T> m_blobZ;
        Blob<T> m_blobD;
        Blob<T> m_blobS;
        Blob<T> m_blobWork;

        public ChangePointDetectionPrimitive(CudaDnn<T> cuda, Log log)
        {
            m_cuda = cuda;
            m_log = log;

            m_blobT = new Blob<T>(cuda, log);
            m_blobT.Name = "T";
            m_blobZ = new Blob<T>(cuda, log);
            m_blobZ.Name = "Z";
            m_blobD = new Blob<T>(cuda, log);
            m_blobD.Name = "D";
            m_blobS = new Blob<T>(cuda, log);
            m_blobS.Name = "S";
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = "Work";
        }

        public void Dispose()
        {
            dispose(ref m_blobT);
            dispose(ref m_blobZ);
            dispose(ref m_blobD);
            dispose(ref m_blobS);
            dispose(ref m_blobWork);
        }

        private void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        public void SetT(int n)
        {
            m_blobT.Reshape(n, n, 1, 1);
            m_blobT.SetData(0);

            m_blobZ.Reshape(n, n, 1, 1);
            m_blobZ.SetData(0);

            m_blobD.Reshape(n, n, 1, 1);
            m_blobD.SetData(0);

            m_blobS.Reshape(n, 1, 1, 1);
            m_blobS.SetData(0);

            m_blobWork.Reshape(n, n, 1, 1);
            m_blobWork.SetData(0);
        }

        public float ComputeTValueAt(int t, int nTau, int nB, T[] rgZ)
        {
            CudaDnn<T> cuda = m_cuda;

            m_blobZ.Reshape(rgZ.Length, 1, 1, 1);
            m_blobZ.mutable_cpu_data = rgZ;

            T fMin = (T)Convert.ChangeType(-nB, typeof(T));
            T fMax = (T)Convert.ChangeType(nB, typeof(T));
            cuda.clip_fwd(m_blobZ.count(), m_blobZ.gpu_data, m_blobZ.mutable_gpu_data, fMin, fMax);

            m_blobD.CopyFrom(m_blobZ, false, true);

            // Compute D[:tau] = 2 / (1 + exp(-Z[:tau]))
            cuda.scal(nTau, -1, m_blobD.mutable_gpu_data);
            cuda.exp(nTau, m_blobD.gpu_data, m_blobD.mutable_gpu_data);
            cuda.add_scalar(nTau, 1, m_blobD.mutable_gpu_data);
            cuda.invert(nTau, m_blobD.gpu_data, m_blobD.mutable_gpu_data);
            cuda.scal(nTau, 2, m_blobD.mutable_gpu_data);

            // Compute D[tau:] = 2 / (1 + exp(Z[tau:]))
            cuda.exp(t - nTau, m_blobD.gpu_data, m_blobD.mutable_gpu_data, nTau, nTau, 1);
            T tOne = (T)Convert.ChangeType(1, typeof(T));
            cuda.add_scalar(t - nTau, tOne, m_blobD.mutable_gpu_data, nTau);
            cuda.invert(t - nTau, m_blobD.gpu_data, m_blobD.mutable_gpu_data, nTau, nTau);
            cuda.scal(t - nTau, 2, m_blobD.mutable_gpu_data, nTau);

            // Compute D = np.log(D)
            cuda.log(t, m_blobD.gpu_data, m_blobD.mutable_gpu_data);


            // Compute statistics for each t.
            // and each change point candidate tau.
            cuda.channel_mean(nTau, 1, 1, nTau, m_blobD.gpu_data, m_blobWork.mutable_gpu_data);
            double dfMean1 = (double)Convert.ChangeType(m_blobWork.GetData(0), typeof(double));
            cuda.channel_mean(t - nTau, 1, 1, t - nTau, m_blobD.gpu_data, m_blobWork.mutable_gpu_data, nTau);
            double dfMean2 = (double)Convert.ChangeType(m_blobWork.GetData(0), typeof(double));
            double dfMean = dfMean1 + dfMean2;
            double dfTauVal = (double)nTau * (double)(t - nTau) / (double)t * dfMean;

            int nIdx = m_blobT.offset(nTau, t);
            m_blobT.SetData(dfTauVal, nIdx);

            return (float)dfTauVal;
        }

        public float[] ComputeSValues()
        {
            CudaDnn<T> cuda = m_cuda;

            int nN = m_blobT.num;
            cuda.channel_max(m_blobT.count(), 1, nN, nN, m_blobT.gpu_data, m_blobS.mutable_gpu_data, false, true);

            float[] rgOut = Utility.ConvertVecF<T>(m_blobS.mutable_cpu_data);
            return rgOut;
        }
    }
}
