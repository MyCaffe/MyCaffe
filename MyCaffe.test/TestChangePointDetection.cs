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
using System.Threading;
using System.IO;
using System.Runtime.InteropServices;

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
        public void TestCPDStationary()
        {
            ChangePointDetectionPrimitivesTest test = new ChangePointDetectionPrimitivesTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IChangePointDetectionTest t in test.Tests)
                {
                    t.TestCPDStationary(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCPDNonStationary()
        {
            ChangePointDetectionPrimitivesTest test = new ChangePointDetectionPrimitivesTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IChangePointDetectionTest t in test.Tests)
                {
                    t.TestCPDNonStationary();
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
        void TestCPDStationary(bool bFull);
        void TestCPDNonStationary();
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
        static object m_syncCuda = new object();

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
        public float Randn(Random rand)
        {
            // Use the Box-Muller transform to generate two independent standard normal random variables
            // See https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
            double u1 = 1.0 - rand.NextDouble(); // Uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius
            double theta = 2.0 * Math.PI * u2; // Angle
                                               // Use one of the normal random variables
            return (float)(r * Math.Cos(theta));
        }

        // A method to generate an array of random floats from a normal distribution
        public float[] Randn(int nTau, double dfMu, double dfSigma2, double dfSigma, params int[] shape)
        {
            Random random = new Random(1024);

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
                array[i] = (float)dfSigma * Randn(random);

                if (i >= nTau)
                {
                    array[i] *= (float)dfSigma2;
                    array[i] += (float)dfMu;
                }
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

        private int? getThreshold(PlotCollection plots, double dfThreshold)
        {
            for (int i = 0; i < plots.Count; i++)
            {
                if (plots[i].Y >= dfThreshold)
                    return i;
            }

            return null;
        }

        private int? getThreshold(Blob<T> blob, double dfThreshold, out int nMaxIdx)
        {
            float[] rgData = Utility.ConvertVecF<T>(blob.mutable_cpu_data);
            float fMax = -float.MaxValue;

            nMaxIdx = -1;

            for (int i = 0; i < rgData.Length; i++)
            {
                if (fMax < rgData[i])
                {
                    fMax = rgData[i];
                    nMaxIdx = i;
                }

                if (rgData[i] >= dfThreshold)
                    return i;
            }

            return null;
        }

        public PlotCollection loadPlots(string strFileCsv)
        {
            PlotCollection plots = new PlotCollection("SPY");

            using (StreamReader sr = new StreamReader(strFileCsv))
            {
                string strLine = sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    strLine = sr.ReadLine();
                    string[] rgstr = strLine.Split(',');
                    DateTime dt = DateTime.Parse(rgstr[0]);
                    double dfOpen = double.Parse(rgstr[1]);
                    double dfHigh = double.Parse(rgstr[2]);
                    double dfLow = double.Parse(rgstr[3]);
                    double dfClose = double.Parse(rgstr[4]);

                    Plot p = new Plot(dt.ToFileTime(), new float[] { (float)dfOpen, (float)dfHigh, (float)dfLow, (float)dfClose });
                    p.Tag = dt;
                    plots.Add(p);
                }
            }

            return plots;
        }

        private float[] preprocess(PlotCollection plots)
        {
            float[] rgData = new float[plots.Count];

            for (int i = 1; i < plots.Count; i++)
            {
                rgData[i] = (plots[i].Y - plots[i - 1].Y);
            }

            return rgData;
        }

        private Tuple<double, double> getStats(float[] rgData, int nIdx, int nWindowSize)
        {
            double dfAve = 0;
            double dfStdDev = 0;

            for (int i = nIdx - nWindowSize; i < nIdx; i++)
            {
                dfAve += rgData[i];
            }

            dfAve /= nWindowSize;

            for (int i = nIdx - nWindowSize; i < nIdx; i++)
            {
                dfStdDev += Math.Pow(rgData[i] - dfAve, 2);
            }

            dfStdDev /= nWindowSize;
            dfStdDev = Math.Sqrt(dfStdDev);

            return new Tuple<double, double>(dfAve, dfStdDev);
        }

        private float[] getWindowData(float[] rgData, int nIdx, int nWindowSize, Tuple<double, double> stats)
        {
            int nIdx1 = 0;
            float[] rgWindow = new float[nWindowSize];

            for (int i = nIdx - nWindowSize; i < nIdx; i++)
            {
                rgWindow[nIdx1] = (float)((rgData[i] - stats.Item1) / stats.Item2);
                nIdx1++;
            }

            return rgWindow;
        }

        private Configuration createDefaultCfg(PlotCollection plots, double dfThresholdNN1, double dfThresholdCs)
        {
            Configuration cfg = SimpleGraphingControl.GetQuickRenderConfiguration("SPY", plots.Count, 2000, 1000, false, ConfigurationAxis.VALUE_RESOLUTION.DAY, null, true, null, true);
            while (cfg.Frames.Count > 1)
            {
                cfg.Frames.RemoveAt(1);
            }

            cfg.Frames[0].Plots[0].PlotFillColor = Color.Transparent;
            cfg.Frames[0].Plots[0].PlotLineColor = Color.Transparent;
            cfg.Frames[0].Plots[0].EnableLabel = true;
            cfg.Frames[0].Plots[0].LineColor = Color.DarkGray;
            cfg.Frames[0].Plots[0].FlagColor = Color.DarkGray;

            ConfigurationPlot plotHighLow = new ConfigurationPlot();
            plotHighLow.PlotType = ConfigurationPlot.PLOTTYPE.HIGHLOW;
            plotHighLow.DataIndex = 0;
            plotHighLow.DataIndexOnRender = 0;
            plotHighLow.Properties.Add("MinLevelVisible", 1);
            cfg.Frames[0].Plots.Add(plotHighLow);

            ConfigurationPlot plotsSnn = new ConfigurationPlot();
            plotsSnn.PlotType = ConfigurationPlot.PLOTTYPE.LINE;
            plotsSnn.LineWidth = 1;
            plotsSnn.PlotShape = ConfigurationPlot.PLOTSHAPE.SQUARE;
            plotsSnn.LineColor = Color.Transparent;
            plotsSnn.PlotFillColor = Color.Red;
            plotsSnn.PlotLineColor = Color.Maroon;
            plotsSnn.FlagColor = Color.Maroon;
            plotsSnn.DataIndex = 1;
            plotsSnn.DataIndexOnRender = 1;
            plotsSnn.EnableLabel = true;
            plotsSnn.Name = "SPY CPD NN (" + dfThresholdNN1.ToString("N2") + ")";
            cfg.Frames[0].Plots.Add(plotsSnn);

            ConfigurationPlot plotScs = new ConfigurationPlot();
            plotScs.PlotType = ConfigurationPlot.PLOTTYPE.LINE;
            plotScs.LineWidth = 1;
            plotScs.PlotShape = ConfigurationPlot.PLOTSHAPE.ELLIPSE;
            plotScs.LineColor = Color.Transparent;
            plotScs.PlotFillColor = Color.Lime;
            plotScs.PlotLineColor = Color.Green;
            plotScs.FlagColor = Color.Green;
            plotScs.DataIndex = 2;
            plotScs.DataIndexOnRender = 2;
            plotScs.EnableLabel = true;
            plotScs.Name = "SPY CPD Cumulative Sum (" + dfThresholdCs.ToString("N2") + ")";
            cfg.Frames[0].Plots.Add(plotScs);

            return cfg;
        }

        public void TestCPDNonStationary()
        {
            if (typeof(T) == typeof(double))
                return;

            CudaDnn<T> cuda = null;
            Log log = new Log("Test CPD");
            ChangePointDetectorContrastiveNN<T> cpdNN = null;
            ChangePointDetectorCumulativeSUM<T> cpdCS = null;
            string strDataPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\cpd\\";
            string strDataFileCsv = strDataPath + "SPY.csv";
            string strResultPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\cpd\\results\\";
            string strResultFile = strResultPath + "SPY_result.png";
            Blob<T> blobX = null;
            Blob<T> blobSnn = null;
            Blob<T> blobScs = null;
            int nWindowSize = 30;
            int nOutMin = 1;
            int nEpochs = 20;
            int nB = 10;
            int nTMin = 5;
            double dfThresholdS1 = 2.5;
            double dfThresholdScs = 2.5;
            Dictionary<DateTime, Tuple<Blob<T>, PlotCollection>> rgSnn = new Dictionary<DateTime, Tuple<Blob<T>, PlotCollection>>();
            Dictionary<DateTime, Tuple<Blob<T>, PlotCollection>> rgScs = new Dictionary<DateTime, Tuple<Blob<T>, PlotCollection>>();

            try
            {
                log.EnableTrace = true;

                if (!Directory.Exists(strResultPath))
                    Directory.CreateDirectory(strResultPath);

                cuda = new CudaDnn<T>(0);

                cpdNN = new ChangePointDetectorContrastiveNN<T>(cuda, log);
                cpdCS = new ChangePointDetectorCumulativeSUM<T>();

                blobX = new Blob<T>(cuda, log);
                blobX.Reshape(nWindowSize, 1, 1, 1);
                cpdNN.Initialize(nWindowSize, blobX, true, nOutMin, nEpochs, nB);

                PlotCollection plots = loadPlots(strDataFileCsv);
                PlotCollection plotsSnn = plots.Clone(0, true, null, false, true, null, true);
                plotsSnn.Name = "SPY CPD NN (" + dfThresholdS1.ToString("N1") + ")";
                PlotCollection plotsScs = plots.Clone(0, true, null, false, true, null, true);
                plotsScs.Name = "SPY CPD Cumulative Sum";
                Configuration cfg = createDefaultCfg(plots, dfThresholdS1, dfThresholdScs);

                Dictionary<DateTime, Tuple<DateTime, double, DateTime, double>> rgMaxes = new Dictionary<DateTime, Tuple<DateTime, double, DateTime, double>>();
                Dictionary<DateTime, int> rgMaxSnn = new Dictionary<DateTime, int>();
                Dictionary<DateTime, int> rgMaxScs = new Dictionary<DateTime, int>();
                Dictionary<DateTime, int> rgMaxBoth = new Dictionary<DateTime, int>();

                float[] rgData = preprocess(plots);
                for (int i = nWindowSize*3; i < rgData.Length; i += 1)
                {
                    Tuple<double, double> stats = getStats(rgData, i, nWindowSize * 3);
                    float[] rgWindow = getWindowData(rgData, i, nWindowSize, stats);
                    blobX.mutable_cpu_data = convert(rgWindow);

                    DateTime dtStart = (DateTime)plots[i-nWindowSize].Tag;
                    DateTime dtEnd = (DateTime)plots[i].Tag;
                    m_log.WriteLine("Running CPD on " + dtStart.ToShortDateString() + " to " + dtEnd.ToShortDateString());

                    // Operate on the current blobX values.
                    blobSnn = cpdNN.ComputeSvalues(nTMin, false);
                    blobScs = cpdCS.ComputeSvalues(blobX);

                    double dfMaxSnn = blobSnn.max_data;
                    double dfMaxScs = blobScs.max_data;

                    int nMaxIdxSnn = -1;
                    int? nIdxSnn = getThreshold(blobSnn, dfThresholdS1, out nMaxIdxSnn);
                    int nMaxIdxScs = -1;
                    int? nIdxScs = getThreshold(blobScs, dfThresholdScs, out nMaxIdxScs);

                    DateTime dtMaxSnn = (DateTime)plotsSnn[i - nWindowSize + nMaxIdxSnn].Tag;
                    DateTime dtMaxScs = (DateTime)plotsScs[i - nWindowSize + nMaxIdxScs].Tag;
                    rgMaxes.Add(dtEnd, new Tuple<DateTime, double, DateTime, double>(dtMaxSnn, dfMaxSnn, dtMaxScs, dfMaxScs));

                    m_log.WriteLine(dtStart.ToShortDateString() + " -->| dtSnn = " + dtMaxSnn.ToShortDateString() + " MaxSnn = " + dfMaxSnn.ToString("N2") + ", " + dtMaxScs.ToShortDateString() + " MaxScs = " + dfMaxScs.ToString("N2") + ", Threshold = " + dfThresholdS1.ToString("N2") + " |<-- " + dtEnd.ToShortDateString());

                    if (!rgMaxSnn.ContainsKey(dtMaxSnn))
                        rgMaxSnn.Add(dtMaxSnn, 1);
                    else
                        rgMaxSnn[dtMaxSnn]++;

                    if (!rgMaxScs.ContainsKey(dtMaxScs))
                        rgMaxScs.Add(dtMaxScs, 1);
                    else
                        rgMaxScs[dtMaxScs]++;

                    if (!rgMaxBoth.ContainsKey(dtMaxSnn))
                        rgMaxBoth.Add(dtMaxSnn, 1);
                    else
                        rgMaxBoth[dtMaxSnn]++;

                    if (!rgMaxBoth.ContainsKey(dtMaxScs))
                        rgMaxBoth.Add(dtMaxScs, 1);
                    else
                        rgMaxBoth[dtMaxScs]++;

                    PlotCollection plotsSnn1 = createPlots(blobSnn);
                    PlotCollection plotsScs1 = createPlots(blobScs);

                    //if (nIdxSnn.HasValue)
                    //{
                    //    nIdxSnn = (i - nWindowSize) + nIdxSnn.Value;
                    //    plotsSnn[nIdxSnn.Value].Active = true;
                    //}

                    //if (nIdxScs.HasValue)
                    //{
                    //    nIdxScs = (i - nWindowSize) + nIdxScs.Value;
                    //    plotsScs[nIdxScs.Value].Active = true;
                    //}

                    //if (rgMaxBoth.Count > 0)
                    //{
                    //    List<KeyValuePair<DateTime, int>> rg = rgMaxBoth.Where(p => p.Value >= 4).ToList();
                    //    for (int j = 0; j < rg.Count; j++)
                    //    {
                    //        int nPlotIdx = plotsSnn.Find(rg[j].Key);
                    //        if (nPlotIdx >= 0)
                    //            plotsSnn[nPlotIdx].Active = true;
                    //    }
                    //}

                    if (rgMaxSnn.Count > 0)
                    {
                        List<KeyValuePair<DateTime, int>> rg = rgMaxSnn.Where(p => p.Value >= 2).ToList();
                        for (int j = 0; j < rg.Count; j++)
                        {
                            int nPlotIdx = plotsSnn.Find(rg[j].Key);
                            if (nPlotIdx >= 0)
                                plotsSnn[nPlotIdx].Active = true;
                        }
                    }

                    if (rgMaxScs.Count > 0)
                    {
                        List<KeyValuePair<DateTime, int>> rg = rgMaxScs.Where(p => p.Value >= 2).ToList();
                        for (int j = 0; j < rg.Count; j++)
                        {
                            int nPlotIdx = plotsScs.Find(rg[j].Key);
                            if (nPlotIdx >= 0)
                                plotsScs[nPlotIdx].Active = true;
                        }
                    }

                    rgSnn.Add(dtEnd, new Tuple<Blob<T>, PlotCollection>(blobSnn, plotsSnn1));
                    rgScs.Add(dtEnd, new Tuple<Blob<T>, PlotCollection>(blobScs, plotsScs1));
                }

                save(strResultPath, rgSnn, "nn");
                save(strResultPath, rgScs, "cs");

                PlotCollectionSet set = new PlotCollectionSet() { plots, plotsSnn, plotsScs };
                Image img = SimpleGraphingControl.QuickRenderEx(set, cfg, 2000, 800, false, ConfigurationAxis.VALUE_RESOLUTION.DAY, true, null, true);
                img.Save(strResultFile);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobSnn);
                dispose(ref blobScs);

                cleanup(rgSnn);
                cleanup(rgScs);

                if (cpdNN != null)
                {
                    cpdNN.Dispose();
                    cpdNN = null;
                }

                if (cuda != null)
                    cuda.Dispose();
            }
        }

        private void save(string strPath, Dictionary<DateTime, Tuple<Blob<T>, PlotCollection>> blobs, string strTag)
        {
            using (StreamWriter sw = new StreamWriter(strPath + strTag + ".csv"))
            {
                StringBuilder sb = new StringBuilder();

                foreach (KeyValuePair<DateTime, Tuple<Blob<T>, PlotCollection>> kv in blobs)
                {
                    sb.Clear();
                    sb.Append(kv.Key.ToShortDateString());
                    sb.Append(',');

                    for (int i = 0; i < kv.Value.Item2.Count; i++)
                    {
                        sb.Append(kv.Value.Item2[i].Y.ToString());
                        sb.Append(',');
                    }

                    sb.Remove(sb.Length - 1, 1);

                    sw.WriteLine(sb.ToString());
                }
            }
        }

        private void cleanup(Dictionary<DateTime, Tuple<Blob<T>, PlotCollection>> blobs)
        {
            foreach (KeyValuePair<DateTime, Tuple<Blob<T>, PlotCollection>> kv in blobs)
            {
                kv.Value.Item1.Dispose();
            }
        }

        public void TestCPDStationary(bool bFull)
        {
            EventWaitHandle evtCancel = new EventWaitHandle(false, EventResetMode.AutoReset, "__GRADIENT_CHECKER_CancelEvent__");

            if (bFull && typeof(T) == typeof(double))
                return;

            double dfStartLowNoise = 0.1;
            double dfStartHighNoise = 2.5;
            double dfNoNoiseChange = 1.0;
            double dfSmallNoiseChangeUp = 1.1;
            double dfSmallNoiseChangeDn = 0.9;
            double dfLargeNoiseChangeUp = 2.5;
            double dfLargeNoiseChangeDn = 0.1;

            List<Tuple<int, double, double, string, ManualResetEvent>> rgScenarios = new List<Tuple<int, double, double, string, ManualResetEvent>>()
            {
                new Tuple<int, double, double, string, ManualResetEvent>(1, dfStartLowNoise, dfNoNoiseChange, "Low Start Noise, No Noise Change", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(2, dfStartLowNoise, dfSmallNoiseChangeUp, "Low Start Noise, Small Noise Change UP", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(3, dfStartLowNoise, dfSmallNoiseChangeDn, "Low Start Noise, Small Noise Change DN", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(4, dfStartLowNoise, dfLargeNoiseChangeUp, "Low Start Noise, Large Noise Change UP", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(5, dfStartLowNoise, dfLargeNoiseChangeDn, "Low Start Noise, Large Noise Change DN", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(6, dfStartHighNoise, dfNoNoiseChange, "High Start Noise, No Noise Change", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(7, dfStartHighNoise, dfSmallNoiseChangeUp, "High Start Noise, Small Noise Change UP", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(8, dfStartHighNoise, dfSmallNoiseChangeDn, "High Start Noise, Small Noise Change DN", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(9, dfStartHighNoise, dfLargeNoiseChangeUp, "High Start Noise, Large Noise Change UP", new ManualResetEvent(false)),
                new Tuple<int, double, double, string, ManualResetEvent>(10, dfStartHighNoise, dfLargeNoiseChangeDn, "High Start Noise, Large Noise Change DN", new ManualResetEvent(false)),
            };

            Stopwatch sw = new Stopwatch();
            sw.Start();

            List<WaitHandle> rgWait = new List<WaitHandle>();
            for (int i = 0; i < rgScenarios.Count; i++)
            {
                if (bFull || i == 3)
                {
                    Thread th = new Thread(new ParameterizedThreadStart(test_cpd_thread));
                    th.Start(new Tuple<Tuple<int, double, double, string, ManualResetEvent>, bool>(rgScenarios[i], bFull));
                    rgScenarios[i].Item5.WaitOne();

                    if (evtCancel.WaitOne(0))
                        return;

                    sw.Stop();
                    double dfPct = (double)i / (double)rgScenarios.Count;
                    m_log.WriteLine("=====================================================================================");
                    m_log.WriteLine("Test #" + i.ToString() + " '" + rgScenarios[i].Item4 + "' at " + dfPct.ToString("P") + " - " + sw.Elapsed.TotalSeconds.ToString("N2") + " sec");
                    sw.Restart();
                }
            }
        }

        private void test_cpd_thread(object obj)
        {
            EventWaitHandle evtCancel = new EventWaitHandle(false, EventResetMode.AutoReset, "__GRADIENT_CHECKER_CancelEvent__");
            Tuple<Tuple<int, double, double, string, ManualResetEvent>, bool> arg = obj as Tuple<Tuple<int, double, double, string, ManualResetEvent>, bool>;
            Tuple<int, double, double, string, ManualResetEvent> tuple = arg.Item1;
            bool bFull = arg.Item2;
            int nThreadIdx = tuple.Item1;
            double dfStartNoise = tuple.Item2;
            double dfNoiseChange = tuple.Item3;
            string strDesc = tuple.Item4;
            ManualResetEvent evtDone = tuple.Item5;
            double dfNoShift = 0.0;
            double dfSmallShift = 0.1;
            double dfLargeShift = 2.5;
            int nN = 150;
            int nTau = 75;
            List<int> rgEpochs = new List<int>() { 10 };

            try
            {
                if (bFull)
                {
                    rgEpochs.Add(20);
                    rgEpochs.Add(50);
                    rgEpochs.Add(100);
                }

                List<Tuple<double, string>> rgThreadConfig = new List<Tuple<double, string>>()
                {
                    new Tuple<double, string>(dfNoShift, "No Shift"),
                    new Tuple<double, string>(dfSmallShift, "Small Shift UP"),
                    new Tuple<double, string>(dfLargeShift, "Large Shift UP"),
                    new Tuple<double, string>(-dfLargeShift, "Large Shift DN"),
                };

                int nTotal = rgEpochs.Count * rgThreadConfig.Count;
                int nIdx = 0;

                Stopwatch sw = new Stopwatch();
                sw.Start();

                int nScenarioIdx = 0;
                for (int j = 0; j < rgEpochs.Count; j++)
                {
                    Trace.WriteLine("Thread #" + nThreadIdx.ToString() + " starting '" + strDesc + "' Epochs = " + rgEpochs[j].ToString() + "...");

                    for (int i = 0; i < rgThreadConfig.Count; i++)
                    {
                        TestCPDEx(nN, nTau, rgThreadConfig[i].Item1, dfNoiseChange, dfStartNoise, rgEpochs[j], strDesc + ", " + rgThreadConfig[i].Item2, 3.5, 2.5, bFull, nThreadIdx, nScenarioIdx);
                        nIdx++;

                        sw.Stop();
                        double dfPct = (double)nIdx / (double)nTotal;
                        Trace.WriteLine("Thread #" + nThreadIdx.ToString() + " '" + strDesc + "' at " + dfPct.ToString("P") + " completed in " + sw.Elapsed.TotalSeconds.ToString() + " sec.");
                        if (evtCancel.WaitOne(0))
                            return;

                        nScenarioIdx++;
                        sw.Restart();
                    }
                }
            }
            finally
            {
                evtDone.Set();
            }
        }

        /// <summary>
        /// Test the Change Point Detection (CPD) algorithm.
        /// </summary>
        /// <param name="nN">Specifies the sample size.</param>
        /// <param name="nTau">Specifies the true change point location.</param>
        /// <param name="dfMu">Specifies the shift applied at and after Tau steps.</param>
        /// <param name="dfSigma2">Specifies the noise applied at and after Tau steps.</param>
        /// <param name="dfSigma">Specifies the noise applied to the entire sample.</param>
        /// <param name="nEpochs">Specifies the number of epochs used for training.</param>
        /// <param name="strDesc">Specifies a description.</param>
        /// <param name="dfTarget">Specifies the threshold level.</param>
        /// <param name="dfTarget2">Optionally, specifies a secondary target.</param>
        /// <param name="bFull">Specifies whether or not to run the full test.</param>
        /// <param name="nThreadIdx">Specifies the thread.</param>
        /// <param name="nScenarioIdx">Specifies the scenario.</param>
        public void TestCPDEx(int nN = 150, int nTau = 75, double dfMu = 0.2, double dfSigma2 = 1.0, double dfSigma = 0.1, int nEpochs = 10, string strDesc = "", double dfTarget = 0.0, double? dfTarget2 = null, bool bFull = false, int nThreadIdx = 0, int nScenarioIdx = 0)
        {
            CudaDnn<T> cuda = null;
            Log log = new Log("Test CPD");
            Blob<T> blobX = null;
            Blob<T> blobSnn = null;
            Blob<T> blobScs = null;
            ChangePointDetectorContrastiveNN<T> cpdNN = null;
            ChangePointDetectorCumulativeSUM<T> cpdCS = null;
            int nB = 10;
            int nOutMin = 10;
            int nTMin = 10;
            Stopwatch sw = new Stopwatch();
            string strType = (typeof(T) == typeof(double)) ? "double" : "float";
            string strResultPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\cpd\\results\\";
            string strResultFile = strResultPath + nThreadIdx.ToString() + "_" + nScenarioIdx.ToString() + "_result_" + strType + "_" + nN.ToString() + "_" + nTau.ToString() + "_" + nEpochs.ToString() + "_" + dfMu.ToString() + "_" + dfSigma2.ToString() + "_" + dfSigma.ToString() + ".png";
            Stopwatch swTiming = new Stopwatch();

            if (!bFull)
                log.EnableTrace = true;

            if (!Directory.Exists(strResultPath))
                Directory.CreateDirectory(strResultPath);

            try
            {
                Trace.WriteLine("Mu = " + dfMu.ToString() + ", Sigma2 = " + dfSigma2.ToString() + ", Sigma = " + dfSigma.ToString() + ", Epochs = " + nEpochs.ToString() + ", Desc = '" + strDesc + "'");

                lock (m_syncCuda)
                {
                    cuda = new CudaDnn<T>(0);
                }

                blobX = new Blob<T>(cuda, log);
                blobX.Name = "X";
                blobX.Reshape(nN, 1, 1, 1);
                blobX.SetData(0);

                // Generate a gaussian signal with a change point at tau.
                float[] rgX = Randn(nTau, dfMu, dfSigma2, dfSigma, nN);
                blobX.mutable_cpu_data = convert(rgX);

                cpdNN = new ChangePointDetectorContrastiveNN<T>(cuda, log, (bFull) ? "0,1" : null);
                cpdCS = new ChangePointDetectorCumulativeSUM<T>();

                m_log.WriteLine("Initializing CPD...");
                sw.Start();
                cpdNN.Initialize(nN, blobX, false, nOutMin, nEpochs, nB);
                double dfTime = sw.Elapsed.TotalMilliseconds;
                m_log.WriteLine("CPD Initialization timing = " + dfTime.ToString("N2") + " ms");

                swTiming.Start();

                sw.Restart();
                m_log.WriteLine("Computing CPD...");
                blobSnn = cpdNN.ComputeSvalues(nTMin, false);
                dfTime = sw.Elapsed.TotalMilliseconds;
                m_log.WriteLine("CPD Compute timing = " + dfTime.ToString("N2") + " ms");

                sw.Restart();
                m_log.WriteLine("Computing Cumulative Sum CPD...");
                blobScs = cpdCS.ComputeSvalues(blobX);

                swTiming.Stop();

                PlotCollection plotsX = createPlots(blobX);
                PlotCollection plotsSnn = createPlots(blobSnn);
                PlotCollection plotsScs = createPlots(blobScs);
                PlotCollectionSet set = new PlotCollectionSet() {  plotsX, plotsSnn, plotsScs };

                ConfigurationTargetLine line = new ConfigurationTargetLine(15, Color.White);
                ConfigurationTargetLine threshold = new ConfigurationTargetLine(dfTarget, Color.Lime, ConfigurationTargetLine.LINE_TYPE.VALUE, true, Color.Black, "Threshold");
                List<ConfigurationTargetLine> rgLines = new List<ConfigurationTargetLine>() { line, threshold };

                if (dfTarget2.HasValue)
                {
                    ConfigurationTargetLine threshold2 = new ConfigurationTargetLine(dfTarget2.Value, Color.Yellow, ConfigurationTargetLine.LINE_TYPE.VALUE, true, Color.Black, "Threshold Low");
                    rgLines.Add(threshold2);
                }

                int? nIdxSnnThreshold = getThreshold(plotsSnn, dfTarget);
                int? nIdxScsThreshold = getThreshold(plotsScs, dfTarget);

                Image img = SimpleGraphingControl.QuickRender(set, 1000, 800, false, null, null, true, rgLines);
                img = renderStats(img, strDesc, nN, nTau, dfMu, dfSigma2, dfSigma, nEpochs, swTiming.Elapsed.TotalSeconds, nIdxSnnThreshold, nIdxScsThreshold);
                img.Save(strResultFile);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobSnn);
                dispose(ref blobScs);

                if (cpdNN != null)
                    cpdNN.Dispose();

                if (cuda != null)
                    cuda.Dispose();
            }
        }

        private Image renderStats(Image img, string strDesc, int nN, int nTau, double dfMu, double dfSigma2, double dfSigma, int nEpochs, double dfSeconds, int? nIdxSThreshold, int? nIdxScsThreshold)
        {
            Pen penTrueCp = new Pen(Color.FromArgb(128, Color.Fuchsia));
            Pen penSnnCp = new Pen(Color.FromArgb(128, Color.Blue));
            Pen penScsCp = new Pen(Color.FromArgb(128, Color.Green));
            Font fontTitle = new Font("Century Gothic", 14, FontStyle.Bold);
            Font fontStats = new Font("Century Gotich", 10);
            Font fontCp = new Font("Century Gotich", 8);

            penSnnCp.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
            penScsCp.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;

            using (Graphics g = Graphics.FromImage(img))
            {
                g.DrawLine(penTrueCp, (nTau + 1) * 5, 0, (nTau + 1) * 5, img.Height);
                g.DrawString("True Change Point", fontCp, Brushes.Fuchsia, (nTau + 1) * 5 + 5, 30);

                if (nIdxScsThreshold.HasValue)
                {
                    g.DrawLine(penScsCp, (nIdxScsThreshold.Value + 1) * 5, 0, (nIdxScsThreshold.Value + 1) * 5, img.Height);
                    g.DrawString("CPD Cumulative Sum Change Point", fontCp, Brushes.Green, (nIdxScsThreshold.Value + 1) * 5 + 5, 45);
                }

                if (nIdxSThreshold.HasValue)
                {
                    g.DrawLine(penSnnCp, (nIdxSThreshold.Value + 1) * 5, 0, (nIdxSThreshold.Value + 1) * 5, img.Height);
                    g.DrawString("CPD NN Change Point", fontCp, Brushes.Blue, (nIdxSThreshold.Value + 1) * 5 + 5, 60);
                }

                int nY = 70;
                if (!string.IsNullOrEmpty(strDesc))
                {
                    g.DrawString(strDesc, fontTitle, Brushes.Black, 10, nY);
                    nY += 40;
                }

                g.DrawString("Total Time: " + dfSeconds.ToString("N2") + " sec", fontStats, Brushes.Black, 10, nY);
                nY += 20;
                g.DrawString("Sample Size: " + nN.ToString(), fontStats, Brushes.Black, 10, nY);
                nY += 20;
                g.DrawString("Epochs: " + nEpochs.ToString(), fontStats, Brushes.Black, 10, nY);
                nY += 20;
                g.DrawString("True Change Point (Tau): " + nTau.ToString(), fontStats, Brushes.Black, 10, nY);
                nY += 20;
                g.DrawString("Shift (after Tau): " + dfMu.ToString("N2"), fontStats, Brushes.Black, 10, nY);
                nY += 20;
                g.DrawString("Noise (after Tau): " + dfSigma2.ToString("N2"), fontStats, Brushes.Black, 10, nY);
                nY += 20;
                g.DrawString("Noise (all): " + dfSigma.ToString("N2"), fontStats, Brushes.Black, 10, nY);
                nY += 20;
            }

            penTrueCp.Dispose();
            penSnnCp.Dispose();
            penScsCp.Dispose();

            return img;
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
