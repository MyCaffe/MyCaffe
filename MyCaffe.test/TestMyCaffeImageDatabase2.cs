using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.db.image;
using MyCaffe.basecode;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using System.IO;
using System.Threading.Tasks;
using System.Threading;

namespace MyCaffe.test
{

    [TestClass]
    public class TestMyCaffeImageDatabase2
    {
        TestingProgressSet m_set = new TestingProgressSet();
        List<SimpleResult> m_rgRes = null;

        public List<SimpleResult> Results
        {
            get { return m_rgRes; }
        }

        public void TestInitialization(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test primary dataset");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            string str;
            Stopwatch sw = new Stopwatch();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);
            try
            {

                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    db.FreeQueryState(strDs, lQueryState);
                    db.CleanUp();

                    sw.Reset();
                    sw.Stop();
                }

                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine("Cleanup Time: " + str + " ms.");
                sw.Restart();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");
        }

        private void Log_OnProgress(object sender, LogProgressArg e)
        {
            m_set.SetProgress(e.Progress);
        }

        public void TestLoadSecondaryDataset(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test secondary dataset");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);
            Stopwatch sw = new Stopwatch();
            string strDs = "MNIST";
            string strDs2 = "CIFAR-10";
            string str;

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                sw.Start();
                long lQueryState = db.InitializeWithDsName(settings, strDs);
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                db.FreeQueryState(strDs, lQueryState);

                sw.Restart();
                db.LoadDatasetByName(strDs2);
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine(strDs2 + " Initialization Time: " + str + " ms.");

                sw.Restart();
                db.CleanUp();
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine("Cleanup Time: " + str + " ms.");

                sw.Restart();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");
        }

        [TestMethod]
        public void TestInitializationLoadAll()
        {
            TestInitialization(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestInitializationLoadOnDemand()
        {
            TestInitialization(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestInitializationLoadLimit()
        {
            TestInitialization(DB_LOAD_METHOD.LOAD_ALL, 10);
        }

        [TestMethod]
        public void TestRefreshSchedule()
        {
            TestRefreshSchedule(false, 3000);
        }

        [TestMethod]
        public void TestRefreshSchedule2()
        {
            TestRefreshSchedule(true, 3000);
        }

        public void TestRefreshSchedule(bool bSetParamsDuringInit, int nLoadLimit)
        {
            TestingProgressSet progress = null;
            PreTest.Init();

            Log log = new Log("Test refresh dataset");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            string str;
            Stopwatch sw = new Stopwatch();

            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);
            try
            {
                progress = new TestingProgressSet();

                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;
                settings.ImageDbLoadLimit = nLoadLimit;

                int nRefreshUpdate = 12 * 1000;
                double dfReplacePct = 1;

                if (bSetParamsDuringInit)
                {
                    settings.ImageDbAutoRefreshScheduledUpdateInMs = nRefreshUpdate;
                    settings.ImageDbAutoRefreshScheduledReplacementPercent = dfReplacePct;

                    log.CHECK_EQ(settings.ImageDbAutoRefreshScheduledReplacementPercent, dfReplacePct, "The default auto refresh scheduled replacement percent should be " + dfReplacePct.ToString());
                    log.CHECK_EQ(settings.ImageDbAutoRefreshScheduledUpdateInMs, nRefreshUpdate, "The default auto refresh scheduled update period should be " + nRefreshUpdate.ToString());
                }
                else
                {
                    log.CHECK_EQ(settings.ImageDbAutoRefreshScheduledReplacementPercent, 0.3, "The default auto refresh scheduled replacement percentage should be 0.3.");
                    log.CHECK_EQ(settings.ImageDbAutoRefreshScheduledUpdateInMs, 10000, "The default auto refresh scheduled update period should be 10000.");
                    settings.ImageDbAutoRefreshScheduledUpdateInMs = 0;
                    settings.ImageDbAutoRefreshScheduledReplacementPercent = 0;
                }


                Stopwatch swInit = new Stopwatch();
                swInit.Start();
                string strDs = "MNIST";
                long lQueryState = db.InitializeWithDsName(settings, strDs);

                double dfInitTime = swInit.Elapsed.TotalMilliseconds;
                Trace.WriteLine("InitTime = " + dfInitTime.ToString("N2") + " ms.");

                DatasetDescriptor ds = db.GetDatasetByName(strDs);
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                if (!bSetParamsDuringInit && settings.ImageDbLoadLimit > 0)
                    db.StartAutomaticRefreshSchedule(strDs, true, false, nRefreshUpdate, dfReplacePct);

                int nPeriodInMs;
                double dfReplacementPct;
                int nTrainingRefreshCount;
                int nTestingRefreshCount;
                bool bRunning = db.GetScheduledAutoRefreshInformation(strDs, out nPeriodInMs, out dfReplacementPct, out nTrainingRefreshCount, out nTestingRefreshCount);

                if (!bRunning)
                    throw new Exception("The scheduled auto refresh should be running!");

                if (nPeriodInMs != nRefreshUpdate)
                    throw new Exception("The periods in ms is not the expected value!");

                if (dfReplacementPct != 1)
                    throw new Exception("The replacement percent is not the expected value!");

                Stopwatch swTimer = new Stopwatch();
                swTimer.Start();
                Dictionary<int, int> rgLabelCounts = new Dictionary<int, int>();
                Dictionary<long, int> rgImageIdCounts = new Dictionary<long, int>();
                int nIdx = 0;
                int nRunTime = (int)(dfInitTime * 2 * 5); // 5 cycles

                sw.Start();
                while (sw.ElapsedMilliseconds < nRunTime)
                {
                    if (swTimer.Elapsed.TotalMilliseconds > 1000)
                    {
                        int nRemaining = (int)((nRunTime) - sw.ElapsedMilliseconds);
                        Trace.WriteLine("Waiting for refresh... " + nRemaining.ToString() + " ms remaining.");
                        swTimer.Restart();

                        double dfPctProgress = sw.ElapsedMilliseconds / (double)nRunTime;
                        progress.SetProgress(dfPctProgress);
                    }

                    SimpleDatum sd = db.QueryImage(lQueryState, ds.TrainingSource.ID, nIdx);
                    nIdx++;

                    if (!rgLabelCounts.ContainsKey(sd.Label))
                        rgLabelCounts.Add(sd.Label, 1);
                    else
                        rgLabelCounts[sd.Label]++;

                    if (!rgImageIdCounts.ContainsKey(sd.ImageID))
                        rgImageIdCounts.Add(sd.ImageID, 1);
                    else
                        rgImageIdCounts[sd.ImageID]++;
                }

                db.StopAutomaticRefreshSchedule(ds.ID, true, true);
                double dfTmp;
                bRunning = db.GetScheduledAutoRefreshInformation(strDs, out nPeriodInMs, out dfTmp, out nTrainingRefreshCount, out nTestingRefreshCount);
                if (bRunning)
                    throw new Exception("The refresh should not be running!");

                sw.Reset();
                db.FreeQueryState(strDs, lQueryState);
                db.CleanUp();
                sw.Stop();

                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine("Cleanup Time: " + str + " ms.");
                sw.Restart();

                double dfTotal = 0;
                foreach (KeyValuePair<int, int> kv in rgLabelCounts)
                {
                    dfTotal += kv.Value;
                }

                Dictionary<int, double> rgLabelPct = new Dictionary<int, double>();
                foreach (KeyValuePair<int, int> kv in rgLabelCounts)
                {
                    double dfPct = kv.Value / dfTotal;
                    rgLabelPct.Add(kv.Key, dfPct);

                    Trace.WriteLine("Label " + kv.Key.ToString() + " => " + dfPct.ToString("P"));
                }

                int nTotalTrainingImagesLoaded = nLoadLimit + (int)(nLoadLimit * dfReplacementPct * nTrainingRefreshCount);
                Trace.WriteLine("Total training images loaded = " + nTotalTrainingImagesLoaded.ToString("N0"));
                Trace.WriteLine("Total unique images queried = " + rgImageIdCounts.Count.ToString("N0"));
                double dfHitPct = rgImageIdCounts.Count / (double)nTotalTrainingImagesLoaded;
                Trace.WriteLine("Hit rate = " + dfHitPct.ToString("P"));
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();

                if (progress != null)
                    progress.Dispose();
            }

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");
        }

        [TestMethod]
        public void TestLoadSecondaryLoadAll()
        {
            TestLoadSecondaryDataset(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestLoadSecondaryLoadOnDemand()
        {
            TestLoadSecondaryDataset(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestUnloadDataset()
        {
            PreTest.Init();

            Log log = new Log("Test primary dataset");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);
            Stopwatch sw = new Stopwatch();
            string str;

            try
            {
                Dictionary<string, long> rgQueryState = new Dictionary<string, long>();

                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;
                    settings.ImageDbLoadLimit = 0;

                    sw.Start();
                    rgQueryState[strDs] = db.InitializeWithDsName(settings, strDs);
                    str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    sw.Reset();
                    sw.Stop();
                }

                sw.Stop();
                sw.Reset();

                double dfTraining;
                double dfTesting;
                double dfPctLoaded;

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[0], out dfTraining, out dfTesting);
                Assert.AreEqual(1, dfPctLoaded);
                Assert.AreEqual(1, dfTraining);
                Assert.AreEqual(1, dfTesting);

                db.UnloadDatasetByName(rgDs[0]);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[0], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[1], out dfTraining, out dfTesting);
                Assert.AreEqual(1, dfPctLoaded);
                Assert.AreEqual(1, dfTraining);
                Assert.AreEqual(1, dfTesting);

                db.UnloadDatasetByName(rgDs[1]);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[1], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[2], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                db.UnloadDatasetByName(rgDs[2]);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[2], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                // First query should restart loading the dataset.
                DatasetDescriptor ds = db.GetDatasetByName(rgDs[0]);
                SimpleDatum sd = db.QueryImage(rgQueryState[rgDs[0]], ds.TrainingSource.ID, 0);
                Thread.Sleep(1000);
                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[0], out dfTraining, out dfTesting);
                Assert.AreNotEqual(0, dfPctLoaded);
                Assert.AreNotEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                if (!db.WaitForDatasetToLoad(rgDs[0], true, false))
                    Assert.Fail("Failed to wait for the dataset to load!");

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[0], out dfTraining, out dfTesting);
                Assert.AreNotEqual(0, dfPctLoaded);
                Assert.AreEqual(1, dfTraining);
                Assert.AreEqual(0, dfTesting);

                sw.Start();
                db.CleanUp();
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine("Cleanup Time: " + str + " ms.");

                sw.Restart();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");

        }

        public void TestQueryRandom(DB_LOAD_METHOD loadMethod, int nLoadLimit, DB_LABEL_SELECTION_METHOD? labelSel = null, DB_ITEM_SELECTION_METHOD? imgSel = null)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            Log log = new Log("Image Database Test");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);
            Stopwatch swNotify = new Stopwatch();

            swNotify.Start();

            try
            {
                for (int k = 0; k < 2; k++)
                {
                    foreach (string strDs in rgDs)
                    {
                        DatasetFactory df = new DatasetFactory();
                        int nDs = df.GetDatasetID(strDs);
                        if (nDs == 0)
                            log.FAIL("The dataset '" + strDs + "' does not exist - you need to load it.");

                        SettingsCaffe settings = new SettingsCaffe();
                        settings.ImageDbLoadMethod = loadMethod;
                        settings.ImageDbLoadLimit = nLoadLimit;

                        Stopwatch sw = new Stopwatch();

                        sw.Start();
                        long lQueryState = db.InitializeWithDsName(settings, strDs);
                        string str = sw.ElapsedMilliseconds.ToString();
                        Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                        if (k == 1)
                        {
                            db.FreeQueryState(strDs, lQueryState);
                            lQueryState = db.CreateQueryState(strDs, false, false);
                        }

                        DatasetDescriptor ds = db.GetDatasetByName(strDs);

                        for (int iter = 0; iter < 3; iter++)
                        {
                            Dictionary<int, List<SimpleDatum>> rgIndexes = new Dictionary<int, List<SimpleDatum>>();
                            int nCount = ds.TrainingSource.ImageCount * 2;
                            int nRefreshCount = nLoadLimit;
                            double dfTotalMs = 0;

                            for (int i = 0; i < nCount; i++)
                            {
                                sw.Restart();
                                SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, 0, labelSel, imgSel);
                                dfTotalMs += sw.ElapsedMilliseconds;
                                sw.Stop();

                                if (!rgIndexes.Keys.Contains(d.Index))
                                    rgIndexes.Add(d.Index, new List<SimpleDatum>() { d });
                                else
                                    rgIndexes[d.Index].Add(d);

                                if (swNotify.Elapsed.TotalMilliseconds > 1000)
                                {
                                    double dfPct = (double)i / nCount;
                                    log.WriteLine("Loading files at " + dfPct.ToString("P") + ", loading " + i.ToString("N0") + " of " + nCount.ToString("N0") + "...");
                                    log.Progress = dfPct;
                                    swNotify.Restart();
                                }

                                if (nLoadLimit > 0)
                                {
                                    nRefreshCount--;
                                    if (nRefreshCount == 0 && i < nCount-1)
                                    {
                                        ((IXImageDatabase2)db).StartRefresh(ds.ID, true, false, 1);
                                        ((IXImageDatabase2)db).WaitForDatasetToRefresh(ds.ID, true, false);
                                        nRefreshCount = nLoadLimit;
                                    }
                                }
                            }

                            string strUnique = (k == 0) ? "UNIQUE INDEXES: " : "NON-UNIQUE indexes: ";

                            str = (dfTotalMs / (double)nCount).ToString();
                            Trace.WriteLine(strUnique + "Average Query Time: " + str + " ms.");

                            str = db.GetLabelQueryHitPercentsAsTextFromSourceName(lQueryState, ds.TrainingSourceName);
                            Trace.WriteLine(strUnique + "Label Query Hit Percents = " + str);


                            // Verify random selection, so no indexes should be the same.
                            Dictionary<int, int> rgCounts = new Dictionary<int, int>();
                            double dfTotal = 0;
                            foreach (KeyValuePair<int, List<SimpleDatum>> kv in rgIndexes)
                            {
                                if (!rgCounts.ContainsKey(kv.Value.Count))
                                    rgCounts.Add(kv.Value.Count, 1);
                                else
                                    rgCounts[kv.Value.Count]++;

                                dfTotal += kv.Value.Count;
                            }

                            List<int> rgMissedIdx = new List<int>();
                            for (int i = 0; i < ds.TrainingSource.ImageCount; i++)
                            {
                                if (!rgIndexes.ContainsKey(i))
                                    rgMissedIdx.Add(i);
                            }

                            dfTotal /= rgIndexes.Count;

                            if (nLoadLimit == 0)
                            {
                                if (k == 0)
                                    log.EXPECT_EQUAL<float>(dfTotal, 2.0, "Unique indexes (complete epoch hits) should be 'almost' guaranteed.");
                                else
                                    log.CHECK_LE(dfTotal, 2.5, "Non-unqiue indexes (complete epoch hits are not guaranteed but are faster) should be less than or equal to 2.5.");
                            }
                            else
                            {
                                Assert.AreEqual(true, dfTotal <= 10.6);
                            }
                        }
                    }

                    db.CleanUp();
                }
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQueryRandomLoadAll()
        {
            TestQueryRandom(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQueryRandomLoadOnDemand()
        {
            TestQueryRandom(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQueryRandomLoadAllLabelBalance()
        {
            TestQueryRandom(DB_LOAD_METHOD.LOAD_ALL, 0, DB_LABEL_SELECTION_METHOD.RANDOM);
        }

        [TestMethod]
        public void TestQueryRandomLoadOnDemandLabelBalance()
        {
            TestQueryRandom(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0, DB_LABEL_SELECTION_METHOD.RANDOM);
        }

        [TestMethod]
        public void TestQueryRandomLoadLimit()
        {
            TestQueryRandom(DB_LOAD_METHOD.LOAD_ALL, 1000);
        }

        public void TestQueryRandom2(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Random 2");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };

            foreach (string strDs in rgDs)
            {
                IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

                try
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    db.SetSelectionMethod(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();
                    Dictionary<int, int> rgCounts = new Dictionary<int, int>();

                    int nCount = 10000;
                    double dfTotalMs = 0;
                    int nCount1 = 0;
                    double dfTotalMs1 = 0;

                    Stopwatch swTimer = new Stopwatch();
                    swTimer.Start();


                    // Randomly query each image and count up the number if times a given label is hit.
                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, 0, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.RANDOM);
                        sw.Stop();
                        dfTotalMs += sw.ElapsedMilliseconds;
                        dfTotalMs1 += sw.ElapsedMilliseconds;
                        nCount1++;

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);

                        if (!rgCounts.Keys.Contains(d.Label))
                            rgCounts.Add(d.Label, 1);
                        else
                            rgCounts[d.Label]++;

                        if (swTimer.Elapsed.TotalMilliseconds > 2000)
                        {
                            double dfPct = (double)i / (double)nCount;
                            Trace.WriteLine("(" + dfPct.ToString("P") + ") ave time = " + (dfTotalMs1 / nCount1).ToString("N3") + " ms.");
                            dfTotalMs1 = 0;
                            nCount1 = 0;
                            swTimer.Restart();
                        }
                    }

                    // Total the label counts and calculate the average and stddev.
                    List<KeyValuePair<int, int>> rgCountsNoLabelBalancing = rgCounts.OrderBy(p => p.Key).ToList();
                    Trace.WriteLine("NO LABEL BALANCING COUNTS");

                    CalculationArray ca = new CalculationArray();
                    foreach (KeyValuePair<int, int> kv in rgCountsNoLabelBalancing)
                    {
                        ca.Add(kv.Value);
                        Trace.WriteLine(kv.Key + " -> " + kv.Value.ToString("N0"));
                    }

                    double dfAve = ca.Average;
                    double dfStdDev1 = ca.CalculateStandardDeviation(dfAve);

                    Trace.WriteLine("Average = " + dfAve.ToString());
                    Trace.WriteLine("StdDev = " + dfStdDev1.ToString());

                    // Load the labels by first selecting the label randomly and then the image randomly from the label set.
                    rg = new Dictionary<int, List<SimpleDatum>>();
                    rgCounts = new Dictionary<int, int>();

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, 0, DB_LABEL_SELECTION_METHOD.RANDOM, DB_ITEM_SELECTION_METHOD.RANDOM);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);

                        if (!rgCounts.Keys.Contains(d.Label))
                            rgCounts.Add(d.Label, 1);
                        else
                            rgCounts[d.Label]++;
                    }

                    // Total the balanced label counts and calculate the average and stddev.
                    List<KeyValuePair<int, int>> rgCountsLabelBalancing = rgCounts.OrderBy(p => p.Key).ToList();
                    Trace.WriteLine("LABEL BALANCING COUNTS");

                    ca = new CalculationArray();

                    foreach (KeyValuePair<int, int> kv in rgCountsLabelBalancing)
                    {
                        ca.Add(kv.Value);
                        Trace.WriteLine(kv.Key + " -> " + kv.Value.ToString("N0"));
                    }

                    dfAve = ca.Average;
                    double dfStdDev2 = ca.CalculateStandardDeviation(dfAve);

                    Trace.WriteLine("Average = " + dfAve.ToString());
                    Trace.WriteLine("StdDev = " + dfStdDev2.ToString());

                    Assert.AreEqual(true, dfStdDev2 < dfStdDev1 * 1.5);

                    str = (dfTotalMs / (double)(nCount * 2)).ToString();
                    Trace.WriteLine("Average Query Time: " + str + " ms.");

                    db.CleanUp();
                }
                finally
                {
                    db.CleanUp(0, true);
                    IDisposable idisp = db as IDisposable;
                    if (idisp != null)
                        idisp.Dispose();
                }
            }
        }

        [TestMethod]
        public void TestQueryRandom2LoadAll()
        {
            TestQueryRandom2(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQueryRandom2LoadOnDemand()
        {
            TestQueryRandom2(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        public void TestQuerySequential(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Sequential");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    DatasetFactory df = new DatasetFactory();
                    int nDs = df.GetDatasetID(strDs);
                    if (nDs == 0)
                        log.FAIL("The dataset '" + strDs + "' does not exist - you need to load it.");

                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, 0, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    Trace.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify sequential selection, so all indexes should be the same.

                    foreach (KeyValuePair<int, List<SimpleDatum>> kv in rg)
                    {
                        Assert.AreEqual(kv.Value.Count, nCount);
                    }
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequentialLoadAll()
        {
            TestQuerySequential(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequentialLoadOnDemand()
        {
            TestQuerySequential(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        public void TestQuerySequential2(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Sequential 2");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    string str = sw.ElapsedMilliseconds.ToString();
                    log.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;
                    List<int> rgIdx = new List<int>();

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);

                        rgIdx.Add(d.Index);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    log.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify sequential selection.

                    rgIdx.Sort();

                    int nIdx = 0;

                    foreach (KeyValuePair<int, List<SimpleDatum>> kv in rg)
                    {
                        int nIdx1 = rgIdx[nIdx];

                        Assert.AreEqual(kv.Value.Count, (nLoadLimit == 0) ? 1 : nLoadLimit);
                        Assert.AreEqual(rg[nIdx1][0].Index, (nLoadLimit == 0) ? nIdx1 : nIdx1 % nLoadLimit);
                        nIdx++;
                    }
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequential2LoadAll()
        {
            TestQuerySequential2(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequential2LoadOnDemand()
        {
            TestQuerySequential2(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        public void TestQuerySequential3(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Sequential 3");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    db.SetSelectionMethod(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    log.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, 0);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    log.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify sequential selection, so all indexes should be the same.

                    foreach (KeyValuePair<int, List<SimpleDatum>> kv in rg)
                    {
                        Assert.AreEqual(kv.Value.Count, nCount);
                    }
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequential3LoadAll()
        {
            TestQuerySequential3(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequential3LoadOnDemand()
        {
            TestQuerySequential3(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        public void TestQuerySequential4(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Sequential 4");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    db.SetSelectionMethod(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    log.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;
                    List<int> rgIdx = new List<int>();

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, i);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);

                        rgIdx.Add(d.Index);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    log.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify sequential selection.

                    int nIdx = 0;

                    rgIdx.Sort();

                    foreach (KeyValuePair<int, List<SimpleDatum>> kv in rg)
                    {
                        int nIdx1 = rgIdx[nIdx];

                        Assert.AreEqual(kv.Value.Count, (nLoadLimit == 0) ? 1 : nLoadLimit);
                        Assert.AreEqual(rg[nIdx1][0].Index, (nLoadLimit == 0) ? nIdx1 : nIdx1 % nLoadLimit);
                        nIdx++;
                    }
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequential4LoadAll()
        {
            TestQuerySequential4(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequential4LoadOnDemand()
        {
            TestQuerySequential4(DB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        public void TestQueryPair(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Pair");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    db.SetSelectionMethod(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    log.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d1 = db.QueryImage(lQueryState, ds.TrainingSource.ID, i);
                        SimpleDatum d2 = db.QueryImage(lQueryState, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.PAIR);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d1.Index))
                            rg.Add(d1.Index, new List<SimpleDatum>() { d1 });
                        else
                            rg[d1.Index].Add(d1);

                        if (nLoadLimit > 0)
                            Assert.AreEqual(true, d1.Index == d2.Index - 1 || d1.Index == nLoadLimit - 1 && d2.Index == 0);
                        else
                            Assert.AreEqual(d1.Index, d2.Index - 1);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    log.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify that all labels are hit.
                    if (nLoadLimit > 0)
                        Assert.AreEqual(rg.Count, nLoadLimit);
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQueryPairLoadAll()
        {
            TestQueryPair(DB_LOAD_METHOD.LOAD_ALL, 0);
        }

        public void TestLoadLimitNextSequential(DB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test Query Load Limit Next Sequential");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    DatasetFactory df = new DatasetFactory();
                    int nDs = df.GetDatasetID(strDs);
                    if (nDs == 0)
                        log.FAIL("The dataset '" + strDs + "' does not exist - you need to load it.");

                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    long lQueryState = db.InitializeWithDsName(settings, strDs);
                    db.SetSelectionMethod(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    log.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();
                    Dictionary<int, List<SimpleDatum>> rgFirst = new Dictionary<int, List<SimpleDatum>>();

                    int nTotal = ds.TrainingSource.ImageCount;
                    int nCount = 0;
                    double dfTotalMs = 0;

                    while (nCount < nTotal)
                    {
                        for (int i = 0; i < nLoadLimit; i++)
                        {
                            sw.Reset();
                            sw.Start();
                            SimpleDatum d1 = db.QueryImage(lQueryState, ds.TrainingSource.ID, i);
                            dfTotalMs += sw.ElapsedMilliseconds;
                            sw.Stop();

                            if (!rg.Keys.Contains(d1.Index))
                                rg.Add(d1.Index, new List<SimpleDatum>() { d1 });
                            else
                                rg[d1.Index].Add(d1);

                            if (nCount == 0)
                            {
                                if (!rgFirst.Keys.Contains(d1.Index))
                                    rgFirst.Add(d1.Index, new List<SimpleDatum>() { d1 });
                                else
                                    rgFirst[d1.Index].Add(d1);
                            }
                        }

                        nCount += nLoadLimit;
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    log.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify that all items have been queried
                    Assert.AreEqual(nTotal, rg.Count);

                    Dictionary<int, List<SimpleDatum>> rgWrapAround = new Dictionary<int, List<SimpleDatum>>();

                    for (int i = 0; i < nLoadLimit; i++)
                    {
                        SimpleDatum d1 = db.QueryImage(lQueryState, ds.TrainingSource.ID, i);

                        if (!rgWrapAround.Keys.Contains(d1.Index))
                            rgWrapAround.Add(d1.Index, new List<SimpleDatum>() { d1 });
                        else
                            rgWrapAround[d1.Index].Add(d1);
                    }

                    // Verify that the reads wrap around to the start.
                    Assert.AreEqual(rgWrapAround.Count, rgFirst.Count);

                    List<KeyValuePair<int, List<SimpleDatum>>> rg1 = new List<KeyValuePair<int, List<SimpleDatum>>>();
                    List<KeyValuePair<int, List<SimpleDatum>>> rg2 = new List<KeyValuePair<int, List<SimpleDatum>>>();

                    foreach (KeyValuePair<int, List<SimpleDatum>> kv in rgWrapAround)
                    {
                        rg1.Add(kv);
                    }

                    foreach (KeyValuePair<int, List<SimpleDatum>> kv in rgFirst)
                    {
                        rg2.Add(kv);
                    }

                    for (int i = 0; i < rg1.Count; i++)
                    {
                        Assert.AreEqual(rg1[i].Key, rg2[i].Key);
                        Assert.AreEqual(rg1[i].Value.Count, rg2[i].Value.Count);

                        for (int j = 0; j < rg1[i].Value.Count; j++)
                        {
                            Assert.AreEqual(rg1[i].Value[j].Label, rg2[i].Value[j].Label);
                        }
                    }
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestMean()
        {
            PreTest.Init();

            Log log = new Log("Test Mean");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName(settings, strDs);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);

                    SimpleDatum d1 = db.QueryItemMean(ds.TrainingSource.ID);
                    SimpleDatum d2 = db.QueryItemMeanFromDataset(ds.ID);
                    SimpleDatum d3 = db.GetItemMean(ds.TrainingSource.ID);

                    byte[] rgB1 = d1.ByteData;
                    byte[] rgB2 = d2.ByteData;
                    byte[] rgB3 = d3.ByteData;

                    Assert.AreEqual(rgB1.Length, rgB2.Length);
                    Assert.AreEqual(rgB2.Length, rgB3.Length);

                    for (int i = 0; i < rgB1.Length; i++)
                    {
                        Assert.AreEqual(rgB1[i], rgB2[i]);
                        Assert.AreEqual(rgB2[i], rgB3[i]);
                    }
                }

                db.CleanUp();
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestGetImagesByDate()
        {
            PreTest.Init();

            Log log = new Log("Test Get Images by Date");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                settings.ImageDbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

                long lQueryState = db.InitializeWithDsName(settings, "MNIST");
                DatasetDescriptor ds = db.GetDatasetByName("MNIST");

                //---------------------------------------------
                // First add a DateTime to each image, which
                // with MNIST makes no sense, but is used 
                // just for testing the sorting.
                //
                // At the same time verify that the images
                // are initially ordered by index.
                //---------------------------------------------

                log.WriteLine("Initializing the dataset with date/time values...");
                sw.Start();

                List<SimpleDatum> rgSd = new List<SimpleDatum>();
                DateTime dt = new DateTime(2000, 1, 1);
                string strDesc = "0";

                for (int i = 0; i < ds.TrainingSource.ImageCount; i++)
                {
                    if (i % 1000 == 0)
                        strDesc = i.ToString();

                    strDesc = strDesc.PadLeft(5, '0');

                    SimpleDatum sd = db.QueryImage(lQueryState, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    sd.TimeStamp = dt;
                    sd.Description = strDesc;
                    dt += TimeSpan.FromMinutes(1);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / (double)ds.TrainingSource.ImageCount;
                        log.WriteLine("Initializing the dataset at " + dfPct.ToString("P"));
                        sw.Restart();
                    }

                    rgSd.Add(sd);
                }

                // Given that we have changed the image descriptions and time, we need to reload the indexing.
                db.ReloadIndexing(ds.ID);

                //---------------------------------------------
                //  Sort by Desc and Time and verify.
                //---------------------------------------------
                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();
                long lQueryState2 = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryState2, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                //---------------------------------------------
                //  Get images at random starting times and
                //  verify that they are in sequence.
                //---------------------------------------------

                // One minute alloted to each image above.
                Random rand = new Random(1701);

                //---------------------------------------------
                //  Verify using Filter Value
                //---------------------------------------------
                for (int i = 0; i < 60; i++)
                {
                    dt = new DateTime(2000, 1, 1);

                    int nFilterVal = i * 1000;
                    string strFilterVal = nFilterVal.ToString().PadLeft(5, '0');
                    int nCount = db.GetImageCount(lQueryState2, ds.TrainingSource.ID, strFilterVal);
                    int nSequenceCount = 10 + rand.Next(50);
                    int nRandomStart = rand.Next(nCount - nSequenceCount);
                    DateTime dtStart = dt + TimeSpan.FromMinutes(nRandomStart + i * 1000);
                    List<SimpleDatum> rgSd1 = db.GetImagesFromTime(lQueryState2, ds.TrainingSource.ID, dtStart, nSequenceCount, strFilterVal);

                    // Verify the count.
                    if (rgSd1.Count != nSequenceCount)
                        log.FAIL("Wrong number of images returned!");

                    DateTime dt1 = dtStart;

                    // Verify that we are in sequence and all have the expected filter value
                    for (int j = 0; j < rgSd1.Count; j++)
                    {
                        if (rgSd1[j].TimeStamp != dt1)
                            log.FAIL("Wrong time for item " + j.ToString());

                        if (rgSd1[j].Description != strFilterVal)
                            log.FAIL("Wrong filter value!");

                        dt1 += TimeSpan.FromMinutes(1);
                    }
                }

                //---------------------------------------------
                //  Sort by Time only and verify.
                //---------------------------------------------
                rgSd = rgSd.OrderBy(p => p.TimeStamp).ToList();
                long lQueryState3 = db.CreateQueryState(ds.ID, false, false, IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryState3, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                //---------------------------------------------
                //  Get images at random starting times and
                //  verify that they are in sequence.
                //---------------------------------------------
                //---------------------------------------------
                //  Verify using Filter Value
                //---------------------------------------------
                for (int i = 0; i < 60; i++)
                {
                    dt = new DateTime(2000, 1, 1);

                    int nCount = ds.TrainingSource.ImageCount;
                    int nSequenceCount = 10 + rand.Next(50);
                    int nRandomStart = rand.Next(nCount - nSequenceCount);
                    DateTime dtStart = dt + TimeSpan.FromMinutes(nRandomStart);
                    List<SimpleDatum> rgSd1 = db.GetImagesFromTime(lQueryState3, ds.TrainingSource.ID, dtStart, nSequenceCount);

                    // Verify the count.
                    if (rgSd1.Count != nSequenceCount)
                        log.FAIL("Wrong number of images returned!");

                    DateTime dt1 = dtStart;

                    // Verify that we are in sequence and all have the expected filter value
                    for (int j = 0; j < rgSd1.Count; j++)
                    {
                        if (rgSd1[j].TimeStamp != dt1)
                            log.FAIL("Wrong time for item " + j.ToString());

                        dt1 += TimeSpan.FromMinutes(1);
                    }
                }
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestSort()
        {
            PreTest.Init();

            Log log = new Log("Test Sort");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                settings.ImageDbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

                long lQueryState = db.InitializeWithDsName(settings, "MNIST");
                DatasetDescriptor ds = db.GetDatasetByName("MNIST");

                //---------------------------------------------
                // First add a DateTime to each image, which
                // with MNIST makes no sense, but is used 
                // just for testing the sorting.
                //
                // At the same time verify that the images
                // are initially ordered by index.
                //---------------------------------------------

                log.WriteLine("Initializing the dataset with date/time values...");
                sw.Start();

                List<SimpleDatum> rgSd = new List<SimpleDatum>();
                DateTime dt = new DateTime(2000, 1, 1);
                string strDesc = "0";

                for (int i = 0; i < ds.TrainingSource.ImageCount; i++)
                {
                    if (i % 1000 == 0)
                        strDesc = i.ToString();

                    strDesc = strDesc.PadLeft(5, '0');

                    SimpleDatum sd = db.QueryImage(lQueryState, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    sd.TimeStamp = dt;
                    sd.Description = strDesc;
                    dt += TimeSpan.FromMinutes(1);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / (double)ds.TrainingSource.ImageCount;
                        log.WriteLine("Initializing the dataset at " + dfPct.ToString("P"));
                        sw.Restart();
                    }

                    rgSd.Add(sd);
                }

                rgSd = rgSd.OrderBy(p => p.Index).ToList();

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryState, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                //---------------------------------------------
                //  Sort by ID and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();
                long lQueryStateByIDdesc = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYID_DESC);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByIDdesc, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.ImageID).ToList();
                long lQueryStateByID = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYID);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByID, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }


                //---------------------------------------------
                //  Sort by Desc and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByIDdesc, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.Index).ToList();
                long lQueryStateByDesc = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYDESC);

                List<string> rgStr = new List<string>();
                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByDesc, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }


                //---------------------------------------------
                //  Sort by Time and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByIDdesc, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.TimeStamp).ToList();
                long lQueryStateByTime = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByTime, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }


                //---------------------------------------------
                //  Sort by Desc and Time and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByIDdesc, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();
                long lQueryStateByDescTime = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByDescTime, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        log.FAIL("The image ordering is not as expected!");
                }
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestCreateDatasetOrganizedByTime()
        {
            PreTest.Init();

            Log log = new Log("CreateDatasetOrganizedByTime");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                settings.ImageDbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

                long lQueryState = db.InitializeWithDsName(settings, "MNIST");
                DatasetDescriptor ds = db.GetDatasetByName("MNIST");

                //---------------------------------------------
                // First add a DateTime to each image, which
                // with MNIST makes no sense, but is used 
                // just for testing the sorting.
                //
                // At the same time verify that the images
                // are initially ordered by index.
                //---------------------------------------------

                Trace.WriteLine("Initializing the dataset with date/time values...");
                sw.Start();

                List<SimpleDatum> rgSd = new List<SimpleDatum>();
                DateTime dt = new DateTime(2000, 1, 1);
                string strDesc = "0";

                for (int i = 0; i < ds.TrainingSource.ImageCount; i++)
                {
                    if (i % 1000 == 0)
                        strDesc = i.ToString();

                    strDesc = strDesc.PadLeft(5, '0');

                    SimpleDatum sd = db.QueryImage(lQueryState, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    sd.TimeStamp = dt;
                    sd.Description = strDesc;
                    dt += TimeSpan.FromMinutes(1);
                    rgSd.Add(sd);

                    if (i < ds.TestingSource.ImageCount)
                    {
                        sd = db.QueryImage(lQueryState, ds.TestingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                        sd.TimeStamp = dt;
                        sd.Description = strDesc;
                        dt += TimeSpan.FromMinutes(1);
                        rgSd.Add(sd);
                    }

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / (double)ds.TrainingSource.ImageCount;
                        log.WriteLine("Initializing the dataset at " + dfPct.ToString("P"));
                        sw.Restart();
                    }
                }


                // Order the items in reverse so that we can test 
                // that the created dataset was actually created
                // chronologically.
                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();
                long lQueryStateByTime = db.CreateQueryState(ds.ID, true, true, IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME);

                List<SimpleDatum> rgSd1 = new List<SimpleDatum>();

                for (int i = 0; i < ds.TrainingSource.ImageCount; i++)
                {
                    SimpleDatum sd = db.QueryImage(lQueryStateByTime, ds.TrainingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                    rgSd1.Add(sd);

                    if (i < ds.TestingSource.ImageCount)
                    {
                        sd = db.QueryImage(lQueryStateByTime, ds.TestingSource.ID, i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
                        rgSd1.Add(sd);
                    }
                }

                // The two lists should be in chronological order.
                if (rgSd1.Count != rgSd.Count)
                    log.FAIL("The list counts are incorrect!");

                for (int i = 0; i < rgSd.Count; i++)
                {
                    if (rgSd1[i].TimeStamp != rgSd[i].TimeStamp)
                        log.FAIL("The time at " + i.ToString() + " is not as expected!");

                    if (rgSd1[i].Description != rgSd[i].Description)
                        log.FAIL("The description at " + i.ToString() + " is not as expected!");
                }
            }
            finally
            {
                db.CleanUp(0, true);
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        private bool testQuery(int nTestIdx, Log log, DatasetDescriptor ds, LabelDescriptor lblDesc, DB_LOAD_METHOD loadMethod, DB_LABEL_SELECTION_METHOD lblSel, DB_ITEM_SELECTION_METHOD imgSel, List<int> rgBoostIdx, ref int nIdx, ref int nTotal)
        {
            TestingProgressSet progress = null;
            IXImageDatabase2 db = null;
            int nSrcId = ds.TrainingSource.ID;
            int nImageCount = 0;
            bool bBoosted = false;

            try
            {
                if (nTestIdx == 2)
                    imgSel = (imgSel | DB_ITEM_SELECTION_METHOD.BOOST);

                int? nLabel = null;
                if (lblDesc != null)
                    nLabel = lblDesc.ActiveLabel;

                if (nTestIdx == 0)
                {
                    if (lblDesc == null)
                    {
                        nTotal += ds.TrainingSource.ImageCount;
                        nTotal += rgBoostIdx.Count;
                    }
                    else
                    {
                        nTotal += lblDesc.ImageCount;
                        nTotal += rgBoostIdx.Where(p => p == nLabel.Value).Count();
                    }
                }
                else
                {
                    Stopwatch sw = new Stopwatch();
                    sw.Start();

                    progress = new TestingProgressSet();

                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = 0;

                    db = new MyCaffeImageDatabase2(log, "default", 1701);
                    long lQueryState = db.InitializeWithDsName(settings, ds.Name);

                    log.WriteLine("Running Test: load " + loadMethod.ToString() + ", label selection " + lblSel.ToString() + ", image selection " + imgSel.ToString());

                    Dictionary<int, int> rgLabelCounts = new Dictionary<int, int>();
                    Dictionary<int, int> rgIndexCounts = new Dictionary<int, int>();
                    List<int> rgImagesNotQueried = new List<int>();

                    Database db1 = new Database();
                    db1.Open(nSrcId);
                    List<DbItem> rgItems = db1.GetAllRawImageIndexes(false);
                    db1.Close();

                    if (lblDesc != null)
                    {
                        if ((imgSel & DB_ITEM_SELECTION_METHOD.BOOST) == DB_ITEM_SELECTION_METHOD.BOOST)
                        {
                            bBoosted = true;
                            rgItems = rgItems.Where(p => p.Label == lblDesc.ActiveLabel && p.Boost > 0).ToList();
                        }
                        else
                        {
                            rgItems = rgItems.Where(p => p.Label == lblDesc.ActiveLabel).ToList();
                        }
                    }
                    else
                    {
                        if ((imgSel & DB_ITEM_SELECTION_METHOD.BOOST) == DB_ITEM_SELECTION_METHOD.BOOST)
                        {
                            bBoosted = true;
                            rgItems = rgItems.Where(p => p.Boost > 0).ToList();
                        }
                    }

                    rgImagesNotQueried = rgItems.Select(p => p.Index).ToList();
                    nImageCount = rgImagesNotQueried.Count;

                    for (int i = 0; i < nImageCount; i++)
                    {
                        SimpleDatum sd = db.QueryImage(lQueryState, nSrcId, i, lblSel, imgSel, nLabel);
                        if (!rgLabelCounts.ContainsKey(sd.Label))
                            rgLabelCounts.Add(sd.Label, 1);
                        else
                            rgLabelCounts[sd.Label]++;

                        if (!rgIndexCounts.ContainsKey(sd.Index))
                            rgIndexCounts.Add(sd.Index, 1);
                        else
                            rgIndexCounts[sd.Index]++;

                        rgImagesNotQueried.Remove(sd.Index);

                        nIdx++;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            progress.SetProgress((double)nIdx / nTotal);
                            sw.Restart();
                        }
                    }

                    if (!bBoosted)
                    {
                        int nMinRemaining = (int)(nImageCount * 0.005);
                        if (lblSel == DB_LABEL_SELECTION_METHOD.RANDOM)
                            nMinRemaining = (int)(nImageCount * 0.03);

                        log.CHECK_LE(rgImagesNotQueried.Count, nMinRemaining, "All images should have been queried!");

                        int nTotal1 = rgLabelCounts.Sum(p => p.Value);
                        Dictionary<int, double> rgProbabilities = new Dictionary<int, double>();

                        foreach (KeyValuePair<int, int> kv in rgLabelCounts)
                        {
                            double dfProb = (double)kv.Value / nTotal1;
                            rgProbabilities.Add(kv.Key, dfProb);
                        }

                        if ((lblSel & DB_LABEL_SELECTION_METHOD.RANDOM) == DB_LABEL_SELECTION_METHOD.RANDOM)
                        {
                            double dfSum = rgProbabilities.Sum(p => p.Value);
                            double dfAve = dfSum / rgProbabilities.Count;

                            double dfThreshold = 0.001;
                            if ((lblSel & DB_LABEL_SELECTION_METHOD.RANDOM) != DB_LABEL_SELECTION_METHOD.RANDOM ||
                                (imgSel & DB_ITEM_SELECTION_METHOD.RANDOM) != DB_ITEM_SELECTION_METHOD.RANDOM)
                                dfThreshold = 0.12;

                            foreach (KeyValuePair<int, double> kv in rgProbabilities)
                            {
                                double dfDiff = Math.Abs(kv.Value - dfAve);
                                log.EXPECT_NEAR_FLOAT(kv.Value, dfAve, dfThreshold, "The probabilities are not correct!");
                            }
                        }
                    }

                    db.CleanUp();
                }

                return true;
            }
            finally
            {
                if (db != null)
                {
                    db.CleanUp(0, true);
                    IDisposable idisp = db as IDisposable;
                    if (idisp != null)
                        idisp.Dispose();
                }

                if (progress != null)
                    progress.Dispose();
            }
        }

        private string CreateDataset(DatasetFactory factory, int nCountTrain, int nCountTest, bool bUseMnistLabels)
        {
            string strName = "test_qry";
            List<RawImage> rgMnistImagesTrain = null;
            List<RawImage> rgMnistImagesTest = null;
            int nMnistIdx = 0;

            if (bUseMnistLabels)
            {
                DatasetDescriptor ds1 = factory.LoadDataset(strName);
                if (ds1 != null && ds1.TrainingSource != null && ds1.TestingSource != null)
                {
                    if (ds1.TrainingSource.ImageCount == 60000 && ds1.TestingSource.ImageCount == 10000)
                        return strName;
                }

                DatasetDescriptor dsMnist = factory.LoadDataset("MNIST");
                nCountTrain = dsMnist.TrainingSource.ImageCount;
                factory.Open(dsMnist.TrainingSource.ID);
                rgMnistImagesTrain = factory.GetRawImagesAt(0, nCountTrain);
                factory.Close();

                nCountTest = dsMnist.TestingSource.ImageCount;
                factory.Open(dsMnist.TestingSource.ID);
                rgMnistImagesTest = factory.GetRawImagesAt(0, nCountTest);
                factory.Close();
            }

            int nSrcTst = factory.AddSource(strName + ".testing", 1, 2, 2, false);
            int nSrcTrn = factory.AddSource(strName + ".training", 1, 2, 2, false);
            int nDs = factory.AddDataset(0, strName, nSrcTst, nSrcTrn);

            byte[] rgData = new byte[4];

            factory.Open(nSrcTrn, 1000);
            factory.DeleteSourceData();

            List<int> rgLabels = new List<int>();
            if (!bUseMnistLabels)
            {
                for (int i = 0; i < 10; i++)
                {
                    rgLabels.Add(i);
                }
            }

            CryptoRandom random = new CryptoRandom(CryptoRandom.METHOD.SYSTEM, 1701);

            for (int i = 0; i < nCountTrain; i++)
            {
                int nLabel = -1;

                if (bUseMnistLabels)
                {
                    nLabel = rgMnistImagesTrain[nMnistIdx].ActiveLabel.Value;
                    nMnistIdx++;

                    if (!rgLabels.Contains(nLabel))
                        rgLabels.Add(nLabel);
                }
                else
                {
                    int nIdx = random.Next(rgLabels.Count);
                    nLabel = rgLabels[nIdx];
                }

                SimpleDatum sd = new SimpleDatum(false, 1, 2, 2, nLabel, DateTime.MinValue, 0, false, i);
                sd.SetData(rgData.ToList(), nLabel);
                factory.PutRawImageCache(i, sd);
            }

            factory.ClearImageCache(true);

            rgLabels = rgLabels.OrderBy(p => p).ToList();
            for (int i = 0; i < rgLabels.Count; i++)
            {
                factory.AddLabel(rgLabels[i], rgLabels[i].ToString());
            }

            factory.Close();

            nMnistIdx = 0;

            factory.Open(nSrcTst, 1000);
            factory.DeleteSourceData();

            for (int i = 0; i < nCountTest; i++)
            {
                int nLabel = -1;

                if (bUseMnistLabels)
                {
                    nLabel = rgMnistImagesTest[nMnistIdx].ActiveLabel.Value;
                    nMnistIdx++;
                }
                else
                {
                    int nIdx = random.Next(rgLabels.Count);
                    nLabel = rgLabels[nIdx];
                }

                SimpleDatum sd = new SimpleDatum(false, 1, 2, 2, nLabel, DateTime.MinValue, 0, false, i);
                sd.SetData(rgData.ToList(), nLabel);
                factory.PutRawImageCache(i, sd);
            }

            factory.ClearImageCache(true);

            rgLabels = rgLabels.OrderBy(p => p).ToList();
            for (int i = 0; i < rgLabels.Count; i++)
            {
                factory.AddLabel(rgLabels[i], rgLabels[i].ToString());
            }

            factory.Close();

            factory.UpdateDatasetCounts(nDs);

            return strName;
        }

        [TestMethod]
        public void TestQueries()
        {
            PreTest.Init();

            TestingProgressSet progress = new TestingProgressSet();
            Log log = new Log("Test Image Database");
            Database db1 = new Database();
            int nSrcId = -1;

            log.EnableTrace = true;

            try
            {
                DatasetFactory factory = new DatasetFactory();

                string strDs = CreateDataset(factory, 10000, 1000, true);

                SettingsCaffe settings = new SettingsCaffe();
                DatasetDescriptor ds = factory.LoadDataset(strDs);
                List<int> rgBoostIdx = new List<int>();
                nSrcId = ds.TrainingSource.ID;

                db1.Open(nSrcId);

                for (int i=0; i<10; i++)
                {                    
                    RawImage img = db1.GetRawImageAt(i * 3);
                    db1.UpdateBoost(img.ID, 1);
                    rgBoostIdx.Add(img.ID);
                }

                db1.Close();

                List<LabelDescriptor> rgLabels = new List<LabelDescriptor>();
                rgLabels.Add(null);
                rgLabels.AddRange(ds.TrainingSource.Labels);

                int nImageCount = ds.TrainingSource.ImageCount;
                int nIdx = 0;
                int nTotal = nImageCount * 4;

                // i = 0, tally up total.
                // i = 1, run non-boost queries.
                // i = 2, run boost queries.
                for (int i = 0; i < 3; i++)
                {
                    foreach (LabelDescriptor lblDesc in rgLabels)
                    {
                        //---------------------------------------------------
                        //  LOAD_ON_DEMAND tests
                        //---------------------------------------------------

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ON_DEMAND, DB_LABEL_SELECTION_METHOD.RANDOM, DB_ITEM_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ON_DEMAND, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ON_DEMAND, DB_LABEL_SELECTION_METHOD.RANDOM, DB_ITEM_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ON_DEMAND, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;


                        //---------------------------------------------------
                        //  LOAD_ALL tests
                        //---------------------------------------------------

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ALL, DB_LABEL_SELECTION_METHOD.RANDOM, DB_ITEM_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ALL, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ALL, DB_LABEL_SELECTION_METHOD.RANDOM, DB_ITEM_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, DB_LOAD_METHOD.LOAD_ALL, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;
                    }
                }
            }
            finally
            {
                if (nSrcId > 0)
                    db1.ResetAllBoosts(nSrcId);
            }
        }

        [TestMethod]
        public void TestDbFilePath()
        {
            PreTest.Init();

            Database db = new Database();
            
            string strFile = db.GetDatabaseFilePath("DNN");
            string strFileImg = db.GetDatabaseImagePath("DNN");

            int nPos = strFileImg.IndexOf(strFile);
            Assert.AreEqual(nPos, 0);

            string strTemp = strFileImg.Substring(strFile.Length);
            Assert.AreEqual(strTemp, "Images\\DNN\\");
        }

        [TestMethod]
        public void TestPutRawImageToDatabase()
        {
            TestPutRawImage(false);
        }

        [TestMethod]
        public void TestPutRawImageToFileSystem()
        {
            TestPutRawImage(true);
        }

        public void TestPutRawImage(bool bSaveImagesToFile)
        {
            PreTest.Init();

            DatasetFactory factory = new DatasetFactory();

            factory.DeleteSources("Test123");
            int nSrcId = factory.AddSource("Test123", 1, 10, 10, false, 0, bSaveImagesToFile);
            factory.Open(nSrcId, 10);

            byte[] rgBytes = new byte[10 * 10];

            for (int i = 0; i < 20; i++)
            {
                rgBytes[i] = (byte)i;
                SimpleDatum sd = new SimpleDatum(false, 1, 10, 10, i, DateTime.MinValue, rgBytes.ToList(), 0, false, i);

                factory.PutRawImageCache(i, sd);
            }

            factory.ClearImageCache(true);

            List<RawImage> rgImg = factory.GetRawImagesAt(0, 20);
            for (int i = 0; i < rgImg.Count; i++)
            {
                SimpleDatum sd = factory.LoadDatum(rgImg[i]);
                bool bEncoded = false;
                byte[] rgData = sd.GetByteData(out bEncoded);

                for (int j = 0; j < 100; j++)
                {
                    if (j <= i)
                        Assert.AreEqual(rgData[j], j);
                    else
                        Assert.AreEqual(rgData[j], 0);
                }
            }

            factory.DeleteSources("Test123");
            factory.Close();
        }

        [TestMethod]
        public void TestGetAllResults()
        {
            PreTest.Init();

            Log log = new Log("Test Get All Results");
            log.EnableTrace = true;
            log.OnProgress += Log_OnProgress;

            string strDs = "MNIST";
            IXImageDatabase2 db = new MyCaffeImageDatabase2(log);

            SettingsCaffe settings = new SettingsCaffe();
            settings.ImageDbLoadMethod = DB_LOAD_METHOD.LOAD_ON_DEMAND;

            long lQueryState = db.InitializeWithDsName(settings, strDs);
            db.SetSelectionMethod(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);
            DatasetDescriptor ds = db.GetDatasetByName(strDs);

            DatasetFactory factory = new DatasetFactory();
            factory.Open(ds.TrainingSource.ID);
            factory.DeleteRawImageResults(ds.TrainingSource.ID);

            int nCount = 100;

            Dictionary<int, Tuple<SimpleDatum, List<Tuple<DateTime, int>>, List<Tuple<SimpleDatum, List<Result>>>>> rgFullSet = new Dictionary<int, Tuple<SimpleDatum, List<Tuple<DateTime, int>>, List<Tuple<SimpleDatum, List<Result>>>>>();
            List<Tuple<SimpleDatum, List<Result>>> rgSd = new List<Tuple<SimpleDatum, List<Result>>>();
            Random random = new Random();

            // Save sample extra data for the first 100 items in the MNIST dataset.
            for (int i = 0; i < nCount; i++)
            {
                SimpleDatum d = db.QueryImage(lQueryState, ds.TrainingSource.ID, i);

                List<Result> rgRes = new List<Result>();

                for (int j = 0; j < 3; j++)
                {
                    rgRes.Add(new Result(j, random.NextDouble()));
                }

                rgSd.Add(new Tuple<SimpleDatum, List<Result>>(d, rgRes));

                if (i >= 3)
                {
                    List<Tuple<DateTime, int>> rgExtra = new List<Tuple<DateTime, int>>();
                    rgExtra.Add(new Tuple<DateTime, int>(DateTime.Now, random.Next(3)));
                    rgExtra.Add(new Tuple<DateTime, int>(DateTime.Now, random.Next(3)));

                    // Save results with a 4 item history, with 3 random items per result and 2 random extra targets.
                    int nResId = factory.PutRawImageResults(ds.TrainingSource.ID, d.Index, d.Label, d.TimeStamp, rgSd, rgExtra);

                    rgFullSet.Add(d.Index, new Tuple<SimpleDatum, List<Tuple<DateTime, int>>, List<Tuple<SimpleDatum, List<Result>>>>(d, rgExtra, new List<Tuple<SimpleDatum, List<Result>>>(rgSd)));
                    rgSd.RemoveAt(0);
                }
            }

            factory.Close();

            // Test GetAllResults by first loading all results previously stored.
            m_rgRes = db.GetAllResults(ds.TrainingSourceName, true);

            // Now verify the data.
            foreach (SimpleResult res in m_rgRes)
            {
                if (!rgFullSet.ContainsKey(res.Index))
                    log.FAIL("Could not find the image index '" + res.Index.ToString() + "' in the full set!");

                Tuple<SimpleDatum, List<Tuple<DateTime, int>>, List<Tuple<SimpleDatum, List<Result>>>> item = rgFullSet[res.Index];
                SimpleDatum sd1 = item.Item1;
                List<Tuple<DateTime, int>> rgTarget = item.Item2;
                List<Tuple<SimpleDatum, List<Result>>> rgSd1 = item.Item3;

                if (sd1.Index != res.Index)
                    log.FAIL("The image indexes do not match!");

                if (sd1.TimeStamp != res.TimeStamp)
                    log.FAIL("The image timestamps do not match!");

                if (rgSd1.Count != res.BatchCount)
                    log.FAIL("The result counts do not match!");

                for (int i = 0; i < rgSd1.Count; i++)
                {
                    for (int j = 0; j < rgSd1[i].Item2.Count; j++)
                    {
                        int nIdx = i * rgSd1[i].Item2.Count + j;
                        float fExpected = (float)rgSd1[i].Item2[j].Score;
                        float fActual = res.Result[nIdx];

                        log.CHECK_EQ(fExpected, fActual, "The expected and actual values do not match!");
                    }
                }

                if (rgTarget.Count != res.Target.Length)
                    log.FAIL("The target counts do not match!");

                for (int i = 0; i < rgTarget.Count; i++)
                {
                    int nExpected = rgTarget[i].Item2;
                    int nActual = res.Target[i];

                    log.CHECK_EQ(nExpected, nActual, "The expected and actual values do not match!");
                }
            }
        }

        // ONLY UNCOMMENT WHEN USING, but do not leave in the Test Cycle.
        //[TestMethod]
        //public void ConvertAllRawImagesToFileBased()
        //{
        //    DatasetFactory factory = new DatasetFactory();

        //    List<int> rgSrc = factory.GetAllDataSourceIDs();

        //    for (int i = 0; i < rgSrc.Count; i++)
        //    {
        //        int nSrcId = rgSrc[i];
        //        SourceDescriptor src = factory.LoadSource(nSrcId);

        //        Trace.WriteLine("Converting data source '" + src.Name + "' - (" + src.ImageCount.ToString("N0") + " images) to file based...");

        //        factory.Open(nSrcId, 500, true);

        //        int nIdx = 0;
        //        int nBatchCount = 1000;
        //        Stopwatch sw = new Stopwatch();

        //        sw.Start();

        //        while (nIdx < src.ImageCount)
        //        {
        //            int nImageCount = Math.Min(nBatchCount, src.ImageCount - nIdx);
        //            bool bResult = factory.ConvertRawImagesSaveToFile(nIdx, nImageCount);

        //            if (sw.Elapsed.TotalMilliseconds > 1000)
        //            {
        //                double dfTotalPct = (double)i / rgSrc.Count;
        //                double dfPct = (double)nIdx / (double)src.ImageCount;
        //                Trace.WriteLine(dfTotalPct.ToString("P") + " (" + (i + 1).ToString() + " of " + rgSrc.Count.ToString() + ") Processing '" + src.Name + "' at " + dfPct.ToString("P"));
        //                sw.Restart();
        //            }

        //            nIdx += nImageCount;
        //        }

        //        factory.UpdateSaveImagesToFile(true);
        //        factory.Close();
        //    }
        //}

        // ONLY UNCOMMENT WHEN USING, but do not leave in the Test Cycle.
        //[TestMethod]
        //public void ConvertAllRawImagesToDatabaseBased()
        //{
        //    DatasetFactory factory = new DatasetFactory();

        //    List<int> rgSrc = factory.GetAllDataSourceIDs();

        //    for (int i = 0; i < rgSrc.Count; i++)
        //    {
        //        int nSrcId = rgSrc[i];
        //        SourceDescriptor src = factory.LoadSource(nSrcId);

        //        Trace.WriteLine("Converting data source '" + src.Name + "' - (" + src.ImageCount.ToString("N0") + " images) to database based...");

        //        factory.Open(nSrcId, 500, true);

        //        int nIdx = 0;
        //        int nBatchCount = 1000;
        //        Stopwatch sw = new Stopwatch();

        //        sw.Start();

        //        while (nIdx < src.ImageCount)
        //        {
        //            int nImageCount = Math.Min(nBatchCount, src.ImageCount - nIdx);
        //            bool bResult = factory.ConvertRawImagesSaveToDatabase(nIdx, nImageCount);

        //            if (sw.Elapsed.TotalMilliseconds > 1000)
        //            {
        //                double dfTotalPct = (double)i / rgSrc.Count;
        //                double dfPct = (double)nIdx / (double)src.ImageCount;
        //                Trace.WriteLine(dfTotalPct.ToString("P") + " (" + (i + 1).ToString() + " of " + rgSrc.Count.ToString() + ") Processing '" + src.Name + "' at " + dfPct.ToString("P"));
        //                sw.Restart();
        //            }

        //            nIdx += nImageCount;
        //        }

        //        factory.UpdateSaveImagesToFile(false);
        //        factory.Close();
        //    }
        //}
    }
}
