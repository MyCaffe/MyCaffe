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
    public class TestMyCaffeImageDatabase
    {
        public void TestInitialization(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test primary dataset");
            log.EnableTrace = true;

            string str;
            Stopwatch sw = new Stopwatch();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase(log);
            try
            {

                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");
        }

        public void TestLoadSecondaryDataset(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            Log log = new Log("Test secondary dataset");
            log.EnableTrace = true;

            IXImageDatabase1 db = new MyCaffeImageDatabase(log);
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
                db.InitializeWithDsName1(settings, strDs);
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                sw.Restart();
                db.LoadDatasetByName1(strDs2);
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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");
        }

        [TestMethod]
        public void TestUnloadDataset()
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();
            Stopwatch sw = new Stopwatch();
            string str;

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
                    settings.ImageDbLoadLimit = 0;

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
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
                Thread.Sleep(5000);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[0], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[1], out dfTraining, out dfTesting);
                Assert.AreEqual(1, dfPctLoaded);
                Assert.AreEqual(1, dfTraining);
                Assert.AreEqual(1, dfTesting);

                db.UnloadDatasetByName(rgDs[1]);
                Thread.Sleep(5000);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[1], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[2], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                db.UnloadDatasetByName(rgDs[2]);
                Thread.Sleep(5000);

                dfPctLoaded = db.GetDatasetLoadedPercentByName(rgDs[2], out dfTraining, out dfTesting);
                Assert.AreEqual(0, dfPctLoaded);
                Assert.AreEqual(0, dfTraining);
                Assert.AreEqual(0, dfTesting);

                sw.Start();
                db.CleanUp();
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine("Cleanup Time: " + str + " ms.");

                sw.Stop();
                sw.Reset();
                sw.Start();
            }
            finally
            {
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
            TestInitialization(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestInitializationLoadOnDemand()
        {
            TestInitialization(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestInitializationLoadLimit()
        {
            TestInitialization(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        [TestMethod]
        public void TestLoadSecondaryLoadAll()
        {
            TestLoadSecondaryDataset(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestLoadSecondaryLoadOnDemand()
        {
            TestLoadSecondaryDataset(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        public void TestQueryRandom(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit, IMGDB_LABEL_SELECTION_METHOD? labelSel = null, IMGDB_IMAGE_SELECTION_METHOD? imgSel = null)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            Log log = new Log("Image Database Test");
            log.EnableTrace = true;
            IXImageDatabase1 db = new MyCaffeImageDatabase(log);

            try
            {
                foreach (string strDs in rgDs)
                {
                    DatasetFactory df = new DatasetFactory();
                    int nDs = df.GetDatasetID(strDs);
                    if (nDs == 0)
                        throw new Exception("The dataset '" + strDs + "' does not exist - you need to load it.");

                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);

                    for (int iter = 0; iter < 3; iter++)
                    {
                        Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();
                        int nCount = ds.TrainingSource.ImageCount * 2;
                        double dfTotalMs = 0;

                        for (int i = 0; i < nCount; i++)
                        {
                            sw.Reset();
                            sw.Start();
                            SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, 0, labelSel, imgSel);
                            dfTotalMs += sw.ElapsedMilliseconds;
                            sw.Stop();

                            if (!rg.Keys.Contains(d.Index))
                                rg.Add(d.Index, new List<SimpleDatum>() { d });
                            else
                                rg[d.Index].Add(d);
                        }

                        str = (dfTotalMs / (double)nCount).ToString();
                        Trace.WriteLine("Average Query Time: " + str + " ms.");

                        str = db.GetLabelQueryHitPercentsAsTextFromSourceName(ds.TrainingSourceName);
                        Trace.WriteLine("Label Query Hit Percents = " + str);


                        // Verify random selection, so no indexes should be the same.
                        Dictionary<int, int> rgCounts = new Dictionary<int, int>();
                        double dfTotal = 0;
                        foreach (KeyValuePair<int, List<SimpleDatum>> kv in rg)
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
                            if (!rg.ContainsKey(i))
                                rgMissedIdx.Add(i);
                        }

                        dfTotal /= rg.Count;

                        if (nLoadLimit == 0)
                            Assert.AreEqual(true, dfTotal <= 2.1);
                        else
                            Assert.AreEqual(true, dfTotal <= 10.6);
                    }
                }

                db.CleanUp();
            }
            finally
            {
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQueryRandomLoadAll()
        {
            TestQueryRandom(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQueryRandomLoadOnDemand()
        {
            TestQueryRandom(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQueryRandomLoadAllLabelBalance()
        {
            TestQueryRandom(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM);
        }

        [TestMethod]
        public void TestQueryRandomLoadOnDemandLabelBalance()
        {
            TestQueryRandom(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM);
        }

        [TestMethod]
        public void TestQueryRandomLoadLimit()
        {
            TestQueryRandom(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQueryRandom2(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };

            foreach (string strDs in rgDs)
            {
                IXImageDatabase1 db = new MyCaffeImageDatabase();

                try
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    db.SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
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
                        SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
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
                        SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
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
                    IDisposable idisp = db as IDisposable;
                    if (idisp != null)
                        idisp.Dispose();
                }
            }
        }

        [TestMethod]
        public void TestQueryRandom2LoadAll()
        {
            TestQueryRandom2(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQueryRandom2LoadOnDemand()
        {
            TestQueryRandom2(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQueryRandom2LoadLimit()
        {
            TestQueryRandom2(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQuerySequential(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    DatasetFactory df = new DatasetFactory();
                    int nDs = df.GetDatasetID(strDs);
                    if (nDs == 0)
                        throw new Exception("The dataset '" + strDs + "' does not exist - you need to load it.");

                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
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
                        SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, 0, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequentialLoadAll()
        {
            TestQuerySequential(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequentialLoadOnDemand()
        {
            TestQuerySequential(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQuerySequentialLoadLimit()
        {
            TestQuerySequential(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQuerySequential2(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;
                    List<int> rgIdx = new List<int>();

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);

                        rgIdx.Add(d.Index);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    Trace.WriteLine("Average Query Time: " + str + " ms.");

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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequential2LoadAll()
        {
            TestQuerySequential2(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequential2LoadOnDemand()
        {
            TestQuerySequential2(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQuerySequential2LoadLimit()
        {
            TestQuerySequential2(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQuerySequential3(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    db.SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
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
                        SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, 0);
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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequential3LoadAll()
        {
            TestQuerySequential3(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequential3LoadOnDemand()
        {
            TestQuerySequential3(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQuerySequential3LoadLimit()
        {
            TestQuerySequential3(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQuerySequential4(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    db.SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);
                    Dictionary<int, List<SimpleDatum>> rg = new Dictionary<int, List<SimpleDatum>>();

                    int nCount = 100;
                    double dfTotalMs = 0;
                    List<int> rgIdx = new List<int>();

                    for (int i = 0; i < nCount; i++)
                    {
                        sw.Reset();
                        sw.Start();
                        SimpleDatum d = db.QueryImage(ds.TrainingSource.ID, i);
                        dfTotalMs += sw.ElapsedMilliseconds;
                        sw.Stop();

                        if (!rg.Keys.Contains(d.Index))
                            rg.Add(d.Index, new List<SimpleDatum>() { d });
                        else
                            rg[d.Index].Add(d);

                        rgIdx.Add(d.Index);
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    Trace.WriteLine("Average Query Time: " + str + " ms.");

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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQuerySequential4LoadAll()
        {
            TestQuerySequential4(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQuerySequential4LoadOnDemand()
        {
            TestQuerySequential4(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, 0);
        }

        [TestMethod]
        public void TestQuerySequential4LoadLimit()
        {
            TestQuerySequential4(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQueryPair(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    db.SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
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
                        SimpleDatum d1 = db.QueryImage(ds.TrainingSource.ID, i);
                        SimpleDatum d2 = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.PAIR);
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
                    Trace.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify that all labels are hit.
                    if (nLoadLimit > 0)
                        Assert.AreEqual(rg.Count, nLoadLimit);
                }

                db.CleanUp();
            }
            finally
            {
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestQueryPairLoadAll()
        {
            TestQueryPair(IMAGEDB_LOAD_METHOD.LOAD_ALL, 0);
        }

        [TestMethod]
        public void TestQueryPairLoadLimit()
        {
            TestQueryPair(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestLoadLimitNextSequential(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    DatasetFactory df = new DatasetFactory();
                    int nDs = df.GetDatasetID(strDs);
                    if (nDs == 0)
                        throw new Exception("The dataset '" + strDs + "' does not exist - you need to load it.");

                    SettingsCaffe settings = new SettingsCaffe();
                    settings.ImageDbLoadMethod = loadMethod;
                    settings.ImageDbLoadLimit = nLoadLimit;

                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    db.SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

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
                            SimpleDatum d1 = db.QueryImage(ds.TrainingSource.ID, i);
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

                        db.LoadNextSet(null);
                        nCount += nLoadLimit;
                    }

                    str = (dfTotalMs / (double)nCount).ToString();
                    Trace.WriteLine("Average Query Time: " + str + " ms.");

                    // Verify that all items have been queried
                    Assert.AreEqual(nTotal, rg.Count);

                    Dictionary<int, List<SimpleDatum>> rgWrapAround = new Dictionary<int, List<SimpleDatum>>();

                    for (int i = 0; i < nLoadLimit; i++)
                    {
                        SimpleDatum d1 = db.QueryImage(ds.TrainingSource.ID, i);

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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestLoadLimitNextSequential()
        {
            TestLoadLimitNextSequential(IMAGEDB_LOAD_METHOD.LOAD_ALL, 1000);
        }

        [TestMethod]
        public void TestMean()
        {
            PreTest.Init();

            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase1 db = new MyCaffeImageDatabase();

            try
            {
                foreach (string strDs in rgDs)
                {
                    SettingsCaffe settings = new SettingsCaffe();
                    Stopwatch sw = new Stopwatch();

                    sw.Start();
                    db.InitializeWithDsName1(settings, strDs);
                    string str = sw.ElapsedMilliseconds.ToString();
                    Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                    DatasetDescriptor ds = db.GetDatasetByName(strDs);

                    SimpleDatum d1 = db.QueryImageMean(ds.TrainingSource.ID);
                    SimpleDatum d2 = db.QueryImageMeanFromDataset(ds.ID);
                    SimpleDatum d3 = db.GetImageMean(ds.TrainingSource.ID);

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
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestGetImagesByDate()
        {
            PreTest.Init();

            Log log = new Log("GetImagesByDate");
            log.EnableTrace = true;

            IXImageDatabase1 db = new MyCaffeImageDatabase(log);

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;

                db.InitializeWithDsName1(settings, "MNIST");
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

                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    sd.TimeStamp = dt;
                    sd.Description = strDesc;
                    dt += TimeSpan.FromMinutes(1);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / (double)ds.TrainingSource.ImageCount;
                        Trace.WriteLine("Initializing the dataset at " + dfPct.ToString("P"));
                        sw.Restart();
                    }

                    rgSd.Add(sd);
                }

                //---------------------------------------------
                //  Sort by Desc and Time and verify.
                //---------------------------------------------
                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
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
                    int nCount = db.GetImageCount(ds.TrainingSource.ID, strFilterVal);
                    int nSequenceCount = 10 + rand.Next(50);
                    int nRandomStart = rand.Next(nCount - nSequenceCount);
                    DateTime dtStart = dt + TimeSpan.FromMinutes(nRandomStart + i * 1000);
                    List<SimpleDatum> rgSd1 = db.GetImagesFromTime(ds.TrainingSource.ID, dtStart, nSequenceCount, strFilterVal);

                    // Verify the count.
                    if (rgSd1.Count != nSequenceCount)
                        throw new Exception("Wrong number of images returned!");

                    DateTime dt1 = dtStart;

                    // Verify that we are in sequence and all have the expected filter value
                    for (int j = 0; j < rgSd1.Count; j++)
                    {
                        if (rgSd1[j].TimeStamp != dt1)
                            throw new Exception("Wrong time for item " + j.ToString());

                        if (rgSd1[j].Description != strFilterVal)
                            throw new Exception("Wrong filter value!");

                        dt1 += TimeSpan.FromMinutes(1);
                    }
                }

                //---------------------------------------------
                //  Sort by Time only and verify.
                //---------------------------------------------
                rgSd = rgSd.OrderBy(p => p.TimeStamp).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
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
                    List<SimpleDatum> rgSd1 = db.GetImagesFromTime(ds.TrainingSource.ID, dtStart, nSequenceCount);

                    // Verify the count.
                    if (rgSd1.Count != nSequenceCount)
                        throw new Exception("Wrong number of images returned!");

                    DateTime dt1 = dtStart;

                    // Verify that we are in sequence and all have the expected filter value
                    for (int j = 0; j < rgSd1.Count; j++)
                    {
                        if (rgSd1[j].TimeStamp != dt1)
                            throw new Exception("Wrong time for item " + j.ToString());

                        dt1 += TimeSpan.FromMinutes(1);
                    }
                }
            }
            finally
            {
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        [TestMethod]
        public void TestSort()
        {
            PreTest.Init();

            Log log = new Log("SortTest");
            log.EnableTrace = true;

            IXImageDatabase1 db = new MyCaffeImageDatabase(log);

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;

                db.InitializeWithDsName1(settings, "MNIST");
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

                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    sd.TimeStamp = dt;
                    sd.Description = strDesc;
                    dt += TimeSpan.FromMinutes(1);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / (double)ds.TrainingSource.ImageCount;
                        Trace.WriteLine("Initializing the dataset at " + dfPct.ToString("P"));
                        sw.Restart();
                    }

                    rgSd.Add(sd);
                }

                rgSd = rgSd.OrderBy(p => p.Index).ToList();

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }

                //---------------------------------------------
                //  Sort by ID and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYID_DESC);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.ImageID).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYID);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }


                //---------------------------------------------
                //  Sort by Desc and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYID_DESC);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.Description).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYDESC);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }


                //---------------------------------------------
                //  Sort by Time and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYID_DESC);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.TimeStamp).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }


                //---------------------------------------------
                //  Sort by Desc and Time and verify.
                //---------------------------------------------

                rgSd = rgSd.OrderByDescending(p => p.ImageID).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYID_DESC);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }

                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYDESC | IMGDB_SORT.BYTIME);

                for (int i = 0; i < rgSd.Count; i++)
                {
                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    if (sd.ImageID != rgSd[i].ImageID)
                        throw new Exception("The image ordering is not as expected!");
                }
            }
            finally
            {
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

            IXImageDatabase1 db = new MyCaffeImageDatabase(log);

            try
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;

                db.InitializeWithDsName1(settings, "MNIST");
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

                    SimpleDatum sd = db.QueryImage(ds.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    sd.TimeStamp = dt;
                    sd.Description = strDesc;
                    dt += TimeSpan.FromMinutes(1);
                    rgSd.Add(sd);

                    if (i < ds.TestingSource.ImageCount)
                    {
                        sd = db.QueryImage(ds.TestingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                        sd.TimeStamp = dt;
                        sd.Description = strDesc;
                        dt += TimeSpan.FromMinutes(1);
                        rgSd.Add(sd);
                    }

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / (double)ds.TrainingSource.ImageCount;
                        Trace.WriteLine("Initializing the dataset at " + dfPct.ToString("P"));
                        sw.Restart();
                    }
                }


                // Order the items in reverse so that we can test 
                // that the created dataset was actually created
                // chronologically.
                db.Sort(ds.TrainingSource.ID, IMGDB_SORT.BYID_DESC);
                db.Sort(ds.TestingSource.ID, IMGDB_SORT.BYID_DESC);

                // Create the new dataset.
                int nNewDsId = db.CreateDatasetOranizedByTime(ds.ID);

                if (nNewDsId >= 0)
                    throw new Exception("The new dataset ID should be < 0.");

                DatasetDescriptor dsNew = db.GetDatasetById(nNewDsId);
                if (dsNew.ID != nNewDsId)
                    throw new Exception("Invalid dataset ID!");

                if (dsNew.TrainingSource.ID >= 0)
                    throw new Exception("The training source ID for the dynamic dataset should be < 0.");

                if (dsNew.TestingSource.ID >= 0)
                    throw new Exception("The testing source ID for the dynamic dataset should be < 0.");

                rgSd = rgSd.OrderBy(p => p.Description).ThenBy(p => p.TimeStamp).ToList();

                List<SimpleDatum> rgSd1 = new List<SimpleDatum>();

                for (int i = 0; i < dsNew.TrainingSource.ImageCount; i++)
                {
                    SimpleDatum sd = db.QueryImage(dsNew.TrainingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    rgSd1.Add(sd);
                }

                for (int i = 0; i < dsNew.TestingSource.ImageCount; i++)
                {
                    SimpleDatum sd = db.QueryImage(dsNew.TestingSource.ID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                    rgSd1.Add(sd);
                }

                // The two lists should be in chronological order.
                if (rgSd1.Count != rgSd.Count)
                    throw new Exception("The list counts are incorrect!");

                for (int i = 0; i < rgSd.Count; i++)
                {
                    if (rgSd1[i].TimeStamp != rgSd[i].TimeStamp)
                        throw new Exception("The time at " + i.ToString() + " is not as expected!");

                    if (rgSd1[i].Description != rgSd[i].Description)
                        throw new Exception("The description at " + i.ToString() + " is not as expected!");
                }

                db.DeleteCreatedDataset(nNewDsId);
            }
            finally
            {
                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
            }
        }

        private bool testQuery(int nTestIdx, Log log, DatasetDescriptor ds, LabelDescriptor lblDesc, IMAGEDB_LOAD_METHOD loadMethod, IMGDB_LABEL_SELECTION_METHOD lblSel, IMGDB_IMAGE_SELECTION_METHOD imgSel, List<int> rgBoostIdx, ref int nIdx, ref int nTotal)
        {
            TestingProgressSet progress = null;
            IXImageDatabase1 db = null;
            int nSrcId = ds.TrainingSource.ID;
            int nImageCount = 0;

            try
            {
                if (nTestIdx == 2)
                    imgSel = (imgSel | IMGDB_IMAGE_SELECTION_METHOD.BOOST);

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

                    db = new MyCaffeImageDatabase(log, "default", 1701);
                    db.InitializeWithDsName1(settings, ds.Name);

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
                        if ((imgSel & IMGDB_IMAGE_SELECTION_METHOD.BOOST) == IMGDB_IMAGE_SELECTION_METHOD.BOOST)
                            rgItems = rgItems.Where(p => p.Label == lblDesc.ActiveLabel && p.Boost > 0).ToList();
                        else
                            rgItems = rgItems.Where(p => p.Label == lblDesc.ActiveLabel).ToList();
                    }
                    else
                    {
                        if ((imgSel & IMGDB_IMAGE_SELECTION_METHOD.BOOST) == IMGDB_IMAGE_SELECTION_METHOD.BOOST)
                            rgItems = rgItems.Where(p => p.Boost > 0).ToList();
                    }

                    rgImagesNotQueried = rgItems.Select(p => p.Index).ToList();
                    nImageCount = rgImagesNotQueried.Count;

                    for (int i = 0; i < nImageCount; i++)
                    {
                        SimpleDatum sd = db.QueryImage(nSrcId, i, lblSel, imgSel, nLabel);
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

                    int nMinRemaining = (int)(nImageCount * 0.005);
                    log.CHECK_LT(rgImagesNotQueried.Count, nMinRemaining, "All images should have been queried!");

                    int nTotal1 = rgLabelCounts.Sum(p => p.Value);
                    Dictionary<int, double> rgProbabilities = new Dictionary<int, double>();

                    foreach (KeyValuePair<int, int> kv in rgLabelCounts)
                    {
                        double dfProb = (double)kv.Value / nTotal1;
                        rgProbabilities.Add(kv.Key, dfProb);
                    }

                    if ((lblSel & IMGDB_LABEL_SELECTION_METHOD.RANDOM) == IMGDB_LABEL_SELECTION_METHOD.RANDOM)
                    {
                        double dfSum = rgProbabilities.Sum(p => p.Value);
                        double dfAve = dfSum / rgProbabilities.Count;

                        double dfThreshold = 0.001;
                        if ((lblSel & IMGDB_LABEL_SELECTION_METHOD.RANDOM) != IMGDB_LABEL_SELECTION_METHOD.RANDOM ||
                            (imgSel & IMGDB_IMAGE_SELECTION_METHOD.RANDOM) != IMGDB_IMAGE_SELECTION_METHOD.RANDOM)
                            dfThreshold = 0.12;

                        foreach (KeyValuePair<int, double> kv in rgProbabilities)
                        {
                            double dfDiff = Math.Abs(kv.Value - dfAve);
                            log.EXPECT_NEAR_FLOAT(kv.Value, dfAve, dfThreshold, "The probabilities are not correct!");
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

            factory.ClearImageCashe(true);

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

            factory.ClearImageCashe(true);

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

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;


                        //---------------------------------------------------
                        //  LOAD_ALL tests
                        //---------------------------------------------------

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ALL, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ALL, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ALL, IMGDB_LABEL_SELECTION_METHOD.RANDOM, IMGDB_IMAGE_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
                            return;

                        if (!testQuery(i, log, ds, lblDesc, IMAGEDB_LOAD_METHOD.LOAD_ALL, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE, rgBoostIdx, ref nIdx, ref nTotal))
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
                SimpleDatum sd = new SimpleDatum(false, 1, 10, 10, i, DateTime.MinValue, rgBytes.ToList(), null, 0, false, i);

                factory.PutRawImageCache(i, sd);
            }

            factory.ClearImageCashe(true);

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

    class CalculationArray
    {
        List<double> m_rgValues = new List<double>();

        public CalculationArray()
        {
        }

        public void Add(double dfVal)
        {
            m_rgValues.Add(dfVal);
        }

        public double Average
        {
            get
            {
                double dfTotal = 0;

                foreach (double df in m_rgValues)
                {
                    dfTotal += df;
                }

                return dfTotal / m_rgValues.Count;
            }
        }

        public double CalculateStandardDeviation(double dfAve)
        {
            double dfTotal = 0;

            foreach (double df in m_rgValues)
            {
                double dfDiff = df - dfAve;
                dfTotal += dfDiff * dfDiff;
            }

            dfTotal /= m_rgValues.Count;

            return Math.Sqrt(dfTotal);
        }
    }
}
