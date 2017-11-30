using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.imagedb;
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();
            Stopwatch sw = new Stopwatch();
            string str;

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
                str = sw.ElapsedMilliseconds.ToString();
                Trace.WriteLine(strDs + " Initialization Time: " + str + " ms.");

                sw.Reset();
                sw.Stop();
            }

            sw.Stop();
            sw.Reset();
            sw.Start();
            db.CleanUp();
            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Cleanup Time: " + str + " ms.");

            sw.Stop();
            sw.Reset();
            sw.Start();

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();

            str = sw.ElapsedMilliseconds.ToString();
            Trace.WriteLine("Dispose Time: " + str + " ms.");
        }

        [TestMethod]
        public void TestUnloadDataset()
        {
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();
            Stopwatch sw = new Stopwatch();
            string str;

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
                settings.ImageDbLoadLimit = 0;

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();

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

        public void TestQueryRandom(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

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
                db.InitializeWithDsName(settings, strDs);
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

                // Verify random selection, so no indexes should be the same.

                double dfTotal = 0;

                foreach (KeyValuePair<int, List<SimpleDatum>> kv in rg)
                {
                    dfTotal += kv.Value.Count;
                }

                dfTotal /= rg.Count;

                if (nLoadLimit == 0)
                    Assert.AreEqual(true, dfTotal <= 1.02);
                else
                    Assert.AreEqual(true, dfTotal <= 10.2);
            }

            db.CleanUp();

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
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
        public void TestQueryRandomLoadLimit()
        {
            TestQueryRandom(IMAGEDB_LOAD_METHOD.LOAD_ALL, 10);
        }

        public void TestQueryRandom2(IMAGEDB_LOAD_METHOD loadMethod, int nLoadLimit)
        {
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };

            foreach (string strDs in rgDs)
            {
                IXImageDatabase db = new MyCaffeImageDatabase();

                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                Stopwatch sw = new Stopwatch();

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

                IDisposable idisp = db as IDisposable;
                if (idisp != null)
                    idisp.Dispose();
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

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
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                Stopwatch sw = new Stopwatch();

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                Stopwatch sw = new Stopwatch();

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                Stopwatch sw = new Stopwatch();

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                settings.ImageDbLoadMethod = loadMethod;
                settings.ImageDbLoadLimit = nLoadLimit;

                Stopwatch sw = new Stopwatch();

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
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
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

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
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
        }

        [TestMethod]
        public void TestLoadLimitNextSequential()
        {
            TestLoadLimitNextSequential(IMAGEDB_LOAD_METHOD.LOAD_ALL, 1000);
        }

        [TestMethod]
        public void TestMean()
        {
            List<string> rgDs = new List<string>() { "MNIST", "CIFAR-10", "MNIST" };
            IXImageDatabase db = new MyCaffeImageDatabase();

            foreach (string strDs in rgDs)
            {
                SettingsCaffe settings = new SettingsCaffe();
                Stopwatch sw = new Stopwatch();

                sw.Start();
                db.InitializeWithDsName(settings, strDs);
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

            IDisposable idisp = db as IDisposable;
            if (idisp != null)
                idisp.Dispose();
        }

        [TestMethod]
        public void TestDbFilePath()
        {
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

            factory.ClearImageCash(true);

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
