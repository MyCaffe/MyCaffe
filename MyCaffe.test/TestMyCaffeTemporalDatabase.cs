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
using MyCaffe.db.temporal;
using SimpleGraphing;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.common;

namespace MyCaffe.test
{

    [TestClass]
    public class TestMyCaffeTemporalDatabase
    {
        private SourceDescriptor loadTestData(int nHistSteps, int nFutSteps, out DateTime dtStart, out DateTime dtEnd)
        {
            DatabaseTemporal db = new DatabaseTemporal();
            int nSteps = nHistSteps + nFutSteps;
            int nBlocks = 2;
            int nItems = 2;
            int nStreams = 3;
            string strName = "$TFT$.src.test";

            int nSrcId = db.GetSourceID(strName);
            if (nSrcId > 0)
                db.DeleteSource(nSrcId);

            nSrcId = db.AddSource(strName, nItems, nStreams, nSteps * 2, true);
            SourceDescriptor src = new SourceDescriptor(nSrcId, strName, nSteps * 2, nStreams, nItems, true, true);

            dtStart = new DateTime(2017, 1, 1);
            dtEnd = dtStart.AddMinutes(nSteps * nBlocks);

            int nValItem1 = db.AddValueItem(nSrcId, 0, "Test Value Item #1");
            int nValItem2 = db.AddValueItem(nSrcId, 1, "Test Value ITem #2");

            // Setup the data schema.
            int nSecPerStep = 60;
            int nOrdering = 0;
            int nValStrm1 = db.AddValueStream(nSrcId, "Test Stream #1", nOrdering++, ValueStreamDescriptor.STREAM_CLASS_TYPE.OBSERVED, ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, dtStart, dtEnd, nSecPerStep);
            int nValStrm2 = db.AddValueStream(nSrcId, "Test Stream #2", nOrdering++, ValueStreamDescriptor.STREAM_CLASS_TYPE.OBSERVED, ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, dtStart, dtEnd, nSecPerStep);
            int nValStrm3 = db.AddValueStream(nSrcId, "Test Stream #3", nOrdering++, ValueStreamDescriptor.STREAM_CLASS_TYPE.KNOWN, ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, dtStart, dtEnd, nSecPerStep);
            int nValStrm4 = db.AddValueStream(nSrcId, "Test Stream #4", nOrdering++, ValueStreamDescriptor.STREAM_CLASS_TYPE.STATIC, ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL);

            RawValueDataCollection dataStatic = new RawValueDataCollection(null);
            dataStatic.Add(new RawValueData(ValueStreamDescriptor.STREAM_CLASS_TYPE.STATIC, ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, nValStrm4));

            RawValueDataCollection data = new RawValueDataCollection(null);
            data.Add(new RawValueData(ValueStreamDescriptor.STREAM_CLASS_TYPE.OBSERVED, ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, nValStrm1));
            data.Add(new RawValueData(ValueStreamDescriptor.STREAM_CLASS_TYPE.OBSERVED, ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, nValStrm2));
            data.Add(new RawValueData(ValueStreamDescriptor.STREAM_CLASS_TYPE.KNOWN, ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, nValStrm3));

            db.Open(nSrcId);
            db.EnableBulk(true);

            // Add the data to item 1
            dataStatic.SetData(new float[] { 1 });
            db.PutRawValue(nSrcId, nValItem1, dataStatic);

            DateTime dt = dtStart;
            for (int i = 0; i < nSteps * nBlocks; i++)
            {
                float[] rgfVal1 = new float[] { (float)((i + 1) * 2), (float)((i + 2) * 3), (float)i };
                data.SetData(dt, rgfVal1);

                db.PutRawValue(nSrcId, nValItem1, data);
                dt += TimeSpan.FromMinutes(1);
            }

            // Add the data to item 2
            dataStatic.SetData(new float[] { 2 });
            db.PutRawValue(nSrcId, nValItem2, dataStatic);

            dt = dtStart;
            for (int i = 0; i < nSteps * nBlocks; i++)
            {
                float[] rgfVal1 = new float[] { (float)((i + 3) * 4), (float)((i + 4) * 8), (float)i };
                data.SetData(dt, rgfVal1);

                db.PutRawValue(nSrcId, nValItem2, data);
                dt += TimeSpan.FromMinutes(1);
            }

            db.SaveRawValues();
            db.Close();

            return src;
        }

        [TestMethod]
        public void TestTemporalSetLoadAll()
        {
            CryptoRandom random = new CryptoRandom();
            Log log = new Log("Test Temporal Set");
            DateTime dtStart;
            DateTime dtEnd;
            DatabaseTemporal db = new DatabaseTemporal();
            int nHistSteps = 90;
            int nFutSteps = 30;
            SourceDescriptor src = loadTestData(nHistSteps, nFutSteps, out dtStart, out dtEnd);
            TemporalSet set = new TemporalSet(log, db, src, DB_LOAD_METHOD.LOAD_ALL, 0, 0, 0, random, nHistSteps, nFutSteps, 1024);
            AutoResetEvent evtCancel = new AutoResetEvent(false);

            int nStreamsNum = 2;
            int nStreamsCat = 1;

            set.Initialize(false, evtCancel);

            for (int i = 0; i < (nHistSteps + nFutSteps); i++)
            {
                int? nItemIdx = null;
                int? nValIdx = null;
                SimpleTemporalDatumCollection rgData = set.GetData(i, ref nItemIdx, ref nValIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                log.CHECK_EQ(rgData.Count, 8, "There should be 8 simple datums (static num, static cat, hist num, hist cat, fut num, fut cat, target, target hist).");

                if (rgData[0] != null)
                    log.CHECK_EQ(rgData[0].ItemCount, 1, "The static item count should = 1.");
                if (rgData[1] != null)
                    log.CHECK_EQ(rgData[1].ItemCount, 1, "The static item count should = 1.");
                if (rgData[2] != null)
                    log.CHECK_EQ(rgData[2].ItemCount, nHistSteps * nStreamsNum, "The historic item count should = " + (nHistSteps * nStreamsNum).ToString());
                if (rgData[3] != null)
                    log.CHECK_EQ(rgData[3].ItemCount, nHistSteps * nStreamsCat, "The historic item count should = " + (nHistSteps * nStreamsCat).ToString());
                if (rgData[4] != null)
                    log.CHECK_EQ(rgData[4].ItemCount, nFutSteps * 1, "The future item count should = " + nFutSteps.ToString() + ".");
                if (rgData[5] != null)
                    log.CHECK_EQ(rgData[5].ItemCount, nFutSteps * 1, "The future item count should = " + nFutSteps.ToString() + ".");
                if (rgData[6] != null)
                    log.CHECK_EQ(rgData[6].ItemCount, nFutSteps * 1, "The target item count should = " + nFutSteps.ToString() + ".");
                if (rgData[7] != null)
                    log.CHECK_EQ(rgData[7].ItemCount, nHistSteps * 1, "The target item count should = " + nHistSteps.ToString() + ".");

                // Verify the static data.float[] 
                log.CHECK(rgData[0] == null, "The static numerical data should be null.");
                float[] rgDataStatic = rgData[1].Data;
                log.CHECK_EQ(rgDataStatic[0], 1, "The static value should = 1.");

                // Verify the historical numeric data.
                float[] rgDataHistNum = rgData[2].Data;
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp1a = (i + j + 1) * 2;
                    float fExp2a = (i + j + 2) * 3;

                    int nIdx = j * nStreamsNum;
                    float fVal = rgDataHistNum[nIdx];
                    log.CHECK_EQ(fVal, fExp1a, "The value should = " + fExp1a.ToString()); // observed #1

                    fVal = rgDataHistNum[nIdx + 1];
                    log.CHECK_EQ(fVal, fExp2a, "The value should = " + fExp2a.ToString()); // observed #2
                }

                // Verify the historical categorical data.
                float[] rgDataHistCat = rgData[3].Data;
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp3a = i + j;

                    int nIdx = j * nStreamsCat;
                    float fVal = rgDataHistCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }

                // Verify the future data.
                log.CHECK(rgData[4] == null, "The future numerical data should be null.");
                float[] rgDataFutCat = rgData[5].Data;
                for (int j = 0; j < nFutSteps; j++)
                {
                    float fExp3a = i + (j + nHistSteps);

                    int nIdx = j;
                    float fVal = rgDataFutCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }
            }

            for (int i = 0; i < (nHistSteps + nFutSteps); i++)
            {
                int? nItemIdx = null;
                int? nValIdx = null;
                SimpleTemporalDatumCollection rgData = set.GetData(i, ref nItemIdx, ref nValIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                log.CHECK_EQ(rgData.Count, 8, "There should be 8 simple datums (static num, static cat, hist num, hist cat, fut num, fut cat, target, target hist).");

                if (rgData[0] != null)
                    log.CHECK_EQ(rgData[0].ItemCount, 1, "The static item count should = 1.");
                if (rgData[1] != null)
                    log.CHECK_EQ(rgData[1].ItemCount, 1, "The static item count should = 1.");
                if (rgData[2] != null)
                    log.CHECK_EQ(rgData[2].ItemCount, nHistSteps * nStreamsNum, "The historic item count should = " + (nHistSteps * nStreamsNum).ToString());
                if (rgData[3] != null)
                    log.CHECK_EQ(rgData[3].ItemCount, nHistSteps * nStreamsCat, "The historic item count should = " + (nHistSteps * nStreamsCat).ToString());
                if (rgData[4] != null)
                    log.CHECK_EQ(rgData[4].ItemCount, nFutSteps * 1, "The future item count should = " + nFutSteps.ToString() + ".");
                if (rgData[5] != null)
                    log.CHECK_EQ(rgData[5].ItemCount, nFutSteps * 1, "The future item count should = " + nFutSteps.ToString() + ".");
                if (rgData[6] != null)
                    log.CHECK_EQ(rgData[6].ItemCount, nFutSteps * 1, "The target item count should = " + nFutSteps.ToString() + ".");
                if (rgData[7] != null)
                    log.CHECK_EQ(rgData[7].ItemCount, nHistSteps * 1, "The target item count should = " + nHistSteps.ToString() + ".");

                // Verify the static data.float[] 
                log.CHECK(rgData[0] == null, "The static numerical data should be null.");
                float[] rgDataStatic = rgData[1].Data;
                log.CHECK_EQ(rgDataStatic[0], 2, "The static value should = 2.");

                // Verify the historical numeric data.
                float[] rgDataHistNum = rgData[2].Data;
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp1a = (i + j + 3) * 4;
                    float fExp2a = (i + j + 4) * 8;

                    int nIdx = j * nStreamsNum;
                    float fVal = rgDataHistNum[nIdx];
                    log.CHECK_EQ(fVal, fExp1a, "The value should = " + fExp1a.ToString()); // observed #1

                    fVal = rgDataHistNum[nIdx + 1];
                    log.CHECK_EQ(fVal, fExp2a, "The value should = " + fExp2a.ToString()); // observed #2
                }

                // Verify the historical categorical data.
                float[] rgDataHistCat = rgData[3].Data;
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp3a = i + j;

                    int nIdx = j * nStreamsCat;
                    float fVal = rgDataHistCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }

                // Verify the future data.
                log.CHECK(rgData[4] == null, "The future numerical data should be null.");
                float[] rgDataFutCat = rgData[5].Data;
                for (int j = 0; j < nFutSteps; j++)
                {
                    float fExp3a = i + (j + nHistSteps);

                    int nIdx = j;
                    float fVal = rgDataFutCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }
            }
        }

        [TestMethod]
        public void TestDirectSequential()
        {
            bool bNormalizedData = false;
            int nHistSteps = 80;
            int nFutureSteps = 30;

            Log log = new Log("Test Data Temporal");
            log.EnableTrace = true;

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadLimit = 0;
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            // Create sine curve data.
            PlotCollection plots = new PlotCollection("SineCurve");
            DateTime dt = DateTime.Now - TimeSpan.FromSeconds(1000);
            for (int i = 0; i < 1000; i++)
            {
                Plot plot = new Plot(dt.ToFileTime(), Math.Sin(i * 0.1));
                plot.Tag = dt;
                plots.Add(plot);
                dt += TimeSpan.FromSeconds(1);
            }

            // Create in-memory database.
            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

            // Create simple, single direct stream and load data.
            Tuple<DatasetDescriptor, int, int> dsd = db.CreateSimpleDirectStream("Direct", "SineCurve", s, prop, plots);
            DatasetDescriptor ds = dsd.Item1;

            // Test iterating through the data sequentially.
            int nIdx = 0;

            for (int i = 0; i < ds.TrainingSource.ImageCount - (nHistSteps + nFutureSteps); i++)
            {
                int? nItemIdx = null;
                int? nValueIdx = null;

                SimpleTemporalDatumCollection data = db.QueryTemporalItem(i, ds.TrainingSource.ID, ref nItemIdx, ref nValueIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                // Observed
                for (int j=0; j<nHistSteps; j++)
                {
                    float fActual = data[2].Data[j];
                    DateTime dtActual = ((DateTime[])data[2].Tag)[j];

                    float fExpected = plots[i + j].Y;
                    DateTime dtExpected = (DateTime)plots[i + j].Tag;

                    log.CHECK_EQ(fActual, fExpected, "The value should = " + fExpected.ToString());
                    log.CHECK_EQ(dtActual.Ticks, dtExpected.Ticks, "The date should = " + dtExpected.ToString());
                }

                // Target Hist
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fActual = data[7].Data[j];
                    DateTime dtActual = ((DateTime[])data[7].Tag)[j];

                    float fExpected = plots[i + j].Y;
                    DateTime dtExpected = (DateTime)plots[i + j].Tag;

                    log.CHECK_EQ(fActual, fExpected, "The value should = " + fExpected.ToString());
                    log.CHECK_EQ(dtActual.Ticks, dtExpected.Ticks, "The date should = " + dtExpected.ToString());
                }

                // Target
                for (int j=0; j<nFutureSteps; j++)
                {
                    float fActual = data[6].Data[j];
                    DateTime dtActual = ((DateTime[])data[6].Tag)[j];

                    float fExpected = plots[i + nHistSteps + j].Y;
                    DateTime dtExpected = (DateTime)plots[i + nHistSteps + j].Tag;

                    log.CHECK_EQ(fActual, fExpected, "The value should = " + fExpected.ToString());
                    log.CHECK_EQ(dtActual.Ticks, dtExpected.Ticks, "The date should = " + dtExpected.ToString());
                }

                nIdx++;
            }
        }

        [TestMethod]
        public void TestDirectRandom()
        {
            bool bNormalizedData = false;
            int nHistSteps = 80;
            int nFutureSteps = 30;

            Log log = new Log("Test Data Temporal");
            log.EnableTrace = true;

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadLimit = 0;
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            // Create sine curve data.
            PlotCollection plots = new PlotCollection("SineCurve");
            DateTime dt = DateTime.Now - TimeSpan.FromSeconds(1000);
            for (int i = 0; i < 1000; i++)
            {
                Plot plot = new Plot(dt.ToFileTime(), Math.Sin(i * 0.1));
                plot.Tag = dt;
                plots.Add(plot);
                dt += TimeSpan.FromSeconds(1);
            }

            // Create in-memory database.
            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

            // Create simple, single direct stream and load data.
            Tuple<DatasetDescriptor, int, int> dsd = db.CreateSimpleDirectStream("Direct", "SineCurve", s, prop, plots);
            DatasetDescriptor ds = dsd.Item1;

            // Test iterating through the data sequentially.
            int nIdx = 0;

            for (int i = 0; i < ds.TrainingSource.ImageCount - (nHistSteps + nFutureSteps); i++)
            {
                int? nItemIdx = null;
                int? nValueIdx = null;

                SimpleTemporalDatumCollection data = db.QueryTemporalItem(i, ds.TrainingSource.ID, ref nItemIdx, ref nValueIdx, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.RANDOM);

                // Observed
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fActual = data[2].Data[j];
                    DateTime dtActual = ((DateTime[])data[2].Tag)[j];

                    float fExpected = plots[nValueIdx.Value + j].Y;
                    DateTime dtExpected = (DateTime)plots[nValueIdx.Value + j].Tag;

                    log.CHECK_EQ(fActual, fExpected, "The value should = " + fExpected.ToString());
                    log.CHECK_EQ(dtActual.Ticks, dtExpected.Ticks, "The date should = " + dtExpected.ToString());
                }

                // Target Hist
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fActual = data[7].Data[j];
                    DateTime dtActual = ((DateTime[])data[7].Tag)[j];

                    float fExpected = plots[nValueIdx.Value + j].Y;
                    DateTime dtExpected = (DateTime)plots[nValueIdx.Value + j].Tag;

                    log.CHECK_EQ(fActual, fExpected, "The value should = " + fExpected.ToString());
                    log.CHECK_EQ(dtActual.Ticks, dtExpected.Ticks, "The date should = " + dtExpected.ToString());
                }

                // Target
                for (int j = 0; j < nFutureSteps; j++)
                {
                    float fActual = data[6].Data[j];
                    DateTime dtActual = ((DateTime[])data[6].Tag)[j];

                    float fExpected = plots[nValueIdx.Value + nHistSteps + j].Y;
                    DateTime dtExpected = (DateTime)plots[nValueIdx.Value + nHistSteps + j].Tag;

                    log.CHECK_EQ(fActual, fExpected, "The value should = " + fExpected.ToString());
                    log.CHECK_EQ(dtActual.Ticks, dtExpected.Ticks, "The date should = " + dtExpected.ToString());
                }

                nIdx++;
            }
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_none_train()
        {
            TestDataset_TFT_commodities_none(Phase.TRAIN);
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_none_test()
        {
            TestDataset_TFT_commodities_none(Phase.TEST);
        }

        public void TestDataset_TFT_commodities_none(Phase phase)
        {
            DatabaseLoader loader = new DatabaseLoader();
            DatasetDescriptor ds = loader.LoadDatasetFromDb("TFT.commodities");

            if (ds == null)
                return;

            Log log = new Log("Test log");
            int nHistSteps = 63;
            int nFutureSteps = 0;
            bool bNormalizedData = false;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            db.InitializeWithDsName1(s, ds.Name);
            List<DateTime> rgMasterTimeSync = db.GetMasterTimeSync(ds.ID, phase);

            //---------------------------------------------------------------------
            // Test non-random queryies.
            //---------------------------------------------------------------------

            int nColItemIdx = 0;
            int nColValueIdx = 0;
            int? nItemIdx = 0;
            int? nValueIdx = 0;
            DB_LABEL_SELECTION_METHOD? itemSel = DB_LABEL_SELECTION_METHOD.NONE;
            DB_ITEM_SELECTION_METHOD? valueSel = DB_ITEM_SELECTION_METHOD.NONE;
            int nSteps = nHistSteps + nFutureSteps;
            int nBatch = 256;
            double dfBatchQueries = (rgMasterTimeSync.Count - nSteps) / (double)nSteps;
            int nBatchQueries = (int)Math.Round(dfBatchQueries) + 1;
            Dictionary<int, int> rgAvailableValid = new Dictionary<int, int>();
            SourceDescriptor src = (phase == Phase.TRAIN) ? ds.TrainingSource : ds.TestingSource;

            for (int m = 0; m < nBatchQueries; m++)
            {
                Dictionary<int, SimpleTemporalDatumCollection> rgData = new Dictionary<int, SimpleTemporalDatumCollection>();
                int nValidCount = 0;
                List<DateTime> rgTimeSync = null;

                for (int k = 0; k < nBatch; k++)
                {
                    nItemIdx = nColItemIdx;
                    nValueIdx = nColValueIdx;

                    SimpleTemporalDatumCollection data = db.QueryTemporalItem(0, src.ID, ref nItemIdx, ref nValueIdx, itemSel, valueSel, DB_INDEX_ORDER.COL_MAJOR, true, false, true, false, null, true);

                    rgData.Add(nColItemIdx, data);
                    if (data != null)
                    {
                        // Get the time sync.
                        SimpleTemporalDatum time = data[8];
                        DateTime dtStart = time.StartTime.Value;
                        List<DateTime> rgTime = new List<DateTime>();

                        for (int j = 0; j < time.Data.Count(); j++)
                        {
                            float fSeconds = time.Data[j];
                            fSeconds *= 10000;
                            fSeconds = (float)Math.Round(fSeconds);
                            DateTime dt = dtStart.AddSeconds(fSeconds);
                            rgTime.Add(dt);
                        }

                        // Verify the time sync.
                        if (rgTimeSync == null)
                        {
                            rgTimeSync = rgTime;
                        }
                        else
                        {
                            for (int j = 0; j < rgTime.Count; j++)
                            {
                                if (rgTime[j] != rgTimeSync[j])
                                    log.FAIL("The time sync error at index " + j.ToString() + "!");
                            }
                        }

                        nValidCount++;
                    }

                    nColItemIdx++;
                }

                rgAvailableValid.Add(m, nValidCount);

                DateTime dtFirst = rgTimeSync[0];
                DateTime dtLast = rgTimeSync[rgTimeSync.Count - 1];
                List<int> rgnInvalidIdx = new List<int>();
                int nExpectedValidCount = 0;

                for (int i = 0; i < src.TemporalDescriptor.ValueItemDescriptorItems.Count; i++)
                {
                    if (src.TemporalDescriptor.ValueItemDescriptorItems[i].Start > dtFirst)
                    {
                        if (rgData[i] != null)
                            rgnInvalidIdx.Add(i);
                        continue;
                    }

                    if (src.TemporalDescriptor.ValueItemDescriptorItems[i].End < dtLast)
                    {
                        if (rgData[i] != null)
                            rgnInvalidIdx.Add(i);
                        continue;
                    }

                    nExpectedValidCount++;
                }

                log.CHECK_EQ(nExpectedValidCount, nValidCount, "The valid count should = " + nExpectedValidCount.ToString());
                log.CHECK_EQ(rgnInvalidIdx.Count, 0, "Invalid records found.");

                nColValueIdx += nSteps;
                nColItemIdx = 0;
            }

            log.EnableTrace = true;
            log.WriteLine("Available " + nSteps.ToString() + " data sets");
            int nMasterIdx = 0;

            foreach (KeyValuePair<int, int> kv in rgAvailableValid)
            {
                DateTime dt = rgMasterTimeSync[nMasterIdx];
                string strStars = "";
                strStars = strStars.PadRight(kv.Value, '*');

                log.WriteLine(dt.ToShortDateString() + "  " + kv.Key.ToString() + " = " + "(" + kv.Value.ToString() + ") " + strStars);

                nMasterIdx += nSteps;
            }
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_random_item()
        {
            DatabaseLoader loader = new DatabaseLoader();
            DatasetDescriptor ds = loader.LoadDatasetFromDb("TFT.commodities");

            if (ds == null)
                return;

            Log log = new Log("Test log");
            int nHistSteps = 63;
            int nFutureSteps = 0;
            bool bNormalizedData = false;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            db.InitializeWithDsName1(s, ds.Name);
            List<DateTime> rgMasterTimeSync = db.GetMasterTimeSync(ds.ID, Phase.TRAIN);

            //---------------------------------------------------------------------
            // Test random item queryies.
            //---------------------------------------------------------------------

            int nColItemIdx = 0;
            int nColValueIdx = 0;
            int? nItemIdx = 0;
            int? nValueIdx = 0;
            DB_LABEL_SELECTION_METHOD? itemSel = DB_LABEL_SELECTION_METHOD.NONE;
            DB_ITEM_SELECTION_METHOD? valueSel = DB_ITEM_SELECTION_METHOD.NONE;
            int nSteps = nHistSteps + nFutureSteps;
            int nBatch = 256;
            int nBatchQueries = (rgMasterTimeSync.Count - nSteps) / nSteps;
            Random random = new Random();

            for (int m = 0; m < nBatchQueries; m++)
            {
                Dictionary<int, SimpleTemporalDatumCollection> rgData = new Dictionary<int, SimpleTemporalDatumCollection>();
                int nValidCount = 0;
                List<DateTime> rgTimeSync = null;

                List<int> rgColIdx = new List<int>();
                for (int n = 0; n < ds.TrainingSource.TemporalDescriptor.ValueItemDescriptors.Count; n++)
                {
                    rgColIdx.Add(n);
                }

                for (int k = 0; k < nBatch; k++)
                {
                    int nIdx1 = random.Next(rgColIdx.Count);
                    nColItemIdx = rgColIdx[nIdx1];
                    rgColIdx.RemoveAt(nIdx1);

                    if (rgColIdx.Count == 0)
                    {
                        for (int n = 0; n < ds.TrainingSource.TemporalDescriptor.ValueItemDescriptors.Count; n++)
                        {
                            rgColIdx.Add(n);
                        }
                    }

                    nItemIdx = nColItemIdx;
                    nValueIdx = nColValueIdx;

                    SimpleTemporalDatumCollection data = db.QueryTemporalItem(0, ds.TrainingSource.ID, ref nItemIdx, ref nValueIdx, itemSel, valueSel, DB_INDEX_ORDER.COL_MAJOR, true, false, true, false, null, true);

                    bool bExists = false;
                    if (!rgData.ContainsKey(nColItemIdx))
                        rgData.Add(nColItemIdx, data);
                    else
                    {
                        rgData[nColItemIdx] = data;
                        bExists = true;
                    }

                    if (data != null)
                    {
                        // Get the time sync.
                        SimpleTemporalDatum time = data[8];
                        DateTime dtStart = time.StartTime.Value;
                        List<DateTime> rgTime = new List<DateTime>();

                        for (int j = 0; j < time.Data.Count(); j++)
                        {
                            float fSeconds = time.Data[j];
                            fSeconds *= 10000;
                            fSeconds = (float)Math.Round(fSeconds);
                            DateTime dt = dtStart.AddSeconds(fSeconds);
                            rgTime.Add(dt);
                        }

                        // Verify the time sync.
                        if (rgTimeSync == null)
                        {
                            rgTimeSync = rgTime;
                        }
                        else
                        {
                            for (int j = 0; j < rgTime.Count; j++)
                            {
                                if (rgTime[j] != rgTimeSync[j])
                                    log.FAIL("The time sync error at index " + j.ToString() + "!");
                            }
                        }

                        if (!bExists)
                            nValidCount++;
                    }
                }

                DateTime dtFirst = rgTimeSync[0];
                DateTime dtLast = rgTimeSync[rgTimeSync.Count - 1];
                List<int> rgnInvalidIdx = new List<int>();
                int nExpectedValidCount = 0;

                for (int i = 0; i < ds.TrainingSource.TemporalDescriptor.ValueItemDescriptorItems.Count; i++)
                {
                    if (ds.TrainingSource.TemporalDescriptor.ValueItemDescriptorItems[i].Start > dtFirst)
                    {
                        if (rgData[i] != null)
                            rgnInvalidIdx.Add(i);
                        continue;
                    }

                    if (ds.TrainingSource.TemporalDescriptor.ValueItemDescriptorItems[i].End < dtLast)
                    {
                        if (rgData[i] != null)
                            rgnInvalidIdx.Add(i);
                        continue;
                    }

                    nExpectedValidCount++;
                }

                log.CHECK_EQ(nExpectedValidCount, nValidCount, "The valid count should = " + nExpectedValidCount.ToString());
                log.CHECK_EQ(rgnInvalidIdx.Count, 0, "Invalid records found.");

                nColValueIdx += nSteps;
                nColItemIdx = 0;
            }
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_random_value()
        {
            DatabaseLoader loader = new DatabaseLoader();
            DatasetDescriptor ds = loader.LoadDatasetFromDb("TFT.commodities");

            if (ds == null)
                return;

            Log log = new Log("Test log");
            int nHistSteps = 63;
            int nFutureSteps = 0;
            bool bNormalizedData = false;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            db.InitializeWithDsName1(s, ds.Name);
            List<DateTime> rgMasterTimeSync = db.GetMasterTimeSync(ds.ID, Phase.TRAIN);

            //---------------------------------------------------------------------
            // Test random value queryies.
            //---------------------------------------------------------------------

            int nColItemIdx = 0;
            int nColValueIdx = 0;
            int? nItemIdx = 0;
            int? nValueIdx = 0;
            DB_LABEL_SELECTION_METHOD? itemSel = DB_LABEL_SELECTION_METHOD.NONE;
            DB_ITEM_SELECTION_METHOD? valueSel = DB_ITEM_SELECTION_METHOD.NONE;
            int nSteps = nHistSteps + nFutureSteps;
            int nBatch = 256;
            int nBatchQueries = (rgMasterTimeSync.Count - nSteps) / nSteps;
            Random random = new Random();

            List<int> rgValueIdx = new List<int>();
            for (int i=0; i<rgMasterTimeSync.Count - nSteps; i++)
            {
                rgValueIdx.Add(i);
            }

            for (int m = 0; m < nBatchQueries; m++)
            {
                Dictionary<int, SimpleTemporalDatumCollection> rgData = new Dictionary<int, SimpleTemporalDatumCollection>();
                int nValidCount = 0;

                for (int k = 0; k < nBatch; k++)
                {
                    int nIdx = random.Next(rgValueIdx.Count);
                    nColValueIdx = rgValueIdx[nIdx];
                    rgValueIdx.RemoveAt(nIdx);

                    if (rgValueIdx.Count == 0)
                    {
                        for (int i = 0; i < rgMasterTimeSync.Count - nSteps; i++)
                        {
                            rgValueIdx.Add(i);
                        }
                    }

                    nItemIdx = nColItemIdx;
                    nValueIdx = nColValueIdx;

                    SimpleTemporalDatumCollection data = db.QueryTemporalItem(0, ds.TrainingSource.ID, ref nItemIdx, ref nValueIdx, itemSel, valueSel, DB_INDEX_ORDER.COL_MAJOR, true, false, true, false, null, true);

                    rgData.Add(nColItemIdx, data);
                    if (data != null)
                    {
                        // Get the time sync.
                        SimpleTemporalDatum time = data[8];
                        DateTime dtStart = time.StartTime.Value;
                        List<DateTime> rgTime = new List<DateTime>();

                        for (int j = 0; j < time.Data.Count(); j++)
                        {
                            float fSeconds = time.Data[j];
                            fSeconds *= 10000;
                            DateTime dt = dtStart.AddSeconds(fSeconds);
                            dt = Utility.RoundDateTime(dt);
                            rgTime.Add(dt);
                        }

                        // Verify the time sync.
                        int nMasterIdx = 0;
                        for (int j=0; j<rgMasterTimeSync.Count; j++)
                        {
                            if (rgMasterTimeSync[j] == rgTime[0])
                            {
                                nMasterIdx = j;
                                break;
                            }
                        }

                        for (int j = 0; j < rgTime.Count; j++)
                        {
                            if (rgTime[j] != rgMasterTimeSync[nMasterIdx + j])
                                log.FAIL("The time sync error at index " + j.ToString() + "!");
                        }

                        nValidCount++;
                    }

                    nColItemIdx++;
                }

                nColItemIdx = 0;
            }
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_none_DataTemporalLayer_train()
        {
            TestDataset_TFT_commodities_none_DataTemporalLayer(Phase.TRAIN);
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_none_DataTemporalLayer_test()
        {
            TestDataset_TFT_commodities_none_DataTemporalLayer(Phase.TEST);
        }

        public void TestDataset_TFT_commodities_none_DataTemporalLayer(Phase phase)
        {
            DatabaseLoader loader = new DatabaseLoader();
            DatasetDescriptor ds = loader.LoadDatasetFromDb("TFT.commodities");

            if (ds == null)
                return;

            Log log = new Log("Test log");
            int nHistSteps = 63;
            int nFutureSteps = 0;
            bool bNormalizedData = false;
            int nSteps = nHistSteps + nFutureSteps;

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            MyCaffeTemporalDatabase db1 = new MyCaffeTemporalDatabase(log, prop);
            db1.InitializeWithDsName1(s, ds.Name);

            CudaDnn<float> cuda = new CudaDnn<float>(0);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data", phase);
            p.data_temporal_param.batch_size = 256;
            p.data_temporal_param.num_historical_steps = (uint)nHistSteps;
            p.data_temporal_param.num_future_steps = (uint)nFutureSteps;
            p.data_temporal_param.shuffle_value_data = false;
            p.data_temporal_param.shuffle_item_data = false;
            p.data_temporal_param.source_type = param.tft.DataTemporalParameter.SOURCE_TYPE.SQL_DB;
            p.data_temporal_param.source = "TFT.commodities";
            p.data_temporal_param.enable_column_major_ordering = true;
            p.data_temporal_param.ignore_future_data = true;
            p.data_temporal_param.output_target_historical = true;
            p.data_temporal_param.output_time = true;
            p.data_temporal_param.output_item_ids = true;
            p.data_temporal_param.value_step_size = nSteps;
            p.data_temporal_param.target_source = param.tft.DataTemporalParameter.TARGET_SOURCE.HISTORICAL;

            Layer<float> layer = Layer<float>.Create(cuda, log, p, null, db1);
            Blob<float> blobTopSn = new Blob<float>(cuda, log);
            Blob<float> blobTopSc = new Blob<float>(cuda, log);
            Blob<float> blobTopHn = new Blob<float>(cuda, log);
            Blob<float> blobTopHc = new Blob<float>(cuda, log);
            Blob<float> blobTopFn = new Blob<float>(cuda, log);
            Blob<float> blobTopFc = new Blob<float>(cuda, log);
            Blob<float> blobTopTrg = new Blob<float>(cuda, log);
            Blob<float> blobTopTrgHist = new Blob<float>(cuda, log);
            Blob<float> blobTopTime = new Blob<float>(cuda, log);
            Blob<float> blobTopID = new Blob<float>(cuda, log);

            BlobCollection<float> colTop = new BlobCollection<float>();
            BlobCollection<float> colBtm = new BlobCollection<float>();
            colTop.Add(blobTopSn);
            colTop.Add(blobTopSc);
            colTop.Add(blobTopHn);
            colTop.Add(blobTopHc);
            colTop.Add(blobTopFn);
            colTop.Add(blobTopFc);
            colTop.Add(blobTopTrg);
            colTop.Add(blobTopTrgHist);
            colTop.Add(blobTopTime);
            colTop.Add(blobTopID);

            try
            {
                layer.Setup(colBtm, colTop);


                MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

                db.InitializeWithDsName1(s, ds.Name);
                List<DateTime> rgMasterTimeSync = db.GetMasterTimeSync(ds.ID, phase);

                //---------------------------------------------------------------------
                // Test non-random queryies.
                //---------------------------------------------------------------------

                int nColItemIdx = 0;
                int nColValueIdx = 0;
                int? nItemIdx = 0;
                int? nValueIdx = 0;
                DB_LABEL_SELECTION_METHOD? itemSel = DB_LABEL_SELECTION_METHOD.NONE;
                DB_ITEM_SELECTION_METHOD? valueSel = DB_ITEM_SELECTION_METHOD.NONE;
                int nBatch = 256;
                double dfBatchQueries = (rgMasterTimeSync.Count - nSteps) / (double)nSteps;
                int nBatchQueries = (int)Math.Round(dfBatchQueries) + 1;
                Dictionary<int, int> rgAvailableValid = new Dictionary<int, int>();
                SourceDescriptor src = (phase == Phase.TRAIN) ? ds.TrainingSource : ds.TestingSource;

                for (int m = 0; m < nBatchQueries; m++)
                {
                    Dictionary<int, SimpleTemporalDatumCollection> rgData = new Dictionary<int, SimpleTemporalDatumCollection>();
                    int nValidCount = 0;
                    List<DateTime> rgTimeSync = null;

                    layer.Forward(colBtm, colTop);

                    float[] rgTime1 = blobTopTime.mutable_cpu_data;
                    float[] rgID = blobTopID.mutable_cpu_data;

                    for (int k = 0; k < nBatch; k++)
                    {
                        nItemIdx = nColItemIdx;
                        nValueIdx = nColValueIdx;

                        SimpleTemporalDatumCollection data = db.QueryTemporalItem(0, src.ID, ref nItemIdx, ref nValueIdx, itemSel, valueSel, DB_INDEX_ORDER.COL_MAJOR, true, false, true, false, null, true);

                        rgData.Add(nColItemIdx, data);
                        if (data != null)
                        {
                            // Get the time sync.
                            SimpleTemporalDatum time = data[8];
                            SimpleTemporalDatum id = data[9];
                            DateTime dtStart = time.StartTime.Value;
                            List<DateTime> rgTime = new List<DateTime>();

                            for (int j = 0; j < time.Data.Count(); j++)
                            {
                                float fSeconds = time.Data[j];
                                fSeconds *= 10000;
                                fSeconds = (float)Math.Round(fSeconds);
                                DateTime dt = dtStart.AddSeconds(fSeconds);
                                dt = Utility.RoundDateTime(dt);
                                rgTime.Add(dt);
                            }

                            // Verify the time sync.
                            if (rgTimeSync == null)
                            {
                                rgTimeSync = rgTime;
                            }
                            else
                            {
                                for (int j = 0; j < rgTime.Count; j++)
                                {
                                    if (rgTime[j] != rgTimeSync[j])
                                        log.FAIL("The time sync error at index " + j.ToString() + "!");
                                }
                            }

                            int nIdx = nSteps * nValidCount;
                            for (int j = 0; j < nSteps; j++)
                            {
                                float fSeconds = time.Data[j];
                                fSeconds *= 10000;
                                fSeconds = (float)Math.Round(fSeconds);
                                DateTime dtExpected = dtStart.AddSeconds(fSeconds);
                                dtExpected = Utility.RoundDateTime(dtExpected);

                                fSeconds = rgTime1[nIdx + j];
                                fSeconds *= 10000;
                                fSeconds = (float)Math.Round(fSeconds);
                                DateTime dtActual = dtStart.AddSeconds(fSeconds);
                                dtActual = Utility.RoundDateTime(dtActual);

                                if (dtActual != dtExpected)
                                    log.FAIL("The actual data time is the the same as the expected time!");
                            }

                            int nIdExpected = (int)id.Data[0];
                            int nIdActual = (int)rgID[nValidCount];

                            if (nIdActual != nIdExpected)
                                log.FAIL("The ID's do not match!"); 

                            nValidCount++;
                        }

                        nColItemIdx++;
                    }

                    rgAvailableValid.Add(m, nValidCount);

                    DateTime dtFirst = rgTimeSync[0];
                    DateTime dtLast = rgTimeSync[rgTimeSync.Count - 1];
                    List<int> rgnInvalidIdx = new List<int>();
                    int nExpectedValidCount = 0;

                    for (int i = 0; i < src.TemporalDescriptor.ValueItemDescriptorItems.Count; i++)
                    {
                        if (src.TemporalDescriptor.ValueItemDescriptorItems[i].Start > dtFirst)
                        {
                            if (rgData[i] != null)
                                rgnInvalidIdx.Add(i);
                            continue;
                        }

                        if (src.TemporalDescriptor.ValueItemDescriptorItems[i].End < dtLast)
                        {
                            if (rgData[i] != null)
                                rgnInvalidIdx.Add(i);
                            continue;
                        }

                        nExpectedValidCount++;
                    }

                    log.CHECK_EQ(nExpectedValidCount, nValidCount, "The valid count should = " + nExpectedValidCount.ToString());
                    log.CHECK_EQ(rgnInvalidIdx.Count, 0, "Invalid records found.");

                    nColValueIdx += nSteps;
                    nColItemIdx = 0;
                }

                log.EnableTrace = true;
                log.WriteLine("Available " + nSteps.ToString() + " data sets");
                int nMasterIdx = 0;

                foreach (KeyValuePair<int, int> kv in rgAvailableValid)
                {
                    DateTime dt = rgMasterTimeSync[nMasterIdx];
                    string strStars = "";
                    strStars = strStars.PadRight(kv.Value, '*');

                    log.WriteLine(dt.ToShortDateString() + "  " + kv.Key.ToString() + " = " + "(" + kv.Value.ToString() + ") " + strStars);

                    nMasterIdx += nSteps;
                }
            }
            finally
            {
                if (colTop != null)
                    colTop.Dispose();

                if (layer != null)
                    layer.Dispose();

                if (cuda != null)
                    cuda.Dispose();
            }
        }

        [TestMethod]
        public void TestDataset_TFT_commodities_random_DataTemporalLayer_train()
        {
            TestDataset_TFT_commodities_random_DataTemporalLayer(Phase.TRAIN);
        }

        public void TestDataset_TFT_commodities_random_DataTemporalLayer(Phase phase)
        {
            DatabaseLoader loader = new DatabaseLoader();
            DatasetDescriptor ds = loader.LoadDatasetFromDb("TFT.commodities");

            if (ds == null)
                return;

            Log log = new Log("Test log");
            int nHistSteps = 63;
            int nFutureSteps = 0;
            bool bNormalizedData = false;
            int nSteps = nHistSteps + nFutureSteps;

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            MyCaffeTemporalDatabase db1 = new MyCaffeTemporalDatabase(log, prop);
            db1.InitializeWithDsName1(s, ds.Name);

            CudaDnn<float> cuda = new CudaDnn<float>(0);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data", phase);
            p.data_temporal_param.batch_size = 256;
            p.data_temporal_param.num_historical_steps = (uint)nHistSteps;
            p.data_temporal_param.num_future_steps = (uint)nFutureSteps;
            p.data_temporal_param.shuffle_value_data = true;
            p.data_temporal_param.shuffle_item_data = true;
            p.data_temporal_param.source_type = param.tft.DataTemporalParameter.SOURCE_TYPE.SQL_DB;
            p.data_temporal_param.source = "TFT.commodities";
            p.data_temporal_param.enable_column_major_ordering = true;
            p.data_temporal_param.ignore_future_data = true;
            p.data_temporal_param.output_target_historical = true;
            p.data_temporal_param.output_time = true;
            p.data_temporal_param.output_item_ids = true;
            p.data_temporal_param.value_step_size = 1;
            p.data_temporal_param.target_source = param.tft.DataTemporalParameter.TARGET_SOURCE.HISTORICAL;
            p.data_temporal_param.seed = 142;

            Layer<float> layer = Layer<float>.Create(cuda, log, p, null, db1);
            Blob<float> blobTopSn = new Blob<float>(cuda, log);
            Blob<float> blobTopSc = new Blob<float>(cuda, log);
            Blob<float> blobTopHn = new Blob<float>(cuda, log);
            Blob<float> blobTopHc = new Blob<float>(cuda, log);
            Blob<float> blobTopFn = new Blob<float>(cuda, log);
            Blob<float> blobTopFc = new Blob<float>(cuda, log);
            Blob<float> blobTopTrg = new Blob<float>(cuda, log);
            Blob<float> blobTopTrgHist = new Blob<float>(cuda, log);
            Blob<float> blobTopTime = new Blob<float>(cuda, log);
            Blob<float> blobTopID = new Blob<float>(cuda, log);

            BlobCollection<float> colTop = new BlobCollection<float>();
            BlobCollection<float> colBtm = new BlobCollection<float>();
            colTop.Add(blobTopSn);
            colTop.Add(blobTopSc);
            colTop.Add(blobTopHn);
            colTop.Add(blobTopHc);
            colTop.Add(blobTopFn);
            colTop.Add(blobTopFc);
            colTop.Add(blobTopTrg);
            colTop.Add(blobTopTrgHist);
            colTop.Add(blobTopTime);
            colTop.Add(blobTopID);

            try
            {
                layer.Setup(colBtm, colTop);


                MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);

                db.InitializeWithDsName1(s, ds.Name);
                List<DateTime> rgMasterTimeSync = db.GetMasterTimeSync(ds.ID, phase);

                //---------------------------------------------------------------------
                // Test non-random queryies.
                //---------------------------------------------------------------------

                int nColItemIdx = 0;
                int nColValueIdx = 0;
                int? nItemIdx = 0;
                int? nValueIdx = 0;
                DB_LABEL_SELECTION_METHOD? itemSel = DB_LABEL_SELECTION_METHOD.NONE;
                DB_ITEM_SELECTION_METHOD? valueSel = DB_ITEM_SELECTION_METHOD.NONE;
                int nBatch = 256;
                double dfBatchQueries = (rgMasterTimeSync.Count - nSteps) / (double)nSteps;
                int nBatchQueries = (int)Math.Round(dfBatchQueries) + 1;
                Dictionary<int, int> rgAvailableValid = new Dictionary<int, int>();
                SourceDescriptor src = (phase == Phase.TRAIN) ? ds.TrainingSource : ds.TestingSource;
                Random random = new Random(142);

                List<int> rgValueIdx = new List<int>();
                for (int i = 0; i < rgMasterTimeSync.Count - nSteps; i++)
                {
                    rgValueIdx.Add(i);
                }

                for (int m = 0; m < nBatchQueries; m++)
                {
                    int nValidCount = 0;
                    List<List<DateTime>> rgrgTimeSync = new List<List<DateTime>>();
                    List<int> rgIdActual = new List<int>();

                    layer.Forward(colBtm, colTop);

                    float[] rgTime1 = blobTopTime.mutable_cpu_data;
                    float[] rgID = blobTopID.mutable_cpu_data;

                    List<int> rgColIdx = new List<int>();
                    for (int n = 0; n < ds.TrainingSource.TemporalDescriptor.ValueItemDescriptors.Count; n++)
                    {
                        rgColIdx.Add(n);
                    }

                    while (rgIdActual.Count < nBatch)
                    {
                        int nIdx1 = random.Next(rgColIdx.Count);
                        nColItemIdx = rgColIdx[nIdx1];
                        rgColIdx.RemoveAt(nIdx1);

                        if (rgColIdx.Count == 0)
                        {
                            for (int n = 0; n < ds.TrainingSource.TemporalDescriptor.ValueItemDescriptors.Count; n++)
                            {
                                rgColIdx.Add(n);
                            }
                        }

                        nIdx1 = random.Next(rgValueIdx.Count);
                        nColValueIdx = rgValueIdx[nIdx1];
                        rgValueIdx.RemoveAt(nIdx1);

                        if (rgValueIdx.Count == 0)
                        {
                            for (int i = 0; i < rgMasterTimeSync.Count - nSteps; i++)
                            {
                                rgValueIdx.Add(i);
                            }
                        }

                        nItemIdx = nColItemIdx;
                        nValueIdx = nColValueIdx;

                        SimpleTemporalDatumCollection data = db.QueryTemporalItem(0, src.ID, ref nItemIdx, ref nValueIdx, itemSel, valueSel, DB_INDEX_ORDER.COL_MAJOR, true, false, true, false, null, true);

                        if (data != null)
                        {
                            // Get the time sync.
                            SimpleTemporalDatum time = data[8];
                            SimpleTemporalDatum id = data[9];
                            DateTime dtStart = time.StartTime.Value;
                            List<DateTime> rgTimeSync = new List<DateTime>();

                            int nIdx = nSteps * nValidCount;
                            for (int j = 0; j < nSteps; j++)
                            {
                                float fSeconds = rgTime1[nIdx + j];
                                if (fSeconds == 0)
                                    log.FAIL("The time should not be 0!");

                                fSeconds *= 10000;
                                fSeconds = (float)Math.Round(fSeconds);
                                DateTime dtActual = dtStart.AddSeconds(fSeconds);
                                dtActual = Utility.RoundDateTime(dtActual);
                                rgTimeSync.Add(dtActual);
                            }

                            rgrgTimeSync.Add(rgTimeSync);

                            int nIdActual = (int)rgID[nValidCount];
                            if (nIdActual == 0)
                                log.FAIL("The ID should not be 0!");

                            rgIdActual.Add(nIdActual);

                            nValidCount++;
                        }

                        nColItemIdx++;
                    }

                    rgAvailableValid.Add(m, nValidCount);

                    List<int> rgnInvalidIdx = new List<int>();
                    int nExpectedValidCount = 0;

                    for (int i = 0; i < rgIdActual.Count; i++)
                    {
                        DateTime dtFirst = rgrgTimeSync[i][0];
                        DateTime dtLast = rgrgTimeSync[i][rgrgTimeSync[i].Count - 1];
                        int nIdActual = rgIdActual[i];

                        int nIdxItem = 0;
                        for (int j=0; j<src.TemporalDescriptor.ValueItemDescriptorItems.Count; j++)
                        {
                            if (src.TemporalDescriptor.ValueItemDescriptorItems[j].Index == nIdActual)
                            {
                                nIdxItem = j;
                                break;
                            }
                        }

                        if (src.TemporalDescriptor.ValueItemDescriptorItems[nIdxItem].Start > dtFirst)
                            continue;

                        if (src.TemporalDescriptor.ValueItemDescriptorItems[nIdxItem].End < dtLast)
                            continue;

                        nExpectedValidCount++;
                    }

                    log.CHECK_EQ(nExpectedValidCount, nValidCount, "The valid count should = " + nExpectedValidCount.ToString());
                    log.CHECK_EQ(rgnInvalidIdx.Count, 0, "Invalid records found.");

                    nColValueIdx += nSteps;
                    nColItemIdx = 0;
                }

                log.EnableTrace = true;
                log.WriteLine("Available " + nSteps.ToString() + " data sets");
                int nMasterIdx = 0;

                foreach (KeyValuePair<int, int> kv in rgAvailableValid)
                {
                    DateTime dt = rgMasterTimeSync[nMasterIdx];
                    string strStars = "";
                    strStars = strStars.PadRight(kv.Value, '*');

                    log.WriteLine(dt.ToShortDateString() + "  " + kv.Key.ToString() + " = " + "(" + kv.Value.ToString() + ") " + strStars);

                    nMasterIdx += nSteps;
                }
            }
            finally
            {
                if (colTop != null)
                    colTop.Dispose();

                if (layer != null)
                    layer.Dispose();

                if (cuda != null)
                    cuda.Dispose();
            }
        }
    }
}
