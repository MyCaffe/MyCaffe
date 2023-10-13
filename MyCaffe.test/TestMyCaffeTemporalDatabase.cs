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
    }
}
