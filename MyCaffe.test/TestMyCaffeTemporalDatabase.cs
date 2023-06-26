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

            int nValStrm1a = db.AddObservedValueStream(nSrcId, nValItem1, "Test Stream #1", ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, 1, dtStart, dtEnd, 60);
            int nValStrm2a = db.AddObservedValueStream(nSrcId, nValItem1, "Test Stream #2", ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, 2, dtStart, dtEnd, 60);
            int nValStrm3a = db.AddKnownValueStream(nSrcId, nValItem1, "Test Stream #3", ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, 3, dtStart, dtEnd, 60);
            int nValStrm4a = db.AddStaticValueStream(nSrcId, nValItem1, "Test Stream #4", ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, 3);

            int nValStrm1b = db.AddObservedValueStream(nSrcId, nValItem2, "Test Stream #1", ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, 1, dtStart, dtEnd, 60);
            int nValStrm2b = db.AddObservedValueStream(nSrcId, nValItem2, "Test Stream #2", ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC, 2, dtStart, dtEnd, 60);
            int nValStrm3b = db.AddKnownValueStream(nSrcId, nValItem2, "Test Stream #3", ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, 3, dtStart, dtEnd, 60);
            int nValStrm4b = db.AddStaticValueStream(nSrcId, nValItem2, "Test Stream #4", ValueStreamDescriptor.STREAM_VALUE_TYPE.CATEGORICAL, 3);

            PlotCollection plots = new PlotCollection();
            DateTime dt = dtStart;
            for (int i = 0; i < nSteps * nBlocks; i++)
            {
                float[] rgfVal1 = new float[] { (float)((i + 1) * 2), (float)((i + 2) * 3), (float)i };

                Plot plot = new Plot(dt.ToFileTimeUtc(), rgfVal1);
                plot.Tag = dt;
                plots.Add(plot);

                dt += TimeSpan.FromMinutes(1);
            }
            plots.SetParameter("Test Stream #1", nValStrm1a);
            plots.SetParameter("Test Stream #2", nValStrm2a);
            plots.SetParameter("Test Stream #3", nValStrm3a);

            db.Open(nSrcId);
            db.PutRawValues(nSrcId, nValItem1, plots);
            db.PutRawValue(nSrcId, nValItem1, nValStrm4a, 1);
            db.Close();

            plots = new PlotCollection();
            dt = dtStart;
            for (int i = 0; i < nSteps * nBlocks; i++)
            {
                float[] rgfVal1 = new float[] { (float)((i + 3) * 4), (float)((i + 4) * 8), (float)i };

                Plot plot = new Plot(dt.ToFileTimeUtc(), rgfVal1);
                plot.Tag = dt;
                plots.Add(plot);

                dt += TimeSpan.FromMinutes(1);
            }
            plots.SetParameter("Test Stream #1", nValStrm1b);
            plots.SetParameter("Test Stream #2", nValStrm2b);
            plots.SetParameter("Test Stream #3", nValStrm3b);

            db.Open(nSrcId);
            db.PutRawValues(nSrcId, nValItem2, plots);
            db.PutRawValue(nSrcId, nValItem2, nValStrm4b, 2);
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
            TemporalSet set = new TemporalSet(log, db, src, DB_LOAD_METHOD.LOAD_ALL, 0, random, nHistSteps, nFutSteps, 1024);
            AutoResetEvent evtCancel = new AutoResetEvent(false);

            int nStreamsNum = 2;
            int nStreamsCat = 1;

            set.Initialize(false, evtCancel);

            for (int i = 0; i < (nHistSteps + nFutSteps); i++)
            {
                SimpleDatum[] rgData = set.GetData(i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                log.CHECK_EQ(rgData.Length, 8, "There should be 8 simple datums (static num, static cat, hist num, hist cat, fut num, fut cat, target, target hist).");

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
                float[] rgDataStatic = rgData[1].GetData<float>();
                log.CHECK_EQ(rgDataStatic[0], 1, "The static value should = 1.");

                // Verify the historical numeric data.
                float[] rgDataHistNum = rgData[2].GetData<float>();
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
                float[] rgDataHistCat = rgData[3].GetData<float>();
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp3a = i + j;

                    int nIdx = j * nStreamsCat;
                    float fVal = rgDataHistCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }

                // Verify the future data.
                log.CHECK(rgData[4] == null, "The future numerical data should be null.");
                float[] rgDataFutCat = rgData[5].GetData<float>();
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
                SimpleDatum[] rgData = set.GetData(i, DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                log.CHECK_EQ(rgData.Length, 8, "There should be 8 simple datums (static num, static cat, hist num, hist cat, fut num, fut cat, target, target hist).");

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
                float[] rgDataStatic = rgData[1].GetData<float>();
                log.CHECK_EQ(rgDataStatic[0], 2, "The static value should = 2.");

                // Verify the historical numeric data.
                float[] rgDataHistNum = rgData[2].GetData<float>();
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
                float[] rgDataHistCat = rgData[3].GetData<float>();
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp3a = i + j;

                    int nIdx = j * nStreamsCat;
                    float fVal = rgDataHistCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }

                // Verify the future data.
                log.CHECK(rgData[4] == null, "The future numerical data should be null.");
                float[] rgDataFutCat = rgData[5].GetData<float>();
                for (int j = 0; j < nFutSteps; j++)
                {
                    float fExp3a = i + (j + nHistSteps);

                    int nIdx = j;
                    float fVal = rgDataFutCat[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }
            }
        }
    }
}
