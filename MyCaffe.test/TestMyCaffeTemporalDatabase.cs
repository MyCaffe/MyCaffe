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

            int nValItem1 = db.AddValueItem(nSrcId, "Test Value Item #1");
            int nValItem2 = db.AddValueItem(nSrcId, "Test Value ITem #2");

            int nValStrm1a = db.AddObservedValueStream(nSrcId, nValItem1, "Test Stream #1", DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 1, dtStart, dtEnd, 60);
            int nValStrm2a = db.AddObservedValueStream(nSrcId, nValItem1, "Test Stream #2", DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 2, dtStart, dtEnd, 60);
            int nValStrm3a = db.AddKnownValueStream(nSrcId, nValItem1, "Test Stream #3", DatabaseTemporal.STREAM_VALUE_TYPE.CATEGORICAL, 3, dtStart, dtEnd, 60);
            int nValStrm4a = db.AddStaticValueStream(nSrcId, nValItem1, "Test Stream #4", DatabaseTemporal.STREAM_VALUE_TYPE.CATEGORICAL, 3);

            int nValStrm1b = db.AddObservedValueStream(nSrcId, nValItem2, "Test Stream #1", DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 1, dtStart, dtEnd, 60);
            int nValStrm2b = db.AddObservedValueStream(nSrcId, nValItem2, "Test Stream #2", DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 2, dtStart, dtEnd, 60);
            int nValStrm3b = db.AddKnownValueStream(nSrcId, nValItem2, "Test Stream #3", DatabaseTemporal.STREAM_VALUE_TYPE.CATEGORICAL, 3, dtStart, dtEnd, 60);
            int nValStrm4b = db.AddStaticValueStream(nSrcId, nValItem2, "Test Stream #4", DatabaseTemporal.STREAM_VALUE_TYPE.CATEGORICAL, 3);

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
            db.PutRawValues(nSrcId, nValItem1, nValStrm4a, 1);
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
            db.PutRawValues(nSrcId, nValItem2, nValStrm4b, 2);
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

            int nStreams = 3;

            set.Initialize(false, evtCancel);

            for (int i = 0; i < (nHistSteps + nFutSteps); i++)
            {
                Tuple<SimpleDatum, SimpleDatum, SimpleDatum> data = set.GetData(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                log.CHECK_EQ(data.Item1.ItemCount, 1, "The static item count should = 1.");
                log.CHECK_EQ(data.Item2.ItemCount, nHistSteps * nStreams, "The historic item count should = " + (nHistSteps * nStreams).ToString());
                log.CHECK_EQ(data.Item3.ItemCount, nFutSteps * 1, "The future item count should = " + nFutSteps.ToString() + ".");

                // Verify the static data.float[] 
                float[] rgDataStatic = data.Item1.GetData<float>();
                log.CHECK_EQ(rgDataStatic[0], 1, "The static value should = " + (i + 1).ToString() + ".");

                // Verify the historical data.
                float[] rgDataHist = data.Item2.GetData<float>();
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp1a = (i + j + 1) * 2;
                    float fExp2a = (i + j + 2) * 3;
                    float fExp3a = i + j;

                    int nIdx = j * nStreams;
                    float fVal = rgDataHist[nIdx];
                    log.CHECK_EQ(fVal, fExp1a, "The value should = " + fExp1a.ToString()); // observed #1

                    fVal = rgDataHist[nIdx + 1];
                    log.CHECK_EQ(fVal, fExp2a, "The value should = " + fExp2a.ToString()); // observed #2

                    fVal = rgDataHist[nIdx + 2];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }

                // Verify the future data.
                float[] rgDataFut = data.Item3.GetData<float>();
                for (int j = 0; j < nFutSteps; j++)
                {
                    float fExp3a = i + (j + nHistSteps);

                    int nIdx = j;
                    float fVal = rgDataFut[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }
            }

            for (int i = 0; i < (nHistSteps + nFutSteps); i++)
            {
                Tuple<SimpleDatum, SimpleDatum, SimpleDatum> data = set.GetData(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE);

                log.CHECK_EQ(data.Item1.ItemCount, 1, "The static item count should = 1.");
                log.CHECK_EQ(data.Item2.ItemCount, nHistSteps * nStreams, "The historic item count should = " + (nHistSteps * nStreams).ToString());
                log.CHECK_EQ(data.Item3.ItemCount, nFutSteps * 1, "The future item count should = " + nFutSteps.ToString() + ".");

                // Verify the static data.float[] 
                float[] rgDataStatic = data.Item1.GetData<float>();
                log.CHECK_EQ(rgDataStatic[0], 2, "The static value should = 2.");

                // Verify the historical data.
                float[] rgDataHist = data.Item2.GetData<float>();
                for (int j = 0; j < nHistSteps; j++)
                {
                    float fExp1a = (i + j + 3) * 4;
                    float fExp2a = (i + j + 4) * 8;
                    float fExp3a = i + j;

                    int nIdx = j * nStreams;
                    float fVal = rgDataHist[nIdx];
                    log.CHECK_EQ(fVal, fExp1a, "The value should = " + fExp1a.ToString());

                    fVal = rgDataHist[nIdx + 1];
                    log.CHECK_EQ(fVal, fExp2a, "The value should = " + fExp2a.ToString());

                    fVal = rgDataHist[nIdx + 2];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString());
                }

                // Verify the future data.
                float[] rgDataFut = data.Item3.GetData<float>();
                for (int j = 0; j < nFutSteps; j++)
                {
                    float fExp3a = i + (j + nHistSteps);

                    int nIdx = j;
                    float fVal = rgDataFut[nIdx];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString()); // known #1
                }
            }
        }
    }
}
