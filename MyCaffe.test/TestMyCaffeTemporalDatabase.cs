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

            int nValStrm1a = db.AddValueStream(nValItem1, "Test Stream #1", DatabaseTemporal.STREAM_CLASS_TYPE.OBSERVED, DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 1, dtStart, dtEnd, 60);
            int nValStrm2a = db.AddValueStream(nValItem1, "Test Stream #2", DatabaseTemporal.STREAM_CLASS_TYPE.OBSERVED, DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 2, dtStart, dtEnd, 60);
            int nValStrm3a = db.AddValueStream(nValItem1, "Test Stream #3", DatabaseTemporal.STREAM_CLASS_TYPE.KNOWN, DatabaseTemporal.STREAM_VALUE_TYPE.CATEGORICAL, 3, dtStart, dtEnd, 60);

            int nValStrm1b = db.AddValueStream(nValItem2, "Test Stream #1", DatabaseTemporal.STREAM_CLASS_TYPE.OBSERVED, DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 1, dtStart, dtEnd, 60);
            int nValStrm2b = db.AddValueStream(nValItem2, "Test Stream #2", DatabaseTemporal.STREAM_CLASS_TYPE.OBSERVED, DatabaseTemporal.STREAM_VALUE_TYPE.NUMERIC, 2, dtStart, dtEnd, 60);
            int nValStrm3b = db.AddValueStream(nValItem2, "Test Stream #3", DatabaseTemporal.STREAM_CLASS_TYPE.KNOWN, DatabaseTemporal.STREAM_VALUE_TYPE.CATEGORICAL, 3, dtStart, dtEnd, 60);

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
            db.Close();

            plots = new PlotCollection();
            dt = dtStart;
            for (int i = 0; i < nSteps * nBlocks; i++)
            {
                float[] rgfVal1 = new float[] { (float)((i+1) * 2), (float)((i+2) * 3), (float)i };

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
            TemporalSet set = new TemporalSet(log, db, src, DB_LOAD_METHOD.LOAD_ALL, 0, random, dtStart, dtEnd, nHistSteps, nFutSteps, 1024);

            int nSteps = nHistSteps + nFutSteps;
            int nStreams = 3;

            set.Initialize();

            for (int i = 0; i < nSteps; i++)
            {
                SimpleDatum sd = set.GetTemporalData(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, 1);

                log.CHECK_EQ(sd.ItemCount, nSteps * nStreams, "The item count should = " + (nSteps * nStreams).ToString());

                float[] rgData = sd.GetData<float>();
                for (int j = 0; j < nSteps; j++)
                {
                    float fExp1a = (i + j + 1) * 2;
                    float fExp2a = (i + j + 2) * 3;
                    float fExp3a = i + j;

                    int nIdx = j * nStreams;
                    float fVal = rgData[nIdx];
                    log.CHECK_EQ(fVal, fExp1a, "The value should = " + fExp1a.ToString());

                    fVal = rgData[nIdx + 1];
                    log.CHECK_EQ(fVal, fExp2a, "The value should = " + fExp2a.ToString());

                    fVal = rgData[nIdx + 2];
                    log.CHECK_EQ(fVal, fExp3a, "The value should = " + fExp3a.ToString());
                }
            }

            for (int i = 0; i < nSteps; i++)
            {
                SimpleDatum sd = set.GetTemporalData(DB_LABEL_SELECTION_METHOD.NONE, DB_ITEM_SELECTION_METHOD.NONE, 1);

                log.CHECK_EQ(sd.ItemCount, nSteps * nStreams, "The item count should = " + (nSteps * nStreams).ToString());

                float[] rgData = sd.GetData<float>();
                for (int j = 0; j < nSteps; j++)
                {
                    float fExp1b = (i + j + 1) * 2;
                    float fExp2b = (i + j + 2) * 3;
                    float fExp3b = i + j;

                    int nIdx = j * nStreams;
                    double dfVal = rgData[nIdx];
                    log.CHECK_EQ(dfVal, fExp1b, "The value should = " + fExp1b.ToString());

                    dfVal = rgData[nIdx + 1];
                    log.CHECK_EQ(dfVal, fExp2b, "The value should = " + fExp2b.ToString());

                    dfVal = rgData[nIdx + 2];
                    log.CHECK_EQ(dfVal, fExp3b, "The value should = " + fExp3b.ToString());
                }
            }
        }
    }
}
