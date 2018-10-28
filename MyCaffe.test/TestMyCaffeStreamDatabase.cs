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
using MyCaffe.db.stream;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffeStreamDatabase
    {
        [TestMethod]
        public void TestInitializationSimple()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=1;";
            string strParam = "Connection=TestCon;Table=TestTbl;Field=TestField;";

            strParam = Utility.Replace(strParam, ';', '|');
            strParam = Utility.Replace(strParam, '=', '~');

            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());

            string strSettings = "QueryCount=5;Start=" + DateTime.Today.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";            
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);
        }

        [TestMethod]
        public void TestQuerySimple()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=1;";
            string strParam = "Connection=TestCon;Table=TestTbl;Field=TestField;";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());

            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + DateTime.Today.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 2 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 2, "The query size item 1 should be 2 for two fields.");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            SimpleDatum sd;
            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            for (int i = 0; i < 2; i++)
            {
                sd = db.Query(int.MaxValue);
                log.CHECK(sd != null, "The SD returned should not be null.");
                log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                for (int j = 0; j < nW; j++)
                {
                    DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                    log.CHECK(dt1 == dt, "The time sync is incorrect.");
                    dt += TimeSpan.FromMinutes(1);

                    double df = sd.RealData[nW + j];
                    int nVal = (int)df;
                    log.CHECK_EQ(nVal, nDataIdx, "The data value is incorrect.");
                    nDataIdx++;
                }
            }

            db.Shutdown();
        }

        [TestMethod]
        public void TestQuerySimpleDual()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=2;";
            string strParam = "Connection=TestCon;Table=TestTbl;Field=TestField;";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";
            strSchema += "Connection1_CustomQueryName=Test2;";
            strSchema += "Connection1_CustomQueryParam=" + strParam + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery2());
            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + dt.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 3, "The query size item 1 should be 3 for three fields (q1:sync,data; q2:data).");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            SimpleDatum sd;
            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            for (int i = 0; i < 2; i++)
            {
                sd = db.Query(int.MaxValue);
                log.CHECK(sd != null, "The SD returned should not be null.");
                log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                for (int j = 0; j < nW; j++)
                {
                    DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                    log.CHECK(dt1 == dt, "The time sync is incorrect.");
                    dt += TimeSpan.FromMinutes(1);

                    double df1 = sd.RealData[(nW * 1) + j];
                    int nVal1 = (int)df1;
                    log.CHECK_EQ(nVal1, nDataIdx, "The data value is incorrect.");

                    double df2 = sd.RealData[(nW * 2) + j];
                    int nVal2 = (int)df2;
                    log.CHECK_EQ(nVal2, nDataIdx, "The data value is incorrect.");

                    nDataIdx++;
                }
            }

            db.Shutdown();
        }

        [TestMethod]
        public void TestQuerySimpleDualSameEnd()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=2;";
            string strParam = "Connection=TestCon;Table=TestTbl;Field=TestField;EndIdx=10;";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";
            strSchema += "Connection1_CustomQueryName=Test2;";
            strSchema += "Connection1_CustomQueryParam=" + strParam + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery2());
            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + dt.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 3, "The query size item 1 should be 3 for three fields (q1:sync,data; q2:data).");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            SimpleDatum sd;
            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            for (int i = 0; i < 3; i++)
            {
                sd = db.Query(int.MaxValue);

                if (i < 2)
                {
                    log.CHECK(sd != null, "The SD returned should not be null.");
                    log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                    log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                    log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                    log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                    for (int j = 0; j < nW; j++)
                    {
                        DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                        log.CHECK(dt1 == dt, "The time sync is incorrect.");
                        dt += TimeSpan.FromMinutes(1);

                        double df1 = sd.RealData[(nW * 1) + j];
                        int nVal1 = (int)df1;
                        log.CHECK_EQ(nVal1, nDataIdx, "The data value is incorrect.");

                        double df2 = sd.RealData[(nW * 2) + j];
                        int nVal2 = (int)df2;
                        log.CHECK_EQ(nVal2, nDataIdx, "The data value is incorrect.");

                        nDataIdx++;
                    }
                }
                else
                {
                    log.CHECK(sd == null, "Since we are past the end, the sd should be null.");
                }
            }

            db.Shutdown();
        }

        [TestMethod]
        public void TestQuerySimpleDualDifferentEnd()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=2;";
            string strParam1 = "Connection=TestCon;Table=TestTbl;Field=TestField;EndIdx=10;";
            string strParam2 = "Connection=TestCon;Table=TestTbl;Field=TestField;EndIdx=15;";

            strParam1 = ParamPacker.Pack(strParam1);
            strParam2 = ParamPacker.Pack(strParam2);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam1 + ";";
            strSchema += "Connection1_CustomQueryName=Test2;";
            strSchema += "Connection1_CustomQueryParam=" + strParam2 + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery2());
            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + dt.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 3, "The query size item 1 should be 3 for three fields (q1:sync,data; q2:data).");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            SimpleDatum sd;
            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            for (int i = 0; i < 3; i++)
            {
                sd = db.Query(int.MaxValue);

                if (i < 2)
                {
                    log.CHECK(sd != null, "The SD returned should not be null.");
                    log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                    log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                    log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                    log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                    for (int j = 0; j < nW; j++)
                    {
                        DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                        log.CHECK(dt1 == dt, "The time sync is incorrect.");
                        dt += TimeSpan.FromMinutes(1);

                        double df1 = sd.RealData[(nW * 1) + j];
                        int nVal1 = (int)df1;
                        log.CHECK_EQ(nVal1, nDataIdx, "The data value is incorrect.");

                        double df2 = sd.RealData[(nW * 2) + j];
                        int nVal2 = (int)df2;
                        log.CHECK_EQ(nVal2, nDataIdx, "The data value is incorrect.");

                        nDataIdx++;
                    }
                }
                else
                {
                    log.CHECK(sd == null, "Since we are past the end, the sd should be null.");
                }
            }

            db.Shutdown();
        }

        [TestMethod]
        public void TestQuerySimpleDualSameEndAndReset()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=2;";
            string strParam = "Connection=TestCon;Table=TestTbl;Field=TestField;EndIdx=10;";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";
            strSchema += "Connection1_CustomQueryName=Test2;";
            strSchema += "Connection1_CustomQueryParam=" + strParam + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery2());
            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + dt.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 3, "The query size item 1 should be 3 for three fields (q1:sync,data; q2:data).");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            SimpleDatum sd;
            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            for (int i = 0; i < 6; i++)
            {
                sd = db.Query(int.MaxValue);

                if (i == 0 || i == 1 || i == 3 || i == 4)
                {
                    log.CHECK(sd != null, "The SD returned should not be null.");
                    log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                    log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                    log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                    log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                    for (int j = 0; j < nW; j++)
                    {
                        DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                        log.CHECK(dt1 == dt, "The time sync is incorrect.");
                        dt += TimeSpan.FromMinutes(1);

                        double df1 = sd.RealData[(nW * 1) + j];
                        int nVal1 = (int)df1;
                        log.CHECK_EQ(nVal1, nDataIdx, "The data value is incorrect.");

                        double df2 = sd.RealData[(nW * 2) + j];
                        int nVal2 = (int)df2;
                        log.CHECK_EQ(nVal2, nDataIdx, "The data value is incorrect.");

                        nDataIdx++;
                    }
                }
                else
                {
                    log.CHECK(sd == null, "Since we are past the end, the sd should be null.");
                    dt = DateTime.Today;
                    nDataIdx = 0;
                    db.Reset(0);
                }
            }

            db.Shutdown();
        }

        [TestMethod]
        public void TestQuerySimpleDualDifferentEndAndReset()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=2;";
            string strParam1 = "Connection=TestCon;Table=TestTbl;Field=TestField;EndIdx=10;";
            string strParam2 = "Connection=TestCon;Table=TestTbl;Field=TestField;EndIdx=15;";

            strParam1 = ParamPacker.Pack(strParam1);
            strParam2 = ParamPacker.Pack(strParam2);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam1 + ";";
            strSchema += "Connection1_CustomQueryName=Test2;";
            strSchema += "Connection1_CustomQueryParam=" + strParam2 + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery2());
            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + dt.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 3, "The query size item 1 should be 3 for three fields (q1:sync,data; q2:data).");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            SimpleDatum sd;
            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            for (int i = 0; i < 6; i++)
            {
                sd = db.Query(int.MaxValue);

                if (i == 0 || i == 1 || i == 3 || i == 4)
                {
                    log.CHECK(sd != null, "The SD returned should not be null.");
                    log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                    log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                    log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                    log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                    for (int j = 0; j < nW; j++)
                    {
                        DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                        log.CHECK(dt1 == dt, "The time sync is incorrect.");
                        dt += TimeSpan.FromMinutes(1);

                        double df1 = sd.RealData[(nW * 1) + j];
                        int nVal1 = (int)df1;
                        log.CHECK_EQ(nVal1, nDataIdx, "The data value is incorrect.");

                        double df2 = sd.RealData[(nW * 2) + j];
                        int nVal2 = (int)df2;
                        log.CHECK_EQ(nVal2, nDataIdx, "The data value is incorrect.");

                        nDataIdx++;
                    }
                }
                else
                {
                    log.CHECK(sd == null, "Since we are past the end, the sd should be null.");
                    dt = DateTime.Today;
                    nDataIdx = 0;
                    db.Reset(0);
                }
            }

            db.Shutdown();
        }

        [TestMethod]
        public void TestQuerySimpleTrippleStressTiming()
        {
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=3;";
            string strParam = "Connection=TestCon;Table=TestTbl;Field=TestField;";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=Test1;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";
            strSchema += "Connection1_CustomQueryName=Test2;";
            strSchema += "Connection1_CustomQueryParam=" + strParam + ";";
            strSchema += "Connection2_CustomQueryName=Test3;";
            strSchema += "Connection2_CustomQueryParam=" + strParam + ";";

            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery1());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery2());
            ((MyCaffeStreamDatabase)db).AddDirectQuery(new CustomQuery3());
            DateTime dt = DateTime.Today;
            string strSettings = "QueryCount=5;Start=" + dt.ToShortDateString() + ";TimeSpanInMs=60000;SegmentSize=1;MaxCount=10;";
            db.Initialize(QUERY_TYPE.SYNCHRONIZED, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 6, "The query size item 1 should be 3 for three fields (q1:sync,data; q2:data).");
            log.CHECK_EQ(rgSize[2], 5, "The query size item 2 should be 5 for the number of items queried.");

            int nDataIdx = 0;
            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            List<SimpleDatum> rgSd = new List<SimpleDatum>();
            Stopwatch sw = new Stopwatch();
            int nIter = 10000;

            sw.Start();

            for (int i = 0; i < nIter; i++)
            {
                rgSd.Add(db.Query(int.MaxValue));
            }

            sw.Stop();

            double dfMs = sw.Elapsed.TotalMilliseconds;
            double dfMsPerQuery = dfMs / nIter;

            log.WriteLine("Total Time = " + dfMs.ToString() + " ms, Ave time per query = " + dfMsPerQuery.ToString() + " ms.");

            for (int i=0; i<rgSd.Count; i++)
            {
                SimpleDatum sd = rgSd[i];
                log.CHECK(sd != null, "The SD returned should not be null.");
                log.CHECK_EQ(sd.ItemCount, nCount, "There should be " + rgSize[1].ToString() + "x" + rgSize[2].ToString() + " items in the data.");
                log.CHECK_EQ(sd.Channels, rgSize[0], "The channels are not as expected.");
                log.CHECK_EQ(sd.Height, rgSize[1], "The height is not as expected.");
                log.CHECK_EQ(sd.Width, rgSize[2], "The width is not as expected.");

                for (int j = 0; j < nW; j++)
                {
                    DateTime dt1 = Utility.ConvertTimeFromMinutes(sd.RealData[j]);
                    log.CHECK(dt1 == dt, "The time sync is incorrect.");
                    dt += TimeSpan.FromMinutes(1);

                    double df1 = sd.RealData[(nW * 1) + j];
                    int nVal1 = (int)df1;
                    log.CHECK_EQ(nVal1, nDataIdx, "The data value is incorrect.");

                    double df2 = sd.RealData[(nW * 2) + j];
                    int nVal2 = (int)df2;
                    log.CHECK_EQ(nVal2, nDataIdx, "The data value is incorrect.");

                    double df3 = sd.RealData[(nW * 3) + j];
                    int nVal3 = (int)df3;
                    log.CHECK_EQ(nVal3, nDataIdx, "The data value is incorrect.");

                    double df4 = sd.RealData[(nW * 4) + j];
                    int nVal4 = (int)df4;
                    log.CHECK_EQ(nVal4, nDataIdx, "The data value is incorrect.");

                    double df5 = sd.RealData[(nW * 5) + j];
                    int nVal5 = (int)df5;
                    log.CHECK_EQ(nVal5, nDataIdx, "The data value is incorrect.");

                    nDataIdx++;
                }
            }

            db.Shutdown();
        }
    
        [TestMethod]
        public void TestDataItem()
        {
            db.stream.DataItem di = new db.stream.DataItem(10);
            Log log = new Log("Test streaming database");
            log.EnableTrace = true;

            for (int i = 0; i < 5; i++)
            {
                bool bComplete = di.Add(i, i);
                log.CHECK(!bComplete, "Data item should not be complete.");                
            }

            for (int i = 5; i < 9; i++)
            {
                bool bComplete = di.Add(i, i);
                log.CHECK(!bComplete, "Data item should not be complete.");
            }

            bool bComplete1 = di.Add(9, 9);
            log.CHECK(bComplete1, "The data should be complete.");

            double[] rgdf = di.GetData();
            log.CHECK_EQ(rgdf.Length, 10, "The data length is incorrect.");

            for (int i = 0; i < 10; i++)
            {
                log.CHECK_EQ(i, rgdf[i], "The data item at index #" + i.ToString() + " is incorrect.");
            }
        }

        protected string getTestPath(string strItem, bool bPathOnly = false, bool bCreateIfMissing = false, bool bUserData = false)
        {
            return TestBase.GetTestPath(strItem, bPathOnly, bCreateIfMissing, bUserData);
        }

        [TestMethod]
        public void TestQueryGeneralText()
        {
            Log log = new Log("Test streaming database with general data");
            log.EnableTrace = true;

            IXStreamDatabase db = new MyCaffeStreamDatabase(log);
            string strSchema = "ConnectionCount=1;";

            string strDataPath = getTestPath("\\MyCaffe\\test_data\\data\\char-rnn", true);
            string strParam = "FilePath=" + strDataPath + ";";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=StdTextFileQuery;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            DateTime dt = DateTime.Today;
            string strSettings = "";
            db.Initialize(QUERY_TYPE.GENERAL, strSchema + strSettings);

            int[] rgSize = db.QuerySize();
            log.CHECK(rgSize != null, "The Query size should not be null.");
            log.CHECK_EQ(rgSize.Length, 3, "The query size should have 3 items.");
            log.CHECK_EQ(rgSize[0], 1, "The query size item 0 should be 1.");
            log.CHECK_EQ(rgSize[1], 1, "The query size item 1 should be 1 for the number of files.");
            log.CHECK_EQ(rgSize[2], 4572882, "The query size item 2 should be 10000 for the maximum number of characters in each of the the files.");

            int nH = rgSize[1];
            int nW = rgSize[2];
            int nCount = nH * nW;

            Stopwatch sw = new Stopwatch();

            sw.Start();

            SimpleDatum sd = db.Query(int.MaxValue);
            SimpleDatum sdEnd = db.Query(int.MaxValue);

            sw.Stop();

            double dfMs = sw.Elapsed.TotalMilliseconds;

            log.WriteLine("Total Time = " + dfMs.ToString() + " ms.");

            log.CHECK(sdEnd == null, "The last query should be null to show no more data exists.");
            log.CHECK_EQ(sd.ItemCount, 4572882, "There should be more than one item in the data.");
            log.CHECK(!sd.IsRealData, "The data should be byte data, not real.");

            db.Shutdown();
        }
    }

    class CustomQuery1 : IXCustomQuery
    {
        string m_strConnection;
        string m_strTable;
        string m_strField;
        int m_nIdx = 0;
        int m_nEndIdx = int.MaxValue;

        public CustomQuery1(string strParam = null)
        {
            if (strParam != null)
            {
                strParam = ParamPacker.UnPack(strParam);
                PropertySet ps = new PropertySet(strParam);
                m_strConnection = ps.GetProperty("Connection");
                m_strTable = ps.GetProperty("Table");
                m_strField = ps.GetProperty("Field");
                m_nEndIdx = ps.GetPropertyAsInt("EndIdx", int.MaxValue);
            }
        }

        public CUSTOM_QUERY_TYPE QueryType
        {
            get { return CUSTOM_QUERY_TYPE.TIME; }
        }

        public string Name
        {
            get { return "Test1"; }
        }

        public int FieldCount
        {
            get { return 2; }
        }

        public IXCustomQuery Clone(string strParam)
        {
            return new CustomQuery1(strParam);
        }

        public void Close()
        {
        }

        public void Open()
        {
        }

        public double[] QueryByTime(DateTime dt, TimeSpan ts, int nCount)
        {
            if (m_nIdx >= m_nEndIdx)
                return null;

            double[] rg = new double[2 * nCount];

            for (int i = 0; i < nCount; i++)
            {
                rg[(2 * i)] = Utility.ConvertTimeToMinutes(dt);
                rg[(2 * i) + 1] = m_nIdx;
                dt += ts;
                m_nIdx++;
            }

            return rg;
        }

        public byte[] QueryBytes()
        {
            throw new NotImplementedException();
        }

        public int GetQuerySize()
        {
            return 1;
        }

        public void Reset()
        {
            m_nIdx = 0;
        }
    }

    class CustomQuery2 : IXCustomQuery
    {
        string m_strConnection;
        string m_strTable;
        string m_strField;
        int m_nIdx = 0;
        int m_nEndIdx = int.MaxValue;

        public CustomQuery2(string strParam = null)
        {
            if (strParam != null)
            {
                strParam = ParamPacker.UnPack(strParam);
                PropertySet ps = new PropertySet(strParam);
                m_strConnection = ps.GetProperty("Connection");
                m_strTable = ps.GetProperty("Table");
                m_strField = ps.GetProperty("Field");
                m_nEndIdx = ps.GetPropertyAsInt("EndIdx", int.MaxValue);
            }
        }

        public CUSTOM_QUERY_TYPE QueryType
        {
            get { return CUSTOM_QUERY_TYPE.TIME; }
        }

        public string Name
        {
            get { return "Test2"; }
        }

        public int FieldCount
        {
            get { return 2; }
        }

        public IXCustomQuery Clone(string strParam)
        {
            return new CustomQuery2(strParam);
        }

        public void Close()
        {
        }

        public void Open()
        {
        }

        public double[] QueryByTime(DateTime dt, TimeSpan ts, int nCount)
        {
            if (m_nIdx >= m_nEndIdx)
                return null;

            double[] rg = new double[2 * nCount];

            for (int i = 0; i < nCount; i++)
            {
                rg[(2 * i)] = Utility.ConvertTimeToMinutes(dt);
                rg[(2 * i) + 1] = m_nIdx;
                dt += ts;
                m_nIdx++;
            }

            return rg;
        }

        public byte[] QueryBytes()
        {
            throw new NotImplementedException();
        }

        public int GetQuerySize()
        {
            return 1;
        }

        public void Reset()
        {
            m_nIdx = 0;
        }
    }

    class CustomQuery3 : IXCustomQuery
    {
        string m_strConnection;
        string m_strTable;
        string m_strField;
        int m_nIdx = 0;
        int m_nEndIdx = int.MaxValue;

        public CUSTOM_QUERY_TYPE QueryType
        {
            get { return CUSTOM_QUERY_TYPE.TIME; }
        }

        public CustomQuery3(string strParam = null)
        {
            if (strParam != null)
            {
                strParam = ParamPacker.UnPack(strParam);
                PropertySet ps = new PropertySet(strParam);
                m_strConnection = ps.GetProperty("Connection");
                m_strTable = ps.GetProperty("Table");
                m_strField = ps.GetProperty("Field");
                m_nEndIdx = ps.GetPropertyAsInt("EndIdx", int.MaxValue);
            }
        }

        public string Name
        {
            get { return "Test3"; }
        }

        public int FieldCount
        {
            get { return 4; }  // sync, data1, data2, data3
        }

        public IXCustomQuery Clone(string strParam)
        {
            return new CustomQuery3(strParam);
        }

        public void Close()
        {
        }

        public void Open()
        {
        }

        public double[] QueryByTime(DateTime dt, TimeSpan ts, int nCount)
        {
            if (m_nIdx >= m_nEndIdx)
                return null;

            double[] rg = new double[4 * nCount];

            for (int i = 0; i < nCount; i++)
            {
                rg[(2 * i)] = Utility.ConvertTimeToMinutes(dt);
                rg[(2 * i) + 1] = m_nIdx;
                rg[(2 * i) + 2] = m_nIdx;
                rg[(2 * i) + 3] = m_nIdx;
                dt += ts;
                m_nIdx++;
            }

            return rg;
        }

        public byte[] QueryBytes()
        {
            throw new NotImplementedException();
        }

        public int GetQuerySize()
        {
            return 1;
        }

        public void Reset()
        {
            m_nIdx = 0;
        }
    }
}
