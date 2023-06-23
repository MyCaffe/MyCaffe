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
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.tft;
using System.Threading;
using System.Diagnostics;
using MyCaffe.param.tft;
using System.IO;
using System.Xml;
using System.ServiceModel.Security;
using System.Data.Entity.ModelConfiguration.Conventions;
using MyCaffe.db.temporal;

/// <summary>
/// Testing the DataTemporal.
/// </remarks> 
namespace MyCaffe.test
{
    enum SOURCE
    {
        ELECTRICITY,
        TRAFFIC,
        VOLATILITY
    }

    [TestClass]
    public class TestTFT_DataTemporal
    {
        private IXDatabaseBase getDatabase(string strName, int nHistSteps, int nFutureSteps, bool bNormalizedData)
        {
            Log log = new Log("Test Data Temporal");
            log.EnableTrace = true;

            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadLimit = 0;
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;

            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());

            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(log, prop);
            db.InitializeWithDsName1(s, "TFT.Electricity");

            return db;
        }

        [TestMethod]
        public void TestForwardTrainElectricitySql()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\electricity\\preprocessed";
            IXDatabaseBase db = null;

            try
            {
                db = getDatabase("TFT.Electricity", 90, 30, true);
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.SQL_DB, "TFT.Electricity", strPath, 100, SOURCE.ELECTRICITY, db);
                }
            }
            finally
            {
                if (db != null)
                    db.CleanUp();

                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestElectricitySql()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\electricity\\preprocessed";
            IXDatabaseBase db = null;

            try
            {
                db = getDatabase("TFT.Electricity", 90, 30, true);
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(30, Phase.TEST, DataTemporalParameter.SOURCE_TYPE.SQL_DB, "TFT.Electricity", strPath, 100, SOURCE.ELECTRICITY, db);
                }
            }
            finally
            {
                if (db != null)
                    db.CleanUp();

                test.Dispose();
            }
        }


        [TestMethod]
        public void TestForwardTrainElectricityNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\electricity\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\data\\electricity\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.ELECTRICITY, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestElectricityNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\electricity\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TEST, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.ELECTRICITY, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRunElectricityNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\electricity\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.RUN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.ELECTRICITY, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestForwardTrainTrafficSql()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\traffic\\preprocessed";
            IXDatabaseBase db = null;

            try
            {
                db = getDatabase("TFT.Traffic", 90, 30, true);
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.SQL_DB, "TFT.Traffic", strPath, 100, SOURCE.TRAFFIC, db);
                }
            }
            finally
            {
                if (db != null)
                    db.CleanUp();

                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestTrafficSql()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\traffic\\preprocessed";
            IXDatabaseBase db = null;

            try
            {
                db = getDatabase("TFT.Traffic", 90, 30, true);
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TEST, DataTemporalParameter.SOURCE_TYPE.SQL_DB, "TFT.Traffic", strPath, 100, SOURCE.TRAFFIC, db);
                }
            }
            finally
            {
                if (db != null)
                    db.CleanUp();

                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTrainTrafficNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\traffic\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\traffic\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.TRAFFIC, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestTrafficNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\traffic\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\traffic\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TEST, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.TRAFFIC, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRunTrafficNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\traffic\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\traffic\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.RUN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.TRAFFIC, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTrainVolatilitySql()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\volatility\\preprocessed";
            IXDatabaseBase db = null;

            try
            {
                db = getDatabase("TFT.Volatility", 90, 30, true);
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.SQL_DB, "TFT.Volatility", strPath, 100, SOURCE.VOLATILITY, null);
                }
            }
            finally
            {
                if (db != null)
                    db.CleanUp();

                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestVolatilitysql()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\volatility\\preprocessed";
            IXDatabaseBase db = null;

            try
            {
                db = getDatabase("TFT.Volatility", 90, 30, true);
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TEST, DataTemporalParameter.SOURCE_TYPE.SQL_DB, "TFT.Volatility", strPath, 100, SOURCE.VOLATILITY, null);
                }
            }
            finally
            {
                if (db != null)
                    db.CleanUp();

                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTrainVolatilityNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\volatility\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\volatility\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.VOLATILITY, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestVolatilityNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\volatility\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\volatility\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.TEST, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.VOLATILITY, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRunVolatilityNpy()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\volatility\\preprocessed";
            //string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\volatility\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(100, Phase.RUN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, strPath, 100, SOURCE.VOLATILITY, null);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBlobLoadNumpyPartial()
        {
            DataTemporalTest test = new DataTemporalTest();

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestBlobLoadNumpyPartial();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

    }

    interface IDataTemporalTest : ITest
    {
        void TestForward(int nMaxIter, Phase phase, DataTemporalParameter.SOURCE_TYPE srcType, string strSrc, string srcPath, int nBatchSize, SOURCE src, IXDatabaseBase db);
        void TestBlobLoadNumpyPartial();
    }

    class DataTemporalTest : TestBase
    {
        public DataTemporalTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT DataTemporal Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DataTemporalTest<double>(strName, nDeviceID, engine);
            else
                return new DataTemporalTest<float>(strName, nDeviceID, engine);
        }
    }

    class DataTemporalTest<T> : TestEx<T>, IDataTemporalTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public DataTemporalTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private string getTestDataPath()
        {
            return "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\static_enrichment_grn\\";
        }


        private string buildModel(int nNumSamples, int nNumHist, int nNumFuture, DataTemporalParameter.SOURCE_TYPE srcType, string strSrc, Phase phase)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";


            //---------------------------------
            //  Data Temporal Input
            //---------------------------------
            LayerParameter data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data");
            data.data_temporal_param.batch_size = (uint)nNumSamples;
            data.data_temporal_param.num_historical_steps = (uint)nNumHist;
            data.data_temporal_param.num_future_steps = (uint)nNumFuture;
            data.data_temporal_param.source = strSrc;
            data.data_temporal_param.source_type = srcType;
            data.data_temporal_param.shuffle_data = false;
            data.data_temporal_param.chunk_count = 1024;
            data.data_temporal_param.drip_refresh_rate_in_sec = 0;
            data.data_temporal_param.max_load_percent = 1.0;
            data.data_temporal_param.seed = 1704;
            data.include.Add(new NetStateRule(phase));
            data.top.Add("x_numeric_static");
            data.top.Add("x_categorical_static");
            data.top.Add("x_numeric_hist");
            data.top.Add("x_categorical_hist");
            data.top.Add("x_numeric_future");
            data.top.Add("x_categorical_future");
            data.top.Add("target");
            p.layer.Add(data);

            return p.ToProto("root").ToString();
        }

        private Tuple<long[], int[]> load(string strPath, string strType, string strFile)
        {
            string strFile1 = strPath + "\\" + strType + "_" + strFile;
            NumpyFile<long> npy = new NumpyFile<long>(m_log);
            npy.OpenRead(strFile1);

            long[] rgData = new long[npy.TotalCount];
            int nOffset = 0;
            long[] rgVal = null;

            for (int i = 0; i < npy.Rows; i++)
            {
                rgVal = npy.LoadRow(rgVal, i);
                Array.Copy(rgVal, 0, rgData, nOffset, rgVal.Length);
                nOffset += rgVal.Length;
            }

            npy.Dispose();

            return new Tuple<long[], int[]>(rgData, npy.Shape);
        }

        private void loadFloat(Blob<T> blob, string strPath, string strType, string strFile)
        {
            string strFile1 = strPath + "\\" + strType + "_" + strFile;
            if (!File.Exists(strFile1))
                return;

            NumpyFile<float> npy = new NumpyFile<float>(m_log);
            npy.OpenRead(strFile1);

            float[] rgData = new float[npy.TotalCount];
            int nOffset = 0;
            float[] rgVal = null;

            for (int i = 0; i < npy.Rows; i++)
            {
                rgVal = npy.LoadRow(rgVal, i);
                Array.Copy(rgVal, 0, rgData, nOffset, rgVal.Length);
                nOffset += rgVal.Length;
            }

            blob.Reshape(npy.Shape);
            blob.mutable_cpu_data = convert(rgData);

            npy.Dispose();
        }

        private void loadLong(Blob<T> blob, string strPath, string strType, string strFile)
        {
            string strFile1 = strPath + "\\" + strType + "_" + strFile;
            if (!File.Exists(strFile1))
                return;

            NumpyFile<long> npy = new NumpyFile<long>(m_log);
            npy.OpenRead(strFile1);

            float[] rgData = new float[npy.TotalCount];
            int nOffset = 0;
            long[] rgVal = null;

            for (int i = 0; i < npy.Rows; i++)
            {
                rgVal = npy.LoadRow(rgVal, i);
                Array.Copy(rgVal, 0, rgData, nOffset, rgVal.Length);
                nOffset += rgVal.Length;
            }

            blob.Reshape(npy.Shape);
            blob.mutable_cpu_data = convert(rgData);

            npy.Dispose();
        }

        private void loadStat(ref int nRowIdx, ref int nColIdx, DataSchema schema, Blob<T> blobStatNum1, Blob<T> blobStatCat1, Blob<T> blobStatNum, Blob<T> blobStatCat, int nNumHist, int nNumFut, int nBatchSize)
        {
            float[] rgStatNum1 = convertF(blobStatNum1.mutable_cpu_data);
            float[] rgStatCat1 = convertF(blobStatCat1.mutable_cpu_data);

            int nStatNumFields = schema.Data.StaticNum.Count;
            int nStatCatFields = schema.Data.StaticCat.Count;
            int nFields = nStatNumFields + nStatCatFields;

            blobStatNum.Reshape(nBatchSize, 1, nStatNumFields, 1);
            blobStatCat.Reshape(nBatchSize, 1, nStatCatFields, 1);

            float[] rgStatNum = convertF(blobStatNum.mutable_cpu_data);
            float[] rgStatCat = convertF(blobStatCat.mutable_cpu_data);

            for (int i = 0; i < nBatchSize; i++)
            {
                int nRowOffset = nRowIdx;
                int nStartIdx = 0;
                int nStartIdx1 = nRowOffset + nStartIdx;

                for (int j = 0; j < 1; j++)
                {
                    int nIdxSrc = nStartIdx1 + j;
                    int nIdxDst = (i * 1 * nFields) + (j * nFields);

                    for (int k = 0; k < nStatNumFields; k++)
                    {
                        rgStatNum[nIdxDst + k] = rgStatNum1[nIdxSrc * nStatNumFields + k];
                    }

                    for (int k = 0; k < nStatCatFields; k++)
                    {
                        rgStatCat[nIdxDst + k] = rgStatCat1[nIdxSrc * nStatCatFields + k];
                    }
                }

                nColIdx++;
                if (nColIdx + nNumHist + nNumFut > schema.Lookups[0][nRowIdx].ValidRangeCount)
                {
                    nColIdx = 0;
                    nRowIdx++;

                    if (nRowIdx >= schema.Lookups[0].Count)
                        nRowIdx = 0;
                }
            }

            if (rgStatNum.Length > 0)
                blobStatNum.mutable_cpu_data = convert(rgStatNum);
            if (rgStatCat.Length > 0)
                blobStatCat.mutable_cpu_data = convert(rgStatCat);
        }

        private void loadNum(ref int nRowIdx, ref int nColIdx, DataSchema schema, Blob<T> blobObsNum, Blob<T> blobKnownNum, Blob<T> blobHistNum, Blob<T> blobFutNum, Blob<T> blobTarget, int nNumHist, int nNumFut, int nBatchSize)
        {
            float[] rgObsNum = convertF(blobObsNum.mutable_cpu_data);
            float[] rgKnownNum = convertF(blobKnownNum.mutable_cpu_data);

            int nObsNumFields = schema.Data.ObservedNum.Count;
            int nObsNumFieldsExplicit = schema.Data.ObservedNumExplicitCount;
            int nKnownNumFields = schema.Data.KnownNum.Count;
            int nFields = nObsNumFieldsExplicit + nKnownNumFields;
            int nFutNumFields = nKnownNumFields;

            blobHistNum.Reshape(nBatchSize, nNumHist, nObsNumFieldsExplicit + nKnownNumFields, 1);
            blobFutNum.Reshape(nBatchSize, nNumFut, nKnownNumFields, 1);
            blobTarget.Reshape(nBatchSize, nNumFut, 1, 1);

            float[] rgHistNum = convertF(blobHistNum.mutable_cpu_data);
            float[] rgFutNum = convertF(blobFutNum.mutable_cpu_data);
            float[] rgTarget = convertF(blobTarget.mutable_cpu_data);

            int nTargetIndex = schema.Data.TargetIndex;
            if (nTargetIndex < 0)
                m_log.FAIL("Could not find the target index in the schema!");

            for (int i = 0; i < nBatchSize; i++)
            {
                int nRowOffset = nRowIdx * schema.Data.Columns;
                int nStartIdx = schema.Lookups[0][nRowIdx].ValidRangeStartIndex;
                int nStartIdx1 = nRowOffset + nStartIdx + nColIdx;

                for (int j = 0; j < nNumHist + nNumFut; j++)
                {
                    int nIdxSrc = nStartIdx1 + j;

                    if (j < nNumHist)
                    {
                        int nIdxDst = (i * nNumHist * nFields) + (j * nFields);
                        int nIdxDst1 = nIdxDst;

                        for (int k = 0; k < nObsNumFields; k++)
                        {
                            if (schema.Data.IsObservedNum(k))
                            {
                                rgHistNum[nIdxDst1] = rgObsNum[nIdxSrc * nObsNumFields + k];
                                nIdxDst1++;
                            }
                        }

                        for (int k = 0; k < nKnownNumFields; k++)
                        {
                            rgHistNum[nIdxDst1] = rgKnownNum[nIdxSrc * nKnownNumFields + k];
                            nIdxDst1++;
                        }
                    }
                    else
                    {
                        int nLocalJ = j - nNumHist;
                        int nIdxDst = (i * nNumFut * nKnownNumFields) + (nLocalJ * nKnownNumFields);
                        int nIdxDst1 = (i * nNumFut) + nLocalJ;

                        rgTarget[nIdxDst1] = rgObsNum[nIdxSrc * nObsNumFields + nTargetIndex];

                        for (int k = 0; k < nKnownNumFields; k++)
                        {
                            rgFutNum[nIdxDst + k] = rgKnownNum[nIdxSrc * nKnownNumFields + k];
                        }
                    }
                }

                nColIdx++;
                if (nColIdx + nNumHist + nNumFut > schema.Lookups[0][nRowIdx].ValidRangeCount)
                {
                    nColIdx = 0;
                    nRowIdx++;

                    if (nRowIdx >= schema.Lookups[0].Count)
                        nRowIdx = 0;
                }
            }

            if (rgHistNum.Length > 0)
                blobHistNum.mutable_cpu_data = convert(rgHistNum);
            if (rgFutNum.Length > 0)
                blobFutNum.mutable_cpu_data = convert(rgFutNum);
            if (rgTarget.Length > 0)
                blobTarget.mutable_cpu_data = convert(rgTarget);
        }

        private void loadCat(ref int nRowIdx, ref int nColIdx, DataSchema schema, Blob<T> blobObsCat, Blob<T> blobKnownCat, Blob<T> blobHistCat, Blob<T> blobFutCat, int nNumHist, int nNumFut, int nBatchSize)
        {
            float[] rgObsCat = convertF(blobObsCat.mutable_cpu_data);
            float[] rgKnownCat = convertF(blobKnownCat.mutable_cpu_data);

            int nObsCatFields = schema.Data.ObservedCat.Count;
            int nKnownCatFields = schema.Data.KnownCat.Count;
            int nFields = nObsCatFields + nKnownCatFields;
            int nFutCatFields = nKnownCatFields;

            blobHistCat.Reshape(nBatchSize, nNumHist, nObsCatFields + nKnownCatFields, 1);
            blobFutCat.Reshape(nBatchSize, nNumFut, nKnownCatFields, 1);

            float[] rgHistCat = convertF(blobHistCat.mutable_cpu_data);
            float[] rgFutCat = convertF(blobFutCat.mutable_cpu_data);

            for (int i = 0; i < nBatchSize; i++)
            {
                int nRowOffset = nRowIdx * schema.Data.Columns;
                int nStartIdx = schema.Lookups[0][nRowIdx].ValidRangeStartIndex;
                int nStartIdx1 = nRowOffset + nStartIdx + nColIdx;

                for (int j = 0; j < nNumHist + nNumFut; j++)
                {
                    int nIdxSrc = nStartIdx1 + j;

                    if (j < nNumHist)
                    {
                        int nIdxDst = (i * nNumHist * nFields) + (j * nFields);

                        for (int k = 0; k < nObsCatFields; k++)
                        {
                            rgHistCat[nIdxDst + k] = rgObsCat[nIdxSrc * nObsCatFields + k];
                        }

                        for (int k = 0; k < nKnownCatFields; k++)
                        {
                            rgHistCat[nIdxDst + nObsCatFields + k] = rgKnownCat[nIdxSrc * nKnownCatFields + k];
                        }
                    }
                    else
                    {
                        int nLocalJ = j - nNumHist;
                        int nIdxDst = (i * nNumFut * nKnownCatFields) + (nLocalJ * nKnownCatFields);
                        int nIdxDst1 = (i * nNumFut) + nLocalJ;

                        for (int k = 0; k < nKnownCatFields; k++)
                        {
                            rgFutCat[nIdxDst + k] = rgKnownCat[nIdxSrc * nKnownCatFields + k];
                        }
                    }
                }

                nColIdx++;
                if (nColIdx + nNumHist + nNumFut > schema.Lookups[0][nRowIdx].ValidRangeCount)
                {
                    nColIdx = 0;
                    nRowIdx++;

                    if (nRowIdx >= schema.Lookups[0].Count)
                        nRowIdx = 0;
                }
            }

            if (rgHistCat.Length > 0)
                blobHistCat.mutable_cpu_data = convert(rgHistCat);
            if (rgFutCat.Length > 0)
                blobFutCat.mutable_cpu_data = convert(rgFutCat);
        }

        private bool verify(int nIter, Blob<T> blob1, int[] rgExpectedShape, Blob<T> blobExpected, Blob<T> blobWork, string strName)
        {
            m_log.CHECK(blob1 != null, "Could not find the blob '" + strName + "'!");
            m_log.CHECK(blob1.CompareShape(rgExpectedShape), "The blob '" + strName + "' has an incorrect shape!");

            if (!blobExpected.Compare(blob1, blobWork, false, 5e-07))
            {
                float[] rgf = convertF(blob1.mutable_cpu_data);
                float[] rgfE = convertF(blobExpected.mutable_cpu_data);
                Dictionary<int, Tuple<float, float, float>> rgErr = new Dictionary<int, Tuple<float, float, float>>();

                for (int i = 0; i < rgf.Length; i++)
                {
                    float f = rgf[i];
                    float fE = rgfE[i];
                    float fDiff = Math.Abs(f - fE);
                    if (fDiff > 5e-07)
                    {
                        if (fE == 0)
                        {
                            m_log.WriteLine("WARNING: Skipping iter " + nIter.ToString() + " for ." + strName + "'.");
                            return false;
                        }
                        else
                        {
                            rgErr.Add(i, new Tuple<float, float, float>(fDiff, f, fE));
                        }
                    }
                }

                if (rgErr.Count > 0)
                    m_log.FAIL("Found " + rgErr.Count.ToString() + " errors.");
            }

            return true;
        }

        /// <summary>
        /// Test the forward pass for self attention
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run training.py on fresh 'test\iter_0' data (run just one iteration to save data)
        ///     with: 
        ///         debug = True
        ///         use_mycaffe = False
        ///         use_mycaffe_data = False
        ///         use_mycaffe_model_direct = False
        ///         use_mycaffe_model = False
        ///         lstm_use_mycaffe = False
        ///         tag = "tft.all"
        ///         test = False
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForward(int nMaxIter, Phase phase, DataTemporalParameter.SOURCE_TYPE srcType, string strSrc, string strSrcPath, int nBatchSize, SOURCE src, IXDatabaseBase db)
        {
            CancelEvent evtCancel = new CancelEvent();
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;
            Blob<T> blobStatNum1 = null;
            Blob<T> blobStatCat1 = null;
            Blob<T> blobObsNum = null;
            Blob<T> blobObsCat = null;
            Blob<T> blobKnownNum = null;
            Blob<T> blobKnownCat = null;
            Blob<T> blobTarget = null;
            Blob<T> blobStatNum = null;
            Blob<T> blobStatCat = null;
            Blob<T> blobHistNum = null;
            Blob<T> blobHistCat = null;
            Blob<T> blobFutNum = null;
            Blob<T> blobFutCat = null;

            Net<T> net = null;
            int nNumSamples = nBatchSize;
            int nNumHist = 90;
            int nNumFuture = 30;
            int nRowIdx = 0;
            int nColIdx = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);
                blobStatNum1 = new Blob<T>(m_cuda, m_log);
                blobStatCat1 = new Blob<T>(m_cuda, m_log);
                blobObsNum = new Blob<T>(m_cuda, m_log);
                blobObsCat = new Blob<T>(m_cuda, m_log);
                blobKnownNum = new Blob<T>(m_cuda, m_log);
                blobKnownCat = new Blob<T>(m_cuda, m_log);
                blobTarget = new Blob<T>(m_cuda, m_log);
                blobStatNum = new Blob<T>(m_cuda, m_log);
                blobStatCat = new Blob<T>(m_cuda, m_log);
                blobHistNum = new Blob<T>(m_cuda, m_log);
                blobHistCat = new Blob<T>(m_cuda, m_log);
                blobFutNum = new Blob<T>(m_cuda, m_log);
                blobFutCat = new Blob<T>(m_cuda, m_log);

                string strType = phase.ToString().ToLower();
                if (strType == "run")
                    strType = "validation";

                if (!File.Exists(strSrcPath + "\\" + strType + "_schema.xml"))
                    throw new Exception("Could not find the schema file '" + strSrcPath + "\\" + strType + "_schema.xml'!  You must run the SignalPop Ai Designer's " + strType + " DataSet Creator.");

                DataSchema schema = DataSchema.Load(strSrcPath + "\\" + strType + "_schema.xml");
                Tuple<long[], int[]> sync = load(strSrcPath, strType, "sync.npy");
                loadFloat(blobStatNum1, strSrcPath, strType, "static_num.npy");
                loadLong(blobStatCat1, strSrcPath, strType, "static_cat.npy");
                loadFloat(blobObsNum, strSrcPath, strType, "observed_num.npy");
                loadLong(blobObsCat, strSrcPath, strType, "observed_cat.npy");
                loadFloat(blobKnownNum, strSrcPath, strType, "known_num.npy");
                loadLong(blobKnownCat, strSrcPath, strType, "known_cat.npy");

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, srcType, strSrc, phase);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                if (db != null)
                    ((IXTemporalDatabaseBase)db).Reset();

                net = new Net<T>(m_cuda, m_log, param, evtCancel, db, phase);

                int[] rgNumStaticShape = new int[] { 0 };
                int[] rgCatStaticShape = new int[] { 0 };
                int[] rgNumHistShape = new int[] { 0 };
                int[] rgCatHistShape = new int[] { 0 };
                int[] rgNumFutShape = new int[] { 0 };
                int[] rgCatFutShape = new int[] { 0 };
                int[] rgTargetShape = new int[] { 0 };

                if (src == SOURCE.ELECTRICITY)
                {
                    rgCatStaticShape = new int[] { nBatchSize, 1 };             // station id
                    rgNumHistShape = new int[] { nBatchSize, nNumHist, 3 };     // log power use, hour, hour from start
                    rgNumFutShape = new int[] { nBatchSize, nNumFuture, 2 };    // hour, hour from start
                    rgTargetShape = new int[] { nBatchSize, nNumFuture, 1 };    // log power use
                }
                else if (src == SOURCE.TRAFFIC)
                {
                    rgCatStaticShape = new int[] { nBatchSize, 1 };             // station id
                    rgNumHistShape = new int[] { nBatchSize, nNumHist, 5 };     // value, sensor_day, time on day, day of week, hour from start
                    rgNumFutShape = new int[] { nBatchSize, nNumFuture, 4 };    // sensor_day, time on day, day of week, hour from start
                    rgTargetShape = new int[] { nBatchSize, nNumFuture, 1 };    // value
                }
                else if (src == SOURCE.VOLATILITY)
                {
                    rgCatStaticShape = new int[] { nBatchSize, 1 };             // region id
                    rgNumHistShape = new int[] { nBatchSize, nNumHist, 2 };     // open_to_close, days from start, 
                    rgCatHistShape = new int[] { nBatchSize, nNumHist, 4 };     // day of week, day of month, week of year, month
                    rgNumFutShape = new int[] { nBatchSize, nNumFuture, 1 };    // days from start
                    rgCatFutShape = new int[] { nBatchSize, nNumFuture, 4 };    // day of week, day of month, week of year, month
                    rgTargetShape = new int[] { nBatchSize, nNumFuture, 1 };    // log vol
                }

                for (int i = 0; i < nMaxIter; i++)
                {
                    BlobCollection<T> colRes = net.Forward();

                    int nRowIdx1 = nRowIdx;
                    int nColIdx1 = nColIdx;

                    loadStat(ref nRowIdx, ref nColIdx, schema, blobStatNum1, blobStatCat1, blobStatNum, blobStatCat, nNumHist, nNumFuture, nBatchSize);

                    nRowIdx = nRowIdx1;
                    nColIdx = nColIdx1;

                    loadCat(ref nRowIdx, ref nColIdx, schema, blobObsCat, blobKnownCat, blobHistCat, blobFutCat, nNumHist, nNumFuture, nBatchSize);

                    nRowIdx = nRowIdx1;
                    nColIdx = nColIdx1;

                    loadNum(ref nRowIdx, ref nColIdx, schema, blobObsNum, blobKnownNum, blobHistNum, blobFutNum, blobTarget, nNumHist, nNumFuture, nBatchSize);

                    blob1 = net.FindBlob("x_numeric_static");
                    verify(i, blob1, rgNumStaticShape, blobStatNum, blobWork, "x_numeric_static");

                    blob1 = net.FindBlob("x_categorical_static");
                    verify(i, blob1, rgCatStaticShape, blobStatCat, blobWork, "x_categorical_static");

                    blob1 = net.FindBlob("x_numeric_hist");
                    verify(i, blob1, rgNumHistShape, blobHistNum, blobWork, "x_numeric_hist");

                    blob1 = net.FindBlob("x_categorical_hist");
                    verify(i, blob1, rgCatHistShape, blobHistCat, blobWork, "x_categorical_hist");

                    blob1 = net.FindBlob("x_numeric_future");
                    verify(i, blob1, rgNumFutShape, blobFutNum, blobWork, "x_numeric_future");

                    blob1 = net.FindBlob("x_categorical_future");
                    verify(i, blob1, rgCatFutShape, blobFutCat, blobWork, "x_categorical_future");

                    blob1 = net.FindBlob("target");
                    verify(i, blob1, rgTargetShape, blobTarget, blobWork, "target");

                    double dfPct = (double)i / nMaxIter;
                    m_log.WriteLine("Testing batch " + i.ToString() + " (" + dfPct.ToString("P") + ")");
                    m_log.Progress = dfPct;
                }
            }
            finally
            {
                evtCancel.Set();
                Thread.Sleep(1000);
                
                dispose(ref blobVal);
                dispose(ref blobWork);
                dispose(ref blobStatNum1);
                dispose(ref blobStatCat1);
                dispose(ref blobStatNum);
                dispose(ref blobStatCat);
                dispose(ref blobObsNum);
                dispose(ref blobObsCat);
                dispose(ref blobKnownNum);
                dispose(ref blobKnownCat);
                dispose(ref blobTarget);
                dispose(ref blobHistNum);
                dispose(ref blobHistCat);
                dispose(ref blobFutNum);
                dispose(ref blobFutCat);

                if (net != null)
                    net.Dispose();
            }
        }

        public static DateTime UnixTimeStampToDateTime(double unixTimeStamp)
        {
            // Unix timestamp is seconds past epoch
            DateTime dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            dateTime = dateTime.AddSeconds(unixTimeStamp).ToLocalTime();
            return dateTime;
        }        

        public void TestBlobLoadNumpyPartial()
        {
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);
            Blob<T> blobData = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);
            string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\";
            string strFile;

            strFile = strPath + "validation_static_feats_categorical.npy";
            blobVal.LoadFromNumpy(strFile);
            blobData.ReshapeLike(blobVal);

            int nStartIdx = 0;
            int nCount = 1024;

            float[] rgData = Utility.ConvertVecF<T>(blobVal.mutable_cpu_data);
            Array.Clear(rgData, 0, rgData.Length);

            while (nStartIdx < blobVal.num)
            {
                Tuple<List<float[]>, int[], List<string>> data = Blob<T>.LoadFromNumpy(strFile, null, int.MaxValue, nStartIdx, nCount);

                for (int i = 0; i < data.Item1.Count; i++)
                {
                    int nItems = data.Item2.Last();
                    int nIdx = (nStartIdx + i) * nItems;
                    
                    Array.Copy(data.Item1[i], 0, rgData, nIdx, nItems);
                }

                nStartIdx += nCount;
            }

            blobData.mutable_cpu_data = convert(rgData);

            double dfMin;
            double dfMax;
            double dfErr = 1e-12;
            m_log.CHECK(blobVal.CompareEx(blobData, blobWork, out dfMin, out dfMax, false, dfErr), "The blobs are not the same.");
        }
    }
}
