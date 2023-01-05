using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.common;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.fillers;
using System.Reflection;
using System.IO;
using System.Diagnostics;
using System.Threading;
using MyCaffe.db.image;
using System.Globalization;
using System.IO.Compression;
using System.Net;

namespace MyCaffe.test
{
    public class TestBase : IDisposable, ITestKnownFailures
    {
        public const int DEFAULT_DEVICE_ID = 0;

        List<ITest> m_rgTests = new List<ITest>();
        string m_strName = "";
        static bool m_bResetOnCleanUp = false;
        protected bool m_bHalf = false;
        IMGDB_VERSION m_imgDbVer = IMGDB_VERSION.DEFAULT;
        static string m_strCudaPath = "";
        static string m_strCulture = "";
        static CultureInfo m_defaultCulture = null;


        public TestBase(string strName, int nDeviceID = DEFAULT_DEVICE_ID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT, object tag = null, bool bHalf = false)
        {
            m_bHalf = bHalf;

            // If an auto test has set the GPUID, us it instead.
            LocalDataStoreSlot lds = Thread.GetNamedDataSlot("GPUID");
            if (lds != null)
            {
                object obj = Thread.GetData(lds);
                if (obj != null)
                {
                    string strGpuId = obj.ToString();
                    if (!string.IsNullOrEmpty(strGpuId))
                    {
                        int nVal;

                        if (int.TryParse(strGpuId, out nVal) && nDeviceID < 4)
                            nDeviceID = nVal;
                    }
                }
            }

            // If an auto test has set the IMGDB_VER, use it instead of the default.
            LocalDataStoreSlot ldsv = Thread.GetNamedDataSlot("IMGDBVER");
            if (ldsv != null)
            {
                object obj = Thread.GetData(ldsv);
                if (obj != null)
                {
                    string strImgDbVer = obj.ToString();
                    if (!string.IsNullOrEmpty(strImgDbVer))
                    {
                        int nVal;

                        if (int.TryParse(strImgDbVer, out nVal) && (nVal == 0 || nVal == 1))
                            m_imgDbVer = (IMGDB_VERSION)nVal;
                    }
                }
            }

            // If an auto test has set the CULTURE, use it instead of the default.
            LocalDataStoreSlot ldsc = Thread.GetNamedDataSlot("CULTURE");
            if (ldsc != null)
            {
                object obj = Thread.GetData(ldsc);
                if (obj != null)
                {
                    string strCulture = obj.ToString();
                    if (!string.IsNullOrEmpty(strCulture))
                    {
                        m_defaultCulture = Thread.CurrentThread.CurrentCulture;
                        Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo(strCulture);
                    }
                }
            }

            LocalDataStoreSlot ldsp = Thread.GetNamedDataSlot("CUDAPATH");
            if (ldsp != null)
            {
                object obj = Thread.GetData(ldsp);
                if (obj != null)
                    m_strCudaPath = obj.ToString();
            }

            string strPath = CudaPath;
            CudaDnn<float>.SetDefaultCudaPath(strPath);
            CudaDnn<double>.SetDefaultCudaPath(strPath);

            m_strName = strName;

            if (create_count == 1)
            {
                ITest iTestF = create(DataType.FLOAT, strName, nDeviceID, engine);
                if (iTestF != null)
                {
                    iTestF.SetParent(this);
                    iTestF.initialize();
                    m_rgTests.Add(iTestF);
                }

                ITest iTestD = create(DataType.DOUBLE, strName, nDeviceID, engine);
                if (iTestD != null)
                {
                    iTestD.SetParent(this);
                    iTestD.initialize();
                    m_rgTests.Add(iTestD);
                }
            }
            else
            {
                for (int i = 0; i < create_count; i++)
                {
                    ITest iTestF = create(i, DataType.FLOAT, strName, nDeviceID, engine);
                    if (iTestF != null)
                    {
                        iTestF.SetParent(this);
                        iTestF.initialize();
                        m_rgTests.Add(iTestF);
                    }

                    ITest iTestD = create(i, DataType.DOUBLE, strName, nDeviceID, engine);
                    if (iTestD != null)
                    {
                        iTestD.SetParent(this);
                        iTestD.initialize();
                        m_rgTests.Add(iTestD);
                    }
                }
            }
        }

        public TestBase()
        {
            m_strName = "";
        }

        public IXImageDatabaseBase createImageDb(Log log, int nSeed = 0)
        {
            if (m_imgDbVer == IMGDB_VERSION.V1)
                return new MyCaffeImageDatabase(log, "default", nSeed);
            else
                return new MyCaffeImageDatabase2(log, "default", nSeed);
        }

        public List<Tuple<string, string, string>> KnownFailures
        {
            get
            {
                List<Tuple<string, string, string>> rgKnownFailures = new List<Tuple<string, string, string>>();
                rgKnownFailures.Add(new Tuple<string, string, string>("TestContrastiveLossLayer", "TestGradientLegacy", "SKIPPED - Values -0.400000 and 0.158390 are NOT within the range 0.01 of one another."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestNeuronLayer", "TestDropoutGradientCuDnn", "SKIPPED - Values 4 and 52.572381 are not within the range of 0.05257 of one another."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestMultiBoxLossLayer", "TestLocGradientGpu", "SKIPPED - CUDA: Invalid parameter."));

                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_1btmA0", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_1btmAp5", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_2btmA0", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_2btmAp5", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_1btmA0_CUDNN", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_1btmAp5_CUDNN", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_2btmA0_CUDNN", "SKIPPED - Testing work in progress."));
                //rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionOctaveLayer", "TestGradient_2btmAp5_CUDNN", "SKIPPED - Testing work in progress."));

                return rgKnownFailures;
            }
        }

        public int GetPriority(string strClass, string strMethod)
        {
            if (strClass == "TestMultiBoxLossLayer" && strMethod.Contains("TestConfGradient"))
                return 1;

            if (strClass == "TestMyCaffeControl" && strMethod.Contains("TestTrainMultiGpu"))
                return 1;

            // If this fails it can fail with the error 'Device encountered a load or storage instruction on an invalid memory address (700)
            if (strClass == "TestLSTMSimpleLayer" && strMethod.Contains("TestGradientClipMask"))
                return 1;

            // If this fails, a corruption of the GPU is possible.
            if (strClass == "TestNCCL")
                return 2;

            // If this fails, all other following tests will fail due to memory usage.
            if (strClass == "TestCudaDnn" && strMethod.Contains("TestMemoryTest"))
                return 3;

            if (strClass == "TestMyCaffeImageDatabase" && strMethod.Contains("TestQueryRandomLoadOnDemandLabelBalance"))
                return 1;

            if (strClass == "TestMyCaffeImageDatabase" && strMethod.Contains("TestQueries"))
                return 1;

            return 0;
        }

        public void EnableTests(params int[] rgIdx)
        {
            for (int i=0; i<m_rgTests.Count; i++)
            {
                if (rgIdx.Contains(i))
                    m_rgTests[i].Enabled = true;
                else
                    m_rgTests[i].Enabled = false;
            }
        }

        protected virtual ITest create(int nIdx, DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            ITest iTest = create(dt, strName, nDeviceID, engine);
            iTest.SetParent(this);
            return iTest;
        }

        protected virtual ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            ITest iTest;

            if (dt == DataType.DOUBLE)
                iTest = new Test<double>(strName, nDeviceID, engine, m_bHalf);
            else
                iTest = new Test<float>(strName, nDeviceID, engine, m_bHalf);

            iTest.SetParent(this);

            return iTest;
        }

        protected virtual int create_count
        {
            get { return 1; }
        }

        protected virtual void dispose()
        {
            if (m_bResetOnCleanUp)
            {
                CudaDnn<float> cuda = new CudaDnn<float>(0, DEVINIT.NONE);
                cuda.ResetDevice();
                cuda.Dispose();
            }

            if (m_defaultCulture != null)
                Thread.CurrentThread.CurrentCulture = m_defaultCulture;
        }

        public static bool ResetOnCleanup
        {
            get { return m_bResetOnCleanUp; }
            set { m_bResetOnCleanUp = value; }
        }

        public static string CudaPath
        {
            get
            {
                if (File.Exists(m_strCudaPath))
                    return m_strCudaPath;

                return CudaDnn<float>.GetCudaDnnDllPath();
            }
        }

        public static string ExecutingAppPath
        {
            get
            {
                string codeBase = Process.GetCurrentProcess().MainModule.FileName;
                UriBuilder uri = new UriBuilder(codeBase);
                string strPath = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(strPath);
            }
        }

        public static string ExecutingAssemblyPath
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string strPath = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(strPath);
            }
        }

        public static string GetTestPath(string strItem, bool bPathOnly = false, bool bCreateIfMissing = false, bool bUserData = false)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);

            if (bUserData)
                strPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);

            if (bPathOnly)
            {
                strPath += strItem;

                if (Directory.Exists(strPath))
                    return strPath;

                if (bCreateIfMissing)
                    Directory.CreateDirectory(strPath);

                if (Directory.Exists(strPath))
                    return strPath;
            }

            string strTemp = strPath + strItem;
            if (File.Exists(strTemp))
                return strTemp;

            strPath = TestBase.ExecutingAssemblyPath;
            int nPos;

            // Remove the build (Release or Debug)
            nPos = strPath.LastIndexOf('\\');
            if (nPos > 0)
                strPath = strPath.Substring(0, nPos);

            // Remove the 'bin'
            nPos = strPath.LastIndexOf('\\');
            if (nPos > 0)
                strPath = strPath.Substring(0, nPos);

            string strTarget = "\\MyCaffe";
            nPos = strItem.IndexOf(strTarget);
            if (nPos >= 0)
                strItem = strItem.Substring(nPos + strTarget.Length);

            return strPath + strItem;
        }

        public List<ITest> Tests
        {
            get { return m_rgTests; }
        }

        public List<ITest> EnabledTests
        {
            get
            {
                List<ITest> rgTest = new List<ITest>();

                foreach (ITest t in m_rgTests)
                {
                    if (t.Enabled)
                        rgTest.Add(t);
                }

                return rgTest;
            }
        }

        public ITest GetTest(DataType dt)
        {
            foreach (ITest t in m_rgTests)
            {
                if (t.DataType == dt)
                    return t;
            }

            return null;
        }

        public Test<T> GetTest<T>()
        {
            if (typeof(T) == typeof(double))
                return (Test<T>)GetTest(DataType.DOUBLE);
            else
                return (Test<T>)GetTest(DataType.FLOAT);
        }

        public void Dispose()
        {
            foreach (ITest t in m_rgTests)
            {
                IDisposable disposable = t as IDisposable;
                disposable.Dispose();
            }

            m_rgTests.Clear();

            dispose();
        }
    }

    public interface ITest
    {
        ICudaDnn Cuda { get; }
        Log Log { get; }
        DataType DataType { get; }
        EngineParameter.Engine engine { get; }
        bool Enabled { get; set; }
        void SetParent(TestBase parent);
        void initialize();
    }

    public class Test<T> : ITest, IDisposable 
    {
        protected Log m_log;
        protected CudaDnn<T> m_cuda;
        protected DataType m_dt;
        protected EngineParameter.Engine m_engine;
        protected bool m_bEnabled = true;
        protected bool m_bHalf = false;
        protected long m_lSeed = 1701;
        TestingActiveGpuSet m_activeGpuId = new TestingActiveGpuSet();
        TestingActiveKernelHandleSet m_activeKernel = new TestingActiveKernelHandleSet();
        TestBase m_parent = null;

        public Test(string strName, int nDeviceID = TestBase.DEFAULT_DEVICE_ID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT, bool bHalf = false)
        {
            PreTest.Init();

            m_dt = (typeof(T) == typeof(double)) ? DataType.DOUBLE : DataType.FLOAT;
            m_cuda = GetCuda(nDeviceID);
            m_bHalf = bHalf;

            string str = name;

            if (str.Length > 0)
                str += " -> ";

            m_log = GetLog(str + strName + " - type " + (typeof(T)).ToString());
            m_engine = engine;
        }

        public void SetParent(TestBase parent)
        {
            m_parent = parent;
        }

        public virtual void initialize()
        {
        }

        protected virtual void dispose()
        {
            m_cuda.Dispose();
            m_cuda = null;
        }

        protected IXImageDatabaseBase createImageDb(Log log, int nSeed = 0)
        {
            return m_parent.createImageDb(log, nSeed);
        }

        protected string getTestPath(string strItem, bool bPathOnly = false, bool bCreateIfMissing = false, bool bUserData = false)
        {
            return TestBase.GetTestPath(strItem, bPathOnly, bCreateIfMissing, bUserData);
        }

        public bool Enabled
        {
            get { return m_bEnabled; }
            set { m_bEnabled = value; }
        }

        public virtual string name
        {
            get { return ""; }
        }

        public CudaDnn<T> GetCuda(int nDeviceID = TestBase.DEFAULT_DEVICE_ID)
        {
            DEVINIT flags = DEVINIT.CUBLAS | DEVINIT.CURAND;

            // NOTE: CudaPath set in TestBase will be used, see CudaPath and SetDefaultCudaPath() above.
            CudaDnn<T> cuda = new CudaDnn<T>(nDeviceID, flags, m_lSeed);
            m_activeGpuId.SetActiveGpu(nDeviceID);
            m_activeKernel.SetActiveKernelHandle(cuda.KernelHandle);

            Trace.WriteLine("TestBase using Cuda Connection: '" + cuda.Path + "'");
          
            return cuda;
        }

        public static Log GetLog(string str)
        {
            Log log = new Log(str + " type -> " + (typeof(T)).ToString());

            log.EnableTrace = true;

            return log;
        }

        public DataType DataType
        {
            get { return m_dt; }
        }

        public EngineParameter.Engine engine
        {
            get { return m_engine; }
        }

        public ICudaDnn Cuda
        {
            get { return m_cuda; }
        }

        public CudaDnn<T> CudaObj
        {
            get { return m_cuda; }
        }

        public Log Log
        {
            get { return m_log; }
        }

        public void Dispose()
        {
            dispose();
        }
    }

    public class TestEx<T> : Test<T>
    {
        protected Blob<T> m_blob_bottom;
        protected Blob<T> m_blob_top;
        BlobCollection<T> m_colBottom = new BlobCollection<T>();
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        protected Filler<T> m_filler;
        protected Blob<T> m_blobWork;
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public TestEx(string strName, List<int> rgBottomShape = null, int nDeviceID = TestBase.DEFAULT_DEVICE_ID, bool bHalf = false)
            : base(strName, nDeviceID, EngineParameter.Engine.DEFAULT, bHalf)
        {
            if (rgBottomShape == null)
                rgBottomShape = new List<int>() { 2, 3, 4, 5 };

            m_blobWork = new Blob<T>(m_cuda, m_log);
            m_blob_bottom = new Blob<T>(m_cuda, m_log, rgBottomShape, true);
            m_blob_top = new Blob<T>(m_cuda, m_log, true);
            m_colBottom.Add(m_blob_bottom);
            m_colTop.Add(m_blob_top);

            FillerParameter fp = getFillerParam();
            m_filler = Filler<T>.Create(m_cuda, m_log, fp);
            m_filler.Fill(m_blob_bottom);
        }

        protected void dispose(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blob_bottom);
            dispose(ref m_blob_top);

            base.dispose();
        }

        protected virtual FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public Filler<T> Filler
        {
            get { return m_filler; }
        }

        public Blob<T> Bottom
        {
            get { return m_blob_bottom; }
        }

        public Blob<T> Top
        {
            get { return m_blob_top; }
        }

        public BlobCollection<T> BottomVec
        {
            get { return m_colBottom; }
        }

        public BlobCollection<T> TopVec
        {
            get { return m_colTop; }
        }

        protected void EXPECT_NEAR(T f1, T f2, double dfErr, string str = "")
        {
            double df1 = (double)Convert.ChangeType(f1, typeof(double));
            double df2 = (double)Convert.ChangeType(f2, typeof(double));
            m_log.EXPECT_NEAR(df1, df2, dfErr, str);
        }

        protected void CHECK_EQ(T f1, T f2, string str)
        {
            double df1 = (double)Convert.ChangeType(f1, typeof(double));
            double df2 = (double)Convert.ChangeType(f2, typeof(double));
            m_log.CHECK_EQ(df1, df2, str);
        }

        protected double convert(T t)
        {
            return (double)Convert.ChangeType(t, typeof(double));
        }

        protected float convertF(T t)
        {
            return (float)Convert.ChangeType(t, typeof(float));
        }

        protected T convert(double df)
        {
            return (T)Convert.ChangeType(df, typeof(T));
        }

        protected float[] convertF(T[] rg)
        {
            return Utility.ConvertVecF<T>(rg);
        }

        protected double[] convert(T[] rg)
        {
            return Utility.ConvertVec<T>(rg);
        }

        protected T[] convert(double[] rg)
        {
            return Utility.ConvertVec<T>(rg);
        }

        protected T[] convert(float[] rg)
        {
            return Utility.ConvertVec<T>(rg);
        }

        protected string loadTestData(string strPath, string strFileName, string strTestPath, string strTestFile)
        {
            string strTestDataFile = downloadTestData(strPath, strFileName);

            if (!File.Exists(strPath + strTestPath + "\\" + strTestFile))
                ZipFile.ExtractToDirectory(strTestDataFile, strPath);

            return strPath + strTestPath + "\\";
        }

        protected string downloadTestData(string strPath, string strFileName)
        {
            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strTestDataFile = strPath + "\\" + strFileName;
            if (!File.Exists(strTestDataFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/" + strFileName;
                    string strFile1 = strFileName;
                    string strFile = strPath + "\\" + strFile1;

                    m_swUpdateTimer.Start();
                    m_dfLastProgress = 0;

                    webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                    webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                    webClient.DownloadFileAsync(new Uri(strUrl), strFile, strFile1);

                    m_evtDownloadDone.WaitOne();
                }
            }

            return strTestDataFile;
        }

        private void WebClient_DownloadFileCompleted(object sender, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            bool bTraceEnabled = m_log.EnableTrace;
            m_log.EnableTrace = true;
            m_log.WriteLine("Downloading done.");
            m_log.EnableTrace = bTraceEnabled;

            m_evtDownloadDone.Set();
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_swUpdateTimer.Elapsed.TotalMilliseconds >= 1000)
            {
                if (m_dfLastProgress != e.ProgressPercentage)
                {
                    m_dfLastProgress = e.ProgressPercentage;
                    string strFile = e.UserState.ToString();
                    bool bTraceEnabled = m_log.EnableTrace;
                    m_log.EnableTrace = true;

                    m_log.Progress = e.ProgressPercentage / 100.0;
                    m_log.WriteLine("Downloading '" + strFile + "' at " + m_log.Progress.ToString("P") + "...");
                    m_log.EnableTrace = bTraceEnabled;
                }

                m_swUpdateTimer.Restart();
            }
        }

        protected void verify(Blob<T> b1, Blob<T> b1exp, bool bCompareDiff, double dfErr = 1e-08)
        {
            float[] rgExpected = (bCompareDiff) ? convertF(b1exp.mutable_cpu_diff) : convertF(b1exp.mutable_cpu_data);
            float[] rgActual = (bCompareDiff) ? convertF(b1.mutable_cpu_diff) : convertF(b1.mutable_cpu_data);

            for (int i = 0; i < rgExpected.Length; i++)
            {
                float fExpected = rgExpected[i];
                float fActual = rgActual[i];

                m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, dfErr, "The values are not as expected!");
            }

            bool bRes = b1.Compare(b1exp, m_blobWork, bCompareDiff, dfErr);
            if (!bRes)
                m_log.FAIL("The blobs are not equal!");
        }
    }
}
