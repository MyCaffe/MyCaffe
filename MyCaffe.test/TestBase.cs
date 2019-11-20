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

namespace MyCaffe.test
{
    public class TestBase : IDisposable, ITestKnownFailures
    {
        public const int DEFAULT_DEVICE_ID = 0;

        List<ITest> m_rgTests = new List<ITest>();
        string m_strName = "";
        static bool m_bResetOnCleanUp = false;
        protected bool m_bHalf = false;


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

            m_strName = strName;

            if (create_count == 1)
            {
                m_rgTests.Add(create(DataType.FLOAT, strName, nDeviceID, engine));
                m_rgTests.Add(create(DataType.DOUBLE, strName, nDeviceID, engine));
            }
            else
            {
                for (int i = 0; i < create_count; i++)
                {
                    ITest iTestF = create(i, DataType.FLOAT, strName, nDeviceID, engine);

                    if (iTestF != null)
                        m_rgTests.Add(iTestF);

                    ITest iTestD = create(i, DataType.DOUBLE, strName, nDeviceID, engine);

                    if (iTestD != null)
                        m_rgTests.Add(iTestD);
                }
            }
        }

        public TestBase()
        {
            m_strName = "";
        }

        public List<Tuple<string, string, string>> KnownFailures
        {
            get
            {
                List<Tuple<string, string, string>> rgKnownFailures = new List<Tuple<string, string, string>>();
                rgKnownFailures.Add(new Tuple<string, string, string>("TestBinaryHashLayer", "TestForward", "SKIPPED - The blob(0) value at 1 is greater that the expected maximum of 0.9."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestBinaryHashLayer", "TestRun", "SKIPPED - The blob(0) value at 10 is greater that the expected maximum of 0.9."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestTripletSelectLayer", "TestGradient", "SKIPPED - currently causes lock-up."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestSimpleTripletLossLayer", "TestGradient", "SKIPPED - Values 0.9923858 and 0.354433 are NOT within the range 0.01 of one another."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestContrastiveLossLayer", "TestGradientLegacy", "SKIPPED - Values -0.400000 and 0.158390 are NOT within the range 0.01 of one another."));

//                rgKnownFailures.Add(new Tuple<string, string, string>("TestMyCaffeImageDatabase", "TestQuerySequential2LoadLimit", "SKIPPED - Assert.AreEqual failed. Expected:<1>. Actual:<10>."));
//                rgKnownFailures.Add(new Tuple<string, string, string>("TestMyCaffeImageDatabase", "TestQuerySequential4LoadLimit", "SKIPPED - Assert.AreEqual failed. Expected:<1>. Actual:<10>."));
//                rgKnownFailures.Add(new Tuple<string, string, string>("TestMyCaffeImageDatabase", "TestQueryPairLoadLimit", "SKIPPED - Assert.AreEqual failed. Expected:<100>. Actual:<10>."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestMyCaffeImageDatabase", "TestLoadLimitNextSequential", "SKIPPED - currently causes lock-up."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionLayer", "TestNDAgainst2D", "SKIPPED - FLOAT:CAFFE Values 0.251427710056305 and -0.860932171344757 are NOT within the range 0.01 of one another.  The 2d and ND backward results are not the same at 1"));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionLayer", "TestGradient3D", "SKIPPED - FLOAT:CAFFE Values 0.107909 and -1.631164 are NOT within the range 0.0016311 of one another."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestDeconvolutionLayer", "TestNDAgainst2D", "SKIPPED - Values - 8.11466598510742 and - 35.0204658508301 are NOT within the range 0.01 of one another.The backward 2D and ND values at 0 should be equal!"));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestDeconvolutionLayer", "TestGradient3D", "SKIPPED - Values 1.7575769 and 0 are NOT within the range 0.001757569 of one another."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestDeconvolutionLayer", "TestGradient3DCuDnn", "SKIPPED - Values 1.7575769 and 0 are NOT within the range 0.001757569 of one another."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestSigmoidCrossEntropyLossLayer", "TestForward", "SKIPPED - Values 14.864425 and 3.398046 are NOT within the range 0.01 of one another."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestSigmoidCrossEntropyLossLayer", "TestGradient", "SKIPPED - Values 0.0185186 and 0.0365138 are NOT within the range 0.01 of one another."));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestReinforcementLossLayer", "TestGradient", "SKIPPED - The current batch size does not equal the size used to load the data!"));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestReinforcementLossLayer", "TestGradientTerminal", "SKIPPED - The current batch size does not equal the size used to load the data!"));

                rgKnownFailures.Add(new Tuple<string, string, string>("TestNeuronLayer", "TestDropoutGradientCuDnn", "SKIPPED - Values 4 and 52.572381 are not within the range of 0.05257 of one another."));

                return rgKnownFailures;
            }
        }

        public int GetPriority(string strClass, string strMethod)
        {
            if (strClass == "TestMyCaffeControl" && strMethod.Contains("TestTrainMultiGpu"))
                return 1;

            // If this fails, a corruption of the GPU is possible.
            if (strClass == "TestNCCL")
                return 2;

            // If this fails, all other following tests will fail due to memory usage.
            if (strClass == "TestCudaDnn" && strMethod.Contains("TestMemoryTest"))
                return 3;

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
            return create(dt, strName, nDeviceID, engine);
        }

        protected virtual ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            if (dt == DataType.DOUBLE)
                return new Test<double>(strName, nDeviceID, engine, m_bHalf);
            else
                return new Test<float>(strName, nDeviceID, engine, m_bHalf);
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
        }

        public static bool ResetOnCleanup
        {
            get { return m_bResetOnCleanUp; }
            set { m_bResetOnCleanUp = value; }
        }

        public static string CudaPath
        {
            get { return CudaDnn<float>.GetCudaDnnDllPath(); }
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

        public Test(string strName, int nDeviceID = TestBase.DEFAULT_DEVICE_ID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT, bool bHalf = false)
        {
            GC.Collect();

            m_dt = (typeof(T) == typeof(double)) ? DataType.DOUBLE : DataType.FLOAT;
            m_cuda = GetCuda(nDeviceID);
            m_bHalf = bHalf;

            string str = name;

            if (str.Length > 0)
                str += " -> ";

            m_log = GetLog(str + strName + " - type " + (typeof(T)).ToString());
            m_engine = engine;
        }

        protected virtual void dispose()
        {
            m_cuda.Dispose();
            m_cuda = null;
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

        public TestEx(string strName, List<int> rgBottomShape = null, int nDeviceID = TestBase.DEFAULT_DEVICE_ID, bool bHalf = false)
            : base(strName, nDeviceID, EngineParameter.Engine.DEFAULT, bHalf)
        {
            if (rgBottomShape == null)
                rgBottomShape = new List<int>() { 2, 3, 4, 5 };

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
            m_blob_bottom.Dispose();
            m_blob_top.Dispose();

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
    }
}
