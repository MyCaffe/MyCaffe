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

namespace MyCaffe.test
{
    public class TestBase : IDisposable, ITestKnownFailures
    {
        public const int DEFAULT_DEVICE_ID = 0;

        List<ITest> m_rgTests = new List<ITest>();
        string m_strName = "";
        static bool m_bResetOnCleanUp = false;

        public TestBase(string strName, int nDeviceID = DEFAULT_DEVICE_ID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT, object tag = null)
        {
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
                rgKnownFailures.Add(new Tuple<string, string, string>("TestTripletSelectLayer", "TestGradient", "SKIPPED - currently causes lock-up."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestMyCaffeImageDatabase", "TestLoadLimitNextSequential", "SKIPPED - currently causes lock-up."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestConvolutionLayer", "TestNDAgainst2D", "SKIPPED - currently causes a CUDA map buffer object failure (14) error on some GPU's.  This appears to corrupt the GPU for all subsequent tests fail with CUDA Missing Configuration (1) errors."));
                rgKnownFailures.Add(new Tuple<string, string, string>("TestDeconvolutionLayer", "TestNDAgainst2D", "SKIPPED - currently causes a CUDA map buffer object failure (14) error on some GPU's.  This appears to corrupt the GPU for all subsequent tests fail with CUDA Missing Configuration (1) errors."));
                return rgKnownFailures;
            }
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
                return new Test<double>(strName, nDeviceID, engine);
            else
                return new Test<float>(strName, nDeviceID, engine);
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
            get
            {
                string strFile = ExecutingAppPath + "\\CudaDnnDll.9.dll";
                if (!File.Exists(strFile))
                    strFile = ExecutingAppPath + "\\CudaDnnDll.8.dll";

                return strFile;
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

        public static string GetTestPath(string strItem, bool bPathOnly = false, bool bCreateIfMissing = false)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);

            if (bPathOnly)
            {
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

        public Test(string strName, int nDeviceID = TestBase.DEFAULT_DEVICE_ID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            m_dt = (typeof(T) == typeof(double)) ? DataType.DOUBLE : DataType.FLOAT;
            m_cuda = GetCuda(nDeviceID);

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

        protected string getTestPath(string strItem, bool bPathOnly = false, bool bCreateIfMissing = false)
        {
            return TestBase.GetTestPath(strItem, bPathOnly, bCreateIfMissing);
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
            long lSeed = 1701;

            // NOTE: CudaPath set in TestBase will be used, see CudaPath and SetDefaultCudaPath() above.
            CudaDnn<T> cuda = new CudaDnn<T>(nDeviceID, flags, lSeed);

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

        public TestEx(string strName, List<int> rgBottomShape = null, int nDeviceID = TestBase.DEFAULT_DEVICE_ID)
            : base(strName, nDeviceID)
        {
            if (rgBottomShape == null)
                rgBottomShape = new List<int>() { 2, 3, 4, 5 };

            m_blob_bottom = new Blob<T>(m_cuda, m_log, rgBottomShape);
            m_blob_top = new Blob<T>(m_cuda, m_log);
            m_colBottom.Add(m_blob_bottom);
            m_colTop.Add(m_blob_top);

            FillerParameter fp = getFillerParam();
            m_filler = Filler<T>.Create(m_cuda, m_log, fp);
            m_filler.Fill(m_blob_bottom);
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

        protected double[] convert(T[] rg)
        {
            double[] rgdf = new double[rg.Length];
            Array.Copy(rg, rgdf, rg.Length);

            return rgdf;
        }

        protected T[] convert(double[] rg)
        {
            T[] rgt = new T[rg.Length];

            if (typeof(T) == typeof(float))
                Array.Copy(Array.ConvertAll(rg, p => Convert.ToSingle(p)), rgt, rg.Length);
            else
                Array.Copy(rg, rgt, rg.Length);

            return rgt;
        }

        protected T[] convert(float[] rg)
        {
            T[] rgt = new T[rg.Length];

            Array.Copy(rg, rgt, rg.Length);

            return rgt;
        }
    }
}
