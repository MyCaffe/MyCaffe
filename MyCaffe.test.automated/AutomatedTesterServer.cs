using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.test.automated
{
    public partial class AutomatedTesterServer : Component
    {
        FileInfo m_fiPath;
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        EventWaitHandle m_evtGlobalCancel = new EventWaitHandle(false, EventResetMode.AutoReset, "__GRADIENT_CHECKER_CancelEvent__");

        public event EventHandler<ProgressChangedEventArgs> OnProgress;
        public event EventHandler<RunWorkerCompletedEventArgs> OnCompleted;

        public AutomatedTesterServer()
        {
            InitializeComponent();
        }

        public AutomatedTesterServer(IContainer container)
        {
            container.Add(this);

            InitializeComponent();

            AppDomain.CurrentDomain.AssemblyResolve += CurrentDomain_AssemblyResolve;
        }

        private System.Reflection.Assembly CurrentDomain_AssemblyResolve(object sender, ResolveEventArgs args)
        {
            if (args.Name.Contains(".resources"))
                return null;

            int nPos = args.Name.IndexOf(',');
            if (nPos >= 0)
            {
                string strName = args.Name.Substring(0, nPos) + ".dll";
                string strPath = m_fiPath.DirectoryName + "\\" + strName;

                Trace.WriteLine("Loading '" + strPath + "'.");

                return Assembly.LoadFile(strPath);
            }

            return null;
        }

        public bool Initialize(string strDbPath, string strInstance)
        {
            TestDatabaseManager dbMgr = new TestDatabaseManager(strInstance);
            bool bExists;
            Exception err = dbMgr.DatabaseExists(out bExists);

            if (err != null)
                throw err;

            if (bExists)
                return false;

            err = dbMgr.CreateDatabase(strDbPath);
            if (err != null)
                throw err;

            return true;
        }

        public bool IsRunning
        {
            get { return m_bw.IsBusy; }
        }

        public void Run(string strTestDllFile, bool bResetAllTests)
        {
            m_evtCancel.Reset();
            m_evtGlobalCancel.Reset();
            m_fiPath = new FileInfo(strTestDllFile);
            m_bw.RunWorkerAsync(new AutoTestParams(strTestDllFile, bResetAllTests));
        }

        public void Abort()
        {
            m_evtCancel.Set();
            m_evtGlobalCancel.Set();
            m_bw.CancelAsync();
        }

        private void m_bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if (OnProgress != null)
                OnProgress(sender, e);
        }

        private void m_bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (OnCompleted != null)
                OnCompleted(sender, e);
        }

        private void m_bw_DoWork(object sender, DoWorkEventArgs e)
        {
            BackgroundWorker bw = sender as BackgroundWorker;
            AutoTestParams param = e.Argument as AutoTestParams;
            AutoTestProgressInfo pi = new AutoTestProgressInfo();
            TestClassCollection colTests = new TestClassCollection();

            colTests.OnRunCompleted += colTests_OnRunCompleted;
            colTests.Load(param.TestDllFile);

            if (param.ResetAllTests)
                colTests.ResetAllTests();
            else
                colTests.LoadFromDatabase();

            colTests.Run(m_evtCancel, false, true);

            while (!bw.CancellationPending && colTests.IsRunning)
            {
                Thread.Sleep(1000);
                string strCurrent = colTests.CurrentTest;

                if (strCurrent.Length > 0)
                    strCurrent = " [" + strCurrent + "]";

                pi.Set(colTests.PercentComplete, colTests.TotalTestTimingString + " completed " + colTests.TotalTestRunCount.ToString("N0") + " of " + colTests.TotalTestCount.ToString("N0") + " (" + colTests.TotalTestFailureCount.ToString("N0") + " failed)." + strCurrent);
                bw.ReportProgress((int)(pi.Progress * 100), pi);
            }

            colTests.SaveToDatabase();
        }

        private void colTests_OnRunCompleted(object sender, EventArgs e)
        {
            m_bw.CancelAsync();
        }
    }

    public class AutoTestParams
    {
        string m_strTestDllFile;
        bool m_bResetAllTests = false;

        public AutoTestParams(string strTestDllFile, bool bResetAllTests)
        {
            m_strTestDllFile = strTestDllFile;
            m_bResetAllTests = bResetAllTests;
        }

        public string TestDllFile
        {
            get { return m_strTestDllFile; }
        }

        public bool ResetAllTests
        {
            get { return m_bResetAllTests; }
        }
    }

    public class AutoTestProgressInfo
    {
        double m_dfProgress = 0;
        string m_strMsg = null;
        bool m_bError = false;

        public AutoTestProgressInfo()
        {
        }

        public void Set(string strMsg)
        {
            m_strMsg = strMsg;
            m_bError = false;
        }

        public void Set(double dfProgress, string strMsg)
        {
            m_strMsg = strMsg;
            m_dfProgress = dfProgress;
            m_bError = false;
        }

        public void Set(double dfProgress)
        {
            m_dfProgress = dfProgress;
            m_bError = false;
        }

        public void Set(Exception excpt)
        {
            m_bError = true;
            m_strMsg = excpt.Message;
        }

        public string Message
        {
            get { return m_strMsg; }
        }

        public double Progress
        {
            get { return m_dfProgress; }
        }

        public bool Error
        {
            get { return m_bError; }
        }

        public override string ToString()
        {
            string str = "(" + m_dfProgress.ToString("P") + ") ";

            if (m_bError)
                str += " ERROR: ";

            str += m_strMsg;

            return str;
        }
    }
}
