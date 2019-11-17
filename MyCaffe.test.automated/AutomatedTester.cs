using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Reflection;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.IO.MemoryMappedFiles;

namespace MyCaffe.test.automated
{
    public partial class AutomatedTester : UserControl
    {
        TestClassCollection m_rgTestClasses = new TestClassCollection();
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        EventWaitHandle m_evtGlobalCancel = new EventWaitHandle(false, EventResetMode.AutoReset, "__GRADIENT_CHECKER_CancelEvent__");
        ListViewColumnSorter m_lstSorter = new ListViewColumnSorter();
        TestingProgressGet m_progress = new TestingProgressGet();
        TestingActiveGpuGet m_activeGpu = new TestingActiveGpuGet();
        FileInfo m_fiPath;
        int m_nGpuId = 0;
        bool m_bSkip = false;

        enum LOADTYPE
        {
            ALL,
            SUCCESS,
            FAILURE,
            NOTEXECUTED
        }

        public AutomatedTester()
        {
            InitializeComponent();
            lstTests.ListViewItemSorter = m_lstSorter;

            AppDomain.CurrentDomain.AssemblyResolve += CurrentDomain_AssemblyResolve;
        }

        private Assembly CurrentDomain_AssemblyResolve(object sender, ResolveEventArgs args)
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

        private void AutomatedTester_Load(object sender, EventArgs e)
        {
            if (m_rgTestClasses.Count > 0)
                load(LOADTYPE.ALL);

            m_lstSorter.SortColumn = 2;
            m_lstSorter.Order = SortOrder.Ascending;
            lstTests.Sort();
        }

        public string TestName
        {
            get { return m_rgTestClasses.Name; }
        }

        public void SaveToDatabase()
        {
            m_rgTestClasses.SaveToDatabase();
        }

        public void LoadFromDatabase()
        {
            m_rgTestClasses.LoadFromDatabase();

            if (IsHandleCreated)
                load(LOADTYPE.ALL);

            m_lstSorter.SortColumn = 1;
            m_lstSorter.Order = SortOrder.Ascending;
            lstTests.Sort();
        }

        private void load(LOADTYPE lt)
        {
            lstTests.Items.Clear();

            int nIdx = 1;

            foreach (TestClass tc in m_rgTestClasses)
            {
                foreach (MethodInfoEx mi in tc.Methods)
                {
                    if (lt == LOADTYPE.ALL || (lt == LOADTYPE.FAILURE && mi.Status == MethodInfoEx.STATUS.Failed) || (lt == LOADTYPE.SUCCESS && mi.Status == MethodInfoEx.STATUS.Passed) || (lt == LOADTYPE.NOTEXECUTED && (mi.Status == MethodInfoEx.STATUS.NotExecuted || mi.Status == MethodInfoEx.STATUS.Pending)))
                    {
                        ListViewItem lvi = new ListViewItem(nIdx.ToString(), (int)mi.Status);
                        lvi.SubItems.Add(mi.Priority.ToString());
                        lvi.SubItems.Add(mi.Status.ToString());
                        lvi.SubItems.Add(tc.Name);
                        lvi.SubItems.Add(mi.Name);
                        lvi.SubItems.Add(mi.ErrorInfo.FullErrorString);
                        lvi.Tag = new KeyValuePair<TestClass, MethodInfoEx>(tc, mi);
                        lvi.SubItems[5].Tag = mi.ErrorInfo;

                        if (mi.Status != MethodInfoEx.STATUS.Passed && mi.Status != MethodInfoEx.STATUS.Failed)
                        {
                            lvi.Checked = true;
                            mi.Enabled = true;
                        }
                        else if (mi.Status == MethodInfoEx.STATUS.Failed && lt == LOADTYPE.FAILURE)
                        {
                            lvi.Checked = true;
                            mi.Enabled = true;
                        }
                        else
                        {
                            mi.Enabled = false;
                        }

                        lstTests.Items.Add(lvi);
                        nIdx++;
                    }
                    else
                    {
                        mi.Enabled = false;
                    }
                }
            }
        }

        public void UpdateStatus()
        {
            int nProgressCount = 0;
            MethodInfoEx miLast = null;

            foreach (ListViewItem lvi in lstTests.Items)
            {
                KeyValuePair<TestClass, MethodInfoEx> kvTc = (KeyValuePair<TestClass, MethodInfoEx>)lvi.Tag;
                MethodInfoEx mi = kvTc.Value;

                if (lvi.ImageIndex != (int)mi.Status)
                {
                    lvi.SubItems[1].Text = mi.Status.ToString();
                    lvi.ImageIndex = (int)mi.Status;

                    if (mi.Status == MethodInfoEx.STATUS.Passed)
                    {
                        lvi.Checked = false;
                        mi.Enabled = false;
                    }

                    if (mi.Status == MethodInfoEx.STATUS.Running)
                        lvi.EnsureVisible();
                }

                double? dfProgress = mi.Progress;
                if (!dfProgress.HasValue)
                    dfProgress = m_progress.GetProgress();

                if (dfProgress.HasValue)
                {
                    nProgressCount++;
                    tsItemProgress.Visible = true;
                    pbItemProgress.Visible = true;
                    tsItemProgress.Text = dfProgress.Value.ToString("P");
                    pbItemProgress.Value = (int)(dfProgress.Value * 100.0);
                }

                int? nActiveGpuID = m_activeGpu.GetActiveGpuID();
                if (nActiveGpuID.HasValue)
                    lblActiveGPUVal.Text = nActiveGpuID.ToString();
                else
                    lblActiveGPUVal.Text = "n\a";

                if (mi.ErrorInfo.Error != null && lvi.SubItems[4].Text.Length == 0)
                {
                    lvi.SubItems[4].Text = mi.ErrorInfo.ShortErrorString;
                    lvi.SubItems[4].Tag = mi.ErrorInfo;
                }

                miLast = mi;
            }

            if (nProgressCount == 0)
            {
                tsItemProgress.Visible = false;
                pbItemProgress.Visible = false;
            }
        }

        public string TestAssemblyPath
        {
            get { return m_rgTestClasses.Path; }
            set 
            {
                if (value == null)
                    return;

                m_fiPath = new FileInfo(value);
                m_rgTestClasses.Load(value);

                if (IsHandleCreated)
                    load(LOADTYPE.ALL);
            }
        }

        public int GpuId
        {
            get { return m_nGpuId; }
            set { m_nGpuId = value; }
        }

        private void btnShowAll_Click(object sender, EventArgs e)
        {
            btnShowFailures.Checked = false;
            btnShowPassed.Checked = false;
            btnShowNotExecuted.Checked = false;
            load(LOADTYPE.ALL);
        }

        private void btnShowPassed_Click(object sender, EventArgs e)
        {
            btnShowFailures.Checked = false;
            btnShowAll.Checked = false;
            btnShowNotExecuted.Checked = false;
            load(LOADTYPE.SUCCESS);
        }

        private void btnShowFailures_Click(object sender, EventArgs e)
        {
            btnShowPassed.Checked = false;
            btnShowAll.Checked = false;
            btnShowNotExecuted.Checked = false;
            load(LOADTYPE.FAILURE);
        }

        private void btnSelectNotExecuted_Click(object sender, EventArgs e)
        {
            btnShowFailures.Checked = false;
            btnShowPassed.Checked = false;
            btnShowAll.Checked = false;
            load(LOADTYPE.NOTEXECUTED);
        }

        private void lstTests_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            ListViewHitTestInfo hti = lstTests.HitTest(e.Location);

            if (hti != null)
            {
                if (hti.Item.SubItems[4].Text.Length > 0)
                {
                    ErrorInfo error = hti.Item.SubItems[4].Tag as ErrorInfo;
                    FormError dlg = new FormError(error);

                    dlg.ShowDialog();
                }
            }
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            btnRun.Enabled = false;
            btnAbort.Enabled = true;

            string[] rgstr = m_rgTestClasses.GetEnabledMethods();

            Trace.WriteLine("Running the following enabled tests...");

            foreach (string str in rgstr)
            {
                Trace.WriteLine("   " + str);
            }

            m_rgTestClasses.Run(m_evtCancel, m_bSkip, false, m_nGpuId);
            m_bSkip = false;
        }

        private void btnAbort_Click(object sender, EventArgs e)
        {
            btnAbort.Enabled = false;
            m_evtCancel.Set();
            m_evtGlobalCancel.Set();
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            if (m_rgTestClasses.IsRunning)
            {
                btnShowAll.Enabled = false;
                btnShowFailures.Enabled = false;
                btnShowPassed.Enabled = false;
                btnShowNotExecuted.Enabled = false;
            }
            else
            {
                btnAbort.Enabled = false;
                btnRun.Enabled = true;
                btnShowAll.Enabled = true;
                btnShowFailures.Enabled = true;
                btnShowPassed.Enabled = true;
                btnShowNotExecuted.Enabled = true;
            }

            tsTotalTests.Text = m_rgTestClasses.TotalTestRunCount.ToString("N0") + " of " + m_rgTestClasses.TotalTestCount.ToString("N0");
            tsFailedTests.Text = m_rgTestClasses.TotalTestFailureCount.ToString("N0");
            tsProgress.Value = (int)(m_rgTestClasses.PercentComplete * 100.0);
            tsProgressPct.Text = m_rgTestClasses.PercentComplete.ToString("P");
            tsTestingTime.Text = m_rgTestClasses.TotalTestTimingString;
            UpdateStatus();
        }

        private void lstTests_ItemChecked(object sender, ItemCheckedEventArgs e)
        {
            KeyValuePair<TestClass, MethodInfoEx> kvTc = (KeyValuePair<TestClass, MethodInfoEx>)e.Item.Tag;
            MethodInfoEx mi = kvTc.Value;

            mi.Enabled = e.Item.Checked;
        }

        private void resetToolStripMenuItem_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lstTests.SelectedItems)
            {
                KeyValuePair<TestClass, MethodInfoEx> kvTc = (KeyValuePair<TestClass, MethodInfoEx>)lvi.Tag;
                MethodInfoEx mi = kvTc.Value;

                mi.Enabled = true;
                mi.Status = MethodInfoEx.STATUS.NotExecuted;

                lvi.Checked = true;
                lvi.SubItems[4].Text = "";
                lvi.SubItems[4].Tag = null;
            }
        }

        private void runToolStripMenuItem_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lstTests.Items)
            {
                if (lvi.Selected)
                    lvi.Checked = true;
                else
                    lvi.Checked = false;
            }

            btnRun_Click(sender, e);
        }

        private void lstTests_ColumnClick(object sender, ColumnClickEventArgs e)
        {
            // Determine if clicked column is already the column that is being sorted.
            if (e.Column == m_lstSorter.SortColumn)
            {
                // Reverse the current sort direction for this column.
                if (m_lstSorter.Order == SortOrder.Ascending)
                    m_lstSorter.Order = SortOrder.Descending;
                else
                    m_lstSorter.Order = SortOrder.Ascending;
            }
            else
            {
                // Set the column number that is to be sorted; default to ascending.
                m_lstSorter.SortColumn = e.Column;
                m_lstSorter.Order = SortOrder.Ascending;
            }

            // Perform the sort with these new sort options.
            lstTests.Sort();
        }

        private void btnGradientTests_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lstTests.Items)
            {
                if (lvi.SubItems[1].Text == "NotExecuted")
                {
                    string strName = lvi.SubItems[3].Text.ToLower();

                    if (strName.Contains("gradient"))
                    {
                        lvi.Selected = true;
                        lvi.Checked = true;
                    }
                    else
                    {
                        lvi.Selected = false;
                        lvi.Checked = false;
                    }
                }
                else
                {
                    lvi.Selected = false;
                    lvi.Checked = false;
                }
            }
        }

        private void btnNonGradientTests_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lstTests.Items)
            {
                if (lvi.SubItems[1].Text == "NotExecuted")
                {
                    string strName = lvi.SubItems[3].Text.ToLower();

                    if (!strName.Contains("gradient"))
                    {
                        lvi.Selected = true;
                        lvi.Checked = true;
                    }
                    else
                    {
                        lvi.Selected = false;
                        lvi.Checked = false;
                    }
                }
                else
                {
                    lvi.Selected = false;
                    lvi.Checked = false;
                }
            }
        }

        private void contextMenuStrip1_Opening(object sender, CancelEventArgs e)
        {
            if (m_rgTestClasses.IsRunning)
            {
                e.Cancel = true;
                return;
            }
        }

        private void skipToolStripMenuItem_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lstTests.Items)
            {
                if (lvi.Selected)
                    lvi.Checked = true;
                else
                    lvi.Checked = false;
            }

            m_bSkip = true;
            btnRun_Click(sender, e);
        }
    }

    class TestClassCollection : IEnumerable<TestClass>, IDisposable 
    {
        List<Tuple<string, string, string>> m_rgKnownFailures = new List<Tuple<string, string, string>>();
        string m_strCurrentTest = "";
        string m_strPath;
        string m_strName;
        List<TestClass> m_rgClasses = new List<TestClass>();
        Task m_testTask = null;
        int m_nCurrentTest = 0;
        int m_nTotalTests = 0;
        Stopwatch m_swTiming = new Stopwatch();

        public event EventHandler OnRunCompleted;

        public TestClassCollection()
        {
        }

        public string Path
        {
            get { return m_strPath; }
        }

        public string Name
        {
            get { return m_strName; }
        }

        public string CurrentTest
        {
            get { return m_strCurrentTest; }
        }

        public int Count
        {
            get { return m_rgClasses.Count; }
        }

        public double PercentComplete
        {
            get
            {
                if (!IsRunning)
                    return 0;

                if (m_nTotalTests == 0)
                    return 0;

                return (double)m_nCurrentTest / (int)m_nTotalTests;
            }
        }

        public int TotalTestCount
        {
            get { return m_nTotalTests; }
        }

        private int totalTestCount
        {
            get
            {
                int nCount = 0;

                foreach (TestClass tc in m_rgClasses)
                {
                    foreach (MethodInfoEx mi in tc.Methods)
                    {
                        nCount++;
                    }
                }

                return nCount;
            }
        }

        public int TotalTestRunCount
        {
            get
            {
                int nCount = 0;

                foreach (TestClass tc in m_rgClasses)
                {
                    foreach (MethodInfoEx mi in tc.Methods)
                    {
                        if (mi.Status != MethodInfoEx.STATUS.NotExecuted &&
                            mi.Status != MethodInfoEx.STATUS.Aborted &&
                            mi.Status != MethodInfoEx.STATUS.Pending)
                            nCount++;
                    }
                }

                return nCount;
            }
        }

        public int TotalTestFailureCount
        {
            get
            {
                int nCount = 0;

                foreach (TestClass tc in m_rgClasses)
                {
                    foreach (MethodInfoEx mi in tc.Methods)
                    {
                        if (mi.Status == MethodInfoEx.STATUS.Failed)
                            nCount++;
                    }
                }

                return nCount;
            }
        }

        public TimeSpan TotalTestTiming
        {
            get { return m_swTiming.Elapsed; }
        }

        public string TotalTestTimingString
        {
            get
            {
                TimeSpan ts = TotalTestTiming;
                string str = "";

                if ((int)ts.TotalHours > 0)
                    str += ts.TotalHours.ToString("N0") + " hrs ";

                if ((int)ts.Minutes > 0)
                    str += ts.Minutes.ToString("N0") + " min ";

                str += ts.Seconds.ToString("N0") + " sec.";

                return str;
            }
        }

        public bool IsRunning
        {
            get
            {
                if (m_testTask == null)
                    return false;

                return !(m_testTask.IsCompleted || m_testTask.IsCanceled);
            }
        }

        public string[] GetEnabledMethods()
        {
            List<string> rgstr = new List<string>();

            foreach (TestClass tc in m_rgClasses)
            {
                foreach (MethodInfoEx mi in tc.Methods)
                {
                    if (mi.Enabled)
                    {
                        rgstr.Add(tc.Name + "." + mi.Name);
                    }
                }
            }

            return rgstr.ToArray();
        }

        public void Run(AutoResetEvent evtCancel, bool bSkip, bool bServerMode = false, int nGpuId = 0)
        {
            m_testTask = Task.Factory.StartNew(new Action<object>(testThread), new Tuple<AutoResetEvent, bool, bool, int>(evtCancel, bSkip, bServerMode, nGpuId), TaskCreationOptions.LongRunning);
        }

        private void testThread(object obj)
        {
            Tuple<AutoResetEvent, bool, bool, int> param = obj as Tuple<AutoResetEvent, bool, bool, int>;
            AutoResetEvent evtCancel = param.Item1;
            bool bSkip = param.Item2;
            bool bServerMode = param.Item3;
            int nGpuId = param.Item4;
            TestClass tcCurrent = null;
            MethodInfoEx miCurrent = null;

            
            m_nCurrentTest = 0;
            m_swTiming.Reset();
            m_swTiming.Start();

            string strSrcStart = "MyCaffe Automated Test Start";
            string strSrcResult = "MyCaffe Automated Test Result";
            string strLog = "Application";

            EventLog eventLogStart = new EventLog(strLog);
            eventLogStart.Source = strSrcStart;

            EventLog eventLogResult = new EventLog(strLog);
            eventLogResult.Source = strSrcResult;

            try
            {
                List<Tuple<TestClass, MethodInfoEx>> rgTests = new List<Tuple<TestClass, MethodInfoEx>>();

                foreach (TestClass tc in m_rgClasses)
                {
                    foreach (MethodInfoEx mi in tc.Methods)
                    {
                        rgTests.Add(new Tuple<TestClass, MethodInfoEx>(tc, mi));
                    }
                }

                rgTests = rgTests.OrderBy(p => p.Item2.Priority).ThenBy(p => p.Item2.Index).ToList();

                foreach (Tuple<TestClass, MethodInfoEx> test in rgTests)
                {
                    tcCurrent = test.Item1;
                    miCurrent = test.Item2;

                    if (evtCancel.WaitOne(0))
                        return;

                    if (miCurrent.Enabled && (!bServerMode || miCurrent.Status == MethodInfoEx.STATUS.NotExecuted))
                    {
                        m_strCurrentTest = tcCurrent.Name + "::" + miCurrent.Name;

                        if (bSkip)
                        {
                            miCurrent.ErrorInfo.SetError(new Exception("SKIPPED"));
                            miCurrent.Status = MethodInfoEx.STATUS.Failed;
                        }
                        else
                        {
                            eventLogStart.WriteEntry("Starting " + tcCurrent.Name + "::" + miCurrent.Name + " test.");

                            miCurrent.Invoke(tcCurrent.Instance, nGpuId);

                            if (miCurrent.Status == MethodInfoEx.STATUS.Failed)
                                eventLogResult.WriteEntry("ERROR " + tcCurrent.Name + "::" + miCurrent.Name + " test - " + miCurrent.Status.ToString() + " Error Information: " + miCurrent.ErrorInfo.FullErrorString, EventLogEntryType.Warning);
                            else
                                eventLogResult.WriteEntry("Completed " + tcCurrent.Name + "::" + miCurrent.Name + " test - " + miCurrent.Status.ToString(), EventLogEntryType.Information);
                        }

                        if (miCurrent.Status != MethodInfoEx.STATUS.Aborted)
                            SaveToDatabase(tcCurrent, miCurrent);
                    }

                    m_nCurrentTest++;
                }
            }
            catch (Exception excpt)
            {
                SaveToDatabase(tcCurrent, miCurrent, excpt);
                tcCurrent.InvokeDispose();

                eventLogStart.WriteEntry("Test Exception Thrown! " + excpt.Message, EventLogEntryType.Error);

                throw excpt;
            }
            finally
            {
                m_swTiming.Stop();

                if (OnRunCompleted != null)
                    OnRunCompleted(this, new EventArgs());

                eventLogStart.Close();
                eventLogResult.Close();
            }
        }

        public void Load(string strPath)
        {
            bool bLoadedKnownFailures = false;

            m_strPath = strPath;
            m_rgClasses.Clear();

            FileInfo fi = new FileInfo(strPath);
            Directory.SetCurrentDirectory(fi.DirectoryName);
            int nIdx = 0;

            try
            {
                Assembly a = Assembly.LoadFile(m_strPath);
                MethodInfo miGetPriority = null;
                TestClass tcBase = null;

                m_strName = a.FullName;

                foreach (Type t in a.GetTypes())
                {
                    TestClass tc = new TestClass(t);

                    foreach (MethodInfo mi in t.GetMethods())
                    {
                        if (tc.Name == "TestBase" && mi.Name == "GetPriority")
                        {
                            tcBase = tc;
                            miGetPriority = mi;
                            break;
                        }
                    }

                    if (miGetPriority != null)
                        break;
                }

                foreach (Type t in a.GetTypes())
                {
                    TestClass tc = new TestClass(t);
                    MethodInfo miDispose = null;

                    foreach (MethodInfo mi in t.GetMethods())
                    {
                        if (mi.Name == "Dispose")
                        {
                            miDispose = mi;
                        }
                        else if (tc.Name == "TestBase" && mi.Name == "get_KnownFailures")
                        {
                            if (!bLoadedKnownFailures)
                            {
                                object obj = mi.Invoke(tc.Instance, null);
                                m_rgKnownFailures = obj as List<Tuple<string, string, string>>;
                                bLoadedKnownFailures = true;
                            }
                        }
                        else if (tc.Name == "TestBase" && mi.Name == "GetPriority")
                        {
                            // Do nothing for we already got the method above.
                        }
                        else
                        {
                            IList<CustomAttributeData> rgAttributes = CustomAttributeData.GetCustomAttributes(mi);

                            foreach (CustomAttributeData data in rgAttributes)
                            {
                                string strAttribute = data.ToString();

                                if (strAttribute.Contains("TestMethodAttribute"))
                                {
                                    int nPriority = 0;
                                    if (miGetPriority != null)
                                    {
                                        object obj = miGetPriority.Invoke(tcBase.Instance, new object[] { tc.Name, mi.Name });
                                        nPriority = (int)obj;
                                    }

                                    tc.AddMethod(mi, nIdx, nPriority);
                                    nIdx++;
                                    break;
                                }
                            }
                        }
                    }

                    if (tc.Methods.Count > 0)
                    {
                        if (miDispose != null)
                        {
                            foreach (MethodInfoEx mi in tc.Methods)
                            {
                                mi.DisposeMethod = miDispose;
                            }

                            tc.AddMethod(miDispose, nIdx, 0);
                            nIdx++;
                        }

                        Add(tc);
                    }
                }

                setInitialSettings();

                m_nTotalTests = totalTestCount;
            }
            catch (Exception excpt)
            {
                string strErr = excpt.Message;

                if (excpt.InnerException != null)
                    strErr += " Inner Exception: " + excpt.InnerException.Message;

                if (excpt is ReflectionTypeLoadException)
                {
                    ReflectionTypeLoadException lexcpt = excpt as ReflectionTypeLoadException;

                    foreach (Exception excpt1 in lexcpt.LoaderExceptions)
                    {
                        strErr += Environment.NewLine;
                        strErr += excpt1.Message;
                    }
                }

                MessageBox.Show("Error! " + strErr);
                return;
            }
        }

        private void setInitialSettings()
        {
            if (m_rgKnownFailures == null)
                return;

            foreach (Tuple<string, string, string> knownFailure in m_rgKnownFailures)
            {
                TestClass testClass = Find(knownFailure.Item1);
                if (testClass != null)
                {
                    MethodInfoEx mi = testClass.Methods.Find(knownFailure.Item2);
                    if (mi != null)
                    {
                        mi.Status = MethodInfoEx.STATUS.Failed;
                        mi.ErrorInfo.SetError(new Exception(knownFailure.Item3));
                    }
                }
            }
        }

        public TestClass Find(string strName)
        {
            foreach (TestClass tc in m_rgClasses)
            {
                if (tc.Name == strName)
                    return tc;
            }

            return null;
        }

        public void Add(TestClass tc)
        {
            if (Find(tc.Name) == null)
                m_rgClasses.Add(tc);
        }

        public TestClass Add(Type t)
        {
            TestClass tc = Find(t.Name);

            if (tc == null)
            {
                tc = new TestClass(t);
                m_rgClasses.Add(tc);
            }

            return tc;
        }

        public void Clear()
        {
            m_rgClasses.Clear();
        }

        public void SaveToDatabase(TestClass tc, MethodInfoEx mi, Exception err = null)
        {
            using (TestingEntities entities = TestEntitiesConnection.CreateEntities())
            {
                List<Session> rgSessions = entities.Sessions.Where(p => p.Session1 == m_strName).ToList();

                if (rgSessions.Count > 0)
                {
                    int nSessionID = rgSessions[0].ID;
                    List<Test> rgTest = entities.Tests.Where(p => p.SessionID == nSessionID && p.TestGroup == tc.Name && p.TestMethod == mi.Name).ToList();

                    if (rgTest.Count > 0)
                    {
                        if (err != null)
                        {
                            rgTest[0].ErrorString = getString(err.Message, 1023);
                            rgTest[0].ErrorLocation = getString(err.StackTrace, 1023);
                        }
                        else if (mi.Status == MethodInfoEx.STATUS.Failed && mi.ErrorInfo != null)
                        {
                            string strErr = mi.ErrorInfo.FullErrorString;
                            if (strErr.Length > 1023)
                                strErr = strErr.Substring(0, 1023);

                            string strInfo = mi.ErrorInfo.FullErrorStringLocation;
                            if (strInfo.Length > 1023)
                                strInfo = strInfo.Substring(0, 1023);

                            rgTest[0].ErrorString = strErr;
                            rgTest[0].ErrorLocation = strInfo;
                        }

                        rgTest[0].Priority = mi.Priority;
                        rgTest[0].Success = (mi.Status == MethodInfoEx.STATUS.Passed) ? true : false;
                        decimal dTiming = Math.Min(9999999, (decimal)mi.TestTiming.TotalMilliseconds);
                        rgTest[0].TestTiming = dTiming;

                        entities.SaveChanges();
                    }
                }
            }
        }

        public void SaveToDatabase()
        {
            using (TestingEntities entities = TestEntitiesConnection.CreateEntities())
            {
                int nTotalTestRunCount = TotalTestRunCount;
                int nTotalTestFailureCount = TotalTestFailureCount;
                double dfFailureRate = (nTotalTestRunCount == 0) ? 0 : ((double)nTotalTestFailureCount / (double)nTotalTestRunCount);
                string strName = m_strName;

                Session s;
                List<Session> rgSessions = entities.Sessions.Where(p => p.Session1 == strName).ToList();

                if (rgSessions.Count > 0)
                {
                    s = rgSessions[0];
                }
                else
                {
                    s = new Session();
                    s.Session1 = strName;
                }

                decimal dTotalTestTiming = 1000 * 60 * 60 * 1;
                if (TotalTestTiming.TotalMilliseconds < (double)dTotalTestTiming)
                    dTotalTestTiming = (decimal)TotalTestTiming.TotalMilliseconds;

                s.TimeStamp = DateTime.Now;
                s.TotalTestsRun = TotalTestRunCount;
                s.TotalTestFailures = TotalTestFailureCount;
                s.TestFailureRate = (decimal)dfFailureRate;
                s.TotalTestTiming = dTotalTestTiming;
                s.Path = m_strPath;

                if (rgSessions.Count == 0)
                    entities.Sessions.AddObject(s);

                entities.SaveChanges();

                foreach (TestClass tc in m_rgClasses)
                {
                    foreach (MethodInfoEx mi in tc.Methods)
                    {
                        Test t;
                        List<Test> rgTest = entities.Tests.Where(p => p.SessionID == s.ID && p.TestGroup == tc.Name && p.TestMethod == mi.Name).ToList();

                        if (rgTest.Count > 0)
                        {
                            t = rgTest[0];
                        }
                        else
                        {
                            t = new Test();
                            t.TestGroup = tc.Name;
                            t.TestMethod = mi.Name;
                            t.SessionID = s.ID;
                        }

                        dTotalTestTiming = 1000 * 60 * 60 * 1;
                        if (mi.TestTiming.TotalMilliseconds < (double)dTotalTestTiming)
                            dTotalTestTiming = (decimal)mi.TestTiming.TotalMilliseconds;

                        t.Priority = mi.Priority;
                        t.ErrorString = getString(mi.ErrorInfo.FullErrorString, 1023);
                        t.ErrorLocation = getString(mi.ErrorInfo.FullErrorStringLocation, 1023);
                        t.Success = (mi.Status == MethodInfoEx.STATUS.Passed) ? true : false;
                        t.TestTiming = dTotalTestTiming;

                        if (rgTest.Count == 0)
                            entities.Tests.AddObject(t);
                    }
                }

                entities.SaveChanges();
            }
        }

        private string getString(string str, int nMax)
        {
            if (str.Length > nMax)
                return str.Substring(0, nMax);

            return str;
        }

        public void LoadFromDatabase()
        {
            using (TestingEntities entities = TestEntitiesConnection.CreateEntities())
            {
                List<Session> rgSessions = entities.Sessions.Where(p => p.Session1 == m_strName && p.Path == m_strPath).OrderByDescending(p => p.TimeStamp).Take(1).ToList();

                if (rgSessions.Count > 0)
                {
                    int nSessionID = rgSessions[0].ID;
                    List<Test> rgTests = entities.Tests.Where(p => p.SessionID == nSessionID).ToList();

                    if (rgTests.Count > 0)
                    {
                        foreach (TestClass tc in m_rgClasses)
                        {
                            List<Test> rgTests0 = rgTests.Where(p => p.TestGroup == tc.Name).ToList();

                            if (rgTests0.Count > 0)
                            {
                                foreach (MethodInfoEx mi in tc.Methods)
                                {
                                    List<Test> rgTests1 = rgTests0.Where(p => p.TestMethod.ToLower() == mi.Name.ToLower()).ToList();

                                    if (rgTests1.Count > 0)
                                    {
                                        if (rgTests1[0].Success.GetValueOrDefault())
                                        {
                                            mi.Status = MethodInfoEx.STATUS.Passed;
                                        }
                                        else if (rgTests1[0].ErrorString.Length > 0)
                                        {
                                            mi.Status = MethodInfoEx.STATUS.Failed;
                                            mi.ErrorInfo.SetErrors(rgTests1[0].ErrorString, rgTests1[0].ErrorLocation);
                                        }
                                        else
                                        {
                                            mi.Status = MethodInfoEx.STATUS.NotExecuted;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public void ResetAllTests()
        {
            using (TestingEntities entities = TestEntitiesConnection.CreateEntities())
            {
                string strName = m_strName;

                List<Session> rgSessions = entities.Sessions.Where(p => p.Session1 == strName).ToList();

                if (rgSessions.Count == 0)
                    return;

                Session s = rgSessions[0];
                string strCmd = "DELETE FROM[Testing].[dbo].[Tests] WHERE SessionID = " + s.ID.ToString();
                entities.ExecuteStoreCommand(strCmd);
            }
        }

        public void Dispose()
        {
            foreach (TestClass tc in m_rgClasses)
            {
                tc.Dispose();
            }
        }

        public IEnumerator<TestClass> GetEnumerator()
        {
            return m_rgClasses.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgClasses.GetEnumerator();
        }
    }

    class TestClass : IDisposable 
    {
        Type m_type;
        object m_instance = null;
        MethodInfo m_miDispose = null;
        MethodCollection m_rgMethods = new MethodCollection();

        public TestClass(Type t)
        {
            m_type = t;
        }

        public string Name
        {
            get { return m_type.Name; }
        }

        public object Instance
        {
            get
            {
                if (m_instance == null)
                    m_instance = Activator.CreateInstance(m_type, null);

                return m_instance;
            }
        }

        public MethodCollection Methods
        {
            get { return m_rgMethods; }
        }

        public void AddMethod(MethodInfo mi, int nIndex, int nPriority)
        {
            if (mi.Name == "Dispose")
                m_miDispose = mi;
            else
                m_rgMethods.Add(mi, nIndex, nPriority);
        }

        public void InvokeMethod(string strName)
        {
            MethodInfoEx mi = m_rgMethods.Find(strName);

            if (mi != null && mi.Enabled)
            {
                if (m_instance == null)
                    m_instance = Activator.CreateInstance(m_type, null);
                mi.MethodInfo.Invoke(m_instance, null);
            }
        }

        public void InvokeDispose()
        {
            if (m_instance != null)
            {
                if (m_miDispose != null)
                    m_miDispose.Invoke(m_instance, null);

                m_instance = null;
            }
        }

        public void Dispose()
        {
            InvokeDispose();
            m_type = null;
            m_rgMethods.Dispose();
        }

        public override string ToString()
        {
            return Name;
        }
    }

    class MethodCollection : IEnumerable<MethodInfoEx>, IDisposable 
    {
        List<MethodInfoEx> m_rgMethods = new List<MethodInfoEx>();

        public MethodCollection()
        {
        }

        public int Count
        {
            get { return m_rgMethods.Count; }
        }

        public void Add(MethodInfo mi, int nIndex, int nPriority)
        {
            if (Find(mi.Name) == null)
                m_rgMethods.Add(new MethodInfoEx(mi, MethodInfoEx.STATUS.NotExecuted, nIndex, nPriority, null));
        }

        public MethodInfoEx Find(string strName)
        {
            foreach (MethodInfoEx mi in m_rgMethods)
            {
                if (mi.Name == strName)
                    return mi;
            }

            return null;
        }

        public void Clear()
        {
            m_rgMethods.Clear();
        }

        public void Dispose()
        {
            Clear();
        }

        public IEnumerator<MethodInfoEx> GetEnumerator()
        {
            return m_rgMethods.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgMethods.GetEnumerator();
        }
    }

    class MethodInfoEx
    {
        MethodInfo m_miDispose = null;
        MethodInfo m_mi;
        STATUS m_status;
        ErrorInfo m_errorInfo = new ErrorInfo();
        bool m_bEnabled = true;
        Task m_taskTest = null;
        Stopwatch m_swTiming = new Stopwatch();
        double? m_dfProgress = null;
        Task m_taskStatus;
        AutoResetEvent m_evtStatusCancel = new AutoResetEvent(false);
        int m_nIndex = 0;
        int m_nPriority = 0;

        public enum STATUS
        {
            Aborted = 0,
            Failed = 1,
            NotExecuted = 2,
            Passed = 3,
            Pending = 4,
            Running = 5
        }

        public MethodInfoEx(MethodInfo mi, STATUS s, int nIndex, int nPriority, Exception excpt)
        {
            m_mi = mi;
            m_status = s;
            m_errorInfo.SetError(excpt);
            m_nIndex = nIndex;
            m_nPriority = nPriority;
        }

        public int Priority
        {
            get { return m_nPriority; }
            set { m_nPriority = value; }
        }

        public int Index
        {
            get { return m_nIndex; }
        }

        public MethodInfo DisposeMethod
        {
            get { return m_miDispose; }
            set { m_miDispose = value; }
        }

        public bool Enabled
        {
            get { return m_bEnabled; }
            set { m_bEnabled = value; }
        }

        public string Name
        {
            get { return m_mi.Name; }
        }

        public MethodInfo MethodInfo
        {
            get { return m_mi; }
        }

        public ErrorInfo ErrorInfo
        {
            get { return m_errorInfo; }
        }

        public TimeSpan TestTiming
        {
            get { return m_swTiming.Elapsed; }
        }

        public STATUS Status
        {
            get { return m_status; }
            set { m_status = value; }
        }

        public double? Progress
        {
            get { return m_dfProgress; }
        }

        public void Invoke(object instance, int nGpuId)
        {
            m_taskTest = Task.Factory.StartNew(new Action<object>(invoke), new Tuple<object, int>(instance, nGpuId));
            m_taskTest.Wait();
        }

        private void status()
        {
            TestingProgressGet progress = new TestingProgressGet();

            while (!m_evtStatusCancel.WaitOne(1000))
            {
                m_dfProgress = progress.GetProgress();
            }
        }

        public void invoke(object obj)
        {
            m_taskStatus = Task.Factory.StartNew(new Action(status));
            Tuple<object, int> arg = obj as Tuple<object, int>;
            object instance = arg.Item1;
            int nGpuId = arg.Item2;

            try
            {
                LocalDataStoreSlot lds = Thread.GetNamedDataSlot("GPUID");
                if (lds == null)
                    lds = Thread.AllocateNamedDataSlot("GPUID");

                if (lds != null)
                    Thread.SetData(lds, nGpuId.ToString());

                m_swTiming.Reset();
                m_swTiming.Start();
                m_status = STATUS.Running;
                m_mi.Invoke(instance, null);
                m_status = STATUS.Passed;
            }
            catch (Exception excpt)
            {
                if (excpt.Message == "Aborted!" || (excpt.InnerException != null && excpt.InnerException.Message == "Aborted!"))
                {
                    m_status = STATUS.Aborted;
                }
                else
                {
                    m_status = STATUS.Failed;
                    m_errorInfo.SetError(excpt);
                }
            }
            finally
            {
                m_swTiming.Stop();

                if (m_miDispose != null)
                    m_miDispose.Invoke(instance, null);

                m_evtStatusCancel.Set();
                Thread.FreeNamedDataSlot("GPUID");
            }
        }

        public override string ToString()
        {
            return Name;
        }
    }

    public class ErrorInfo
    {
        object m_syncObj = new object();
        Exception m_excpt;
        List<string> m_rgstrErrors = new List<string>();
        List<string> m_rgstrLocations = new List<string>();

        public ErrorInfo()
        {
        }

        public Exception Error
        {
            get { return m_excpt; }
        }

        public void SetError(Exception e)
        {
            lock (m_syncObj)
            {
                m_excpt = e;

                m_rgstrErrors = new List<string>();
                m_rgstrLocations = new List<string>();

                while (e != null)
                {
                    m_rgstrErrors.Add(e.Message);
                    m_rgstrLocations.Add(e.StackTrace);

                    e = e.InnerException;
                }

                m_rgstrErrors.Reverse();
                m_rgstrLocations.Reverse();
            }
        }

        public void SetErrors(string str, string strLocation)
        {
            lock (m_syncObj)
            {
                string[] rgStr = str.Split('\n');

                m_rgstrErrors = new List<string>();

                foreach (string str0 in rgStr)
                {
                    m_rgstrErrors.Add(str0.Trim(' ', '\n', '\r'));
                }

                string[] rgStrL = strLocation.Split('\n');

                m_rgstrLocations = new List<string>();

                foreach (string str0 in rgStrL)
                {
                    m_rgstrLocations.Add(str0.Trim(' ', '\n', '\r'));
                }

                m_excpt = new Exception(str);
            }
        }

        public string ShortErrorString
        {
            get
            {
                lock (m_syncObj)
                {
                    return m_rgstrErrors[0];
                }
            }
        }

        public string ShortErrorLocation
        {
            get
            {
                lock (m_syncObj)
                {
                    return m_rgstrLocations[0];
                }
            }
        }

        public string FullErrorString
        {
            get
            {
                lock (m_syncObj)
                {
                    string strErr = "";

                    for (int i = 0; i < m_rgstrErrors.Count; i++)
                    {
                        strErr += m_rgstrErrors[i];

                        if (i < m_rgstrErrors.Count - 1)
                        {
                            strErr += Environment.NewLine;
                            strErr += Environment.NewLine;
                        }
                    }

                    return strErr;
                }
            }
        }

        public string FullErrorStringLocation
        {
            get
            {
                lock (m_syncObj)
                {
                    string strErr = "";

                    for (int i = 0; i < m_rgstrLocations.Count; i++)
                    {
                        strErr += m_rgstrLocations[i];

                        if (i < m_rgstrLocations.Count - 1)
                        {
                            strErr += Environment.NewLine;
                        }
                    }

                    return strErr;
                }
            }
        }
    }
}
