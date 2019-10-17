using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.test.automated;
using MyCaffe.common;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using MyCaffe.param;
using MyCaffe.gym;
using System.IO;
using System.Net;
using System.Reflection;
using MyCaffe.test;

namespace MyCaffe.app
{
    public partial class FormMain : Form
    {
        CancelEvent m_evtCancel = new CancelEvent();
        CancelEvent m_evtCaffeCancel = new CancelEvent();
        AutoResetEvent m_evtCommandRead = new AutoResetEvent(false);
        AutoResetEvent m_evtThreadDone = new AutoResetEvent(false);
        IXMyCaffeNoDb<float> m_caffeRun = null;
        COMMAND m_Cmd = COMMAND.NONE;
        byte[] m_rgTrainedWeights = null;
        SimpleDatum m_sdImageMean = null;
        AutomatedTesterServer m_autoTest = new AutomatedTesterServer();
        bool m_bLoading = false;
        bool m_bTesting = false;
        bool m_bCaffeCreated = false;
        FormWait m_dlgWait = null;
        StreamWriter m_swResNetTest = null;
        MODE m_mode = MODE.UNKNOWN;
        double m_dfLastLoss = 0;
        int m_nLastTrainingIteration = 0;
        double m_dfTotalTiming = 0;
        int m_nTimingCount = 0;
        Stopwatch m_swGlobalTiming = null;
        string m_strTestLogDir = null;
        Log m_log;
        Task m_trainerTask = null;
        CancelEvent m_evtCancelTraining = new CancelEvent();
        Task m_nsTask = null;
        CancelEvent m_evtCancelNs = new CancelEvent();
        MyCaffeGymUiServiceHost m_gymHost;
        TestingProgressGet m_progress = new TestingProgressGet();
        string m_strNsResults = null;
        string m_strAtariRom = "pong";
        int m_nMinComputeMajor = 0;
        int m_nMinComputeMinor = 0;
        string m_strDllPath = null;

        delegate void fnVerifyGpu(int nGpuId);
        delegate void fnSetStatus(string strMsg, STATUS status, bool bBreath);
        delegate void fnNsDone();

        enum STATUS
        {
            INFO,
            INFO2,
            WARNING,
            ERROR
        }

        enum COMMAND
        {
            NONE,
            CREATE,
            DESTROY,
            TRAIN,
            TEST,
            DEVICEINFO,
            SPECIALTEST_ALEXNETCIFAR,
            SPECIALTEST_RESNETCIFAR
        }

        enum MODE
        {
            UNKNOWN,
            WDM,
            TCC
        }

        public FormMain()
        {
            InitializeComponent();

            m_log = new Log("Test Run");
            m_log.OnWriteLine += Log_OnWriteLine;
            m_caffeRun = new MyCaffeControl<float>(new SettingsCaffe(), m_log, m_evtCancel);
            m_progress.Initialize();

            if (lvStatus is ListViewEx)
                ((ListViewEx)lvStatus).RowHeight = 12;
        }

        private void FormMain_Load(object sender, EventArgs e)
        {
            try
            {
                List<string> rgSqlInst = DatabaseInstanceQuery.GetInstances();

                m_bwProcess.RunWorkerAsync();

                if (!File.Exists("index.chm"))
                    localHelpToolStripMenuItem.Enabled = false;

                if (rgSqlInst == null || rgSqlInst.Count == 0)
                {
                    setStatus("You must download and install 'Microsoft SQL' or 'Microsoft SQL Express' first!", STATUS.WARNING);
                    setStatus("see 'https://www.microsoft.com/en-us/sql-server/sql-server-editions-express'");
                    setStatus("");
                    return;
                }
                else if (rgSqlInst.Count == 1)
                {
                    if (rgSqlInst[0] != ".\\MSSQLSERVER")
                        EntitiesConnection.GlobalDatabaseServerName = rgSqlInst[0];
                }
                else
                {
                    FormSqlInstances dlg = new FormSqlInstances(rgSqlInst);

                    if (dlg.ShowDialog() == DialogResult.OK)
                    {
                        if (dlg.Instance != ".\\MSSQLSERVER")
                            EntitiesConnection.GlobalDatabaseServerName = dlg.Instance;
                    }
                    else
                    {
                        setStatus("You are NOT connected to SQL.", STATUS.WARNING);
                    }
                }

                setStatus("Using SQL Instance '" + EntitiesConnection.GlobalDatabaseServerName + "'", STATUS.INFO2);

                DatabaseManagement dbMgr = new DatabaseManagement("DNN", "", EntitiesConnection.GlobalDatabaseServerName);
                bool bExists;
                Exception err = dbMgr.DatabaseExists(out bExists);

                if (err != null)
                    setStatus("ERROR: " + err.Message, STATUS.ERROR);
                else if (!bExists)
                    createDatabaseToolStripMenuItem_Click(this, new EventArgs());
                else
                    setStatus("Using database '" + dbMgr.Name + "'", STATUS.INFO2);

                setStatus("", STATUS.INFO2);

                m_autoTest.OnProgress += m_autoTest_OnProgress;
                m_autoTest.OnCompleted += m_autoTest_OnCompleted;

                setStatus("The MyCaffe Test App supports two different types of automated testing:", STATUS.INFO2);
                setStatus(" 1.) User interface based automated testing via the 'Test | Run Autotests UI', and", STATUS.INFO2);
                setStatus(" 2.) Server based automated testing via the 'Test | Start Server Autotests' menu.", STATUS.INFO2);
                setStatus("Server auto tests can easily integrate into other applications.", STATUS.INFO2);
                setStatus("NOTE: Known test failures are pre-set with a FAILURE status.", STATUS.INFO2);
                setStatus("----------------------------------------------------------------------------------", STATUS.INFO2);

                DatasetFactory factory = new DatasetFactory();
                int nCifarID = factory.GetDatasetID("CIFAR-10");
                int nMnistID = factory.GetDatasetID("MNIST");

                if (nCifarID == 0 || nMnistID == 0)
                {
                    setStatus(" !Before running any automated tests, make sure to load the following datasets:", STATUS.WARNING);

                    if (nCifarID == 0)
                        setStatus("    CIFAR-10", STATUS.WARNING);

                    if (nMnistID == 0)
                        setStatus("    MNIST (1 channel)", STATUS.WARNING);

                    setStatus(" see the 'Database' menu.", STATUS.WARNING);
                }

                m_dlgWait = new FormWait();
                m_bwInit.RunWorkerAsync();
                m_dlgWait.ShowDialog();
                m_bwUrlCheck.RunWorkerAsync();

                m_gymHost = new MyCaffeGymUiServiceHost();

                try
                {
                    m_gymHost.Open();
                }
                catch (Exception excpt)
                {
                    setStatus(excpt.Message, STATUS.ERROR);
                    m_gymHost = null;
                }
            }
            catch (Exception excpt)
            {
                string strErr = excpt.Message;

                if (excpt.InnerException != null)
                    strErr += " " + excpt.InnerException.Message;

                if (strErr.Contains("login") && strErr.Contains("DNN"))
                    strErr += " Make sure that this user can access the DNN database - this setting is made using the SQL Management Studio.";

                setStatus("ERROR: " + strErr, STATUS.ERROR);
            }
        }

        private void m_bwInit_DoWork(object sender, DoWorkEventArgs e)
        {
            List<string> rgstrGpu = new List<string>();

            // Setup the GPU menu with all GPU's in the system and 
            //  select the first GPU as the default for testing.
            CudaDnn<float> cuda = new CudaDnn<float>(0);
            int nDeviceCount = cuda.GetDeviceCount();
            for (int i = 0; i < nDeviceCount; i++)
            {
                string strDevice = cuda.GetDeviceName(i);
                rgstrGpu.Add(strDevice);
            }

            m_strDllPath = cuda.GetRequiredCompute(out m_nMinComputeMajor, out m_nMinComputeMinor);

            cuda.Dispose();

            e.Result = rgstrGpu;
        }

        private void m_bwInit_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            m_dlgWait.Close();

            if (e.Error != null)
            {
                MessageBox.Show("Initializing Error: " + e.Error.Message, "Initialization Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                List<string> rgstrGpu = e.Result as List<string>;
                int nGpu = Properties.Settings.Default.GPU;

                if (rgstrGpu != null)
                {
                    for (int i = 0; i < rgstrGpu.Count; i++)
                    {
                        string strDevice = rgstrGpu[i];
                        ToolStripMenuItem menu = new ToolStripMenuItem(strDevice);
                        menu.Tag = i;
                        menu.Click += menuGpu_Click;

                        if (i == nGpu)
                        {
                            menu.Checked = true;
                            verifyGpuId(strDevice, i);
                        }

                        gpuToolStripMenuItem.DropDownItems.Add(menu);
                    }
                }
            }
        }

        private void menuGpu_Click(object sender, EventArgs e)
        {
            ToolStripMenuItem menu = sender as ToolStripMenuItem;

            foreach (ToolStripMenuItem item in gpuToolStripMenuItem.DropDownItems)
            {
                item.Checked = false;
            }

            menu.Checked = true;

            verifyGpuId(menu.Text, (int)menu.Tag);
        }

        private int getGpu()
        {
            int nGpuId = 0;

            foreach (ToolStripMenuItem menu in gpuToolStripMenuItem.DropDownItems)
            {
                if (menu.Checked)
                {
                    nGpuId = (int)menu.Tag;
                    break;
                }
            }

            return nGpuId;
        }

        private void getMajorMinor(string strName, out int nMajor, out int nMinor)
        {
            nMajor = 0;
            nMinor = 0;

            string strTarget = "compute ";
            int nPos = strName.IndexOf(strTarget);
            if (nPos <= 0)
                return;

            strName = strName.Substring(nPos + strTarget.Length);
            strName = strName.TrimEnd(')');

            nPos = strName.IndexOf('.');
            if (nPos <= 0)
                return;

            string strMajor = strName.Substring(0, nPos);
            string strMinor = strName.Substring(nPos + 1);

            nMajor = int.Parse(strMajor);
            nMinor = int.Parse(strMinor);
        }

        private void verifyGpuId(string strName, int nGpuId)
        {
            int nActualMajor;
            int nActualMinor;
            getMajorMinor(strName, out nActualMajor, out nActualMinor);

            if (nActualMajor < m_nMinComputeMajor || (nActualMajor == m_nMinComputeMajor && nActualMinor < m_nMinComputeMinor))
            {
                setStatus("The current DLL '" + m_strDllPath + "' requires a minimum compute value of " + m_nMinComputeMajor.ToString() + "." + m_nMinComputeMinor.ToString(), STATUS.WARNING);
                setStatus("The current GPU (ID=" + nGpuId.ToString() + ") has a lower compute of " + nActualMajor.ToString() + "." + nActualMinor, STATUS.ERROR);
                setStatus("Please use a different GPU or use a different CudaDNN dll version that supports a lower compute value.", STATUS.WARNING);
            }
        }

        private string getGpuName()
        {
            foreach (ToolStripMenuItem menu in gpuToolStripMenuItem.DropDownItems)
            {
                if (menu.Checked)
                    return ((int)menu.Tag).ToString() + ": " + menu.Text;
            }

            return "No GPU Selected";
        }

        private bool checkURL(string url)
        {
            bool pageExists = false;
            try
            {
                WebRequest r = WebRequest.Create(url);

                if (r is HttpWebRequest)
                {
                    HttpWebRequest request = (HttpWebRequest)r;
                    request.Method = WebRequestMethods.Http.Head;
                    HttpWebResponse response = (HttpWebResponse)request.GetResponse();
                    pageExists = response.StatusCode == HttpStatusCode.OK;
                }
                else
                {
                    pageExists = true;
                }
            }
            catch (Exception)
            {
                return false;
            }

            return pageExists;
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_evtCancel.Set();
            m_evtThreadDone.WaitOne();
            Close();
        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormAbout dlg = new app.FormAbout();

            dlg.ShowDialog();
        }

        private void createDatabaseToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormCreateDatabase dlg = new app.FormCreateDatabase();

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                try
                {
                    MyCaffeImageDatabase.CreateDatabase(dlg.DatabaseName, dlg.DatabasePath);
                    setStatus("Database '" + dlg.DatabaseName + "' created in location: '" + dlg.DatabasePath + "'");
                }
                catch (Exception excpt)
                {
                    setStatus("ERROR Creating Database! " + excpt.Message);
                }
            }
        }

        private void runAutotestsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            TestDatabaseManager dbMgrTest = new TestDatabaseManager(EntitiesConnection.GlobalDatabaseServerName);
            bool bExists;
            Exception err = dbMgrTest.DatabaseExists(out bExists);

            if (err != null)
            {
                setStatus("ERROR Querying Testing Database! " + err.Message);
                return;
            }

            if (!bExists)
            {
                FormCreateDatabase dlg = new FormCreateDatabase(dbMgrTest.DatabaseName, "You must create the 'Testing' Database first");

                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                err = dbMgrTest.CreateDatabase(dlg.DatabasePath);
                if (err != null)
                {
                    setStatus("ERROR Creating Testing Database! " + err.Message);
                    return;
                }

                setStatus("Testing database created.");
            }

            openFileDialogAutoTests.InitialDirectory = initialDirectory;
            if (openFileDialogAutoTests.ShowDialog() == DialogResult.OK)
            {
                FormAutomatedTests dlg = new FormAutomatedTests(openFileDialogAutoTests.FileName, getGpu());

                setStatus("Running automatic tests.");
                dlg.ShowDialog();
            }
        }

        private string initialDirectory
        {
            get
            {
                string codeBase = Process.GetCurrentProcess().MainModule.FileName;
                UriBuilder uri = new UriBuilder(codeBase);
                string strPath = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(strPath);
            }
        }

        private void setStatus(string str, STATUS status = STATUS.INFO, bool bBreathe = false)
        {
            int nMaxLines = 2000;

            ListViewItem lvi = new ListViewItem(str);

            if (status == STATUS.ERROR || str.Contains("ERROR"))
                lvi.BackColor = Color.Salmon;
            else if (status == STATUS.WARNING || str.Contains("WARING"))
                lvi.BackColor = Color.Yellow;
            else if (status == STATUS.INFO2)
            {
                lvi.BackColor = Color.AliceBlue;
                lvi.ForeColor = Color.SteelBlue;
            }

            lvStatus.Items.Add(lvi);

            if (lvi.BackColor == Color.Salmon && str.Contains("storage instruction"))
            {
                lvi = new ListViewItem("CUDA storage instruction errors can corrupt the current application's GPU state - Please restart this application.");
                lvi.BackColor = Color.Yellow;
                lvStatus.Items.Add(lvi);
            }

            while (lvStatus.Items.Count > nMaxLines)
            {
                lvStatus.Items.RemoveAt(0);
            }

            if (bBreathe)
            {
                Thread.Sleep(0);
                Application.DoEvents();
            }

            lvi.EnsureVisible();
        }

        private void Log_OnWriteLine(object sender, LogArg e)
        {
            Invoke(new fnSetStatus(setStatus), e.Message, STATUS.INFO, true);
        }

        private void loadMNISTToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormMnist dlg = new FormMnist();

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                m_bLoading = true;
                m_evtCancel.Set();
                loadMNISTToolStripMenuItem.Enabled = false;
                loadCIFAR10ToolStripMenuItem.Enabled = false;
                loadVOC2007ToolStripMenuItem.Enabled = false;
                trainMNISTToolStripMenuItem.Enabled = false;
                testMNISTToolStripMenuItem.Enabled = false;
                createMyCaffeToolStripMenuItem.Enabled = false;
                destroyMyCaffeToolStripMenuItem.Enabled = false;
                deviceInformationToolStripMenuItem.Enabled = false;
                runTestImageToolStripMenuItem.Enabled = false;
                m_bwLoadMnistDatabase.RunWorkerAsync(dlg.Parameters);
            }
        }

        private void loadCIFAR10ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormCifar10 dlg = new app.FormCifar10();

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                m_bLoading = true;
                loadMNISTToolStripMenuItem.Enabled = false;
                loadCIFAR10ToolStripMenuItem.Enabled = false;
                loadVOC2007ToolStripMenuItem.Enabled = false;
                trainMNISTToolStripMenuItem.Enabled = false;
                testMNISTToolStripMenuItem.Enabled = false;
                createMyCaffeToolStripMenuItem.Enabled = false;
                destroyMyCaffeToolStripMenuItem.Enabled = false;
                deviceInformationToolStripMenuItem.Enabled = false;
                runTestImageToolStripMenuItem.Enabled = false;
                m_bwLoadCiFar10Database.RunWorkerAsync(dlg.Parameters);
            }
        }

        private void loadVOC2007ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormVOC dlg = new FormVOC();

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                m_bLoading = true;
                loadMNISTToolStripMenuItem.Enabled = false;
                loadCIFAR10ToolStripMenuItem.Enabled = false;
                loadVOC2007ToolStripMenuItem.Enabled = false;
                trainMNISTToolStripMenuItem.Enabled = false;
                testMNISTToolStripMenuItem.Enabled = false;
                createMyCaffeToolStripMenuItem.Enabled = false;
                destroyMyCaffeToolStripMenuItem.Enabled = false;
                deviceInformationToolStripMenuItem.Enabled = false;
                runTestImageToolStripMenuItem.Enabled = false;
                cancelToolStripMenuItem.Enabled = true;
                m_bwLoadVOCDatabase.RunWorkerAsync(dlg.Parameters);
            }
        }

        private void m_bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                setStatus("ERROR: " + e.Error.Message, STATUS.ERROR);
                runTestImageToolStripMenuItem.Enabled = false;
            }
            else if (e.Cancelled)
            {
                setStatus("ABORTED!", STATUS.WARNING);
                runTestImageToolStripMenuItem.Enabled = false;
            }
            else
            {
                setStatus("COMPLETED.");
            }

            startAutotestsToolStripMenuItem.Enabled = !m_bLoading;
            runAutotestsToolStripMenuItem.Enabled = !m_bLoading;
            createDatabaseToolStripMenuItem.Enabled = !m_bLoading;
            trainMNISTToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
            testMNISTToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
            destroyMyCaffeToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
            deviceInformationToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
            createMyCaffeToolStripMenuItem.Enabled = !m_bLoading && !m_bCaffeCreated;
            specialTestsToolStripMenuItem.Enabled = !m_bLoading && !m_bCaffeCreated;
            loadMNISTToolStripMenuItem.Enabled = !m_bLoading;
            loadCIFAR10ToolStripMenuItem.Enabled = !m_bLoading;
            loadVOC2007ToolStripMenuItem.Enabled = !m_bLoading;
            runTestImageToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
            cancelToolStripMenuItem.Enabled = false;
        }

        private void m_bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            ProgressInfo pi = e.UserState as ProgressInfo;

            runTestImageToolStripMenuItem.Enabled = false;

            if (pi.Alive.HasValue)
            {
                if (pi.Alive.Value == true)
                {
                    startAutotestsToolStripMenuItem.Enabled = !m_bLoading;
                    runAutotestsToolStripMenuItem.Enabled = !m_bLoading;
                    createDatabaseToolStripMenuItem.Enabled = !m_bLoading;
                    loadMNISTToolStripMenuItem.Enabled = !m_bLoading;
                    loadCIFAR10ToolStripMenuItem.Enabled = !m_bLoading;
                    createMyCaffeToolStripMenuItem.Enabled = !m_bLoading && !m_bCaffeCreated && !m_bTesting;
                    destroyMyCaffeToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
                    trainMNISTToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
                    testMNISTToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
                    deviceInformationToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
                    runTestImageToolStripMenuItem.Enabled = !m_bLoading && m_bCaffeCreated;
                }
                else
                {
                    startAutotestsToolStripMenuItem.Enabled = !m_bLoading;
                    runAutotestsToolStripMenuItem.Enabled = !m_bLoading;
                    createDatabaseToolStripMenuItem.Enabled = !m_bLoading;
                    loadMNISTToolStripMenuItem.Enabled = !m_bLoading;
                    loadCIFAR10ToolStripMenuItem.Enabled = !m_bLoading;
                    createMyCaffeToolStripMenuItem.Enabled = !m_bLoading && !m_bCaffeCreated;
                }

                setStatus(pi.Message);

                if (pi.Message == "MyCaffe Training completed.")
                {
                    if (m_rgTrainedWeights != null)
                    {
                        string strModel = System.Text.Encoding.UTF8.GetString(Properties.Resources.lenet_train_test);
                        m_caffeRun.LoadToRun(strModel, m_rgTrainedWeights, new BlobShape(1, 1, 28, 28), m_sdImageMean);
                        runTestImageToolStripMenuItem.Enabled = true;
                    }
                    else
                    {
                        runTestImageToolStripMenuItem.Enabled = false;
                    }
                }
            }
            else
            {
                setStatus("(" + pi.Percentage.ToString("P") + ") " + pi.Message);
            }
        }

        private void m_bwLoadMnistDatabase_DoWork(object sender, DoWorkEventArgs e)
        {
            MnistDataLoader loader = new app.MnistDataLoader(e.Argument as MnistDataParameters);

            loader.OnProgress += loader_OnProgress;
            loader.OnError += loader_OnError;
            loader.OnCompleted += loader_OnCompleted;

            loader.LoadDatabase();            
        }

        private void m_bwLoadCiFar10Database_DoWork(object sender, DoWorkEventArgs e)
        {
            CiFar10DataLoader loader = new CiFar10DataLoader(e.Argument as CiFar10DataParameters);

            loader.OnProgress += loader_OnProgress;
            loader.OnError += loader_OnError;
            loader.OnCompleted += loader_OnCompleted;

            loader.LoadDatabase();
        }

        private void m_bwLoadVOCDatabase_DoWork(object sender, DoWorkEventArgs e)
        {
            m_evtCaffeCancel.Reset();
            VOCDataLoader loader = new VOCDataLoader(e.Argument as VOCDataParameters, m_evtCaffeCancel);

            loader.OnProgress += loader_OnProgress;
            loader.OnError += loader_OnError;
            loader.OnCompleted += loader_OnCompleted;

            loader.LoadDatabase();
        }

        private void loader_OnCompleted(object sender, EventArgs e)
        {
            m_bLoading = false;
        }

        private void loader_OnError(object sender, ProgressArgs e)
        {
            if (sender.GetType() == typeof(MnistDataLoader))
                m_bwLoadMnistDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else if (sender.GetType() == typeof(CiFar10DataLoader))
                m_bwLoadCiFar10Database.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else if (sender.GetType() == typeof(VOCDataLoader))
                m_bwLoadVOCDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);

            m_bLoading = false;
        }

        private void loader_OnProgress(object sender, ProgressArgs e)
        {
            if (sender.GetType() == typeof(MnistDataLoader))
                m_bwLoadMnistDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else if (sender.GetType() == typeof(CiFar10DataLoader))
                m_bwLoadCiFar10Database.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else if (sender.GetType() == typeof(VOCDataLoader))
                m_bwLoadVOCDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
        }

        private void createMyCaffeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_bCaffeCreated = true;
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadCIFAR10ToolStripMenuItem.Enabled = false;
            loadVOC2007ToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            specialTestsToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();
            m_evtCaffeCancel.Reset();

            if (!m_bwProcess.IsBusy)
                m_bwProcess.RunWorkerAsync();

            m_Cmd = COMMAND.CREATE;
            m_evtCommandRead.Set();
        }

        private void destroyMyCaffeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            runTestImageToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();

            m_Cmd = COMMAND.DESTROY;
            m_evtCommandRead.Set();
        }

        private void trainMNISTToolStripMenuItem_Click(object sender, EventArgs e)
        {
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            loadCIFAR10ToolStripMenuItem.Enabled = false;
            loadVOC2007ToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();
            m_evtCaffeCancel.Reset();

            m_Cmd = COMMAND.TRAIN;
            m_evtCommandRead.Set();
            cancelToolStripMenuItem.Enabled = true;
        }

        private void testMNISTToolStripMenuItem_Click(object sender, EventArgs e)
        {
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            loadCIFAR10ToolStripMenuItem.Enabled = false;
            loadVOC2007ToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();
            m_evtCaffeCancel.Reset();

            m_Cmd = COMMAND.TEST;
            m_evtCommandRead.Set();
            cancelToolStripMenuItem.Enabled = true;
        }

        private void runTestImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            FormTestImage dlg = new app.FormTestImage();

            dlg.ShowDialog();

            if (dlg.Image != null && m_caffeRun != null && m_rgTrainedWeights != null)
            {
                Bitmap bmp = ImageTools.ResizeImage(dlg.Image, 28, 28);
                Stopwatch sw = new Stopwatch();

                sw.Start();
                ResultCollection res = m_caffeRun.Run(bmp);
                sw.Stop();

                setStatus("====================================");
                setStatus("Detected Label = " + res.DetectedLabel.ToString() + " in " + sw.Elapsed.TotalMilliseconds.ToString("N4") + " ms.");
                setStatus("--Results--");

                foreach (KeyValuePair<int, double> kv in res.ResultsSorted)
                {
                    setStatus("Label " + kv.Key.ToString() + " -> " + kv.Value.ToString("N5"));
                }
            }
        }

        private void deviceInformationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            loadCIFAR10ToolStripMenuItem.Enabled = false;
            loadVOC2007ToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();

            m_Cmd = COMMAND.DEVICEINFO;
            m_evtCommandRead.Set();
        }

        private void abortToolStripMenuItem_Click(object sender, EventArgs e)
        {
            abortToolStripMenuItem.Enabled = false;
            m_evtCaffeCancel.Set();
            m_evtCancel.Set();
        }

        private void cancelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_evtCaffeCancel.Set();
        }

        private void m_bwProcess_DoWork(object sender, DoWorkEventArgs e)
        {
            BackgroundWorker bw = sender as BackgroundWorker;
            Log log = new Log("MyCaffe");
            MyCaffeControl<float> caffe = null;

            log.OnWriteLine += log_OnWriteLine1;

            while (!m_evtCancel.WaitOne(0))
            {
                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.AddRange(m_evtCancel.Handles);
                rgWait.Add(m_evtCommandRead);

                int nWait = WaitHandle.WaitAny(rgWait.ToArray());
                if (nWait > 0)
                {
                    try
                    {
                        switch (m_Cmd)
                        {
                            case COMMAND.CREATE:
                                SettingsCaffe settings = new SettingsCaffe();
                                settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
                                settings.EnableRandomInputSelection = true;
                                settings.GpuIds = getGpu().ToString();

                                caffe = new MyCaffeControl<float>(settings, log, m_evtCaffeCancel);

                                string strSolver = System.Text.Encoding.UTF8.GetString(Properties.Resources.lenet_solver);
                                string strModel = System.Text.Encoding.UTF8.GetString(Properties.Resources.lenet_train_test);

                                caffe.Load(Phase.TRAIN, strSolver, strModel, null);
                                bw.ReportProgress(1, new ProgressInfo(1, 1, "MyCaffe Created.", null, true));
                                break;

                            case COMMAND.DESTROY:
                                m_bCaffeCreated = false;
                                caffe.Dispose();
                                caffe = null;
                                bw.ReportProgress(0, new ProgressInfo(0, 0, "MyCaffe Destroyed", null, false));
                                break;

                            case COMMAND.TRAIN:
                                caffe.Train(5000);
                                m_rgTrainedWeights = caffe.GetWeights();
                                m_sdImageMean = caffe.GetImageMean();
                                bw.ReportProgress(0, new ProgressInfo(0, 0, "MyCaffe Training completed.", null, true));
                                break;

                            case COMMAND.TEST:
                                double dfAccuracy = caffe.Test(100);
                                log.WriteLine("Accuracy = " + dfAccuracy.ToString("P"));
                                bw.ReportProgress(0, new ProgressInfo(0, 0, "MyCaffe Testing completed.", null, true));
                                break;

                            case COMMAND.DEVICEINFO:
                                string str1 = caffe.GetDeviceName(getGpu());
                                str1 += Environment.NewLine;
                                str1 += caffe.Cuda.GetDeviceInfo(getGpu(), true);
                                bw.ReportProgress(0, new ProgressInfo(0, 0, str1, null, true));
                                break;

                            case COMMAND.SPECIALTEST_ALEXNETCIFAR:
                            case COMMAND.SPECIALTEST_RESNETCIFAR:
                                bw.ReportProgress(0, new ProgressInfo(0, 0, "Starting special test " + m_Cmd.ToString(), null, true));
                                caffe = runTest(m_Cmd, log);
                                bw.ReportProgress(0, new ProgressInfo(0, 0, "Completed special test " + m_Cmd.ToString(), null, true));
                                break;
                        }
                    }
                    catch (Exception excpt)
                    {
                        log.WriteError(excpt);
                    }
                }

                m_evtCaffeCancel.Reset();
            }

            if (caffe != null)
            {
                caffe.Dispose();
                m_evtCancel.Reset();
                m_bCaffeCreated = false;
            }

            m_evtThreadDone.Set();
        }

        private void log_OnWriteLine1(object sender, LogArg e)
        {
            int nTotal = 1000;
            Exception err = (e.Error) ? new Exception(e.Message) : null;
            ProgressInfo pi = new ProgressInfo((int)(nTotal * e.Progress), nTotal, e.Message, err);

            m_bwProcess.ReportProgress((int)pi.Percentage, pi);
        }

        private void FormMain_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (m_caffeRun != null)
            {
                ((IDisposable)m_caffeRun).Dispose();
                m_caffeRun = null;
                m_autoTest.Abort();
            }

            Properties.Settings.Default.GPU = getGpu();
            Properties.Settings.Default.Save();

            m_evtCancelTraining.Set();
            m_evtCancelNs.Set();
        }

        #region Server Based Autotesting

        private void startToolStripMenuItem_Click(object sender, EventArgs e)
        {
            openFileDialogAutoTests.InitialDirectory = initialDirectory;
            if (openFileDialogAutoTests.ShowDialog() == DialogResult.OK)
            {
                runAutotestsToolStripMenuItem.Enabled = false;
                startAutotestsToolStripMenuItem.Enabled = false;
                abortAutotestsToolStripMenuItem.Enabled = true;
                m_autoTest.Initialize("c:\\temp", EntitiesConnection.GlobalDatabaseServerName);
                m_autoTest.Run(openFileDialogAutoTests.FileName, false, getGpu());
            }
        }

        private void startWithResetToolStripMenuItem_Click(object sender, EventArgs e)
        {
            openFileDialogAutoTests.InitialDirectory = initialDirectory;
            if (openFileDialogAutoTests.ShowDialog() == DialogResult.OK)
            {
                if (MessageBox.Show("Resetting the test database will delete all test results for the '" + openFileDialogAutoTests.FileName + "'!  Do you want to continue?", "Delete Test Configuration", MessageBoxButtons.YesNo, MessageBoxIcon.Exclamation) != DialogResult.Yes)
                    return;

                runAutotestsToolStripMenuItem.Enabled = false;
                startAutotestsToolStripMenuItem.Enabled = false;
                abortAutotestsToolStripMenuItem.Enabled = true;
                m_autoTest.Initialize("c:\\temp", EntitiesConnection.GlobalDatabaseServerName);
                m_autoTest.Run(openFileDialogAutoTests.FileName, true, getGpu());
            }
        }

        private void abortAutotestsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            abortAutotestsToolStripMenuItem.Enabled = false;
            m_autoTest.Abort();
        }

        private void m_autoTest_OnCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
                setStatus("AutoTest ERROR: " + e.Error.Message);
            else if (e.Cancelled)
                setStatus("AutoTest ABORTED!");
            else
                setStatus("AutoTest COMPLETED!");

            startAutotestsToolStripMenuItem.Enabled = true;
            runAutotestsToolStripMenuItem.Enabled = true;
        }

        private void m_autoTest_OnProgress(object sender, ProgressChangedEventArgs e)
        {
            AutoTestProgressInfo pi = e.UserState as AutoTestProgressInfo;
            setStatus(pi.ToString());
        }

        #endregion

        private void helpToolStripMenuItem_Click(object sender, EventArgs e)
        {
        }

        private void localHelpToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Process p = new Process();
            p.StartInfo = new ProcessStartInfo("index.chm");
            p.Start();
        }

        private void onlineHelpToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(Properties.Settings.Default.OnlineHelpUrl);
            p.Start();
        }

        private void m_bwUrlCheck_DoWork(object sender, DoWorkEventArgs e)
        {
            e.Result = checkURL(Properties.Settings.Default.OnlineHelpUrl);
        }

        private void m_bwUrlCheck_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
                return;

            onlineHelpToolStripMenuItem.Enabled = (bool)e.Result;
        }

        #region Special Tests

        private void alexNetToolStripMenuItem_Click(object sender, EventArgs e)
        {
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            loadCIFAR10ToolStripMenuItem.Enabled = false;
            loadVOC2007ToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            specialTestsToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();
            m_evtCaffeCancel.Reset();

            if (!m_bwProcess.IsBusy)
                m_bwProcess.RunWorkerAsync();

            m_Cmd = COMMAND.SPECIALTEST_ALEXNETCIFAR;
            m_evtCommandRead.Set();
        }

        private void resNet56CifarAccuracyBugToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBox.Show("The ResNet56 results will be placed in the 'C:\\ProgramData\\MyCaffe\\test_data\\models\\resnet56\\cifar' directory.", "ResNet56 Results", MessageBoxButtons.OK, MessageBoxIcon.Information);

            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            loadCIFAR10ToolStripMenuItem.Enabled = false;
            loadVOC2007ToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            specialTestsToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();
            m_evtCaffeCancel.Reset();

            if (!m_bwProcess.IsBusy)
                m_bwProcess.RunWorkerAsync();

            m_Cmd = COMMAND.SPECIALTEST_RESNETCIFAR;
            m_evtCommandRead.Set();
        }

        private MyCaffeControl<float> runTest(COMMAND cmd, Log log)
        {
            try
            {
                m_bTesting = true;

                switch (cmd)
                {
                    case COMMAND.SPECIALTEST_ALEXNETCIFAR:
                        return runTest_alexnetcifar(log);

                    case COMMAND.SPECIALTEST_RESNETCIFAR:
                        return runTest_resnetcifar(log);

                    default:
                        log.WriteLine("WARNING: Unknown test command '" + cmd.ToString() + "'.");
                        return null;
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_bTesting = false;
            }
        }

        private MyCaffeControl<float> runTest_alexnetcifar(Log log)
        {
            MyCaffeControl<float> caffe = null;
            SettingsCaffe settings = new SettingsCaffe();
            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND;
            settings.EnableRandomInputSelection = true;
            settings.GpuIds = getGpu().ToString();

            log.WriteLine("Running AlexNet-Cifar test on GPU " + settings.GpuIds + "...");

            caffe = new MyCaffeControl<float>(settings, log, m_evtCaffeCancel);

            string strSolver = System.Text.Encoding.UTF8.GetString(Properties.Resources.alexnet_cifar_solver);
            string strModel = System.Text.Encoding.UTF8.GetString(Properties.Resources.alexnet_cifar_train_val);

            caffe.Load(Phase.TRAIN, strSolver, strModel, null);
            caffe.Train();

            return caffe;
        }

        private MyCaffeControl<float> runTest_resnetcifar(Log log)
        {
            MyCaffeControl<float> caffe = null;
            SettingsCaffe settings = new SettingsCaffe();
            int nGpuId = getGpu();
            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            settings.EnableRandomInputSelection = true;
            settings.GpuIds = nGpuId.ToString();

            log.WriteLine("Running ResNet56-Cifar test on GPU " + settings.GpuIds + "...");

            caffe = new MyCaffeControl<float>(settings, log, m_evtCaffeCancel);

            string strSolver = System.Text.Encoding.UTF8.GetString(Properties.Resources.resnet56_cifar_solver);
            string strModel = System.Text.Encoding.UTF8.GetString(Properties.Resources.resnet56_cifar_train_val);

            // Use the OnTestingIteration event to log the ongoing results.
            caffe.OnTestingIteration += Caffe_OnTestingIteration;

            // Use the OnTrainingIteration event to save the last error.
            caffe.OnTrainingIteration += Caffe_OnTrainingIteration;

            // Load the model.
            caffe.Load(Phase.TRAIN, strSolver, strModel, null);

            // Get current mode used TCC or WDM
            string strInfo = caffe.Cuda.GetDeviceP2PInfo(nGpuId);
            if (strInfo.Contains("TCC Driver = YES"))
                m_mode = MODE.TCC;
            else if (strInfo.Contains("TCC Driver = NO"))
                m_mode = MODE.WDM;
            else
                m_mode = MODE.UNKNOWN;

            // Start training
            caffe.Train();

            if (m_swResNetTest != null)
            {
                if (!string.IsNullOrEmpty(m_strTestLogDir))
                    log.WriteLine("The ResNet test logs are in the directory '" + m_strTestLogDir + "'");

                m_swResNetTest.Close();
                m_swResNetTest.Dispose();
                m_swResNetTest = null;
            }

            return caffe;
        }

        private void Caffe_OnTrainingIteration(object sender, TrainingIterationArgs<float> e)
        {
            m_nLastTrainingIteration = e.Iteration;
            m_dfLastLoss = e.SmoothedLoss;
            m_dfTotalTiming += e.Timing;
            m_nTimingCount++;

            if (m_swGlobalTiming == null)
            {
                m_swGlobalTiming = new Stopwatch();
                m_swGlobalTiming.Start();
            }
        }

        private void Caffe_OnTestingIteration(object sender, TestingIterationArgs<float> e)
        {
            if (m_swResNetTest == null)
            {
                string strLog = GetTestPath("\\MyCaffe\\test_data\\models\\resnet56\\cifar", true, true, false);
                m_strTestLogDir = strLog;
                strLog += "\\resnet56_cifar_" + m_mode.ToString() + "_log.csv";

                if (File.Exists(strLog))
                    File.Delete(strLog);

                m_swResNetTest = new StreamWriter(strLog);
                m_swResNetTest.WriteLine("Iteration, Loss, Accuracy, Ave Timing (ms), Global Ave Timing (ms)");
                m_swResNetTest.Flush();
            }

            double dfAveTiming = (m_nTimingCount == 0) ? 0 : m_dfTotalTiming / m_nTimingCount;
            double dfGlobalTiming = (m_swGlobalTiming == null) ? 0 : m_swGlobalTiming.Elapsed.TotalMilliseconds / m_nTimingCount;
            m_swResNetTest.WriteLine(m_nLastTrainingIteration.ToString() + "," + m_dfLastLoss.ToString() + "," + e.Accuracy.ToString() + "," + dfAveTiming.ToString("N2") + "," + dfGlobalTiming.ToString("N2"));
            m_swResNetTest.Flush();
            m_dfTotalTiming = 0;
            m_nTimingCount = 0;
            m_swGlobalTiming = null;
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

            strPath = ExecutingAssemblyPath;
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

        #endregion

        private void FormMain_Resize(object sender, EventArgs e)
        {
            lvStatus.Columns[0].Width = Width - 24;
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            lblGpu.Text = "Using GPU " + getGpuName();
        }

        private void startCartPoleTrainerToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (startCartPoleTrainerToolStripMenuItem.Text.Contains("Start"))
            {
                bool bShowUi = false;
                bool bUseAccelTrain = false;
                bool bAllowDiscountReset = false;
                string strTrainer = "SIMPLE";
                int nIterations = 500000;
                int nMiniBatch = 1;

                FormCustomTraining dlg = new FormCustomTraining("Cart-Pole");
                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                nIterations = dlg.Iterations;
                nMiniBatch = dlg.MiniBatch;
                bShowUi = dlg.ShowUserInterface;
                bUseAccelTrain = dlg.UseAcceleratedTraining;
                bAllowDiscountReset = dlg.AllowDiscountReset;
                strTrainer = dlg.Trainer;

                Text = "MyCaffe - Cart-Pole " + strTrainer + " [MiniBatch = " + nMiniBatch.ToString() + ", UseAccelTrain=" + bUseAccelTrain.ToString() + ", DiscountReset=" + bAllowDiscountReset.ToString() + "]";

                m_log.WriteLine("starting policy gradient cart-pole test...");
                m_evtCancelTraining.Reset();
                m_trainerTask = Task.Factory.StartNew(new Action<object>(trainerThread), new Settings(m_evtCancelTraining, getGpu(), "Cart-Pole", strTrainer, nIterations, nMiniBatch, bShowUi, bUseAccelTrain, bAllowDiscountReset, false, false, false, 0, 0));
                startAtariTrainerToolStripMenuItem.Enabled = false;
                startNeuralStyleTransferToolStripMenuItem.Enabled = false;
                startCartPoleTrainerToolStripMenuItem.Text = "Stop Cart-Pole Training";
            }
            else
            {
                Text = "MyCaffe";

                m_log.WriteLine("stopping policy gradient cart-pole test...");
                m_evtCancelTraining.Set();                
                m_trainerTask = null;
                startAtariTrainerToolStripMenuItem.Enabled = true;
                startNeuralStyleTransferToolStripMenuItem.Enabled = true;
                startCartPoleTrainerToolStripMenuItem.Text = "Start Cart-Pole Training";
            }
        }

        private void startAtariTrainerToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (startAtariTrainerToolStripMenuItem.Text.Contains("Start"))
            {
                bool bShowUi = false;
                bool bUseAccelTrain = false;
                bool bAllowDiscountReset = false;
                bool bAllowNegativeRewards = false;
                bool bTerminateOnRallyEnd = false;
                bool bLoadWeights = false;
                double dfVMin = -10;
                double dfVMax = 10;
                string strTrainer = "SIMPLE";
                int nIterations = 500000;
                int nMiniBatch = 1;

                FormCustomTraining dlg = new FormCustomTraining("ATARI");
                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                m_strAtariRom = dlg.RomName;
                nIterations = dlg.Iterations;
                nMiniBatch = dlg.MiniBatch;
                bShowUi = dlg.ShowUserInterface;
                bUseAccelTrain = dlg.UseAcceleratedTraining;
                bAllowDiscountReset = dlg.AllowDiscountReset;
                bAllowNegativeRewards = dlg.AllowNegativeRewards;
                bTerminateOnRallyEnd = dlg.TerminateOnRallyEnd;
                bLoadWeights = dlg.LoadWeights;
                strTrainer = dlg.Trainer;
                dfVMin = dlg.VMin;
                dfVMax = dlg.VMax;

                string strVMinMax = "";

                if (strTrainer.Contains("C51"))
                    strVMinMax = ", VMin=" + dfVMin.ToString("N1") + " VMax=" + dfVMax.ToString("N1");

                Text = "MyCaffe - ATARI(" + m_strAtariRom + ") " + strTrainer + " [MiniBatch = " + nMiniBatch.ToString() + ", UseAccelTrain=" + bUseAccelTrain.ToString() + ", DiscountReset=" + bAllowDiscountReset.ToString() + strVMinMax + "]";

                m_log.WriteLine("starting " + strTrainer + " ATARI (" + m_strAtariRom + ") test...");
                m_evtCancelTraining.Reset();
                m_trainerTask = Task.Factory.StartNew(new Action<object>(trainerThread), new Settings(m_evtCancelTraining, getGpu(), "ATARI", strTrainer, nIterations, nMiniBatch, bShowUi, bUseAccelTrain, bAllowDiscountReset, bAllowNegativeRewards, bTerminateOnRallyEnd, bLoadWeights, dfVMin, dfVMax));
                startCartPoleTrainerToolStripMenuItem.Enabled = false;
                startNeuralStyleTransferToolStripMenuItem.Enabled = false;
                startAtariTrainerToolStripMenuItem.Text = "Stop ATARI Training";
            }
            else
            {
                Text = "MyCaffe";

                m_log.WriteLine("stopping ATARI test...");
                m_evtCancelTraining.Set();
                m_trainerTask = null;
                startCartPoleTrainerToolStripMenuItem.Enabled = true;
                startNeuralStyleTransferToolStripMenuItem.Enabled = true;
                startAtariTrainerToolStripMenuItem.Text = "Start ATARI Training";
            }
        }

        private void trainerThread(object obj)
        {
            Settings arg = obj as Settings;
            CancelEvent evtCancel = arg.Cancel;
            string strGym = arg.Gym;
            MyCaffeCustomTrainerTest<float> test = new MyCaffeCustomTrainerTest<float>(strGym, arg.Gpu, EngineParameter.Engine.DEFAULT);
            int nIterations = arg.Iterations;
            int nMiniBatch = arg.MiniBatch;
            bool bShowUi = arg.ShowUi;
            bool bUseAccelTrain = arg.UseAcceleratedTraining;
            bool bAllowDiscountReset = arg.AllowDiscountReset;
            string strTrainer = arg.Trainer;

            test.Log.OnWriteLine += Log_OnWriteLine1;
            test.CancelEvent.AddCancelOverride(evtCancel);

            try
            {
                if (strGym == "Cart-Pole")
                {
                    if (strTrainer.Contains("DQN"))
                        test.TrainCartPoleDqmDual(bShowUi, strTrainer, nIterations, nMiniBatch, bUseAccelTrain, bAllowDiscountReset);
                    else
                        test.TrainCartPolePGDual(bShowUi, strTrainer, nIterations, nMiniBatch, bUseAccelTrain, bAllowDiscountReset);
                }
                else if (strGym == "ATARI")
                {
                    if (strTrainer.Contains("DQN"))
                        test.TrainAtariDqnDual(bShowUi, strTrainer, nIterations, nMiniBatch, bUseAccelTrain, 1, m_strAtariRom, arg.AllowNegativeRewards, arg.TerminateOnRallyEnd, arg.LoadWeights);
                    else if (strTrainer.Contains("C51"))
                        test.TrainAtariC51Dual(bShowUi, strTrainer, nIterations, nMiniBatch, bUseAccelTrain, 1, m_strAtariRom, arg.AllowNegativeRewards, arg.TerminateOnRallyEnd, arg.LoadWeights, arg.VMin, arg.VMax);
                    else
                        test.TrainAtariPGDual(bShowUi, strTrainer, nIterations, nMiniBatch, bUseAccelTrain, bAllowDiscountReset, m_strAtariRom, arg.AllowNegativeRewards, arg.TerminateOnRallyEnd);
                }

                if (evtCancel.WaitOne(0))
                    test.Log.WriteLine("training aborted.");
                else
                    test.Log.WriteLine("training done.");
            }
            catch (Exception excpt)
            {
                test.Log.WriteError(excpt);
            }

            test.Log.OnWriteLine -= Log_OnWriteLine1;
            test.Dispose();
        }

        private void Log_OnWriteLine1(object sender, LogArg e)
        {
            if (e.Error)
                m_log.WriteError(new Exception(e.Message));
            else
                m_log.WriteLine(e.Message);
        }

        private void startNeuralStyleTransferToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (startNeuralStyleTransferToolStripMenuItem.Text.Contains("Start"))
            {
                string strStyleFile = Properties.Settings.Default.NsStyleImgFile;
                string strContentFile = Properties.Settings.Default.NsContentImgFile;
                string strModelName = Properties.Settings.Default.NsModelName;
                string strSolverType = Properties.Settings.Default.NsSolverType;
                int nIterations = Properties.Settings.Default.NsIterations;
                double dfLr = Properties.Settings.Default.NsLearningRate;
                string strResultPath = Properties.Settings.Default.NsResultPath;
                int nIntermediateIterations = Properties.Settings.Default.NsIntermediateIterations;
                double dfTvLoss = Properties.Settings.Default.NsTVLoss;
                int nMaxImageSize = Properties.Settings.Default.NsMaxImageSize;

                if (string.IsNullOrEmpty(strResultPath))
                {
                    string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
                    strPath += "\\MyCaffe\\test_data\\data\\images\\result";

                    if (!Directory.Exists(strPath))
                        Directory.CreateDirectory(strPath);

                    strResultPath = strPath;
                }

                if (string.IsNullOrEmpty(strStyleFile))
                {
                    strStyleFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
                    strStyleFile += "\\MyCaffe\\test_data\\data\\images\\style\\starry_night.jpg";
                }

                if (string.IsNullOrEmpty(strContentFile))
                {
                    strContentFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
                    strContentFile += "\\MyCaffe\\test_data\\data\\images\\content\\sanfrancisco.jpg";
                }

                FormNeuralStyle dlg = new FormNeuralStyle(strStyleFile, strContentFile, nIterations, strModelName, strSolverType, dfLr, strResultPath, nIntermediateIterations, dfTvLoss, nMaxImageSize);

                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                Properties.Settings.Default.NsStyleImgFile = dlg.Info.StyleImageFile;
                Properties.Settings.Default.NsContentImgFile = dlg.Info.ContentImageFile;
                Properties.Settings.Default.NsModelName = dlg.Info.ModelName;
                Properties.Settings.Default.NsSolverType = dlg.Info.SolverType;
                Properties.Settings.Default.NsIterations = dlg.Info.Iterations;
                Properties.Settings.Default.NsLearningRate = dlg.Info.LearningRate;
                Properties.Settings.Default.NsResultPath = dlg.Info.ResultPath;
                Properties.Settings.Default.NsIntermediateIterations = dlg.Info.IntermediateIterations;
                Properties.Settings.Default.NsTVLoss = dlg.Info.TVLoss;
                Properties.Settings.Default.NsMaxImageSize = dlg.Info.MaxImageSize;
                Properties.Settings.Default.Save();

                m_log.WriteLine("starting neural style transfer...");
                m_log.WriteLine("Results to be placed in: " + strResultPath);
                m_evtCancelNs.Reset();
                m_nsTask = Task.Factory.StartNew(new Action<object>(nsThread), new Tuple<CancelEvent, NeuralStyleInfo>(m_evtCancelNs, dlg.Info));
                startCartPoleTrainerToolStripMenuItem.Enabled = false;
                startAtariTrainerToolStripMenuItem.Enabled = false;
                startNeuralStyleTransferToolStripMenuItem.Text = "Stop Neural Style Transfer";
            }
            else
            {
                m_log.WriteLine("stopping neural style transfer...");
                m_evtCancelNs.Set();
                startCartPoleTrainerToolStripMenuItem.Enabled = true;
                startAtariTrainerToolStripMenuItem.Enabled = true;
                startNeuralStyleTransferToolStripMenuItem.Text = "Start Neural Style Transfer";
            }
        }

        private void nsDone()
        {
            startCartPoleTrainerToolStripMenuItem.Enabled = true;
            startAtariTrainerToolStripMenuItem.Enabled = true;
            startNeuralStyleTransferToolStripMenuItem.Text = "Start Neural Style Transfer";

            if (File.Exists(m_strNsResults))
            {
                Process p = new Process();
                p.StartInfo = new ProcessStartInfo(m_strNsResults);
                p.Start();
            }
        }

        private void nsThread(object obj)
        {
            Tuple<CancelEvent, NeuralStyleInfo> arg = obj as Tuple<CancelEvent, NeuralStyleInfo>;
            CancelEvent evtCancel = arg.Item1;
            NeuralStyleInfo info = arg.Item2;
            NeuralStyleTransferTest<float> test = new NeuralStyleTransferTest<float>("Neural Style Test", getGpu(), EngineParameter.Engine.DEFAULT);

            test.Log.OnWriteLine += Log_OnWriteLine1;
            test.CancelEvent.AddCancelOverride(evtCancel);
            m_strNsResults = test.TestNeuralStyleTransfer(info.StyleImageFile, info.ContentImageFile, info.Iterations, info.IntermediateIterations, info.ResultPath, info.ModelName, info.SolverType, info.LearningRate, info.TVLoss, info.MaxImageSize);

            if (evtCancel.WaitOne(0))
                test.Log.WriteLine("training aborted.");
            else
                test.Log.WriteLine("training done.");

            test.Log.OnWriteLine -= log_OnWriteLine1;
            test.Dispose();

            this.Invoke(new fnNsDone(nsDone));
        }

        private void showGymUiToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                EventWaitHandle evtOpenUi = EventWaitHandle.OpenExisting("_MyCaffeTrainer_OpenUi_");
                if (evtOpenUi != null)
                    evtOpenUi.Set();
            }
            catch (Exception excpt)
            {
            }
        }
    }

    class Settings
    {
        CancelEvent m_evtCancel;
        string m_strTrainer;
        string m_strGym;
        bool m_bShowUi;
        bool m_bUseAccelTrain;
        bool m_bAllowDiscountReset;
        bool m_bAllowNegativeRewards;
        bool m_bTerminateOnRallyEnd;
        bool m_bLoadWeights;
        int m_nIterations;
        int m_nMiniBatch;
        double m_dfVMin;
        double m_dfVMax;
        int m_nGPU;

        public Settings(CancelEvent evtCancel, int nGpu, string strGym, string strTrainer, int nIterations, int nMiniBatch, bool bShowUi, bool bUseAccelTrain, bool bAllowDiscountReset, bool bAllowNegRewards, bool bTerminateOnRallyEnd, bool bLoadWeights, double dfVMin, double dfVMax)
        {
            m_evtCancel = evtCancel;
            m_nGPU = nGpu;
            m_strGym = strGym;
            m_strTrainer = strTrainer;
            m_nIterations = nIterations;
            m_nMiniBatch = nMiniBatch;
            m_bShowUi = bShowUi;
            m_bUseAccelTrain = bUseAccelTrain;
            m_bAllowDiscountReset = bAllowDiscountReset;
            m_bAllowNegativeRewards = bAllowNegRewards;
            m_bTerminateOnRallyEnd = bTerminateOnRallyEnd;
            m_bLoadWeights = bLoadWeights;
            m_dfVMin = dfVMin;
            m_dfVMax = dfVMax;
        }

        public CancelEvent Cancel
        {
            get { return m_evtCancel; }
        }

        public int Gpu
        {
            get { return m_nGPU; }
        }

        public string Gym
        {
            get { return m_strGym; }
        }

        public string Trainer
        {
            get { return m_strTrainer; }
        }

        public int Iterations
        {
            get { return m_nIterations; }
        }

        public int MiniBatch
        {
            get { return m_nMiniBatch; }
        }

        public bool ShowUi
        {
            get { return m_bShowUi; }
        }

        public bool UseAcceleratedTraining
        {
            get { return m_bUseAccelTrain; }
        }

        public bool AllowDiscountReset
        {
            get { return m_bAllowDiscountReset; }
        }

        public bool AllowNegativeRewards
        {
            get { return m_bAllowNegativeRewards; }
        }

        public bool TerminateOnRallyEnd
        {
            get { return m_bTerminateOnRallyEnd; }
        }

        public bool LoadWeights
        {
            get { return m_bLoadWeights; }
        }

        public double VMin
        {
            get { return m_dfVMin; }
        }

        public double VMax
        {
            get { return m_dfVMax; }
        }
    }
}
