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
        Task m_pgTask = null;
        CancelEvent m_evtCancelPG = new CancelEvent();
        MyCaffeGymUiServiceHost m_gymHost;

        delegate void fnSetStatus(string strMsg, STATUS status, bool bBreath);

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
                            menu.Checked = true;

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
        }

        private int getGpu()
        {
            foreach (ToolStripMenuItem menu in gpuToolStripMenuItem.DropDownItems)
            {
                if (menu.Checked)
                    return (int)menu.Tag;
            }

            return 0;
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
                trainMNISTToolStripMenuItem.Enabled = false;
                testMNISTToolStripMenuItem.Enabled = false;
                createMyCaffeToolStripMenuItem.Enabled = false;
                destroyMyCaffeToolStripMenuItem.Enabled = false;
                deviceInformationToolStripMenuItem.Enabled = false;
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
                trainMNISTToolStripMenuItem.Enabled = false;
                testMNISTToolStripMenuItem.Enabled = false;
                createMyCaffeToolStripMenuItem.Enabled = false;
                destroyMyCaffeToolStripMenuItem.Enabled = false;
                deviceInformationToolStripMenuItem.Enabled = false;
                m_bwLoadCiFar10Database.RunWorkerAsync(dlg.Parameters);
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
            cancelToolStripMenuItem.Enabled = false;
        }

        private void m_bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            ProgressInfo pi = e.UserState as ProgressInfo;

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

        private void loader_OnCompleted(object sender, EventArgs e)
        {
            m_bLoading = false;
        }

        private void loader_OnError(object sender, ProgressArgs e)
        {
            if (sender.GetType() == typeof(MnistDataLoader))
                m_bwLoadMnistDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else
                m_bwLoadCiFar10Database.ReportProgress((int)e.Progress.Percentage, e.Progress);

            m_bLoading = false;
        }

        private void loader_OnProgress(object sender, ProgressArgs e)
        {
            if (sender.GetType() == typeof(MnistDataLoader))
                m_bwLoadMnistDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else
                m_bwLoadCiFar10Database.ReportProgress((int)e.Progress.Percentage, e.Progress);
        }

        private void createMyCaffeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            m_bCaffeCreated = true;
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
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
                                string str1 = caffe.GetDeviceName(0);
                                str1 += Environment.NewLine;
                                str1 += caffe.Cuda.GetDeviceInfo(0, true);
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

            m_evtCancelPG.Set();
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

                FormCustomTraining dlg = new FormCustomTraining("Cart-Pole");
                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                bShowUi = dlg.ShowUserInterface;
                bUseAccelTrain = dlg.UseAcceleratedTraining;
                bAllowDiscountReset = dlg.AllowDiscountReset;
                strTrainer = dlg.Trainer;

                m_log.WriteLine("starting policy gradient cart-pole test...");
                m_evtCancelPG.Reset();
                m_pgTask = Task.Factory.StartNew(new Action<object>(pgTrainerThread), new Tuple<CancelEvent, string, bool, bool, bool, string>(m_evtCancelPG, "Cart-Pole", bShowUi, bUseAccelTrain, bAllowDiscountReset, strTrainer));
                startAtariTrainerToolStripMenuItem.Enabled = false;
                startCartPoleTrainerToolStripMenuItem.Text = "Stop Cart-Pole Training";
            }
            else
            {
                m_log.WriteLine("stopping policy gradient cart-pole test...");
                m_evtCancelPG.Set();                
                m_pgTask = null;
                startAtariTrainerToolStripMenuItem.Enabled = true;
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
                string strTrainer = "SIMPLE";

                FormCustomTraining dlg = new FormCustomTraining("ATARI");
                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                bShowUi = dlg.ShowUserInterface;
                bUseAccelTrain = dlg.UseAcceleratedTraining;
                bAllowDiscountReset = dlg.AllowDiscountReset;
                strTrainer = dlg.Trainer;

                m_log.WriteLine("starting policy gradient ATARI test...");
                m_evtCancelPG.Reset();
                m_pgTask = Task.Factory.StartNew(new Action<object>(pgTrainerThread), new Tuple<CancelEvent, string, bool, bool, bool, string>(m_evtCancelPG, "ATARI", bShowUi, bUseAccelTrain, bAllowDiscountReset, strTrainer));
                startCartPoleTrainerToolStripMenuItem.Enabled = false;
                startAtariTrainerToolStripMenuItem.Text = "Stop ATARI Training";
            }
            else
            {
                m_log.WriteLine("stopping policy gradient ATARI test...");
                m_evtCancelPG.Set();
                m_pgTask = null;
                startCartPoleTrainerToolStripMenuItem.Enabled = true;
                startAtariTrainerToolStripMenuItem.Text = "Start ATARI Training";
            }
        }

        private void pgTrainerThread(object obj)
        {
            Tuple<CancelEvent, string, bool, bool, bool, string> arg = obj as Tuple<CancelEvent, string, bool, bool, bool, string>;
            CancelEvent evtCancel = arg.Item1;
            string strGym = arg.Item2;
            MyCaffeCustomTrainerTest<float> test = new MyCaffeCustomTrainerTest<float>(strGym, 0, EngineParameter.Engine.DEFAULT);
            int nIterations = 500000;
            bool bShowUi = arg.Item3;
            bool bUseAccelTrain = arg.Item4;
            bool bAllowDiscountReset = arg.Item5;
            string strTrainer = arg.Item6;

            test.Log.OnWriteLine += Log_OnWriteLine1;
            test.CancelEvent.AddCancelOverride(evtCancel);

            if (strGym == "Cart-Pole")
                test.TrainCartPolePG(bShowUi, "PG." + strTrainer, nIterations, bUseAccelTrain, bAllowDiscountReset);
            else if (strGym == "ATARI")
                test.TrainAtariPG(bShowUi, "PG." + strTrainer, nIterations, bUseAccelTrain, bAllowDiscountReset);

            if (evtCancel.WaitOne(0))
                test.Log.WriteLine("training aborted.");
            else
                test.Log.WriteLine("training done.");

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
    }
}
