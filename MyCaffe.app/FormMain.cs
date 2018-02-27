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
using MyCaffe.imagedb;
using MyCaffe.test.automated;
using MyCaffe.common;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using MyCaffe.param;
using System.IO;
using System.Net;

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
        bool m_bCaffeCreated = false;
        FormWait m_dlgWait = null;

        enum COMMAND
        {
            NONE,
            CREATE,
            DESTROY,
            TRAIN,
            TEST,
            DEVICEINFO
        }

        public FormMain()
        {
            InitializeComponent();

            Log log = new Log("Test Run");
            log.OnWriteLine += Log_OnWriteLine;
            m_caffeRun = new MyCaffeControl<float>(new SettingsCaffe(), log, m_evtCancel);
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
                    setStatus("You must download and install 'Microsoft SQL' or 'Microsoft SQL Express' first!");
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
                        setStatus("You are NOT connected to SQL.");
                    }
                }

                setStatus("Using SQL Instance '" + EntitiesConnection.GlobalDatabaseServerName + "'", false);

                DatabaseManagement dbMgr = new DatabaseManagement("DNN", "", EntitiesConnection.GlobalDatabaseServerName);
                bool bExists;
                Exception err = dbMgr.DatabaseExists(out bExists);

                if (err != null)
                    setStatus("ERROR: " + err.Message);
                else if (!bExists)
                    createDatabaseToolStripMenuItem_Click(this, new EventArgs());
                else
                    setStatus("Using database '" + dbMgr.Name + "'");

                setStatus("");

                m_autoTest.OnProgress += m_autoTest_OnProgress;
                m_autoTest.OnCompleted += m_autoTest_OnCompleted;

                setStatus("The MyCaffe Test App supports two different types of automated testing:");
                setStatus(" 1.) User interface based automated testing via the 'Test | Run Autotests UI', and");
                setStatus(" 2.) Server based automated testing via the 'Test | Start Server Autotests' menu.");
                setStatus("Server auto tests can easily integrate into other applications.");
                setStatus("NOTE: Known test failures are pre-set with a FAILURE status.");
                setStatus("----------------------------------------------------------------------------------");

                DatasetFactory factory = new DatasetFactory();
                int nCifarID = factory.GetDatasetID("CIFAR-10");
                int nMnistID = factory.GetDatasetID("MNIST");

                if (nCifarID == 0 || nMnistID == 0)
                {
                    setStatus(" !Before running any automated tests, make sure to load the following datasets:");

                    if (nCifarID == 0)
                        setStatus("    CIFAR-10");

                    if (nMnistID == 0)
                        setStatus("    MNIST");

                    setStatus(" see the 'Database' menu.");
                }

                m_dlgWait = new FormWait();
                m_bwInit.RunWorkerAsync();
                m_dlgWait.ShowDialog();
                m_bwUrlCheck.RunWorkerAsync();
            }
            catch (Exception excpt)
            {
                string strErr = excpt.Message;

                if (excpt.InnerException != null)
                    strErr += " " + excpt.InnerException.Message;

                if (strErr.Contains("login") && strErr.Contains("DNN"))
                    strErr += " Make sure that this user can access the DNN database - this setting is made using the SQL Management Studio.";

                setStatus("ERROR: " + strErr);
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

                if (rgstrGpu != null)
                {
                    for (int i = 0; i < rgstrGpu.Count; i++)
                    {
                        string strDevice = rgstrGpu[i];
                        ToolStripMenuItem menu = new ToolStripMenuItem(strDevice);
                        menu.Tag = i;
                        menu.Click += menuGpu_Click;

                        if (i == 0)
                            menu.Checked = true;

                        gPUToolStripMenuItem.DropDownItems.Add(menu);
                    }
                }
            }
        }

        private void menuGpu_Click(object sender, EventArgs e)
        {
            ToolStripMenuItem menu = sender as ToolStripMenuItem;

            foreach (ToolStripMenuItem item in gPUToolStripMenuItem.DropDownItems)
            {
                item.Checked = false;
            }

            menu.Checked = true;
        }

        private int getGpu()
        {
            foreach (ToolStripMenuItem menu in gPUToolStripMenuItem.DropDownItems)
            {
                if (menu.Checked)
                    return (int)menu.Tag;
            }

            return 0;
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
            catch (Exception e)
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

        private void setStatus(string str, bool bNewLine = true)
        {
            int nMaxLines = 2000;

            if (bNewLine)
                edtStatus.Text += Environment.NewLine;

            edtStatus.Text += str;

            if (edtStatus.Lines.Length > nMaxLines)
            {
                List<string> rgstr = new List<string>(edtStatus.Lines);

                while (rgstr.Count > nMaxLines)
                {
                    rgstr.RemoveAt(0);
                }

                edtStatus.Lines = rgstr.ToArray();
            }

            edtStatus.SelectionLength = 0;
            edtStatus.SelectionStart = edtStatus.Text.Length;
            edtStatus.ScrollToCaret();
        }

        private void Log_OnWriteLine(object sender, LogArg e)
        {
            setStatus(e.Message);
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
                setStatus("ERROR: " + e.Error.Message);
                runTestImageToolStripMenuItem.Enabled = false;
            }
            else if (e.Cancelled)
            {
                setStatus("ABORTED!");
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
                    createMyCaffeToolStripMenuItem.Enabled = !m_bLoading && !m_bCaffeCreated;
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
    }
}
