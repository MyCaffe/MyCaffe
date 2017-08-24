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

namespace MyCaffe.app
{
    public partial class FormMain : Form
    {
        CancelEvent m_evtCancel = new CancelEvent();
        AutoResetEvent m_evtCommandRead = new AutoResetEvent(false);
        AutoResetEvent m_evtThreadDone = new AutoResetEvent(false);
        IXMyCaffeNoDb<float> m_caffeRun;
        COMMAND m_Cmd = COMMAND.NONE;
        byte[] m_rgTrainedWeights = null;
        SimpleDatum m_sdImageMean = null;
        AutomatedTesterServer m_autoTest = new AutomatedTesterServer();

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
            m_bwProcess.RunWorkerAsync();

            m_autoTest.OnProgress += M_autoTest_OnProgress;
            m_autoTest.OnCompleted += M_autoTest_OnCompleted;

            setStatus("The MyCaffe Test App supports two different types of automated testing:");
            setStatus(" 1.) User interface based automated testing via the 'Test | Run Autotests UI', and");
            setStatus(" 2.) Server based automated testing via the 'Test | Start Server Autotests' menu.");
            setStatus("Server auto tests can easily integrate into other applications.");
            setStatus("----------------------------------------------------------------------------------");
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
            TestDatabaseManager dbMgr = new TestDatabaseManager();
            bool bExists;
            Exception err = dbMgr.DatabaseExists(out bExists);

            if (err != null)
            {
                setStatus("ERROR Querying Testing Database! " + err.Message);
                return;
            }

            if (!bExists)
            {
                FormCreateDatabase dlg = new FormCreateDatabase(dbMgr.DatabaseName, "You must create the 'Testing' Database first");

                if (dlg.ShowDialog() != DialogResult.OK)
                    return;

                err = dbMgr.CreateDatabase(dlg.DatabasePath);
                if (err != null)
                {
                    setStatus("ERROR Creating Testing Database! " + err.Message);
                    return;
                }

                setStatus("Testing database created.");
            }

            if (openFileDialogAutoTests.ShowDialog() == DialogResult.OK)
            {
                FormAutomatedTests dlg = new FormAutomatedTests(openFileDialogAutoTests.FileName);

                setStatus("Running automatic tests.");
                dlg.ShowDialog();
            }
        }

        private void setStatus(string str)
        {
            int nMaxLines = 2000;

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

            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            createMyCaffeToolStripMenuItem.Enabled = true;
            loadMNISTToolStripMenuItem.Enabled = true;
            loadCIFAR10ToolStripMenuItem.Enabled = true;
        }

        private void m_bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            ProgressInfo pi = e.UserState as ProgressInfo;

            if (pi.Alive.HasValue)
            {
                if (pi.Alive.Value == true)
                {
                    loadMNISTToolStripMenuItem.Enabled = true;
                    loadCIFAR10ToolStripMenuItem.Enabled = true;
                    destroyMyCaffeToolStripMenuItem.Enabled = true;
                    trainMNISTToolStripMenuItem.Enabled = true;
                    testMNISTToolStripMenuItem.Enabled = true;
                    deviceInformationToolStripMenuItem.Enabled = true;
                }
                else
                {
                    loadMNISTToolStripMenuItem.Enabled = true;
                    loadCIFAR10ToolStripMenuItem.Enabled = true;
                    createMyCaffeToolStripMenuItem.Enabled = true;
                }

                setStatus(pi.Message);

                if (pi.Message == "MyCaffe Traning completed.")
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

        private void m_bwLoadDatabase_DoWork(object sender, DoWorkEventArgs e)
        {
            MnistDataLoader loader = new app.MnistDataLoader(e.Argument as MnistDataParameters);

            loader.OnProgress += loader_OnProgress;
            loader.OnError += loader_OnError;

            loader.LoadDatabase();            
        }

        private void m_bwLoadCiFar10Database_DoWork(object sender, DoWorkEventArgs e)
        {
            CiFar10DataLoader loader = new CiFar10DataLoader(e.Argument as CiFar10DataParameters);

            loader.OnProgress += loader_OnProgress;
            loader.OnError += loader_OnError;

            loader.LoadDatabase();
        }

        private void loader_OnError(object sender, ProgressArgs e)
        {
            if (sender.GetType() == typeof(MnistDataLoader))
                m_bwLoadMnistDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
            else
                m_bwLoadCiFar10Database.ReportProgress((int)e.Progress.Percentage, e.Progress);
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
            createMyCaffeToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            loadMNISTToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            abortToolStripMenuItem.Enabled = true;
            m_evtCancel.Reset();

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
            m_Cmd = COMMAND.TRAIN;
            m_evtCommandRead.Set();
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
            m_Cmd = COMMAND.TEST;
            m_evtCommandRead.Set();
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
            m_evtCancel.Set();
        }

        private void m_bwProcess_DoWork(object sender, DoWorkEventArgs e)
        {
            BackgroundWorker bw = sender as BackgroundWorker;
            Log log = new Log("MyCaffe");
            MyCaffeControl<float> caffe = null;

            while (!m_evtCancel.WaitOne(0))
            {
                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.AddRange(m_evtCancel.Handles);
                rgWait.Add(m_evtCommandRead);

                int nWait = WaitHandle.WaitAny(rgWait.ToArray());

                if (nWait > 0)
                {
                    switch (m_Cmd)
                    {
                        case COMMAND.CREATE:
                            SettingsCaffe settings = new SettingsCaffe();
                            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
                            settings.EnableRandomInputSelection = true;

                            log = new Log("MyCaffe");
                            log.OnWriteLine += log_OnWriteLine1;

                            caffe = new MyCaffeControl<float>(settings, log, m_evtCancel);

                            string strSolver = System.Text.Encoding.UTF8.GetString(Properties.Resources.lenet_solver);
                            string strModel = System.Text.Encoding.UTF8.GetString(Properties.Resources.lenet_train_test);

                            caffe.Load(Phase.TRAIN, strSolver, strModel, null);
                            bw.ReportProgress(1, new ProgressInfo(1, 1, "MyCaffe Created.", null, true));
                            break;

                        case COMMAND.DESTROY:
                            caffe.Dispose();
                            caffe = null;
                            log = null;
                            bw.ReportProgress(0, new ProgressInfo(0, 0, "MyCaffe Destroyed", null, false));
                            m_evtCancel.Reset();
                            break;

                        case COMMAND.TRAIN:
                            caffe.Train(5000);
                            m_rgTrainedWeights = caffe.GetWeights();
                            m_sdImageMean = caffe.GetImageMean();
                            bw.ReportProgress(0, new ProgressInfo(0, 0, "MyCaffe Traning completed.", null, true));
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
            }

            if (caffe != null)
            {
                caffe.Dispose();
                m_evtCancel.Reset();
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
            if (openFileDialogAutoTests.ShowDialog() == DialogResult.OK)
            {
                runAutotestsToolStripMenuItem.Enabled = false;
                startAutotestsToolStripMenuItem.Enabled = false;
                abortAutotestsToolStripMenuItem.Enabled = true;
                m_autoTest.Initialize("c:\\temp");
                m_autoTest.Run(openFileDialogAutoTests.FileName, false);
            }
        }

        private void startWithResetToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (openFileDialogAutoTests.ShowDialog() == DialogResult.OK)
            {
                if (MessageBox.Show("Resetting the test database will delete all test results for the '" + openFileDialogAutoTests.FileName + "'!  Do you want to continue?", "Delete Test Configuration", MessageBoxButtons.YesNo, MessageBoxIcon.Exclamation) != DialogResult.Yes)
                    return;

                runAutotestsToolStripMenuItem.Enabled = false;
                startAutotestsToolStripMenuItem.Enabled = false;
                abortAutotestsToolStripMenuItem.Enabled = true;
                m_autoTest.Initialize("c:\\temp");
                m_autoTest.Run(openFileDialogAutoTests.FileName, true);
            }
        }

        private void abortAutotestsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            abortAutotestsToolStripMenuItem.Enabled = false;
            m_autoTest.Abort();
        }

        private void M_autoTest_OnCompleted(object sender, RunWorkerCompletedEventArgs e)
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

        private void M_autoTest_OnProgress(object sender, ProgressChangedEventArgs e)
        {
            AutoTestProgressInfo pi = e.UserState as AutoTestProgressInfo;
            setStatus(pi.ToString());
        }

        #endregion
    }
}
