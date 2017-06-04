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

namespace MyCaffe.app
{
    public partial class FormMain : Form
    {
        CancelEvent m_evtCancel = new CancelEvent();
        AutoResetEvent m_evtCommandRead = new AutoResetEvent(false);
        AutoResetEvent m_evtThreadDone = new AutoResetEvent(false);
        COMMAND m_Cmd = COMMAND.NONE;

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
        }

        private void FormMain_Load(object sender, EventArgs e)
        {
            m_bwProcess.RunWorkerAsync();
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
            edtStatus.Text += Environment.NewLine;
            edtStatus.Text += str;
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
                trainMNISTToolStripMenuItem.Enabled = false;
                testMNISTToolStripMenuItem.Enabled = false;
                createMyCaffeToolStripMenuItem.Enabled = false;
                destroyMyCaffeToolStripMenuItem.Enabled = false;
                deviceInformationToolStripMenuItem.Enabled = false;
                m_bwLoadDatabase.RunWorkerAsync(dlg.Parameters);
            }
        }

        private void m_bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
                setStatus("ERROR: " + e.Error.Message);
            else if (e.Cancelled)
                setStatus("ABORTED!");
            else
                setStatus("COMPLETED.");

            trainMNISTToolStripMenuItem.Enabled = false;
            testMNISTToolStripMenuItem.Enabled = false;
            destroyMyCaffeToolStripMenuItem.Enabled = false;
            deviceInformationToolStripMenuItem.Enabled = false;
            createMyCaffeToolStripMenuItem.Enabled = true;
            loadMNISTToolStripMenuItem.Enabled = true;
        }

        private void m_bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            ProgressInfo pi = e.UserState as ProgressInfo;

            if (pi.Alive.HasValue)
            {
                if (pi.Alive.Value == true)
                {
                    loadMNISTToolStripMenuItem.Enabled = true;
                    destroyMyCaffeToolStripMenuItem.Enabled = true;
                    trainMNISTToolStripMenuItem.Enabled = true;
                    testMNISTToolStripMenuItem.Enabled = true;
                    deviceInformationToolStripMenuItem.Enabled = true;
                }
                else
                {
                    loadMNISTToolStripMenuItem.Enabled = true;
                    createMyCaffeToolStripMenuItem.Enabled = true;
                }

                setStatus(pi.Message);
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

        private void loader_OnError(object sender, ProgressArgs e)
        {
            m_bwLoadDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
        }

        private void loader_OnProgress(object sender, ProgressArgs e)
        {
            m_bwLoadDatabase.ReportProgress((int)e.Progress.Percentage, e.Progress);
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
                WaitHandle[] rgWait = new WaitHandle[] { m_evtCancel.Handle, m_evtCommandRead };
                int nWait = WaitHandle.WaitAny(rgWait);

                if (nWait > 0)
                {
                    switch (m_Cmd)
                    {
                        case COMMAND.CREATE:
                            SettingsCaffe settings = new SettingsCaffe();
                            settings.ImageDbLoadMethod = SettingsCaffe.IMAGEDB_LOAD_METHOD.LOAD_ALL;
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
                            caffe.Train();
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
    }
}
