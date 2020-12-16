using MyCaffe.basecode;
using MyCaffe.data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormMnist : Form
    {
        Dictionary<Button, TextBox> m_rgItems = new Dictionary<Button, TextBox>();
        MnistDataParameters m_param = null;
        bool m_bSqlLoaded;
        WebClient m_webClient = null;
        List<string> m_rgstrDownloadedFiles = new List<string>();
        List<string> m_rgstrSrcFiles = new List<string>();
        int m_nDownloadFileIdx = 0;

        public FormMnist(bool bSqlLoaded)
        {
            InitializeComponent();

            m_rgstrSrcFiles = new List<string>()
            {
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
            };

            m_bSqlLoaded = bSqlLoaded;
            edtTrainImagesFile.Tag = m_rgstrSrcFiles[0];
            edtTrainLabelsFile.Tag = m_rgstrSrcFiles[1];
            edtTestImagesFile.Tag = m_rgstrSrcFiles[2];
            edtTestLabelsFile.Tag = m_rgstrSrcFiles[3];

            m_rgItems.Add(btnBrowseGz1, edtTrainImagesFile);
            m_rgItems.Add(btnBrowseGz2, edtTrainLabelsFile);
            m_rgItems.Add(btnBrowseGz3, edtTestImagesFile);
            m_rgItems.Add(btnBrowseGz4, edtTestLabelsFile);
        }

        public MnistDataParameters Parameters
        {
            get { return m_param; }
        }

        private void FormMnist_Load(object sender, EventArgs e)
        {
            string strFile;

            strFile = Properties.Settings.Default.MnistFile1;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtTestImagesFile.Text = strFile;

            strFile = Properties.Settings.Default.MnistFile2;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtTestLabelsFile.Text = strFile;

            strFile = Properties.Settings.Default.MnistFile3;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtTrainImagesFile.Text = strFile;

            strFile = Properties.Settings.Default.MnistFile4;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtTrainLabelsFile.Text = strFile;

            string strFolder = Properties.Settings.Default.MnistExportFolder;
            if (!string.IsNullOrEmpty(strFolder) && Directory.Exists(strFolder))
                edtExportFolder.Text = strFolder;

            chkExportToFile.Checked = Properties.Settings.Default.ExportMnistToFile;

            if (!m_bSqlLoaded)
            {
                chkExportToFile.Checked = true;
                chkExportToFile.Enabled = false;
            }
        }

        private void lblDownloadSite_MouseHover(object sender, EventArgs e)
        {
            lblDownloadSite.ForeColor = Color.SkyBlue;
        }

        private void lblDownloadSite_MouseLeave(object sender, EventArgs e)
        {
            lblDownloadSite.ForeColor = Color.Blue;
        }

        private void lblDownloadSite_Click(object sender, EventArgs e)
        {
            string strUrl = "http://" + lblDownloadSite.Text;

            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(strUrl);
            p.Start();
        }

        private void btnBrowseGz_Click(object sender, EventArgs e)
        {
            TextBox edt = m_rgItems[(Button)sender];

            openFileDialogGz.FileName = edt.Tag.ToString();
            openFileDialogGz.Title = "Select the " + edt.Tag.ToString() + " GZ file.";

            if (openFileDialogGz.ShowDialog() == DialogResult.OK)
            {
                edt.Text = openFileDialogGz.FileName;
            }
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            bool bEnable = true;

            foreach (KeyValuePair<Button, TextBox> kv in m_rgItems)
            {
                if (kv.Value.Text == null || kv.Value.Text.Length == 0)
                {
                    bEnable = false;
                    break;
                }

                FileInfo fi = new FileInfo(kv.Value.Text);

                if (!fi.Exists)
                {
                    bEnable = false;
                    break;
                }
            }

            btnOK.Enabled = bEnable;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (chkExportToFile.Checked && string.IsNullOrEmpty(edtExportFolder.Text))
            {
                MessageBox.Show("You must specify a folder to export into!", "No Export Folder", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                DialogResult = DialogResult.None;
                edtExportFolder.Focus();
                return;
            }

            m_param = new MnistDataParameters(edtTrainImagesFile.Text, edtTrainLabelsFile.Text, edtTestImagesFile.Text, edtTestLabelsFile.Text, chkExportToFile.Checked, edtExportFolder.Text);
        }

        private void FormMnist_FormClosing(object sender, FormClosingEventArgs e)
        {
            string strFile;

            strFile = edtTestImagesFile.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.MnistFile1 = strFile;

            strFile = edtTestLabelsFile.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.MnistFile2 = strFile;

            strFile = edtTrainImagesFile.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.MnistFile3 = strFile;

            strFile = edtTrainLabelsFile.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.MnistFile4 = strFile;

            string strFolder = edtExportFolder.Text;
            if (!string.IsNullOrEmpty(strFolder) && Directory.Exists(strFolder))
                Properties.Settings.Default.MnistExportFolder = strFolder;

            Properties.Settings.Default.ExportMnistToFile = chkExportToFile.Checked;

            Properties.Settings.Default.Save();
        }

        private void chkExportToFile_CheckedChanged(object sender, EventArgs e)
        {
            edtExportFolder.Enabled = chkExportToFile.Checked;
        }

        private void btnBrowseFolder_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.SelectedPath = edtExportFolder.Text;

            if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
                edtExportFolder.Text = folderBrowserDialog1.SelectedPath;
        }

        private void btnDownload_Click(object sender, EventArgs e)
        {
            string strUrl = lblDownloadSite.Text;

            string strFileName = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            strFileName += "\\MyCaffe\\downloads\\mnist\\";

            if (!Directory.Exists(strFileName))
                Directory.CreateDirectory(strFileName);

            if (m_nDownloadFileIdx == m_rgstrSrcFiles.Count)
                m_nDownloadFileIdx = 0;
           
            string strFile = m_rgstrSrcFiles[m_nDownloadFileIdx];
            strFileName += strFile;
            m_nDownloadFileIdx++;

            if (m_webClient != null)
                m_webClient.CancelAsync();

            if (m_webClient == null)
            {
                m_webClient = new WebClient();
                m_webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                m_webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
            }

            btnDownload.Enabled = false;
            lblDownloadPct.Enabled = false;
            m_webClient.DownloadFileAsync(new Uri("http://" + strUrl + "//" + strFile), strFileName, new Tuple<Label, Button, string>(lblDownloadPct, btnDownload, strFileName));
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            double dfPct = (e.BytesReceived / (double)e.TotalBytesToReceive);
            Tuple<Label, Button, string> edt = e.UserState as Tuple<Label, Button, string>;
            edt.Item1.Text = dfPct.ToString("P");
        }

        private void WebClient_DownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            if (e.Cancelled)
                return;

            Tuple<Label, Button, string> edt = e.UserState as Tuple<Label, Button, string>;
            string strDir = Path.GetDirectoryName(edt.Item3);

            m_rgstrDownloadedFiles.Add(edt.Item3);

            if (m_rgstrDownloadedFiles.Count < 4)
            {
                btnDownload_Click(sender, new EventArgs());
            }
            else
            {
                edt.Item1.Enabled = true;
                edt.Item2.Enabled = true;

                edtTrainImagesFile.Text = m_rgstrDownloadedFiles[0];
                edtTrainLabelsFile.Text = m_rgstrDownloadedFiles[1];
                edtTestImagesFile.Text = m_rgstrDownloadedFiles[2];
                edtTestLabelsFile.Text = m_rgstrDownloadedFiles[3];
            }

            if (m_webClient != null)
            {
                m_webClient.Dispose();
                m_webClient = null;
            }
        }
    }
}
