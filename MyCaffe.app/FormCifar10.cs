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
    public partial class FormCifar10 : Form
    {
        Dictionary<Button, TextBox> m_rgItems = new Dictionary<Button, TextBox>();
        CiFar10DataParameters m_param = null;
        WebClient m_webClient = null;

        public FormCifar10()
        {
            InitializeComponent();

            edtCifarDataFile1.Tag = "data_batch_1.bin";
            edtCifarDataFile2.Tag = "data_batch_2.bin";
            edtCifarDataFile3.Tag = "data_batch_3.bin";
            edtCifarDataFile4.Tag = "data_batch_4.bin";
            edtCifarDataFile5.Tag = "data_batch_5.bin";
            edtCifarDataFile6.Tag = "test_batch.bin";

            m_rgItems.Add(btnBrowseBin1, edtCifarDataFile1);
            m_rgItems.Add(btnBrowseBin2, edtCifarDataFile2);
            m_rgItems.Add(btnBrowseBin3, edtCifarDataFile3);
            m_rgItems.Add(btnBrowseBin4, edtCifarDataFile4);
            m_rgItems.Add(btnBrowseBin5, edtCifarDataFile5);
            m_rgItems.Add(btnBrowseBin6, edtCifarDataFile6);
        }

        public CiFar10DataParameters Parameters
        {
            get { return m_param; }
        }

        private void FormCiFar10_Load(object sender, EventArgs e)
        {
            string strFile;

            strFile = Properties.Settings.Default.CiFarFile1;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtCifarDataFile1.Text = strFile;

            strFile = Properties.Settings.Default.CiFarFile2;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtCifarDataFile2.Text = strFile;

            strFile = Properties.Settings.Default.CiFarFile3;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtCifarDataFile3.Text = strFile;

            strFile = Properties.Settings.Default.CiFarFile4;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtCifarDataFile4.Text = strFile;

            strFile = Properties.Settings.Default.CiFarFile5;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtCifarDataFile5.Text = strFile;

            strFile = Properties.Settings.Default.CiFarFileTest;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtCifarDataFile6.Text = strFile;
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

        private void btnBrowseBin_Click(object sender, EventArgs e)
        {
            TextBox edt = m_rgItems[(Button)sender];

            openFileDialogBin.FileName = edt.Tag.ToString();
            openFileDialogBin.Title = "Select the " + edt.Tag.ToString() + " BIN file.";

            if (openFileDialogBin.ShowDialog() == DialogResult.OK)
            {
                edt.Text = openFileDialogBin.FileName;
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
            m_param = new CiFar10DataParameters(edtCifarDataFile1.Text, edtCifarDataFile2.Text, edtCifarDataFile3.Text, edtCifarDataFile4.Text, edtCifarDataFile5.Text, edtCifarDataFile6.Text);
        }

        private void FormCifar10_FormClosing(object sender, FormClosingEventArgs e)
        {
            string strFile;

            strFile = edtCifarDataFile1.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.CiFarFile1 = strFile;

            strFile = edtCifarDataFile2.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.CiFarFile2 = strFile;

            strFile = edtCifarDataFile3.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.CiFarFile3 = strFile;

            strFile = edtCifarDataFile4.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.CiFarFile4 = strFile;

            strFile = edtCifarDataFile5.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.CiFarFile5 = strFile;

            strFile = edtCifarDataFile6.Text;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.CiFarFileTest = strFile;

            Properties.Settings.Default.Save();

            if (m_webClient != null)
            {
                m_webClient.CancelAsync();
                m_webClient = null;
            }
        }

        private void btnDownload_Click(object sender, EventArgs e)
        {
            string strUrl = lblDownloadSite.Text;

            string strFileName = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            strFileName += "\\MyCaffe\\downloads\\";

            if (!Directory.Exists(strFileName))
                Directory.CreateDirectory(strFileName);

            strFileName += "cifar-10.gz";

            if (m_webClient != null)
                m_webClient.CancelAsync();

            m_webClient = new WebClient();
            m_webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
            m_webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
            btnDownload.Enabled = false;
            lblDownloadPct.Enabled = false;
            m_webClient.DownloadFileAsync(new Uri("http://" + strUrl), strFileName, new Tuple<Label, Button, string>(lblDownloadPct, btnDownload, strFileName));
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            double dfPct = (e.BytesReceived / (double)e.TotalBytesToReceive);
            Tuple<Label, Button, string> edt = e.UserState as Tuple<Label, Button, string>;
            edt.Item1.Text = dfPct.ToString("P");
        }

        private void WebClient_DownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            Tuple<Label, Button, string> edt = e.UserState as Tuple<Label, Button, string>;
            edt.Item1.Enabled = true;
            edt.Item2.Enabled = true;

            string strDir = Path.GetDirectoryName(edt.Item3);
            TarFile.ExtractTarGz(edt.Item3, strDir);

            edtCifarDataFile1.Text = strDir + "\\cifar-10-batches-bin\\data_batch_1.bin";
            edtCifarDataFile2.Text = strDir + "\\cifar-10-batches-bin\\data_batch_2.bin";
            edtCifarDataFile3.Text = strDir + "\\cifar-10-batches-bin\\data_batch_3.bin";
            edtCifarDataFile4.Text = strDir + "\\cifar-10-batches-bin\\data_batch_4.bin";
            edtCifarDataFile5.Text = strDir + "\\cifar-10-batches-bin\\data_batch_5.bin";
            edtCifarDataFile6.Text = strDir + "\\cifar-10-batches-bin\\test_batch.bin";
        }
    }
}
