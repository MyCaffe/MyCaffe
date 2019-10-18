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
    public partial class FormVOC : Form
    {
        Dictionary<Button, TextBox> m_rgItems = new Dictionary<Button, TextBox>();
        VOCDataParameters m_param = null;
        List<WebClient> m_rgWebClients = new List<WebClient>();

        public FormVOC()
        {
            InitializeComponent();
            string strFile;

            strFile = Properties.Settings.Default.VocFile1;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtDataFile1.Text = strFile;

            strFile = Properties.Settings.Default.VocFile2;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtDataFile2.Text = strFile;

            strFile = Properties.Settings.Default.VocFile3;
            if (!string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                edtDataFile3.Text = strFile;

            edtDataFile1.Tag = "VOCtrainval_11-May-2012.tar";
            edtDataFile2.Tag = "VOCtrainval_06-Nov-2007.tar";
            edtDataFile3.Tag = "VOCtest_06-Nov-2007.tar";
            btnDownload1.Tag = new Tuple<string, TextBox, Label, Button, Button>("VOCtrainval_11-May-2012.tar", edtDataFile1, lblDownloadPct1, btnDownload1, btnBrowseTar1);
            btnDownload2.Tag = new Tuple<string, TextBox, Label, Button, Button>("VOCtrainval_06-Nov-2007.tar", edtDataFile2, lblDownloadPct2, btnDownload2, btnBrowseTar2);
            btnDownload3.Tag = new Tuple<string, TextBox, Label, Button, Button>("VOCtest_06-Nov-2007.tar", edtDataFile3, lblDownloadPct3, btnDownload3, btnBrowseTar3);

            m_rgItems.Add(btnBrowseTar1, edtDataFile1);
            m_rgItems.Add(btnBrowseTar2, edtDataFile2);
            m_rgItems.Add(btnBrowseTar3, edtDataFile3);
        }

        public VOCDataParameters Parameters
        {
            get { return m_param; }
        }

        private void FormCiFar10_Load(object sender, EventArgs e)
        {
            chkExtractFiles.Checked = Properties.Settings.Default.ExpandFiles;
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
            openFileDialogTar.FileName = edt.Tag.ToString();
            openFileDialogTar.Title = "Select the " + edt.Tag.ToString() + " BIN file.";

            if (openFileDialogTar.ShowDialog() == DialogResult.OK)
            {
                edt.Text = openFileDialogTar.FileName;
            }
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            bool bEnable = false;

            if (btnDownload1.Enabled && btnDownload2.Enabled && btnDownload3.Enabled)
                bEnable = true;

            btnOK.Enabled = bEnable;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_param = new app.VOCDataParameters(edtDataFile1.Text, edtDataFile2.Text, edtDataFile3.Text, chkExtractFiles.Checked);
        }

        private void btnDownload_Click(object sender, EventArgs e)
        {
            if (folderBrowserDialog1.ShowDialog() != DialogResult.OK)
                return;
            
            Button btn = (Button)sender;
            Tuple<string, TextBox, Label, Button, Button> edt = (Tuple<string, TextBox, Label, Button, Button>)btn.Tag;
            edt.Item2.Enabled = false;
            edt.Item4.Enabled = false;
            edt.Item5.Enabled = false;

            WebClient webClient = new WebClient();
            webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
            webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
            m_rgWebClients.Add(webClient);
            getFile(webClient, folderBrowserDialog1.SelectedPath, edt);
        }

        private void getFile(WebClient web, string strFolder, Tuple<string, TextBox, Label, Button, Button> edt)
        {
            string strDstFile = strFolder.TrimEnd('\\') + "\\" + edt.Item1;
            TextBox edtFile = edt.Item2;

            edtFile.Text = strDstFile;

            string strSubDir = (edt.Item1.Contains("2012")) ? "voc2012" : "voc2007";
            string strSrcFile = "http://" + lblDownloadSite.Text + "/" + strSubDir + "/" + edt.Item1;

            web.DownloadFileAsync(new Uri(strSrcFile), strDstFile, edt);   
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            double dfPct = (e.BytesReceived / (double)e.TotalBytesToReceive);
            Tuple<string, TextBox, Label, Button, Button> edt = (Tuple<string, TextBox, Label, Button, Button>)e.UserState;
            edt.Item3.Text = dfPct.ToString("P");
        }

        private void WebClient_DownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            Tuple<string, TextBox, Label, Button, Button> edt = (Tuple<string, TextBox, Label, Button, Button>)e.UserState;
            edt.Item2.Enabled = true;
            edt.Item4.Enabled = true;
            edt.Item5.Enabled = true;
            m_rgWebClients.Remove((WebClient)sender);
        }

        private void FormVOC_FormClosing(object sender, FormClosingEventArgs e)
        {
            string strFile;

            strFile = edtDataFile1.Text;
            if (btnDownload1.Enabled && !string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.VocFile1 = strFile;

            strFile = edtDataFile2.Text;
            if (btnDownload1.Enabled && !string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.VocFile2 = strFile;

            strFile = edtDataFile3.Text;
            if (btnDownload1.Enabled && !string.IsNullOrEmpty(strFile) && File.Exists(strFile))
                Properties.Settings.Default.VocFile3 = strFile;

            Properties.Settings.Default.ExpandFiles = chkExtractFiles.Checked;

            Properties.Settings.Default.Save();

            foreach (WebClient web in m_rgWebClients)
            {
                web.CancelAsync();
            }
        }
    }
}
