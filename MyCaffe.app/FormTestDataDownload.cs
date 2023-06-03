using MyCaffe.data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormTestDataDownload : Form
    {
        StringBuilder m_sb = new StringBuilder();
        string m_strTargetFolder;
        List<string> m_rgstrUrl;
        Stopwatch m_sw = new Stopwatch();

        public FormTestDataDownload(string strTargetFolder, List<string> rgstrUrl)
        {
            m_rgstrUrl = rgstrUrl;
            m_strTargetFolder = strTargetFolder;
            InitializeComponent();
        }

        private void FormTestDataDownload_Load(object sender, EventArgs e)
        {
            lblTargetFolder.Text = m_strTargetFolder;
        }

        private void setStatus(string str)
        {
            m_sb.Append(Environment.NewLine);
            m_sb.Append(str);
            edtStatus.Text = m_sb.ToString();
            edtStatus.SelectionStart = edtStatus.Text.Length;
            edtStatus.ScrollToCaret();
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            m_sw.Start();

            if (!Directory.Exists(m_strTargetFolder))
            {
                if (MessageBox.Show("The target folder '" + m_strTargetFolder + "' does not exist - do you want to create it?", "Missing Target Folder", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Exclamation) != DialogResult.Yes)
                    return;

                Directory.CreateDirectory(m_strTargetFolder);
            }

            m_bw.RunWorkerAsync();
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            m_bw.CancelAsync();
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            btnStart.Enabled = !m_bw.IsBusy;
            btnStop.Enabled = m_bw.IsBusy && !m_bw.CancellationPending;
        }

        private void m_bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            ProgressInfo pi = e.UserState as ProgressInfo;
            pbProgress.Value = (int)(pi.Percentage * 100);
            lblPct.Text = (pi.Percentage * 100).ToString("N2") + "%";

            if (pi.Percentage < 1)
            {
                if (pi.Message == "Bytes")
                {
                    double dfMbDownloaded = (double)pi.Index / 1000000.0;
                    double dfMbTotal = (double)pi.Total / 1000000.0;
                    lblStatus.Text = "downloading " + dfMbDownloaded.ToString("N2") + " of " + dfMbTotal.ToString("N2") + " MB...";
                }
                else
                {
                    lblStatus.Text = "downloading...";
                }
            }
            else
                lblStatus.Text = "idle.";

            if (pi.Message != "Bytes")
                setStatus(pi.Message);

            Application.DoEvents();

            m_sw.Restart();
        }

        private void m_bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                lblStatus.Text = "ERROR!";
                setStatus("ERROR: " + e.Error.Message);
            }
            else if (e.Cancelled)
            {
                lblStatus.Text = "CANCELLED!";
                setStatus("CANCELLED!");
            }
            else
            {
                lblStatus.Text = "DONE!";
                setStatus("DONE!");
            }
        }

        private void m_bw_DoWork(object sender, DoWorkEventArgs e)
        {
            ProgressInfo pi;
            BackgroundWorker bw = sender as BackgroundWorker;

            for (int i = 0; i < m_rgstrUrl.Count; i++)
            {
                string strFile;
                if (!download(bw, i, m_rgstrUrl.Count, m_strTargetFolder, m_rgstrUrl[i], out strFile))
                {
                    e.Cancel = true;
                    return;
                }

                if (!expand(bw, i, m_rgstrUrl.Count, m_strTargetFolder, strFile))
                {
                    e.Cancel = true;
                    return;
                }
            }
        }

        private bool download(BackgroundWorker bw, int nIdx, int nTotal, string strTargetFolder, string strUrl, out string strTargetFile)
        {
            // Download the file and report status.
            string strFileName = strUrl.Substring(strUrl.LastIndexOf('/') + 1);
            strTargetFile = strTargetFolder + "\\" + strFileName;
            ProgressInfo pi;

            if (File.Exists(strTargetFile))
            {
                pi = new ProgressInfo(nIdx, nTotal, "File '" + strTargetFile + "' already exists, skipping.");
                bw.ReportProgress((int)(pi.Percentage * 100), pi);
                return true;
            }

            pi = new ProgressInfo(nIdx, nTotal, "Downloading '" + strUrl + "' to '" + strTargetFile + "'...");
            bw.ReportProgress((int)(pi.Percentage * 100), pi);

            using (WebClient client = new WebClient())
            {
                client.DownloadProgressChanged += client_DownloadProgressChanged; ;
                client.DownloadFileCompleted += client_DownloadFileCompleted; ;
                client.DownloadFileAsync(new Uri(strUrl), strTargetFile);
                
                while (client.IsBusy)
                {
                    if (bw.CancellationPending)
                    {
                        client.CancelAsync();
                        return false;
                    }

                    System.Threading.Thread.Sleep(250);
                }
            }

            return true;
        }

        private bool expand(BackgroundWorker bw, int nIdx, int nTotal, string strTargetFolder, string strFile)
        {
            try
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                using (ZipArchive zip = ZipFile.OpenRead(strFile))
                {
                    ProgressInfo pi = new ProgressInfo(nIdx, nTotal, "Extracting '" + strFile + "' to '" + strTargetFolder + "'...");
                    bw.ReportProgress((int)(pi.Percentage * 100), pi);

                    nIdx = 0;
                    nTotal = zip.Entries.Count;

                    foreach (ZipArchiveEntry entry in zip.Entries)
                    {
                        if (bw.CancellationPending)
                            return false;

                        string strPath = Path.Combine(strTargetFolder, entry.FullName.Replace('/', '\\'));
                        if (entry.Length == 0)
                        {
                            if (!Directory.Exists(strPath))
                                Directory.CreateDirectory(strPath);
                        }
                        else
                        {
                            entry.ExtractToFile(strPath, true);
                        }
                        nIdx++;

                        if (sw.Elapsed.TotalMilliseconds > 2000)
                        {
                            pi = new ProgressInfo(nIdx, nTotal, "Extracting '" + entry.FullName + "'...");
                            bw.ReportProgress((int)(pi.Percentage * 100), pi);
                            sw.Restart();
                        }
                    }
                }
            }
            catch (Exception excpt)
            {
                ProgressInfo pi = new ProgressInfo(nIdx, nTotal, "ERROR: expanding '" + strFile + "' - " + excpt.Message + " You may need to re-download the zip file.");
                bw.ReportProgress((int)(pi.Percentage * 100), pi);
                return false;
            }

            return true;
        }

        private void client_DownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            ProgressInfo pi = new ProgressInfo(1, 1, "download completed.");
            m_bw.ReportProgress((int)(pi.Percentage * 100), pi);
        }

        private void client_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_sw.Elapsed.TotalMilliseconds < 1000)
                return;

            double dfBytesReceivedInMb = e.BytesReceived / 1024 / 1024.0;
            double dfTotalBytesToReceiveInMb = e.TotalBytesToReceive / 1024 / 1024.0;

            ProgressInfo pi = new ProgressInfo(e.BytesReceived, e.TotalBytesToReceive, "Bytes");
            m_bw.ReportProgress((int)(pi.Percentage * 100), pi);
        }

        private void btnDeleteExistingFiles_Click(object sender, EventArgs e)
        {
            string[] rgstrFiles = Directory.GetFiles(m_strTargetFolder);

            if (rgstrFiles.Length > 0)
            {
                for (int i = 0; i < rgstrFiles.Length; i++)
                {
                    File.Delete(rgstrFiles[i]);
                }

                MessageBox.Show("Deleted " + rgstrFiles.Length.ToString() + " files.", "Delete Files", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                MessageBox.Show("No files to delete.", "Delete Files", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }
    }
}
