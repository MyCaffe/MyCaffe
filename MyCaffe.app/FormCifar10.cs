using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormCifar10 : Form
    {
        Dictionary<Button, TextBox> m_rgItems = new Dictionary<Button, TextBox>();
        CiFar10DataParameters m_param = null;

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
            m_param = new app.CiFar10DataParameters(edtCifarDataFile1.Text, edtCifarDataFile2.Text, edtCifarDataFile3.Text, edtCifarDataFile4.Text, edtCifarDataFile5.Text, edtCifarDataFile6.Text);
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
        }
    }
}
