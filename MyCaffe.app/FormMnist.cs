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
    public partial class FormMnist : Form
    {
        Dictionary<Button, TextBox> rgItems = new Dictionary<Button, TextBox>();
        MnistDataParameters m_param = null;

        public FormMnist()
        {
            InitializeComponent();

            edtTrainImagesFile.Tag = "train-images-idx3-ubyte";
            edtTrainLabelsFile.Tag = "train-labels-idx1-ubyte";
            edtTestImagesFile.Tag = "t10k-images-idx3-ubyte";
            edtTestLabelsFile.Tag = "t10k-labels-idx1-ubyte";

            rgItems.Add(btnBrowseGz1, edtTrainImagesFile);
            rgItems.Add(btnBrowseGz2, edtTrainLabelsFile);
            rgItems.Add(btnBrowseGz3, edtTestImagesFile);
            rgItems.Add(btnBrowseGz4, edtTestLabelsFile);
        }

        public MnistDataParameters Parameters
        {
            get { return m_param; }
        }

        private void FormMnist_Load(object sender, EventArgs e)
        {

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
            TextBox edt = rgItems[(Button)sender];

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

            foreach (KeyValuePair<Button, TextBox> kv in rgItems)
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
            m_param = new MnistDataParameters(edtTrainImagesFile.Text, edtTrainLabelsFile.Text, edtTestImagesFile.Text, edtTestLabelsFile.Text);
        }
    }
}
