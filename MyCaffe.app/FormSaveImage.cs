using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormSaveImage : Form
    {
        Size m_szImage;
        string m_strFolder;
        bool m_bEnableSaving;

        public FormSaveImage()
        {
            InitializeComponent();
        }

        public Size ImageSize
        {
            get { return m_szImage; }
        }

        public string Folder
        {
            get { return m_strFolder; }
        }

        public bool EnableSaving
        {
            get { return m_bEnableSaving; }
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            if (folderBrowserDialogTestImages.ShowDialog() == DialogResult.OK)
                edtFolder.Text = folderBrowserDialogTestImages.SelectedPath;
        }

        private void FormSaveImage_Load(object sender, EventArgs e)
        {
            edtW.Text = Properties.Settings.Default.ResizeW.ToString();
            edtH.Text = Properties.Settings.Default.ResizeH.ToString();
            edtFolder.Text = Properties.Settings.Default.TestImageFolder;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            int nH;
            if (!int.TryParse(edtH.Text, out nH) || nH < 2 || nH > 512)
            {
                MessageBox.Show("The 'H' is incorrect - please specify a valid integer in the range [2,512].", "Invalid 'H' Value", MessageBoxButtons.OK, MessageBoxIcon.Error);
                DialogResult = DialogResult.None;
                edtH.Focus();
                return;
            }

            int nW;
            if (!int.TryParse(edtW.Text, out nW) || nW < 2 || nW > 512)
            {
                MessageBox.Show("The 'W' is incorrect - please specify a valid integer in the range [2,512].", "Invalid 'W' Value", MessageBoxButtons.OK, MessageBoxIcon.Error);
                DialogResult = DialogResult.None;
                edtW.Focus();
                return;
            }

            m_szImage = new Size(nW, nH);

            if (edtFolder.Text.Length == 0)
            {
                MessageBox.Show("You must specify an output foler!", "Missing Output Folder", MessageBoxButtons.OK, MessageBoxIcon.Error);
                DialogResult = DialogResult.None;
                edtFolder.Focus();
                return;
            }

            m_strFolder = edtFolder.Text;
            m_bEnableSaving = chkEnableSavingTestImages.Checked;

            Properties.Settings.Default.TestImageFolder = edtFolder.Text;
            Properties.Settings.Default.ResizeW = m_szImage.Width;
            Properties.Settings.Default.ResizeH = m_szImage.Height;
            Properties.Settings.Default.Save();
        }
    }
}
