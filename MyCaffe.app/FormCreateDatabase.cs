using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormCreateDatabase : Form
    {
        string m_strTitle;
        string m_strName;
        string m_strPath = "c:\\temp";

        public FormCreateDatabase(string strName = "DNN", string strTitle = "")
        {
            m_strTitle = strTitle;
            m_strName = strName;
            InitializeComponent();
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            folderBrowserDialog.SelectedPath = edtPath.Text;

            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
                edtPath.Text = folderBrowserDialog.SelectedPath;
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            if (Directory.Exists(edtPath.Text))
                btnOK.Enabled = true;
            else
                btnOK.Enabled = false;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_strName = edtName.Text;
            m_strPath = edtPath.Text;
        }

        public string DatabaseName
        {
            get { return m_strName; }
        }

        public string DatabasePath
        {
            get { return m_strPath; }
        }

        private void FormCreateDatabase_Load(object sender, EventArgs e)
        {
            edtName.Text = m_strName;
            edtPath.Text = m_strPath;

            if (m_strTitle.Length > 0)
                Text = m_strTitle;
        }
    }
}
