using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MyCaffe.test.automated
{
    public partial class FormAutomatedTests : Form
    {
        string m_strPath;
        string m_strCulture;
        int m_nGpuID = 0;
        DB_VERSION m_imgDbVer = DB_VERSION.DEFAULT;
        string m_strCudaPath = "";

        public FormAutomatedTests(string strPath, int nGpuID, DB_VERSION imgDbVer, string strCulture, string strCudaPath)
        {
            m_strCulture = strCulture;
            m_strPath = strPath;
            m_nGpuID = nGpuID;
            m_imgDbVer = imgDbVer;
            m_strCudaPath = strCudaPath;
            InitializeComponent();
        }

        private void FormAutomatedTests_Load(object sender, EventArgs e)
        {
            automatedTester1.TestAssemblyPath = m_strPath;
            automatedTester1.GpuId = m_nGpuID;
            automatedTester1.ImageDbVersion = m_imgDbVer;
            automatedTester1.CudaPath = m_strCudaPath;
            automatedTester1.Culture = m_strCulture;
            automatedTester1.LoadFromDatabase();
            Text = "Automated Tests [" + automatedTester1.TestName + "]";
        }

        private void FormAutomatedTests_FormClosing(object sender, FormClosingEventArgs e)
        {
            automatedTester1.SaveToDatabase();
            automatedTester1.Dispose();
        }
    }
}
