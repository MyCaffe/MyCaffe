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
        int m_nGpuID = 0;
        IMGDB_VERSION m_imgDbVer = IMGDB_VERSION.DEFAULT;
        string m_strCudaPath = "";

        public FormAutomatedTests(string strPath, int nGpuID, IMGDB_VERSION imgDbVer, string strCudaPath)
        {
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
            automatedTester1.LoadFromDatabase();
            Text = "Automated Tests [" + automatedTester1.TestName + "]";
        }

        private void FormAutomatedTests_FormClosing(object sender, FormClosingEventArgs e)
        {
            automatedTester1.SaveToDatabase();            
        }
    }
}
