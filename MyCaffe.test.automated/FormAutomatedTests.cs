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

        public FormAutomatedTests(string strPath)
        {
            m_strPath = strPath;
            InitializeComponent();
        }

        private void FormAutomatedTests_Load(object sender, EventArgs e)
        {
            automatedTester1.TestAssemblyPath = m_strPath;
            automatedTester1.LoadFromDatabase();
            Text = "Automated Tests [" + automatedTester1.TestName + "]";
        }

        private void FormAutomatedTests_FormClosing(object sender, FormClosingEventArgs e)
        {
            automatedTester1.SaveToDatabase();            
        }
    }
}
