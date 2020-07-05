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
    public partial class FormSqlInstances : Form
    {
        string m_strInstance;
        string m_strCuda;
        List<string> m_rgstrInstances;
        Dictionary<string, string> m_rgstrCuda;

        public FormSqlInstances(List<string> rgSql, Dictionary<string, string> rgCuda)
        {
            m_rgstrInstances = rgSql;
            m_rgstrCuda = rgCuda;
            InitializeComponent();
        }

        private void FormSqlInstances_Load(object sender, EventArgs e)
        {
            int nIdxFound = 0;
            int nIdx = 0;

            foreach (string strSql in m_rgstrInstances)
            {
                if (strSql == Properties.Settings.Default.StartupDbInstance)
                    nIdxFound = nIdx;

                cmbSql.Items.Add(strSql);
                nIdx++;
            }

            cmbSql.SelectedIndex = nIdxFound;

            nIdxFound = 0;
            nIdx = 0;

            foreach (KeyValuePair<string, string> kv in m_rgstrCuda)
            {
                if (kv.Key == Properties.Settings.Default.StartupCudaPath)
                    nIdxFound = nIdx;

                cmbCuda.Items.Add(kv.Key);
                nIdx++;
            }

            cmbCuda.SelectedIndex = nIdxFound;
        }

        public string Instance
        {
            get { return m_strInstance; }
        }

        public string CudaPath
        {
            get { return m_rgstrCuda[m_strCuda]; }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_strInstance = cmbSql.Text;
            m_strCuda = cmbCuda.Text;

            Properties.Settings.Default.StartupDbInstance = m_strInstance;
            Properties.Settings.Default.StartupCudaPath = m_strCuda;
            Properties.Settings.Default.Save();
        }
    }
}
