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
        List<string> m_rgstrInstances;

        public FormSqlInstances(List<string> rgSql)
        {
            m_rgstrInstances = rgSql;
            InitializeComponent();
        }

        private void FormSqlInstances_Load(object sender, EventArgs e)
        {
            foreach (string strSql in m_rgstrInstances)
            {
                cmbSql.Items.Add(strSql);
            }

            cmbSql.SelectedIndex = 0;
        }

        public string Instance
        {
            get { return m_strInstance; }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_strInstance = cmbSql.Text;
        }
    }
}
