using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace MyCaffe.param.ui
{
    public partial class DictionaryParameterEditorControl : UserControl /** @private */
    {
        IWindowsFormsEditorService m_svc;
        string m_strVal;

        public DictionaryParameterEditorControl(string strVal, IWindowsFormsEditorService svc)
        {
            m_svc = svc;
            m_strVal = strVal;
            InitializeComponent();
        }

        public string Value
        {
            get
            {
                m_strVal = "";

                foreach (DataGridViewRow row in dgvItems.Rows)
                {
                    if (row.Cells[0].Value != null)
                    {
                        string strItem = row.Cells[0].Value.ToString() + "=";
                        if (row.Cells[1].Value != null)
                            strItem += row.Cells[1].Value.ToString();

                        m_strVal += strItem + ";";
                    }
                }

                return m_strVal.TrimEnd(';');
            }
        }

        private void SolverParameterEditorControl_Load(object sender, EventArgs e)
        {
            string[] rgstrItem = m_strVal.Split(';');

            for (int i = 0; i < rgstrItem.Length; i++)
            {
                string[] rgstrVal = rgstrItem[i].Split('=');

                if (rgstrVal.Count() == 2)
                    dgvItems.Rows.Add(rgstrVal[0], rgstrVal[1]);
            }
        }
    }
}
