using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.param.ui
{
    /// <summary>
    /// The FormProperty window is used to edit a given key/value pair.
    /// </summary>
    public partial class FormProperty : Form
    {
        bool m_bNew = false;
        string m_strName = null;
        string m_strVal = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bNew">Specifies that this is a new item.</param>
        /// <param name="strName">When editing, specifies the existing name.</param>
        /// <param name="strVal">When editing, specifies the existing value.</param>
        public FormProperty(bool bNew, string strName, string strVal)
        {
            InitializeComponent();

            m_bNew = bNew;
            m_strName = strName;
            m_strVal = strVal;
        }

        /// <summary>
        /// Returns the Key (e.g. the property name).
        /// </summary>
        public string Key
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the property value.
        /// </summary>
        public string Value
        {
            get { return m_strVal; }
        }

        private void FormProperty_Load(object sender, EventArgs e)
        {
            edtName.Text = m_strName;
            edtValue.Text = m_strVal;

            if (m_bNew)
                Text = "New Property";
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_strName = edtName.Text;
            m_strVal = edtValue.Text;
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            if (edtName.Text.Length > 0 && edtValue.Text.Length > 0)
                btnOK.Enabled = true;
            else
                btnOK.Enabled = false;
        }
    }
}
