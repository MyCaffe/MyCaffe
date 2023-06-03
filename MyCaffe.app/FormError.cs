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
    public partial class FormError : Form
    {
        string m_strType;
        Exception m_err;

        public FormError(string strErrorType, Exception err)
        {
            m_strType = strErrorType;
            m_err = err;
            InitializeComponent();
        }

        private void FormError_Load(object sender, EventArgs e)
        {
            Text += " - " + m_strType;

            Exception err = m_err;

            while (err != null)
            {
                edtError.Text += err.Message;
                edtError.Text += Environment.NewLine;
                err = err.InnerException;
            }

            if (err != null)
                edtLocation.Text = err.StackTrace;
        }
    }
}
