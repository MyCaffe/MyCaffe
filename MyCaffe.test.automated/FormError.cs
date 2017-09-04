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
    public partial class FormError : Form
    {
        ErrorInfo m_error;

        public FormError(ErrorInfo error)
        {
            m_error = error;
            InitializeComponent();
        }

        private void FormError_Load(object sender, EventArgs e)
        {
            edtError.Text = m_error.FullErrorString;
            edtLocation.Text = m_error.FullErrorStringLocation;
        }
    }
}
