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
    public partial class FormCustomTraining : Form
    {
        string m_strName;
        bool m_bShowUi = false;
        bool m_bUseAcceleratedTraining = false;
        string m_strTrainer = "";

        public FormCustomTraining(string strName)
        {
            InitializeComponent();
            m_strName = strName;

            if (strName == "ATARI")
            {
                m_strTrainer = "SIMPLE";
                m_bShowUi = false;
                m_bUseAcceleratedTraining = false;
            }
            else
            {
                m_strTrainer = "MT";
                m_bShowUi = false;
                m_bUseAcceleratedTraining = true;
            }
        }

        private void FromCustomTraining_Load(object sender, EventArgs e)
        {
            lblGymName.Text = m_strName;

            chkShowUi.Checked = m_bShowUi;
            chkUseAcceleratedTraining.Checked = m_bUseAcceleratedTraining;

            if (m_strTrainer == "SIMPLE")
                radSimple.Checked = true;
            else if (m_strTrainer == "ST")
                radSingleThread.Checked = true;
            else
                radMultiThread.Checked = true;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_bShowUi = chkShowUi.Checked;
            m_bUseAcceleratedTraining = chkUseAcceleratedTraining.Checked;

            if (radSimple.Checked)
                m_strTrainer = "SIMPLE";
            else if (radSingleThread.Checked)
                m_strTrainer = "ST";
            else
                m_strTrainer = "MT";
        }

        public bool ShowUserInterface
        {
            get { return m_bShowUi; }
        }

        public bool UseAcceleratedTraining
        {
            get { return m_bUseAcceleratedTraining; }
        }

        public string Trainer
        {
            get { return m_strTrainer; }
        }
    }
}
