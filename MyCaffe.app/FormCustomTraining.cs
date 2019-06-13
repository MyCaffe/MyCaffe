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
        bool m_bAllowDiscountReset = false;
        bool m_bAllowNegativeRewards = false;
        bool m_bTerminateOnRallyEnd = false;
        bool m_bAllowC51 = false;
        string m_strTrainer = "";
        string m_strRomName = "";

        public FormCustomTraining(string strName)
        {
            InitializeComponent();
            m_strName = strName;

            if (strName == "ATARI")
            {
                m_strTrainer = "PG.SIMPLE";
                m_bShowUi = false;
                m_bUseAcceleratedTraining = false;
                m_bAllowDiscountReset = true;
                m_bAllowC51 = true;
                grpRom.Visible = true;

            }
            else
            {
                m_strTrainer = "PG.MT";
                m_bShowUi = false;
                m_bUseAcceleratedTraining = true;
                m_bAllowDiscountReset = false;
                grpRom.Visible = false;
                chkTerminateOnRallyEnd.Visible = false;
                chkAllowNegativeRewards.Visible = false;
            }
        }

        private void FromCustomTraining_Load(object sender, EventArgs e)
        {
            lblGymName.Text = m_strName;

            chkShowUi.Checked = m_bShowUi;
            chkUseAcceleratedTraining.Checked = m_bUseAcceleratedTraining;
            chkAllowDiscountReset.Checked = m_bAllowDiscountReset;

            if (m_strTrainer == "PG.SIMPLE")
                radPGSimple.Checked = true;
            else if (m_strTrainer == "PG.ST")
                radPGSingleThread.Checked = true;
            else if (m_strTrainer == "C51.ST")
                radC51SingleThread.Checked = true;
            else
                radPGMultiThread.Checked = true;

            radC51SingleThread.Enabled = m_bAllowC51;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_bShowUi = chkShowUi.Checked;
            m_bUseAcceleratedTraining = chkUseAcceleratedTraining.Checked;
            m_bAllowDiscountReset = chkAllowDiscountReset.Checked;
            m_bAllowNegativeRewards = chkAllowNegativeRewards.Checked;
            m_bTerminateOnRallyEnd = chkTerminateOnRallyEnd.Checked;

            if (radPGSimple.Checked)
                m_strTrainer = "PG.SIMPLE";
            else if (radPGSingleThread.Checked)
                m_strTrainer = "PG.ST";
            else if (radC51SingleThread.Checked)
                m_strTrainer = "C51.ST";
            else
                m_strTrainer = "PG.MT";

            if (radAtariBreakout.Checked)
                m_strRomName = "breakout";
            else
                m_strRomName = "pong";
        }

        public string RomName
        {
            get { return m_strRomName; }
        }

        public bool ShowUserInterface
        {
            get { return m_bShowUi; }
        }

        public bool UseAcceleratedTraining
        {
            get { return m_bUseAcceleratedTraining; }
        }

        public bool AllowDiscountReset
        {
            get { return m_bAllowDiscountReset; }
        }

        public bool TerminateOnRallyEnd
        {
            get { return m_bTerminateOnRallyEnd; }
        }

        public bool AllowNegativeRewards
        {
            get { return m_bAllowNegativeRewards; }
        }

        public string Trainer
        {
            get { return m_strTrainer; }
        }
    }
}
