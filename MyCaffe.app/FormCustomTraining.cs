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
        bool m_bLoadWeights = true;
        string m_strTrainer = "";
        string m_strRomName = "";
        double m_dfVMin = -10;
        double m_dfVMax = 10;
        int m_nIterations = 1000000;

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
                radC51SingleThread.Enabled = false;
                radNoisyNetSingleThread.Enabled = false;
                radNoisyNetSimple.Enabled = true;
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
            else if (m_strTrainer == "NOISYDQN.ST")
                radNoisyNetSingleThread.Checked = true;
            else if (m_strTrainer == "NOISYDQN.SIMPLE")
                radNoisyNetSimple.Checked = true;
            else
                radPGMultiThread.Checked = true;

            radC51SingleThread.Enabled = m_bAllowC51;

            edtVMin.Text = Properties.Settings.Default.CustVmin.ToString();
            edtVMax.Text = Properties.Settings.Default.CustVmax.ToString();
            edtIterations.Text = m_nIterations.ToString();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (!int.TryParse(edtIterations.Text, out m_nIterations) || m_nIterations < 1)
            {
                MessageBox.Show("The 'Iterations' value is invalid - please enter a valid positive integer value greater than or equal to 1.", "Invalid Iterations", MessageBoxButtons.OK, MessageBoxIcon.Error);
                edtIterations.Focus();
                DialogResult = DialogResult.None;
                return;
            }

            m_bShowUi = chkShowUi.Checked;
            m_bUseAcceleratedTraining = chkUseAcceleratedTraining.Checked;
            m_bAllowDiscountReset = chkAllowDiscountReset.Checked;
            m_bAllowNegativeRewards = chkAllowNegativeRewards.Checked;
            m_bTerminateOnRallyEnd = chkTerminateOnRallyEnd.Checked;
            m_bLoadWeights = chkLoadWeights.Checked;

            if (radPGSimple.Checked)
                m_strTrainer = "PG.SIMPLE";
            else if (radPGSingleThread.Checked)
                m_strTrainer = "PG.ST";
            else if (radC51SingleThread.Checked)
                m_strTrainer = "C51.ST";
            else if (radNoisyNetSingleThread.Checked)
                m_strTrainer = "NOISYDQN.ST";
            else if (radNoisyNetSimple.Checked)
                m_strTrainer = "NOISYDQN.SIMPLE";
            else
                m_strTrainer = "PG.MT";

            if (radAtariBreakout.Checked)
                m_strRomName = "breakout";
            else if (radAtariPong.Checked)
                m_strRomName = "pong";
            else
                m_strRomName = "";

            if (radC51SingleThread.Checked)
            {
                double dfVMin;
                double dfVMax;

                if (!double.TryParse(edtVMin.Text, out dfVMin))
                {
                    MessageBox.Show("The 'VMin' value is invalid.  Please enter a valid number.", "Invalid VMin", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    edtVMin.Focus();
                    DialogResult = DialogResult.None;
                    return;
                }

                if (!double.TryParse(edtVMax.Text, out dfVMax))
                {
                    MessageBox.Show("The 'VMax' value is invalid.  Please enter a valid number.", "Invalid VMax", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    edtVMax.Focus();
                    DialogResult = DialogResult.None;
                    return;
                }

                if (dfVMax <= dfVMin)
                {
                    MessageBox.Show("The 'VMax' value must be greater than the 'VMin' value.", "Invalid VMin,VMax", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    edtVMin.Focus();
                    DialogResult = DialogResult.None;
                    return;
                }

                m_dfVMin = dfVMin;
                m_dfVMax = dfVMax;

                Properties.Settings.Default.CustVmin = m_dfVMin;
                Properties.Settings.Default.CustVmax = m_dfVMax;
                Properties.Settings.Default.Save();
            }
        }

        public string RomName
        {
            get { return m_strRomName; }
        }

        public int Iterations
        {
            get { return m_nIterations; }
        }

        public double VMin
        {
            get { return m_dfVMin; }
        }

        public double VMax
        {
            get { return m_dfVMax; }
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

        public bool LoadWeights
        {
            get { return m_bLoadWeights; }
        }

        public string Trainer
        {
            get { return m_strTrainer; }
        }

        private void radC51SingleThread_CheckedChanged(object sender, EventArgs e)
        {
            lblVMin.Visible = radC51SingleThread.Checked;
            lblVMax.Visible = radC51SingleThread.Checked;
            edtVMin.Visible = radC51SingleThread.Checked;
            edtVMax.Visible = radC51SingleThread.Checked;
            btnReset.Visible = radC51SingleThread.Checked;

            if (radC51SingleThread.Checked)
            {
                radAtariBreakout.Checked = true;
                chkAllowNegativeRewards.Checked = true;
                chkTerminateOnRallyEnd.Checked = true;
                chkUseAcceleratedTraining.Checked = false;
                chkAllowDiscountReset.Checked = false;
            }
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            edtVMin.Text = "-10";
            edtVMax.Text = "10";
        }

        private void radNoisyNet_CheckedChanged(object sender, EventArgs e)
        {
            if (radNoisyNetSingleThread.Checked)
            {
                radAtariBreakout.Checked = true;
                chkAllowNegativeRewards.Checked = true;
                chkTerminateOnRallyEnd.Checked = true;
                chkUseAcceleratedTraining.Checked = false;
                chkAllowDiscountReset.Checked = false;
            }
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            if (radC51SingleThread.Checked ||
                radNoisyNetSingleThread.Checked)
            {
                radAtariBreakout.Enabled = true;
                radAtariBreakout.Checked = true;
                radAtariPong.Enabled = false;
            }
            if (radPGSimple.Checked ||
                radPGSingleThread.Checked ||
                radPGMultiThread.Checked)
            {
                radAtariBreakout.Enabled = false;
                radAtariPong.Checked = true;
                radAtariPong.Enabled = true;
            }
        }

        private void radNoisyNetSimple_CheckedChanged(object sender, EventArgs e)
        {
            if (radNoisyNetSimple.Checked)
                chkUseAcceleratedTraining.Checked = false;
        }
    }
}
