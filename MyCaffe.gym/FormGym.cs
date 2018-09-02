using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.gym
{
    public partial class FormGym : Form
    {
        MyCaffeGymControl m_ctrl;
        Dictionary<string, int> m_rgActionSpace;

        public FormGym(MyCaffeGymControl ctrl)
        {
            InitializeComponent();
            ctrl.Dock = DockStyle.Fill;
            toolStripContainer1.ContentPanel.Controls.Add(ctrl);
            m_ctrl = ctrl;
            m_rgActionSpace = ctrl.GetActionSpace();
        }

        public MyCaffeGymControl GymControl
        {
            get { return m_ctrl; }
        }

        public string GymName
        {
            get { return (m_ctrl == null) ? "UNKNOWN" : m_ctrl.GymName; }
        }

        public void Stop()
        {
            m_ctrl.Stop();
        }

        private void FormGym_Load(object sender, EventArgs e)
        {
            Text += " - " + GymName;
            m_ctrl.Render(null);
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            if (m_ctrl == null)
            {
                btnRun.Enabled = false;
                btnStop.Enabled = false;
            }
            else
            {
                btnRun.Enabled = !m_ctrl.IsRunning;
                btnStop.Enabled = m_ctrl.IsRunning && !m_ctrl.IsStopping;
            }

            btnMoveLeft.Visible = m_rgActionSpace.ContainsKey("MoveLeft");
            btnMoveRight.Visible = m_rgActionSpace.ContainsKey("MoveRight");
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            m_ctrl.Start();
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            m_ctrl.Stop();
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            m_ctrl.Reset();
        }

        private void btnMoveCartLeft_Click(object sender, EventArgs e)
        {
            m_ctrl.RunAction(m_rgActionSpace["MoveLeft"]);
        }

        private void btnMoveCartRight_Click(object sender, EventArgs e)
        {
            m_ctrl.RunAction(m_rgActionSpace["MoveRight"]);
        }

        private void FormGym_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason == CloseReason.WindowsShutDown)
                return;

            Hide();
            e.Cancel = true;
        }
    }
}
