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
        FormActionImage m_dlgActionImage;

        public FormGym(MyCaffeGymControl ctrl)
        {
            InitializeComponent();
            ctrl.Dock = DockStyle.Fill;
            toolStripContainer1.ContentPanel.Controls.Add(ctrl);
            m_ctrl = ctrl;
            m_ctrl.OnObservation += ctrl_OnObservation;
            m_rgActionSpace = ctrl.GetActionSpace();
            m_dlgActionImage = new FormActionImage();
            m_dlgActionImage.FormClosing += dlgActionImage_FormClosing;
        }

        private void ctrl_OnObservation(object sender, ObservationArgs e)
        {
            m_dlgActionImage.SetImage(e.Observation.Image);
        }

        private void dlgActionImage_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason == CloseReason.WindowsShutDown)
                return;

            m_dlgActionImage.Hide();
            e.Cancel = true;
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

        private void btnShowActionImage_Click(object sender, EventArgs e)
        {
            if (btnShowActionImage.Checked)
            {
                if (m_dlgActionImage.Visible)
                    m_dlgActionImage.BringToFront();
                else
                    m_dlgActionImage.Show(this);
            }
            else
            {
                if (m_dlgActionImage.Visible)
                    m_dlgActionImage.Hide();
            }
        }
    }
}
