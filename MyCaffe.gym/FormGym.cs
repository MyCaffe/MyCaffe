using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.gym
{
    /// <summary>
    /// The FormGym displays the gym visualization.
    /// </summary>
    public partial class FormGym : Form
    {
        string m_strName;
        MyCaffeGymControl m_ctrl;
        FormActionImage m_dlgActionImage;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the Gym.</param>
        /// <param name="ctrl">Specifies the MyCaffeGymControl instance to use.</param>
        public FormGym(string strName, MyCaffeGymControl ctrl = null)
        {
            InitializeComponent();

            m_strName = strName;

            if (ctrl == null)
                ctrl = new MyCaffeGymControl();

            ctrl.Dock = DockStyle.Fill;
            toolStripContainer1.ContentPanel.Controls.Add(ctrl);
            m_ctrl = ctrl;
            m_dlgActionImage = new FormActionImage();
            m_dlgActionImage.FormClosing += dlgActionImage_FormClosing;
        }

        /// <summary>
        /// Returns the Gym name.
        /// </summary>
        public string GymName
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Renders the bitmap and action image (if exists).
        /// </summary>
        /// <param name="bmp">Specifies the gym visualization.</param>
        /// <param name="bmpAction">Optionally, specifies the action image.</param>
        public void Render(Image bmp, Image bmpAction)
        {
            m_ctrl.Render(m_strName, bmp);

            if (bmpAction == null)
            {
                btnShowActionImage.Enabled = false;
            }
            else
            {
                btnShowActionImage.Enabled = true;
                m_dlgActionImage.SetImage(bmpAction);
            }
        }

        /// <summary>
        /// Renders the bitmap and action image (if exists).
        /// </summary>
        /// <param name="rgData">Specifies the gym data used to render the visualization.</param>
        /// <param name="bmp">Specifies the gym visualization.</param>
        /// <param name="bmpAction">Optionally, specifies the action image.</param>
        public void Render(double[] rgData, Image bmp, Image bmpAction)
        {
            m_ctrl.Render(true, m_strName, rgData, bmp);

            if (bmpAction == null)
            {
                btnShowActionImage.Enabled = false;
            }
            else
            {
                btnShowActionImage.Enabled = true;
                m_dlgActionImage.SetImage(bmpAction);
            }
        }

        private void dlgActionImage_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason == CloseReason.WindowsShutDown)
                return;

            m_dlgActionImage.Hide();
            e.Cancel = true;
        }

        private void FormGym_Load(object sender, EventArgs e)
        {
            Text = "MyCaffe Test Gym - " + m_strName;
        }

        private void FormGym_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason == CloseReason.WindowsShutDown || e.CloseReason == CloseReason.ApplicationExitCall)
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

        private void btnRecord_Click(object sender, EventArgs e)
        {
            if (btnRecord.Checked)
            {
                btnRecord.Image = Properties.Resources.record_on;
                m_ctrl.EnableRecording = true;
            }
            else
            {
                btnRecord.Image = Properties.Resources.record_off;
                m_ctrl.EnableRecording = false;
            }
        }

        private void btnDeleteRecordingData_Click(object sender, EventArgs e)
        {
            m_ctrl.DeleteRecordingData();
        }

        private void timerUI_Tick(object sender, EventArgs e)
        {
            btnDeleteRecordingData.Enabled = m_ctrl.HasRecordingData;
        }
    }
}
