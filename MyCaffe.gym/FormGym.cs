using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.gym
{
    public partial class FormGym : Form
    {
        string m_strName;
        MyCaffeGymControl m_ctrl;
        FormActionImage m_dlgActionImage;

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

        public string GymName
        {
            get { return m_strName; }
        }

        public void Render(Bitmap bmp, Bitmap bmpAction)
        {
            m_ctrl.Render(m_strName, bmp);

            if (bmp == null)
            {
                btnShowActionImage.Enabled = false;
            }
            else
            {
                btnShowActionImage.Enabled = true;
                m_dlgActionImage.SetImage(bmpAction);
            }
        }

        public void Render(double[] rgData, Bitmap bmp, Bitmap bmpAction)
        {
            m_ctrl.Render(true, m_strName, rgData, bmp);

            if (bmp == null)
                btnShowActionImage.Enabled = false;
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
    }
}
