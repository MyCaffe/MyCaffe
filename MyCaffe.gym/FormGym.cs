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
        FormActionImage m_dlgActionImage;

        public FormGym(MyCaffeGymControl ctrl)
        {
            InitializeComponent();
            ctrl.Dock = DockStyle.Fill;
            toolStripContainer1.ContentPanel.Controls.Add(ctrl);
            m_ctrl = ctrl;
            m_dlgActionImage = new FormActionImage();
            m_dlgActionImage.FormClosing += dlgActionImage_FormClosing;
        }

        public void Render(string strName, Bitmap bmp, Bitmap bmpAction)
        {
            m_ctrl.Render(strName, bmp);
            m_dlgActionImage.SetImage(bmpAction);
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
