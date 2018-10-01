using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;
using MyCaffe.basecode;
using System.Collections;
using MyCaffe.basecode.descriptors;
using System.Diagnostics;

namespace MyCaffe.gym
{
    public partial class MyCaffeGymControl : UserControl
    {
        string m_strName = "";
        Bitmap m_bmp = null;
        GymCollection m_colGym = new GymCollection();

        public MyCaffeGymControl()
        {
            InitializeComponent();
            m_colGym.Load();
        }

        public string GymName
        {
            get { return m_strName; }
        }

        private void MyCaffeGymControl_Resize(object sender, EventArgs e)
        {
        }

        private void MyCaffeGymControl_Load(object sender, EventArgs e)
        {
        }

        public void Render(string strName, Image bmp)
        {
            m_strName = strName;
            m_bmp = new Bitmap(bmp);

            if (IsHandleCreated && Visible)
                Invalidate(true);
        }

        public void Render(bool bShowUi, string strName, double[] rgData, Image bmp)
        {
            m_strName = strName;

            IXMyCaffeGym igym = m_colGym.Find(strName);

            if (bmp != null)
                m_bmp = new Bitmap(bmp);
            else
                m_bmp = igym.Render(bShowUi, Width, Height, rgData, false).Item1;

            if (IsHandleCreated && Visible)
                Invalidate(true);
        }

        private void MyCaffeGymControl_Paint(object sender, PaintEventArgs e)
        {
            if (m_bmp != null)
                e.Graphics.DrawImage(m_bmp, new Point(0, 0));
        }
    }
}
