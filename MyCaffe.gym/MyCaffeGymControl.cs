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

/// <summary>
/// The MyCaffe.gym namespace contains all classes related to the Gym's supported by MyCaffe.
/// </summary>
namespace MyCaffe.gym
{
    /// <summary>
    /// The MyCaffeGymControl displays the actual Gym visualizations.
    /// </summary>
    public partial class MyCaffeGymControl : UserControl
    {
        string m_strName = "";
        Bitmap m_bmp = null;
        GymCollection m_colGym = new GymCollection();
        List<Exception> m_loadErrors;

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeGymControl()
        {
            InitializeComponent();
            m_loadErrors = m_colGym.Load();
        }

        /// <summary>
        /// Returns any load errors that may have occured while loading the gyms.
        /// </summary>
        public List<Exception> LoadErrors
        {
            get { return m_loadErrors; }
        }

        /// <summary>
        /// Returns the GymName.
        /// </summary>
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

        /// <summary>
        /// Renders the Gym visualization.
        /// </summary>
        /// <param name="strName">Specifies the Gym Name.</param>
        /// <param name="bmp">Specifies the Gym image to visualize.</param>
        public void Render(string strName, Image bmp)
        {
            m_strName = strName;
            m_bmp = new Bitmap(bmp);

            if (IsHandleCreated && Visible)
                Invalidate(true);
        }

        /// <summary>
        /// Renders the Gym visualizations.
        /// </summary>
        /// <param name="bShowUi">Specifies whether or not to render for the user interface.</param>
        /// <param name="strName">Specifies the Gym name.</param>
        /// <param name="rgData">Specifies the Gym data.</param>
        /// <param name="bmp">Specifies the Gym image to use.</param>
        public void Render(bool bShowUi, string strName, double[] rgData, Image bmp)
        {
            m_strName = strName;

            IXMyCaffeGym igym = m_colGym.Find(strName);

            if (bmp != null)
            {
                m_bmp = new Bitmap(bmp);
            }
            else
            {
                Tuple<Bitmap, SimpleDatum> data = igym.Render(bShowUi, Width, Height, rgData, false);
                if (data != null)
                    m_bmp = data.Item1;
            }

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
