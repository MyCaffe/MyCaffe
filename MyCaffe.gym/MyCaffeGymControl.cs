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
using System.IO;

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
        bool m_bEnableRecording = false;
        int m_nRecordingIndex = 0;
        string m_strRecordingFolder;

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeGymControl()
        {
            InitializeComponent();
            m_loadErrors = m_colGym.Load();
            m_strRecordingFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments) + "\\MyCaffe\\gym\\recordings";
        }

        /// <summary>
        /// Enable or disable recording.  When recording is enabled, each image is saved to the recording folder.
        /// </summary>
        /// <remarks>
        /// The recording folder is located in the My Documents folder under MyDocuments\MyCaffe\gym\recordings.
        /// </remarks>
        public bool EnableRecording
        {
            get { return m_bEnableRecording; }
            set { m_bEnableRecording = value; }
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

        private void record(Bitmap bmp)
        {
            if (!m_bEnableRecording)
                return;

            if (!System.IO.Directory.Exists(m_strRecordingFolder))
                System.IO.Directory.CreateDirectory(m_strRecordingFolder);

            string strFile = m_strRecordingFolder + "\\" + m_nRecordingIndex.ToString("000000") + ".png";
            bmp.Save(strFile, System.Drawing.Imaging.ImageFormat.Png);

            m_nRecordingIndex++;
        }

        /// <summary>
        /// Returns whether or not the Gym has any recording data.
        /// </summary>
        public bool HasRecordingData
        {
            get
            {
                if (Directory.Exists(m_strRecordingFolder))
                {
                    string[] rgstrFiles = Directory.GetFiles(m_strRecordingFolder, "*.png");
                    return (rgstrFiles.Length > 0);
                }

                return false;
            }
        }

        /// <summary>
        /// Delete any recording data that exists.
        /// </summary>
        public void DeleteRecordingData()
        {
            if (Directory.Exists(m_strRecordingFolder))
            {
                string[] rgstrFiles = Directory.GetFiles(m_strRecordingFolder, "*.png");

                foreach (string strFile in rgstrFiles)
                {
                    File.Delete(strFile);
                }

                m_nRecordingIndex = 0;
            }
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

            record(m_bmp);
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

            record(m_bmp);
        }

        private void MyCaffeGymControl_Paint(object sender, PaintEventArgs e)
        {
            if (m_bmp != null)
                e.Graphics.DrawImage(m_bmp, new Point(0, 0));
        }
    }
}
