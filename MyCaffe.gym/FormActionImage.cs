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
    /// <summary>
    /// The FormActionImage displays the action image (if one exists)
    /// </summary>
    public partial class FormActionImage : Form
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public FormActionImage()
        {
            InitializeComponent();
        }

        private void FormActionImage_Load(object sender, EventArgs e)
        {

        }

        /// <summary>
        /// Set the image to display.
        /// </summary>
        /// <param name="bmp"></param>
        public void SetImage(Image bmp)
        {
            pictureBox1.Image = bmp;
        }
    }
}
