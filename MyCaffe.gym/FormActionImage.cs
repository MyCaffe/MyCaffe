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
    public partial class FormActionImage : Form
    {
        public FormActionImage()
        {
            InitializeComponent();
        }

        private void FormActionImage_Load(object sender, EventArgs e)
        {

        }

        public void SetImage(Bitmap bmp)
        {
            pictureBox1.Image = bmp;
        }
    }
}
