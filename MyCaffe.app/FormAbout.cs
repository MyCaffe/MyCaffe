using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormAbout : Form
    {
        public FormAbout()
        {
            InitializeComponent();
        }

        private void lblWebUrl_Click(object sender, EventArgs e)
        {
            Process p = new Process();

            p.StartInfo = new ProcessStartInfo("http://www.signalpop.com");
            p.Start();
        }

        private void lblWebUrl_MouseHover(object sender, EventArgs e)
        {
            lblWebUrl.ForeColor = Color.Blue;
        }

        private void lblWebUrl_MouseLeave(object sender, EventArgs e)
        {
            lblWebUrl.ForeColor = Color.CornflowerBlue;
        }

        private void FormAbout_Load(object sender, EventArgs e)
        {
            this.lblProduct.Text = AssemblyProduct;
            this.lblVersion.Text = String.Format("Version {0}", AssemblyVersion);
            edtLicense.Text = MyCaffeControl<float>.GetLicenseTextEx(Properties.Resources.ALE_license_cain);
            edtLicense.SelectionLength = 0;
            edtLicense.SelectionStart = 0;
            edtLicense.ScrollToCaret();
        }

        public string AssemblyVersion
        {
            get
            {
                return Assembly.GetExecutingAssembly().GetName().Version.ToString();
            }
        }

        public string AssemblyProduct
        {
            get
            {
                object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyProductAttribute), false);
                if (attributes.Length == 0)
                {
                    return "";
                }
                return ((AssemblyProductAttribute)attributes[0]).Product;
            }
        }
    }
}
