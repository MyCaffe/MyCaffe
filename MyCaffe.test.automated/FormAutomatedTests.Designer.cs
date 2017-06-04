namespace MyCaffe.test.automated
{
    partial class FormAutomatedTests
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.automatedTester1 = new MyCaffe.test.automated.AutomatedTester();
            this.SuspendLayout();
            // 
            // automatedTester1
            // 
            this.automatedTester1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.automatedTester1.Location = new System.Drawing.Point(0, 0);
            this.automatedTester1.Name = "automatedTester1";
            this.automatedTester1.Size = new System.Drawing.Size(913, 685);
            this.automatedTester1.TabIndex = 0;
            this.automatedTester1.TestAssemblyPath = null;
            // 
            // FormAutomatedTests
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(913, 685);
            this.Controls.Add(this.automatedTester1);
            this.MinimizeBox = false;
            this.Name = "FormAutomatedTests";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Automated Tests";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FormAutomatedTests_FormClosing);
            this.Load += new System.EventHandler(this.FormAutomatedTests_Load);
            this.ResumeLayout(false);

        }

        #endregion

        private AutomatedTester automatedTester1;
    }
}