namespace MyCaffe.app
{
    partial class FormTestImage
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormTestImage));
            this.SuspendLayout();
            // 
            // FormTestImage
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(544, 544);
            this.DoubleBuffered = true;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormTestImage";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Draw Test Image";
            this.Load += new System.EventHandler(this.FormTestImage_Load);
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.FormTestImage_Paint);
            this.MouseClick += new System.Windows.Forms.MouseEventHandler(this.FormTestImage_MouseClick);
            this.MouseMove += new System.Windows.Forms.MouseEventHandler(this.FormTestImage_MouseMove);
            this.MouseUp += new System.Windows.Forms.MouseEventHandler(this.FormTestImage_MouseUp);
            this.ResumeLayout(false);

        }

        #endregion
    }
}