namespace MyCaffe.app
{
    partial class FormAbout
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormAbout));
            this.edtLicense = new System.Windows.Forms.TextBox();
            this.lblLicense = new System.Windows.Forms.Label();
            this.lblWebUrl = new System.Windows.Forms.Label();
            this.lblDescription = new System.Windows.Forms.Label();
            this.lblVersion = new System.Windows.Forms.Label();
            this.lblProduct = new System.Windows.Forms.Label();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.lblBroughtToYou = new System.Windows.Forms.Label();
            this.lblMyCaffeUrl = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // edtLicense
            // 
            this.edtLicense.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.edtLicense.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.edtLicense.Location = new System.Drawing.Point(15, 309);
            this.edtLicense.Multiline = true;
            this.edtLicense.Name = "edtLicense";
            this.edtLicense.ReadOnly = true;
            this.edtLicense.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.edtLicense.Size = new System.Drawing.Size(586, 263);
            this.edtLicense.TabIndex = 12;
            // 
            // lblLicense
            // 
            this.lblLicense.Font = new System.Drawing.Font("Corbel", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblLicense.Location = new System.Drawing.Point(12, 288);
            this.lblLicense.Name = "lblLicense";
            this.lblLicense.Size = new System.Drawing.Size(458, 18);
            this.lblLicense.TabIndex = 4;
            this.lblLicense.Text = "License:";
            // 
            // lblWebUrl
            // 
            this.lblWebUrl.Font = new System.Drawing.Font("Century Gothic", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblWebUrl.ForeColor = System.Drawing.Color.CornflowerBlue;
            this.lblWebUrl.Location = new System.Drawing.Point(372, 262);
            this.lblWebUrl.Name = "lblWebUrl";
            this.lblWebUrl.Size = new System.Drawing.Size(134, 17);
            this.lblWebUrl.TabIndex = 5;
            this.lblWebUrl.Text = "www.signalpop.com";
            this.lblWebUrl.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.lblWebUrl.Click += new System.EventHandler(this.lblWebUrl_Click);
            this.lblWebUrl.MouseLeave += new System.EventHandler(this.lblWebUrl_MouseLeave);
            this.lblWebUrl.MouseHover += new System.EventHandler(this.lblWebUrl_MouseHover);
            // 
            // lblDescription
            // 
            this.lblDescription.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblDescription.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDescription.Location = new System.Drawing.Point(259, 163);
            this.lblDescription.Name = "lblDescription";
            this.lblDescription.Size = new System.Drawing.Size(344, 71);
            this.lblDescription.TabIndex = 7;
            this.lblDescription.Text = "MyCaffe is a complete re-write of the open-source Caffe project in C# for Windows" +
    " .NET programmers.";
            // 
            // lblVersion
            // 
            this.lblVersion.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblVersion.Location = new System.Drawing.Point(263, 53);
            this.lblVersion.Name = "lblVersion";
            this.lblVersion.Size = new System.Drawing.Size(342, 15);
            this.lblVersion.TabIndex = 9;
            // 
            // lblProduct
            // 
            this.lblProduct.Font = new System.Drawing.Font("Century Gothic", 18F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblProduct.Location = new System.Drawing.Point(263, 19);
            this.lblProduct.Name = "lblProduct";
            this.lblProduct.Size = new System.Drawing.Size(338, 34);
            this.lblProduct.TabIndex = 11;
            this.lblProduct.Text = "Product:";
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox1.Image")));
            this.pictureBox1.Location = new System.Drawing.Point(12, 12);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(220, 275);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 3;
            this.pictureBox1.TabStop = false;
            // 
            // lblBroughtToYou
            // 
            this.lblBroughtToYou.Font = new System.Drawing.Font("Century Gothic", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblBroughtToYou.ForeColor = System.Drawing.Color.SlateGray;
            this.lblBroughtToYou.Location = new System.Drawing.Point(261, 262);
            this.lblBroughtToYou.Name = "lblBroughtToYou";
            this.lblBroughtToYou.Size = new System.Drawing.Size(112, 17);
            this.lblBroughtToYou.TabIndex = 5;
            this.lblBroughtToYou.Text = "brought to you by";
            this.lblBroughtToYou.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.lblBroughtToYou.Click += new System.EventHandler(this.lblWebUrl_Click);
            this.lblBroughtToYou.MouseLeave += new System.EventHandler(this.lblWebUrl_MouseLeave);
            this.lblBroughtToYou.MouseHover += new System.EventHandler(this.lblWebUrl_MouseHover);
            // 
            // lblMyCaffeUrl
            // 
            this.lblMyCaffeUrl.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblMyCaffeUrl.Font = new System.Drawing.Font("Century Gothic", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblMyCaffeUrl.ForeColor = System.Drawing.Color.CornflowerBlue;
            this.lblMyCaffeUrl.Location = new System.Drawing.Point(260, 235);
            this.lblMyCaffeUrl.Name = "lblMyCaffeUrl";
            this.lblMyCaffeUrl.Size = new System.Drawing.Size(343, 27);
            this.lblMyCaffeUrl.TabIndex = 7;
            this.lblMyCaffeUrl.Text = "http://www.mycaffe.org";
            this.lblMyCaffeUrl.Click += new System.EventHandler(this.lblMyCaffeUrl_Click);
            this.lblMyCaffeUrl.MouseLeave += new System.EventHandler(this.lblMyCaffeUrl_MouseLeave);
            this.lblMyCaffeUrl.MouseHover += new System.EventHandler(this.lblMyCaffeUrl_MouseHover);
            // 
            // FormAbout
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(615, 584);
            this.Controls.Add(this.edtLicense);
            this.Controls.Add(this.lblLicense);
            this.Controls.Add(this.lblBroughtToYou);
            this.Controls.Add(this.lblWebUrl);
            this.Controls.Add(this.lblMyCaffeUrl);
            this.Controls.Add(this.lblDescription);
            this.Controls.Add(this.lblVersion);
            this.Controls.Add(this.lblProduct);
            this.Controls.Add(this.pictureBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "FormAbout";
            this.Text = "About MyCaffe";
            this.Load += new System.EventHandler(this.FormAbout_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox edtLicense;
        private System.Windows.Forms.Label lblLicense;
        private System.Windows.Forms.Label lblWebUrl;
        private System.Windows.Forms.Label lblDescription;
        private System.Windows.Forms.Label lblVersion;
        private System.Windows.Forms.Label lblProduct;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Label lblBroughtToYou;
        private System.Windows.Forms.Label lblMyCaffeUrl;
    }
}