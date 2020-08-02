namespace MyCaffe.app
{
    partial class FormSaveImage
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormSaveImage));
            this.label1 = new System.Windows.Forms.Label();
            this.edtFolder = new System.Windows.Forms.TextBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.edtH = new System.Windows.Forms.TextBox();
            this.edtW = new System.Windows.Forms.TextBox();
            this.lblX = new System.Windows.Forms.Label();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.folderBrowserDialogTestImages = new System.Windows.Forms.FolderBrowserDialog();
            this.chkEnableSavingTestImages = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(23, 19);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(52, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Directory:";
            // 
            // edtFolder
            // 
            this.edtFolder.Location = new System.Drawing.Point(81, 16);
            this.edtFolder.Name = "edtFolder";
            this.edtFolder.Size = new System.Drawing.Size(724, 20);
            this.edtFolder.TabIndex = 1;
            // 
            // btnBrowse
            // 
            this.btnBrowse.Location = new System.Drawing.Point(811, 16);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(29, 20);
            this.btnBrowse.TabIndex = 2;
            this.btnBrowse.Text = "...";
            this.btnBrowse.UseVisualStyleBackColor = true;
            this.btnBrowse.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // edtH
            // 
            this.edtH.Location = new System.Drawing.Point(81, 42);
            this.edtH.Name = "edtH";
            this.edtH.Size = new System.Drawing.Size(43, 20);
            this.edtH.TabIndex = 4;
            this.edtH.Text = "28";
            this.edtH.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // edtW
            // 
            this.edtW.Location = new System.Drawing.Point(148, 42);
            this.edtW.Name = "edtW";
            this.edtW.Size = new System.Drawing.Size(43, 20);
            this.edtW.TabIndex = 6;
            this.edtW.Text = "28";
            this.edtW.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lblX
            // 
            this.lblX.AutoSize = true;
            this.lblX.Location = new System.Drawing.Point(130, 45);
            this.lblX.Name = "lblX";
            this.lblX.Size = new System.Drawing.Size(12, 13);
            this.lblX.TabIndex = 5;
            this.lblX.Text = "x";
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(684, 66);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 9;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(765, 66);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 10;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.ForeColor = System.Drawing.SystemColors.ActiveCaption;
            this.label3.Location = new System.Drawing.Point(208, 45);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 13);
            this.label3.TabIndex = 11;
            this.label3.Text = "(valid range [2, 512])";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(13, 45);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(62, 13);
            this.label4.TabIndex = 7;
            this.label4.Text = "Image Size:";
            // 
            // folderBrowserDialogTestImages
            // 
            this.folderBrowserDialogTestImages.Description = "Select test image output folder.";
            this.folderBrowserDialogTestImages.RootFolder = System.Environment.SpecialFolder.MyComputer;
            // 
            // chkEnableSavingTestImages
            // 
            this.chkEnableSavingTestImages.AutoSize = true;
            this.chkEnableSavingTestImages.Checked = true;
            this.chkEnableSavingTestImages.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkEnableSavingTestImages.Location = new System.Drawing.Point(81, 72);
            this.chkEnableSavingTestImages.Name = "chkEnableSavingTestImages";
            this.chkEnableSavingTestImages.Size = new System.Drawing.Size(149, 17);
            this.chkEnableSavingTestImages.TabIndex = 12;
            this.chkEnableSavingTestImages.Text = "Enable saving test images";
            this.chkEnableSavingTestImages.UseVisualStyleBackColor = true;
            // 
            // FormSaveImage
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(852, 101);
            this.Controls.Add(this.chkEnableSavingTestImages);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.lblX);
            this.Controls.Add(this.edtW);
            this.Controls.Add(this.edtH);
            this.Controls.Add(this.btnBrowse);
            this.Controls.Add(this.edtFolder);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormSaveImage";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Save Test Image Settings";
            this.Load += new System.EventHandler(this.FormSaveImage_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox edtFolder;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.TextBox edtH;
        private System.Windows.Forms.TextBox edtW;
        private System.Windows.Forms.Label lblX;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialogTestImages;
        private System.Windows.Forms.CheckBox chkEnableSavingTestImages;
    }
}