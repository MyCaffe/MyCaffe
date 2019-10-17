namespace MyCaffe.app
{
    partial class FormVOC
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormVOC));
            this.lblDownload = new System.Windows.Forms.Label();
            this.lblDownloadSite = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.edtDataFile1 = new System.Windows.Forms.TextBox();
            this.btnBrowseTar1 = new System.Windows.Forms.Button();
            this.openFileDialogTar = new System.Windows.Forms.OpenFileDialog();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.timerUI = new System.Windows.Forms.Timer(this.components);
            this.label2 = new System.Windows.Forms.Label();
            this.edtDataFile2 = new System.Windows.Forms.TextBox();
            this.btnBrowseTar2 = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.edtDataFile3 = new System.Windows.Forms.TextBox();
            this.btnBrowseTar3 = new System.Windows.Forms.Button();
            this.btnDownload1 = new System.Windows.Forms.Button();
            this.btnDownload2 = new System.Windows.Forms.Button();
            this.btnDownload3 = new System.Windows.Forms.Button();
            this.lblDownloadPct1 = new System.Windows.Forms.Label();
            this.lblDownloadPct2 = new System.Windows.Forms.Label();
            this.lblDownloadPct3 = new System.Windows.Forms.Label();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.chkExtractFiles = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // lblDownload
            // 
            this.lblDownload.Location = new System.Drawing.Point(16, 13);
            this.lblDownload.Name = "lblDownload";
            this.lblDownload.Size = new System.Drawing.Size(443, 16);
            this.lblDownload.TabIndex = 0;
            this.lblDownload.Text = "If you have not already done so, download the .tar VOC files and expand the data " +
    "files from:";
            // 
            // lblDownloadSite
            // 
            this.lblDownloadSite.ForeColor = System.Drawing.Color.Blue;
            this.lblDownloadSite.Location = new System.Drawing.Point(456, 13);
            this.lblDownloadSite.Name = "lblDownloadSite";
            this.lblDownloadSite.Size = new System.Drawing.Size(234, 16);
            this.lblDownloadSite.TabIndex = 0;
            this.lblDownloadSite.Text = "host.robots.ox.ac.uk/pascal/VOC/";
            this.lblDownloadSite.Click += new System.EventHandler(this.lblDownloadSite_Click);
            this.lblDownloadSite.MouseLeave += new System.EventHandler(this.lblDownloadSite_MouseLeave);
            this.lblDownloadSite.MouseHover += new System.EventHandler(this.lblDownloadSite_MouseHover);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(31, 47);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(146, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "VOCtrainval_11-May-2012.tar";
            // 
            // edtDataFile1
            // 
            this.edtDataFile1.Location = new System.Drawing.Point(183, 44);
            this.edtDataFile1.Name = "edtDataFile1";
            this.edtDataFile1.Size = new System.Drawing.Size(472, 20);
            this.edtDataFile1.TabIndex = 2;
            // 
            // btnBrowseTar1
            // 
            this.btnBrowseTar1.Location = new System.Drawing.Point(661, 44);
            this.btnBrowseTar1.Name = "btnBrowseTar1";
            this.btnBrowseTar1.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseTar1.TabIndex = 3;
            this.btnBrowseTar1.Text = "...";
            this.btnBrowseTar1.UseVisualStyleBackColor = true;
            this.btnBrowseTar1.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // openFileDialogTar
            // 
            this.openFileDialogTar.Filter = "VOC Data Files (*.tar)|*.tar||";
            this.openFileDialogTar.Title = "Select the VOC data file ";
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(659, 129);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 4;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(740, 129);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 4;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            // 
            // timerUI
            // 
            this.timerUI.Enabled = true;
            this.timerUI.Interval = 250;
            this.timerUI.Tick += new System.EventHandler(this.timerUI_Tick);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(31, 73);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(146, 13);
            this.label2.TabIndex = 1;
            this.label2.Text = "VOCtrainval_06-Nov-2007.tar";
            // 
            // edtDataFile2
            // 
            this.edtDataFile2.Location = new System.Drawing.Point(183, 70);
            this.edtDataFile2.Name = "edtDataFile2";
            this.edtDataFile2.Size = new System.Drawing.Size(472, 20);
            this.edtDataFile2.TabIndex = 2;
            // 
            // btnBrowseTar2
            // 
            this.btnBrowseTar2.Location = new System.Drawing.Point(661, 70);
            this.btnBrowseTar2.Name = "btnBrowseTar2";
            this.btnBrowseTar2.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseTar2.TabIndex = 3;
            this.btnBrowseTar2.Text = "...";
            this.btnBrowseTar2.UseVisualStyleBackColor = true;
            this.btnBrowseTar2.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(31, 99);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(129, 13);
            this.label3.TabIndex = 1;
            this.label3.Text = "VOCtest_06-Nov-2007.tar";
            // 
            // edtDataFile3
            // 
            this.edtDataFile3.Location = new System.Drawing.Point(183, 96);
            this.edtDataFile3.Name = "edtDataFile3";
            this.edtDataFile3.Size = new System.Drawing.Size(472, 20);
            this.edtDataFile3.TabIndex = 2;
            // 
            // btnBrowseTar3
            // 
            this.btnBrowseTar3.Location = new System.Drawing.Point(661, 96);
            this.btnBrowseTar3.Name = "btnBrowseTar3";
            this.btnBrowseTar3.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseTar3.TabIndex = 3;
            this.btnBrowseTar3.Text = "...";
            this.btnBrowseTar3.UseVisualStyleBackColor = true;
            this.btnBrowseTar3.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // btnDownload1
            // 
            this.btnDownload1.Location = new System.Drawing.Point(696, 44);
            this.btnDownload1.Name = "btnDownload1";
            this.btnDownload1.Size = new System.Drawing.Size(69, 20);
            this.btnDownload1.TabIndex = 3;
            this.btnDownload1.Text = "download...";
            this.btnDownload1.UseVisualStyleBackColor = true;
            this.btnDownload1.Click += new System.EventHandler(this.btnDownload_Click);
            // 
            // btnDownload2
            // 
            this.btnDownload2.Location = new System.Drawing.Point(696, 69);
            this.btnDownload2.Name = "btnDownload2";
            this.btnDownload2.Size = new System.Drawing.Size(69, 20);
            this.btnDownload2.TabIndex = 3;
            this.btnDownload2.Text = "download...";
            this.btnDownload2.UseVisualStyleBackColor = true;
            this.btnDownload2.Click += new System.EventHandler(this.btnDownload_Click);
            // 
            // btnDownload3
            // 
            this.btnDownload3.Location = new System.Drawing.Point(696, 96);
            this.btnDownload3.Name = "btnDownload3";
            this.btnDownload3.Size = new System.Drawing.Size(69, 20);
            this.btnDownload3.TabIndex = 3;
            this.btnDownload3.Text = "download...";
            this.btnDownload3.UseVisualStyleBackColor = true;
            this.btnDownload3.Click += new System.EventHandler(this.btnDownload_Click);
            // 
            // lblDownloadPct1
            // 
            this.lblDownloadPct1.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDownloadPct1.Location = new System.Drawing.Point(771, 47);
            this.lblDownloadPct1.Name = "lblDownloadPct1";
            this.lblDownloadPct1.Size = new System.Drawing.Size(53, 17);
            this.lblDownloadPct1.TabIndex = 1;
            this.lblDownloadPct1.Text = "0.00 %";
            this.lblDownloadPct1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // lblDownloadPct2
            // 
            this.lblDownloadPct2.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDownloadPct2.Location = new System.Drawing.Point(771, 72);
            this.lblDownloadPct2.Name = "lblDownloadPct2";
            this.lblDownloadPct2.Size = new System.Drawing.Size(53, 17);
            this.lblDownloadPct2.TabIndex = 1;
            this.lblDownloadPct2.Text = "0.00 %";
            this.lblDownloadPct2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // lblDownloadPct3
            // 
            this.lblDownloadPct3.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDownloadPct3.Location = new System.Drawing.Point(771, 98);
            this.lblDownloadPct3.Name = "lblDownloadPct3";
            this.lblDownloadPct3.Size = new System.Drawing.Size(53, 17);
            this.lblDownloadPct3.TabIndex = 1;
            this.lblDownloadPct3.Text = "0.00 %";
            this.lblDownloadPct3.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // folderBrowserDialog1
            // 
            this.folderBrowserDialog1.Description = "Select the folder for the downloaded VOC files";
            this.folderBrowserDialog1.RootFolder = System.Environment.SpecialFolder.MyComputer;
            // 
            // chkExtractFiles
            // 
            this.chkExtractFiles.AutoSize = true;
            this.chkExtractFiles.Checked = true;
            this.chkExtractFiles.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkExtractFiles.Location = new System.Drawing.Point(183, 133);
            this.chkExtractFiles.Name = "chkExtractFiles";
            this.chkExtractFiles.Size = new System.Drawing.Size(396, 17);
            this.chkExtractFiles.TabIndex = 5;
            this.chkExtractFiles.Text = "Extract files (when unchecked, only extracted files are added to the database).";
            this.chkExtractFiles.UseVisualStyleBackColor = true;
            // 
            // FormVOC
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(827, 164);
            this.Controls.Add(this.chkExtractFiles);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.btnBrowseTar3);
            this.Controls.Add(this.btnBrowseTar2);
            this.Controls.Add(this.btnDownload3);
            this.Controls.Add(this.btnDownload2);
            this.Controls.Add(this.btnDownload1);
            this.Controls.Add(this.btnBrowseTar1);
            this.Controls.Add(this.edtDataFile3);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.edtDataFile2);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.edtDataFile1);
            this.Controls.Add(this.lblDownloadPct3);
            this.Controls.Add(this.lblDownloadPct2);
            this.Controls.Add(this.lblDownloadPct1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lblDownloadSite);
            this.Controls.Add(this.lblDownload);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormVOC";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "VOC 2007 and 2012 Data Files";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FormVOC_FormClosing);
            this.Load += new System.EventHandler(this.FormCiFar10_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblDownload;
        private System.Windows.Forms.Label lblDownloadSite;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox edtDataFile1;
        private System.Windows.Forms.Button btnBrowseTar1;
        private System.Windows.Forms.OpenFileDialog openFileDialogTar;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Timer timerUI;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox edtDataFile2;
        private System.Windows.Forms.Button btnBrowseTar2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox edtDataFile3;
        private System.Windows.Forms.Button btnBrowseTar3;
        private System.Windows.Forms.Button btnDownload1;
        private System.Windows.Forms.Button btnDownload2;
        private System.Windows.Forms.Button btnDownload3;
        private System.Windows.Forms.Label lblDownloadPct1;
        private System.Windows.Forms.Label lblDownloadPct2;
        private System.Windows.Forms.Label lblDownloadPct3;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.CheckBox chkExtractFiles;
    }
}