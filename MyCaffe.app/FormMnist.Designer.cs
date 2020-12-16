namespace MyCaffe.app
{
    partial class FormMnist
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormMnist));
            this.lblDownload = new System.Windows.Forms.Label();
            this.lblDownloadSite = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.edtTrainImagesFile = new System.Windows.Forms.TextBox();
            this.btnBrowseGz1 = new System.Windows.Forms.Button();
            this.openFileDialogGz = new System.Windows.Forms.OpenFileDialog();
            this.label2 = new System.Windows.Forms.Label();
            this.edtTrainLabelsFile = new System.Windows.Forms.TextBox();
            this.btnBrowseGz2 = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.edtTestImagesFile = new System.Windows.Forms.TextBox();
            this.btnBrowseGz3 = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.edtTestLabelsFile = new System.Windows.Forms.TextBox();
            this.btnBrowseGz4 = new System.Windows.Forms.Button();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.timerUI = new System.Windows.Forms.Timer(this.components);
            this.chkExportToFile = new System.Windows.Forms.CheckBox();
            this.label5 = new System.Windows.Forms.Label();
            this.edtExportFolder = new System.Windows.Forms.TextBox();
            this.btnBrowseFolder = new System.Windows.Forms.Button();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.lblDownloadPct = new System.Windows.Forms.Label();
            this.btnDownload = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // lblDownload
            // 
            this.lblDownload.Location = new System.Drawing.Point(16, 13);
            this.lblDownload.Name = "lblDownload";
            this.lblDownload.Size = new System.Drawing.Size(494, 16);
            this.lblDownload.TabIndex = 0;
            this.lblDownload.Text = "If you have not already done so, download the four .gz MNIST data files from:";
            // 
            // lblDownloadSite
            // 
            this.lblDownloadSite.ForeColor = System.Drawing.Color.Blue;
            this.lblDownloadSite.Location = new System.Drawing.Point(386, 13);
            this.lblDownloadSite.Name = "lblDownloadSite";
            this.lblDownloadSite.Size = new System.Drawing.Size(147, 16);
            this.lblDownloadSite.TabIndex = 1;
            this.lblDownloadSite.Text = "yann.lecun.com/exdb/mnist/";
            this.lblDownloadSite.Click += new System.EventHandler(this.lblDownloadSite_Click);
            this.lblDownloadSite.MouseLeave += new System.EventHandler(this.lblDownloadSite_MouseLeave);
            this.lblDownloadSite.MouseHover += new System.EventHandler(this.lblDownloadSite_MouseHover);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(19, 87);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(133, 13);
            this.label1.TabIndex = 8;
            this.label1.Text = "train-images-idx3-ubyte file:";
            // 
            // edtTrainImagesFile
            // 
            this.edtTrainImagesFile.Location = new System.Drawing.Point(158, 84);
            this.edtTrainImagesFile.Name = "edtTrainImagesFile";
            this.edtTrainImagesFile.Size = new System.Drawing.Size(461, 20);
            this.edtTrainImagesFile.TabIndex = 9;
            // 
            // btnBrowseGz1
            // 
            this.btnBrowseGz1.Location = new System.Drawing.Point(625, 84);
            this.btnBrowseGz1.Name = "btnBrowseGz1";
            this.btnBrowseGz1.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz1.TabIndex = 10;
            this.btnBrowseGz1.Text = "...";
            this.btnBrowseGz1.UseVisualStyleBackColor = true;
            this.btnBrowseGz1.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // openFileDialogGz
            // 
            this.openFileDialogGz.Filter = "MNIST Data Files (*.gz)|*.gz||";
            this.openFileDialogGz.Title = "Select the MNIST data file ";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(25, 113);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(127, 13);
            this.label2.TabIndex = 11;
            this.label2.Text = "train-labels-idx1-ubyte file:";
            // 
            // edtTrainLabelsFile
            // 
            this.edtTrainLabelsFile.Location = new System.Drawing.Point(158, 110);
            this.edtTrainLabelsFile.Name = "edtTrainLabelsFile";
            this.edtTrainLabelsFile.Size = new System.Drawing.Size(461, 20);
            this.edtTrainLabelsFile.TabIndex = 12;
            // 
            // btnBrowseGz2
            // 
            this.btnBrowseGz2.Location = new System.Drawing.Point(625, 110);
            this.btnBrowseGz2.Name = "btnBrowseGz2";
            this.btnBrowseGz2.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz2.TabIndex = 13;
            this.btnBrowseGz2.Text = "...";
            this.btnBrowseGz2.UseVisualStyleBackColor = true;
            this.btnBrowseGz2.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(19, 35);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(134, 13);
            this.label3.TabIndex = 2;
            this.label3.Text = "t10k-images-idx3-ubyte file:";
            // 
            // edtTestImagesFile
            // 
            this.edtTestImagesFile.Location = new System.Drawing.Point(158, 32);
            this.edtTestImagesFile.Name = "edtTestImagesFile";
            this.edtTestImagesFile.Size = new System.Drawing.Size(461, 20);
            this.edtTestImagesFile.TabIndex = 3;
            // 
            // btnBrowseGz3
            // 
            this.btnBrowseGz3.Location = new System.Drawing.Point(625, 32);
            this.btnBrowseGz3.Name = "btnBrowseGz3";
            this.btnBrowseGz3.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz3.TabIndex = 4;
            this.btnBrowseGz3.Text = "...";
            this.btnBrowseGz3.UseVisualStyleBackColor = true;
            this.btnBrowseGz3.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(24, 61);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(128, 13);
            this.label4.TabIndex = 5;
            this.label4.Text = "t10k-labels-idx1-ubyte file:";
            // 
            // edtTestLabelsFile
            // 
            this.edtTestLabelsFile.Location = new System.Drawing.Point(158, 58);
            this.edtTestLabelsFile.Name = "edtTestLabelsFile";
            this.edtTestLabelsFile.Size = new System.Drawing.Size(461, 20);
            this.edtTestLabelsFile.TabIndex = 6;
            // 
            // btnBrowseGz4
            // 
            this.btnBrowseGz4.Location = new System.Drawing.Point(625, 58);
            this.btnBrowseGz4.Name = "btnBrowseGz4";
            this.btnBrowseGz4.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz4.TabIndex = 7;
            this.btnBrowseGz4.Text = "...";
            this.btnBrowseGz4.UseVisualStyleBackColor = true;
            this.btnBrowseGz4.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(498, 190);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 14;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(579, 190);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 15;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            // 
            // timerUI
            // 
            this.timerUI.Enabled = true;
            this.timerUI.Interval = 250;
            this.timerUI.Tick += new System.EventHandler(this.timerUI_Tick);
            // 
            // chkExportToFile
            // 
            this.chkExportToFile.AutoSize = true;
            this.chkExportToFile.Location = new System.Drawing.Point(158, 136);
            this.chkExportToFile.Name = "chkExportToFile";
            this.chkExportToFile.Size = new System.Drawing.Size(193, 17);
            this.chkExportToFile.TabIndex = 16;
            this.chkExportToFile.Text = "Export to file only (SQL not needed)";
            this.chkExportToFile.UseVisualStyleBackColor = true;
            this.chkExportToFile.CheckedChanged += new System.EventHandler(this.chkExportToFile_CheckedChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(80, 163);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(72, 13);
            this.label5.TabIndex = 11;
            this.label5.Text = "Export Folder:";
            // 
            // edtExportFolder
            // 
            this.edtExportFolder.Enabled = false;
            this.edtExportFolder.Location = new System.Drawing.Point(158, 159);
            this.edtExportFolder.Name = "edtExportFolder";
            this.edtExportFolder.Size = new System.Drawing.Size(461, 20);
            this.edtExportFolder.TabIndex = 12;
            this.edtExportFolder.Text = "\\ProgramData\\MyCaffe\\test_data\\mnist";
            // 
            // btnBrowseFolder
            // 
            this.btnBrowseFolder.Location = new System.Drawing.Point(625, 159);
            this.btnBrowseFolder.Name = "btnBrowseFolder";
            this.btnBrowseFolder.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseFolder.TabIndex = 13;
            this.btnBrowseFolder.Text = "...";
            this.btnBrowseFolder.UseVisualStyleBackColor = true;
            this.btnBrowseFolder.Click += new System.EventHandler(this.btnBrowseFolder_Click);
            // 
            // folderBrowserDialog1
            // 
            this.folderBrowserDialog1.Description = "Select the export folder.";
            this.folderBrowserDialog1.RootFolder = System.Environment.SpecialFolder.MyComputer;
            this.folderBrowserDialog1.SelectedPath = "\\ProgramData\\MyCaffe\\test_data";
            // 
            // lblDownloadPct
            // 
            this.lblDownloadPct.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDownloadPct.Location = new System.Drawing.Point(239, 194);
            this.lblDownloadPct.Name = "lblDownloadPct";
            this.lblDownloadPct.Size = new System.Drawing.Size(53, 17);
            this.lblDownloadPct.TabIndex = 25;
            this.lblDownloadPct.Text = "0.00 %";
            this.lblDownloadPct.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // btnDownload
            // 
            this.btnDownload.Location = new System.Drawing.Point(158, 190);
            this.btnDownload.Name = "btnDownload";
            this.btnDownload.Size = new System.Drawing.Size(75, 23);
            this.btnDownload.TabIndex = 24;
            this.btnDownload.Text = "Download";
            this.btnDownload.UseVisualStyleBackColor = true;
            this.btnDownload.Click += new System.EventHandler(this.btnDownload_Click);
            // 
            // FormMnist
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(666, 225);
            this.Controls.Add(this.lblDownloadPct);
            this.Controls.Add(this.btnDownload);
            this.Controls.Add(this.chkExportToFile);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.btnBrowseGz4);
            this.Controls.Add(this.btnBrowseGz3);
            this.Controls.Add(this.btnBrowseFolder);
            this.Controls.Add(this.btnBrowseGz2);
            this.Controls.Add(this.btnBrowseGz1);
            this.Controls.Add(this.edtTestLabelsFile);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.edtTestImagesFile);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.edtExportFolder);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.edtTrainLabelsFile);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.edtTrainImagesFile);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lblDownloadSite);
            this.Controls.Add(this.lblDownload);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormMnist";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "MNIST Data Files";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FormMnist_FormClosing);
            this.Load += new System.EventHandler(this.FormMnist_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblDownload;
        private System.Windows.Forms.Label lblDownloadSite;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox edtTrainImagesFile;
        private System.Windows.Forms.Button btnBrowseGz1;
        private System.Windows.Forms.OpenFileDialog openFileDialogGz;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox edtTrainLabelsFile;
        private System.Windows.Forms.Button btnBrowseGz2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox edtTestImagesFile;
        private System.Windows.Forms.Button btnBrowseGz3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox edtTestLabelsFile;
        private System.Windows.Forms.Button btnBrowseGz4;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Timer timerUI;
        private System.Windows.Forms.CheckBox chkExportToFile;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox edtExportFolder;
        private System.Windows.Forms.Button btnBrowseFolder;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.Label lblDownloadPct;
        private System.Windows.Forms.Button btnDownload;
    }
}