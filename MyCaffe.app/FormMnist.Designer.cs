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
            this.lblDownloadSite.TabIndex = 0;
            this.lblDownloadSite.Text = "yann.lecun.com/exdb/mnist/";
            this.lblDownloadSite.Click += new System.EventHandler(this.lblDownloadSite_Click);
            this.lblDownloadSite.MouseLeave += new System.EventHandler(this.lblDownloadSite_MouseLeave);
            this.lblDownloadSite.MouseHover += new System.EventHandler(this.lblDownloadSite_MouseHover);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(19, 47);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(133, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "train-images-idx3-ubyte file:";
            // 
            // edtTrainImagesFile
            // 
            this.edtTrainImagesFile.Location = new System.Drawing.Point(158, 44);
            this.edtTrainImagesFile.Name = "edtTrainImagesFile";
            this.edtTrainImagesFile.Size = new System.Drawing.Size(461, 20);
            this.edtTrainImagesFile.TabIndex = 2;
            // 
            // btnBrowseGz1
            // 
            this.btnBrowseGz1.Location = new System.Drawing.Point(625, 44);
            this.btnBrowseGz1.Name = "btnBrowseGz1";
            this.btnBrowseGz1.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz1.TabIndex = 3;
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
            this.label2.Location = new System.Drawing.Point(25, 73);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(127, 13);
            this.label2.TabIndex = 1;
            this.label2.Text = "train-labels-idx1-ubyte file:";
            // 
            // edtTrainLabelsFile
            // 
            this.edtTrainLabelsFile.Location = new System.Drawing.Point(158, 70);
            this.edtTrainLabelsFile.Name = "edtTrainLabelsFile";
            this.edtTrainLabelsFile.Size = new System.Drawing.Size(461, 20);
            this.edtTrainLabelsFile.TabIndex = 2;
            // 
            // btnBrowseGz2
            // 
            this.btnBrowseGz2.Location = new System.Drawing.Point(625, 70);
            this.btnBrowseGz2.Name = "btnBrowseGz2";
            this.btnBrowseGz2.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz2.TabIndex = 3;
            this.btnBrowseGz2.Text = "...";
            this.btnBrowseGz2.UseVisualStyleBackColor = true;
            this.btnBrowseGz2.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(19, 99);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(134, 13);
            this.label3.TabIndex = 1;
            this.label3.Text = "t10k-images-idx3-ubyte file:";
            // 
            // edtTestImagesFile
            // 
            this.edtTestImagesFile.Location = new System.Drawing.Point(158, 96);
            this.edtTestImagesFile.Name = "edtTestImagesFile";
            this.edtTestImagesFile.Size = new System.Drawing.Size(461, 20);
            this.edtTestImagesFile.TabIndex = 2;
            // 
            // btnBrowseGz3
            // 
            this.btnBrowseGz3.Location = new System.Drawing.Point(625, 96);
            this.btnBrowseGz3.Name = "btnBrowseGz3";
            this.btnBrowseGz3.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz3.TabIndex = 3;
            this.btnBrowseGz3.Text = "...";
            this.btnBrowseGz3.UseVisualStyleBackColor = true;
            this.btnBrowseGz3.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(24, 125);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(128, 13);
            this.label4.TabIndex = 1;
            this.label4.Text = "t10k-labels-idx1-ubyte file:";
            // 
            // edtTestLabelsFile
            // 
            this.edtTestLabelsFile.Location = new System.Drawing.Point(158, 122);
            this.edtTestLabelsFile.Name = "edtTestLabelsFile";
            this.edtTestLabelsFile.Size = new System.Drawing.Size(461, 20);
            this.edtTestLabelsFile.TabIndex = 2;
            // 
            // btnBrowseGz4
            // 
            this.btnBrowseGz4.Location = new System.Drawing.Point(625, 122);
            this.btnBrowseGz4.Name = "btnBrowseGz4";
            this.btnBrowseGz4.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseGz4.TabIndex = 3;
            this.btnBrowseGz4.Text = "...";
            this.btnBrowseGz4.UseVisualStyleBackColor = true;
            this.btnBrowseGz4.Click += new System.EventHandler(this.btnBrowseGz_Click);
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(498, 155);
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
            this.btnCancel.Location = new System.Drawing.Point(579, 155);
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
            // FormMnist
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(666, 190);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.btnBrowseGz4);
            this.Controls.Add(this.btnBrowseGz3);
            this.Controls.Add(this.btnBrowseGz2);
            this.Controls.Add(this.btnBrowseGz1);
            this.Controls.Add(this.edtTestLabelsFile);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.edtTestImagesFile);
            this.Controls.Add(this.label3);
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
    }
}