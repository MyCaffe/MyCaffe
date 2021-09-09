namespace MyCaffe.app
{
    partial class FormCifar10
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormCifar10));
            this.lblDownload = new System.Windows.Forms.Label();
            this.lblDownloadSite = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.edtCifarDataFile1 = new System.Windows.Forms.TextBox();
            this.btnBrowseBin1 = new System.Windows.Forms.Button();
            this.openFileDialogBin = new System.Windows.Forms.OpenFileDialog();
            this.btnOK = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.timerUI = new System.Windows.Forms.Timer(this.components);
            this.label2 = new System.Windows.Forms.Label();
            this.edtCifarDataFile2 = new System.Windows.Forms.TextBox();
            this.btnBrowseBin2 = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.edtCifarDataFile3 = new System.Windows.Forms.TextBox();
            this.btnBrowseBin3 = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.edtCifarDataFile4 = new System.Windows.Forms.TextBox();
            this.btnBrowseBin4 = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.edtCifarDataFile5 = new System.Windows.Forms.TextBox();
            this.btnBrowseBin5 = new System.Windows.Forms.Button();
            this.label6 = new System.Windows.Forms.Label();
            this.edtCifarDataFile6 = new System.Windows.Forms.TextBox();
            this.btnBrowseBin6 = new System.Windows.Forms.Button();
            this.btnDownload = new System.Windows.Forms.Button();
            this.lblDownloadPct = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // lblDownload
            // 
            this.lblDownload.Location = new System.Drawing.Point(16, 13);
            this.lblDownload.Name = "lblDownload";
            this.lblDownload.Size = new System.Drawing.Size(494, 16);
            this.lblDownload.TabIndex = 0;
            this.lblDownload.Text = "If you have not already done so, download the .gz CIFAR-10 and expand the data fi" +
    "les from:";
            // 
            // lblDownloadSite
            // 
            this.lblDownloadSite.ForeColor = System.Drawing.Color.Blue;
            this.lblDownloadSite.Location = new System.Drawing.Point(456, 13);
            this.lblDownloadSite.Name = "lblDownloadSite";
            this.lblDownloadSite.Size = new System.Drawing.Size(234, 16);
            this.lblDownloadSite.TabIndex = 1;
            this.lblDownloadSite.Text = "www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
            this.lblDownloadSite.Click += new System.EventHandler(this.lblDownloadSite_Click);
            this.lblDownloadSite.MouseLeave += new System.EventHandler(this.lblDownloadSite_MouseLeave);
            this.lblDownloadSite.MouseHover += new System.EventHandler(this.lblDownloadSite_MouseHover);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(31, 47);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(90, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "data_batch_1.bin";
            // 
            // edtCifarDataFile1
            // 
            this.edtCifarDataFile1.Location = new System.Drawing.Point(127, 44);
            this.edtCifarDataFile1.Name = "edtCifarDataFile1";
            this.edtCifarDataFile1.Size = new System.Drawing.Size(528, 20);
            this.edtCifarDataFile1.TabIndex = 3;
            // 
            // btnBrowseBin1
            // 
            this.btnBrowseBin1.Location = new System.Drawing.Point(661, 44);
            this.btnBrowseBin1.Name = "btnBrowseBin1";
            this.btnBrowseBin1.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseBin1.TabIndex = 4;
            this.btnBrowseBin1.Text = "...";
            this.btnBrowseBin1.UseVisualStyleBackColor = true;
            this.btnBrowseBin1.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // openFileDialogBin
            // 
            this.openFileDialogBin.Filter = "CIFAR-10 Data Files (*.bin)|*.bin||";
            this.openFileDialogBin.Title = "Select the CIFAR-10 data file ";
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(534, 208);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 21;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(615, 208);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 22;
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
            this.label2.Size = new System.Drawing.Size(90, 13);
            this.label2.TabIndex = 5;
            this.label2.Text = "data_batch_2.bin";
            // 
            // edtCifarDataFile2
            // 
            this.edtCifarDataFile2.Location = new System.Drawing.Point(127, 70);
            this.edtCifarDataFile2.Name = "edtCifarDataFile2";
            this.edtCifarDataFile2.Size = new System.Drawing.Size(528, 20);
            this.edtCifarDataFile2.TabIndex = 6;
            // 
            // btnBrowseBin2
            // 
            this.btnBrowseBin2.Location = new System.Drawing.Point(661, 70);
            this.btnBrowseBin2.Name = "btnBrowseBin2";
            this.btnBrowseBin2.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseBin2.TabIndex = 7;
            this.btnBrowseBin2.Text = "...";
            this.btnBrowseBin2.UseVisualStyleBackColor = true;
            this.btnBrowseBin2.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(31, 99);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(90, 13);
            this.label3.TabIndex = 8;
            this.label3.Text = "data_batch_3.bin";
            // 
            // edtCifarDataFile3
            // 
            this.edtCifarDataFile3.Location = new System.Drawing.Point(127, 96);
            this.edtCifarDataFile3.Name = "edtCifarDataFile3";
            this.edtCifarDataFile3.Size = new System.Drawing.Size(528, 20);
            this.edtCifarDataFile3.TabIndex = 9;
            // 
            // btnBrowseBin3
            // 
            this.btnBrowseBin3.Location = new System.Drawing.Point(661, 96);
            this.btnBrowseBin3.Name = "btnBrowseBin3";
            this.btnBrowseBin3.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseBin3.TabIndex = 10;
            this.btnBrowseBin3.Text = "...";
            this.btnBrowseBin3.UseVisualStyleBackColor = true;
            this.btnBrowseBin3.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(31, 125);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(90, 13);
            this.label4.TabIndex = 11;
            this.label4.Text = "data_batch_4.bin";
            // 
            // edtCifarDataFile4
            // 
            this.edtCifarDataFile4.Location = new System.Drawing.Point(127, 122);
            this.edtCifarDataFile4.Name = "edtCifarDataFile4";
            this.edtCifarDataFile4.Size = new System.Drawing.Size(528, 20);
            this.edtCifarDataFile4.TabIndex = 12;
            // 
            // btnBrowseBin4
            // 
            this.btnBrowseBin4.Location = new System.Drawing.Point(661, 122);
            this.btnBrowseBin4.Name = "btnBrowseBin4";
            this.btnBrowseBin4.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseBin4.TabIndex = 13;
            this.btnBrowseBin4.Text = "...";
            this.btnBrowseBin4.UseVisualStyleBackColor = true;
            this.btnBrowseBin4.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(31, 151);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(90, 13);
            this.label5.TabIndex = 14;
            this.label5.Text = "data_batch_5.bin";
            // 
            // edtCifarDataFile5
            // 
            this.edtCifarDataFile5.Location = new System.Drawing.Point(127, 148);
            this.edtCifarDataFile5.Name = "edtCifarDataFile5";
            this.edtCifarDataFile5.Size = new System.Drawing.Size(528, 20);
            this.edtCifarDataFile5.TabIndex = 15;
            // 
            // btnBrowseBin5
            // 
            this.btnBrowseBin5.Location = new System.Drawing.Point(661, 148);
            this.btnBrowseBin5.Name = "btnBrowseBin5";
            this.btnBrowseBin5.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseBin5.TabIndex = 16;
            this.btnBrowseBin5.Text = "...";
            this.btnBrowseBin5.UseVisualStyleBackColor = true;
            this.btnBrowseBin5.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(47, 177);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(74, 13);
            this.label6.TabIndex = 17;
            this.label6.Text = "test_batch.bin";
            // 
            // edtCifarDataFile6
            // 
            this.edtCifarDataFile6.Location = new System.Drawing.Point(127, 174);
            this.edtCifarDataFile6.Name = "edtCifarDataFile6";
            this.edtCifarDataFile6.Size = new System.Drawing.Size(528, 20);
            this.edtCifarDataFile6.TabIndex = 18;
            // 
            // btnBrowseBin6
            // 
            this.btnBrowseBin6.Location = new System.Drawing.Point(661, 174);
            this.btnBrowseBin6.Name = "btnBrowseBin6";
            this.btnBrowseBin6.Size = new System.Drawing.Size(29, 20);
            this.btnBrowseBin6.TabIndex = 19;
            this.btnBrowseBin6.Text = "...";
            this.btnBrowseBin6.UseVisualStyleBackColor = true;
            this.btnBrowseBin6.Click += new System.EventHandler(this.btnBrowseBin_Click);
            // 
            // btnDownload
            // 
            this.btnDownload.Location = new System.Drawing.Point(127, 208);
            this.btnDownload.Name = "btnDownload";
            this.btnDownload.Size = new System.Drawing.Size(75, 23);
            this.btnDownload.TabIndex = 20;
            this.btnDownload.Text = "Download";
            this.btnDownload.UseVisualStyleBackColor = true;
            this.btnDownload.Click += new System.EventHandler(this.btnDownload_Click);
            // 
            // lblDownloadPct
            // 
            this.lblDownloadPct.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblDownloadPct.Location = new System.Drawing.Point(208, 212);
            this.lblDownloadPct.Name = "lblDownloadPct";
            this.lblDownloadPct.Size = new System.Drawing.Size(53, 17);
            this.lblDownloadPct.TabIndex = 23;
            this.lblDownloadPct.Text = "0.00 %";
            this.lblDownloadPct.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // FormCifar10
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(702, 243);
            this.Controls.Add(this.lblDownloadPct);
            this.Controls.Add(this.btnDownload);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.btnBrowseBin6);
            this.Controls.Add(this.btnBrowseBin5);
            this.Controls.Add(this.btnBrowseBin4);
            this.Controls.Add(this.btnBrowseBin3);
            this.Controls.Add(this.btnBrowseBin2);
            this.Controls.Add(this.btnBrowseBin1);
            this.Controls.Add(this.edtCifarDataFile6);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.edtCifarDataFile5);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.edtCifarDataFile4);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.edtCifarDataFile3);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.edtCifarDataFile2);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.edtCifarDataFile1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lblDownloadSite);
            this.Controls.Add(this.lblDownload);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormCifar10";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "CIFAR-10 Data File";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FormCifar10_FormClosing);
            this.Load += new System.EventHandler(this.FormCiFar10_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label lblDownload;
        private System.Windows.Forms.Label lblDownloadSite;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox edtCifarDataFile1;
        private System.Windows.Forms.Button btnBrowseBin1;
        private System.Windows.Forms.OpenFileDialog openFileDialogBin;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Timer timerUI;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox edtCifarDataFile2;
        private System.Windows.Forms.Button btnBrowseBin2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox edtCifarDataFile3;
        private System.Windows.Forms.Button btnBrowseBin3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox edtCifarDataFile4;
        private System.Windows.Forms.Button btnBrowseBin4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox edtCifarDataFile5;
        private System.Windows.Forms.Button btnBrowseBin5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox edtCifarDataFile6;
        private System.Windows.Forms.Button btnBrowseBin6;
        private System.Windows.Forms.Button btnDownload;
        private System.Windows.Forms.Label lblDownloadPct;
    }
}