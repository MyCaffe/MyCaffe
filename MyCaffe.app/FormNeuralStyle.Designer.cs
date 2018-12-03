namespace MyCaffe.app
{
    partial class FormNeuralStyle
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormNeuralStyle));
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnOK = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.cmbModel = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.cmbSolver = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.edtLearningRate = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.edtIterations = new System.Windows.Forms.TextBox();
            this.edtIntermediateIterations = new System.Windows.Forms.TextBox();
            this.chkIntermediateOutput = new System.Windows.Forms.CheckBox();
            this.label5 = new System.Windows.Forms.Label();
            this.edtResultPath = new System.Windows.Forms.TextBox();
            this.btnBrowseResultPath = new System.Windows.Forms.Button();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.edtTvLoss = new System.Windows.Forms.TextBox();
            this.chkEnableTvLoss = new System.Windows.Forms.CheckBox();
            this.edtStyleImageFile = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.btnBrowseStyle = new System.Windows.Forms.Button();
            this.edtContentImageFile = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.btnBrowseContent = new System.Windows.Forms.Button();
            this.lblGymName = new System.Windows.Forms.Label();
            this.lblGym = new System.Windows.Forms.Label();
            this.openFileDialogStyle = new System.Windows.Forms.OpenFileDialog();
            this.openFileDialogContent = new System.Windows.Forms.OpenFileDialog();
            this.SuspendLayout();
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(476, 229);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 16;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(395, 229);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 15;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(52, 49);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(39, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Model:";
            // 
            // cmbModel
            // 
            this.cmbModel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.cmbModel.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbModel.FormattingEnabled = true;
            this.cmbModel.Items.AddRange(new object[] {
            "VGG19",
            "GOOGLENET"});
            this.cmbModel.Location = new System.Drawing.Point(97, 46);
            this.cmbModel.Name = "cmbModel";
            this.cmbModel.Size = new System.Drawing.Size(104, 21);
            this.cmbModel.TabIndex = 1;
            // 
            // label2
            // 
            this.label2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(207, 49);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(40, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Solver:";
            // 
            // cmbSolver
            // 
            this.cmbSolver.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.cmbSolver.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbSolver.FormattingEnabled = true;
            this.cmbSolver.Items.AddRange(new object[] {
            "LBFGS",
            "ADAM",
            "RMSPROP",
            "SGD"});
            this.cmbSolver.Location = new System.Drawing.Point(253, 46);
            this.cmbSolver.Name = "cmbSolver";
            this.cmbSolver.Size = new System.Drawing.Size(104, 21);
            this.cmbSolver.TabIndex = 3;
            this.cmbSolver.SelectedIndexChanged += new System.EventHandler(this.cmbSolver_SelectedIndexChanged);
            // 
            // label3
            // 
            this.label3.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(363, 50);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(77, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Learning Rate:";
            // 
            // edtLearningRate
            // 
            this.edtLearningRate.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.edtLearningRate.Location = new System.Drawing.Point(446, 47);
            this.edtLearningRate.Name = "edtLearningRate";
            this.edtLearningRate.Size = new System.Drawing.Size(67, 20);
            this.edtLearningRate.TabIndex = 5;
            this.edtLearningRate.Text = "1.0";
            this.edtLearningRate.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(387, 102);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(53, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "Iterations:";
            // 
            // edtIterations
            // 
            this.edtIterations.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.edtIterations.Location = new System.Drawing.Point(446, 99);
            this.edtIterations.Name = "edtIterations";
            this.edtIterations.Size = new System.Drawing.Size(67, 20);
            this.edtIterations.TabIndex = 9;
            this.edtIterations.Text = "1000";
            this.edtIterations.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // edtIntermediateIterations
            // 
            this.edtIntermediateIterations.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.edtIntermediateIterations.Enabled = false;
            this.edtIntermediateIterations.Location = new System.Drawing.Point(446, 125);
            this.edtIntermediateIterations.Name = "edtIntermediateIterations";
            this.edtIntermediateIterations.Size = new System.Drawing.Size(67, 20);
            this.edtIntermediateIterations.TabIndex = 11;
            this.edtIntermediateIterations.Text = "0";
            this.edtIntermediateIterations.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // chkIntermediateOutput
            // 
            this.chkIntermediateOutput.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.chkIntermediateOutput.AutoSize = true;
            this.chkIntermediateOutput.Location = new System.Drawing.Point(256, 127);
            this.chkIntermediateOutput.Name = "chkIntermediateOutput";
            this.chkIntermediateOutput.Size = new System.Drawing.Size(184, 17);
            this.chkIntermediateOutput.TabIndex = 10;
            this.chkIntermediateOutput.Text = "Enable intermediate output every:";
            this.chkIntermediateOutput.UseVisualStyleBackColor = true;
            this.chkIntermediateOutput.CheckedChanged += new System.EventHandler(this.chkIntermediateOutput_CheckedChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(27, 206);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(65, 13);
            this.label5.TabIndex = 12;
            this.label5.Text = "Result Path:";
            // 
            // edtResultPath
            // 
            this.edtResultPath.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.edtResultPath.Location = new System.Drawing.Point(98, 203);
            this.edtResultPath.Name = "edtResultPath";
            this.edtResultPath.ReadOnly = true;
            this.edtResultPath.Size = new System.Drawing.Size(415, 20);
            this.edtResultPath.TabIndex = 13;
            this.edtResultPath.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // btnBrowseResultPath
            // 
            this.btnBrowseResultPath.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnBrowseResultPath.Location = new System.Drawing.Point(519, 203);
            this.btnBrowseResultPath.Name = "btnBrowseResultPath";
            this.btnBrowseResultPath.Size = new System.Drawing.Size(32, 20);
            this.btnBrowseResultPath.TabIndex = 14;
            this.btnBrowseResultPath.Text = "...";
            this.btnBrowseResultPath.UseVisualStyleBackColor = true;
            this.btnBrowseResultPath.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // folderBrowserDialog1
            // 
            this.folderBrowserDialog1.RootFolder = System.Environment.SpecialFolder.MyComputer;
            // 
            // edtTvLoss
            // 
            this.edtTvLoss.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.edtTvLoss.Enabled = false;
            this.edtTvLoss.Location = new System.Drawing.Point(446, 73);
            this.edtTvLoss.Name = "edtTvLoss";
            this.edtTvLoss.Size = new System.Drawing.Size(67, 20);
            this.edtTvLoss.TabIndex = 7;
            this.edtTvLoss.Text = "0";
            this.edtTvLoss.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // chkEnableTvLoss
            // 
            this.chkEnableTvLoss.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.chkEnableTvLoss.AutoSize = true;
            this.chkEnableTvLoss.Location = new System.Drawing.Point(278, 75);
            this.chkEnableTvLoss.Name = "chkEnableTvLoss";
            this.chkEnableTvLoss.Size = new System.Drawing.Size(166, 17);
            this.chkEnableTvLoss.TabIndex = 6;
            this.chkEnableTvLoss.Text = "Enable TV-loss for smoothing:";
            this.chkEnableTvLoss.UseVisualStyleBackColor = true;
            this.chkEnableTvLoss.CheckedChanged += new System.EventHandler(this.chkEnableTvLoss_CheckedChanged);
            // 
            // edtStyleImageFile
            // 
            this.edtStyleImageFile.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.edtStyleImageFile.Location = new System.Drawing.Point(98, 151);
            this.edtStyleImageFile.Name = "edtStyleImageFile";
            this.edtStyleImageFile.ReadOnly = true;
            this.edtStyleImageFile.Size = new System.Drawing.Size(415, 20);
            this.edtStyleImageFile.TabIndex = 13;
            this.edtStyleImageFile.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(27, 154);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(65, 13);
            this.label6.TabIndex = 12;
            this.label6.Text = "Style Image:";
            // 
            // btnBrowseStyle
            // 
            this.btnBrowseStyle.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnBrowseStyle.Location = new System.Drawing.Point(519, 151);
            this.btnBrowseStyle.Name = "btnBrowseStyle";
            this.btnBrowseStyle.Size = new System.Drawing.Size(32, 20);
            this.btnBrowseStyle.TabIndex = 14;
            this.btnBrowseStyle.Text = "...";
            this.btnBrowseStyle.UseVisualStyleBackColor = true;
            this.btnBrowseStyle.Click += new System.EventHandler(this.btnBrowseStyle_Click);
            // 
            // edtContentImageFile
            // 
            this.edtContentImageFile.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.edtContentImageFile.Location = new System.Drawing.Point(98, 177);
            this.edtContentImageFile.Name = "edtContentImageFile";
            this.edtContentImageFile.ReadOnly = true;
            this.edtContentImageFile.Size = new System.Drawing.Size(415, 20);
            this.edtContentImageFile.TabIndex = 13;
            this.edtContentImageFile.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(13, 180);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(79, 13);
            this.label7.TabIndex = 12;
            this.label7.Text = "Content Image:";
            // 
            // btnBrowseContent
            // 
            this.btnBrowseContent.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnBrowseContent.Location = new System.Drawing.Point(519, 177);
            this.btnBrowseContent.Name = "btnBrowseContent";
            this.btnBrowseContent.Size = new System.Drawing.Size(32, 20);
            this.btnBrowseContent.TabIndex = 14;
            this.btnBrowseContent.Text = "...";
            this.btnBrowseContent.UseVisualStyleBackColor = true;
            this.btnBrowseContent.Click += new System.EventHandler(this.btnBrowseContent_Click);
            // 
            // lblGymName
            // 
            this.lblGymName.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblGymName.BackColor = System.Drawing.Color.LightSteelBlue;
            this.lblGymName.Font = new System.Drawing.Font("Century Gothic", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblGymName.ForeColor = System.Drawing.Color.SteelBlue;
            this.lblGymName.Location = new System.Drawing.Point(223, 9);
            this.lblGymName.Name = "lblGymName";
            this.lblGymName.Size = new System.Drawing.Size(340, 27);
            this.lblGymName.TabIndex = 17;
            // 
            // lblGym
            // 
            this.lblGym.BackColor = System.Drawing.Color.SteelBlue;
            this.lblGym.Font = new System.Drawing.Font("Century Gothic", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblGym.ForeColor = System.Drawing.Color.LightSteelBlue;
            this.lblGym.Location = new System.Drawing.Point(0, 9);
            this.lblGym.Name = "lblGym";
            this.lblGym.Size = new System.Drawing.Size(220, 27);
            this.lblGym.TabIndex = 18;
            this.lblGym.Text = "Neural Style Transfer";
            // 
            // openFileDialogStyle
            // 
            this.openFileDialogStyle.DefaultExt = "png";
            this.openFileDialogStyle.Filter = "Image Files (*.png)|*.png||";
            this.openFileDialogStyle.Title = "Select the Style image";
            // 
            // openFileDialogContent
            // 
            this.openFileDialogContent.DefaultExt = "png";
            this.openFileDialogContent.Filter = "Image Files (*.png)|*.png||";
            this.openFileDialogContent.Title = "Select the Content image";
            // 
            // FormNeuralStyle
            // 
            this.AcceptButton = this.btnOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new System.Drawing.Size(562, 264);
            this.Controls.Add(this.lblGymName);
            this.Controls.Add(this.lblGym);
            this.Controls.Add(this.chkEnableTvLoss);
            this.Controls.Add(this.btnBrowseContent);
            this.Controls.Add(this.btnBrowseStyle);
            this.Controls.Add(this.btnBrowseResultPath);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.edtContentImageFile);
            this.Controls.Add(this.edtStyleImageFile);
            this.Controls.Add(this.edtResultPath);
            this.Controls.Add(this.edtIntermediateIterations);
            this.Controls.Add(this.chkIntermediateOutput);
            this.Controls.Add(this.edtIterations);
            this.Controls.Add(this.edtTvLoss);
            this.Controls.Add(this.edtLearningRate);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.cmbSolver);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.cmbModel);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormNeuralStyle";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Neural Style Settings";
            this.Load += new System.EventHandler(this.FormNeuralStyle_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox cmbModel;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ComboBox cmbSolver;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox edtLearningRate;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox edtIterations;
        private System.Windows.Forms.TextBox edtIntermediateIterations;
        private System.Windows.Forms.CheckBox chkIntermediateOutput;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox edtResultPath;
        private System.Windows.Forms.Button btnBrowseResultPath;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.TextBox edtTvLoss;
        private System.Windows.Forms.CheckBox chkEnableTvLoss;
        private System.Windows.Forms.TextBox edtStyleImageFile;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button btnBrowseStyle;
        private System.Windows.Forms.TextBox edtContentImageFile;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Button btnBrowseContent;
        private System.Windows.Forms.Label lblGymName;
        private System.Windows.Forms.Label lblGym;
        private System.Windows.Forms.OpenFileDialog openFileDialogStyle;
        private System.Windows.Forms.OpenFileDialog openFileDialogContent;
    }
}