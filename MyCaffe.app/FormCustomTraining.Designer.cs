namespace MyCaffe.app
{
    partial class FormCustomTraining
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormCustomTraining));
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnOK = new System.Windows.Forms.Button();
            this.lblGym = new System.Windows.Forms.Label();
            this.lblGymName = new System.Windows.Forms.Label();
            this.chkShowUi = new System.Windows.Forms.CheckBox();
            this.chkUseAcceleratedTraining = new System.Windows.Forms.CheckBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.radNoisyNetSingleThread = new System.Windows.Forms.RadioButton();
            this.radC51SingleThread = new System.Windows.Forms.RadioButton();
            this.radPGMultiThread = new System.Windows.Forms.RadioButton();
            this.radPGSingleThread = new System.Windows.Forms.RadioButton();
            this.radPGSimple = new System.Windows.Forms.RadioButton();
            this.chkAllowDiscountReset = new System.Windows.Forms.CheckBox();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.grpRom = new System.Windows.Forms.GroupBox();
            this.radAtariBreakout = new System.Windows.Forms.RadioButton();
            this.radAtariPong = new System.Windows.Forms.RadioButton();
            this.chkAllowNegativeRewards = new System.Windows.Forms.CheckBox();
            this.chkTerminateOnRallyEnd = new System.Windows.Forms.CheckBox();
            this.chkLoadWeights = new System.Windows.Forms.CheckBox();
            this.lblVMin = new System.Windows.Forms.Label();
            this.edtVMin = new System.Windows.Forms.TextBox();
            this.lblVMax = new System.Windows.Forms.Label();
            this.edtVMax = new System.Windows.Forms.TextBox();
            this.btnReset = new System.Windows.Forms.Button();
            this.radNoisyNetSimple = new System.Windows.Forms.RadioButton();
            this.groupBox1.SuspendLayout();
            this.grpRom.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(611, 194);
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
            this.btnOK.Location = new System.Drawing.Point(530, 194);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 15;
            this.btnOK.Text = "OK";
            this.btnOK.UseVisualStyleBackColor = true;
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // lblGym
            // 
            this.lblGym.BackColor = System.Drawing.Color.SteelBlue;
            this.lblGym.Font = new System.Drawing.Font("Century Gothic", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblGym.ForeColor = System.Drawing.Color.LightSteelBlue;
            this.lblGym.Location = new System.Drawing.Point(-1, 9);
            this.lblGym.Name = "lblGym";
            this.lblGym.Size = new System.Drawing.Size(72, 27);
            this.lblGym.TabIndex = 0;
            this.lblGym.Text = "Gym:";
            // 
            // lblGymName
            // 
            this.lblGymName.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblGymName.BackColor = System.Drawing.Color.LightSteelBlue;
            this.lblGymName.Font = new System.Drawing.Font("Century Gothic", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblGymName.ForeColor = System.Drawing.Color.SteelBlue;
            this.lblGymName.Location = new System.Drawing.Point(73, 9);
            this.lblGymName.Name = "lblGymName";
            this.lblGymName.Size = new System.Drawing.Size(625, 27);
            this.lblGymName.TabIndex = 1;
            // 
            // chkShowUi
            // 
            this.chkShowUi.AutoSize = true;
            this.chkShowUi.Location = new System.Drawing.Point(12, 51);
            this.chkShowUi.Name = "chkShowUi";
            this.chkShowUi.Size = new System.Drawing.Size(120, 17);
            this.chkShowUi.TabIndex = 2;
            this.chkShowUi.Text = "Show user-interface";
            this.toolTip1.SetToolTip(this.chkShowUi, "Show the gym user interface.");
            this.chkShowUi.UseVisualStyleBackColor = true;
            // 
            // chkUseAcceleratedTraining
            // 
            this.chkUseAcceleratedTraining.AutoSize = true;
            this.chkUseAcceleratedTraining.Location = new System.Drawing.Point(12, 74);
            this.chkUseAcceleratedTraining.Name = "chkUseAcceleratedTraining";
            this.chkUseAcceleratedTraining.Size = new System.Drawing.Size(141, 17);
            this.chkUseAcceleratedTraining.TabIndex = 3;
            this.chkUseAcceleratedTraining.Text = "Use accelerated training";
            this.toolTip1.SetToolTip(this.chkUseAcceleratedTraining, "Enable accelerated training which focuses on gradient changes (Note this works be" +
        "st with Cart-Pole)");
            this.chkUseAcceleratedTraining.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.radNoisyNetSimple);
            this.groupBox1.Controls.Add(this.radNoisyNetSingleThread);
            this.groupBox1.Controls.Add(this.radC51SingleThread);
            this.groupBox1.Controls.Add(this.radPGMultiThread);
            this.groupBox1.Controls.Add(this.radPGSingleThread);
            this.groupBox1.Controls.Add(this.radPGSimple);
            this.groupBox1.Location = new System.Drawing.Point(12, 120);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(674, 51);
            this.groupBox1.TabIndex = 13;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Trainer";
            // 
            // radNoisyNetSingleThread
            // 
            this.radNoisyNetSingleThread.AutoSize = true;
            this.radNoisyNetSingleThread.Location = new System.Drawing.Point(420, 19);
            this.radNoisyNetSingleThread.Name = "radNoisyNetSingleThread";
            this.radNoisyNetSingleThread.Size = new System.Drawing.Size(133, 17);
            this.radNoisyNetSingleThread.TabIndex = 4;
            this.radNoisyNetSingleThread.Text = "NoisyNet Single-thread";
            this.radNoisyNetSingleThread.UseVisualStyleBackColor = true;
            this.radNoisyNetSingleThread.CheckedChanged += new System.EventHandler(this.radNoisyNet_CheckedChanged);
            // 
            // radC51SingleThread
            // 
            this.radC51SingleThread.AutoSize = true;
            this.radC51SingleThread.Location = new System.Drawing.Point(310, 19);
            this.radC51SingleThread.Name = "radC51SingleThread";
            this.radC51SingleThread.Size = new System.Drawing.Size(109, 17);
            this.radC51SingleThread.TabIndex = 3;
            this.radC51SingleThread.Text = "C51 Single-thread";
            this.radC51SingleThread.UseVisualStyleBackColor = true;
            this.radC51SingleThread.CheckedChanged += new System.EventHandler(this.radC51SingleThread_CheckedChanged);
            // 
            // radPGMultiThread
            // 
            this.radPGMultiThread.AutoSize = true;
            this.radPGMultiThread.Location = new System.Drawing.Point(206, 19);
            this.radPGMultiThread.Name = "radPGMultiThread";
            this.radPGMultiThread.Size = new System.Drawing.Size(98, 17);
            this.radPGMultiThread.TabIndex = 2;
            this.radPGMultiThread.Text = "PG Multi-thread";
            this.radPGMultiThread.UseVisualStyleBackColor = true;
            // 
            // radPGSingleThread
            // 
            this.radPGSingleThread.AutoSize = true;
            this.radPGSingleThread.Location = new System.Drawing.Point(95, 19);
            this.radPGSingleThread.Name = "radPGSingleThread";
            this.radPGSingleThread.Size = new System.Drawing.Size(105, 17);
            this.radPGSingleThread.TabIndex = 1;
            this.radPGSingleThread.Text = "PG Single-thread";
            this.radPGSingleThread.UseVisualStyleBackColor = true;
            // 
            // radPGSimple
            // 
            this.radPGSimple.AutoSize = true;
            this.radPGSimple.Checked = true;
            this.radPGSimple.Location = new System.Drawing.Point(15, 19);
            this.radPGSimple.Name = "radPGSimple";
            this.radPGSimple.Size = new System.Drawing.Size(74, 17);
            this.radPGSimple.TabIndex = 0;
            this.radPGSimple.TabStop = true;
            this.radPGSimple.Text = "PG Simple";
            this.radPGSimple.UseVisualStyleBackColor = true;
            // 
            // chkAllowDiscountReset
            // 
            this.chkAllowDiscountReset.AutoSize = true;
            this.chkAllowDiscountReset.Location = new System.Drawing.Point(12, 97);
            this.chkAllowDiscountReset.Name = "chkAllowDiscountReset";
            this.chkAllowDiscountReset.Size = new System.Drawing.Size(120, 17);
            this.chkAllowDiscountReset.TabIndex = 4;
            this.chkAllowDiscountReset.Text = "Allow discount reset";
            this.toolTip1.SetToolTip(this.chkAllowDiscountReset, "Allowing the discount reset, resets the running sum on non-zero reward values.");
            this.chkAllowDiscountReset.UseVisualStyleBackColor = true;
            // 
            // grpRom
            // 
            this.grpRom.Controls.Add(this.radAtariBreakout);
            this.grpRom.Controls.Add(this.radAtariPong);
            this.grpRom.Location = new System.Drawing.Point(12, 177);
            this.grpRom.Name = "grpRom";
            this.grpRom.Size = new System.Drawing.Size(200, 40);
            this.grpRom.TabIndex = 14;
            this.grpRom.TabStop = false;
            this.grpRom.Text = "ATARI ROM";
            this.grpRom.Visible = false;
            // 
            // radAtariBreakout
            // 
            this.radAtariBreakout.AutoSize = true;
            this.radAtariBreakout.Location = new System.Drawing.Point(95, 17);
            this.radAtariBreakout.Name = "radAtariBreakout";
            this.radAtariBreakout.Size = new System.Drawing.Size(68, 17);
            this.radAtariBreakout.TabIndex = 1;
            this.radAtariBreakout.TabStop = true;
            this.radAtariBreakout.Text = "Breakout";
            this.radAtariBreakout.UseVisualStyleBackColor = true;
            // 
            // radAtariPong
            // 
            this.radAtariPong.AutoSize = true;
            this.radAtariPong.Checked = true;
            this.radAtariPong.Location = new System.Drawing.Point(15, 17);
            this.radAtariPong.Name = "radAtariPong";
            this.radAtariPong.Size = new System.Drawing.Size(50, 17);
            this.radAtariPong.TabIndex = 0;
            this.radAtariPong.TabStop = true;
            this.radAtariPong.Text = "Pong";
            this.radAtariPong.UseVisualStyleBackColor = true;
            // 
            // chkAllowNegativeRewards
            // 
            this.chkAllowNegativeRewards.AutoSize = true;
            this.chkAllowNegativeRewards.Location = new System.Drawing.Point(218, 51);
            this.chkAllowNegativeRewards.Name = "chkAllowNegativeRewards";
            this.chkAllowNegativeRewards.Size = new System.Drawing.Size(135, 17);
            this.chkAllowNegativeRewards.TabIndex = 5;
            this.chkAllowNegativeRewards.Text = "Allow negative rewards";
            this.chkAllowNegativeRewards.UseVisualStyleBackColor = true;
            // 
            // chkTerminateOnRallyEnd
            // 
            this.chkTerminateOnRallyEnd.AutoSize = true;
            this.chkTerminateOnRallyEnd.Location = new System.Drawing.Point(218, 74);
            this.chkTerminateOnRallyEnd.Name = "chkTerminateOnRallyEnd";
            this.chkTerminateOnRallyEnd.Size = new System.Drawing.Size(130, 17);
            this.chkTerminateOnRallyEnd.TabIndex = 6;
            this.chkTerminateOnRallyEnd.Text = "Terminate on rally end";
            this.chkTerminateOnRallyEnd.UseVisualStyleBackColor = true;
            // 
            // chkLoadWeights
            // 
            this.chkLoadWeights.AutoSize = true;
            this.chkLoadWeights.Location = new System.Drawing.Point(218, 97);
            this.chkLoadWeights.Name = "chkLoadWeights";
            this.chkLoadWeights.Size = new System.Drawing.Size(127, 17);
            this.chkLoadWeights.TabIndex = 7;
            this.chkLoadWeights.Text = "Load weights (if exist)";
            this.chkLoadWeights.UseVisualStyleBackColor = true;
            // 
            // lblVMin
            // 
            this.lblVMin.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.lblVMin.AutoSize = true;
            this.lblVMin.Location = new System.Drawing.Point(601, 52);
            this.lblVMin.Name = "lblVMin";
            this.lblVMin.Size = new System.Drawing.Size(34, 13);
            this.lblVMin.TabIndex = 8;
            this.lblVMin.Text = "VMin:";
            this.lblVMin.Visible = false;
            // 
            // edtVMin
            // 
            this.edtVMin.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.edtVMin.Location = new System.Drawing.Point(641, 49);
            this.edtVMin.Name = "edtVMin";
            this.edtVMin.Size = new System.Drawing.Size(45, 20);
            this.edtVMin.TabIndex = 9;
            this.edtVMin.Text = "-10";
            this.edtVMin.Visible = false;
            // 
            // lblVMax
            // 
            this.lblVMax.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.lblVMax.AutoSize = true;
            this.lblVMax.Location = new System.Drawing.Point(598, 75);
            this.lblVMax.Name = "lblVMax";
            this.lblVMax.Size = new System.Drawing.Size(37, 13);
            this.lblVMax.TabIndex = 10;
            this.lblVMax.Text = "VMax:";
            this.lblVMax.Visible = false;
            // 
            // edtVMax
            // 
            this.edtVMax.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.edtVMax.Location = new System.Drawing.Point(641, 72);
            this.edtVMax.Name = "edtVMax";
            this.edtVMax.Size = new System.Drawing.Size(45, 20);
            this.edtVMax.TabIndex = 11;
            this.edtVMax.Text = "10";
            this.edtVMax.Visible = false;
            // 
            // btnReset
            // 
            this.btnReset.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnReset.Location = new System.Drawing.Point(641, 98);
            this.btnReset.Name = "btnReset";
            this.btnReset.Size = new System.Drawing.Size(45, 21);
            this.btnReset.TabIndex = 12;
            this.btnReset.Text = "reset";
            this.btnReset.UseVisualStyleBackColor = true;
            this.btnReset.Visible = false;
            this.btnReset.Click += new System.EventHandler(this.btnReset_Click);
            // 
            // radNoisyNetSimple
            // 
            this.radNoisyNetSimple.AutoSize = true;
            this.radNoisyNetSimple.Enabled = false;
            this.radNoisyNetSimple.Location = new System.Drawing.Point(559, 19);
            this.radNoisyNetSimple.Name = "radNoisyNetSimple";
            this.radNoisyNetSimple.Size = new System.Drawing.Size(102, 17);
            this.radNoisyNetSimple.TabIndex = 5;
            this.radNoisyNetSimple.TabStop = true;
            this.radNoisyNetSimple.Text = "NoisyNet Simple";
            this.radNoisyNetSimple.UseVisualStyleBackColor = true;
            // 
            // FormCustomTraining
            // 
            this.AcceptButton = this.btnOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new System.Drawing.Size(698, 229);
            this.Controls.Add(this.btnReset);
            this.Controls.Add(this.edtVMax);
            this.Controls.Add(this.lblVMax);
            this.Controls.Add(this.edtVMin);
            this.Controls.Add(this.lblVMin);
            this.Controls.Add(this.chkLoadWeights);
            this.Controls.Add(this.chkTerminateOnRallyEnd);
            this.Controls.Add(this.chkAllowNegativeRewards);
            this.Controls.Add(this.grpRom);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.chkAllowDiscountReset);
            this.Controls.Add(this.chkUseAcceleratedTraining);
            this.Controls.Add(this.chkShowUi);
            this.Controls.Add(this.lblGymName);
            this.Controls.Add(this.lblGym);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormCustomTraining";
            this.ShowInTaskbar = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "Reinforcement Training Test";
            this.Load += new System.EventHandler(this.FromCustomTraining_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.grpRom.ResumeLayout(false);
            this.grpRom.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnOK;
        private System.Windows.Forms.Label lblGym;
        private System.Windows.Forms.Label lblGymName;
        private System.Windows.Forms.CheckBox chkShowUi;
        private System.Windows.Forms.CheckBox chkUseAcceleratedTraining;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.RadioButton radPGMultiThread;
        private System.Windows.Forms.RadioButton radPGSingleThread;
        private System.Windows.Forms.RadioButton radPGSimple;
        private System.Windows.Forms.ToolTip toolTip1;
        private System.Windows.Forms.CheckBox chkAllowDiscountReset;
        private System.Windows.Forms.RadioButton radC51SingleThread;
        private System.Windows.Forms.GroupBox grpRom;
        private System.Windows.Forms.RadioButton radAtariBreakout;
        private System.Windows.Forms.RadioButton radAtariPong;
        private System.Windows.Forms.CheckBox chkAllowNegativeRewards;
        private System.Windows.Forms.CheckBox chkTerminateOnRallyEnd;
        private System.Windows.Forms.CheckBox chkLoadWeights;
        private System.Windows.Forms.Label lblVMin;
        private System.Windows.Forms.TextBox edtVMin;
        private System.Windows.Forms.Label lblVMax;
        private System.Windows.Forms.TextBox edtVMax;
        private System.Windows.Forms.Button btnReset;
        private System.Windows.Forms.RadioButton radNoisyNetSingleThread;
        private System.Windows.Forms.RadioButton radNoisyNetSimple;
    }
}