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
            this.radMultiThread = new System.Windows.Forms.RadioButton();
            this.radSingleThread = new System.Windows.Forms.RadioButton();
            this.radSimple = new System.Windows.Forms.RadioButton();
            this.chkAllowDiscountReset = new System.Windows.Forms.CheckBox();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnCancel
            // 
            this.btnCancel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancel.Location = new System.Drawing.Point(202, 181);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(75, 23);
            this.btnCancel.TabIndex = 5;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            // 
            // btnOK
            // 
            this.btnOK.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnOK.Location = new System.Drawing.Point(121, 181);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(75, 23);
            this.btnOK.TabIndex = 6;
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
            this.lblGym.TabIndex = 7;
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
            this.lblGymName.Size = new System.Drawing.Size(216, 27);
            this.lblGymName.TabIndex = 7;
            // 
            // chkShowUi
            // 
            this.chkShowUi.AutoSize = true;
            this.chkShowUi.Location = new System.Drawing.Point(12, 51);
            this.chkShowUi.Name = "chkShowUi";
            this.chkShowUi.Size = new System.Drawing.Size(120, 17);
            this.chkShowUi.TabIndex = 8;
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
            this.chkUseAcceleratedTraining.TabIndex = 8;
            this.chkUseAcceleratedTraining.Text = "Use accelerated training";
            this.toolTip1.SetToolTip(this.chkUseAcceleratedTraining, "Enable accelerated training which focuses on gradient changes (Note this works be" +
        "st with Cart-Pole)");
            this.chkUseAcceleratedTraining.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.radMultiThread);
            this.groupBox1.Controls.Add(this.radSingleThread);
            this.groupBox1.Controls.Add(this.radSimple);
            this.groupBox1.Location = new System.Drawing.Point(12, 120);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(263, 51);
            this.groupBox1.TabIndex = 9;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Trainer";
            // 
            // radMultiThread
            // 
            this.radMultiThread.AutoSize = true;
            this.radMultiThread.Location = new System.Drawing.Point(170, 19);
            this.radMultiThread.Name = "radMultiThread";
            this.radMultiThread.Size = new System.Drawing.Size(80, 17);
            this.radMultiThread.TabIndex = 2;
            this.radMultiThread.Text = "Multi-thread";
            this.radMultiThread.UseVisualStyleBackColor = true;
            // 
            // radSingleThread
            // 
            this.radSingleThread.AutoSize = true;
            this.radSingleThread.Location = new System.Drawing.Point(77, 19);
            this.radSingleThread.Name = "radSingleThread";
            this.radSingleThread.Size = new System.Drawing.Size(87, 17);
            this.radSingleThread.TabIndex = 1;
            this.radSingleThread.Text = "Single-thread";
            this.radSingleThread.UseVisualStyleBackColor = true;
            // 
            // radSimple
            // 
            this.radSimple.AutoSize = true;
            this.radSimple.Checked = true;
            this.radSimple.Location = new System.Drawing.Point(15, 19);
            this.radSimple.Name = "radSimple";
            this.radSimple.Size = new System.Drawing.Size(56, 17);
            this.radSimple.TabIndex = 0;
            this.radSimple.TabStop = true;
            this.radSimple.Text = "Simple";
            this.radSimple.UseVisualStyleBackColor = true;
            // 
            // chkAllowDiscountReset
            // 
            this.chkAllowDiscountReset.AutoSize = true;
            this.chkAllowDiscountReset.Location = new System.Drawing.Point(12, 97);
            this.chkAllowDiscountReset.Name = "chkAllowDiscountReset";
            this.chkAllowDiscountReset.Size = new System.Drawing.Size(120, 17);
            this.chkAllowDiscountReset.TabIndex = 8;
            this.chkAllowDiscountReset.Text = "Allow discount reset";
            this.toolTip1.SetToolTip(this.chkAllowDiscountReset, "Allowing the discount reset, resets the running sum on non-zero reward values.");
            this.chkAllowDiscountReset.UseVisualStyleBackColor = true;
            // 
            // FormCustomTraining
            // 
            this.AcceptButton = this.btnOK;
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.CancelButton = this.btnCancel;
            this.ClientSize = new System.Drawing.Size(289, 216);
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
        private System.Windows.Forms.RadioButton radMultiThread;
        private System.Windows.Forms.RadioButton radSingleThread;
        private System.Windows.Forms.RadioButton radSimple;
        private System.Windows.Forms.ToolTip toolTip1;
        private System.Windows.Forms.CheckBox chkAllowDiscountReset;
    }
}