namespace MyCaffe.gym
{
    partial class FormGym
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormGym));
            this.toolStripContainer1 = new System.Windows.Forms.ToolStripContainer();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.btnShowActionImage = new System.Windows.Forms.ToolStripButton();
            this.btnRecord = new System.Windows.Forms.ToolStripButton();
            this.btnDeleteRecordingData = new System.Windows.Forms.ToolStripButton();
            this.timerUI = new System.Windows.Forms.Timer(this.components);
            this.toolStripContainer1.BottomToolStripPanel.SuspendLayout();
            this.toolStripContainer1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // toolStripContainer1
            // 
            // 
            // toolStripContainer1.BottomToolStripPanel
            // 
            this.toolStripContainer1.BottomToolStripPanel.Controls.Add(this.toolStrip1);
            // 
            // toolStripContainer1.ContentPanel
            // 
            this.toolStripContainer1.ContentPanel.Size = new System.Drawing.Size(512, 487);
            this.toolStripContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.toolStripContainer1.Location = new System.Drawing.Point(0, 0);
            this.toolStripContainer1.Name = "toolStripContainer1";
            this.toolStripContainer1.Size = new System.Drawing.Size(512, 537);
            this.toolStripContainer1.TabIndex = 0;
            this.toolStripContainer1.Text = "toolStripContainer1";
            // 
            // toolStrip1
            // 
            this.toolStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.btnShowActionImage,
            this.btnRecord,
            this.btnDeleteRecordingData});
            this.toolStrip1.Location = new System.Drawing.Point(3, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(81, 25);
            this.toolStrip1.TabIndex = 0;
            // 
            // btnShowActionImage
            // 
            this.btnShowActionImage.CheckOnClick = true;
            this.btnShowActionImage.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnShowActionImage.Enabled = false;
            this.btnShowActionImage.Image = ((System.Drawing.Image)(resources.GetObject("btnShowActionImage.Image")));
            this.btnShowActionImage.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnShowActionImage.Name = "btnShowActionImage";
            this.btnShowActionImage.Size = new System.Drawing.Size(23, 22);
            this.btnShowActionImage.Text = "Show action image";
            this.btnShowActionImage.Click += new System.EventHandler(this.btnShowActionImage_Click);
            // 
            // btnRecord
            // 
            this.btnRecord.CheckOnClick = true;
            this.btnRecord.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnRecord.Image = ((System.Drawing.Image)(resources.GetObject("btnRecord.Image")));
            this.btnRecord.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnRecord.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnRecord.Name = "btnRecord";
            this.btnRecord.Size = new System.Drawing.Size(23, 22);
            this.btnRecord.Text = "Record";
            this.btnRecord.ToolTipText = "Record data saved to \'MyDocuments\\MyCaffe\\gym\\recordings\'";
            this.btnRecord.Click += new System.EventHandler(this.btnRecord_Click);
            // 
            // btnDeleteRecordingData
            // 
            this.btnDeleteRecordingData.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnDeleteRecordingData.Image = ((System.Drawing.Image)(resources.GetObject("btnDeleteRecordingData.Image")));
            this.btnDeleteRecordingData.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnDeleteRecordingData.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnDeleteRecordingData.Name = "btnDeleteRecordingData";
            this.btnDeleteRecordingData.Size = new System.Drawing.Size(23, 22);
            this.btnDeleteRecordingData.Text = "Delete Recording Data";
            this.btnDeleteRecordingData.Click += new System.EventHandler(this.btnDeleteRecordingData_Click);
            // 
            // timerUI
            // 
            this.timerUI.Enabled = true;
            this.timerUI.Interval = 250;
            this.timerUI.Tick += new System.EventHandler(this.timerUI_Tick);
            // 
            // FormGym
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(512, 537);
            this.Controls.Add(this.toolStripContainer1);
            this.DoubleBuffered = true;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "FormGym";
            this.Text = "Test Gym";
            this.TopMost = true;
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FormGym_FormClosing);
            this.Load += new System.EventHandler(this.FormGym_Load);
            this.toolStripContainer1.BottomToolStripPanel.ResumeLayout(false);
            this.toolStripContainer1.BottomToolStripPanel.PerformLayout();
            this.toolStripContainer1.ResumeLayout(false);
            this.toolStripContainer1.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ToolStripContainer toolStripContainer1;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton btnShowActionImage;
        private System.Windows.Forms.ToolStripButton btnRecord;
        private System.Windows.Forms.ToolStripButton btnDeleteRecordingData;
        private System.Windows.Forms.Timer timerUI;
    }
}