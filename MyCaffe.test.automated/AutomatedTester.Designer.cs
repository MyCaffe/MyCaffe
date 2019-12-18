namespace MyCaffe.test.automated
{
    partial class AutomatedTester
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(AutomatedTester));
            this.lstTests = new System.Windows.Forms.ListView();
            this.colIdx = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colPriority = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colResult = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colTestClass = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colTestMethod = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colError = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.contextMenuStrip1 = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.resetToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.runToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.skipToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.imageListUI = new System.Windows.Forms.ImageList(this.components);
            this.toolStripContainer1 = new System.Windows.Forms.ToolStripContainer();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.btnShowAll = new System.Windows.Forms.ToolStripButton();
            this.btnShowPassed = new System.Windows.Forms.ToolStripButton();
            this.btnShowFailures = new System.Windows.Forms.ToolStripButton();
            this.btnShowNotExecuted = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.btnRun = new System.Windows.Forms.ToolStripButton();
            this.btnAbort = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.btnGradientTests = new System.Windows.Forms.ToolStripButton();
            this.btnNonGradientTests = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.btnCurrent = new System.Windows.Forms.ToolStripButton();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.tsTotalTests = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel3 = new System.Windows.Forms.ToolStripStatusLabel();
            this.tsFailedTests = new System.Windows.Forms.ToolStripStatusLabel();
            this.tsProgress = new System.Windows.Forms.ToolStripProgressBar();
            this.tsProgressPct = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.tsTestingTime = new System.Windows.Forms.ToolStripStatusLabel();
            this.pbItemProgress = new System.Windows.Forms.ToolStripProgressBar();
            this.tsItemProgress = new System.Windows.Forms.ToolStripStatusLabel();
            this.lblActiveGPU = new System.Windows.Forms.ToolStripStatusLabel();
            this.lblActiveGPUVal = new System.Windows.Forms.ToolStripStatusLabel();
            this.timerUI = new System.Windows.Forms.Timer(this.components);
            this.lblActiveKernel = new System.Windows.Forms.ToolStripStatusLabel();
            this.lblKernelHandleVal = new System.Windows.Forms.ToolStripStatusLabel();
            this.contextMenuStrip1.SuspendLayout();
            this.toolStripContainer1.ContentPanel.SuspendLayout();
            this.toolStripContainer1.TopToolStripPanel.SuspendLayout();
            this.toolStripContainer1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // lstTests
            // 
            this.lstTests.CheckBoxes = true;
            this.lstTests.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.colIdx,
            this.colPriority,
            this.colResult,
            this.colTestClass,
            this.colTestMethod,
            this.colError});
            this.lstTests.ContextMenuStrip = this.contextMenuStrip1;
            this.lstTests.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lstTests.Font = new System.Drawing.Font("Century Gothic", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lstTests.FullRowSelect = true;
            this.lstTests.GridLines = true;
            this.lstTests.HideSelection = false;
            this.lstTests.LargeImageList = this.imageListUI;
            this.lstTests.Location = new System.Drawing.Point(0, 0);
            this.lstTests.Name = "lstTests";
            this.lstTests.Size = new System.Drawing.Size(1252, 579);
            this.lstTests.SmallImageList = this.imageListUI;
            this.lstTests.TabIndex = 0;
            this.lstTests.UseCompatibleStateImageBehavior = false;
            this.lstTests.View = System.Windows.Forms.View.Details;
            this.lstTests.ColumnClick += new System.Windows.Forms.ColumnClickEventHandler(this.lstTests_ColumnClick);
            this.lstTests.ItemChecked += new System.Windows.Forms.ItemCheckedEventHandler(this.lstTests_ItemChecked);
            this.lstTests.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.lstTests_MouseDoubleClick);
            // 
            // colIdx
            // 
            this.colIdx.Text = "Index";
            this.colIdx.Width = 78;
            // 
            // colPriority
            // 
            this.colPriority.Text = "Priority";
            // 
            // colResult
            // 
            this.colResult.Text = "Result";
            this.colResult.Width = 180;
            // 
            // colTestClass
            // 
            this.colTestClass.Text = "Test Class";
            this.colTestClass.Width = 190;
            // 
            // colTestMethod
            // 
            this.colTestMethod.Text = "Test Method";
            this.colTestMethod.Width = 202;
            // 
            // colError
            // 
            this.colError.Text = "Error Message";
            this.colError.Width = 730;
            // 
            // contextMenuStrip1
            // 
            this.contextMenuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.resetToolStripMenuItem,
            this.runToolStripMenuItem,
            this.skipToolStripMenuItem});
            this.contextMenuStrip1.Name = "contextMenuStrip1";
            this.contextMenuStrip1.Size = new System.Drawing.Size(103, 70);
            this.contextMenuStrip1.Opening += new System.ComponentModel.CancelEventHandler(this.contextMenuStrip1_Opening);
            // 
            // resetToolStripMenuItem
            // 
            this.resetToolStripMenuItem.Name = "resetToolStripMenuItem";
            this.resetToolStripMenuItem.Size = new System.Drawing.Size(102, 22);
            this.resetToolStripMenuItem.Text = "Reset";
            this.resetToolStripMenuItem.Click += new System.EventHandler(this.resetToolStripMenuItem_Click);
            // 
            // runToolStripMenuItem
            // 
            this.runToolStripMenuItem.Name = "runToolStripMenuItem";
            this.runToolStripMenuItem.Size = new System.Drawing.Size(102, 22);
            this.runToolStripMenuItem.Text = "Run";
            this.runToolStripMenuItem.Click += new System.EventHandler(this.runToolStripMenuItem_Click);
            // 
            // skipToolStripMenuItem
            // 
            this.skipToolStripMenuItem.Name = "skipToolStripMenuItem";
            this.skipToolStripMenuItem.Size = new System.Drawing.Size(102, 22);
            this.skipToolStripMenuItem.Text = "Skip";
            this.skipToolStripMenuItem.Click += new System.EventHandler(this.skipToolStripMenuItem_Click);
            // 
            // imageListUI
            // 
            this.imageListUI.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageListUI.ImageStream")));
            this.imageListUI.TransparentColor = System.Drawing.Color.Fuchsia;
            this.imageListUI.Images.SetKeyName(0, "aborted.png");
            this.imageListUI.Images.SetKeyName(1, "failed.png");
            this.imageListUI.Images.SetKeyName(2, "not_executed.png");
            this.imageListUI.Images.SetKeyName(3, "passed.png");
            this.imageListUI.Images.SetKeyName(4, "pending.png");
            this.imageListUI.Images.SetKeyName(5, "running.png");
            // 
            // toolStripContainer1
            // 
            // 
            // toolStripContainer1.ContentPanel
            // 
            this.toolStripContainer1.ContentPanel.Controls.Add(this.lstTests);
            this.toolStripContainer1.ContentPanel.Size = new System.Drawing.Size(1252, 579);
            this.toolStripContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.toolStripContainer1.Location = new System.Drawing.Point(0, 0);
            this.toolStripContainer1.Name = "toolStripContainer1";
            this.toolStripContainer1.Size = new System.Drawing.Size(1252, 604);
            this.toolStripContainer1.TabIndex = 1;
            this.toolStripContainer1.Text = "toolStripContainer1";
            // 
            // toolStripContainer1.TopToolStripPanel
            // 
            this.toolStripContainer1.TopToolStripPanel.Controls.Add(this.toolStrip1);
            // 
            // toolStrip1
            // 
            this.toolStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.btnShowAll,
            this.btnShowPassed,
            this.btnShowFailures,
            this.btnShowNotExecuted,
            this.toolStripSeparator1,
            this.btnRun,
            this.btnAbort,
            this.toolStripSeparator2,
            this.btnGradientTests,
            this.btnNonGradientTests,
            this.toolStripSeparator3,
            this.btnCurrent});
            this.toolStrip1.Location = new System.Drawing.Point(3, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(237, 25);
            this.toolStrip1.TabIndex = 0;
            // 
            // btnShowAll
            // 
            this.btnShowAll.Checked = true;
            this.btnShowAll.CheckOnClick = true;
            this.btnShowAll.CheckState = System.Windows.Forms.CheckState.Checked;
            this.btnShowAll.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnShowAll.Image = ((System.Drawing.Image)(resources.GetObject("btnShowAll.Image")));
            this.btnShowAll.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnShowAll.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnShowAll.Name = "btnShowAll";
            this.btnShowAll.Size = new System.Drawing.Size(23, 22);
            this.btnShowAll.Text = "Show All";
            this.btnShowAll.Click += new System.EventHandler(this.btnShowAll_Click);
            // 
            // btnShowPassed
            // 
            this.btnShowPassed.CheckOnClick = true;
            this.btnShowPassed.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnShowPassed.Image = ((System.Drawing.Image)(resources.GetObject("btnShowPassed.Image")));
            this.btnShowPassed.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnShowPassed.Name = "btnShowPassed";
            this.btnShowPassed.Size = new System.Drawing.Size(23, 22);
            this.btnShowPassed.Text = "Show Passed";
            this.btnShowPassed.Click += new System.EventHandler(this.btnShowPassed_Click);
            // 
            // btnShowFailures
            // 
            this.btnShowFailures.CheckOnClick = true;
            this.btnShowFailures.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnShowFailures.Image = ((System.Drawing.Image)(resources.GetObject("btnShowFailures.Image")));
            this.btnShowFailures.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnShowFailures.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnShowFailures.Name = "btnShowFailures";
            this.btnShowFailures.Size = new System.Drawing.Size(23, 22);
            this.btnShowFailures.Text = "Show Failures";
            this.btnShowFailures.Click += new System.EventHandler(this.btnShowFailures_Click);
            // 
            // btnShowNotExecuted
            // 
            this.btnShowNotExecuted.CheckOnClick = true;
            this.btnShowNotExecuted.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnShowNotExecuted.Image = ((System.Drawing.Image)(resources.GetObject("btnShowNotExecuted.Image")));
            this.btnShowNotExecuted.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnShowNotExecuted.Name = "btnShowNotExecuted";
            this.btnShowNotExecuted.Size = new System.Drawing.Size(23, 22);
            this.btnShowNotExecuted.Text = "Show Not Executed";
            this.btnShowNotExecuted.Click += new System.EventHandler(this.btnSelectNotExecuted_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // btnRun
            // 
            this.btnRun.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnRun.Image = ((System.Drawing.Image)(resources.GetObject("btnRun.Image")));
            this.btnRun.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnRun.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(23, 22);
            this.btnRun.Text = "Run Tests";
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // btnAbort
            // 
            this.btnAbort.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnAbort.Enabled = false;
            this.btnAbort.Image = ((System.Drawing.Image)(resources.GetObject("btnAbort.Image")));
            this.btnAbort.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnAbort.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnAbort.Name = "btnAbort";
            this.btnAbort.Size = new System.Drawing.Size(23, 22);
            this.btnAbort.Text = "Abort Tests";
            this.btnAbort.Click += new System.EventHandler(this.btnAbort_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // btnGradientTests
            // 
            this.btnGradientTests.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnGradientTests.Image = ((System.Drawing.Image)(resources.GetObject("btnGradientTests.Image")));
            this.btnGradientTests.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnGradientTests.Name = "btnGradientTests";
            this.btnGradientTests.Size = new System.Drawing.Size(23, 22);
            this.btnGradientTests.Text = "Gradient tests";
            this.btnGradientTests.Click += new System.EventHandler(this.btnGradientTests_Click);
            // 
            // btnNonGradientTests
            // 
            this.btnNonGradientTests.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnNonGradientTests.Image = ((System.Drawing.Image)(resources.GetObject("btnNonGradientTests.Image")));
            this.btnNonGradientTests.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnNonGradientTests.Name = "btnNonGradientTests";
            this.btnNonGradientTests.Size = new System.Drawing.Size(23, 22);
            this.btnNonGradientTests.Text = "Non-gradient tests";
            this.btnNonGradientTests.Click += new System.EventHandler(this.btnNonGradientTests_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
            // 
            // btnCurrent
            // 
            this.btnCurrent.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.btnCurrent.Image = ((System.Drawing.Image)(resources.GetObject("btnCurrent.Image")));
            this.btnCurrent.ImageScaling = System.Windows.Forms.ToolStripItemImageScaling.None;
            this.btnCurrent.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnCurrent.Name = "btnCurrent";
            this.btnCurrent.Size = new System.Drawing.Size(23, 22);
            this.btnCurrent.Text = "Go to running test.";
            this.btnCurrent.Click += new System.EventHandler(this.btnCurrent_Click);
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.tsTotalTests,
            this.toolStripStatusLabel3,
            this.tsFailedTests,
            this.tsProgress,
            this.tsProgressPct,
            this.toolStripStatusLabel2,
            this.tsTestingTime,
            this.pbItemProgress,
            this.tsItemProgress,
            this.lblActiveGPU,
            this.lblActiveGPUVal,
            this.lblActiveKernel,
            this.lblKernelHandleVal});
            this.statusStrip1.Location = new System.Drawing.Point(0, 579);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(1252, 25);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(63, 20);
            this.toolStripStatusLabel1.Text = "Total Tests:";
            this.toolStripStatusLabel1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // tsTotalTests
            // 
            this.tsTotalTests.AutoSize = false;
            this.tsTotalTests.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
            this.tsTotalTests.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenOuter;
            this.tsTotalTests.Name = "tsTotalTests";
            this.tsTotalTests.Size = new System.Drawing.Size(90, 20);
            this.tsTotalTests.Text = "0";
            this.tsTotalTests.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // toolStripStatusLabel3
            // 
            this.toolStripStatusLabel3.Name = "toolStripStatusLabel3";
            this.toolStripStatusLabel3.Size = new System.Drawing.Size(78, 20);
            this.toolStripStatusLabel3.Text = "   Failed Tests:";
            // 
            // tsFailedTests
            // 
            this.tsFailedTests.AutoSize = false;
            this.tsFailedTests.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
            this.tsFailedTests.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenOuter;
            this.tsFailedTests.Name = "tsFailedTests";
            this.tsFailedTests.Size = new System.Drawing.Size(50, 20);
            this.tsFailedTests.Text = "0";
            this.tsFailedTests.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // tsProgress
            // 
            this.tsProgress.ForeColor = System.Drawing.Color.Lime;
            this.tsProgress.Name = "tsProgress";
            this.tsProgress.Size = new System.Drawing.Size(200, 19);
            this.tsProgress.Step = 1;
            this.tsProgress.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            // 
            // tsProgressPct
            // 
            this.tsProgressPct.AutoSize = false;
            this.tsProgressPct.BackColor = System.Drawing.Color.Black;
            this.tsProgressPct.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tsProgressPct.ForeColor = System.Drawing.Color.Lime;
            this.tsProgressPct.Name = "tsProgressPct";
            this.tsProgressPct.Size = new System.Drawing.Size(55, 20);
            this.tsProgressPct.Text = "0.00 %";
            this.tsProgressPct.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(82, 20);
            this.toolStripStatusLabel2.Text = "  Testing Time:";
            // 
            // tsTestingTime
            // 
            this.tsTestingTime.AutoSize = false;
            this.tsTestingTime.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Top) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right) 
            | System.Windows.Forms.ToolStripStatusLabelBorderSides.Bottom)));
            this.tsTestingTime.BorderStyle = System.Windows.Forms.Border3DStyle.SunkenOuter;
            this.tsTestingTime.Name = "tsTestingTime";
            this.tsTestingTime.Size = new System.Drawing.Size(110, 20);
            this.tsTestingTime.Text = "0";
            this.tsTestingTime.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // pbItemProgress
            // 
            this.pbItemProgress.ForeColor = System.Drawing.Color.Lime;
            this.pbItemProgress.Name = "pbItemProgress";
            this.pbItemProgress.Size = new System.Drawing.Size(100, 19);
            this.pbItemProgress.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.pbItemProgress.Visible = false;
            // 
            // tsItemProgress
            // 
            this.tsItemProgress.AutoSize = false;
            this.tsItemProgress.BackColor = System.Drawing.Color.Black;
            this.tsItemProgress.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tsItemProgress.ForeColor = System.Drawing.Color.Lime;
            this.tsItemProgress.Name = "tsItemProgress";
            this.tsItemProgress.Size = new System.Drawing.Size(55, 20);
            this.tsItemProgress.Text = "0.00%";
            this.tsItemProgress.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.tsItemProgress.Visible = false;
            // 
            // lblActiveGPU
            // 
            this.lblActiveGPU.Name = "lblActiveGPU";
            this.lblActiveGPU.Size = new System.Drawing.Size(83, 20);
            this.lblActiveGPU.Text = "Active GPU ID:";
            this.lblActiveGPU.ToolTipText = "Active GPU ID";
            // 
            // lblActiveGPUVal
            // 
            this.lblActiveGPUVal.AutoSize = false;
            this.lblActiveGPUVal.BackColor = System.Drawing.Color.Black;
            this.lblActiveGPUVal.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblActiveGPUVal.ForeColor = System.Drawing.Color.Lime;
            this.lblActiveGPUVal.Name = "lblActiveGPUVal";
            this.lblActiveGPUVal.Size = new System.Drawing.Size(30, 20);
            this.lblActiveGPUVal.Text = "n/a";
            // 
            // timerUI
            // 
            this.timerUI.Enabled = true;
            this.timerUI.Interval = 250;
            this.timerUI.Tick += new System.EventHandler(this.timerUI_Tick);
            // 
            // lblActiveKernel
            // 
            this.lblActiveKernel.Name = "lblActiveKernel";
            this.lblActiveKernel.Size = new System.Drawing.Size(79, 20);
            this.lblActiveKernel.Text = "Active Kernel:";
            this.lblActiveKernel.ToolTipText = "Active kernel handle";
            // 
            // lblKernelHandleVal
            // 
            this.lblKernelHandleVal.AutoSize = false;
            this.lblKernelHandleVal.BackColor = System.Drawing.Color.Black;
            this.lblKernelHandleVal.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblKernelHandleVal.ForeColor = System.Drawing.Color.Lime;
            this.lblKernelHandleVal.Name = "lblKernelHandleVal";
            this.lblKernelHandleVal.Size = new System.Drawing.Size(38, 20);
            this.lblKernelHandleVal.Text = "n/a";
            // 
            // AutomatedTester
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.toolStripContainer1);
            this.Name = "AutomatedTester";
            this.Size = new System.Drawing.Size(1252, 604);
            this.Load += new System.EventHandler(this.AutomatedTester_Load);
            this.contextMenuStrip1.ResumeLayout(false);
            this.toolStripContainer1.ContentPanel.ResumeLayout(false);
            this.toolStripContainer1.TopToolStripPanel.ResumeLayout(false);
            this.toolStripContainer1.TopToolStripPanel.PerformLayout();
            this.toolStripContainer1.ResumeLayout(false);
            this.toolStripContainer1.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ListView lstTests;
        private System.Windows.Forms.ColumnHeader colResult;
        private System.Windows.Forms.ColumnHeader colTestClass;
        private System.Windows.Forms.ColumnHeader colTestMethod;
        private System.Windows.Forms.ColumnHeader colError;
        private System.Windows.Forms.ImageList imageListUI;
        private System.Windows.Forms.ToolStripContainer toolStripContainer1;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton btnShowAll;
        private System.Windows.Forms.ToolStripButton btnShowPassed;
        private System.Windows.Forms.ToolStripButton btnShowFailures;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton btnRun;
        private System.Windows.Forms.ToolStripButton btnAbort;
        private System.Windows.Forms.ToolStripButton btnShowNotExecuted;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.ToolStripStatusLabel tsTotalTests;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel3;
        private System.Windows.Forms.ToolStripStatusLabel tsFailedTests;
        private System.Windows.Forms.ToolStripProgressBar tsProgress;
        private System.Windows.Forms.ToolStripStatusLabel tsProgressPct;
        private System.Windows.Forms.Timer timerUI;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
        private System.Windows.Forms.ToolStripStatusLabel tsTestingTime;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip1;
        private System.Windows.Forms.ToolStripMenuItem resetToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem runToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripButton btnGradientTests;
        private System.Windows.Forms.ToolStripButton btnNonGradientTests;
        private System.Windows.Forms.ColumnHeader colIdx;
        private System.Windows.Forms.ToolStripMenuItem skipToolStripMenuItem;
        private System.Windows.Forms.ToolStripProgressBar pbItemProgress;
        private System.Windows.Forms.ToolStripStatusLabel tsItemProgress;
        private System.Windows.Forms.ToolStripStatusLabel lblActiveGPU;
        private System.Windows.Forms.ToolStripStatusLabel lblActiveGPUVal;
        private System.Windows.Forms.ColumnHeader colPriority;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripButton btnCurrent;
        private System.Windows.Forms.ToolStripStatusLabel lblActiveKernel;
        private System.Windows.Forms.ToolStripStatusLabel lblKernelHandleVal;
    }
}
