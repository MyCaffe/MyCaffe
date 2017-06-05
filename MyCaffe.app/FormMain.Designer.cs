namespace MyCaffe.app
{
    partial class FormMain
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormMain));
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.databaseToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.createDatabaseToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.loadMNISTToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.loadCIFAR10ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.testToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.runAutotestsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.createMyCaffeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.deviceInformationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.destroyMyCaffeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.trainMNISTToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.testMNISTToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.abortToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.helpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.aboutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFileDialogAutoTests = new System.Windows.Forms.OpenFileDialog();
            this.edtStatus = new System.Windows.Forms.TextBox();
            this.m_bwLoadMnistDatabase = new System.ComponentModel.BackgroundWorker();
            this.m_bwProcess = new System.ComponentModel.BackgroundWorker();
            this.m_bwLoadCiFar10Database = new System.ComponentModel.BackgroundWorker();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.databaseToolStripMenuItem,
            this.testToolStripMenuItem,
            this.helpToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(588, 24);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
            this.fileToolStripMenuItem.Text = "&File";
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(92, 22);
            this.exitToolStripMenuItem.Text = "E&xit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // databaseToolStripMenuItem
            // 
            this.databaseToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.createDatabaseToolStripMenuItem,
            this.loadMNISTToolStripMenuItem,
            this.loadCIFAR10ToolStripMenuItem});
            this.databaseToolStripMenuItem.Name = "databaseToolStripMenuItem";
            this.databaseToolStripMenuItem.Size = new System.Drawing.Size(67, 20);
            this.databaseToolStripMenuItem.Text = "&Database";
            // 
            // createDatabaseToolStripMenuItem
            // 
            this.createDatabaseToolStripMenuItem.Name = "createDatabaseToolStripMenuItem";
            this.createDatabaseToolStripMenuItem.Size = new System.Drawing.Size(161, 22);
            this.createDatabaseToolStripMenuItem.Text = "&Create Database";
            this.createDatabaseToolStripMenuItem.Click += new System.EventHandler(this.createDatabaseToolStripMenuItem_Click);
            // 
            // loadMNISTToolStripMenuItem
            // 
            this.loadMNISTToolStripMenuItem.Name = "loadMNISTToolStripMenuItem";
            this.loadMNISTToolStripMenuItem.Size = new System.Drawing.Size(161, 22);
            this.loadMNISTToolStripMenuItem.Text = "Load MNIST...";
            this.loadMNISTToolStripMenuItem.Click += new System.EventHandler(this.loadMNISTToolStripMenuItem_Click);
            // 
            // loadCIFAR10ToolStripMenuItem
            // 
            this.loadCIFAR10ToolStripMenuItem.Name = "loadCIFAR10ToolStripMenuItem";
            this.loadCIFAR10ToolStripMenuItem.Size = new System.Drawing.Size(161, 22);
            this.loadCIFAR10ToolStripMenuItem.Text = "Load CIFAR-10...";
            this.loadCIFAR10ToolStripMenuItem.Click += new System.EventHandler(this.loadCIFAR10ToolStripMenuItem_Click);
            // 
            // testToolStripMenuItem
            // 
            this.testToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.runAutotestsToolStripMenuItem,
            this.toolStripSeparator2,
            this.createMyCaffeToolStripMenuItem,
            this.deviceInformationToolStripMenuItem,
            this.destroyMyCaffeToolStripMenuItem,
            this.trainMNISTToolStripMenuItem,
            this.testMNISTToolStripMenuItem,
            this.abortToolStripMenuItem});
            this.testToolStripMenuItem.Name = "testToolStripMenuItem";
            this.testToolStripMenuItem.Size = new System.Drawing.Size(41, 20);
            this.testToolStripMenuItem.Text = "&Test";
            // 
            // runAutotestsToolStripMenuItem
            // 
            this.runAutotestsToolStripMenuItem.Name = "runAutotestsToolStripMenuItem";
            this.runAutotestsToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.runAutotestsToolStripMenuItem.Text = "&Run Autotests";
            this.runAutotestsToolStripMenuItem.Click += new System.EventHandler(this.runAutotestsToolStripMenuItem_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(172, 6);
            // 
            // createMyCaffeToolStripMenuItem
            // 
            this.createMyCaffeToolStripMenuItem.Name = "createMyCaffeToolStripMenuItem";
            this.createMyCaffeToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.createMyCaffeToolStripMenuItem.Text = "Create MyCaffe";
            this.createMyCaffeToolStripMenuItem.Click += new System.EventHandler(this.createMyCaffeToolStripMenuItem_Click);
            // 
            // deviceInformationToolStripMenuItem
            // 
            this.deviceInformationToolStripMenuItem.Enabled = false;
            this.deviceInformationToolStripMenuItem.Name = "deviceInformationToolStripMenuItem";
            this.deviceInformationToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.deviceInformationToolStripMenuItem.Text = "Device Information";
            this.deviceInformationToolStripMenuItem.Click += new System.EventHandler(this.deviceInformationToolStripMenuItem_Click);
            // 
            // destroyMyCaffeToolStripMenuItem
            // 
            this.destroyMyCaffeToolStripMenuItem.Enabled = false;
            this.destroyMyCaffeToolStripMenuItem.Name = "destroyMyCaffeToolStripMenuItem";
            this.destroyMyCaffeToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.destroyMyCaffeToolStripMenuItem.Text = "Destroy MyCaffe";
            this.destroyMyCaffeToolStripMenuItem.Click += new System.EventHandler(this.destroyMyCaffeToolStripMenuItem_Click);
            // 
            // trainMNISTToolStripMenuItem
            // 
            this.trainMNISTToolStripMenuItem.Enabled = false;
            this.trainMNISTToolStripMenuItem.Name = "trainMNISTToolStripMenuItem";
            this.trainMNISTToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.trainMNISTToolStripMenuItem.Text = "Train MNIST";
            this.trainMNISTToolStripMenuItem.Click += new System.EventHandler(this.trainMNISTToolStripMenuItem_Click);
            // 
            // testMNISTToolStripMenuItem
            // 
            this.testMNISTToolStripMenuItem.Enabled = false;
            this.testMNISTToolStripMenuItem.Name = "testMNISTToolStripMenuItem";
            this.testMNISTToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.testMNISTToolStripMenuItem.Text = "Test MNIST";
            this.testMNISTToolStripMenuItem.Click += new System.EventHandler(this.testMNISTToolStripMenuItem_Click);
            // 
            // abortToolStripMenuItem
            // 
            this.abortToolStripMenuItem.Enabled = false;
            this.abortToolStripMenuItem.Name = "abortToolStripMenuItem";
            this.abortToolStripMenuItem.Size = new System.Drawing.Size(175, 22);
            this.abortToolStripMenuItem.Text = "Abort";
            this.abortToolStripMenuItem.Click += new System.EventHandler(this.abortToolStripMenuItem_Click);
            // 
            // helpToolStripMenuItem
            // 
            this.helpToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.aboutToolStripMenuItem});
            this.helpToolStripMenuItem.Name = "helpToolStripMenuItem";
            this.helpToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
            this.helpToolStripMenuItem.Text = "&Help";
            // 
            // aboutToolStripMenuItem
            // 
            this.aboutToolStripMenuItem.Name = "aboutToolStripMenuItem";
            this.aboutToolStripMenuItem.Size = new System.Drawing.Size(107, 22);
            this.aboutToolStripMenuItem.Text = "&About";
            this.aboutToolStripMenuItem.Click += new System.EventHandler(this.aboutToolStripMenuItem_Click);
            // 
            // openFileDialogAutoTests
            // 
            this.openFileDialogAutoTests.DefaultExt = "dll";
            this.openFileDialogAutoTests.Filter = "Auto Test Files (*.dll)|*.dll||";
            this.openFileDialogAutoTests.Title = "Select the Auto Test DLL";
            // 
            // edtStatus
            // 
            this.edtStatus.Dock = System.Windows.Forms.DockStyle.Fill;
            this.edtStatus.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.edtStatus.Location = new System.Drawing.Point(0, 24);
            this.edtStatus.Multiline = true;
            this.edtStatus.Name = "edtStatus";
            this.edtStatus.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.edtStatus.Size = new System.Drawing.Size(588, 428);
            this.edtStatus.TabIndex = 1;
            // 
            // m_bwLoadMnistDatabase
            // 
            this.m_bwLoadMnistDatabase.WorkerReportsProgress = true;
            this.m_bwLoadMnistDatabase.WorkerSupportsCancellation = true;
            this.m_bwLoadMnistDatabase.DoWork += new System.ComponentModel.DoWorkEventHandler(this.m_bwLoadDatabase_DoWork);
            this.m_bwLoadMnistDatabase.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.m_bw_ProgressChanged);
            this.m_bwLoadMnistDatabase.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.m_bw_RunWorkerCompleted);
            // 
            // m_bwProcess
            // 
            this.m_bwProcess.WorkerReportsProgress = true;
            this.m_bwProcess.WorkerSupportsCancellation = true;
            this.m_bwProcess.DoWork += new System.ComponentModel.DoWorkEventHandler(this.m_bwProcess_DoWork);
            this.m_bwProcess.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.m_bw_ProgressChanged);
            this.m_bwProcess.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.m_bw_RunWorkerCompleted);
            // 
            // m_bwLoadCiFar10Database
            // 
            this.m_bwLoadCiFar10Database.WorkerReportsProgress = true;
            this.m_bwLoadCiFar10Database.WorkerSupportsCancellation = true;
            this.m_bwLoadCiFar10Database.DoWork += new System.ComponentModel.DoWorkEventHandler(this.m_bwLoadCiFar10Database_DoWork);
            this.m_bwLoadCiFar10Database.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.m_bw_ProgressChanged);
            this.m_bwLoadCiFar10Database.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.m_bw_RunWorkerCompleted);
            // 
            // FormMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(588, 452);
            this.Controls.Add(this.edtStatus);
            this.Controls.Add(this.menuStrip1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "FormMain";
            this.Text = "MyCaffe";
            this.Load += new System.EventHandler(this.FormMain_Load);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem databaseToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem createDatabaseToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem testToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem runAutotestsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem helpToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem aboutToolStripMenuItem;
        private System.Windows.Forms.OpenFileDialog openFileDialogAutoTests;
        private System.Windows.Forms.TextBox edtStatus;
        private System.Windows.Forms.ToolStripMenuItem deviceInformationToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem loadMNISTToolStripMenuItem;
        private System.ComponentModel.BackgroundWorker m_bwLoadMnistDatabase;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripMenuItem trainMNISTToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem testMNISTToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem abortToolStripMenuItem;
        private System.ComponentModel.BackgroundWorker m_bwProcess;
        private System.Windows.Forms.ToolStripMenuItem createMyCaffeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem destroyMyCaffeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem loadCIFAR10ToolStripMenuItem;
        private System.ComponentModel.BackgroundWorker m_bwLoadCiFar10Database;
    }
}

