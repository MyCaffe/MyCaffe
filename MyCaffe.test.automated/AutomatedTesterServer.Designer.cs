namespace MyCaffe.test.automated
{
    partial class AutomatedTesterServer
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
            this.m_bw = new System.ComponentModel.BackgroundWorker();
            // 
            // m_bw
            // 
            this.m_bw.WorkerReportsProgress = true;
            this.m_bw.WorkerSupportsCancellation = true;
            this.m_bw.DoWork += new System.ComponentModel.DoWorkEventHandler(this.m_bw_DoWork);
            this.m_bw.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.m_bw_ProgressChanged);
            this.m_bw.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.m_bw_RunWorkerCompleted);

        }

        #endregion

        private System.ComponentModel.BackgroundWorker m_bw;
    }
}
