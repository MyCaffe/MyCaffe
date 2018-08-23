using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;
using MyCaffe.basecode;
using System.Collections;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.gym
{
    public partial class MyCaffeGymControl : UserControl
    {
        Log m_log;
        IXMyCaffeGym m_igym;
        Bitmap m_bmp = null;
        Tuple<Tuple<double,double,double>[], double, bool> m_state;
        bool m_bStopping = false;
        bool m_bRendering = false;
        Observation m_observation = null;

        public MyCaffeGymControl(Log log)
        {
            m_log = log;
            InitializeComponent();
        }

        public void Initialize(IXMyCaffeGym igym)
        {
            m_igym = igym;
        }

        public void Start()
        {
            if (m_bwGym.IsBusy)
                return;

            m_bwGym.RunWorkerAsync(m_igym);
        }

        public void Stop()
        {
            m_bwGym.CancelAsync();
            m_bStopping = true;
        }

        public string GymName
        {
            get { return m_igym.Name; }
        }

        public bool IsRunning
        {
            get { return m_bwGym.IsBusy; }
        }

        public bool IsStopping
        {
            get { return m_bStopping; }
        }

        private void MyCaffeGymControl_Resize(object sender, EventArgs e)
        {
        }

        private void MyCaffeGymControl_Load(object sender, EventArgs e)
        {
        }

        public void Render()
        {
            if (m_bRendering)
                return;

            try
            {             
                m_bRendering = true;
                m_bmp = m_igym.Render(Width, Height);

                if (m_state != null)
                    m_observation = new Observation(new Bitmap(m_bmp), m_state.Item1, m_state.Item2, m_state.Item3);

                if (IsHandleCreated && Visible)
                    Invalidate(true);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_bRendering = false;
            }
        }

        public DatasetDescriptor GetDataset(int nType)
        {
            return m_igym.GetDataset((DATA_TYPE)nType);
        }

        public void Reset()
        {
            m_observation = null;
            m_igym.Reset();
        }

        public void RunAction(int nAction)
        {
            m_igym.AddAction(nAction);
        }

        public Dictionary<string, int> GetActionSpace()
        {
            return m_igym.GetActionSpace();
        }

        public Observation GetLastObservation()
        {
            return m_observation;
        }

        private void m_bwGym_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            Render();
        }

        private void m_bwGym_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            Render();
        }

        private void m_bwGym_DoWork(object sender, DoWorkEventArgs e)
        {
            BackgroundWorker bw = sender as BackgroundWorker;
            IXMyCaffeGym igym = e.Argument as IXMyCaffeGym;

            igym.Initialize(m_log);

            while (!bw.CancellationPending)
            {
                m_state = igym.Step();
                bw.ReportProgress(1);
                Thread.Sleep(20);   // roughly 50 frames/second
            }

            igym.Close();
        }

        private void MyCaffeGymControl_Paint(object sender, PaintEventArgs e)
        {
            if (m_bmp != null)
                e.Graphics.DrawImage(m_bmp, new Point(0, 0));
        }
    }
}
