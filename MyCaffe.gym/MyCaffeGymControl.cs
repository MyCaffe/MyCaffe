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
using System.Diagnostics;

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
        double[] m_rgdfInit = null;
        int m_nIndex = -1;

        public event EventHandler<ObservationArgs> OnObservation;

        public MyCaffeGymControl(Log log, double[] rgdfInit)
        {
            m_rgdfInit = rgdfInit;
            m_log = log;
            InitializeComponent();
        }

        public void Initialize(IXMyCaffeGym igym)
        {
            m_igym = igym;
        }

        public int Index
        {
            get { return m_nIndex; }
            set { m_nIndex = value; }
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

        public void Render(Bitmap bmp)
        {
            if (m_bRendering)
                return;

            try
            {             
                m_bRendering = true;
                m_bmp = bmp;

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
            m_state = m_igym.Reset();

            if (OnObservation != null)
            {
                Bitmap bmpAction;
                m_igym.Render(Width, Height, out bmpAction);
                OnObservation(this, new ObservationArgs(m_igym.Name, m_nIndex, new Observation(bmpAction, m_state.Item1, m_state.Item2, m_state.Item3)));
            }
        }

        public void RunAction(int nAction)
        {
            m_igym.Run(nAction);
        }

        public Dictionary<string, int> GetActionSpace()
        {
            return m_igym.GetActionSpace();
        }

        public Observation GetLastObservation(bool bReset)
        {
            Observation obs = m_observation;

            if (bReset)
                m_observation = null;

            return obs;
        }

        private void m_bwGym_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
        }

        private void m_bwGym_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            Bitmap bmp = e.UserState as Bitmap;
            Render(bmp);
        }

        private void m_bwGym_DoWork(object sender, DoWorkEventArgs e)
        {
            BackgroundWorker bw = sender as BackgroundWorker;
            IXMyCaffeGym igym = e.Argument as IXMyCaffeGym;

            igym.Initialize(m_log, m_rgdfInit);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            while (!bw.CancellationPending)
            {
                m_state = igym.Step();
                Bitmap bmpAction;
                Bitmap bmp = m_igym.Render(Width, Height, out bmpAction);

                if (OnObservation != null)
                    OnObservation(this, new ObservationArgs(m_igym.Name, m_nIndex, new Observation(bmpAction, m_state.Item1, m_state.Item2, m_state.Item3)));

                bw.ReportProgress(1, bmp);
                Thread.Sleep(20); // roughly 50 frames a second.
            }

            igym.Close();
        }

        private void MyCaffeGymControl_Paint(object sender, PaintEventArgs e)
        {
            if (m_bmp != null)
                e.Graphics.DrawImage(m_bmp, new Point(0, 0));
        }
    }

    public class ObservationArgs : EventArgs
    {
        string m_strName;
        int m_nIdx;
        Observation m_obs;

        public ObservationArgs(string strName, int nIdx, Observation obs)
        {
            m_strName = strName;
            m_nIdx = nIdx;
            m_obs = obs;
        }

        public string Name
        {
            get { return m_strName; }
        }

        public int Index
        {
            get { return m_nIdx; }
        }

        public Observation Observation
        {
            get { return m_obs; }
        }
    }
}
