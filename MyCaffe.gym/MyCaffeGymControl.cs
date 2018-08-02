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

namespace MyCaffe.gym
{
    public partial class MyCaffeGymControl : UserControl
    {
        Log m_log;
        IxMycaffeGym m_igym;
        Bitmap m_bmp = null;
        Tuple<double[], double, bool> m_state;
        bool m_bStopping = false;
        Observations m_rgObservations = new Observations(10);

        public event EventHandler<OnObservationArgs> OnObservation;

        public MyCaffeGymControl(Log log)
        {
            m_log = log;
            InitializeComponent();
        }

        public void Initialize(IxMycaffeGym igym)
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
            m_bmp = m_igym.Render(Width, Height);

            if (m_state != null)
            {
                m_rgObservations.Add(new Observation(m_bmp, m_state.Item1, m_state.Item2, m_state.Item3));

                if (OnObservation != null)
                    OnObservation(this, new OnObservationArgs(m_rgObservations));
            }

            Invalidate(true);
        }

        public void Reset()
        {
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
            return m_rgObservations.LastObservation;
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
            IxMycaffeGym igym = e.Argument as IxMycaffeGym;

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

    public class OnObservationArgs : EventArgs
    {
        Observations m_rgObservations;

        public OnObservationArgs(Observations rgObs)
        {
            m_rgObservations = rgObs;
        }

        public Observations Observations
        {
            get { return m_rgObservations; }
        }
    }

    public class Observations : IEnumerable<Observation>
    {
        List<Observation> m_rgObservation = new List<Observation>();
        object m_syncObj = new object();
        int m_nMax;

        public Observations(int nMax)
        {
            m_nMax = nMax;
        }

        public void Add(Observation obs)
        {
            lock (m_syncObj)
            {
                m_rgObservation.Add(obs);

                while (m_rgObservation.Count > m_nMax)
                {
                    m_rgObservation.RemoveAt(0);
                }
            }
        }

        public Observation LastObservation
        {
            get
            {
                lock (m_syncObj)
                {
                    if (m_rgObservation.Count == 0)
                        return null;

                    return m_rgObservation[m_rgObservation.Count - 1];
                }
            }
        }

        public IEnumerator<Observation> GetEnumerator()
        {
            return m_rgObservation.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgObservation.GetEnumerator();
        }

        public int Count
        {
            get { return m_rgObservation.Count; }
        }

        public Observation this[int nIdx]
        {
            get { return m_rgObservation[nIdx]; }
        }
    }
}
