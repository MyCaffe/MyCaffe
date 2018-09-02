using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.gym
{
    public partial class MyCaffeGymRegistry : Component
    {
        Control m_ctrlParent;
        Dictionary<string, List<FormGym>> m_rgGym = new Dictionary<string, List<FormGym>>();
        GymCollection m_gymCollection = new GymCollection();
        Log m_log;
        object m_sync = new object();

        public event EventHandler<ObservationArgs> OnObservation;

        delegate int fnopen(string strName, bool bAutoStart, bool bShowUi, bool bShowOnlyFirst, double[] rgdfInit);
        delegate void fnopenUi(string strName, int nIdx);
        delegate bool fnclose(string strName, int nIdx);

        public MyCaffeGymRegistry()
        {
            InitializeComponent();
        }

        public MyCaffeGymRegistry(IContainer container)
        {
            container.Add(this);

            InitializeComponent();            
        }

        public void Initialize(Control ctrlParent, Log log)
        {
            m_ctrlParent = ctrlParent;
            m_gymCollection.Load();
            m_log = log;
        }

        public FormGym Find(string strName, int nIdx)
        {
            if (!m_rgGym.ContainsKey(strName))
                return null;

            if (m_rgGym[strName].Count <= nIdx)
                return null;

            return m_rgGym[strName][nIdx];
        }

        public bool RemoveAll(string strName)
        {
            if (!m_rgGym.ContainsKey(strName))
                return false;

            m_rgGym[strName].Clear();
            m_rgGym.Remove(strName);

            return true;
        }

        public int Open()
        {
            FormGyms dlg = new FormGyms();

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                return Open(dlg.SelectedGym.Name, false, true, false, null);

            return -1;
        }

        public int Open(string strName, bool bAutoStart, bool bShowUI, bool bShowOnlyFirst, double[] rgdfInit)
        {
            return (int)m_ctrlParent.Invoke(new fnopen(open), strName, bAutoStart, bShowUI, bShowOnlyFirst, rgdfInit);
        }

        public void Open(string strName, int nIdx)
        {
            m_ctrlParent.Invoke(new fnopenUi(openUi), strName, nIdx);
        }

        private void openUi(string strName, int nIdx)
        {
            FormGym gym = Find(strName, nIdx);

            if (gym != null)
            {
                gym.Show();
                gym.BringToFront();
            }
        }

        private int open(string strName, bool bAutoStart, bool bShowUi, bool bShowOnlyFirst, double[] rgdfInit)
        {
            IXMyCaffeGym igym = m_gymCollection.Find(strName);
            if (igym == null)
                return -1;

            MyCaffeGymControl ctrl = new MyCaffeGymControl(m_log, rgdfInit);
            ctrl.OnObservation += ctrl_OnObservation;
            ctrl.Initialize(igym.Clone());
            FormGym dlg = new FormGym(ctrl);

            lock (m_sync)
            {
                if (!m_rgGym.ContainsKey(strName))
                    m_rgGym.Add(strName, new List<FormGym>());

                m_rgGym[strName].Add(dlg);
                ctrl.Index = m_rgGym[strName].Count - 1;
            }

            if (bAutoStart)
            {
                dlg.GymControl.Start();
                dlg.GymControl.Reset();
            }

            if (bShowUi)
            {
                if (!bShowOnlyFirst || ctrl.Index == 0)
                    dlg.Show();
            }

            return ctrl.Index;
        }

        private void ctrl_OnObservation(object sender, ObservationArgs e)
        {
            if (OnObservation != null)
                OnObservation(sender, e);
        }

        public bool CloseAll(string strName)
        {
            if (!string.IsNullOrEmpty(strName))
            {
                if (!m_rgGym.ContainsKey(strName))
                    return false;
            }

            List<string> rgstrRemove = new List<string>();

            foreach (KeyValuePair<string, List<FormGym>> kv in m_rgGym)
            {
                if (string.IsNullOrEmpty(strName) || kv.Key == strName)
                {
                    int nCount = kv.Value.Count;

                    for (int i = 0; i < nCount; i++)
                    {
                        Close(kv.Key, i);
                    }

                    rgstrRemove.Add(kv.Key);
                }
            }

            foreach (string strRemove in rgstrRemove)
            {
                RemoveAll(strRemove);
            }

            if (string.IsNullOrEmpty(strName))
                m_rgGym.Clear();

            return true;
        }

        public bool Close(string strName, int nIdx)
        {
            return (bool)m_ctrlParent.Invoke(new fnclose(close), strName, nIdx);
        }

        private bool close(string strName, int nIdx)
        {
            FormGym dlg = Find(strName, nIdx);
            if (dlg == null)
                return false;

            dlg.Stop();
            dlg.Close();
            m_rgGym[strName][nIdx] = null;

            for (int i = 0; i < m_rgGym[strName].Count; i++)
            {
                if (m_rgGym[strName][i] != null)
                    return true;
            }

            return true;
        }

        public DatasetDescriptor GetDataset(string strName, int nType)
        {
            IXMyCaffeGym igym = m_gymCollection.Find(strName);
            if (igym == null)
                return null;

            return igym.GetDataset((DATA_TYPE)nType);
        }

        public Dictionary<string, int> GetActionSpace(string strName)
        {
            IXMyCaffeGym igym = m_gymCollection.Find(strName);
            if (igym == null)
                return null;

            return igym.GetActionSpace();
        }

        public bool Run(string strName, int nIdx, int nAction)
        {
            FormGym dlg = Find(strName, nIdx);
            if (dlg == null)
                return false;

            dlg.GymControl.RunAction(nAction);
            return true;
        }

        public bool Reset(string strName, int nIdx)
        {
            FormGym dlg = Find(strName, nIdx);
            if (dlg == null)
                return false;

            dlg.GymControl.Reset();
            return true;
        }

        public Observation GetObservation(string strName, int nIdx, bool bReset)
        {
            FormGym dlg = Find(strName, nIdx);
            if (dlg == null)
                return null;

            return dlg.GymControl.GetLastObservation(bReset);
        }
    }

    public class MyCaffeGymRegistrar
    {
        static MyCaffeGymRegistry m_registry = new MyCaffeGymRegistry();
        static AutoResetEvent m_evtCancel = new AutoResetEvent(false);

        public static void Initialize(Control ctrlParent, Log log)
        {
            m_registry.Initialize(ctrlParent, log);
            Task.Factory.StartNew(new Action(hostingThread));
        }

        public static MyCaffeGymRegistry Registry
        {
            get { return m_registry; }
        }

        public static void Shutdown()
        {
            m_evtCancel.Set();
        }

        private static void hostingThread()
        {
            // Create a service host with an named pipe endpoint
            using (var host = new ServiceHost(typeof(MyCaffeGymService), new Uri("net.pipe://localhost")))
            {
                host.AddServiceEndpoint(typeof(IXMyCaffeGymService), new NetNamedPipeBinding(), "MyCaffeGymService");
                host.Open();

                while (!m_evtCancel.WaitOne(100))
                {
                }

                host.Close();
            }
        }
    }
}
