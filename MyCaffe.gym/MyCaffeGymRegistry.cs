using MyCaffe.basecode;
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
        List<FormGym> m_rgGym = new List<FormGym>();

        delegate bool fnopen(string strName, bool bAutoStart, bool bShowUi);
        delegate bool fnclose(string strName);

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

            GymCollection col = new GymCollection();

            col.Load();

            foreach (IXMyCaffeGym igym in col)
            {
                MyCaffeGymControl ctrl = new MyCaffeGymControl(log);
                ctrl.Initialize(igym);
                FormGym dlg = new FormGym(ctrl);
                m_rgGym.Add(dlg);
            }
        }

        public FormGym Find(string strName)
        {
            foreach (FormGym dlg in m_rgGym)
            {
                if (dlg.GymName == strName)
                    return dlg;
            }

            return null;
        }

        public void Open()
        {
            FormGyms dlg = new FormGyms();

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                Open(dlg.SelectedGym.Name, false, true);
            }
        }

        public bool Open(string strName, bool bAutoStart, bool bShowUI)
        {
            return (bool)m_ctrlParent.Invoke(new fnopen(open), strName, bAutoStart, bShowUI);
        }

        private bool open(string strName, bool bAutoStart, bool bShowUi)
        {
            FormGym dlg = Find(strName);
            if (dlg == null)
                return false;

            if (bAutoStart)
            {
                dlg.GymControl.Start();
                dlg.GymControl.Reset();
            }

            if (bShowUi)
                dlg.Show();

            return true;
        }

        public bool Close(string strName)
        {
            return (bool)m_ctrlParent.Invoke(new fnclose(close), strName);
        }

        private bool close(string strName)
        {
            FormGym dlg = Find(strName);
            if (dlg == null)
                return false;

            dlg.Hide();
            return true;
        }

        public Dictionary<string, int> GetActionSpace(string strName)
        {
            FormGym dlg = Find(strName);
            if (dlg == null)
                return null;

            return dlg.GymControl.GetActionSpace();
        }

        public bool Run(string strName, int nAction)
        {
            FormGym dlg = Find(strName);
            if (dlg == null)
                return false;

            dlg.GymControl.RunAction(nAction);
            return true;
        }

        public Observation GetObservation(string strName)
        {
            FormGym dlg = Find(strName);
            if (dlg == null)
                return null;

            return dlg.GymControl.GetLastObservation();
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
