using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.PerSession)]
    public class MyCaffeGymUiService : IXMyCaffeGymUiService
    {
        static Dictionary<int, FormGym> m_rgGyms = new Dictionary<int, FormGym>();
        static object m_syncObjGym = new object();
        IXMyCaffeGymUiCallback m_callback;


        public MyCaffeGymUiService()
        {
            m_callback = OperationContext.Current.GetCallbackChannel<IXMyCaffeGymUiCallback>();
        }

        private void dlg_FormClosing(object sender, System.Windows.Forms.FormClosingEventArgs e)
        {
            m_callback.Closing();
        }

        public void CloseUi(int nId)
        {
            if (!m_rgGyms.ContainsKey(nId))
                return;

            m_rgGyms[nId].Hide();
            m_callback.Closing();
        }

        public int OpenUi(string strName, int nId)
        {
            lock (m_syncObjGym)
            {
                if (m_rgGyms.ContainsKey(nId))
                {
                    m_rgGyms[nId].BringToFront();
                    return nId;
                }

                FormGym dlg = new FormGym(strName);
                dlg.FormClosing += dlg_FormClosing;
                dlg.Show();
                dlg.BringToFront();

                nId = m_rgGyms.Count;
                m_rgGyms.Add(nId, dlg);
            }

            return nId;
        }

        public void Render(int nId, Observation obs)
        {
            if (!m_rgGyms.ContainsKey(nId))
                return;

            if (!m_rgGyms[nId].Visible)
                return;

            double[] rgData = obs.State.Select(p => p.Item1).ToArray();

            m_rgGyms[nId].Render(rgData, obs.Image);
        }
    }
}
