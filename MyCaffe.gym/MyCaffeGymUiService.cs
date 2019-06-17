using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The MyCaffeGymUiService provides the service used to show the Gym visualizations.
    /// </summary>
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.PerSession)]
    public class MyCaffeGymUiService : IXMyCaffeGymUiService
    {
        static Dictionary<int, FormGym> m_rgGyms = new Dictionary<int, FormGym>();
        static object m_syncObjGym = new object();
        IXMyCaffeGymUiCallback m_callback;

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeGymUiService()
        {
            m_callback = OperationContext.Current.GetCallbackChannel<IXMyCaffeGymUiCallback>();
        }

        private void dlg_FormClosing(object sender, System.Windows.Forms.FormClosingEventArgs e)
        {
            if (m_callback != null)
            {
                m_callback.Closing();
                m_callback = null;
            }
        }

        /// <summary>
        /// Close the user interface of a Gym.
        /// </summary>
        /// <param name="nId">Specifies the Gym id (used when multiple Gym's of the same name are used).</param>
        public void CloseUi(int nId)
        {
            if (!m_rgGyms.ContainsKey(nId))
                return;

            m_rgGyms[nId].Hide();

            if (m_callback != null)
            {
                m_callback.Closing();
                m_callback = null;
            }
        }

        /// <summary>
        /// Open the Gym user interface.
        /// </summary>
        /// <param name="strName">Specifies the Gym name.</param>
        /// <param name="nId">Specifies the ID of the Gym.</param>
        /// <returns>The ID of the Gym opened is returned.</returns>
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

        /// <summary>
        /// Render an observation on the Gym user-interface.
        /// </summary>
        /// <param name="nId">Specifies the Gym ID.</param>
        /// <param name="obs">Specifies the Observation to visualize.</param>
        public void Render(int nId, Observation obs)
        {
            if (!m_rgGyms.ContainsKey(nId))
                return;

            if (!m_rgGyms[nId].Visible)
                return;

            if (obs.RequireDisplayImage && obs.ImageDisplay == null)
                return;

            double[] rgData = obs.State;

            m_rgGyms[nId].Render(rgData, obs.ImageDisplay, obs.Image);
        }

        /// <summary>
        /// Returns <i>true</i> when the visualization is open, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nId">Specifies the Gym ID.</param>
        /// <returns>Returns <i>true</i> when the visualization is open, <i>false</i> otherwise.</returns>
        public bool IsOpen(int nId)
        {
            if (!m_rgGyms.ContainsKey(nId))
                return false;

            return m_rgGyms[nId].Visible;
        }
    }
}
