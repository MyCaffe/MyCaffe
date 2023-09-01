using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.ServiceModel.Description;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The MyCaffeGymUiProxy is used to interact with the MyCaffeGymUiService.
    /// </summary>
    public class MyCaffeGymUiProxy : DuplexClientBase<IXMyCaffeGymUiService>       
    {
        object m_sync = new object();
        bool m_bOpen = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="ctx">Specifies the context.</param>
        /// <param name="binding">Optionally, specifies the binding.  Specifying a binding can be useful when you need to set larger buffer sizes.</param>
        public MyCaffeGymUiProxy(InstanceContext ctx, NetNamedPipeBinding binding = null)
            : base(ctx, new ServiceEndpoint(ContractDescription.GetContract(typeof(IXMyCaffeGymUiService)),
                   (binding == null) ? new NetNamedPipeBinding() : binding, new EndpointAddress("net.pipe://localhost/MyCaffeGym/gymui")))
        {
        }

        /// <summary>
        /// Open the Gym user interface.
        /// </summary>
        /// <param name="strName">Specifies the Gym name.</param>
        /// <param name="nId">Specifies the ID of the Gym.</param>
        /// <param name="bStartRecording">Optionally, specifies to open with recording on.</param>
        /// <returns>The ID of the Gym opened is returned.</returns>
        public int OpenUi(string strName, int nId, bool bStartRecording = false)
        {
            m_bOpen = true;
            return Channel.OpenUi(strName, nId, bStartRecording);
        }

        /// <summary>
        /// Closes the Gym user interface.
        /// </summary>
        /// <param name="nId">Specifies the ID of the Gym.</param>
        public void CloseUi(int nId)
        {
            lock (m_sync)
            {
                Channel.CloseUi(nId);
                m_bOpen = false;
            }
        }

        /// <summary>
        /// Render the observation of the Gym.
        /// </summary>
        /// <param name="nId">Specifies the ID of the Gym.</param>
        /// <param name="obs">Specifies the Observation to render.</param>
        public void Render(int nId, Observation obs)
        {
            try
            {
                lock (m_sync)
                {
                    if (!m_bOpen)
                        return;

                    Channel.Render(nId, obs);
                }
            }
            catch
            {
            }
        }

        /// <summary>
        /// Returns whether or not the Gym user interface is visible or not.
        /// </summary>
        /// <param name="nId">Specifies the ID of the Gym.</param>
        /// <returns>Returns <i>true</i> if the Gym is visible, <i>false</i> otherwise.</returns>
        public bool IsOpen(int nId)
        {
            return Channel.IsOpen(nId);
        }
    }
}
