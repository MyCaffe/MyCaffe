using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The MyCaffeGymUiServiceHost provides the hosting service that listens for users of the user interface service.
    /// </summary>
    public class MyCaffeGymUiServiceHost : ServiceHost
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeGymUiServiceHost()
            : base(typeof(MyCaffeGymUiService), new Uri[] { new Uri("net.pipe://localhost/MyCaffeGym") })
        {
            AddServiceEndpoint(typeof(IXMyCaffeGymUiService), new NetNamedPipeBinding(), "gymui");
        }
    }
}
