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
        /// <param name="nMaxBuffer">Optionally, specifies the maximum buffer to use for out-going messages (in-going must be set on the client-side binding), default = 2147483647.
        /// </param>
        public MyCaffeGymUiServiceHost(int nMaxBuffer = 2147483647)
            : base(typeof(MyCaffeGymUiService), new Uri[] { new Uri("net.pipe://localhost/MyCaffeGym") })
        {
            NetNamedPipeBinding binding = new NetNamedPipeBinding();
            binding.MaxReceivedMessageSize = nMaxBuffer;
            binding.MaxBufferSize = nMaxBuffer;
            binding.MaxBufferPoolSize = nMaxBuffer;

            AddServiceEndpoint(typeof(IXMyCaffeGymUiService), binding, "gymui");
        }
    }
}
