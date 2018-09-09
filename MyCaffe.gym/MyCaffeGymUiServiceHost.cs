using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class MyCaffeGymUiServiceHost : ServiceHost
    {
        public MyCaffeGymUiServiceHost()
            : base(typeof(MyCaffeGymUiService), new Uri[] { new Uri("net.pipe://localhost/MyCaffeGym") })
        {
            AddServiceEndpoint(typeof(IXMyCaffeGymUiService), new NetNamedPipeBinding(), "gymui");
        }
    }
}
