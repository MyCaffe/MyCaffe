using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.ServiceModel.Description;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class MyCaffeGymUiProxy : DuplexClientBase<IXMyCaffeGymUiService>       
    {
        public MyCaffeGymUiProxy(InstanceContext ctx)
            : base(ctx, new ServiceEndpoint(ContractDescription.GetContract(typeof(IXMyCaffeGymUiService)),
                   new NetNamedPipeBinding(), new EndpointAddress("net.pipe://localhost/MyCaffeGym/gymui")))
        {
        }

        public int OpenUi(string strName, int nId)
        {
            return Channel.OpenUi(strName, nId);
        }

        public void CloseUi(int nId)
        {
            Channel.CloseUi(nId);
        }

        public void Render(int nId, Observation obs)
        {
            Channel.Render(nId, obs);
        }
    }
}
