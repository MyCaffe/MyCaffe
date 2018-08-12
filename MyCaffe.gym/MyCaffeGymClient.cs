using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class MyCaffeGymClient
    {
        IXMyCaffeGymService m_igym;

        public MyCaffeGymClient()
        {
            // Consume the service
            var factory = new ChannelFactory<IXMyCaffeGymService>(new NetNamedPipeBinding(), new EndpointAddress("net.pipe://localhost/MyCaffeGymService"));
            m_igym = factory.CreateChannel();
        }

        public void Open(string strName, bool bAutoStart, bool bShowUi)
        {
            m_igym.Open(strName, bAutoStart, bShowUi);
        }

        public void Close()
        {
            m_igym.Close();
        }

        public DatasetDescriptor GetDataset(string strName, int nType = 0)
        {
            byte[] rgDs = m_igym.GetDataset(strName, nType);
            return DatasetDescriptor.Deserialize(rgDs);
        }

        public string Name
        {
            get {return m_igym.GetName(); }
        }

        public Dictionary<string, int> ActionSpace
        {
            get { return m_igym.GetActionSpace(); }
        }

        public void Run(int nAction)
        {
            m_igym.Run(nAction);
        }

        public void Reset()
        {
            m_igym.Reset();
        }

        public Observation GetObservation()
        {
            return m_igym.GetLastObservation();
        }
    }
}
