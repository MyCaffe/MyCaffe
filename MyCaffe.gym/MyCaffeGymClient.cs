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

        public int Open(string strName, bool bAutoStart, bool bShowUi, bool bShowOnyFirst)
        {
            return m_igym.Open(strName, bAutoStart, bShowUi, bShowOnyFirst);
        }

        public void Close(string strName, int nIdx)
        {
            m_igym.Close(strName, nIdx);
        }

        public void CloseAll(string strName)
        {
            m_igym.CloseAll(strName);
        }

        public DatasetDescriptor GetDataset(string strName, int nType = 0)
        {
            byte[] rgDs = m_igym.GetDataset(strName, nType);
            return DatasetDescriptor.Deserialize(rgDs);
        }

        public Dictionary<string, int> GetActionSpace(string strName)
        {
            return m_igym.GetActionSpace(strName);
        }

        public void Run(string strName, int nIdx, int nAction)
        {
            m_igym.Run(strName, nIdx, nAction);
        }

        public void Reset(string strName, int nIdx)
        {
            m_igym.Reset(strName, nIdx);
        }

        public Observation GetObservation(string strName, int nIdx)
        {
            return m_igym.GetLastObservation(strName, nIdx);
        }
    }
}
