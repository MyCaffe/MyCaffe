using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class MyCaffeGymClient : IDisposable, IXMyCaffeGymCallback
    {
        DuplexChannelFactory<IXMyCaffeGymService> m_factory = null;
        IXMyCaffeGymService m_igym;

        public event EventHandler<ObservationArgs> OnNewObservation;

        public MyCaffeGymClient()
        {
            // Consume the service
            m_factory = new DuplexChannelFactory<IXMyCaffeGymService>(new InstanceContext(this), new NetNamedPipeBinding(), new EndpointAddress("net.pipe://localhost/MyCaffeGymService"));
            m_igym = m_factory.CreateChannel();
        }

        public void Dispose()
        {
            if (m_igym != null)
            {
                m_igym.CloseAll(null);
                m_igym = null;
            }

            if (m_factory != null)
            {
                m_factory.Close();
                m_factory = null;
            }
        }

        public int Open(string strName, bool bAutoStart, bool bShowUi, bool bShowOnyFirst, double[] rgdfInit)
        {
            return m_igym.Open(strName, bAutoStart, bShowUi, bShowOnyFirst, rgdfInit);
        }

        public void Close(string strName, int nIdx)
        {
            m_igym.Close(strName, nIdx);
        }

        public void CloseAll(string strName)
        {
            m_igym.CloseAll(strName);
        }

        public void OpenUi(string strName, int nIndex)
        {
            m_igym.OpenUi(strName, nIndex);
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

        public void OnObservation(string strName, int nIdx, Observation obs)
        {
            if (OnNewObservation != null)
                OnNewObservation(this, new ObservationArgs(strName, nIdx, obs));
        }
    }
}
