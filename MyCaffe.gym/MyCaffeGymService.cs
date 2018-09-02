using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;

namespace MyCaffe.gym
{
    [ServiceBehavior(InstanceContextMode = InstanceContextMode.PerSession)]
    public class MyCaffeGymService : IXMyCaffeGymService, IDisposable
    {
        IXMyCaffeGymCallback m_icallback = null;

        public MyCaffeGymService()
        {
            MyCaffeGymRegistrar.Registry.OnObservation += Registry_OnObservation;
            m_icallback = OperationContext.Current.GetCallbackChannel<IXMyCaffeGymCallback>();
        }

        public void Dispose()
        {
            MyCaffeGymRegistrar.Registry.OnObservation -= Registry_OnObservation;
        }

        private void Registry_OnObservation(object sender, ObservationArgs e)
        {
            if (m_icallback != null)
                m_icallback.OnObservation(e.Name, e.Index, e.Observation);
        }

        public int Open(string strName, bool bAutoStart, bool bShowUi, bool bShowOnlyFirst, double[] rgdfInit)
        {
            return MyCaffeGymRegistrar.Registry.Open(strName, bAutoStart, bShowUi, bShowOnlyFirst, rgdfInit);
        }

        public void Close(string strName, int nIdx)
        {
            MyCaffeGymRegistrar.Registry.Close(strName, nIdx);
        }

        public void CloseAll(string strName)
        {
            MyCaffeGymRegistrar.Registry.CloseAll(strName);
        }

        public void OpenUi(string strName, int nIdx)
        {
            MyCaffeGymRegistrar.Registry.Open(strName, nIdx);
        }

        public Dictionary<string, int> GetActionSpace(string strName)
        {
            return MyCaffeGymRegistrar.Registry.GetActionSpace(strName);
        }

        public Observation GetLastObservation(string strName, int nIdx, bool bReset)
        {
            return MyCaffeGymRegistrar.Registry.GetObservation(strName, nIdx, bReset);
        }

        public void Run(string strName, int nIdx, int nAction)
        {
            MyCaffeGymRegistrar.Registry.Run(strName, nIdx, nAction);
        }

        public void Reset(string strName, int nIdx)
        {
            MyCaffeGymRegistrar.Registry.Reset(strName, nIdx);
        }

        public byte[] GetDataset(string strName, int nType)
        {
            DatasetDescriptor ds = MyCaffeGymRegistrar.Registry.GetDataset(strName, nType);
            return DatasetDescriptor.Serialize(ds);
        }
    }
}
