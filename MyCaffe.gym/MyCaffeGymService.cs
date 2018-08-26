using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;

namespace MyCaffe.gym
{
    public class MyCaffeGymService : IXMyCaffeGymService
    {
        public int Open(string strName, bool bAutoStart, bool bShowUi, bool bShowOnlyFirst)
        {
            return MyCaffeGymRegistrar.Registry.Open(strName, bAutoStart, bShowUi, bShowOnlyFirst);
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

        public Observation GetLastObservation(string strName, int nIdx)
        {
            return MyCaffeGymRegistrar.Registry.GetObservation(strName, nIdx);
        }

        public void Run(string strName, int nIdx, int nAction = 0)
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
