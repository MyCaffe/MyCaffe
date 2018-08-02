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
        string m_strGym = null;

        public void Open(string strName, bool bAutoStart, bool bShowUi)
        {
            m_strGym = strName;
            MyCaffeGymRegistrar.Registry.Open(strName, bAutoStart, bShowUi);
        }

        public void Close()
        {
            MyCaffeGymRegistrar.Registry.Close(m_strGym);
            m_strGym = null;
        }

        public string GetName()
        {
            return m_strGym;
        }

        public Dictionary<string, int> GetActionSpace()
        {
            return MyCaffeGymRegistrar.Registry.GetActionSpace(m_strGym);
        }

        public Observation GetLastObservation()
        {
            return MyCaffeGymRegistrar.Registry.GetObservation(m_strGym);
        }

        public void Run(int nAction = 0)
        {
            MyCaffeGymRegistrar.Registry.Run(m_strGym, nAction);
        }
    }
}
