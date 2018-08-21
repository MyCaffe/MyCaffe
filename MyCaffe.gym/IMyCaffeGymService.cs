using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;

namespace MyCaffe.gym
{
    [ServiceContract]
    public interface IXMyCaffeGymService
    {
        [OperationContract]
        int Open(string strName, bool bAutoStart, bool bShowUi, bool bShowOnlyFirst);
        [OperationContract]
        void Close(string strName, int nIdx);
        [OperationContract]
        void CloseAll(string strName);
        [OperationContract]
        byte[] GetDataset(string strName, int nType);
        [OperationContract]
        Dictionary<string, int> GetActionSpace(string strName);
        [OperationContract]
        void Run(string strName, int nIdx, int nAction);
        [OperationContract]
        void Reset(string strName, int nIdx);
        [OperationContract]
        Observation GetLastObservation(string strName, int nIdx);
    }

    [DataContract]
    public class Observation
    {
        Bitmap m_image;
        double[] m_rgState;
        double m_dfReward;
        bool m_bDone;

        public Observation(Bitmap img, double[] rgState, double dfReward, bool bDone)
        {
            m_image = img;
            m_rgState = rgState;
            m_dfReward = dfReward;
            m_bDone = bDone;
        }

        [DataMember]
        public Bitmap Image
        {
            get { return m_image; }
            set { m_image = value; }
        }

        [DataMember]
        public double[] State
        {
            get { return m_rgState; }
            set { m_rgState = value; }
        }

        [DataMember]
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }

        [DataMember]
        public bool Done
        {
            get { return m_bDone; }
            set { m_bDone = value; }
        }
    }
}
