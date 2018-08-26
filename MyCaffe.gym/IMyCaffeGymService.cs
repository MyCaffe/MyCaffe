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
        void OpenUi(string strName, int nIdx);
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
        Tuple<double,double,double>[] m_rgState;
        Bitmap m_image;
        double m_dfReward;
        bool m_bDone;

        public Observation(Bitmap img, Tuple<double,double,double>[] rgState, double dfReward, bool bDone)
        {
            m_image = img;
            m_rgState = rgState;
            m_dfReward = dfReward;
            m_bDone = bDone;
        }

        public Observation Clone()
        {
            Bitmap img = new Bitmap(m_image);
            List<Tuple<double, double, double>> rgVal = new List<Tuple<double, double, double>>();

            foreach (Tuple<double,double,double> item in m_rgState)
            {
                rgVal.Add(new Tuple<double, double, double>(item.Item1, item.Item2, item.Item3));
            }

            return new Observation(img, rgVal.ToArray(), m_dfReward, m_bDone);
        }

        public static double[] GetValues(Tuple<double,double,double>[] rg, bool bNormalize)
        {
            List<double> rgState = new List<double>();

            for (int i = 0; i < rg.Length; i++)
            {
                if (bNormalize)
                    rgState.Add((rg[i].Item1 - rg[i].Item2) / (rg[i].Item3 - rg[i].Item2));
                else
                    rgState.Add(rg[i].Item1);
            }

            return rgState.ToArray();
        }

        [DataMember]
        public Tuple<double,double,double>[] State
        {
            get { return m_rgState; }
            set { m_rgState = value; }
        }

        [DataMember]
        public Bitmap Image
        {
            get { return m_image; }
            set { m_image = value; }
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
