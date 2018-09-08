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
    [ServiceContract(SessionMode=SessionMode.Required)]
    public interface IXMyCaffeGymService
    {
        [OperationContract(IsOneWay = true)]
        void Open(string strName);
        [OperationContract(IsOneWay = true)]
        void Close();
        [OperationContract(IsOneWay = true)]
        void Render(Observation obs);
    }

    [DataContract]
    public class Observation
    {
        Tuple<double, double, double>[] m_rgState;
        double m_dfReward;
        bool m_bDone;
        Bitmap m_image;

        public Observation(Bitmap img, Tuple<double,double,double>[] rgState, double dfReward, bool bDone)
        {
            m_rgState = rgState;
            m_dfReward = dfReward;
            m_bDone = bDone;
            m_image = img;
        }

        public Observation Clone()
        {
            Bitmap bmp = (m_image == null) ? null : new Bitmap(m_image);

            List<Tuple<double, double, double>> rgState = new List<Tuple<double, double, double>>();
            foreach (Tuple<double, double, double> item in m_rgState)
            {
                rgState.Add(new Tuple<double, double, double>(item.Item1, item.Item2, item.Item3));
            }

            return new Observation(bmp, rgState.ToArray(), m_dfReward, m_bDone);
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
