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
    [ServiceContract(SessionMode=SessionMode.Required, CallbackContract=typeof(IXMyCaffeGymUiCallback))]
    public interface IXMyCaffeGymUiService
    {
        [OperationContract(IsOneWay = false)]
        int OpenUi(string strName, int nId);
        [OperationContract(IsOneWay = true)]
        void CloseUi(int nId);
        [OperationContract(IsOneWay = true)]
        void Render(int nId, Observation obs);
        [OperationContract(IsOneWay = false)]
        bool IsOpen(int nId);
    }

    public interface IXMyCaffeGymUiCallback
    {
        [OperationContract(IsOneWay=true)]
        void Closing();
    }

    [DataContract]
    public class Observation
    {
        Tuple<double, double, double, bool>[] m_rgState;
        double m_dfReward;
        bool m_bDone;
        Bitmap m_image;
        Bitmap m_imgDisplay;
        bool m_bRequireDisplayImage = false;

        public Observation(Bitmap imgDisp, Bitmap img, bool bRequireDisplayImg, Tuple<double,double,double, bool>[] rgState, double dfReward, bool bDone)
        {
            m_rgState = rgState;
            m_dfReward = dfReward;
            m_bDone = bDone;
            m_image = img;
            m_imgDisplay = imgDisp;
            m_bRequireDisplayImage = bRequireDisplayImg;
        }

        public Observation Clone()
        {
            Bitmap bmp = (m_image == null) ? null : new Bitmap(m_image);
            Bitmap bmpDisp = (m_imgDisplay == null) ? null : new Bitmap(m_imgDisplay);

            List<Tuple<double, double, double, bool>> rgState = new List<Tuple<double, double, double, bool>>();
            foreach (Tuple<double, double, double, bool> item in m_rgState)
            {
                rgState.Add(new Tuple<double, double, double, bool>(item.Item1, item.Item2, item.Item3, item.Item4));
            }

            return new Observation(bmpDisp, bmp, m_bRequireDisplayImage, rgState.ToArray(), m_dfReward, m_bDone);
        }

        public static double[] GetValues(Tuple<double,double,double, bool>[] rg, bool bNormalize, bool bGetAllData = false)
        {
            List<double> rgState = new List<double>();

            for (int i = 0; i < rg.Length; i++)
            {
                if (rg[i].Item4 || bGetAllData)
                {
                    if (bNormalize)
                        rgState.Add((rg[i].Item1 - rg[i].Item2) / (rg[i].Item3 - rg[i].Item2));
                    else
                        rgState.Add(rg[i].Item1);
                }
            }

            return rgState.ToArray();
        }

        [DataMember]
        public Tuple<double,double,double, bool>[] State
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
        public Bitmap ImageDisplay
        {
            get { return m_imgDisplay; }
            set { m_imgDisplay = value; }
        }

        [DataMember]
        public bool RequireDisplayImage
        {
            get { return m_bRequireDisplayImage; }
            set { m_bRequireDisplayImage = value; }
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
