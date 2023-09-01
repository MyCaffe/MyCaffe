using MyCaffe.basecode;
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
    /// <summary>
    /// The IXMYCaffeGymUiService interface provides access to the MyCaffeGymUiService used
    /// to display the visualizations of each Gym as they run.
    /// </summary>
    [ServiceContract(SessionMode=SessionMode.Required, CallbackContract=typeof(IXMyCaffeGymUiCallback))]
    public interface IXMyCaffeGymUiService
    {
        /// <summary>
        /// Open the user interface of a Gym.
        /// </summary>
        /// <param name="strName">Specifies the Gym name who's user-interface is to be displayed.</param>
        /// <param name="nId">Specifies the Gym id (used when multiple Gym's of the same name are used).</param>
        /// <param name="bStartRecording">Optionally, specifies to open with recording on.</param>
        /// <returns>The ID of the Gym opened is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int OpenUi(string strName, int nId, bool bStartRecording = false);
        /// <summary>
        /// Close the user interface of a Gym.
        /// </summary>
        /// <param name="nId">Specifies the Gym id (used when multiple Gym's of the same name are used).</param>
        [OperationContract(IsOneWay = true)]
        void CloseUi(int nId);
        /// <summary>
        /// Render an observation on the Gym user-interface.
        /// </summary>
        /// <param name="nId">Specifies the Gym ID.</param>
        /// <param name="obs">Specifies the Observation to visualize.</param>
        [OperationContract(IsOneWay = true)]        
        void Render(int nId, Observation obs);
        /// <summary>
        /// Returns <i>true</i> when the visualization is open, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nId">Specifies the Gym ID.</param>
        /// <returns>Returns <i>true</i> when the visualization is open, <i>false</i> otherwise.</returns>
        [OperationContract(IsOneWay = false)]
        bool IsOpen(int nId);
    }

    /// <summary>
    /// The IXMyCaffeGymUiCallback is used to interact with the user of the IXMyCaffeGymUiService interface.
    /// </summary>
    public interface IXMyCaffeGymUiCallback
    {
        /// <summary>
        /// The Closing method is called when closing the Gym user interface.
        /// </summary>
        [OperationContract(IsOneWay=true)]
        void Closing();
    }

    /// <summary>
    /// The Observation contains data describing the Gym as it runs.
    /// </summary>
    [DataContract]
    public class Observation
    {
        double[] m_rgState;
        double m_dfReward;
        bool m_bDone;
        Bitmap m_img;
        Bitmap m_imgDisplay;
        bool m_bRequireDisplayImage = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="imgDisp">Specifies the display image.</param>
        /// <param name="img">Optionally, specifies the action image.</param>
        /// <param name="bRequireDisplayImg">Specifies whether a display image is required.</param>
        /// <param name="rgState">Specifies state data of the Gym.</param>
        /// <param name="dfReward">Specifies the reward.</param>
        /// <param name="bDone">Specifies the done state of the Gym.</param>
        public Observation(Bitmap imgDisp, Bitmap img, bool bRequireDisplayImg, double[] rgState, double dfReward, bool bDone)
        {
            m_rgState = rgState;
            m_dfReward = dfReward;
            m_bDone = bDone;
            m_img = img;
            m_imgDisplay = imgDisp;
            m_bRequireDisplayImage = bRequireDisplayImg;
        }

        /// <summary>
        /// Returns a copy of the Observation.
        /// </summary>
        /// <returns>The Observation copy is returned.</returns>
        public Observation Clone()
        {
            Bitmap bmp = (m_img == null) ? null : new Bitmap(m_img);
            Bitmap bmpDisp = (m_imgDisplay == null) ? null : new Bitmap(m_imgDisplay);
            double[] rgState = new double[m_rgState.Length];
            Array.Copy(m_rgState, rgState, rgState.Length);

            return new Observation(bmpDisp, bmp, m_bRequireDisplayImage, rgState, m_dfReward, m_bDone);
        }

        /// <summary>
        /// Get/set the state data.
        /// </summary>
        [DataMember]
        public double[] State
        {
            get { return m_rgState; }
            set { m_rgState = value; }
        }

        /// <summary>
        /// Get/set the action image, if it exists.
        /// </summary>
        [DataMember]
        public Bitmap Image
        {
            get { return m_img; }
            set { m_img = value; }
        }

        /// <summary>
        /// Get/set the image to display.
        /// </summary>
        [DataMember]
        public Bitmap ImageDisplay
        {
            get { return m_imgDisplay; }
            set { m_imgDisplay = value; }
        }

        /// <summary>
        /// Get/set whether or not the image for display (ImageDisplay) is required.
        /// </summary>
        [DataMember]
        public bool RequireDisplayImage
        {
            get { return m_bRequireDisplayImage; }
            set { m_bRequireDisplayImage = value; }
        }

        /// <summary>
        /// Get/set the reward.
        /// </summary>
        [DataMember]
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }

        /// <summary>
        /// Get/set the done state.
        /// </summary>
        [DataMember]
        public bool Done
        {
            get { return m_bDone; }
            set { m_bDone = value; }
        }
    }
}
