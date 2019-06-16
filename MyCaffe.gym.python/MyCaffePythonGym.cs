using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.ServiceModel;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using MyCaffe.basecode;

namespace MyCaffe.gym.python
{
    /// <summary>
    /// The MyCaffePythonGym class provides a simple interface that can easily be used from within Python.
    /// </summary>
    /// <remarks>
    /// To use within Python first install the PythonNet package with the command:
    /// <code>
    ///   pip install pythonnet
    /// </code>
    /// 
    /// Next, from within python use the following code to access the MyCaffePythonGym.
    /// <code>
    /// import clr
    /// clr.AddReference('path\\MyCaffe.gym.python.dll')
    /// from MyCaffe.gym.python import *
    /// 
    /// gym = MyCaffePythonGym()
    /// gym.Initialize('ATARI', 'FrameSkip=4;GameROM=C:\\Program~Files\\SignalPop\\AI~Designer\\roms\\pong.bin')
    /// gym.OpenUi()
    /// gym.Step(action, 1)
    /// gym.CloseUi();
    /// </code>
    /// 
    /// NOTE: Using the OpenUi function requires that a UI service host is already running.  The MyCaffe Test Application
    /// automatically provides a service host - just run this application before running your Python script and
    /// the test application will then handle the user interface display.
    /// 
    /// To get the latest MyCaffe Test Application, see https://github.com/MyCaffe/MyCaffe/releases
    /// </remarks>
    public class MyCaffePythonGym : IXMyCaffeGymUiCallback
    {
        int m_nUiId = -1;
        IXMyCaffeGym m_igym = null;
        MyCaffeGymUiProxy m_gymui = null;
        Log m_log = new Log("MyCaffePythonGym");
        Tuple<SimpleDatum, double, bool> m_state = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffePythonGym()
        {
        }

        /// <summary>
        /// The Initialize method loads the gym specified.
        /// </summary>
        /// <param name="strGym">Specifies the name of the gym to load.</param>
        /// <param name="strParam">Specifies the semi-colon separated parameters passed to the gym.</param>
        /// <returns>0 is returned on success.</returns>
        /// <remarks>
        /// The following gyms are supported: 'ATARI', 'Cart-Pole'
        /// </remarks>
        public int Initialize(string strGym, string strParam)
        {
            m_log.EnableTrace = true;

            GymCollection col = new GymCollection();
            col.Load();

            m_igym = col.Find(strGym);
            if (m_igym == null)
                throw new Exception("Could not find the gym '" + strGym + "'!");

            m_igym.Initialize(m_log, new PropertySet(strParam));

            return 0;
        }

        /// <summary>
        /// Returns the name of the gym.
        /// </summary>
        public string Name
        {
            get
            {
                if (m_igym == null)
                    throw new Exception("You must call 'Initialize' first!");

                return "MyCaffe " + m_igym.Name;
            }
        }

        /// <summary>
        /// Returns the action values
        /// </summary>
        public int[] Actions
        {
            get
            {
                if (m_igym == null)
                    throw new Exception("You must call 'Initialize' first!");

                List<int> rg = new List<int>();
                Dictionary<string, int> rgActions = m_igym.GetActionSpace();
                int nActionIdx = 0;

                foreach (KeyValuePair<string, int> kv in rgActions)
                {
                    rg.Add(nActionIdx);
                    nActionIdx++;
                }

                return rg.ToArray();
            }
        }

        /// <summary>
        /// Returns the action names
        /// </summary>
        public string[] ActionNames
        {
            get
            {
                if (m_igym == null)
                    throw new Exception("You must call 'Initialize' first!");

                List<string> rg = new List<string>();
                Dictionary<string, int> rgActions = m_igym.GetActionSpace();

                foreach (KeyValuePair<string, int> kv in rgActions)
                {
                    rg.Add(kv.Key);
                }

                return rg.ToArray();
            }
        }

        /// <summary>
        /// Returns the terminal state from the last state.
        /// </summary>
        public bool IsTerminal
        {
            get { return (m_state == null) ? true : m_state.Item3; }
        }

        /// <summary>
        /// Returns the reward from the last state.
        /// </summary>
        public double Reward
        {
            get { return (m_state == null) ? 0 : m_state.Item2; }
        }

        /// <summary>
        /// Returns the data from the last state.
        /// </summary>
        public List<double> Data
        {
            get { return (m_state == null) ? null : m_state.Item1.GetData<double>().ToList(); }
        }

        /// <summary>
        /// Returns the data as an image compatible with CV2.
        /// </summary>
        /// <param name="bGrayscale">Optionally, specifies to return gray scale data (one channel).</param>
        /// <param name="dfScale">Optionally, specifies the scale to apply to each item.</param>
        /// <param name="nClipUpToRow">Optionally, specifies the number of rows starting from the top to clip and set as 0.</param>
        public List<List<List<double>>> GetDataAsImage(bool bGrayscale = false, double dfScale = 1, int nClipUpToRow = 0)
        {
            List<List<List<double>>> rgrgrgData = new List<List<List<double>>>();
            List<double> rgData1 = Data;
            int nChannels = m_state.Item1.Channels;
            int nHeight = m_state.Item1.Height;
            int nWidth = m_state.Item1.Width;

            for (int h = 0; h < nHeight; h++)
            {
                List<List<double>> rgrgData = new List<List<double>>();

                for (int w = 0; w < nWidth; w++)
                {
                    List<double> rgData = new List<double>();

                    if (bGrayscale)
                    {
                        double dfSum = 0;

                        for (int c = 0; c < nChannels; c++)
                        {
                            int nIdx = (c * nHeight * nWidth) + (h * nWidth) + w;
                            dfSum += rgData1[nIdx];
                        }

                        if (h < nClipUpToRow)
                            dfSum = 0;

                        rgData.Add((dfSum / nChannels) * dfScale);
                    }
                    else
                    {
                        for (int c = 0; c < nChannels; c++)
                        {
                            int nIdx = (c * nHeight * nWidth) + (h * nWidth) + w;
                            double dfVal = rgData1[nIdx];

                            if (h < nClipUpToRow)
                                dfVal = 0;

                            rgData.Add(dfVal * dfScale);
                        }
                    }

                    rgrgData.Add(rgData);
                }

                rgrgrgData.Add(rgrgData);
            }

            return rgrgrgData;
        }

        /// <summary>
        /// Resets the gym to its initial state.
        /// </summary>
        /// <returns>A tuple containing a double[] with the data, a double with the reward and a bool with the terminal state is returned.</returns>
        public Tuple<List<double>, double, bool> Reset()
        {
            if (m_igym == null)
                throw new Exception("You must call 'Initialize' first!");

            Tuple<State, double, bool> state = m_igym.Reset();

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            Tuple<Bitmap, SimpleDatum> data = m_igym.Render(bIsOpen, 512, 512, true);
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(false, out nDataLen);
            Observation obs = new Observation(data.Item1, ImageData.GetImage(data.Item2), m_igym.RequiresDisplayImage, stateData.RealData, state.Item2, state.Item3);

            if (bIsOpen)
            {
                m_gymui.Render(m_nUiId, obs);
                Thread.Sleep(m_igym.UiDelay);
            }

            m_state = new Tuple<SimpleDatum, double, bool>(data.Item2, state.Item2, state.Item3);

            return new Tuple<List<double>, double, bool>(m_state.Item1.GetData<double>().ToList(), m_state.Item2, m_state.Item3);
        }

        /// <summary>
        /// Steps the gym one or more steps with a given action.
        /// </summary>
        /// <param name="nAction">Specifies the action to run.</param>
        /// <param name="nSteps">Specifies the number of steps to run the action.</param>
        /// <returns>A tuple containing a double[] with the data, a double with the reward and a bool with the terminal state is returned.</returns>
        public Tuple<List<double>, double, bool> Step(int nAction, int nSteps)
        {
            if (m_igym == null)
                throw new Exception("You must call 'Initialize' first!");

            for (int i = 0; i < nSteps - 1; i++)
            {
                m_igym.Step(nAction);
            }

            Tuple<State, double, bool> state = m_igym.Step(nAction);

            bool bIsOpen = (m_nUiId >= 0) ? true : false;
            Tuple<Bitmap, SimpleDatum> data = m_igym.Render(bIsOpen, 512, 512, true);
            int nDataLen = 0;
            SimpleDatum stateData = state.Item1.GetData(false, out nDataLen);
            Observation obs = new Observation(data.Item1, ImageData.GetImage(data.Item2), m_igym.RequiresDisplayImage, stateData.RealData, state.Item2, state.Item3);

            if (bIsOpen)
            {
                m_gymui.Render(m_nUiId, obs);
                Thread.Sleep(m_igym.UiDelay);
            }

            m_state = new Tuple<SimpleDatum, double, bool>(data.Item2, state.Item2, state.Item3);

            return new Tuple<List<double>, double, bool>(m_state.Item1.GetData<double>().ToList(), m_state.Item2, m_state.Item3);
        }

        /// <summary>
        /// The OpenUi method opens the user interface to visualize the gym as it progresses.
        /// </summary>
        public void OpenUi()
        {
            if (m_gymui != null)
                return;

            try
            {
                m_gymui = new MyCaffeGymUiProxy(new InstanceContext(this));
                m_gymui.Open();
                m_nUiId = m_gymui.OpenUi(Name, m_nUiId);
            }
            catch (Exception excpt)
            {
                throw new Exception("You need to run the MyCaffe Test Application which supports the gym user interface host.", excpt);
            }
        }

        /// <summary>
        /// The CloseUi method closes the user interface if it is open.
        /// </summary>
        public void CloseUi()
        {
            if (m_gymui == null)
                return;

            m_gymui.CloseUi(0);
            m_gymui.Close();
            m_gymui = null;
            m_nUiId = -1;
        }

        /// <summary>
        /// The Closing method is a call-back method called when the gym closes.
        /// </summary>
        public void Closing()
        {
        }
    }
}
