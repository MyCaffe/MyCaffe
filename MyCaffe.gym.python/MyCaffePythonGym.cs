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
        List<SimpleDatum> m_rgData = new List<SimpleDatum>();

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
        /// Returns the data int a 3D form compatible with CV2.
        /// </summary>
        /// <param name="bGrayscale">Optionally, specifies to return gray scale data (one channel).</param>
        /// <param name="dfScale">Optionally, specifies the scale to apply to each item.</param>
        /// <param name="rgData1">Optionally, specifies the data as a flat array.</param>
        /// <returns>The data is returned as a multi-dimensional array.</returns>
        public List<List<List<double>>> GetDataAs3D(bool bGrayscale = false, double dfScale = 1, SimpleDatum sd = null)
        {
            double[] rgData1 = (sd == null) ? Data.ToArray() : sd.GetData<double>();
            int nChannels = (sd == null) ? m_state.Item1.Channels : sd.Channels;
            int nHeight = (sd == null) ? m_state.Item1.Height : sd.Height;
            int nWidth = (sd == null) ? m_state.Item1.Width : sd.Width;
            List<List<List<double>>> rgrgrgData = new List<List<List<double>>>();

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

                        rgData.Add((dfSum / nChannels) * dfScale);
                    }
                    else
                    {
                        for (int c = 0; c < nChannels; c++)
                        {
                            int nIdx = (c * nHeight * nWidth) + (h * nWidth) + w;
                            double dfVal = rgData1[nIdx];

                            rgData.Add(dfVal * dfScale);
                        }
                    }

                    rgrgData.Add(rgData);
                }

                rgrgrgData.Add(rgrgData);
            }

            return rgrgrgData;
        }

        private SimpleDatum preprocess(SimpleDatum sd, bool bGrayscale, double dfScale)
        {
            if (!bGrayscale && dfScale == 1)
                return sd;

            double[] rgData = sd.GetData<double>();
            int nCount = sd.Height * sd.Width;
            int nChannels = sd.Channels;
            bool bIsReal = sd.IsRealData;
            byte[] rgByteData = null;
            double[] rgRealData = null;

            if (bIsReal && !bGrayscale)
            {
                nCount *= sd.Channels;
                rgRealData = new double[nCount];
            }
            else
            {
                bIsReal = false;
                nChannels = 1;
                rgByteData = new byte[nCount];
            }

            for (int h = 0; h < sd.Height; h++)
            {
                for (int w = 0; w < sd.Width; w++)
                {
                    int nIdx = (h * sd.Width) + w;
                    int nIdxSrc = nIdx * sd.Channels;

                    if (rgRealData != null)
                    {
                        for (int c = 0; c < sd.Channels; c++)
                        {
                            double dfVal = rgData[nIdxSrc + c] * dfScale;
                            rgRealData[nIdxSrc + c] = dfVal;
                        }
                    }
                    else
                    {
                        double dfSum = 0;

                        for (int c = 0; c < sd.Channels; c++)
                        {
                            dfSum += rgData[nIdxSrc + c];
                        }

                        double dfVal = ((dfSum / sd.Channels) * dfScale);
                        if (dfVal > 255)
                            dfVal = 255;

                        if (dfVal < 0)
                            dfVal = 0;

                        rgByteData[nIdx] = (byte)dfVal;
                    }
                }
            }

            SimpleDatum sdResult = new SimpleDatum(bIsReal, nChannels, sd.Width, sd.Height, sd.Label, sd.TimeStamp, rgByteData, rgRealData, sd.Boost, sd.AutoLabeled, sd.Index);
            sdResult.Tag = sd.Tag;

            return sdResult;
        }

        /// <summary>
        /// Returns stacked data in a 3D form compatible with CV2.
        /// </summary>
        /// <param name="bReset">Specifies to reset the stack or not.</param>
        /// <param name="nFrames">Optionally, specifies the number of frames (default = 4).</param>
        /// <param name="nStacks">Optionally, specifies the number of stacks (default = 4).</param>
        /// <param name="bGrayscale">Optionally, specifies to return gray scale data (default = true, one channel).</param>
        /// <param name="dfScale">Optionally, specifies the scale to apply to each item (default = 1.0).</param>
        /// <returns>The data is returned as a multi-dimensional array.</returns>
        public List<List<List<double>>> GetDataAsStacked3D(bool bReset, int nFrames = 4, int nStacks = 4, bool bGrayscale = true, double dfScale = 1)
        {
            SimpleDatum sd = preprocess(m_state.Item1, bGrayscale, dfScale);

            if (bReset)
            {
                m_rgData.Clear();

                for (int i = 0; i < nFrames * nStacks; i++)
                {
                    m_rgData.Add(sd);
                }
            }
            else
            {
                m_rgData.Add(sd);
                m_rgData.RemoveAt(0);
            }

            SimpleDatum[] rgSd = new SimpleDatum[nStacks];

            for (int i = 0; i < nStacks; i++)
            {
                int nIdx = ((nStacks - i) * nFrames) - 1;
                rgSd[i] = m_rgData[nIdx];
            }

            SimpleDatum sd1 = new SimpleDatum(rgSd.ToList());

            return GetDataAs3D(false, 1, sd1);
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
