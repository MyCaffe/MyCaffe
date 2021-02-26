using AleControlLib;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The Atari Gym provides acess to the Atari-2600 Emulator from Stella (https://github.com/stella-emu/stella)
    /// via a slightly modified version of the Arcade-Learning-Envrionment (ALE) from mgbellemare 
    /// (https://github.com/mgbellemare/Arcade-Learning-Environment).
    /// </summary>
    /// <remarks>
    /// This gym is a rewrite of the original atari gym provided by OpenAi under the MIT license and located
    /// on GitHub at: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
    /// License: https://github.com/openai/gym/blob/master/LICENSE.md
    /// 
    /// The original Atari-2600 Emulator from Stella (https://github.com/stella-emu/stella) is 
    /// distributed under the GPL license (https://github.com/stella-emu/stella/blob/master/License.txt)
    /// 
    /// The original Arcade-Learning-Envrionment (ALE) from mgbellemare 
    /// (https://github.com/mgbellemare/Arcade-Learning-Environment) also distributed under the GPL license
    /// (https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/License.txt).
    /// 
    /// The Arcade-Learning-Environment (ALE) uses the Simple DrectMedia Layer (SDL) which is a cross-platform
    /// library designed to make it easy to write multi-media software, such as games and emulators.
    /// The SDL source code is available from: http://www.libsdl.org/ and the library is distrubted under 
    /// the terms of the GNU LGPL license: http://www.gnu.org/copyleft/lesser.html
    /// </remarks>
    public class AtariGym : IXMyCaffeGym, IDisposable
    {
        string m_strName = "ATARI";
        IALE m_ale = null;
        AtariState m_state = new AtariState();
        Log m_log;
        ACTION[] m_rgActionsRaw;
        List<int> m_rgFrameSkip = new List<int>();
        CryptoRandom m_random;
        Dictionary<string, int> m_rgActions = new Dictionary<string, int>();
        List<KeyValuePair<string, int>> m_rgActionSet;
        DATA_TYPE m_dt = DATA_TYPE.BLOB;
        COLORTYPE m_ct = COLORTYPE.CT_COLOR;
        bool m_bPreprocess = true;
        bool m_bForceGray = false;
        DirectBitmap m_bmpRaw = null;
        DirectBitmap m_bmpActionRaw = null;
        bool m_bEnableNumSkip = true;
        int m_nFrameSkip = -1;

        /// <summary>
        /// The constructor.
        /// </summary>
        public AtariGym()
        {
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_ale != null)
            {
                m_ale.Shutdown();
                m_ale = null;               
            }

            if (m_bmpRaw != null)
            {
                m_bmpRaw.Dispose();
                m_bmpRaw = null;
            }

            if (m_bmpActionRaw != null)
            {
                m_bmpActionRaw.Dispose();
                m_bmpActionRaw = null;
            }
        }

        /// <summary>
        /// Initialize the gym with the specified properties.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="properties">Specifies the properties containing Gym specific initialization parameters.</param>
        /// <remarks>
        /// The AtariGym uses the following initialization properties.
        ///   GameRom='path to .rom file'
        /// </remarks>
        public void Initialize(Log log, PropertySet properties)
        {
            m_log = log;

            if (m_ale != null)
            {
                m_ale.Shutdown();
                m_ale = null;
            }

            m_ale = new ALE();
            m_ale.Initialize();
            m_ale.EnableDisplayScreen = false;
            m_ale.EnableSound = false;
            m_ale.EnableColorData = properties.GetPropertyAsBool("EnableColor", false);
            m_ale.EnableRestrictedActionSet = true;
            m_ale.EnableColorAveraging = true;
            m_ale.AllowNegativeRewards = properties.GetPropertyAsBool("AllowNegativeRewards", false);
            m_ale.EnableTerminateOnRallyEnd = properties.GetPropertyAsBool("TerminateOnRallyEnd", false);
            m_ale.RandomSeed = (int)DateTime.Now.Ticks;
            m_ale.RepeatActionProbability = 0.0f; // disable action repeatability

            if (properties == null)
                throw new Exception("The properties must be specified with the 'GameROM' set the the Game ROM file path.");

            string strROM = properties.GetProperty("GameROM");
            if (strROM.Contains('~'))
                strROM = Utility.Replace(strROM, '~', ' ');
            else
                strROM = Utility.Replace(strROM, "[sp]", ' ');

            if (!File.Exists(strROM))
                throw new Exception("Could not find the game ROM file specified '" + strROM + "'!");

            if (properties.GetPropertyAsBool("UseGrayscale", false))
                m_ct = COLORTYPE.CT_GRAYSCALE;

            m_bPreprocess = properties.GetPropertyAsBool("Preprocess", true);
            m_bForceGray = properties.GetPropertyAsBool("ActionForceGray", false);
            m_bEnableNumSkip = properties.GetPropertyAsBool("EnableNumSkip", true);
            m_nFrameSkip = properties.GetPropertyAsInt("FrameSkip", -1);

            m_ale.Load(strROM);
            m_rgActionsRaw = m_ale.ActionSpace;
            m_random = new CryptoRandom();
            m_rgFrameSkip = new List<int>();

            if (m_nFrameSkip < 0)
            {
                for (int i = 2; i < 5; i++)
                {
                    m_rgFrameSkip.Add(i);
                }
            }
            else
            {
                m_rgFrameSkip.Add(m_nFrameSkip);
            }

            m_rgActions.Add(ACTION.ACT_PLAYER_A_LEFT.ToString(), (int)ACTION.ACT_PLAYER_A_LEFT);
            m_rgActions.Add(ACTION.ACT_PLAYER_A_RIGHT.ToString(), (int)ACTION.ACT_PLAYER_A_RIGHT);

            if (!properties.GetPropertyAsBool("EnableBinaryActions", false))
                m_rgActions.Add(ACTION.ACT_PLAYER_A_FIRE.ToString(), (int)ACTION.ACT_PLAYER_A_FIRE);

            m_rgActionSet = m_rgActions.ToList();

            Reset(false);
        }

        /// <summary>
        /// Create a new copy of the gym.
        /// </summary>
        /// <param name="properties">Optionally, specifies the properties to initialize the new copy with.</param>
        /// <returns>The new Gym copy is returned.</returns>
        public IXMyCaffeGym Clone(PropertySet properties = null)
        {
            AtariGym gym = new AtariGym();

            if (properties != null)
                gym.Initialize(m_log, properties);

            return gym;
        }

        /// <summary>
        /// Returns <i>true</i> indicating that this Gym requires a display image.
        /// </summary>
        public bool RequiresDisplayImage
        {
            get { return true; }
        }

        /// <summary>
        /// Returns the selected data type.
        /// </summary>
        public DATA_TYPE SelectedDataType
        {
            get { return m_dt; }
        }

        /// <summary>
        /// Returns the data types supported by this gym.
        /// </summary>
        public DATA_TYPE[] SupportedDataType
        {
            get { return new DATA_TYPE[] { DATA_TYPE.BLOB }; }
        }

        /// <summary>
        /// Returns the gym's name.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Returns the delay to use (if any) when the user-display is visible.
        /// </summary>
        public int UiDelay
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the testinng percent of -1, which then uses the default of 0.2.
        /// </summary>
        public double TestingPercent
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the action space as a dictionary of name,actionid pairs.
        /// </summary>
        /// <returns>The action space is returned.</returns>
        public Dictionary<string, int> GetActionSpace()
        {
            return m_rgActions;
        }

        /// <summary>
        /// Shutdown and close the gym.
        /// </summary>
        public void Close()
        {
            if (m_ale != null)
            {
                m_ale.Shutdown();
                m_ale = null;
            }
        }

        /// <summary>
        /// Render the gym's current state on a bitmap and SimpleDatum.
        /// </summary>
        /// <param name="bShowUi">When <i>true</i> the Bitmap is drawn.</param>
        /// <param name="nWidth">Specifies the width used to size the Bitmap.</param>
        /// <param name="nHeight">Specifies the height used to size the Bitmap.</param>
        /// <param name="bGetAction">When <i>true</i> the action data is returned as a SimpleDatum.</param>
        /// <returns>A tuple optionally containing a Bitmap and/or Simpledatum is returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, bool bGetAction)
        {
            List<double> rgData = new List<double>();
            return Render(bShowUi, nWidth, nHeight, rgData.ToArray(), bGetAction);
        }

        /// <summary>
        /// Render the gyms specified data.
        /// </summary>
        /// <param name="bShowUi">When <i>true</i> the Bitmap is drawn.</param>
        /// <param name="nWidth">Specifies the width used to size the Bitmap.</param>
        /// <param name="nHeight">Specifies the height used to size the Bitmap.</param>
        /// <param name="rgData">Specifies the gym data to render.</param>
        /// <param name="bGetAction">When <i>true</i> the action data is returned as a SimpleDatum.</param>
        /// <returns>A tuple optionally containing a Bitmap and/or Simpledatum is returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, double[] rgData, bool bGetAction)
        {
            float fWid;
            float fHt;

            m_ale.GetScreenDimensions(out fWid, out fHt);
            byte[] rgRawData = m_ale.GetScreenData(m_ct);

            Tuple<DirectBitmap, SimpleDatum> data = getData(m_ct, (int)fWid, (int)fHt, 35, 2, rgRawData, m_bPreprocess, bGetAction, m_bForceGray);
            Bitmap bmp = null;

            if (bShowUi)
            {
                if (data.Item1.Bitmap.Width != nWidth || data.Item1.Bitmap.Height != nHeight)
                    bmp = ImageTools.ResizeImage(data.Item1.Bitmap, nWidth, nHeight);
                else
                    bmp = new Bitmap(data.Item1.Bitmap);
            }

            return new Tuple<Bitmap, SimpleDatum>(bmp, data.Item2);
        }

        private Tuple<DirectBitmap, SimpleDatum> getData(COLORTYPE ct, int nWid, int nHt, int nOffset, int nDownsample, byte[] rg, bool bPreprocess, bool bGetAction, bool bForceGray)
        {
            int nChannels = (bPreprocess || bForceGray) ? 1 : 3;
            int nSize = Math.Min(nWid, nHt);
            int nDsSize = nSize / nDownsample;
            int nX = 0;
            int nY = 0;
            bool bY = false;
            bool bX = false;

            if (m_bmpRaw != null && (m_bmpRaw.Width != nWid || m_bmpRaw.Height != nHt))
            {
                m_bmpRaw.Dispose();
                m_bmpRaw = null;
            }

            if (m_bmpActionRaw != null && (m_bmpActionRaw.Width != nDsSize || m_bmpActionRaw.Height != nDsSize))
            {
                m_bmpActionRaw.Dispose();
                m_bmpActionRaw = null;
            }

            if (m_bmpRaw == null)
                m_bmpRaw = new DirectBitmap(nWid, nHt);

            if (m_bmpActionRaw == null)
                m_bmpActionRaw = new DirectBitmap(nDsSize, nDsSize);

            DirectBitmap bmp = m_bmpRaw;
            Valuemap dataV = null;
            Bytemap dataB = null;

            if (bGetAction)
            {
                if (m_bPreprocess)
                    dataV = new Valuemap(nChannels, nDsSize, nDsSize);
                else
                    dataB = new Bytemap(nChannels, nDsSize, nDsSize);
            }

            for (int y = 0; y < nHt; y++)
            {
                if (y % nDownsample == 0 && y > nOffset && y < nOffset + nSize)
                    bY = true;
                else
                    bY = false;

                for (int x = 0; x < nWid; x++)
                {
                    int nIdx = (y * nWid) + x;
                    int nR = rg[nIdx];
                    int nG = nR;
                    int nB = nR;

                    if (x % nDownsample == 0)
                        bX = true;
                    else
                        bX = false;

                    if (ct == COLORTYPE.CT_COLOR)
                    {
                        nG = rg[nIdx + (nWid * nHt * 1)];
                        nB = rg[nIdx + (nWid * nHt * 2)];
                    }

                    Color clr = Color.FromArgb(nR, nG, nB);

                    bmp.SetPixel(x, y, clr);

                    if (bForceGray)
                    {
                        int nClr = (nR + nG + nB) / 3;
                        clr = Color.FromArgb(nClr, nClr, nClr);
                    }

                    if (bY && bX && (dataB != null || dataV != null))
                    {
                        if (bPreprocess)
                        {
                            if (nR != 144 && nR != 109 && nR != 0)
                                dataV.SetPixel(nX, nY, 1.0);
                        }
                        else
                        {
                            dataB.SetPixel(nX, nY, clr);
                        }

                        nX++;
                    }
                }

                if (bY)
                {
                    nX = 0;
                    nY++;
                }
            }

            SimpleDatum sd = null;

            if (m_bPreprocess)
            {
                if (dataV != null)
                    sd = new SimpleDatum(dataV);
            }
            else
            {
                if (dataB != null)
                    sd = new SimpleDatum(dataB);
            }

            return new Tuple<DirectBitmap, SimpleDatum>(bmp, sd);
        }

        /// <summary>
        /// Reset the state of the gym.
        /// </summary>
        /// <param name="bGetLabel">Not used.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Reset(bool bGetLabel)
        {
            m_state = new AtariState();
            m_ale.ResetGame();
            return new Tuple<State, double, bool>(m_state.Clone(), 1, m_ale.GameOver);
        }

        /// <summary>
        /// Step the gym one step in its simulation.
        /// </summary>
        /// <param name="nAction">Specifies the action to run on the gym.</param>
        /// <param name="bGetLabel">Not used.</param>
        /// <param name="propExtra">Optionally, specifies extra parameters.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Step(int nAction, bool bGetLabel, PropertySet propExtra = null)
        {
            ACTION action = (ACTION)m_rgActionSet[nAction].Value;
            double dfReward = m_ale.Act(action);

            if (m_bEnableNumSkip)
            {
                int nIdx = 0;

                if (m_rgFrameSkip.Count > 1)
                    nIdx = m_random.Next(m_rgFrameSkip.Count);

                int nNumSkip = m_rgFrameSkip[nIdx];

                for (int i = 0; i < nNumSkip; i++)
                {
                    dfReward += m_ale.Act(action);
                }
            }

            return new Tuple<State, double, bool>(new AtariState(), dfReward, m_ale.GameOver);
        }

        /// <summary>
        /// Returns the dataset descriptor of the dynamic dataset produced by the Gym.
        /// </summary>
        /// <param name="dt">Specifies the data-type to use.</param>
        /// <param name="log">Optionally, specifies the output log to use (default = <i>null</i>).</param>
        /// <returns>The dataset descriptor is returned.</returns>
        public DatasetDescriptor GetDataset(DATA_TYPE dt, Log log = null)
        {
            if (dt == DATA_TYPE.DEFAULT)
                dt = DATA_TYPE.BLOB;

            if (dt != DATA_TYPE.BLOB)
            {
                if (log == null)
                    log = m_log;

                if (log != null)
                    log.WriteLine("WARNING: This gym only supports the BLOB type, the datatype will be changed to BLOB.");
                else
                    throw new Exception("This gym only supports the BLOB type.");

                dt = DATA_TYPE.BLOB;
            }

            int nH = 80;
            int nW = 80;
            int nC = 1;

            SourceDescriptor srcTrain = new SourceDescriptor((int)GYM_DS_ID.ATARI, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor((int)GYM_SRC_TEST_ID.ATARI, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor((int)GYM_SRC_TRAIN_ID.ATARI, Name, null, null, srcTrain, srcTest, "AtariGym", "Atari Gym", null, GYM_TYPE.DYNAMIC);

            m_dt = dt;

            return ds;
        }
    }

    class AtariState : State /** @private */
    {
        public AtariState()
        {
        }

        public AtariState(AtariState s)
        {
        }

        public override State Clone()
        {
            return new AtariState(this);
        }

        public override SimpleDatum GetData(bool bNormalize, out int nDataLen)
        {
            nDataLen = 0;
            return new SimpleDatum();
        }
    }
}
