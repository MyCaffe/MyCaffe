using AleControlLib;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
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
    /// Licence: https://github.com/openai/gym/blob/master/LICENSE.md
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
        string m_strParam = "";
        IALE m_ale = null;
        AtariState m_state = new AtariState();
        Log m_log;
        ACTION[] m_rgActionsRaw;
        List<int> m_rgFrameSkip = new List<int>();
        CryptoRandom m_random;
        Dictionary<string, int> m_rgActions = new Dictionary<string, int>();
        List<KeyValuePair<string, int>> m_rgActionSet;
        DATA_TYPE m_dt = DATA_TYPE.BLOB;

        public AtariGym()
        {
        }

        public void Dispose()
        {
            if (m_ale != null)
            {
                m_ale.Shutdown();
                m_ale = null;               
            }
        }

        public void Initialize(Log log, string strParameter, double[] rgdfInit)
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
            m_ale.EnableColorData = false;
            m_ale.EnableRestrictedActionSet = true;
            m_ale.EnableColorAveraging = true;
            m_ale.RandomSeed = DateTime.Now.Millisecond;

            if (!File.Exists(strParameter))
                throw new Exception("Could not find the game ROM file specified '" + strParameter + "'!");

            m_strParam = strParameter;
            m_ale.Load(m_strParam);
            m_rgActionsRaw = m_ale.ActionSpace;
            m_random = new CryptoRandom();

            for (int i = 2; i < 5; i++)
            {
                m_rgFrameSkip.Add(i);
            }

            m_rgActions.Add(ACTION.ACT_PLAYER_A_LEFT.ToString(), (int)ACTION.ACT_PLAYER_A_LEFT);
            m_rgActions.Add(ACTION.ACT_PLAYER_A_RIGHT.ToString(), (int)ACTION.ACT_PLAYER_A_RIGHT);
            m_rgActionSet = m_rgActions.ToList();

            Reset();
        }


        public IXMyCaffeGym Clone(bool bInitialize)
        {
            AtariGym gym = new AtariGym();

            if (bInitialize)
            {
                List<double> rgdfInit = new List<double>();
                gym.Initialize(m_log, m_strParam, rgdfInit.ToArray());
            }

            return gym;
        }

        public DATA_TYPE SelectedDataType
        {
            get { return m_dt; }
        }

        public string Name
        {
            get { return m_strName; }
        }

        public int UiDelay
        {
            get { return 0; }
        }

        public Dictionary<string, int> GetActionSpace()
        {
            return m_rgActions;
        }

        public void Close()
        {
            if (m_ale != null)
            {
                m_ale.Shutdown();
                m_ale = null;
            }
        }

        public Bitmap Render(int nWidth, int nHeight, out Bitmap bmpAction)
        {
            List<double> rgData = new List<double>();
            return Render(nWidth, nHeight, rgData.ToArray(), out bmpAction);
        }

        public Bitmap Render(int nWidth, int nHeight, double[] rgData, out Bitmap bmpAction)
        {
            COLORTYPE ct = COLORTYPE.CT_COLOR;
            float fWid;
            float fHt;

            m_ale.GetScreenDimensions(out fWid, out fHt);
            byte[] rgRawData = m_ale.GetScreenData(ct);

            Bitmap bmp = getBitmap(ct, (int)fWid, (int)fHt, rgRawData);
            int nSize = Math.Min((int)fWid, (int)fHt);
            bmpAction = new Bitmap(nSize, nSize);

            using (Graphics g = Graphics.FromImage(bmpAction))
            {
                int nY = 0;
                if (fHt > fWid)
                    nY = (int)(fHt - fWid);

                Rectangle rcDst = new Rectangle(0, 0, nSize, nSize);
                Rectangle rcSrc = new Rectangle(0, nY, nSize, nSize);

                g.DrawImage(bmp, rcDst, rcSrc, GraphicsUnit.Pixel);
            }

            if (bmpAction.Width != 80 || bmpAction.Height != 80)
                bmpAction = ImageTools.ResizeImage(bmpAction, 80, 80);

            if (bmp.Width != nWidth || bmp.Height != nHeight)
                bmp = ImageTools.ResizeImage(bmp, nWidth, nHeight);

            return bmp;
        }

        private Bitmap getBitmap(COLORTYPE ct, int nWid, int nHt, byte[] rg)
        {
            Bitmap bmp = new Bitmap(nWid, nHt);

            for (int y = 0; y < nHt; y++)
            {
                for (int x = 0; x < nWid; x++)
                {
                    int nIdx = (y * nWid) + x;
                    int nR = rg[nIdx];
                    int nG = nR;
                    int nB = nR;

                    if (ct == COLORTYPE.CT_COLOR)
                    {
                        nG = rg[nIdx + (nWid * nHt * 1)];
                        nB = rg[nIdx + (nWid * nHt * 2)];
                    }

                    Color clr = Color.FromArgb(nR, nG, nB);
                    bmp.SetPixel(x, y, clr);
                }
            }

            return bmp;
        }

        public Tuple<State, double, bool> Reset()
        {
            m_state = new AtariState();
            m_ale.ResetGame();
            return new Tuple<State, double, bool>(m_state.Clone(), 1, m_ale.GameOver);
        }

        public Tuple<State, double, bool> Step(int nAction)
        {
            ACTION action = (ACTION)m_rgActionSet[nAction].Value;
            double dfReward = m_ale.Act(action);
            int nIdx = m_random.Next(m_rgFrameSkip.Count);
            int nNumSkip = m_rgFrameSkip[nIdx];

            for (int i = 0; i < nNumSkip; i++)
            {
                dfReward += m_ale.Act((ACTION)nAction);                
            }

            return new Tuple<State, double, bool>(new AtariState(), dfReward, m_ale.GameOver);
        }

        public DatasetDescriptor GetDataset(DATA_TYPE dt)
        {
            if (dt == DATA_TYPE.DEFAULT)
                dt = DATA_TYPE.BLOB;

            if (dt != DATA_TYPE.BLOB)
                throw new Exception("This gym only supports the BLOB data type at this time.");

            int nH = 80;
            int nW = 80;
            int nC = 1;

            SourceDescriptor srcTrain = new SourceDescriptor(9999988, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor(9999989, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor(9999989, Name, null, null, srcTrain, srcTest, "AtariGym", "Atari Gym", null, true);

            m_dt = dt;

            return ds;
        }
    }

    class AtariState : State
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

        public override Tuple<double,double,double, bool>[] ToArray()
        {
            List<Tuple<double, double, double, bool>> rg = new List<Tuple<double, double, double, bool>>();
            return rg.ToArray();
        }
    }
}
