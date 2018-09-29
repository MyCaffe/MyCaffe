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
            m_ale.EnableColorData = false;
            m_ale.EnableRestrictedActionSet = true;
            m_ale.EnableColorAveraging = true;
            m_ale.RandomSeed = (int)DateTime.Now.Ticks;
            m_ale.RepeatActionProbability = 0.0f; // disable action repeatability

            if (properties == null)
                throw new Exception("The properties must be specified with the 'GameROM' set the the Game ROM file path.");

            string strROM = properties.GetProperty("GameROM");
            if (!File.Exists(strROM))
                throw new Exception("Could not find the game ROM file specified '" + strROM + "'!");

            if (properties.GetPropertyAsBool("UseGrayscale", false))
                m_ct = COLORTYPE.CT_GRAYSCALE;

            m_bPreprocess = properties.GetPropertyAsBool("Preprocess", true);

            m_ale.Load(strROM);
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


        public IXMyCaffeGym Clone(PropertySet properties = null)
        {
            AtariGym gym = new AtariGym();

            if (properties != null)
                gym.Initialize(m_log, properties);

            return gym;
        }

        public DATA_TYPE SelectedDataType
        {
            get { return m_dt; }
        }

        public DATA_TYPE[] SupportedDataType
        {
            get { return new DATA_TYPE[] { DATA_TYPE.BLOB }; }
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
            float fWid;
            float fHt;

            m_ale.GetScreenDimensions(out fWid, out fHt);
            byte[] rgRawData = m_ale.GetScreenData(m_ct);

            Tuple<DirectBitmap, DirectBitmap> bmps = getBitmaps(m_ct, (int)fWid, (int)fHt, 35, 2, rgRawData);

            if (m_bPreprocess)
            {
                DirectBitmap bmpRawAction = bmps.Item2;

                for (int y = 0; y < bmpRawAction.Height; y++)
                {
                    for (int x = 0; x < bmpRawAction.Width; x++)
                    {
                        Color clr = bmpRawAction.GetPixel(x, y);
                        int nR = clr.R;

                        if (nR == 144 || nR == 109)
                            nR = 0;       // erase background (type 1 and 2)
                        else if (nR != 0)
                            nR = 255;     // everything else (paddles, ball) just set to 1

                        bmpRawAction.SetPixel(x, y, Color.FromArgb(nR, nR, nR));
                    }
                }
            }

            Bitmap bmp;
            if (bmps.Item1.Bitmap.Width != nWidth || bmps.Item1.Bitmap.Height != nHeight)
                bmp = ImageTools.ResizeImage(bmps.Item1.Bitmap, nWidth, nHeight);
            else
                bmp = new Bitmap(bmps.Item1.Bitmap);

            bmpAction = new Bitmap(bmps.Item2.Bitmap);

            bmps.Item1.Dispose();
            bmps.Item2.Dispose();

            return bmp;
        }

        private Tuple<DirectBitmap, DirectBitmap> getBitmaps(COLORTYPE ct, int nWid, int nHt, int nOffset, int nDownsample, byte[] rg)
        {
            int nSize = Math.Min(nWid, nHt);
            int nDsSize = nSize / nDownsample;
            int nX = 0;
            int nY = 0;
            bool bY = false;
            bool bX = false;
            DirectBitmap bmp = new DirectBitmap(nWid, nHt);
            DirectBitmap bmpA = new DirectBitmap(nDsSize, nDsSize);

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

                    if (bY && bX)
                    {
                        bmpA.SetPixel(nX, nY, clr);
                        nX++;
                    }
                }

                if (bY)
                {
                    nX = 0;
                    nY++;
                }
            }

            return new Tuple<DirectBitmap, DirectBitmap>(bmp, bmpA);
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
                dfReward += m_ale.Act(action);                
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

    /// <summary>
    /// See https://stackoverflow.com/questions/24701703/c-sharp-faster-alternatives-to-setpixel-and-getpixel-for-bitmaps-for-windows-f
    /// </summary>
    public class DirectBitmap : IDisposable
    {
        public Bitmap Bitmap { get; private set; }
        public Int32[] Bits { get; private set; }
        public bool Disposed { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }

        protected GCHandle BitsHandle { get; private set; }

        public DirectBitmap(int width, int height)
        {
            Width = width;
            Height = height;
            Bits = new Int32[width * height];
            BitsHandle = GCHandle.Alloc(Bits, GCHandleType.Pinned);
            Bitmap = new Bitmap(width, height, width * 4, PixelFormat.Format32bppPArgb, BitsHandle.AddrOfPinnedObject());
        }

        public void SetPixel(int x, int y, Color colour)
        {
            int index = x + (y * Width);
            int col = colour.ToArgb();

            Bits[index] = col;
        }

        public Color GetPixel(int x, int y)
        {
            int index = x + (y * Width);
            int col = Bits[index];
            Color result = Color.FromArgb(col);

            return result;
        }

        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;

            if (Bitmap != null)
                Bitmap.Dispose();

            BitsHandle.Free();
        }
    }
}
