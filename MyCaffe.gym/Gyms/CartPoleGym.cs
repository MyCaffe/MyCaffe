using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The CartPole Gym provides a simulation of a cart with a balancing pole standing on top of it.
    /// </summary>
    /// <remarks>
    /// This gym is a rewrite of the original gym provided by OpenAi under the MIT license and located
    /// on GitHub at: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    /// License: https://github.com/openai/gym/blob/master/LICENSE.md
    /// 
    /// OpenAi notes that their implementation is a 'classic cart-pole system implemented by Rich Sutton et al.'
    /// copied from http://incompleteideas.net/sutton/book/code/pole.c with permalink: https://perma.cc/C9ZM-652R
    /// </remarks>
    public class CartPoleGym : IXMyCaffeGym
    {
        string m_strName = "Cart-Pole";
        double m_dfGravity = 9.8;
        double m_dfMassCart = 1.0;
        double m_dfMassPole = 0.1;
        double m_dfTotalMass;
        double m_dfLength = 0.5; // actually half the pole's length
        double m_dfPoleMassLength;
        double m_dfForce = 10;
        bool m_bAdditive = false;
        double m_dfTau = 0.02; // seconds between state updates.
        Dictionary<string, int> m_rgActionSpace;
        Bitmap m_bmp = null;
        int m_nSteps = 0;
        int m_nMaxSteps = 0;
        ColorMapper m_clrMap = null;
        DATA_TYPE m_dt = DATA_TYPE.VALUES;

        // Angle at which to fail the episode
        double m_dfThetaThreshold = CartPoleState.MAX_THETA;
        double m_dfXThreshold = CartPoleState.MAX_X;

        Random m_random = new Random();
        CartPoleState m_state = new CartPoleState();
        int? m_nStepsBeyondDone = null;
        Log m_log;

        /// <summary>
        /// Defines the actions to perform.
        /// </summary>
        public enum ACTION
        {
            /// <summary>
            /// Move the cart left.
            /// </summary>
            MOVELEFT,
            /// <summary>
            /// Move the cart right.
            /// </summary>
            MOVERIGHT
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public CartPoleGym()
        {
            m_dfTotalMass = m_dfMassPole + m_dfMassCart;
            m_dfPoleMassLength = m_dfMassPole * m_dfLength;

            m_rgActionSpace = new Dictionary<string, int>();
            m_rgActionSpace.Add("MoveLeft", 0);
            m_rgActionSpace.Add("MoveRight", 1);
        }

        /// <summary>
        /// Initialize the gym with the specified properties.
        /// </summary>
        /// <param name="log">Specifies the output log to use.</param>
        /// <param name="properties">Specifies the properties containing Gym specific initialization parameters.</param>
        /// <remarks>
        /// The AtariGym uses the following initialization properties.
        ///   Init1=value - specifies the default force to use.
        ///   Init2=value - specifies whether to use an additive force (1) or not (0).
        /// </remarks>
        public void Initialize(Log log, PropertySet properties)
        {
            m_dfForce = 10;
            m_bAdditive = false;

            if (properties != null)
            {
                m_dfForce = properties.GetPropertyAsDouble("Init1", 10);
                m_bAdditive = (properties.GetPropertyAsDouble("Init2", 0) == 0) ? false : true;
            }

            m_log = log;
            m_nMaxSteps = 0;
            Reset(false);
        }


        /// <summary>
        /// Create a new copy of the gym.
        /// </summary>
        /// <param name="properties">Optionally, specifies the properties to initialize the new copy with.</param>
        /// <returns>The new Gym copy is returned.</returns>
        public IXMyCaffeGym Clone(PropertySet properties = null)
        {
            CartPoleGym gym = new CartPoleGym();

            if (properties != null)
                gym.Initialize(m_log, properties);

            return gym;
        }

        /// <summary>
        /// Returns <i>false</i> indicating that this Gym does not require a display image.
        /// </summary>
        public bool RequiresDisplayImage
        {
            get { return false; }
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
            get { return new DATA_TYPE[] { DATA_TYPE.VALUES, DATA_TYPE.BLOB }; }
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
            get { return 20; }
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
            return m_rgActionSpace;
        }

        private void processAction(ACTION? a)
        {
            if (a.HasValue)
            {
                switch (a)
                {
                    case ACTION.MOVELEFT:
                        m_state.ForceMag = (m_state.ForceMag * ((m_bAdditive) ? 1 : 0)) + m_dfForce * -1;
                        break;

                    case ACTION.MOVERIGHT:
                        m_state.ForceMag = (m_state.ForceMag * ((m_bAdditive) ? 1 : 0)) + m_dfForce * 1;
                        break;
                }
            }
        }

        /// <summary>
        /// Shutdown and close the gym.
        /// </summary>
        public void Close()
        {
        }

        /// <summary>
        /// Not used.
        /// </summary>
        public void ResetValue()
        {
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

            rgData.Add(m_state.X);
            rgData.Add(m_state.XDot);
            rgData.Add(m_state.Theta);
            rgData.Add(m_state.ThetaDot);
            rgData.Add(m_state.ForceMag);
            rgData.Add(m_nSteps);

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
        /// <param name="predictions">Optionally, specifies the future predictions.</param>
        /// <returns>A tuple optionally containing a Bitmap and/or Simpledatum is returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, double[] rgData, bool bGetAction, FuturePredictions predictions = null)
        { 
            Bitmap bmp = new Bitmap(nWidth, nHeight);

            double dfX = rgData[0];
            double dfTheta = rgData[2];
            double dfThetaInDegrees = dfTheta * (180.0 / Math.PI);
            double dfForceMag = rgData[4];
            int nSteps = (int)rgData[5];

            m_nSteps = nSteps;
            m_nMaxSteps = Math.Max(nSteps, m_nMaxSteps);

            using (Graphics g = Graphics.FromImage(bmp))
            {
                Rectangle rc = new Rectangle(0, 0, bmp.Width, bmp.Height);
                g.FillRectangle(Brushes.White, rc);

                float fScreenWidth = g.VisibleClipBounds.Width;
                float fScreenHeight = g.VisibleClipBounds.Height;
                float fWorldWidth = (float)(m_dfXThreshold * 2);
                float fScale = fScreenWidth / fWorldWidth;
                float fCartY = 100; // Top of Cart;
                float fPoleWidth = 10;
                float fPoleLen = fScale * 1.0f;
                float fCartWidth = 50;
                float fCartHeight = 30;

                float fL = -fCartWidth / 2;
                float fR = fCartWidth / 2;
                float fT = fCartHeight / 2;
                float fB = -fCartHeight / 2;
                float fAxleOffset = 0;
                GeomCart cart = new GeomCart(fL, fR, fT, fB, Color.SkyBlue, Color.Black);

                fL = -fPoleWidth / 2;
                fR = fPoleWidth / 2;
                fT = fPoleLen - fPoleWidth / 2;
                fB = --fPoleWidth / 2;
                GeomPole pole = new GeomPole(fL, fR, fT, fB, Color.Tan, Color.Black);

                fL = 0;
                fR = fScreenWidth;
                fT = fCartY;
                fB = fT;
                GeomLine track = new GeomLine(fL, fR, fT, fB, Color.Black, Color.Black);

                fL = 0;
                fR = fScreenWidth;
                fT = fCartY - 40;
                fB = fT + 10;

                if (m_clrMap == null)
                    m_clrMap = new ColorMapper(fL, fR, Color.Fuchsia, Color.Red);

                GeomRectangle posbar = new GeomRectangle(fL, fR, fT, fB, Color.Black, Color.Transparent, m_clrMap);

                float fCartX = (float)dfX * fScale + fScreenWidth / 2;   // middle of the cart.
                cart.SetLocation(fCartX, fCartY);
                pole.SetRotation((float)-dfThetaInDegrees);
                cart.Attach(pole, fAxleOffset);

                GeomView view = new GeomView();

                view.RenderText(g, "Current Force = " + dfForceMag.ToString(), 10, 10);
                view.RenderText(g, "X = " + dfX.ToString("N02"), 10, 24);
                view.RenderText(g, "Theta = " + dfTheta.ToString("N02") + " radians", 10, 36);
                view.RenderText(g, "Theta = " + dfThetaInDegrees.ToString("N02") + " degrees", 10, 48);
                view.RenderSteps(g, m_nSteps, m_nMaxSteps);

                // Render the objects.
                view.AddObject(posbar);
                view.AddObject(track);
                view.AddObject(cart);
                view.Render(g);

                SimpleDatum sdAction = null;

                if (bGetAction)
                    sdAction = getActionData(fCartX, fCartY, bmp);

                m_bmp = bmp;

                return new Tuple<Bitmap, SimpleDatum>(bmp, sdAction);
            }
        }

        private SimpleDatum getActionData(float fX, float fY, Bitmap bmpSrc)
        {
            double dfWid = 156;
            double dfHt = 156;
            double dfX = fX - (dfWid * 0.5);
            double dfY = (bmpSrc.Height - fY) - (dfHt * 0.75);

            RectangleF rc = new RectangleF((float)dfX, (float)dfY, (float)dfWid, (float)dfHt);
            Bitmap bmp = new Bitmap((int)dfWid, (int)dfHt);

            using (Graphics g = Graphics.FromImage(bmp))
            {
                RectangleF rc1 = new RectangleF(0, 0, (float)dfWid, (float)dfHt);
                g.FillRectangle(Brushes.Black, rc1);
                g.DrawImage(bmpSrc, rc1, rc, GraphicsUnit.Pixel);
            }

            return ImageData.GetImageDataD(bmp, 3, false, -1);
        }

        /// <summary>
        /// Reset the state of the gym.
        /// </summary>
        /// <param name="bGetLabel">Not used.</param>
        /// <param name="props">Optionally, specifies extra properties.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Reset(bool bGetLabel, PropertySet props = null)
        {
            double dfX = randomUniform(-0.05, 0.05);
            double dfXDot = randomUniform(-0.05, 0.05);
            double dfTheta = randomUniform(-0.05, 0.05);
            double dfThetaDot = randomUniform(-0.05, 0.05);
            m_nStepsBeyondDone = null;
            m_nSteps = 0;

            m_state = new CartPoleState(dfX, dfXDot, dfTheta, dfThetaDot);
            return new Tuple<State, double, bool>(m_state.Clone(), 1, false);
        }

        private double randomUniform(double dfMin, double dfMax)
        {
            double dfRange = dfMax - dfMin;
            return dfMin + (m_random.NextDouble() * dfRange);
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
            CartPoleState state = new CartPoleState(m_state);
            double dfReward = 0;

            processAction((ACTION)nAction);

            double dfX = state.X;
            double dfXDot = state.XDot;
            double dfTheta = state.Theta;
            double dfThetaDot = state.ThetaDot;
            double dfForce = m_state.ForceMag;
            double dfCosTheta = Math.Cos(dfTheta);
            double dfSinTheta = Math.Sin(dfTheta);
            double dfTemp = (dfForce + m_dfPoleMassLength * dfThetaDot * dfThetaDot * dfSinTheta) / m_dfTotalMass;
            double dfThetaAcc = (m_dfGravity * dfSinTheta - dfCosTheta * dfTemp) / (m_dfLength * ((4.0 / 3.0) - m_dfMassPole * dfCosTheta * dfCosTheta / m_dfTotalMass));
            double dfXAcc = dfTemp - m_dfPoleMassLength * dfThetaAcc * dfCosTheta / m_dfTotalMass;

            dfX += m_dfTau * dfXDot;
            dfXDot += m_dfTau * dfXAcc;
            dfTheta += m_dfTau * dfThetaDot;
            dfThetaDot += m_dfTau * dfThetaAcc;

            CartPoleState stateOut = m_state;
            m_state = new CartPoleState(dfX, dfXDot, dfTheta, dfThetaDot);

            bool bDone = false;

            if (dfX < -m_dfXThreshold || dfX > m_dfXThreshold ||
                dfTheta < -m_dfThetaThreshold || dfTheta > m_dfThetaThreshold)
                bDone = true;

            if (!bDone)
            {
                dfReward = 1.0;
            }
            else if (!m_nStepsBeyondDone.HasValue)
            {
                // Pole just fell!
                m_nStepsBeyondDone = 0;
                dfReward = 1.0;
            }
            else
            {
                if (m_nStepsBeyondDone.Value == 0)
                    m_log.WriteLine("WARNING: You are calling 'step()' even though this environment has already returned done = True.  You should always call 'reset()'");

                m_nStepsBeyondDone++;
                dfReward = 0.0;
            }

            m_nSteps++;
            m_nMaxSteps = Math.Max(m_nMaxSteps, m_nSteps);

            stateOut.Steps = m_nSteps;
            return new Tuple<State, double, bool>(stateOut.Clone(), dfReward, bDone);
        }

        /// <summary>
        /// Returns the dataset descriptor of the dynamic dataset produced by the Gym.
        /// </summary>
        /// <param name="dt">Specifies the data-type to use.</param>
        /// <param name="log">Optionally, specifies the output log to use (default = <i>null</i>).</param>
        /// <returns>The dataset descriptor is returned.</returns>
        public DatasetDescriptor GetDataset(DATA_TYPE dt, Log log = null)
        {
            int nH = 1;
            int nW = 1;
            int nC = 4;

            if (dt == DATA_TYPE.DEFAULT)
                dt = DATA_TYPE.VALUES;

            if (dt == DATA_TYPE.BLOB)
            {
                nH = 156;
                nW = 156;
                nC = 3;
            }

            SourceDescriptor srcTrain = new SourceDescriptor((int)GYM_DS_ID.CARTPOLE, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor((int)GYM_SRC_TEST_ID.CARTPOLE, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor((int)GYM_SRC_TRAIN_ID.CARTPOLE, Name, null, null, srcTrain, srcTest, "CartPoleGym", "CartPole Gym", null, GYM_TYPE.DYNAMIC);

            m_dt = dt;

            return ds;
        }
    }

    class GeomCart : GeomPolygon /** @private */
    {
        GeomPole m_pole;

        public GeomCart(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
        }

        public void Attach(GeomPole pole, float fXOffset)
        {
            m_pole = pole;
            m_pole.SetLocation(Location.X + fXOffset, Location.Y);
        }

        public override void Render(Graphics g)
        {
            base.Render(g);
            m_pole.Render(g);
        }
    }

    class GeomPole : GeomPolygon /** @private */
    {
        GeomEllipse m_axis;

        public GeomPole(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
            float fWid = fR - fL;
            m_axis = new GeomEllipse(fL, fR, fB - fWid, fB, Color.Brown, Color.Black);
        }

        public override void SetLocation(float fX, float fY)
        {
            m_axis.SetLocation(fX, fY);
            base.SetLocation(fX, fY);
        }

        public override void Render(Graphics g)
        {
            base.Render(g);
            m_axis.Render(g);
        }
    }

    class CartPoleState : State /** @private */
    {
        double m_dfX = 0;
        double m_dfXDot = 0;
        double m_dfTheta = 0;
        double m_dfThetaDot = 0;
        double m_dfForceMag = 0;
        int m_nSteps = 0;

        public const double MAX_X = 2.4;
        public const double MAX_THETA = 20 * (Math.PI/180);

        public CartPoleState(double dfX = 0, double dfXDot = 0, double dfTheta = 0, double dfThetaDot = 0)
        {
            m_dfX = dfX;
            m_dfXDot = dfXDot;
            m_dfTheta = dfTheta;
            m_dfThetaDot = dfThetaDot;
            m_dfForceMag = 0;
        }

        public CartPoleState(CartPoleState s)
        {
            m_dfX = s.m_dfX;
            m_dfXDot = s.m_dfXDot;
            m_dfTheta = s.m_dfTheta;
            m_dfThetaDot = s.m_dfThetaDot;
            m_dfForceMag = s.m_dfForceMag;
            m_nSteps = s.m_nSteps;
        }

        public int Steps
        {
            get { return m_nSteps; }
            set { m_nSteps = value; }
        }

        public double ForceMag
        {
            get { return m_dfForceMag; }
            set { m_dfForceMag = value; }
        }

        public double X
        {
            get { return m_dfX; }
            set { m_dfX = value; }
        }

        public double XDot
        {
            get { return m_dfXDot; }
            set { m_dfXDot = value; }
        }

        public double Theta
        {
            get { return m_dfTheta; }
            set { m_dfTheta = value; }
        }

        public double ThetaDot
        {
            get { return m_dfThetaDot; }
            set { m_dfThetaDot = value; }
        }

        public double ThetaInDegrees
        {
            get
            {
                return m_dfTheta * (180.0/Math.PI);
            }
        }

        public override State Clone()
        {
            return new CartPoleState(this);
        }

        public override SimpleDatum GetData(bool bNormalize, out int nDataLen)
        {
            int nScale = 4;
            nDataLen = 4;
            Valuemap data = new Valuemap(1, 6, 1);

            data.SetPixel(0, 0, getValue(m_dfX, -MAX_X, MAX_X, bNormalize));
            data.SetPixel(0, 1, getValue(m_dfXDot, -MAX_X * nScale, MAX_X * nScale, bNormalize));
            data.SetPixel(0, 2, getValue(m_dfTheta, -MAX_THETA, MAX_THETA, bNormalize));
            data.SetPixel(0, 3, getValue(m_dfThetaDot, -MAX_THETA * nScale * 2, MAX_THETA * nScale * 2, bNormalize));
            data.SetPixel(0, 4, getValue(m_dfForceMag, -100, 100, bNormalize));
            data.SetPixel(0, 5, m_nSteps);

            return new SimpleDatum(data);
        }

        private double getValue(double dfVal, double dfMin, double dfMax, bool bNormalize)
        {
            if (!bNormalize)
                return dfVal;

            return (dfVal - dfMin) / (dfMax - dfMin);
        }
    }
}
