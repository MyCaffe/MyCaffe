using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The CartPole Gym provides a simulation of a cart with a balancing pole standing on top of it.
    /// </summary>
    /// <remarks>
    /// This gym is a rewrite of the original gym provided by OpenAi under the MIT license and located
    /// on GitHub at: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    /// Licence: https://github.com/openai/gym/blob/master/LICENSE.md
    /// 
    /// OpenAi notes that their implementation is a 'classic cart-pole system implemented by Rich Sutton et al.'
    /// copied from http://incompleteideas.net/sutton/book/code/pole.c with permalink: https://perma.cc/C9ZM-652R
    /// </remarks>
    public class CartPoleGym : IXMyCaffeGym
    {
        string m_strName = "Cart Pole";
        List<int> m_rgActions = new List<int>();
        object m_objActionSync = new object();
        double m_dfGravity = 9.8;
        double m_dfMassCart = 1.0;
        double m_dfMassPole = 0.1;
        double m_dfTotalMass;
        double m_dfLength = 0.5; // actually half the pole's length
        double m_dfPoleMassLength;
        double m_dfForceMag = 0.0;
        double m_dfTau = 0.02; // seconds between state updates.
        bool m_bDone = false;
        Dictionary<string, int> m_rgActionSpace;
        Bitmap m_bmp = null;
        int m_nSteps = 0;
        int m_nMaxSteps = 0;

        // Angle at which to fail the episode
        double m_dfThetaThreshold = CartPoleState.MAX_THETA;
        double m_dfXThreshold = CartPoleState.MAX_X;

        Random m_random = new Random();
        CartPoleState m_state = new CartPoleState();
        int? m_nStepsBeyondDone = null;
        Log m_log;

        public enum ACTION
        {
            MOVELEFT,
            MOVERIGHT
        }


        public CartPoleGym()
        {
            m_dfTotalMass = m_dfMassPole + m_dfMassCart;
            m_dfPoleMassLength = m_dfMassPole * m_dfLength;

            m_rgActionSpace = new Dictionary<string, int>();
            m_rgActionSpace.Add("MoveLeft", 0);
            m_rgActionSpace.Add("MoveRight", 1);
        }

        public IXMyCaffeGym Clone()
        {
            return new CartPoleGym();
        }

        public string Name
        {
            get { return m_strName; }
        }

        public void AddAction(int nAction)
        {
            lock (m_objActionSync)
            {
                m_rgActions.Add(nAction);
            }
        }

        public Dictionary<string, int> GetActionSpace()
        {
            return m_rgActionSpace;
        }

        private int? getNextAction()
        {
            lock (m_objActionSync)
            {
                if (m_rgActions.Count == 0)
                    return null;

                int nAction = m_rgActions[0];
                m_rgActions.RemoveAt(0);
                return nAction;
            }
        }

        private ACTION? getNextActionValue()
        {
            return (ACTION?)getNextAction();
        }

        private void processActions()
        {
            ACTION? a = getNextActionValue();

            while (a.HasValue)
            {
                if (a == ACTION.MOVELEFT)
                    m_dfForceMag = -10;
                if (a == ACTION.MOVERIGHT)
                    m_dfForceMag = 10;

                a = getNextActionValue();
            }
        }

        public void Close()
        {
        }

        public void Initialize(Log log)
        {
            m_log = log;
            m_nMaxSteps = 0;
            Reset();
        }

        public Bitmap Render(int nWidth, int nHeight)
        {
            Bitmap bmp = new Bitmap(nWidth, nHeight);

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

                if (m_state != null)
                {
                    float fCartX = (float)m_state.X * fScale + fScreenWidth / 2;   // middle of the cart.
                    cart.SetLocation(fCartX, fCartY);
                    pole.SetRotation((float)-m_state.Theta);
                    cart.Attach(pole, fAxleOffset);

                    GeomView view = new GeomView();

                    view.RenderText(g, "Current Force = " + m_dfForceMag.ToString(), 10, 10);
                    view.RenderText(g, "X = " + m_state.X.ToString("N02"), 10, 24);
                    view.RenderText(g, "Theta = " + m_state.Theta.ToString("N02"), 10, 36);
                    view.RenderSteps(g, m_nSteps, m_nMaxSteps);

                    // Render the objects.
                    view.AddObject(track);
                    view.AddObject(cart);
                    view.Render(g);
                }

                m_bmp = bmp;

                return bmp;
            }
        }

        public Bitmap Image
        {
            get { return m_bmp; }
        }

        public void Reset()
        {
            double dfX = randomUniform(-0.05, 0.05);
            double dfXDot = randomUniform(-0.05, 0.05);
            double dfTheta = randomUniform(-0.05, 0.05);
            double dfThetaDot = randomUniform(-0.05, 0.05);
            m_dfForceMag = 0;
            m_nStepsBeyondDone = null;
            m_bDone = false;
            m_nSteps = 0;

            lock (m_objActionSync)
            {
                m_rgActions.Clear();
            }

            m_state = new CartPoleState(dfX, dfXDot, dfTheta, dfThetaDot);
        }

        private double randomUniform(double dfMin, double dfMax)
        {
            double dfRange = dfMax - dfMin;
            return dfMin + (m_random.NextDouble() * dfRange);
        }

        public Tuple<Tuple<double,double,double>[], double, bool> Step()
        {
            CartPoleState state = new CartPoleState(m_state);
            double dfReward = 0;

            if (!m_bDone)
            {
                processActions();

                double dfX = state.X;
                double dfXDot = state.XDot;
                double dfTheta = state.Theta;
                double dfThetaDot = state.ThetaDot;
                double dfForce = m_dfForceMag;
                double dfCosTheta = Math.Cos(dfTheta);
                double dfSinTheta = Math.Sin(dfTheta);
                double dfTemp = (dfForce + m_dfPoleMassLength * dfThetaDot * dfThetaDot * dfSinTheta) / m_dfTotalMass;
                double dfThetaAcc = (m_dfGravity * dfSinTheta - dfCosTheta * dfTemp) / (m_dfLength * (4.0 / 3.0 - m_dfMassPole * dfCosTheta * dfCosTheta / m_dfTotalMass));
                double dfXAcc = dfTemp - m_dfPoleMassLength * dfThetaAcc * dfCosTheta / m_dfTotalMass;

                dfX = dfX + m_dfTau * dfXDot;
                dfXDot = dfXDot + m_dfTau * dfXAcc;
                dfTheta = dfTheta + m_dfTau * dfThetaDot;
                dfThetaDot = dfThetaDot + m_dfTau * dfThetaAcc;

                m_state = new CartPoleState(dfX, dfXDot, dfTheta, dfThetaDot);

                bool bXDone = false;
                bool bThetaDone = false;

                if (dfX < -m_dfXThreshold || dfX > m_dfXThreshold)
                    bXDone = true;

                if (dfTheta < -m_dfThetaThreshold || dfTheta > m_dfThetaThreshold)
                    bThetaDone = true;

                if (bXDone || bThetaDone)
                    m_bDone = true;

                if (!m_bDone)
                {
                    dfReward = 1.0;
                }
                else if (!m_nStepsBeyondDone.HasValue)
                {
                    m_nStepsBeyondDone = 0;

                    if (bXDone) // ran off track
                        dfReward = -1;
                    else // pole fell
                        dfReward = 0.0;
                }
                else
                {
                    if (m_nStepsBeyondDone.Value == 0)
                        m_log.WriteLine("You are calling 'step()' even though this environment has already returned done = True.  You should always call 'reset()'");
                    m_nStepsBeyondDone++;
                    dfReward = 0.0;
                }

                m_nSteps++;
                m_nMaxSteps = Math.Max(m_nMaxSteps, m_nSteps);
            }

            return new Tuple<Tuple<double,double,double>[], double, bool>(m_state.ToArray(), dfReward, m_bDone);
        }

        public DatasetDescriptor GetDataset(DATA_TYPE dt)
        {
            int nH = 1;
            int nW = 1;
            int nC = 4;

            if (dt == DATA_TYPE.BLOB)
            {
                nH = 512;
                nW = 512;
                nC = 1;
            }

            SourceDescriptor srcTrain = new SourceDescriptor(9999998, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor(9999999, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor(9999999, Name, null, null, srcTrain, srcTest, "CartPoleGym", "CartPole Gym", null, true);

            return ds;
        }
    }

    class GeomCart : GeomPolygon
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

    class GeomPole : GeomPolygon
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

    class CartPoleState : State
    {
        double m_dfX = 0;
        double m_dfXDot = 0;
        double m_dfTheta = 0;
        double m_dfThetaDot = 0;

        public const double MAX_X = 2.4;
        public const double MAX_THETA = 15;

        public CartPoleState(double dfX = 0, double dfXDot = 0, double dfTheta = 0, double dfThetaDot = 0)
        {
            m_dfX = dfX;
            m_dfXDot = dfXDot;
            m_dfTheta = dfTheta;
            m_dfThetaDot = dfThetaDot;
        }

        public CartPoleState(CartPoleState s)
        {
            m_dfX = s.m_dfX;
            m_dfXDot = s.m_dfXDot;
            m_dfTheta = s.m_dfTheta;
            m_dfThetaDot = s.m_dfThetaDot;
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

        public Tuple<double,double,double>[] ToArray()
        {
            List<Tuple<double, double, double>> rg = new List<Tuple<double, double, double>>();

            rg.Add(new Tuple<double, double, double>(m_dfX, -MAX_X, MAX_X));
            rg.Add(new Tuple<double, double, double>(m_dfXDot, -MAX_X * 3, MAX_X * 3));
            rg.Add(new Tuple<double, double, double>(m_dfTheta, -MAX_THETA, MAX_THETA));
            rg.Add(new Tuple<double, double, double>(m_dfThetaDot, -MAX_THETA * 3, MAX_THETA * 3));

            return rg.ToArray();
        }
    }
}
