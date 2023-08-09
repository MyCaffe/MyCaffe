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
    /// The Curve Gym provides a simulation of continuous curve such as Sin or Cos.
    /// </summary>
    public class CurveGym : IXMyCaffeGym
    {
        string m_strName = "Curve";
        Dictionary<string, int> m_rgActionSpace;
        Bitmap m_bmp = null;
        int m_nSteps = 0;
        int m_nMaxSteps = int.MaxValue;
        ColorMapper m_clrMap = null;
        DATA_TYPE m_dt = DATA_TYPE.VALUES;
        GeomGraph m_geomGraph;
        GeomPolyLine m_geomTargetLine;
        GeomPolyLine m_geomPredictLine;
        CURVE_TYPE m_curveType = CURVE_TYPE.SIN;
        float m_fXScale = 512;
        float m_fYScale = 150;
        float m_fX = 0;
        float m_fInc = (float)(Math.PI * 2.0f / 360.0f);
        float m_fMax = (float)(Math.PI * 2.0f);
        List<DataPoint> m_rgPrevPoints = new List<DataPoint>();
        int m_nMaxPlots = 500;
        float m_fLastY = 0;
        bool m_bRenderImage = true;

        Random m_random = new Random();
        CurveState m_state = new CurveState();
        Log m_log;

        /// <summary>
        /// Defines the actions to perform.
        /// </summary>
        public enum ACTION
        {
            /// <summary>
            /// Move the cart left.
            /// </summary>
            MOVEUP,
            /// <summary>
            /// Move the cart right.
            /// </summary>
            MOVEDN
        }

        /// <summary>
        /// Defines the curve types.
        /// </summary>
        public enum CURVE_TYPE
        {
            /// <summary>
            /// Specifies to use a Sin curve as the target.
            /// </summary>
            SIN,
            /// <summary>
            /// Specifies to use a Cos curve as the target.
            /// </summary>
            COS,
            /// <summary>
            /// Specifies to use a Random curve as the target.
            /// </summary>
            RANDOM
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public CurveGym()
        {
            m_rgActionSpace = new Dictionary<string, int>();
            m_rgActionSpace.Add("MoveUp", 0);
            m_rgActionSpace.Add("MoveDn", 1);
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
            m_log = log;
            m_nMaxSteps = 0;

            int nCurveType = properties.GetPropertyAsInt("CurveType", -1);
            if (nCurveType != -1)
                m_curveType = (CURVE_TYPE)nCurveType;

            Reset(false);
        }


        /// <summary>
        /// Create a new copy of the gym.
        /// </summary>
        /// <param name="properties">Optionally, specifies the properties to initialize the new copy with.</param>
        /// <returns>The new Gym copy is returned.</returns>
        public IXMyCaffeGym Clone(PropertySet properties = null)
        {
            CurveGym gym = new CurveGym();

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
            get { return 1; }
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

        private void processAction(ACTION? a, double? dfOverride = null)
        {
            if (dfOverride.HasValue)
            {
                m_state.PredictedY = dfOverride.Value;
            }
            else
            {
                if (a.HasValue)
                {
                    switch (a)
                    {
                        case ACTION.MOVEUP:
                            m_state.PredictedY += 0.1;
                            break;

                        case ACTION.MOVEDN:
                            m_state.PredictedY -= 0.1;
                            break;
                    }
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
            rgData.Add(m_state.Y);
            rgData.Add(m_state.PredictedY);
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
        /// <returns>A tuple optionally containing a Bitmap and/or Simpledatum is returned.</returns>
        public Tuple<Bitmap, SimpleDatum> Render(bool bShowUi, int nWidth, int nHeight, double[] rgData, bool bGetAction)
        { 

            double dfX = rgData[0];
            double dfY = rgData[1];
            double dfPredictedY = rgData[2];
            int nSteps = (int)rgData[3];

            m_nSteps = nSteps;
            m_nMaxSteps = Math.Max(nSteps, m_nMaxSteps);

            SimpleDatum sdAction = null;
            Bitmap bmp = null;

            if (m_bRenderImage)
            {
                bmp = new Bitmap(nWidth, nHeight);
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    Rectangle rc = new Rectangle(0, 0, bmp.Width, bmp.Height);
                    g.FillRectangle(Brushes.White, rc);

                    float fScreenWidth = g.VisibleClipBounds.Width;
                    float fScreenHeight = g.VisibleClipBounds.Height;
                    float fWorldWidth = (float)m_fXScale;
                    float fWorldHeight = (float)m_fYScale * 2;
                    float fScale = fScreenHeight / fWorldHeight;

                    float fL = 0;
                    float fR = fWorldWidth;
                    float fT = fWorldHeight / 2;
                    float fB = -fWorldHeight / 2;

                    if (m_geomGraph == null)
                    {
                        m_geomGraph = new GeomGraph(fL, fR, fT, fB, Color.Azure, Color.SteelBlue);
                        m_geomGraph.SetLocation(0, fScale * (fWorldHeight / 2));
                    }
                    if (m_geomTargetLine == null)
                    {
                        m_geomTargetLine = new GeomPolyLine(fL, fR, fT, fB, Color.Blue, Color.Blue, m_nMaxPlots);
                        m_geomTargetLine.Polygon.Clear();
                        m_geomTargetLine.SetLocation(0, fScale * (fWorldHeight / 2));
                    }
                    m_geomTargetLine.Polygon.Add(new PointF((float)dfX, (float)(dfY * m_fYScale)));

                    if (m_geomPredictLine == null)
                    {
                        m_geomPredictLine = new GeomPolyLine(fL, fR, fT, fB, Color.Orange, Color.Orange, m_nMaxPlots);
                        m_geomPredictLine.Polygon.Clear();
                        m_geomPredictLine.SetLocation(0, fScale * (fWorldHeight / 2));
                    }
                    m_geomPredictLine.Polygon.Add(new PointF((float)dfX, (float)(dfPredictedY * m_fYScale)));

                    GeomView view = new GeomView();

                    view.RenderText(g, "X = " + dfX.ToString("N02"), 10, 24);
                    view.RenderText(g, "Y = " + dfY.ToString("N02"), 10, 36);
                    view.RenderText(g, "Predicted Y = " + dfPredictedY.ToString("N02"), 10, 48);
                    view.RenderText(g, "Curve Type = " + m_curveType.ToString(), 10, 60);
                    view.RenderSteps(g, m_nSteps, m_nMaxSteps);

                    // Render the objects.
                    view.AddObject(m_geomGraph);
                    view.AddObject(m_geomTargetLine);
                    view.AddObject(m_geomPredictLine);
                    view.Render(g);

                    if (bGetAction)
                        sdAction = getActionData((float)dfX, (float)dfY, (float)fWorldWidth, (float)fWorldHeight, bmp);

                    m_bmp = bmp;
                }
            }

            return new Tuple<Bitmap, SimpleDatum>(bmp, sdAction);
        }

        private SimpleDatum getActionData(float fX, float fY, float fWid, float fHt, Bitmap bmpSrc)
        {
            double dfX = (fWid * 0.85);
            double dfY = (bmpSrc.Height - fY) - (fHt * 0.75);

            RectangleF rc = new RectangleF((float)dfX, (float)dfY, fWid, fHt);
            Bitmap bmp = new Bitmap((int)fWid, (int)fHt);

            using (Graphics g = Graphics.FromImage(bmp))
            {
                RectangleF rc1 = new RectangleF(0, 0, (float)fWid, (float)fHt);
                g.FillRectangle(Brushes.Black, rc1);
                g.DrawImage(bmpSrc, rc1, rc, GraphicsUnit.Pixel);
            }

            return ImageData.GetImageDataD(bmp, 3, false, -1);
        }

        /// <summary>
        /// Reset the state of the gym.
        /// </summary>
        /// <param name="bGetLabel">Not used.</param>
        /// <param name="props">Optionally, specifies the property set to use.</param>
        /// <returns>A tuple containing state data, the reward, and the done state is returned.</returns>
        public Tuple<State, double, bool> Reset(bool bGetLabel, PropertySet props = null)
        {
            double dfX = 0;
            double dfY = 0;
            double dfPredictedY = 0;

            m_bRenderImage = true;
            m_rgPrevPoints.Clear();
            m_nSteps = 0;

            m_fLastY = 0;
            m_state = new CurveState(dfX, dfY, dfPredictedY);

            if (props != null)
            {
                bool bTraining = props.GetPropertyAsBool("Training", false);
                if (bTraining)
                    m_bRenderImage = false;

                m_fLastY = (float)props.GetPropertyAsDouble("TrainingStart", 0);
            }

            return new Tuple<State, double, bool>(m_state.Clone(), 1, false);
        }

        private double randomUniform(double dfMin, double dfMax)
        {
            double dfRange = dfMax - dfMin;
            return dfMin + (m_random.NextDouble() * dfRange);
        }

        private double calculateTarget(double dfX)
        {
            switch (m_curveType)
            {
                case CURVE_TYPE.SIN:
                    return Math.Sin(dfX);

                case CURVE_TYPE.COS:
                    return Math.Cos(dfX);

                case CURVE_TYPE.RANDOM:
                    float fCurve = m_fLastY + (float)(m_random.NextDouble() - 0.5) * (float)(m_random.NextDouble() * 0.20f);
                    if (fCurve > 1.0f)
                        fCurve = 1.0f;
                    if (fCurve < -1.0f)
                        fCurve = -1.0f;
                    m_fLastY = fCurve;
                    return fCurve;

                default:
                    throw new Exception(Name + " does not support the curve type '" + m_curveType.ToString() + "'.");
            }
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
            CurveState state = new CurveState(m_state);
            double dfReward = 0;
            double? dfOverride = null;

            m_bRenderImage = true;

            if (propExtra != null)
            {
                bool bTraining = propExtra.GetPropertyAsBool("Training", false);
                if (bTraining)
                    m_bRenderImage = false;

                double dfVal = propExtra.GetPropertyAsDouble("override_prediction", double.MaxValue);
                if (dfVal != double.MaxValue)   
                    dfOverride = dfVal;
            }

            processAction((ACTION)nAction, dfOverride);

            double dfX = state.X;
            double dfY = state.Y;
            double dfPredictedY = ((dfOverride.HasValue) ? dfOverride.Value : state.PredictedY);

            m_fX += m_fInc;
            if (m_fX > m_fMax)
                m_fX = 0;

            dfY = calculateTarget(m_fX);
            dfX += 1;

            float[] rgInput = new float[] { m_fX };
            float[] rgMask = new float[] { 1 };
            float fTarget = (float)dfY;
            float fTime = (float)(dfX / m_nMaxPlots);

            DataPoint pt = new DataPoint(rgInput, rgMask, fTarget,  (float)dfPredictedY, fTime);
            m_rgPrevPoints.Add(pt);

            if (m_rgPrevPoints.Count > m_nMaxPlots)
                m_rgPrevPoints.RemoveAt(0);

            CurveState stateOut = m_state;
            m_state = new CurveState(dfX, dfY, dfPredictedY, m_rgPrevPoints);

            dfReward = 1.0 - Math.Abs(dfPredictedY - dfY);
            if (dfReward < -1)
                dfReward = -1;

            m_nSteps++;
            m_nMaxSteps = Math.Max(m_nMaxSteps, m_nSteps);

            stateOut.Steps = m_nSteps;
            return new Tuple<State, double, bool>(stateOut.Clone(), dfReward, false);
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
            int nC = 1;

            if (dt == DATA_TYPE.DEFAULT)
                dt = DATA_TYPE.VALUES;

            if (dt == DATA_TYPE.BLOB)
            {
                nH = 156;
                nW = 156;
                nC = 3;
            }

            SourceDescriptor srcTrain = new SourceDescriptor((int)GYM_DS_ID.CURVE, Name + ".training", nW, nH, nC, false, false);
            SourceDescriptor srcTest = new SourceDescriptor((int)GYM_SRC_TEST_ID.CURVE, Name + ".testing", nW, nH, nC, false, false);
            DatasetDescriptor ds = new DatasetDescriptor((int)GYM_SRC_TRAIN_ID.CURVE, Name, null, null, srcTrain, srcTest, "CurveGym", "Curve Gym", null, GYM_TYPE.DYNAMIC);

            m_dt = dt;

            return ds;
        }
    }

    class GeomGraph : GeomPolygon /** @private */
    {
        public GeomGraph(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
        }

        public override void Render(Graphics g)
        {
            base.Render(g);
        }
    }

    class CurveState : State /** @private */
    {
        double m_dfX = 0;
        double m_dfY = 0;
        double m_dfPredictedY = 0;
        int m_nSteps = 0;

        public const double MAX_X = 2.4;
        public const double MAX_Y = 2.4;

        public CurveState(double dfX = 0, double dfY = 0, double dfPredictedY = 0, List<DataPoint> rgPoints = null)
        {
            m_dfX = dfX;
            m_dfY = dfY;
            m_dfPredictedY = dfPredictedY;

            if (rgPoints != null)
                m_rgPrevPoints = rgPoints;
        }

        public CurveState(CurveState s)
        {
            m_dfX = s.m_dfX;
            m_dfY = s.m_dfY;
            m_dfPredictedY = s.m_dfPredictedY;
            m_nSteps = s.m_nSteps;
            m_rgPrevPoints = new List<DataPoint>();

            if (s.m_rgPrevPoints != null)
                m_rgPrevPoints.AddRange(s.m_rgPrevPoints);
        }

        public int Steps
        {
            get { return m_nSteps; }
            set { m_nSteps = value; }
        }

        public double X
        {
            get { return m_dfX; }
            set { m_dfX = value; }
        }

        public double Y
        {
            get { return m_dfY; }
            set { m_dfY = value; }
        }

        public double PredictedY
        {
            get { return m_dfPredictedY; }
            set { m_dfPredictedY = value; }
        }

        public override State Clone()
        {
            return new CurveState(this);
        }

        public override SimpleDatum GetData(bool bNormalize, out int nDataLen)
        {
            nDataLen = 4;
            Valuemap data = new Valuemap(1, 4, 1);

            data.SetPixel(0, 0, getValue(m_dfX, -MAX_X, MAX_X, bNormalize));
            data.SetPixel(0, 1, getValue(m_dfY, -MAX_Y, MAX_Y, bNormalize));
            data.SetPixel(0, 2, getValue(m_dfPredictedY, -MAX_Y, MAX_Y, bNormalize));
            data.SetPixel(0, 3, m_nSteps);

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
