using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace MyCaffe.gym
{
    /// <summary>
    /// The GoomObj is the base class for all other gometric objects used to draw Gym objects.
    /// </summary>
    abstract class GeomObj
    {
        /// <summary>
        /// Specifies the location of the object.
        /// </summary>
        protected PointF m_location = new PointF(0, 0);
        /// <summary>
        /// Specifies the points of the object.
        /// </summary>
        protected List<PointF> m_rgPoints = new List<PointF>();
        /// <summary>
        /// Specifies the fill color of the object.
        /// </summary>
        protected Color m_clrFill = Color.LightGray;
        /// <summary>
        /// Specifies the border color of the object.
        /// </summary>
        protected Color m_clrBorder = Color.Black;
        /// <summary>
        /// Specifies the rotation of the object.
        /// </summary>
        protected float m_fRotation = 0;
        /// <summary>
        /// Specifies that the colors have changed.
        /// </summary>
        protected bool m_bClrDirty = false;
        System.Drawing.Drawing2D.GraphicsState m_gstate = null;
        RectangleF m_rcBounds;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        public GeomObj(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
        {
            Set(fL, fR, fT, fB);
            m_clrFill = clrFill;
            m_clrBorder = clrBorder;
        }

        /// <summary>
        /// Set the points of the object.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        public void Set(float fL, float fR, float fT, float fB)
        {
            m_rgPoints.Clear();
            m_rgPoints.Add(new PointF(fL, fB));
            m_rgPoints.Add(new PointF(fL, fT));
            m_rgPoints.Add(new PointF(fR, fT));
            m_rgPoints.Add(new PointF(fR, fB));
            m_rgPoints.Add(new PointF(fL, fB));
            m_rcBounds = new RectangleF(fL, fT, (int)(fR - fL), (int)Math.Abs(fB - fT));
        }

        /// <summary>
        /// Called just before rendering the object.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to render.</param>
        protected void prerender(Graphics g)
        {
            m_gstate = g.Save();
            g.TranslateTransform(m_location.X, m_location.Y);
            g.RotateTransform(m_fRotation);
        }

        /// <summary>
        /// Called just after rendering the object.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to render.</param>
        protected void postrender(Graphics g)
        {
            if (m_gstate != null)
            {
                g.Restore(m_gstate);
                m_gstate = null;
            }
        }

        /// <summary>
        /// Returns the bounds of the object.
        /// </summary>
        public RectangleF Bounds
        {
            get { return m_rcBounds; }
        }

        /// <summary>
        /// Set the colors of the object.
        /// </summary>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        public void SetColors(Color clrFill, Color clrBorder)
        {
            m_clrFill = clrFill;
            m_clrBorder = clrBorder;
            m_bClrDirty = true;
        }

        /// <summary>
        /// Returns the width of the points.
        /// </summary>
        /// <param name="rg">Optionally, specifies the override points.</param>
        /// <returns>The width is returned.</returns>
        public float Width(PointF[] rg)
        {
            if (rg == null)
                rg = m_rgPoints.ToArray();

            return rg[2].X - rg[0].X;
        }

        /// <summary>
        /// Returns the height of the points.
        /// </summary>
        /// <param name="rg">Optionally, specifies the override points.</param>
        /// <returns>The height is returned.</returns>
        public float Height(PointF[] rg)
        {
            if (rg == null)
                rg = m_rgPoints.ToArray();

            return rg[0].Y - rg[1].Y;
        }

        /// <summary>
        /// Returns the left bottom.
        /// </summary>
        /// <param name="rg">Optionally, specifies the override points.</param>
        /// <returns>The left-bottom is returned.</returns>
        public PointF LeftBottom(PointF[] rg = null)
        {
            if (rg == null)
                rg = m_rgPoints.ToArray();

            return rg[0];
        }

        /// <summary>
        /// Returns the left top.
        /// </summary>
        /// <param name="rg">Optionally, specifies the override points.</param>
        /// <returns>The left-top is returned.</returns>
        public PointF LeftTop(PointF[] rg = null)
        {
            if (rg == null)
                rg = m_rgPoints.ToArray();

            return rg[1];
        }

        /// <summary>
        /// Returns the right top.
        /// </summary>
        /// <param name="rg">Optionally, specifies the override points.</param>
        /// <returns>The right-top is returned.</returns>
        public PointF RightTop(PointF[] rg = null)
        {
            if (rg == null)
                rg = m_rgPoints.ToArray();

            return rg[2];
        }

        /// <summary>
        /// Returns the right bottom.
        /// </summary>
        /// <param name="rg">Optionally, specifies the override points.</param>
        /// <returns>The right-bottom is returned.</returns>
        public PointF RightBottom(PointF[] rg = null)
        {
            if (rg == null)
                rg = m_rgPoints.ToArray();

            return rg[3];
        }

        /// <summary>
        /// Returns the location of the object.
        /// </summary>
        public PointF Location
        {
            get { return m_location; }
        }

        /// <summary>
        /// Returns the rotation of the object.
        /// </summary>
        public float Rotation
        {
            get { return m_fRotation; }
        }

        /// <summary>
        /// Sets the object location.
        /// </summary>
        /// <param name="fX">Specifies the location x coordinate.</param>
        /// <param name="fY">Specifies the location y coordinate.</param>
        public virtual void SetLocation(float fX, float fY)
        {
            m_location = new PointF(fX, fY);
        }

        /// <summary>
        /// Sets the rotation of the object.
        /// </summary>
        /// <param name="fR">Specifies the rotation.</param>
        public virtual void SetRotation(float fR)
        {
            m_fRotation = fR;
        }

        /// <summary>
        /// Returns the bounds as a Polygon.
        /// </summary>
        public List<PointF> Polygon
        {
            get { return m_rgPoints; }
        }

        /// <summary>
        /// Returns the fill color.
        /// </summary>
        public Color FillColor
        {
            get { return m_clrFill; }
        }

        /// <summary>
        /// Returns the border color.
        /// </summary>
        public Color BorderColor
        {
            get { return m_clrBorder; }
        }

        /// <summary>
        /// Override used to render the object.
        /// </summary>
        /// <param name="g"></param>
        public abstract void Render(Graphics g);
    }

    /// <summary>
    /// The GeomLine object is used to render a line.
    /// </summary>
    class GeomLine : GeomObj
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        public GeomLine(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
        }

        /// <summary>
        /// Renders the line on the Graphics specified.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public override void Render(Graphics g)
        {
            prerender(g);
            PointF[] rg = m_rgPoints.ToArray();
            Pen p = new Pen(m_clrBorder, 1.0f);
            g.DrawLine(p, LeftBottom(rg), RightBottom(rg));
            p.Dispose();
            postrender(g);
        }
    }

    /// <summary>
    /// The GeomEllipse object is used to render an ellipse.
    /// </summary>
    class GeomEllipse : GeomObj
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        public GeomEllipse(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
        }

        /// <summary>
        /// Renders the ellipse on the Graphics specified.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public override void Render(Graphics g)
        {
            prerender(g);
            PointF[] rg = m_rgPoints.ToArray();
            RectangleF rc = new RectangleF(LeftTop(rg).X, LeftTop(rg).Y, Width(rg), Height(rg));
            Brush br = new SolidBrush(m_clrFill);
            Pen p = new Pen(m_clrBorder, 1.0f);
            g.FillEllipse(br, rc);
            g.DrawEllipse(p, rc);
            p.Dispose();
            br.Dispose();
            postrender(g);
        }
    }

    /// <summary>
    /// The GeomEllipse object is used to render an rectangle.
    /// </summary>
    class GeomRectangle : GeomObj, IDisposable
    {
        Font m_font = null;
        string m_strName = null;
        ColorMapper m_clrMap = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        /// <param name="clrMap">Optionally, specifies a color-map used to color the rectangle based on the x position of the object.</param>
        public GeomRectangle(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder, ColorMapper clrMap = null)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
            m_clrMap = clrMap;
        }

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            if (m_font != null)
            {
                m_font.Dispose();
                m_font = null;
            }
        }

        /// <summary>
        /// Renders the rectangle on the Graphics specified.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public override void Render(Graphics g)
        {
            prerender(g);

            PointF[] rg = m_rgPoints.ToArray();

            float fX1 = LeftTop(rg).X;
            float fY1 = LeftTop(rg).Y;
            float fW1 = Width(rg);
            float fH1 = Math.Abs(Height(rg));

            RectangleF rc = new RectangleF(fX1, fY1, fW1, fH1);
            Brush br = new SolidBrush(m_clrFill);
            Pen p = new Pen(m_clrBorder, 1.0f);
            g.FillRectangle(br, rc);

            if (m_clrMap != null)
            {
                float fX = 0;
                float fWid = rc.Width / 20;

                for (int i = 0; i < 20; i++)
                {
                    RectangleF rc2 = new RectangleF(fX, rc.Y, fWid, rc.Height);
                    Color clr = m_clrMap.GetColor(fX);
                    Brush br1 = new SolidBrush(clr);
                    g.FillRectangle(br1, rc2);
                    br1.Dispose();
                    fX += rc2.Width;
                }
            }

            g.DrawRectangle(p, rc.X, rc.Y, rc.Width, rc.Height);
            p.Dispose();
            br.Dispose();

            if (!string.IsNullOrEmpty(m_strName))
            {
                if (m_font == null)
                    m_font = new Font("Calibri Light", 8.0f);

                Color clr = Color.FromArgb(255, m_clrFill);
                Brush br1 = new SolidBrush(clr);

                SizeF sz = g.MeasureString(m_strName, m_font);
                float fDy = 200;
                g.MultiplyTransform(new System.Drawing.Drawing2D.Matrix(1, 0, 0, -1, 0, fDy));
                g.DrawString(m_strName, m_font, br1, new PointF(rc.X + (rc.Width - sz.Width) / 2, rc.Y - sz.Height));

                br1.Dispose();
            }

            postrender(g);
        }

        /// <summary>
        /// Synchronize the position of the rectangle to cover the area of the poly line.
        /// </summary>
        /// <param name="line">Specifies the poly line.</param>
        /// <param name="nOffsetFromEnd">Specifies the offset from the line end where the rectangle is to start.</param>
        /// <param name="nWidth">Specifies the prediction box width.</param>
        /// <param name="strName">Optionally, specifies a name for the box.</param>
        public void SyncLocation(GeomPolyLine line, int nOffsetFromEnd, int nWidth, string strName = null)
        {
            if (line.Polygon.Count == 0)
                return;

            RectangleF rc = line.Bounds;

            float fL = line.Polygon[line.Polygon.Count - 1].X + nOffsetFromEnd;
            float fR = fL + nWidth;
            float fT = rc.Y - 44;
            float fB = fT + rc.Height;

            Set(fL, fR, fT, fB);

            m_location.X = line.Location.X;
            m_strName = strName;
        }
    }

    /// <summary>
    /// The GeomPolygon object is used to render an polygon.
    /// </summary>
    class GeomPolygon : GeomObj
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        public GeomPolygon(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
        }

        /// <summary>
        /// Renders the rectangle on the Graphics specified.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public override void Render(Graphics g)
        {
            prerender(g);

            if (m_rgPoints.Count >= 2)
            {
                PointF[] rg = m_rgPoints.ToArray();
                Brush br = new SolidBrush(m_clrFill);
                Pen p = new Pen(m_clrBorder, 1.0f);
                g.FillPolygon(br, rg);
                g.DrawPolygon(p, rg);
                p.Dispose();
                br.Dispose();
            }

            postrender(g);
        }
    }

    /// <summary>
    /// The GeomPolyLine object is used to render an spline.
    /// </summary>
    class GeomPolyLine : GeomObj
    {
        protected int m_nMaxPlots = int.MaxValue;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        /// <param name="nMaxPlots">Specifies the maximum number of plots.</param>
        public GeomPolyLine(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder, int nMaxPlots = int.MaxValue)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
            m_nMaxPlots = nMaxPlots;
        }

        private void clipAndShift()
        {
            if (m_rgPoints.Count > m_nMaxPlots)
            {
                float fShift = m_rgPoints[m_rgPoints.Count - 1].X - m_rgPoints[m_rgPoints.Count - 2].X;
                int nDiff = m_rgPoints.Count - m_nMaxPlots;
                m_rgPoints.RemoveRange(0, nDiff);
                m_location.X -= (fShift * nDiff);
            }
        }

        /// <summary>
        /// Renders the rectangle on the Graphics specified.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public override void Render(Graphics g)
        {
            prerender(g);

            if (m_rgPoints.Count >= 2)
            {
                clipAndShift();

                PointF[] rg = m_rgPoints.ToArray();
                Pen p = new Pen(m_clrBorder, 1.0f);
                g.DrawLines(p, rg);
                p.Dispose();
            }

            postrender(g);
        }
    }

    /// <summary>
    /// The GeomPolyLinesSet contains a set of lines.
    /// </summary>
    class GeomPolyLineSet : GeomPolyLine
    {
        int m_nMaxPlotsOffset = 0;
        List<PointF[]> m_rgrgPoints = new List<PointF[]>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fL">Specifies the left position.</param>
        /// <param name="fR">Specifies the right position.</param>
        /// <param name="fT">Specifies the top position.</param>
        /// <param name="fB">Specifies the bottom position.</param>
        /// <param name="clrFill">Specifies the fill color.</param>
        /// <param name="clrBorder">Specifies the border color.</param>
        /// <param name="nMaxPlots">Specifies the maximum number of plots.</param>
        /// <param name="nMaxPlotsOffset">Optionally, specifies an offset to the maximum number of plots (default = 0).</param>
        public GeomPolyLineSet(float fL, float fR, float fT, float fB, Color clrFill, Color clrBorder, int nMaxPlots = int.MaxValue, int nMaxPlotsOffset = 0)
            : base(fL, fR, fT, fB, clrFill, clrBorder)
        {
            m_nMaxPlotsOffset = nMaxPlotsOffset;
            m_nMaxPlots = nMaxPlots;
        }

        /// <summary>
        /// Add a new set of future predictions by converting the predictions to a set of points.
        /// </summary>
        /// <param name="fX"></param>
        /// <param name="predictions"></param>
        /// <param name="fYScale"></param>
        public void Add(float fX, FuturePredictions predictions, float fYScale)
        {
            List<PointF> rg = new List<PointF>();

            for (int i=0; i<predictions.Predictions.Count; i++)
            {
                float fY = predictions.Predictions[i] * fYScale;
                rg.Add(new PointF(fX, fY));
                fX += 1;
            }
            
            m_rgrgPoints.Add(rg.ToArray());
        }

        /// <summary>
        /// Returns the set of poly lines..
        /// </summary>
        public List<PointF[]> PolyLines
        {
            get { return m_rgrgPoints; }
        }

        private void clipAndShift()
        {
            if (m_rgrgPoints.Count > (m_nMaxPlots + m_nMaxPlotsOffset))
            {
                PointF[] rg1 = m_rgrgPoints[m_rgrgPoints.Count - 1];
                PointF[] rg2 = m_rgrgPoints[m_rgrgPoints.Count - 2];

                float fShift = rg1[rg1.Length-1].X - rg2[rg2.Length-1].X;
                int nDiff = m_rgrgPoints.Count - (m_nMaxPlots + m_nMaxPlotsOffset);
                m_rgrgPoints.RemoveRange(0, nDiff);
                m_location.X -= (fShift * nDiff);
            }
        }

        /// <summary>
        /// Renders the rectangle on the Graphics specified.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public override void Render(Graphics g)
        {
            prerender(g);

            if (m_rgrgPoints.Count >= 2)
            {
                clipAndShift();

                int nStart = 0;
                int nCount = m_rgrgPoints.Count;

                if (nCount > 50)
                {
                    nStart = m_rgrgPoints.Count - 50;
                    nCount = 50;
                }

                int nAlpha = 255 - (nCount * 4);
                if (nAlpha > 16)
                    nAlpha = 16;

                for (int i=nStart; i<m_rgrgPoints.Count; i++)
                {
                    if (i == m_rgrgPoints.Count - 1)
                        nAlpha = 255;

                    Pen p = new Pen(Color.FromArgb(nAlpha, m_clrBorder), 1.0f);
                    g.DrawLines(p, m_rgrgPoints[i]);
                    p.Dispose();
                    nAlpha += 1;
                }
            }

            postrender(g);
        }
    }

    /// <summary>
    /// The GeomView manages and renders a collection of Geometric objects.
    /// </summary>
    class GeomView
    {
        List<GeomObj> m_rgObj = new List<GeomObj>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public GeomView()
        {
        }

        /// <summary>
        /// Add a new geometric object to the view.
        /// </summary>
        /// <param name="obj">Specifies the object to add.</param>
        public void AddObject(GeomObj obj)
        {
            m_rgObj.Add(obj);
        }

        /// <summary>
        /// Render text at a location.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        /// <param name="str">Specifies the text to draw.</param>
        /// <param name="fX">Specifies the left most x-coordinate where the text is drawn.</param>
        /// <param name="fY">Specifies the top most y-coordinate where the text is drawn.</param>
        /// <param name="br">Optionally, specifies the brush to use (default = Black)</param>
        public void RenderText(Graphics g, string str, float fX, float fY, Brush br = null)
        {
            if (br == null)
                br = Brushes.Black;

            Font font = new Font("Century Gothic", 9.0f);
            g.DrawString(str, font, br, new PointF(fX, fY));
            font.Dispose();
        }

        /// <summary>
        /// Renders the Gym step information.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        /// <param name="nSteps">Specifies the current steps.</param>
        /// <param name="nMax">Specifies the maximum number of steps.</param>
        public void RenderSteps(Graphics g, int nSteps, int nMax)
        {
            RectangleF rc = g.VisibleClipBounds;
            Font fontStep = new Font("Century Gothic", 7.0f);
            string strStep = nSteps.ToString();
            string strMax = nMax.ToString();
            string strSteps = "Steps";
            SizeF szStep = g.MeasureString(strStep, fontStep);
            SizeF szSteps = g.MeasureString(strSteps, fontStep);
            SizeF szMax = g.MeasureString(strMax, fontStep);

            float fX = 10;
            float fY = rc.Bottom - (szSteps.Height * 2);

            g.DrawString(strSteps, fontStep, Brushes.Black, new PointF(fX, fY));
            fX += szSteps.Width;

            float fMaxWid = Math.Max(nMax, nSteps);
            float fStepWid = nSteps;

            if (fMaxWid + fX + 20 > rc.Width)
            {
                fMaxWid = rc.Width - (fX + 20);
                fStepWid = ((float)nSteps / (float)Math.Max(nMax, nSteps)) * fMaxWid;
            }

            Rectangle rcMax = new Rectangle((int)fX, (int)fY, (int)fMaxWid, (int)(szStep.Height));
            Rectangle rcStep = new Rectangle((int)fX, (int)fY, (int)fStepWid, (int)(szStep.Height));
            Brush br = (nSteps < nMax) ? Brushes.Orange : Brushes.Lime;
            Pen pen = (nSteps < nMax) ? Pens.Brown : Pens.DarkGreen;

            g.FillRectangle(br, rcStep);
            g.DrawRectangle(pen, rcMax);

            fX = rcStep.Right - szStep.Width / 2;
            fY = rcStep.Bottom;

            g.DrawString(strStep, fontStep, Brushes.Brown, new PointF(fX, fY));

            fX = rcMax.Right - szMax.Width / 2;
            fY = rcMax.Bottom;

            g.DrawString(strMax, fontStep, Brushes.DarkGreen, new PointF(fX, fY));

            fontStep.Dispose();
        }

        /// <summary>
        /// Renders the view.
        /// </summary>
        /// <param name="g">Specifies the Graphics used to draw.</param>
        public void Render(Graphics g)
        {
            System.Drawing.Drawing2D.GraphicsState gstate = g.Save();

            g.TranslateTransform(0, -g.VisibleClipBounds.Height);
            g.ScaleTransform(1, -1, System.Drawing.Drawing2D.MatrixOrder.Append);

            g.DrawRectangle(Pens.SteelBlue, 1, 1, 2, 2);
            g.DrawLine(Pens.SteelBlue, 1, 3, 1, 4);
            g.DrawLine(Pens.SteelBlue, 3, 3, 4, 4);
            g.DrawLine(Pens.SteelBlue, 3, 1, 4, 1);

            foreach (GeomObj obj in m_rgObj)
            {
                obj.Render(g);
            }

            g.Restore(gstate);
        }
    }
}
