﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    class BucketAnalysis
    {
        BucketCollection m_colNeg;
        BucketCollection m_colPos;

        public BucketAnalysis(BucketCollection colNeg, BucketCollection colPos)
        {
            m_colNeg = colNeg;
            m_colPos = colPos;
        }

        public string SaveAnalysis(string strPath, string strName, string strType, string strPeriod, double dfPctFromMid)
        {
            if (string.IsNullOrEmpty(strPath))
                return null;

            if (!Directory.Exists(strPath))
                return null;

            string strFile = Path.Combine(strPath, strName + "." + strPeriod + ".statistics.png");

            Bitmap bmp = new Bitmap(1000, 1000);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);

                Pen pen = new Pen(Color.Gray, 2.0f);
                Font font = new Font("Century Gothic", 10);

                g.DrawLine(pen, 500, 0, 500, 1000);
                g.DrawLine(pen, 0, 900, 1000, 900);

                CurveCollection curveLeft = new CurveCollection();
                int nX = 500;

                for (int i = m_colNeg.Count - 1; i >= 0; i--)
                {
                    Bucket b = m_colNeg[i];
                    curveLeft.Add(m_colNeg[i], nX);
                    nX -= 10;
                }

                CurveCollection curveRight = new CurveCollection();
                nX = 500;

                for (int i = 0; i < m_colPos.Count; i++)
                {
                    Bucket b = m_colPos[i];
                    curveRight.Add(m_colPos[i], nX);
                    nX += 10;
                }

                CurveCollection curve = new CurveCollection();
                curve.Add(curveLeft, true);
                curve.Add(curveRight, false, 1);
                int nMaxCount = curve.Translate(0, 0, 1000, 900);

                curve.Render(g, 500, 900, 1000, 900, nMaxCount, strType, dfPctFromMid);
            }

            bmp.Save(strFile);

            return strFile;
        }
    }

    class CurveItem
    {
        PointF m_pt;
        Bucket m_bucket;
        int m_nX;

        public CurveItem(Bucket bucket, int nX)
        {
            m_bucket = bucket;
            m_nX = nX;
        }

        public Bucket Bucket
        {
            get { return m_bucket; }
        }

        public int X
        {
            get { return m_nX; }
        }

        public PointF Point
        {
            get { return m_pt; }
            set { m_pt = value; }
        }

        public override string ToString()
        {
            return "X=" + m_nX.ToString() + ", Count=" + m_bucket.Count.ToString() + " Range:" + m_bucket.Minimum.ToString() + "," + m_bucket.Maximum.ToString() + " Pt=" + m_pt.X.ToString() + "," + m_pt.Y.ToString();
        }
    }

    class CurveCollection
    {
        int m_nMidIdx = -1;
        List<CurveItem> m_rgCurves = new List<CurveItem>();

        public CurveCollection()
        {
        }

        private void renderYaxis(Graphics g, Font font, int nCount, int nMaxCount, int nX, int nY, int nWid, int nHt, bool bDrawTick, bool bDrawLabel)
        {
            SizeF sz;

            // Draw y labels
            int nStep = nMaxCount / nCount;
            int nStepCount = nStep;
            float fStep = nHt / (float)nCount;
            float fY = nY - fStep;

            for (int i = 1; i < nCount; i++)
            {
                if (bDrawTick)
                    g.DrawLine(Pens.Gray, nX - 5, fY, nX + 5, fY);

                if (bDrawLabel)
                {
                    string str = nStepCount.ToString("N0");
                    sz = g.MeasureString(str, font);
                    g.DrawString(str, font, Brushes.Black, nX + 5, fY - sz.Height / 2);
                }

                nStepCount += nStep;
                fY -= fStep;
            }
        }

        private void renderXaxisLeft(Graphics g, Font font, int nCount, int nMaxCount, int nX, int nY, int nWid, int nHt, bool bDrawTicks, bool bDrawLabel)
        {
            SizeF sz;

            // Draw left x labels
            int nStep = nMaxCount / nCount;
            int nStepCount = nStep;
            float fStep = nHt / (float)nCount;
            int nIdx = m_nMidIdx - 1;
            float fX = nX - fStep;

            while (fX > 0 && nIdx >= 0)
            {
                string strX = m_rgCurves[nIdx].Bucket.MidPoint.ToString("N4") + Environment.NewLine + "idx=" + nIdx.ToString();
                nIdx--;

                sz = g.MeasureString(strX, font);
                float fXtxt = fX - sz.Width / 2;
                if (fXtxt < 0)
                    break;

                if (bDrawTicks)
                    g.DrawLine(Pens.Gray, fX, nY - 5, fX, nY + 5);
                if (bDrawLabel && nIdx % 5 == 0)
                    g.DrawString(strX, font, Brushes.Black, fXtxt, nY + 5);
                fX -= fStep;
            }
        }

        private void renderXaxisRight(Graphics g, Font font, int nCount, int nMaxCount, int nX, int nY, int nWid, int nHt, bool bDrawTicks, bool bDrawLabel)
        {
            SizeF sz;

            // Draw right x labels
            int nStep = nMaxCount / nCount;
            int nStepCount = nStep;
            float fStep = nHt / (float)nCount;
            int nIdx = m_nMidIdx + 1;
            float fX = nX + fStep;

            while (fX < nWid && nIdx < m_rgCurves.Count)
            {
                string strX = m_rgCurves[nIdx].Bucket.MidPoint.ToString("N4") + Environment.NewLine + "idx=" + nIdx.ToString();
                nIdx++;

                sz = g.MeasureString(strX, font);
                float fXtxt = fX - sz.Width / 2;
                if (fXtxt + sz.Width > nWid)
                    break;

                if (bDrawTicks)
                    g.DrawLine(new Pen(Color.Gray, 1.0f), fX, nY - 5, fX, nY + 5);
                if (bDrawLabel && nIdx % 5 == 0)
                    g.DrawString(strX, font, Brushes.Black, fXtxt, nY + 5);
                fX += fStep;
            }
        }

        private void renderGrid(Graphics g, int nX, int nY, int nWid, int nHt, int nCount, int nMaxCount, int nNegative25PctIdx, int nPositive25PctIdx)
        {
            // Draw grid
            int nStep = nMaxCount / nCount;
            int nStepCount = nStep;
            float fStep = nHt / (float)nCount;
            float fY = nY - fStep;

            // Draw horizontal grid lines
            for (int i = 1; i < nCount; i++)
            {
                g.DrawLine(Pens.LightGray, 0, fY, nWid, fY);
                nStepCount += nStep;
                fY -= fStep;
            }

            // Draw vertical grid lines
            int nIdx = m_nMidIdx - 1;
            float fX = nX - fStep;

            while (fX > 0 && nIdx >= 0)
            {
                if (nIdx == nNegative25PctIdx)
                {
                    Pen pen = new Pen(Color.Red, 2.0f);
                    g.DrawLine(pen, fX, nY, fX, nY - nHt);
                    pen.Dispose();
                }

                g.DrawLine(Pens.LightGray, fX, nY, fX, nY - nHt);
                nIdx--;
                fX -= fStep;
            }

            nIdx = m_nMidIdx + 1;
            fX = nX + fStep;

            while (fX < nWid && nIdx < m_rgCurves.Count)
            {
                if (nIdx == nPositive25PctIdx)
                {
                    Pen pen = new Pen(Color.Red, 2.0f);
                    g.DrawLine(pen, fX, nY, fX, nY - nHt);
                    pen.Dispose();
                }

                g.DrawLine(Pens.LightGray, fX, nY, fX, nY - nHt);
                nIdx++;
                fX += fStep;
            }
        }

        public void Render(Graphics g, int nX, int nY, int nWid, int nHt, int nMaxCount, string strType, double dfPctFromMid)
        {
            Font font = new Font("Century Gothic", 8.0f);
            int nCount = 100;

            int nTotalPositive = m_rgCurves.Where(p => p.Bucket.MidPoint > 0).Sum(p => p.Bucket.Count);
            int nTotalNegative = m_rgCurves.Where(p => p.Bucket.MidPoint < 0).Sum(p => p.Bucket.Count);
            int nTotalPositive25Pct = (int)(nTotalPositive * dfPctFromMid);
            int nTotalNegative25Pct = (int)(nTotalNegative * dfPctFromMid);

            int nNegative25PctIdx = 0;
            int nTotal = 0;
            for (int i = m_nMidIdx; i >= 0; i--)
            {
                nTotal += m_rgCurves[i].Bucket.Count;
                if (nTotal >= nTotalNegative25Pct)
                {
                    nNegative25PctIdx = i;
                    break;
                }
            }

            int nPositive25PctIdx = 0;
            nTotal = 0;
            for (int i = m_nMidIdx; i < m_rgCurves.Count; i++)
            {
                nTotal += m_rgCurves[i].Bucket.Count;
                if (nTotal >= nTotalPositive25Pct)
                {
                    nPositive25PctIdx = i;
                    break;
                }
            }

            renderYaxis(g, font, nCount, nMaxCount, nX, nY, nWid, nHt, true, false);
            renderXaxisLeft(g, font, nCount, nMaxCount, nX, nY, nWid, nHt, true, false);
            renderXaxisRight(g, font, nCount, nMaxCount, nX, nY, nWid, nHt, true, false);
            renderGrid(g, nX, nY, nWid, nHt, nCount, nMaxCount, nNegative25PctIdx, nPositive25PctIdx);
            renderYaxis(g, font, nCount, nMaxCount, nX, nY, nWid, nHt, false, true);
            renderXaxisLeft(g, font, nCount, nMaxCount, nX, nY, nWid, nHt, false, true);
            renderXaxisRight(g, font, nCount, nMaxCount, nX, nY, nWid, nHt, false, true);

            // Draw the curve
            Pen pen = new Pen(Color.Blue, 2.0f);

            for (int i = 0; i < m_rgCurves.Count; i++)
            {
                List<PointF> rgPt = m_rgCurves.Select(p => p.Point).ToList();
                g.DrawPolygon(pen, rgPt.ToArray());
            }

            font.Dispose();
            font = new Font("Century Gothic", 12.0f);

            string strTitle = strType;
            SizeF sz = g.MeasureString(strTitle, font);
            g.DrawString(strTitle, font, Brushes.Black, nX + nWid / 2 - sz.Width / 2, nY + nHt + 10);

            // Cleanup
            pen.Dispose();
            font.Dispose();
        }

        public int Translate(int nX, int nY, int nWid, int nHt)
        {
            nHt -= 10;
            nY += 10;

            int nMaxCount = m_rgCurves.Max(p => p.Bucket.Count);
            nMaxCount = (int)((nMaxCount * 100.0) / 100.0) + 100;

            for (int i = 0; i < m_rgCurves.Count; i++)
            {
                int nX1 = nX + m_rgCurves[i].X;

                double dfPct = (double)m_rgCurves[i].Bucket.Count / (double)nMaxCount;
                float fHt = (float)(dfPct * nHt);

                m_rgCurves[i].Point = new PointF(nX1, nY + nHt - fHt);
            }

            return nMaxCount;
        }

        public void Add(CurveCollection curve, bool bReverse, int nOffset = 0)
        {
            if (bReverse)
            {
                for (int i = curve.Count - (nOffset + 1); i >= 0; i--)
                {
                    m_rgCurves.Add(new CurveItem(curve[i].Bucket, curve[i].X));
                }

                if (m_nMidIdx == -1)
                    m_nMidIdx = m_rgCurves.Count - 1;
            }
            else
            {
                for (int i = nOffset; i < curve.Count; i++)
                {
                    m_rgCurves.Add(new CurveItem(curve[i].Bucket, curve[i].X));
                }

                if (m_nMidIdx == -1)
                    m_nMidIdx = m_rgCurves.Count - 1;
            }
        }

        public void Add(Bucket bucket, int nX)
        {
            m_rgCurves.Add(new CurveItem(bucket, nX));
        }

        public int Count
        {
            get { return m_rgCurves.Count; }
        }

        public CurveItem this[int nIdx]
        {
            get { return m_rgCurves[nIdx]; }
        }
    }
}