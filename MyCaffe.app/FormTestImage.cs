using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyCaffe.app
{
    public partial class FormTestImage : Form
    {
        int m_nGridXRes = 28;
        int m_nGridYRes = 28;
        double m_dfResX = 0;
        double m_dfResY = 0;
        Point m_ptLast = new Point();
        Bitmap m_bmp;
        List<Point> m_rgPoints = new List<Point>();
        List<Point> m_rgRawPoints = new List<Point>();

        public FormTestImage()
        {
            InitializeComponent();
        }

        public Bitmap Image
        {
            get { return m_bmp; }
        }

        private void FormTestImage_Load(object sender, EventArgs e)
        {
            m_bmp = new Bitmap(ClientRectangle.Width, ClientRectangle.Height);
            m_dfResX = m_bmp.Width / (double)m_nGridXRes;
            m_dfResY = m_bmp.Height / (double)m_nGridYRes;
            updateImage();
        }

        private void FormTestImage_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.DrawImage(m_bmp, 0, 0);   
        }

        private void FormTestImage_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            if (m_ptLast.IsEmpty)
            {
                m_ptLast = e.Location;
                return;
            }

            using (Graphics g = Graphics.FromImage(m_bmp))
            {
                Pen p = new Pen(Brushes.White, 10.0f);
                g.DrawLine(p, m_ptLast, e.Location);
                m_ptLast = e.Location;
                p.Dispose();
            }

            Invalidate();
        }

        private void FormTestImage_MouseUp(object sender, MouseEventArgs e)
        {
            m_ptLast = new Point();
            m_rgPoints.Clear();

            for (int y = 0; y < m_nGridYRes; y++)
            {
                for (int x = 0; x < m_nGridXRes; x++)
                {
                    Point pt = new Point(x, y);
                    int nHash = pt.GetHashCode();

                        bool bContainsBlack = false;

                    for (int j = 0; j < (int)m_dfResY; j++)
                    {
                        for (int i = 0; i < (int)m_dfResX; i++)
                        {
                            int nX = (int)(x * m_dfResX) + i;
                            int nY = (int)(y * m_dfResY) + j;
                            Color clr = m_bmp.GetPixel(nX, nY);

                            if (clr.R == 255 && clr.G == 255 && clr.B == 255)
                            {
                                m_rgPoints.Add(pt);
                                bContainsBlack = true;
                                break;
                            }
                        }

                        if (bContainsBlack)
                            break;
                    }
                }
            }

            updateImage();
        }

        private void FormTestImage_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                m_rgPoints.Clear();
                updateImage();
            }
        }

        private void updateImage()
        {
            using (Graphics g = Graphics.FromImage(m_bmp))
            {
                g.FillRectangle(Brushes.Black, 0, 0, m_bmp.Width, m_bmp.Height);

                foreach (Point pt in m_rgPoints)
                {
                    Rectangle rc = new Rectangle((int)(pt.X * m_dfResX), (int)(pt.Y * m_dfResY), (int)m_dfResX, (int)m_dfResY);
                    g.FillRectangle(Brushes.White, rc);
                }

                drawGrid(g, m_bmp.Width, m_bmp.Height);
            }

            Invalidate();
        }

        private void drawGrid(Graphics g, int nW, int nH)
        {
            Pen p = new Pen(Brushes.LightGray, 1.0f);
            p.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;

            for (int x = 0; x < m_nGridXRes; x++)
            {
                g.DrawLine(p, (int)(x * m_dfResX), 0, (int)(x * m_dfResX), nH);
            }

            for (int y = 0; y < m_nGridYRes; y++)
            {
                g.DrawLine(p, 0, (int)(y * m_dfResY), nW, (int)(y * m_dfResY));
            }
        }
    }
}
