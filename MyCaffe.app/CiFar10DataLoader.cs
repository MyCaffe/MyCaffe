using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using System.Drawing;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.app
{
    public class CiFar10DataLoader
    {
        CiFar10DataParameters m_param;
        DatasetFactory m_factory = new DatasetFactory();

        public event EventHandler<ProgressArgs> OnProgress;
        public event EventHandler<ProgressArgs> OnError;

        public CiFar10DataLoader(CiFar10DataParameters param)
        {
            m_param = param;
        }

        public void LoadDatabase()
        {
            int nIdx = 0;
            int nTotal = 50000;

            reportProgress(nIdx, 0, "Loading database...");

            loadFile(m_param.DataBatchFile1, "CIFAR-10.training", nTotal, ref nIdx);
            loadFile(m_param.DataBatchFile2, "CIFAR-10.training", nTotal, ref nIdx);
            loadFile(m_param.DataBatchFile3, "CIFAR-10.training", nTotal, ref nIdx);
            loadFile(m_param.DataBatchFile4, "CIFAR-10.training", nTotal, ref nIdx);
            loadFile(m_param.DataBatchFile5, "CIFAR-10.training", nTotal, ref nIdx);

            nIdx = 0;
            nTotal = 10000;
            loadFile(m_param.TestBatchFile, "CIFAR-10.testing", nTotal, ref nIdx);

            DatasetFactory factory = new DatasetFactory();
            SourceDescriptor srcTrain = factory.LoadSource("CIFAR-10.training");
            SourceDescriptor srcTest = factory.LoadSource("CIFAR-10.testing");
            DatasetDescriptor ds = new DatasetDescriptor(0, "CIFAR-10", null, null, srcTrain, srcTest, "CIFAR-10", "CiFar-10 Dataset");
            factory.AddDataset(ds);
            factory.UpdateDatasetCounts(ds.ID);
        }

        private void loadFile(string strImagesFile, string strSourceName, int nTotal, ref int nIdx)
        {
            Stopwatch sw = new Stopwatch();
            int nStart = nIdx;

            reportProgress(nIdx, nTotal, " Source: " + strSourceName);
            reportProgress(nIdx, nTotal, "  loading " + strImagesFile + "...");

            FileStream fs = null;

            try
            {
                fs = new FileStream(strImagesFile, FileMode.Open, FileAccess.Read);
                using (BinaryReader br = new BinaryReader(fs))
                {
                    fs = null;

                    int nSrcId = m_factory.AddSource(strSourceName, 3, 32, 32, false, 0, true);

                    m_factory.Open(nSrcId);
                    if (nIdx == 0)
                        m_factory.DeleteSourceData();

                    sw.Start();

                    for (int i = 0; i < 10000; i++)
                    {
                        int nLabel = (int)br.ReadByte();
                        byte[] rgImgBytes = br.ReadBytes(3072);
                        Bitmap img = createImage(rgImgBytes);

                        Datum d = ImageData.GetImageData(img, 3, false, nLabel);

                        m_factory.PutRawImageCache(nIdx, d);
                        nIdx++;

                        if (sw.ElapsedMilliseconds > 1000)
                        {
                            reportProgress(nStart + i, nTotal, "loading " + strImagesFile + "   " + i.ToString("N0") + " of 10,000...");
                            sw.Restart();
                        }
                    }

                    m_factory.ClearImageCash(true);

                    if (nIdx == nTotal)
                        m_factory.UpdateSourceCounts();
                }
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }
        }

        private Bitmap createImage(byte[] rgImg)
        {
            int nRoffset = 0;
            int nGoffset = 1024;
            int nBoffset = 2048;
            int nX = 0;
            int nY = 0;

            Bitmap bmp = new Bitmap(32, 32);

            for (int i = 0; i < 1024; i++)
            {
                byte bR = rgImg[nRoffset + i];
                byte bG = rgImg[nGoffset + i];
                byte bB = rgImg[nBoffset + i];
                Color clr = Color.FromArgb(bR, bG, bB);

                bmp.SetPixel(nX, nY, clr);

                nX++;

                if (nX == 32)
                {
                    nY++;
                    nX = 0;
                }
            }

            return bmp;
        }

        private void reportProgress(int nIdx, int nTotal, string strMsg)
        {
            if (OnProgress != null)
                OnProgress(this, new ProgressArgs(new ProgressInfo(nIdx, nTotal, strMsg)));
        }

        private void reportError(int nIdx, int nTotal, Exception err)
        {
            if (OnError != null)
                OnError(this, new ProgressArgs(new ProgressInfo(nIdx, nTotal, "ERROR", err)));
        }
    }
}
