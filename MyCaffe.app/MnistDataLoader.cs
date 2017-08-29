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
using MyCaffe.basecode.descriptors;

namespace MyCaffe.app
{
    public class MnistDataLoader
    {
        MnistDataParameters m_param;
        DatasetFactory m_factory = new DatasetFactory();

        public event EventHandler<ProgressArgs> OnProgress;
        public event EventHandler<ProgressArgs> OnError;
        public event EventHandler OnCompleted;

        public MnistDataLoader(MnistDataParameters param)
        {
            m_param = param;
        }

        public void LoadDatabase()
        {
            int nIdx = 0;
            int nTotal = 0;

            reportProgress(nIdx, nTotal, "Unpacking files...");
            string strTrainImagesBin = expandFile(m_param.TrainImagesFile);
            string strTrainLabelsBin = expandFile(m_param.TrainLabelsFile);
            string strTestImagesBin = expandFile(m_param.TestImagesFile);
            string strTestLabelsBin = expandFile(m_param.TestLabelsFile);

            reportProgress(nIdx, nTotal, "Loading database...");

            loadFile(strTrainImagesBin, strTrainLabelsBin, "MNIST.training");
            loadFile(strTestImagesBin, strTestLabelsBin, "MNIST.testing");

            DatasetFactory factory = new DatasetFactory();
            SourceDescriptor srcTrain = factory.LoadSource("MNIST.training");
            SourceDescriptor srcTest = factory.LoadSource("MNIST.testing");
            DatasetDescriptor ds = new DatasetDescriptor(0, "MNIST", null, null, srcTrain, srcTest, "MNIST", "MNIST Character Dataset");
            factory.AddDataset(ds);
            factory.UpdateDatasetCounts(ds.ID);

            if (OnCompleted != null)
                OnCompleted(this, new EventArgs());
        }

        private void loadFile(string strImagesFile, string strLabelsFile, string strSourceName)
        {
            Stopwatch sw = new Stopwatch();

            reportProgress(0, 0, " Source: " + strSourceName);
            reportProgress(0, 0, "  loading " + strImagesFile + "...");

            BinaryFile image_file = new app.BinaryFile(strImagesFile);
            BinaryFile label_file = new app.BinaryFile(strLabelsFile);

            try
            {
                // Verify the files
                uint magicImg = image_file.ReadUInt32();
                uint magicLbl = label_file.ReadUInt32();

                if (magicImg != 2051)
                    throw new Exception("Incorrect image file magic.");

                if (magicLbl != 2049)
                    throw new Exception("Incorrect label file magic.");

                uint num_items = image_file.ReadUInt32();
                uint num_labels = label_file.ReadUInt32();

                if (num_items != num_labels)
                    throw new Exception("The number of items must be equal to the number of labels!");


                // Add the data source to the database.
                uint rows = image_file.ReadUInt32();
                uint cols = image_file.ReadUInt32();
                int nChannels = 1;  // black and white

                int nSrcId = m_factory.AddSource(strSourceName, nChannels, (int)cols, (int)rows, false, 0, true);

                m_factory.Open(nSrcId);
                m_factory.DeleteSourceData();

                // Storing to database;
                byte[] rgLabel;
                byte[] rgPixels;

                Datum datum = new Datum(false, nChannels, (int)cols, (int)rows, -1, DateTime.MinValue, null, null, 0, false, -1);

                reportProgress(0, (int)num_items, "  loading a total of " + num_items.ToString() + " items.");
                reportProgress(0, (int)num_items, "   (with rows: " + rows.ToString() + ", cols: " + cols.ToString() + ")");

                sw.Start();

                for (int i = 0; i < num_items; i++)
                {
                    rgPixels = image_file.ReadBytes((int)(rows * cols));
                    rgLabel = label_file.ReadBytes(1);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        reportProgress(i, (int)num_items, " loading data...");
                        sw.Restart();
                    }

                    datum.SetData(rgPixels.ToList(), (int)rgLabel[0]);
                    m_factory.PutRawImageCache(i, datum);
                }

                m_factory.ClearImageCash(true);
                m_factory.UpdateSourceCounts();

                reportProgress((int)num_items, (int)num_items, " loading completed.");
            }
            finally
            {
                image_file.Dispose();
                label_file.Dispose();
            }
        }

        private string expandFile(string strFile)
        {
            FileInfo fi = new FileInfo(strFile);
            string strNewFile = fi.DirectoryName;
            int nPos = fi.Name.LastIndexOf('.');

            if (nPos >= 0)
                strNewFile += "\\" + fi.Name.Substring(0, nPos) + ".bin";
            else
                strNewFile += "\\" + fi.Name + ".bin";

            if (!File.Exists(strNewFile))
            {
                using (FileStream fs = fi.OpenRead())
                {
                    using (FileStream fsBin = File.Create(strNewFile))
                    {
                        using (GZipStream decompStrm = new GZipStream(fs, CompressionMode.Decompress))
                        {
                            decompStrm.CopyTo(fsBin);
                        }
                    }
                }
            }

            return strNewFile;
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

    public class ProgressArgs : EventArgs
    {
        ProgressInfo m_pi;

        public ProgressArgs(ProgressInfo pi)
        {
            m_pi = pi;
        }

        public ProgressInfo Progress
        {
            get { return m_pi; }
        }
    }

    class BinaryFile : IDisposable
    {
        FileStream m_file;
        BinaryReader m_reader;

        public BinaryFile(string strFile)
        {
            m_file = File.Open(strFile, FileMode.Open, FileAccess.Read, FileShare.Read);
            m_reader = new BinaryReader(m_file);
        }

        public void Dispose()
        {
            m_reader.Close();
        }

        public BinaryReader Reader
        {
            get { return m_reader; }
        }

        public UInt32 ReadUInt32()
        {
            UInt32 nVal = m_reader.ReadUInt32();

            return swap_endian(nVal);
        }

        public byte[] ReadBytes(int nCount)
        {
            return m_reader.ReadBytes(nCount);
        }

        private UInt32 swap_endian(UInt32 nVal)
        {
            nVal = ((nVal << 8) & 0xFF00FF00) | ((nVal >> 8) & 0x00FF00FF);
            return (nVal << 16) | (nVal >> 16);
        }
    }
}
