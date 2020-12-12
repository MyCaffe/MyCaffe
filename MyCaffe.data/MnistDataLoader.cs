// Copyright (c) 2018-2020 SignalPop LLC and contributors. All rights reserved.
// License: Apache 2.0
// License: https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE
// Original Source: https://github.com/MyCaffe/MyCaffe/blob/master/MyCaffe.data/MnistDataLoader.cs
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using System.Threading;
using System.Drawing;

namespace MyCaffe.data
{
    /// <summary>
    /// The MnistDataLoader is used to create the MNIST dataset and load it into the database managed by the MyCaffe Image Database.
    /// </summary>
    /// <remarks>
    /// @see [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
    /// </remarks>
    public class MnistDataLoader
    {
        MnistDataLoaderLite m_extractor;
        MnistDataParameters m_param;
        Log m_log;
        CancelEvent m_evtCancel;

        /// <summary>
        /// The OnProgress event fires during the creation process to show the progress.
        /// </summary>
        public event EventHandler<ProgressArgs> OnProgress;
        /// <summary>
        /// The OnError event fires when an error occurs.
        /// </summary>
        public event EventHandler<ProgressArgs> OnError;
        /// <summary>
        /// The OnComplete event fires once the dataset creation has completed.
        /// </summary>
        public event EventHandler OnCompleted;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="param">Specifies the creation parameters.</param>
        /// <param name="log">Specifies the output log used to show status updates.</param>
        /// <param name="evtCancel">Specifies the cancel event used to abort the creation process.</param>
        public MnistDataLoader(MnistDataParameters param, Log log, CancelEvent evtCancel)
        {
            m_extractor = new MnistDataLoaderLite(Path.GetDirectoryName(param.TrainImagesFile));
            m_extractor.OnProgress += m_extractor_OnProgress;
            m_extractor.OnError += m_extractor_OnError;

            m_param = param;
            m_log = log;
            m_evtCancel = evtCancel;
            m_evtCancel.Reset();
        }

        private void m_extractor_OnError(object sender, ProgressArgs e)
        {
            if (OnError != null)
                OnError(sender, e);
        }

        private void m_extractor_OnProgress(object sender, ProgressArgs e)
        {
            if (OnProgress != null)
                OnProgress(sender, e);
        }

        private string dataset_name
        {
            get { return "MNIST"; }
        }

        /// <summary>
        /// Create the dataset and load it into the database.
        /// </summary>
        /// <param name="nCreatorID">Specifies the creator ID.</param>
        /// <returns>On successful creation, <i>true</i> is returned, otherwise <i>false</i> is returned on abort.</returns>
        public bool LoadDatabase(int nCreatorID = 0)
        {
            int nIdx = 0;
            int nTotal = 0;

            try
            {
                List<Tuple<byte[], int>> rgTrainImg;
                List<Tuple<byte[], int>> rgTestImg;

                m_extractor.ExtractImages(out rgTrainImg, out rgTestImg);

                reportProgress(nIdx, nTotal, "Loading " + dataset_name + " database...");

                DatasetFactory factory = null;
                string strExportFolder = null;

                if (m_param.ExportToFile)
                {
                    strExportFolder = m_param.ExportPath.TrimEnd('\\') + "\\";
                    if (!Directory.Exists(strExportFolder))
                        Directory.CreateDirectory(strExportFolder);
                }

                string strTrainSrc = "training";
                if (!m_param.ExportToFile)
                {
                    factory = new DatasetFactory();

                    strTrainSrc = dataset_name + "." + strTrainSrc;
                    int nSrcId = factory.GetSourceID(strTrainSrc);
                    if (nSrcId != 0)
                        factory.DeleteSourceData(nSrcId);
                }

                if (!loadFile(factory, rgTrainImg, m_extractor.Channels, m_extractor.Height, m_extractor.Width, strTrainSrc, strExportFolder))
                    return false;

                string strTestSrc = "testing";
                if (!m_param.ExportToFile)
                {
                    strTestSrc = dataset_name + "." + strTestSrc;
                    int nSrcId = factory.GetSourceID(strTestSrc);
                    if (nSrcId != 0)
                        factory.DeleteSourceData(nSrcId);
                }

                if (!loadFile(factory, rgTestImg, m_extractor.Channels, m_extractor.Height, m_extractor.Width, strTestSrc, strExportFolder))
                    return false;

                if (!m_param.ExportToFile)
                {
                    SourceDescriptor srcTrain = factory.LoadSource(strTrainSrc);
                    SourceDescriptor srcTest = factory.LoadSource(strTestSrc);
                    DatasetDescriptor ds = new DatasetDescriptor(nCreatorID, dataset_name, null, null, srcTrain, srcTest, dataset_name, dataset_name + " Character Dataset");
                    factory.AddDataset(ds);
                    factory.UpdateDatasetCounts(ds.ID);
                }

                return true;
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                if (OnCompleted != null)
                    OnCompleted(this, new EventArgs());
            }
        }

        private bool loadFile(DatasetFactory factory, List<Tuple<byte[], int>> rgData, int nC, int nH, int nW, string strSourceName, string strExportPath)
        {
            if (strExportPath != null)
            {
                strExportPath += strSourceName;

                if (!Directory.Exists(strExportPath))
                    Directory.CreateDirectory(strExportPath);
            }

            Stopwatch sw = new Stopwatch();

            reportProgress(0, 0, " Source: " + strSourceName);

            try
            {
                if (factory != null)
                {
                    int nSrcId = factory.AddSource(strSourceName, nC, nW, nH, false, 0, true);

                    factory.Open(nSrcId, 500, Database.FORCE_LOAD.NONE, m_log);
                    factory.DeleteSourceData();
                }

                // Storing to database;
                int nLabel;
                byte[] rgPixels;

                Datum datum = new Datum(false, nC, nW, nH, -1, DateTime.MinValue, new List<byte>(), 0, false, -1);
                string strAction = (m_param.ExportToFile) ? "exporing" : "loading";

                reportProgress(0, rgData.Count, "  " + strAction + " a total of " + rgData.Count.ToString() + " items.");
                reportProgress(0, rgData.Count, "   (with rows: " + nH.ToString() + ", cols: " + nW.ToString() + ")");

                sw.Start();

                List<SimpleDatum> rgImg = new List<SimpleDatum>();

                FileStream fsFileDesc = null;
                StreamWriter swFileDesc = null;
                if (m_param.ExportToFile)
                {
                    string strFile = strExportPath + "\\file_list.txt";
                    fsFileDesc = File.OpenWrite(strFile);
                    swFileDesc = new StreamWriter(fsFileDesc);
                }

                for (int i = 0; i < rgData.Count; i++)
                {
                    rgPixels = rgData[i].Item1;
                    nLabel = rgData[i].Item2;

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        reportProgress(i, rgData.Count, " " + strAction + " data...");
                        sw.Restart();
                    }

                    datum.SetData(rgPixels, nLabel);

                    if (factory != null)
                        factory.PutRawImageCache(i, datum, 5);
                    else if (strExportPath != null)
                        saveToFile(strExportPath, i, datum, swFileDesc);

                    rgImg.Add(new SimpleDatum(datum));

                    if (m_evtCancel.WaitOne(0))
                        return false;
                }

                if (swFileDesc != null)
                {
                    swFileDesc.Flush();
                    swFileDesc.Close();
                    swFileDesc.Dispose();

                    fsFileDesc.Close();
                    fsFileDesc.Dispose();
                }

                if (factory != null)
                {
                    factory.ClearImageCache(true);
                    factory.UpdateSourceCounts();
                    factory.SaveImageMean(SimpleDatum.CalculateMean(m_log, rgImg.ToArray(), new WaitHandle[] { new ManualResetEvent(false) }), true);
                }

                reportProgress(rgData.Count, rgData.Count, " " + strAction + " completed.");
            }
            finally
            {
            }

            return true;
        }

        private void saveToFile(string strPath, int nIdx, Datum d, StreamWriter sw)
        {
            string strFile = strPath.TrimEnd('\\') + "\\" + getImageFileName(nIdx, d);
            Bitmap bmp = ImageData.GetImage(d);

            bmp.Save(strFile);
            bmp.Dispose();

            if (sw != null)
                sw.WriteLine(strFile + " " + d.Label.ToString());
        }

        private string getImageFileName(int nIdx, SimpleDatum sd)
        {
            return "img_" + nIdx.ToString() + "-" + sd.Label.ToString() + ".png";
        }

        private void Log_OnWriteLine(object sender, LogArg e)
        {
            reportProgress((int)(e.Progress * 1000), 1000, e.Message);
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
}
