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
using System.Drawing;
using MyCaffe.basecode.descriptors;
using System.Threading;

namespace MyCaffe.app
{
    public class VOCDataLoader
    {
        List<SimpleDatum> m_rgImg = new List<SimpleDatum>();
        VOCDataParameters m_param;
        DatasetFactory m_factory = new DatasetFactory();
        CancelEvent m_evtCancel = null;

        public event EventHandler<ProgressArgs> OnProgress;
        public event EventHandler<ProgressArgs> OnError;
        public event EventHandler OnCompleted;

        public VOCDataLoader(VOCDataParameters param, CancelEvent evtCancel)
        {
            m_param = param;
            m_evtCancel = evtCancel;
        }

        public void LoadDatabase()
        {
            try
            {
                reportProgress(0, 0, "Loading database...");

                int nIdx = 0;
                int nTotal = 40178 + 10935;
                Log log = new Log("VOC");
                log.OnWriteLine += Log_OnWriteLine;

                DatasetFactory factory = new DatasetFactory();

                if (!loadFile(m_param.DataBatchFile1, "VOC.training", nTotal, ref nIdx, log, m_param.ExtractFiles))
                    return;

                if (!loadFile(m_param.DataBatchFile2, "VOC.training", nTotal, ref nIdx, log, m_param.ExtractFiles))
                    return;

                SourceDescriptor srcTrain = factory.LoadSource("VOC.training");
#warning TODO: Save image mean, datasources and dataset.
                //m_factory.SaveImageMean(SimpleDatum.CalculateMean(log, m_rgImg.ToArray(), new WaitHandle[] { new ManualResetEvent(false) }), true, srcTrain.ID);

                m_rgImg = new List<SimpleDatum>();
                nIdx = 0;
                nTotal = 10347;
                if (!loadFile(m_param.DataBatchFile3, "VOC.testing", nTotal, ref nIdx, log, m_param.ExtractFiles))
                    return;

                SourceDescriptor srcTest = factory.LoadSource("VOC.testing");
                //m_factory.SaveImageMean(SimpleDatum.CalculateMean(log, m_rgImg.ToArray(), new WaitHandle[] { new ManualResetEvent(false) }), true, srcTest.ID);

                DatasetDescriptor ds = new DatasetDescriptor(0, "VOC", null, null, srcTrain, srcTest, "VOC", "VOC Dataset");
                //factory.AddDataset(ds);
                //factory.UpdateDatasetCounts(ds.ID);
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

        private void Log_OnWriteLine(object sender, LogArg e)
        {
            reportProgress((int)(e.Progress * 1000), 1000, e.Message);
        }

        private bool loadFile(string strImagesFile, string strSourceName, int nTotal, ref int nIdx, Log log, bool bExtractFiles)
        {
            Stopwatch sw = new Stopwatch();

            reportProgress(nIdx, nTotal, " Source: " + strSourceName);
            reportProgress(nIdx, nTotal, "  loading " + strImagesFile + "...");

            FileStream fs = null;

            try
            {
                int nPos = strImagesFile.ToLower().LastIndexOf(".tar");
                string strPath = strImagesFile.Substring(0, nPos);

                if (!Directory.Exists(strPath))
                    Directory.CreateDirectory(strPath);

                if (bExtractFiles)
                {
                    log.Progress = (double)nIdx / nTotal;
                    log.WriteLine("Extracting files from '" + strImagesFile + "'...");

                    if ((nIdx = TarFile.ExtractTar(strImagesFile, strPath, m_evtCancel, log, nTotal, nIdx)) == 0)
                    {
                        log.WriteLine("Aborted.");
                        return false;
                    }
                }

#warning TODO: Add files to database.
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }

            return true;
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
