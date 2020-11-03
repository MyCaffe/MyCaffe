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
using MyCaffe.param.ssd;
using System.Xml;
using System.Xml.Linq;

namespace MyCaffe.data
{
    /// <summary>
    /// The VOCDataLoader is used to create the VOC0712 (VOC2007 and VOC2012) dataset and load it into the database managed by the MyCaffe Image Database.
    /// </summary>
    /// <remarks>
    /// @see [VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
    /// </remarks>
    public class VOCDataLoader
    {
        List<SimpleDatum> m_rgImg = new List<SimpleDatum>();
        VOCDataParameters m_param;
        DatasetFactory m_factory = new DatasetFactory();
        CancelEvent m_evtCancel = null;
        Log m_log;

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
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specfies the cancel event used to cancel the process.</param>
        public VOCDataLoader(VOCDataParameters param, Log log, CancelEvent evtCancel)
        {
            m_param = param;
            m_log = log;
            m_evtCancel = evtCancel;
            m_evtCancel.Reset();
        }

        private string dataset_name
        {
            get { return "VOC0712"; }
        }

        /// <summary>
        /// Create the dataset and load it into the database.
        /// </summary>
        /// <param name="nCreatorID">Specifies the creator ID.</param>
        /// <returns>On successful creation, <i>true</i> is returned, otherwise <i>false</i> is returned on abort.</returns>
        public bool LoadDatabase(int nCreatorID = 0)
        {
            try
            {
                reportProgress(0, 0, "Loading " + dataset_name + " database...");

                int nIdx = 0;
                int nTotal = 5011 + 17125;
                int nExtractIdx = 0;
                int nExtractTotal = 10935 + 40178;

                // Get the label map.
                LabelMap labelMap = loadLabelMap();
                Dictionary<string, int> rgNameToLabel = labelMap.MapToLabel(m_log, true);
                string strSrc = dataset_name + ".training";

                int nSrcId = m_factory.GetSourceID(strSrc);
                if (nSrcId > 0)
                    m_factory.DeleteSourceData(nSrcId);

                List<Tuple<int, string, Size>> rgFileSizes = new List<Tuple<int, string, Size>>();

                if (!loadFile(m_param.DataBatchFileTrain2007, strSrc, nExtractTotal, ref nExtractIdx, nTotal, ref nIdx, m_log, m_param.ExtractFiles, rgNameToLabel, rgFileSizes))
                    return false;

                if (!loadFile(m_param.DataBatchFileTrain2012, strSrc, nExtractTotal, ref nExtractIdx, nTotal, ref nIdx, m_log, m_param.ExtractFiles, rgNameToLabel, rgFileSizes))
                    return false;

                string strDir = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\ssd\\VOC0712\\";
                if (!Directory.Exists(strDir))
                    Directory.CreateDirectory(strDir);

                saveFileSizes(rgFileSizes, strDir + "train_name_size.txt");

                SourceDescriptor srcTrain = m_factory.LoadSource(strSrc);

                rgFileSizes = new List<Tuple<int, string, Size>>();
                m_rgImg = new List<SimpleDatum>();
                nIdx = 0;
                nTotal = 4952;
                nExtractIdx = 0;
                nExtractTotal = 10347;

                strSrc = dataset_name + ".testing";

                nSrcId = m_factory.GetSourceID(strSrc);
                if (nSrcId > 0)
                    m_factory.DeleteSourceData(nSrcId);

                if (!loadFile(m_param.DataBatchFileTest2007, strSrc, nExtractTotal, ref nExtractIdx, nTotal, ref nIdx, m_log, m_param.ExtractFiles, rgNameToLabel, rgFileSizes))
                    return false;

                saveFileSizes(rgFileSizes, strDir + "test_name_size.txt");

                SourceDescriptor srcTest = m_factory.LoadSource(strSrc);

                DatasetDescriptor ds = new DatasetDescriptor(nCreatorID, dataset_name, null, null, srcTrain, srcTest, dataset_name, dataset_name + " Dataset");
                m_factory.AddDataset(ds);
                m_factory.UpdateDatasetCounts(ds.ID);

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

        private void saveFileSizes(List<Tuple<int, string, Size>> rgFileSizes, string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (StreamWriter sw = new StreamWriter(strFile))
            {
                foreach (Tuple<int, string, Size> item in rgFileSizes)
                {
                    string strLine = item.Item1.ToString() + ", ";
                    strLine += item.Item2.ToString() + ", ";
                    strLine += item.Item3.Height.ToString() + ", ";
                    strLine += item.Item3.Width.ToString();
                    sw.WriteLine(strLine);
                }
            }
        }

        private void addLabels(int nSrcId, Dictionary<string, int> rgNameToLabel)
        {
            foreach (KeyValuePair<string, int> kv in rgNameToLabel)
            {
                m_factory.AddLabel(kv.Value, kv.Key, nSrcId);
            }
        }

        private bool loadFile(string strImagesFile, string strSourceName, int nExtractTotal, ref int nExtractIdx, int nTotal, ref int nIdx, Log log, bool bExtractFiles, Dictionary<string, int> rgNameToLabel, List<Tuple<int, string, Size>> rgFileSizes)
        {
            Stopwatch sw = new Stopwatch();

            reportProgress(nIdx, nTotal, " Source: " + strSourceName);
            reportProgress(nIdx, nTotal, "  loading " + strImagesFile + "...");

            FileStream fs = null;

            try
            {
                int nSrcId = m_factory.AddSource(strSourceName, 3, -1, -1, false);
                addLabels(nSrcId, rgNameToLabel);
                m_factory.Open(nSrcId, 500, Database.FORCE_LOAD.NONE, log);

                int nPos = strImagesFile.ToLower().LastIndexOf(".tar");
                string strPath = strImagesFile.Substring(0, nPos);

                if (!Directory.Exists(strPath))
                    Directory.CreateDirectory(strPath);

                if (bExtractFiles)
                {
                    log.Progress = (double)nIdx / nExtractTotal;
                    log.WriteLine("Extracting files from '" + strImagesFile + "'...");

                    if ((nExtractIdx = TarFile.ExtractTar(strImagesFile, strPath, m_evtCancel, log, nExtractTotal, nExtractIdx)) == 0)
                    {
                        log.WriteLine("Aborted.");
                        return false;
                    }
                }

                // Load the annotations.
                SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;
                int nResizeHeight = 0;
                int nResizeWidth = 0;

                // Create the training database images.
                // Create the master list file.
                List<Tuple<string, string>> rgFiles = createFileList(log, strPath);

                sw.Start();
                for (int i = 0; i < rgFiles.Count; i++)
                {
                    SimpleDatum datum = loadDatum(log, rgFiles[i].Item1, rgFiles[i].Item2, nResizeHeight, nResizeWidth, type, rgNameToLabel);
                    m_factory.PutRawImageCache(nIdx, datum);
                    nIdx++;

                    if (m_evtCancel.WaitOne(0))
                    {
                        log.WriteLine("Aborted.");
                        return false;
                    }

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        log.Progress = (double)nIdx / nTotal;
                        log.WriteLine("Loading file " + i.ToString() + " of " + rgFiles.Count.ToString() + "...");
                        sw.Restart();
                    }

                    rgFileSizes.Add(new Tuple<int, string, Size>(nIdx, rgFiles[i].Item1, new Size(datum.Width, datum.Height)));
                }

                m_factory.ClearImageCashe(true);
                m_factory.Close();
            }
            finally
            {
                if (fs != null)
                    fs.Dispose();
            }

            return true;
        }

        private SimpleDatum loadDatum(Log log, string strImgFile, string strAnnotationFile, int nResizeHeight, int nResizeWidth, SimpleDatum.ANNOTATION_TYPE type, Dictionary<string, int> rgNameToLabel)
        {
            Bitmap bmp = new Bitmap(strImgFile);

            if (nResizeHeight == 0)
                nResizeHeight = bmp.Height;

            if (nResizeWidth == 0)
                nResizeWidth = bmp.Width;

            if (nResizeHeight != bmp.Height || nResizeWidth != bmp.Width)
            {
                Bitmap bmpNew = ImageTools.ResizeImage(bmp, nResizeWidth, nResizeHeight);
                bmp.Dispose();
                bmp = bmpNew;
            }

            SimpleDatum datum = ImageData.GetImageDataD(bmp, 3, false, 0, false);
            loadAnnotationFile(log, strAnnotationFile, datum, type, rgNameToLabel);

            return datum;
        }

        private bool loadAnnotationFile(Log log, string strFile, SimpleDatum datum, SimpleDatum.ANNOTATION_TYPE type, Dictionary<string, int> rgNameToLabel)
        {
            if (type != SimpleDatum.ANNOTATION_TYPE.BBOX)
            {
                log.FAIL("Unknown annotation type '" + type.ToString() + "'!");
                return false;
            }

            datum.annotation_group = new AnnotationGroupCollection();
            datum.annotation_type = type;

            string strExt = Path.GetExtension(strFile).ToLower();

            switch (strExt)
            {
                case ".xml":
                    return loadXmlAnnotationFile(log, strFile, datum, rgNameToLabel);

                default:
                    log.FAIL("Unknown annotation file type '" + strExt + "'!");
                    break;
            }

            return true;
        }

        private bool loadXmlAnnotationFile(Log log, string strFile, SimpleDatum datum, Dictionary<string, int> rgNameToLabel)
        {
            XDocument doc = XDocument.Load(strFile);
            XElement size = doc.Descendants("size").First();
            XElement val;

            val = size.Descendants("width").First();
            int nWidth = int.Parse(val.Value);

            val = size.Descendants("height").First();
            int nHeight = int.Parse(val.Value);

            val = size.Descendants("depth").First();
            int nChannels = int.Parse(val.Value);

            if (datum.Height != nHeight || datum.Width != nWidth || datum.Channels != nChannels)
                log.FAIL("Inconsistent image size, expected (" + datum.Channels.ToString() + "," + datum.Height.ToString() + "," + datum.Width.ToString() + ") but annotation has size (" + nChannels.ToString() + "," + nHeight.ToString() + "," + nWidth.ToString() + ").");

            int nInstanceId = 0;

            List<XElement> objects = doc.Descendants("object").ToList();
            foreach (XElement obj in objects)
            {
                val = obj.Descendants("name").First();
                string strName = val.Value;

                val = obj.Descendants("difficult").First();
                bool bDifficult = (val.Value == "0") ? false : true;

                XElement bndbox = obj.Descendants("bndbox").First();

                val = bndbox.Descendants("xmin").First();
                float fxmin = BaseParameter.parseFloat(val.Value);
                if (fxmin > nWidth || fxmin < 0)
                    log.WriteLine("WARNING: '" + strFile + "' bounding box exceeds image boundary.");

                val = bndbox.Descendants("ymin").First();
                float fymin = BaseParameter.parseFloat(val.Value);
                if (fymin > nHeight || fymin < 0)
                    log.WriteLine("WARNING: '" + strFile + "' bounding box exceeds image boundary.");

                val = bndbox.Descendants("xmax").First();
                float fxmax = BaseParameter.parseFloat(val.Value);
                if (fxmax > nWidth || fxmax < 0)
                    log.WriteLine("WARNING: '" + strFile + "' bounding box exceeds image boundary.");

                val = bndbox.Descendants("ymax").First();
                float fymax = BaseParameter.parseFloat(val.Value);
                if (fymax > nHeight || fymax < 0)
                    log.WriteLine("WARNING: '" + strFile + "' bounding box exceeds image boundary.");

                if (!rgNameToLabel.ContainsKey(strName))
                {
                    log.FAIL("Could not find the label '" + strName + "' in the label mapping!");
                    return false;
                }

                int nLabel = rgNameToLabel[strName];
                NormalizedBBox bbox = new NormalizedBBox(fxmin / nWidth, fymin / nHeight, fxmax / nWidth, fymax / nHeight, nLabel, bDifficult);
                datum.SetLabel(nLabel);

                foreach (AnnotationGroup g in datum.annotation_group)
                {
                    if (nLabel == g.group_label)
                    {
                        if (g.annotations.Count == 0)
                            nInstanceId = 0;
                        else
                            nInstanceId = g.annotations[g.annotations.Count - 1].instance_id + 1;

                        g.annotations.Add(new Annotation(bbox, nInstanceId));
                        bbox = null;
                        break;
                    }
                }

                if (bbox != null)
                {
                    nInstanceId = 0;
                    AnnotationGroup grp = new AnnotationGroup(null, nLabel);
                    grp.annotations.Add(new Annotation(bbox, nInstanceId));
                    datum.annotation_group.Add(grp);
                    bbox = null;
                }
            }

            return true;
        }

        private List<Tuple<string, string>> createFileList(Log log, string strFile)
        {
            List<Tuple<string, string>> rgFiles = new List<Tuple<string, string>>();

            string strPath = strFile;
            int nPos = strFile.ToLower().LastIndexOf(".tar");
            if (nPos > 0)
                strPath = strFile.Substring(0, nPos);

            strPath += "\\VOCdevkit\\VOC";

            if (strPath.Contains("2012"))
                strPath += "2012";

            if (strPath.Contains("2007"))
                strPath += "2007";

            loadFileList(log, strPath, rgFiles);
            return rgFiles;
        }

        private void loadFileList(Log log, string strPath, List<Tuple<string, string>> rgFiles)
        {
            log.WriteLine("Creating the list file for " + dataset_name + " dataset...");

            string strImgPath = strPath + "\\JPEGImages";
            string strLabelPath = strPath + "\\Annotations";

            string[] rgImgFiles = Directory.GetFiles(strImgPath);
            string[] rgLabelFiles = Directory.GetFiles(strLabelPath);

            if (rgImgFiles.Length != rgLabelFiles.Length)
            {
                log.FAIL("The image path '" + strImgPath + "' has " + rgImgFiles.Length.ToString() + " files and label path '" + strLabelPath + "' has " + rgLabelFiles.Length.ToString() + " files - both paths should have the same number of files!");
                return;
            }

            for (int i = 0; i < rgImgFiles.Length; i++)
            {
                rgFiles.Add(new Tuple<string, string>(rgImgFiles[i], rgLabelFiles[i]));
            }
        }

        private string test_data_path
        {
            get
            {
                string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData).TrimEnd('\\');
                strPath += "\\MyCaffe\\test_data";
                return strPath;
            }
        }

        private string getDataFile(string strSubDir, string strFileName)
        {
            string strPath = test_data_path;
            strPath += "\\data\\ssd\\" + strSubDir + "\\" + strFileName;
            return strPath;
        }

        private LabelMap loadLabelMap()
        {
            string strFile = getDataFile(dataset_name, "labelmap_voc.prototxt");
            RawProto proto = RawProtoFile.LoadFromFile(strFile);
            return LabelMap.FromProto(proto);
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
