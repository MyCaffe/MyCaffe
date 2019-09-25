using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.fillers;
using MyCaffe.param;
using MyCaffe.param.ssd;

namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// Generate the detection output based on location and confidence predictions by doing non maximum supression.  Intended for use with MultiBox detection method used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DetectionOutputLayer<T> : Layer<T>
    {
        int m_nNumClasses;
        bool m_bShareLocations;
        int m_nNumLocClasses;
        int m_nBackgroundLabelId;
        PriorBoxParameter.CodeType m_codeType;
        bool m_bVarianceEncodedInTarget;
        int m_nKeepTopK;
        float m_fConfidenceThreshold;
        int m_nNumPriors;
        float m_fNmsThreshold;
        int m_nTopK;
        float m_fEta;

        bool m_bNeedSave = false;
        string m_strOutputDir;
        string m_strOutputNamePrefix;
        SaveOutputParameter.OUTPUT_FORMAT m_outputFormat = SaveOutputParameter.OUTPUT_FORMAT.VOC;
        Dictionary<int, string> m_rgLabelToName = new Dictionary<int, string>();
        Dictionary<int, string> m_rgLabelToDisplayName = new Dictionary<int, string>();
        List<string> m_rgstrNames = new List<string>();
        List<SizeF> m_rgSizes = new List<SizeF>();
        int m_nNumTestImage;
        int m_nNameCount;
        ResizeParameter m_resizeParam = null;

        PropertyTree m_detections = new PropertyTree();

        bool m_bVisualize;
        float m_fVisualizeThreshold;
        DataTransformer<T> m_transformer;
        string m_strSaveFile;
        Blob<T> m_blobBboxPreds;
        Blob<T> m_blobBboxPermute;
        Blob<T> m_blobConfPermute;
        BBoxUtility<T> m_bboxUtil;

        /// <summary>
        /// The DetectionOutputLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type DETECTION_OUTPUT with parameter detection_output_param.
        /// </param>
        public DetectionOutputLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DETECTION_OUTPUT;
            m_blobBboxPreds = new Blob<T>(cuda, log);
            m_blobBboxPreds.Name = "bbox preds";
            m_blobBboxPermute = new Blob<T>(cuda, log);
            m_blobBboxPermute.Name = "bbox permute";
            m_blobConfPermute = new Blob<T>(cuda, log);
            m_blobConfPermute.Name = "bbox conf";
            m_bboxUtil = new BBoxUtility<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobBboxPreds != null)
            {
                m_blobBboxPreds.Dispose();
                m_blobBboxPreds = null;
            }

            if (m_blobBboxPermute != null)
            {
                m_blobBboxPermute.Dispose();
                m_blobBboxPermute = null;
            }

            if (m_blobConfPermute != null)
            {
                m_blobConfPermute.Dispose();
                m_blobConfPermute = null;
            }

            if (m_bboxUtil != null)
            {
                m_bboxUtil.Dispose();
                m_bboxUtil = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobBboxPreds);
                col.Add(m_blobBboxPermute);
                col.Add(m_blobConfPermute);

                return col;
            }
        }

        /// <summary>
        /// Returns the minimum number of bottom (input) Blobs: loc pred, conf pred, prior bbox
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Returns the maximum number of bottom (input) Blobs: loc pred, conf pred, prior bbox
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 4; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: det
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GT(m_param.detection_output_param.num_classes, 0, "There must be at least one class specified.");
            m_nNumClasses = (int)m_param.detection_output_param.num_classes;

            m_bShareLocations = m_param.detection_output_param.share_location;
            m_nNumLocClasses = (m_bShareLocations) ? 1 : m_nNumClasses;
            m_nBackgroundLabelId = m_param.detection_output_param.background_label_id;
            m_codeType = m_param.detection_output_param.code_type;
            m_bVarianceEncodedInTarget = m_param.detection_output_param.variance_encoded_in_target;
            m_nKeepTopK = m_param.detection_output_param.keep_top_k;
            m_fConfidenceThreshold = m_param.detection_output_param.confidence_threshold.GetValueOrDefault(-float.MaxValue);

            // Parameters used in nms.
            m_fNmsThreshold = m_param.detection_output_param.nms_param.nms_threshold;
            m_log.CHECK_GE(m_fNmsThreshold, 0, "The nms_threshold must be non negative.");
            m_fEta = m_param.detection_output_param.nms_param.eta;
            m_log.CHECK_GT(m_fEta, 0, "The nms_param.eta must be > 0.");
            m_log.CHECK_LE(m_fEta, 1, "The nms_param.eta must be < 0.");

            m_nTopK = m_param.detection_output_param.nms_param.top_k.GetValueOrDefault(-1);

            m_strOutputDir = m_param.detection_output_param.save_output_param.output_directory;
            m_bNeedSave = !string.IsNullOrEmpty(m_strOutputDir);
            if (m_bNeedSave && !Directory.Exists(m_strOutputDir))
                Directory.CreateDirectory(m_strOutputDir);

            m_strOutputNamePrefix = m_param.detection_output_param.save_output_param.output_name_prefix;
            m_outputFormat = m_param.detection_output_param.save_output_param.output_format;

            if (!string.IsNullOrEmpty(m_param.detection_output_param.save_output_param.label_map_file))
            {
                string strLabelMapFile = m_param.detection_output_param.save_output_param.label_map_file;
                if (!File.Exists(strLabelMapFile))
                {
                    // Ignore saving if there is no label map file.
                    m_log.WriteLine("WARNING: Could not find the label_map_file '" + strLabelMapFile + "'!");
                    m_bNeedSave = false;
                }
                else
                {
                    LabelMap label_map;

                    try
                    {
                        RawProto proto = RawProto.FromFile(strLabelMapFile);
                        label_map = LabelMap.FromProto(proto);
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception("Failed to read label map file!", excpt);
                    }

                    try
                    {
                        m_rgLabelToName = label_map.MapToName(m_log, true, false);
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception("Failed to convert the label to name!", excpt);
                    }

                    try
                    {
                        m_rgLabelToDisplayName = label_map.MapToName(m_log, true, true);
                    }
                    catch (Exception excpt)
                    {
                        throw new Exception("Failed to convert the label to display name!", excpt);
                    }
                }
            }
            else
            {
                m_bNeedSave = false;
            }

            if (!string.IsNullOrEmpty(m_param.detection_output_param.save_output_param.name_size_file))
            {
                string strNameSizeFile = m_param.detection_output_param.save_output_param.name_size_file;
                if (!File.Exists(strNameSizeFile))
                {
                    // Ignore saving if there is no name size file.
                    m_log.WriteLine("WARNING: Could not find the name_size_file '" + strNameSizeFile + "'!");
                    m_bNeedSave = false;
                }
                else
                {
                    using (StreamReader sr = new StreamReader(strNameSizeFile))
                    {
                        string strName;
                        int nHeight;
                        int nWidth;

                        string strLine = sr.ReadLine();
                        while (strLine != null)
                        {
                            string[] rgstr = strLine.Split(' ');
                            if (rgstr.Length != 3)
                                throw new Exception("Invalid name_size_file format, expected 'name' 'height' 'width'");

                            strName = rgstr[0];
                            nHeight = int.Parse(rgstr[1]);
                            nWidth = int.Parse(rgstr[2]);

                            m_rgstrNames.Add(strName);
                            m_rgSizes.Add(new SizeF(nWidth, nHeight));

                            strLine = sr.ReadLine();
                        }
                    }

                    if (m_param.detection_output_param.save_output_param.num_test_image.HasValue)
                        m_nNumTestImage = (int)m_param.detection_output_param.save_output_param.num_test_image.Value;
                    else
                        m_nNumTestImage = m_rgstrNames.Count;

                    m_log.CHECK_LE(m_nNumTestImage, m_rgstrNames.Count, "The number of test images cannot exceed the number of names.");
                }
            }
            else
            {
                m_bNeedSave = false;
            }

            if (m_param.detection_output_param.save_output_param.resize_param != null)
                m_resizeParam = m_param.detection_output_param.save_output_param.resize_param;

            m_nNameCount = 0;

            m_bVisualize = m_param.detection_output_param.visualize;
            if (m_bVisualize)
            {
                m_fVisualizeThreshold = m_param.detection_output_param.visualize_threshold.GetValueOrDefault(0.6f);
                m_transformer = new DataTransformer<T>(m_cuda, m_log, m_param.transform_param, m_phase, 0, 0, 0);
                m_transformer.InitRand();
                m_strSaveFile = m_param.detection_output_param.save_file;
            }

            m_blobBboxPreds.ReshapeLike(colBottom[0]);

            if (!m_bShareLocations)
                m_blobBboxPermute.ReshapeLike(colBottom[0]);

            m_blobConfPermute.ReshapeLike(colBottom[1]);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_bNeedSave)
            {
                m_log.CHECK_LE(m_nNameCount, m_rgstrNames.Count, "The name count must be <= the number of names.");

                if (m_nNameCount % m_nNumTestImage == 0)
                {
                    // Clean all outputs.
                    if (m_outputFormat == SaveOutputParameter.OUTPUT_FORMAT.VOC)
                    {
                        string strDir = m_strOutputDir;

                        foreach (KeyValuePair<int, string> kv in m_rgLabelToName)
                        {
                            if (kv.Key == m_nBackgroundLabelId)
                                continue;

                            string strFile = strDir.TrimEnd('\\') + "\\" + kv.Value + ".txt";
                            if (File.Exists(strFile))
                                File.Delete(strFile);
                        }
                    }
                }
            }

            m_log.CHECK_EQ(colBottom[0].num, colBottom[1].num, "The bottom[0] and bottom[1] must have the same 'num'.");

            m_blobBboxPreds.ReshapeLike(colBottom[0]);

            if (!m_bShareLocations)
                m_blobBboxPermute.ReshapeLike(colBottom[0]);

            m_blobConfPermute.ReshapeLike(colBottom[1]);

            m_nNumPriors = colBottom[2].height / 4;
            m_log.CHECK_EQ(m_nNumPriors * m_nNumLocClasses * 4, colBottom[0].channels, "The number of priors must match the number of location predictions (bottom[0]).");
            m_log.CHECK_EQ(m_nNumPriors * m_nNumClasses, colBottom[1].channels, "The number of priors must match the number of confidence predictions (bottom[1]).");

            // num() and channels() are 1.
            List<int> rgTopShape = Utility.Create<int>(2, 1);
            // Since the number of bboxes to be kept is unknown before nms, we manually set it to (fake) 1.
            rgTopShape.Add(1);
            // Each row is a 7 dimension vecotr, which stores:
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            rgTopShape.Add(7);

            colTop[0].Reshape(rgTopShape);
        }

        private string getFileName(string strLabel, string strExt)
        {
            string strFile = m_strOutputDir.TrimEnd('\\');
            strFile += "\\";
            strFile += m_strOutputNamePrefix;
            strFile += strLabel;
            strFile += ".";
            strFile += strExt;

            return strFile;
        }

        /// <summary>
        /// Do non-maximum suppression (nms) on prediction results.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (at least 2)
        ///  -# @f$ (N \times C1 \times 1 \times 1) @f$ the location predictions with C1 predictions.
        ///  -# @f$ (N \times C2 \times 1 \times 1) @f$ the confidence predictions with C2 predictions.
        ///  -# @f$ (N \times 2 \times C3 \times 1) @f$ the prior bounding boxes with C3 values.
        /// </param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (1 \times 1 \times N \times 7) @f$ N is the number of detections after, and each row is: [image_id, label, confidence, xmin, ymin, xmax, ymax].
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float[] rgfLocData = convertF(colBottom[0].mutable_cpu_data);
            float[] rgfConfData = convertF(colBottom[1].mutable_cpu_data);
            float[] rgfPriorData = convertF(colBottom[2].mutable_cpu_data);
            int nNum = colBottom[0].num;

            // Retrieve all location predictions.
            List<LabelBBox> rgAllLocPreds = m_bboxUtil.GetLocPredictions(rgfLocData, nNum, m_nNumPriors, m_nNumLocClasses, m_bShareLocations);

            // Retrieve all confidence scores.
            List<Dictionary<int, List<float>>> rgAllConfScores = m_bboxUtil.GetConfidenceScores(rgfConfData, m_nNumClasses, m_nNumPriors, m_nNumClasses);

            // Retrieve all prior bboxes, which is the same within a batch since we assume all
            // images in a batch are of the same dimension.
            List<List<float>> rgrgPriorVariances;
            List<NormalizedBBox> rgPriorBboxes = m_bboxUtil.GetPrior(rgfPriorData, m_nNumPriors, out rgrgPriorVariances);

            // Decode all loc predictions to bboxes.
            bool bClipBbox = false;
            List<LabelBBox> rgAllDecodeBboxes = m_bboxUtil.DecodeAll(rgAllLocPreds, rgPriorBboxes, rgrgPriorVariances, nNum, m_bShareLocations, m_nNumLocClasses, m_nBackgroundLabelId, m_codeType, m_bVarianceEncodedInTarget, bClipBbox);

            int nNumKept = 0;
            List<Dictionary<int, List<int>>> rgAllIndices = new List<Dictionary<int, List<int>>>();

            for (int i=0; i < nNum; i++)
            {
                LabelBBox decode_bboxes = rgAllDecodeBboxes[i];
                Dictionary<int, List<float>> rgConfScores = rgAllConfScores[i];
                Dictionary<int, List<int>> rgIndices = new Dictionary<int, List<int>>();
                int nNumDet = 0;

                for (int c = 0; c < m_nNumClasses; c++)
                {
                    // Ignore background class.
                    if (c == m_nBackgroundLabelId)
                        continue;

                    // Something bad happened if there are no predictions for the current label.
                    if (!rgConfScores.ContainsKey(c))
                        m_log.FAIL("Could not find confidence predictions for label '" + c.ToString() + "'!");

                    List<float> rgfScores = rgConfScores[c];
                    int nLabel = (m_bShareLocations) ? -1 : c;

                    // Something bad happened if there are no locations for the current label.
                    if (!decode_bboxes.Contains(nLabel))
                        m_log.FAIL("Could not find location predictions for the label '" + nLabel.ToString() + "'!");

                    List<NormalizedBBox> rgBboxes = decode_bboxes[nLabel];
                    List<int> rgIndexes;
                    m_bboxUtil.ApplyNMSFast(rgBboxes, rgfScores, m_fConfidenceThreshold, m_fNmsThreshold, m_fEta, m_nTopK, out rgIndexes);
                    rgIndices[c] = rgIndexes;
                    nNumDet += rgIndexes.Count;
                }

                if (m_nKeepTopK > -1 && nNumDet > m_nKeepTopK)
                {
                    List<Tuple<float, Tuple<int, int>>> rgScoreIndexPairs = new List<Tuple<float, Tuple<int, int>>>();

                    foreach (KeyValuePair<int, List<int>> kv in rgIndices)
                    {
                        int nLabel = kv.Key;
                        List<int> rgLabelIndices = kv.Value;

                        // Something bad happend for the current label.
                        if (!rgConfScores.ContainsKey(nLabel))
                            m_log.FAIL("Could not find location predictions for label " + nLabel.ToString() + "!");

                        List<float> rgScores = rgConfScores[nLabel];
                        for (int j = 0; j < rgLabelIndices.Count; j++)
                        {
                            int nIdx = rgLabelIndices[j];
                            m_log.CHECK_LT(nIdx, rgScores.Count, "The current index must be less than the number of scores!");
                            rgScoreIndexPairs.Add(new Tuple<float, Tuple<int, int>>(rgScores[nIdx], new Tuple<int, int>(nLabel, nIdx)));
                        }
                    }

                    // Keep top k results per image.
                    rgScoreIndexPairs = rgScoreIndexPairs.OrderByDescending(p => p.Item1).ToList();
                    if (rgScoreIndexPairs.Count > m_nKeepTopK)
                        rgScoreIndexPairs = rgScoreIndexPairs.Take(m_nKeepTopK).ToList();

                    // Store the new indices.
                    Dictionary<int, List<int>> rgNewIndices = new Dictionary<int, List<int>>();
                    for (int j = 0; j < rgScoreIndexPairs.Count; j++)
                    {
                        int nLabel = rgScoreIndexPairs[j].Item2.Item1;
                        int nIdx = rgScoreIndexPairs[j].Item2.Item2;

                        if (!rgNewIndices.ContainsKey(nLabel))
                            rgNewIndices.Add(nLabel, new List<int>());

                        rgNewIndices[nLabel].Add(nIdx);
                    }

                    rgAllIndices.Add(rgNewIndices);
                    nNumKept += m_nKeepTopK;
                }
                else
                {
                    rgAllIndices.Add(rgIndices);
                    nNumKept += nNumDet;
                }
            }

            List<int> rgTopShape = Utility.Create<int>(2, 1);
            rgTopShape.Add(nNumKept);
            rgTopShape.Add(7);
            float[] rgfTopData = null;

            if (nNumKept == 0)
            {
                m_log.WriteLine("WARNING: Could not find any detections.");
                rgTopShape[2] = nNum;
                colTop[0].Reshape(rgTopShape);

                colTop[0].SetData(-1);
                rgfTopData = convertF(colTop[0].mutable_cpu_data);
                int nOffset = 0;

                // Generate fake results per image.
                for (int i = 0; i < nNum; i++)
                {
                    rgfTopData[nOffset + 0] = i;
                    nOffset += 7;
                }
            }
            else
            {
                colTop[0].Reshape(rgTopShape);
                rgfTopData = convertF(colTop[0].mutable_cpu_data);
            }

            int nCount = 0;
            string strDir = m_strOutputDir;

            for (int i = 0; i < nNum; i++)
            {
                Dictionary<int, List<float>> rgConfScores = rgAllConfScores[i];
                LabelBBox decode_bboxes = rgAllDecodeBboxes[i];

                foreach (KeyValuePair<int, List<int>> kv in rgAllIndices[i])
                {
                    int nLabel = kv.Key;

                    // Something bad happened if there are no predictions for the current label.
                    if (!rgConfScores.ContainsKey(nLabel))
                        m_log.FAIL("Could not find confidence predictions for label '" + nLabel.ToString() + "'!");

                    List<float> rgfScores = rgConfScores[nLabel];
                    int nLocLabel = (m_bShareLocations) ? -1 : nLabel;

                    // Something bad happened if therea re no predictions for the current label.
                    if (!decode_bboxes.Contains(nLabel))
                        m_log.FAIL("COuld not find location predictions for label '" + nLabel.ToString() + "'!");

                    List<NormalizedBBox> rgBboxes = decode_bboxes[nLabel];
                    List<int> rgIndices = kv.Value;

                    if (m_bNeedSave)
                    {
                        m_log.CHECK(m_rgLabelToName.ContainsKey(nLabel), "The label to name mapping does not contain the label '" + nLabel.ToString() + "'!");
                        m_log.CHECK_LT(m_nNameCount, m_rgstrNames.Count, "The name count must be less than the number of names.");
                    }

                    for (int j = 0; j < rgIndices.Count; j++)
                    {
                        int nIdx = rgIndices[j];
                        rgfTopData[nCount * 7 + 0] = i;
                        rgfTopData[nCount * 7 + 1] = nLabel;
                        rgfTopData[nCount * 7 + 2] = rgfScores[nIdx];

                        NormalizedBBox bbox = rgBboxes[nIdx];
                        rgfTopData[nCount * 7 + 3] = bbox.xmin;
                        rgfTopData[nCount * 7 + 4] = bbox.ymin;
                        rgfTopData[nCount * 7 + 5] = bbox.xmax;
                        rgfTopData[nCount * 7 + 6] = bbox.ymax;

                        if (m_bNeedSave)
                        {
                            NormalizedBBox out_bbox = m_bboxUtil.Output(bbox, m_rgSizes[m_nNameCount], m_resizeParam);

                            float fScore = rgfTopData[nCount * 7 + 2];
                            float fXmin = out_bbox.xmin;
                            float fYmin = out_bbox.ymin;
                            float fXmax = out_bbox.xmax;
                            float fYmax = out_bbox.ymax;

                            PropertyTree pt_xmin = new PropertyTree();
                            pt_xmin.Put("", Math.Round(fXmin * 100) / 100);

                            PropertyTree pt_ymin = new PropertyTree();
                            pt_ymin.Put("", Math.Round(fYmin * 100) / 100);

                            PropertyTree pt_wd = new PropertyTree();
                            pt_wd.Put("", Math.Round((fXmax - fXmin) * 100) / 100);

                            PropertyTree pt_ht = new PropertyTree();
                            pt_ht.Put("", Math.Round((fYmax - fYmin) * 100) / 100);

                            PropertyTree cur_bbox = new PropertyTree();
                            cur_bbox.AddChild("", pt_xmin);
                            cur_bbox.AddChild("", pt_ymin);
                            cur_bbox.AddChild("", pt_wd);
                            cur_bbox.AddChild("", pt_ht);

                            PropertyTree cur_det = new PropertyTree();
                            cur_det.Put("image_id", m_rgstrNames[m_nNameCount]);
                            if (m_outputFormat == SaveOutputParameter.OUTPUT_FORMAT.ILSVRC)
                                cur_det.Put("category_id", nLabel);
                            else
                                cur_det.Put("category_id", m_rgLabelToName[nLabel]);

                            cur_det.AddChild("bbox", cur_bbox);
                            cur_det.Put("score", fScore);

                            m_detections.AddChild("", cur_det);
                        }

                        nCount++;
                    }
                }

                if (m_bNeedSave)
                {
                    m_nNameCount++;

                    if (m_nNameCount % m_nNumTestImage == 0)
                    {
                        if (m_outputFormat == SaveOutputParameter.OUTPUT_FORMAT.VOC)
                        {
                            Dictionary<string, StreamWriter> rgOutFiles = new Dictionary<string, StreamWriter>();

                            for (int c = 0; c < m_nNumClasses; c++)
                            {
                                if (c == m_nBackgroundLabelId)
                                    continue;

                                string strLabelName = m_rgLabelToName[c];
                                string strFile = getFileName(strLabelName, "txt");
                                rgOutFiles.Add(strLabelName, new StreamWriter(strFile));
                            }

                            foreach (PropertyTree pt in m_detections.Children)
                            {
                                string strLabel = pt.Get("category_id").Value;
                                if (!rgOutFiles.ContainsKey(strLabel))
                                {
                                    m_log.WriteLine("WARNING! Cannot find '" + strLabel + "' label in the output files!");
                                    continue;
                                }

                                string strImageName = pt.Get("image_id").Value;
                                float fScore = (float)pt.Get("score").Numeric;

                                List<int> bbox = new List<int>();
                                foreach (Property elm in pt.GetChildren("bbox"))
                                {
                                    bbox.Add((int)elm.Numeric);
                                }

                                string strLine = strImageName;
                                strLine += " " + fScore.ToString();
                                strLine += " " + bbox[0].ToString() + " " + bbox[1].ToString();
                                strLine += " " + (bbox[0] + bbox[2]).ToString();
                                strLine += " " + (bbox[1] + bbox[3]).ToString();
                                rgOutFiles[strLabel].WriteLine(strLine);
                            }

                            for (int c = 0; c < m_nNumClasses; c++)
                            {
                                if (c == m_nBackgroundLabelId)
                                    continue;

                                string strLabel = m_rgLabelToName[c];
                                rgOutFiles[strLabel].Flush();
                                rgOutFiles[strLabel].Close();
                                rgOutFiles[strLabel].Dispose();
                            }
                        }
                        else if (m_outputFormat == SaveOutputParameter.OUTPUT_FORMAT.COCO)
                        {
                            string strFile = getFileName("", "json");
                            using (StreamWriter sw = new StreamWriter(strFile))
                            {
                                PropertyTree output = new PropertyTree();
                                output.AddChild("detections", m_detections);
                                string strOut = output.ToJson();
                                sw.Write(strOut);
                            }
                        }
                        else if (m_outputFormat == SaveOutputParameter.OUTPUT_FORMAT.ILSVRC)
                        {
                            string strFile = getFileName("", "txt");
                            using (StreamWriter sw = new StreamWriter(strFile))
                            {
                                foreach (PropertyTree pt in m_detections.Children)
                                {
                                    int nLabel = (int)pt.Get("category_id").Numeric;
                                    string strImageName = pt.Get("image_id").Value;
                                    float fScore = (float)pt.Get("score").Numeric;

                                    List<int> bbox = new List<int>();
                                    foreach (Property elm in pt.GetChildren("bbox"))
                                    {
                                        bbox.Add((int)elm.Numeric);
                                    }

                                    string strLine = strImageName;
                                    strLine += " " + fScore.ToString();
                                    strLine += " " + bbox[0].ToString() + " " + bbox[1].ToString();
                                    strLine += " " + (bbox[0] + bbox[2]).ToString();
                                    strLine += " " + (bbox[1] + bbox[3]).ToString();
                                    sw.WriteLine(strLine);
                                }
                            }
                        }

                        m_nNameCount = 0;
                        m_detections.Clear();
                    }
                }

                if (m_bVisualize)
                {
#warning DetectionOutputLayer - does not visualize detections yet.
                    // TBD.
                }
            }
        }

        /// <summary>
        /// Does not implement.
        /// </summary>
        /// <param name="colTop">Not used.</param>
        /// <param name="rgbPropagateDown">Not used.</param>
        /// <param name="colBottom">Not Used.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            throw new NotImplementedException();
        }
    }
}
