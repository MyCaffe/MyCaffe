using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;
using MyCaffe.param.ssd;

namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// The DetectionEvaluateLayer generates the detection evaluation based on the DetectionOutputLayer and 
    /// ground truth bounding box labels.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DetectionEvaluateLayer<T> : Layer<T>
    {
        int m_nNumClasses;
        int m_nBackgroundLabelId;
        float m_fOverlapThreshold;
        bool m_bEvaluateDifficultGt;
        List<SizeF> m_rgSizes = new List<SizeF>();
        int m_nCount;
        bool m_bUseNormalizedBbox;
        ResizeParameter m_resizeParam = null;
        BBoxUtility<T> m_bboxUtil;

        /// <summary>
        /// The DetectionEvaluateLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type DETECTION_EVALUATE with parameter detection_evaluate_param.
        /// </param>
        public DetectionEvaluateLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DETECTION_EVALUATE;
            m_bboxUtil = new BBoxUtility<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
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

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: det res, gt
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
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
            m_log.CHECK_GT(m_param.detection_evaluate_param.num_classes, 1, "There must be at least one class!");
            m_nNumClasses = (int)m_param.detection_evaluate_param.num_classes;

            m_nBackgroundLabelId = (int)m_param.detection_evaluate_param.background_label_id;
            m_fOverlapThreshold = m_param.detection_evaluate_param.overlap_threshold;
            m_log.CHECK_GT(m_fOverlapThreshold, 0.0f, "The overlap_threshold must be non-negative.");

            m_bEvaluateDifficultGt = m_param.detection_evaluate_param.evaulte_difficult_gt;

            if (File.Exists(m_param.detection_evaluate_param.name_size_file))
            {
                using (StreamReader sr = new StreamReader(m_param.detection_evaluate_param.name_size_file))
                {
                    string strLine = sr.ReadLine();

                    while (strLine != null)
                    {
                        string[] rgstr = strLine.Split(' ');
                        if (rgstr.Length == 3)
                        {
                            string strName = rgstr[0];
                            int nHeight = int.Parse(rgstr[1]);
                            int nWidth = int.Parse(rgstr[2]);

                            m_rgSizes.Add(new SizeF(nWidth, nHeight));
                        }

                        strLine = sr.ReadLine();
                    }
                }
            }

            m_nCount = 0;

            // If there is no name_size_provided, use normalized bbox to evaluate.
            m_bUseNormalizedBbox = (m_rgSizes.Count == 0) ? true : false;

            // Retrieve resize parameter if there is one provided.
            m_resizeParam = m_param.detection_evaluate_param.resize_param;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_LE(m_nCount, m_rgSizes.Count, "The count must be less than or equal to the number of Sizes.");
            m_log.CHECK_EQ(colBottom[0].num, 1, "The bottom[0].num must = 1.");
            m_log.CHECK_EQ(colBottom[0].channels, 1, "The bottom[0].channels must = 1.");
            m_log.CHECK_EQ(colBottom[0].width, 7, "The bottom[0].width must = 7.");
            m_log.CHECK_EQ(colBottom[1].num, 1, "The bottom[1].num must = 1.");
            m_log.CHECK_EQ(colBottom[1].channels, 1, "The bottom[1].channels must = 1.");
            m_log.CHECK_EQ(colBottom[1].width, 8, "The bottom[1].width must = 8.");

            // num() and channels() are 1.
            List<int> rgTopShape = Utility.Create<int>(2, 1);
            int nNumPosClasses = (m_nBackgroundLabelId == -1) ? m_nNumClasses : m_nNumClasses - 1;
            int nNumValidDet = 0;
            int nOffset = 0;

            float[] rgfDetData = convertF(colBottom[0].mutable_cpu_data);
            for (int i = 0; i < colBottom[0].height; i++)
            {
                if (rgfDetData[1 + nOffset] != -1)
                    nNumValidDet++;

                nOffset += 7;
            }

            rgTopShape.Add(nNumPosClasses + nNumValidDet);

            // Each row is a 5 dimension vector, which stores
            // [image_id, label, confidence, true_pos, false_pos]
            rgTopShape.Add(5);

            colTop[0].Reshape(rgTopShape);
        }

        private int sortBboxDescending(NormalizedBBox b1, NormalizedBBox b2)
        {
            if (b1.score < b2.score)
                return 1;

            if (b1.score > b2.score)
                return -1;

            return 0;
        }

        /// <summary>
        /// Evaluate the detection output.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 2)
        ///  -# @f$ (1 \times 1 \times N \times 7) @f$ N detection results.
        ///  -# @f$ (1 \times 1 \times M \times 7) @f$ M ground truth.
        /// </param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (1 \times 1 \times N \times 4) @f$ N is the number of detections, and each row is: [image_id, label, confidence, true_pos, false_pos].
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float[] rgfDetData = convertF(colBottom[0].mutable_cpu_data);
            float[] rgfGtData = convertF(colBottom[1].mutable_cpu_data);

            // Retrieve all detection results.
            Dictionary<int, LabelBBox> rgAllDetections = m_bboxUtil.GetDetectionResults(rgfDetData, colBottom[0].height, m_nBackgroundLabelId);

            // Retrieve all ground truth (including difficult ones).
            Dictionary<int, LabelBBox> rgAllGtBboxes = m_bboxUtil.GetGroundTruthEx(rgfGtData, colBottom[1].height, m_nBackgroundLabelId, true);

            colTop[0].SetData(0);
            float[] rgfTopData = convertF(colTop[0].mutable_cpu_data);
            int nNumDet = 0;

            // Insert number of ground truth for each label.
            Dictionary<int, int> rgNumPos = new Dictionary<int, int>();
            List<KeyValuePair<int, LabelBBox>> rgAllGtBboxList = rgAllGtBboxes.ToList();

            foreach (KeyValuePair<int, LabelBBox> kv in rgAllGtBboxList)
            {
                List<KeyValuePair<int, List<NormalizedBBox>>> kvLabels = kv.Value.ToList();
                foreach (KeyValuePair<int, List<NormalizedBBox>> kvLabel in kvLabels)
                {
                    int nCount = 0;

                    if (m_bEvaluateDifficultGt)
                    {
                        nCount = kvLabel.Value.Count;
                    }
                    else
                    {
                        // Get number of non difficult ground truth.
                        for (int i = 0; i < kvLabel.Value.Count; i++)
                        {
                            if (kvLabel.Value[i].difficult)
                                nCount++;
                        }
                    }

                    if (!rgNumPos.ContainsKey(kvLabel.Key))
                        rgNumPos.Add(kvLabel.Key, nCount);
                    else
                        rgNumPos[kvLabel.Key] += nCount;
                }
            }

            for (int c = 0; c < m_nNumClasses; c++)
            {
                if (c == m_nBackgroundLabelId)
                    continue;

                rgfTopData[nNumDet * 5 + 0] = -1;
                rgfTopData[nNumDet * 5 + 1] = c;

                if (!rgNumPos.ContainsKey(c))
                    rgfTopData[nNumDet * 5 + 2] = 0;
                else
                    rgfTopData[nNumDet * 5 + 2] = rgNumPos[c];

                rgfTopData[nNumDet * 5 + 3] = -1;
                rgfTopData[nNumDet * 5 + 4] = -1;
                nNumDet++;
            }

            // Insert detection evaluate status.
            foreach (KeyValuePair<int, LabelBBox> kv in rgAllDetections)
            {
                int nImageId = kv.Key;
                LabelBBox detections = kv.Value;

                // No ground truth for current image.  All detections become false_pos.
                if (!rgAllGtBboxes.ContainsKey(nImageId))
                {
                    List<KeyValuePair<int, List<NormalizedBBox>>> kvLabels = detections.ToList();
                    foreach (KeyValuePair<int, List<NormalizedBBox>> kvLabel in kvLabels)
                    {
                        int nLabel = kvLabel.Key;
                        if (nLabel == -1)
                            continue;

                        List<NormalizedBBox> bboxes = kvLabel.Value;
                        for (int i = 0; i < bboxes.Count; i++)
                        {
                            rgfTopData[nNumDet * 5 + 0] = nImageId;
                            rgfTopData[nNumDet * 5 + 1] = nLabel;
                            rgfTopData[nNumDet * 5 + 2] = bboxes[i].score;
                            rgfTopData[nNumDet * 5 + 3] = 0;
                            rgfTopData[nNumDet * 5 + 4] = 1;
                            nNumDet++;
                        }
                    }
                }

                // Gound truth's exist for current image.
                else
                {
                    LabelBBox label_bboxes = rgAllGtBboxes[nImageId];

                    List<KeyValuePair<int, List<NormalizedBBox>>> kvLabels = detections.ToList();
                    foreach (KeyValuePair<int, List<NormalizedBBox>> kvLabel in kvLabels)
                    {
                        int nLabel = kvLabel.Key;
                        if (nLabel == -1)
                            continue;

                        List<NormalizedBBox> bboxes = kvLabel.Value;

                        // No ground truth for current label. All detectiosn become false_pos
                        if (!label_bboxes.Contains(nLabel))
                        {
                            for (int i = 0; i < bboxes.Count; i++)
                            {
                                rgfTopData[nNumDet * 5 + 0] = nImageId;
                                rgfTopData[nNumDet * 5 + 1] = nLabel;
                                rgfTopData[nNumDet * 5 + 2] = bboxes[i].score;
                                rgfTopData[nNumDet * 5 + 3] = 0;
                                rgfTopData[nNumDet * 5 + 4] = 1;
                                nNumDet++;
                            }
                        }

                        // Ground truth for current label found.
                        else
                        {
                            List<NormalizedBBox> gt_bboxes = label_bboxes[nLabel];
                            // Scale ground truth if needed.
                            if (!m_bUseNormalizedBbox)
                            {
                                m_log.CHECK_LE(m_nCount, m_rgSizes.Count, "The count must be <= the sizes count.");
                                for (int i = 0; i < gt_bboxes.Count; i++)
                                {
                                    gt_bboxes[i] = m_bboxUtil.Output(gt_bboxes[i], m_rgSizes[m_nCount], m_resizeParam);
                                }
                            }

                            List<bool> rgbVisited = Utility.Create<bool>(gt_bboxes.Count, false);

                            // Sort detections in decending order based on scores.
                            if (bboxes.Count > 1)
                                bboxes.Sort(new Comparison<NormalizedBBox>(sortBboxDescending));

                            for (int i = 0; i < bboxes.Count; i++)
                            {
                                rgfTopData[nNumDet * 5 + 0] = nImageId;
                                rgfTopData[nNumDet * 5 + 1] = nLabel;
                                rgfTopData[nNumDet * 5 + 2] = bboxes[i].score;

                                if (!m_bUseNormalizedBbox)
                                    bboxes[i] = m_bboxUtil.Output(bboxes[i], m_rgSizes[m_nCount], m_resizeParam);

                                // Compare with each ground truth bbox.
                                float fOverlapMax = -1;
                                int nJmax = -1;

                                for (int j = 0; j < gt_bboxes.Count; j++)
                                {
                                    float fOverlap = m_bboxUtil.JaccardOverlap(bboxes[i], gt_bboxes[j], m_bUseNormalizedBbox);
                                    if (fOverlap > fOverlapMax)
                                    {
                                        fOverlapMax = fOverlap;
                                        nJmax = j;
                                    }
                                }

                                if (fOverlapMax >= m_fOverlapThreshold)
                                {
                                    if (m_bEvaluateDifficultGt || (!m_bEvaluateDifficultGt && !gt_bboxes[nJmax].difficult))
                                    {
                                        // True positive.
                                        if (!rgbVisited[nJmax])
                                        {
                                            rgfTopData[nNumDet * 5 + 3] = 1;
                                            rgfTopData[nNumDet * 5 + 4] = 0;
                                            rgbVisited[nJmax] = true;
                                        }
                                        // False positive (multiple detectioN).
                                        else
                                        {
                                            rgfTopData[nNumDet * 5 + 3] = 0;
                                            rgfTopData[nNumDet * 5 + 4] = 1;
                                        }
                                    }
                                }
                                else
                                {
                                    // False positive.
                                    rgfTopData[nNumDet * 5 + 3] = 0;
                                    rgfTopData[nNumDet * 5 + 4] = 1;
                                }

                                nNumDet++;
                            }
                        }
                    }
                }

                if (m_rgSizes.Count > 0)
                {
                    m_nCount++;

                    // Reset count after a full iteration through the DB.
                    if (m_nCount == m_rgSizes.Count)
                        m_nCount = 0;
                }
            }

            colTop[0].mutable_cpu_data = convert(rgfTopData);
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
