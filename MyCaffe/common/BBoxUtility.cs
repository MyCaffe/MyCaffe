using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.param.ssd;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The BBox class processes the NormalizedBBox data used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class BBoxUtility<T>
    {
        CudaDnn<T> m_cuda;
        Log m_log;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to Cuda and cuDNN.</param>
        /// <param name="log">Specifies the output log.</param>
        public BBoxUtility(CudaDnn<T> cuda, Log log)
        {
            m_cuda = cuda;
            m_log = log;
        }

        /// <summary>
        /// Compute the average precision given true positive and false positive vectors.
        /// </summary>
        /// <param name="rgTp">Specifies a list of scores and true positive.</param>
        /// <param name="nNumPos">Specifies the number of true positives.</param>
        /// <param name="rgFp">Specifies a list of scores and false positive.</param>
        /// <param name="strApVersion">Specifies the different ways of computing the Average Precisions.
        /// @see [Tag: Average Precision](https://sanchom.wordpress.com/tag/average-precision/)
        ///   
        /// Versions:
        /// 11point: The 11-point interpolated average precision, used in VOC2007.
        /// MaxIntegral: maximally interpolated AP. Used in VOC2012/ILSVRC.
        /// Integral: the natrual integral of the precision-recall curve.
        /// <param name="rgPrec">Returns the computed precisions.</param>
        /// <param name="rgRec">Returns the computed recalls.</param>
        /// <returns>The Average Precision value is returned.</returns>
        public float ComputeAP(List<Tuple<float, int>> rgTp, int nNumPos, List<Tuple<float, int>> rgFp, string strApVersion, out List<float> rgPrec, out List<float> rgRec)
        {
            float fEps = 1e-6f;
            int nNum = rgTp.Count;

            // Make sure that rgTp and rgFp have complement values.
            for (int i = 0; i < nNum; i++)
            {
                m_log.CHECK_LE(Math.Abs(rgTp[i].Item1 - rgFp[i].Item1), fEps, "The Tp[i] - Fp[i] is less than the threshold " + fEps.ToString());
                m_log.CHECK_EQ(rgTp[i].Item2, 1 - rgFp[i].Item2, "The Tp[i].second should be one less than Fp[i].second!");
            }

            rgPrec = new List<float>();
            rgRec = new List<float>();
            float fAp = 0;

            if (rgTp.Count == 0 || nNumPos == 0)
                return fAp;

            // Compute cumsum of rgTp
            List<int> rgTpCumSum = CumSum(rgTp);
            m_log.CHECK_EQ(rgTpCumSum.Count, nNum, "The tp cumulative sum should equal the number of rgTp items (" + nNum.ToString() + ")");

            // Compute cumsum of rgFp
            List<int> rgFpCumSum = CumSum(rgFp);
            m_log.CHECK_EQ(rgFpCumSum.Count, nNum, "The fp cumulative sum should equal the number of rgFp items (" + nNum.ToString() + ")");

            // Compute precision.
            for (int i = 0; i < nNum; i++)
            {
                rgPrec.Add((float)rgTpCumSum[i] / (float)(rgTpCumSum[i] + rgFpCumSum[i]));
            }

            // Compute recall
            for (int i = 0; i < nNum; i++)
            {
                m_log.CHECK_LE(rgTpCumSum[i], nNumPos, "The Tp cumulative sum must be less than the num pos of " + nNumPos.ToString());
                rgRec.Add((float)rgTpCumSum[i] / nNumPos);
            }

            switch (strApVersion)
            {
                // VOC2007 style for computing AP
                case "11point":
                    {
                        List<float> rgMaxPrec = Utility.Create<float>(11, 0);
                        int nStartIdx = nNum - 1;

                        for (int j = 10; j >= 0; j--)
                        {
                            for (int i = nStartIdx; i >= 0; i--)
                            {
                                if (rgRec[i] < j / 10.0f)
                                {
                                    nStartIdx = i;
                                    if (j > 0)
                                        rgMaxPrec[j - 1] = rgMaxPrec[j];
                                    break;
                                }
                                else
                                {
                                    if (rgMaxPrec[j] < rgPrec[i])
                                        rgMaxPrec[j] = rgPrec[i];
                                }
                            }
                        }
                        for (int j = 10; j >= 0; j--)
                        {
                            fAp += rgMaxPrec[j] / 11.0f;
                        }
                    }
                    break;

                // VOC2012 or ILSVRC style of computing AP.
                case "MaxIntegral":
                    {
                        float fCurRec = rgRec.Last();
                        float fCurPrec = rgPrec.Last();

                        for (int i = nNum - 2; i >= 0; i--)
                        {
                            fCurPrec = Math.Max(rgPrec[i], fCurPrec);
                            float fAbsRec = Math.Abs(fCurRec - rgRec[i]);
                            if (fAbsRec > fEps)
                                fAp += fCurPrec * fAbsRec;
                            fCurRec = rgRec[i];
                        }
                        fAp += fCurRec * fCurPrec;
                    }
                    break;

                // Natural integral.
                case "Integral":
                    {
                        float fPrevRec = 0.0f;
                        for (int i = 0; i < nNum; i++)
                        {
                            float fAbsRec = Math.Abs(rgRec[i] - fPrevRec);
                            if (fAbsRec > fEps)
                                fAp += rgPrec[i] * fAbsRec;
                            fPrevRec = rgRec[i];
                        }
                    }
                    break;

                default:
                    m_log.FAIL("Unknown ap version '" + strApVersion + "'!");
                    break;
            }

            return fAp;
        }

        /// <summary>
        /// Calculate the cumulative sum of a set of pairs.
        /// </summary>
        /// <param name="rgPairs"></param>
        /// <returns></returns>
        public List<int> CumSum(List<Tuple<float, int>> rgPairs)
        {
            // Sort the pairs based on the first item of the pair.
            List<Tuple<float, int>> rgSortPairs = rgPairs.OrderByDescending(p => p.Item1).ToList();
            List<int> rgCumSum = new List<int>();

            for (int i = 0; i < rgSortPairs.Count; i++)
            {
                if (i == 0)
                    rgCumSum.Add(rgSortPairs[i].Item2);
                else
                    rgCumSum.Add(rgCumSum.Last() + rgSortPairs[i].Item2);
            }

            return rgCumSum;
        }

        /// <summary>
        /// Create the TopK ordered score list.
        /// </summary>
        /// <param name="rgScores">Specifies the scores.</param>
        /// <param name="rgIdx">Specifies the indexes.</param>
        /// <param name="nTopK">Specifies the top k items or -1 for all items.</param>
        /// <returns>The items listed by highest score is returned.</returns>
        List<Tuple<float, int>> GetTopKScoreIndex(List<float> rgScores, List<int> rgIdx, int nTopK)
        {
            List<Tuple<float, int>> rgItems = new List<Tuple<float, int>>();

            for (int i = 0; i < rgScores.Count; i++)
            {
                rgItems.Add(new Tuple<float, int>(rgScores[i], rgIdx[i]));
            }

            rgItems = rgItems.OrderByDescending(p => p.Item1).ToList();

            if (nTopK > -1)
            {
                List<Tuple<float, int>> rgItems1 = new List<Tuple<float, int>>();

                for (int i = 0; i < nTopK; i++)
                {
                    rgItems1.Add(rgItems[i]);
                }

                rgItems = rgItems1;
            }

            return rgItems;
        }

        /// <summary>
        /// Create the max ordered score list.
        /// </summary>
        /// <param name="rgScores">Specifies the scores.</param>
        /// <param name="fThreshold">Specifies the threshold of score to consider.</param>
        /// <param name="nTopK">Specifies the top k items or -1 for all items.</param>
        /// <returns>The items listed by highest score is returned.</returns>
        List<Tuple<float, int>> GetMaxScoreIndex(List<float> rgScores, float fThreshold, int nTopK)
        {
            List<Tuple<float, int>> rgItems = new List<Tuple<float, int>>();

            for (int i = 0; i < rgScores.Count; i++)
            {
                if (rgScores[i] > fThreshold)
                    rgItems.Add(new Tuple<float, int>(rgScores[i], i));
            }

            rgItems = rgItems.OrderByDescending(p => p.Item1).ToList();

            if (nTopK > -1)
            {
                List<Tuple<float, int>> rgItems1 = new List<Tuple<float, int>>();

                for (int i = 0; i < nTopK; i++)
                {
                    rgItems1.Add(rgItems[i]);
                }

                rgItems = rgItems1;
            }

            return rgItems;
        }

        /// <summary>
        /// Do a fast non maximum supression given bboxes and scores.
        /// </summary>
        /// <param name="rgBBoxes">Specifies a set of bounding boxes.</param>
        /// <param name="rgScores">Specifies a seto of corresponding confidences.</param>
        /// <param name="fScoreThreshold">Specifies the score threshold used in non maximum suppression.</param>
        /// <param name="fNmsThreshold">Specifies the nms threshold used in non maximum suppression.</param>
        /// <param name="fEta">Specifies the eta value.</param>
        /// <param name="nTopK">Specifies the top k picked indices or -1 for all.</param>
        /// <param name="rgIndices">Returns the kept indices of bboxes after nms.</param>
        public void ApplyNMSFast(List<NormalizedBBox> rgBBoxes, List<float> rgScores, float fScoreThreshold, float fNmsThreshold, float fEta, int nTopK, out List<int> rgIndices)
        {
            rgIndices = new List<int>();

            // Sanity check.
            m_log.CHECK_EQ(rgBBoxes.Count, rgScores.Count, "The number of BBoxes and scores must be the same.");

            List<Tuple<float, int>> rgScoresIndex = GetMaxScoreIndex(rgScores, fScoreThreshold, nTopK);

            // Do nms.
            float fAdaptiveThreshold = fNmsThreshold;

            while (rgScoresIndex.Count > 0)
            {
                int nIdx = rgScoresIndex[0].Item2;
                bool bKeep = true;

                for (int k = 0; k < rgIndices.Count; k++)
                {
                    if (!bKeep)
                        break;

                    int nKeptIdx = rgIndices[k];
                    float fOverlap = JaccardOverlap(rgBBoxes[nIdx], rgBBoxes[nKeptIdx]);

                    if (fOverlap <= fAdaptiveThreshold)
                        bKeep = true;
                    else
                        bKeep = false;
                }

                if (bKeep)
                    rgIndices.Add(nIdx);

                rgScoresIndex.RemoveAt(0);

                if (bKeep && fEta < 1 && fAdaptiveThreshold > 0.5f)
                    fAdaptiveThreshold *= fEta;
            }
        }

        /// <summary>
        /// Do non maximum supression given bboxes and scores.
        /// </summary>
        /// <param name="rgBBoxes">Specifies a set of bounding boxes.</param>
        /// <param name="rgScores">Specifies a seto of corresponding confidences.</param>
        /// <param name="fThreshold">Specifies the threshold used in non maximum suppression.</param>
        /// <param name="nTopK">Specifies the top k picked indices or -1 for all.</param>
        /// <param name="bReuseOverlaps">Specifies whether or not to use and update overlaps (true) or alwasy compute the overlap (false).</param>
        /// <param name="rgOverlaps">Returns the overlaps between pairs of bboxes if bReuseOverlaps is true.</param>
        /// <param name="rgIndices">Returns the kept indices of bboxes after nms.</param>
        public void ApplyNMS(List<NormalizedBBox> rgBBoxes, List<float> rgScores, float fThreshold, int nTopK, bool bReuseOverlaps, out Dictionary<int, Dictionary<int, float>> rgOverlaps, out List<int> rgIndices)
        {
            rgIndices = new List<int>();
            rgOverlaps = new Dictionary<int, Dictionary<int, float>>();

            // Sanity check.
            m_log.CHECK_EQ(rgBBoxes.Count, rgScores.Count, "The number of BBoxes and scores must be the same.");

            // Get top_k scores (with corresponding indices)
            List<int> rgIdx = new List<int>();
            for (int i = 0; i < rgScores.Count; i++)
            {
                rgIdx.Add(i);
            }

            List<Tuple<float, int>> rgScoresIndex = GetTopKScoreIndex(rgScores, rgIdx, nTopK);

            // Do nms.
            while (rgScoresIndex.Count > 0)
            {
                // Get the current highest score box.
                int nBestIdx = rgScoresIndex[0].Item2;
                NormalizedBBox best_bbox = rgBBoxes[nBestIdx];
                float fSize = Size(best_bbox);

                // Erase small box.
                if (fSize < 1e-5f)
                {
                    rgScoresIndex.RemoveAt(0);
                    continue;
                }

                rgIndices.Add(nBestIdx);

                // Erase the best box.
                rgScoresIndex.RemoveAt(0);

                // Stop if finding enough boxes for nms.
                if (nTopK > -1 && rgIndices.Count >= nTopK)
                    break;

                // Compute overlap between best_bbox and other remaining bboxes.
                // Remove a bbox if the overlap with the best_bbox is larger than nms_threshold.
                int nIdx = 0;
                while (nIdx < rgScoresIndex.Count)
                {
                    Tuple<float, int> item = rgScoresIndex[nIdx];
                    int nCurIdx = item.Item2;
                    NormalizedBBox cur_bbox = rgBBoxes[nCurIdx];
                    fSize = Size(cur_bbox);

                    if (fSize < 1e-5f)
                    {
                        rgScoresIndex.RemoveAt(nIdx);
                        continue;
                    }

                    float fCurOverlap = 0.0f;

                    if (bReuseOverlaps)
                    {
                        // Use the compute overlap
                        if (rgOverlaps.ContainsKey(nBestIdx) &&
                            rgOverlaps[nBestIdx].ContainsKey(nCurIdx))
                            fCurOverlap = rgOverlaps[nBestIdx][nCurIdx];
                        else if (rgOverlaps.ContainsKey(nCurIdx) &&
                            rgOverlaps[nCurIdx].ContainsKey(nBestIdx))
                            fCurOverlap = rgOverlaps[nCurIdx][nBestIdx];
                        else
                        {
                            fCurOverlap = JaccardOverlap(best_bbox, cur_bbox);

                            if (!rgOverlaps.ContainsKey(nBestIdx))
                            {
                                rgOverlaps.Add(nBestIdx, new Dictionary<int, float>());
                                rgOverlaps[nBestIdx].Add(nCurIdx, fCurOverlap);
                            }
                            else if (!rgOverlaps[nBestIdx].ContainsKey(nCurIdx))
                            {
                                rgOverlaps[nBestIdx].Add(nCurIdx, fCurOverlap);
                            }
                            else
                            {
                                rgOverlaps[nBestIdx][nCurIdx] = fCurOverlap;
                            }
                        }
                    }
                    else
                    {
                        fCurOverlap = JaccardOverlap(best_bbox, cur_bbox);
                    }

                    // Remove if necessary
                    if (fCurOverlap > fThreshold)
                        rgScoresIndex.RemoveAt(nIdx);
                    else
                        nIdx++;
                }
            }
        }

        /// <summary>
        /// Get detection results from rgData.
        /// </summary>
        /// <param name="rgData">Specifies a 1 x 1 x nNumDet x 7 blob data.</param>
        /// <param name="nNumDet">Specifies the number of detections.</param>
        /// <param name="nBackgroundLabelId">Specifies the label for the background class which is used to do a
        /// sanity check so that no detection contains it.</param>
        /// <returns>The detection results are returned for each class from each image.</returns>
        public Dictionary<int, LabelBBox> GetDetectionResults(float[] rgData, int nNumDet, int nBackgroundLabelId)
        {
            Dictionary<int, LabelBBox> rgAllDetections = new Dictionary<int, LabelBBox>();

            for (int i = 0; i < nNumDet; i++)
            {
                int nStartIdx = i * 7;
                int nItemId = (int)rgData[nStartIdx];
                if (nItemId == -1)
                    continue;

                int nLabel = (int)rgData[nStartIdx + 1];
                m_log.CHECK_NE(nBackgroundLabelId, nLabel, "Found background label in the detection results.");

                NormalizedBBox bbox = new NormalizedBBox(rgData[nStartIdx + 3],
                                                         rgData[nStartIdx + 4],
                                                         rgData[nStartIdx + 5],
                                                         rgData[nStartIdx + 6],
                                                         nLabel,
                                                         false,
                                                         rgData[nStartIdx + 2]);
                bbox.size = Size(bbox);

                if (!rgAllDetections.ContainsKey(nItemId))
                    rgAllDetections.Add(nItemId, new LabelBBox());

                rgAllDetections[nItemId].Add(nLabel, bbox);
            }

            return rgAllDetections;
        }

        /// <summary>
        /// Get the prior boundary boxes from the rgPriorData.
        /// </summary>
        /// <param name="rgPriorData">Specifies the prior data as a 1 x 2 x nNumPriors x 4 x 1 blob.</param>
        /// <param name="nNumPriors">Specifies the number of priors.</param>
        /// <param name="rgPriorBboxes">Specifies the prior box list in the format of NormalizedBBox.</param>
        /// <param name="rgPriorVariances">Specifies the prior variances need by prior bboxes.</param>
        public void GetPrior(float[] rgPriorData, int nNumPriors, out List<NormalizedBBox> rgPriorBboxes, out List<List<float>> rgPriorVariances)
        {
            rgPriorBboxes = new List<NormalizedBBox>();
            rgPriorVariances = new List<List<float>>();

            for (int i = 0; i < nNumPriors; i++)
            {
                int nStartIdx = i * 4;
                NormalizedBBox bbox = new NormalizedBBox(rgPriorData[nStartIdx + 0],
                                                         rgPriorData[nStartIdx + 1],
                                                         rgPriorData[nStartIdx + 2],
                                                         rgPriorData[nStartIdx + 3]);
                bbox.size = Size(bbox);
                rgPriorBboxes.Add(bbox);
            }

            for (int i = 0; i < nNumPriors; i++)
            {
                int nStartIdx = (nNumPriors + i) * 4;
                List<float> rgVariance = new List<float>();

                for (int j = 0; j < 4; j++)
                {
                    rgVariance.Add(rgPriorData[nStartIdx + j]);
                }

                rgPriorVariances.Add(rgVariance);
            }
        }

        private int getLabel(int nPredIdx, int nNumPredsPerClass, int nNumClasses, int nBackgroundLabel, Dictionary<int, List<int>> rgMatchIndices, List<NormalizedBBox> rgGtBoxes)
        {
            int nLabel = nBackgroundLabel;

            if (rgMatchIndices != null && rgMatchIndices.Count > 0 && rgGtBoxes != null && rgGtBoxes.Count > 0)
            {
                List<KeyValuePair<int, List<int>>> rgMatches = rgMatchIndices.ToList();

                foreach (KeyValuePair<int, List<int>> match in rgMatches)
                {
                    List<int> rgMatchIdx = match.Value;
                    m_log.CHECK_EQ(rgMatchIdx.Count, nNumPredsPerClass, "The match count should equal the number of predictions per class.");

                    if (rgMatchIdx[nPredIdx] > -1)
                    {
                        int nIdx = rgMatchIdx[nPredIdx];
                        m_log.CHECK_LT(nIdx, rgGtBoxes.Count, "The match index should be less than the number of ground truth boxes.");
                        nLabel = rgGtBoxes[nIdx].label;

                        m_log.CHECK_GE(nLabel, 0, "The label must be >= 0.");
                        m_log.CHECK_NE(nLabel, nBackgroundLabel, "The label cannot equal the background label.");
                        m_log.CHECK_LT(nLabel, nNumClasses, "The label must be less than the number of classes.");

                        // A prior can only be matched to one ground-truth bbox.
                        return nLabel;
                    }
                }
            }

            return nLabel;
        }

        /// <summary>
        /// Compute the confidence loss for each prior from rgConfData.
        /// </summary>
        /// <param name="rgConfData">Specifies the nNum x nNumPredsPerClass * nNumClasses blob of confidence data.</param>
        /// <param name="nNum">Specifies the number of images.</param>
        /// <param name="nNumPredsPerClass">Specifies the number of predictions per class.</param>
        /// <param name="nNumClasses">Specifies the number of classes.</param>
        /// <param name="nBackgroundLabelId">Specifies the background label.</param>
        /// <param name="loss_type">Specifies the loss type used to compute the confidence.</param>
        /// <param name="rgAllMatchIndices">Specifies all match indices storing a mapping between predictions and ground truth.</param>
        /// <param name="rgAllGtBoxes">Specifies all ground truth bboxes from the batch.</param>
        /// <returns>The confidence loss values are returned with confidence loss per location for each image.</returns>
        public List<List<float>> ComputeConfLoss(float[] rgConfData, int nNum, int nNumPredsPerClass, int nNumClasses, int nBackgroundLabelId, MultiBoxLossParameter.ConfLossType loss_type, List<Dictionary<int, List<int>>> rgAllMatchIndices = null, Dictionary<int, List<NormalizedBBox>> rgAllGtBoxes = null)
        {
            List<List<float>> rgrgConfLoss = new List<List<float>>();
            int nOffset = 0;

            for (int i = 0; i < nNum; i++)
            {
                List<float> rgConfLoss = new List<float>();

                for (int p = 0; p < nNumPredsPerClass; p++)
                {
                    int nStartIdx = p * nNumPredsPerClass;
                    // Get the label index.
                    int nLabel = getLabel(p, nNumPredsPerClass, nNumClasses, nBackgroundLabelId, (rgAllMatchIndices == null) ? null : rgAllMatchIndices[i], (rgAllGtBoxes == null) ? null : rgAllGtBoxes[i]);
                    float fLoss = 0;

                    switch (loss_type)
                    {
                        case MultiBoxLossParameter.ConfLossType.SOFTMAX:
                            {
                                m_log.CHECK_GE(nLabel, 0, "The label must be >= 0 for the SOFTMAX loss type.");
                                m_log.CHECK_LT(nLabel, nNumClasses, "The label must be < NumClasses for the SOFTMAX loss type.");
                                // Compute softmax probability.
                                // We need to subtract the max to avoid numerical issues.
                                float fMaxVal = -float.MaxValue;
                                for (int c = 0; c < nNumClasses; c++)
                                {
                                    float fVal = rgConfData[nOffset + nStartIdx + c];
                                    fMaxVal = Math.Max(fMaxVal, fVal);
                                }

                                float fSum = 0;
                                for (int c = 0; c < nNumClasses; c++)
                                {
                                    float fVal = rgConfData[nOffset + nStartIdx + c];
                                    fSum += (float)Math.Exp(fVal - fMaxVal);
                                }

                                float fValAtLabel = rgConfData[nOffset + nStartIdx + nLabel];
                                float fProb = (float)Math.Exp(fValAtLabel - fMaxVal) / fSum;
                                fLoss = (float)-Math.Log(Math.Max(fProb, float.MinValue));
                            }
                            break;

                        case MultiBoxLossParameter.ConfLossType.LOGISTIC:
                            {
                                int nTarget = 0;
                                for (int c = 0; c < nNumClasses; c++)
                                {
                                    nTarget = (c == nLabel) ? 1 : 0;
                                    float fInput = rgConfData[nOffset + nStartIdx + c];
                                    fLoss -= fInput * (nTarget - ((fInput >= 0) ? 1.0f : 0.0f)) - (float)Math.Log(1 + Math.Exp(fInput - 2 * fInput * ((fInput >= 0) ? 1.0f : 0.0f)));
                                }
                            }
                            break;

                        default:
                            m_log.FAIL("Unknown loss type '" + loss_type.ToString() + "'!");
                            break;
                    }

                    rgConfLoss.Add(fLoss);
                }

                rgrgConfLoss.Add(rgConfLoss);
                nOffset += nNumPredsPerClass * nNumClasses;
            }

            return rgrgConfLoss;
        }

        /// <summary>
        /// Calculate the confidence predictions from rgConfData.
        /// </summary>
        /// <param name="rgConfData">Specifies the nNum x nNumPredsPerClass * nNumClasses blob of confidence data.</param>
        /// <param name="nNum">Specifies the number of images.</param>
        /// <param name="nNumPredsPerClass">Specifies the number of predictions per class.</param>
        /// <param name="nNumClasses">Specifies the number of classes.</param>
        /// <returns>The confidence scores are returned as the confidence predictions which contains a confidence prediction for an image.</returns>
        public List<Dictionary<int, List<float>>> GetConfidenceScores(float[] rgConfData, int nNum, int nNumPredsPerClass, int nNumClasses)
        {
            List<Dictionary<int, List<float>>> rgConfPreds = new List<Dictionary<int, List<float>>>();
            int nOffset = 0;

            for (int i = 0; i < nNum; i++)
            {
                Dictionary<int, List<float>> rgLabelScores = new Dictionary<int, List<float>>();

                for (int p = 0; p < nNumPredsPerClass; p++)
                {
                    int nStartIdx = p * nNumClasses;

                    for (int c = 0; c < nNumClasses; c++)
                    {
                        float fConf = rgConfData[nOffset + nStartIdx + c];

                        if (!rgLabelScores.ContainsKey(c))
                            rgLabelScores.Add(c, new List<float>());

                        rgLabelScores[c].Add(fConf);
                    }
                }

                rgConfPreds.Add(rgLabelScores);
                nOffset += nNumPredsPerClass * nNumClasses;
            }

            return rgConfPreds;
        }

        /// <summary>
        /// Create a set of local predictions from the rgLocData.
        /// </summary>
        /// <param name="rgLocData">Specifies the nNum x nNumPredsPerClass * nNumLocClasses * 4 blbo with prediction initialization data.</param>
        /// <param name="nNum">Specifies the number of images.</param>
        /// <param name="nNumPredsPerClass">Specifies the number of predictions per class.</param>
        /// <param name="nNumLocClasses">Specifies the number of local classes  It is 1 if bShareLocation is true; and it is equal to the number
        /// of classes needed to predict otherwise.</param>
        /// <param name="bShareLocation">Specifies whether or not to share the location.  If true, all classes share the same location prediction.</param>
        /// <returns>A list of created location predictions is returned as a list of LabelBBox items where each item contains a location prediction
        /// for the image.</returns>
        public List<LabelBBox> GetLocPredictions(float[] rgLocData, int nNum, int nNumPredsPerClass, int nNumLocClasses, bool bShareLocation)
        {
            List<LabelBBox> rgLocPreds = new List<LabelBBox>();

            if (bShareLocation)
                m_log.CHECK_EQ(nNumLocClasses, 1, "When shareing locations, the nNumLocClasses must be 1.");

            int nOffset = 0;

            for (int i = 0; i < nNum; i++)
            {
                LabelBBox labelBbox = new LabelBBox();

                for (int p = 0; p < nNumPredsPerClass; p++)
                {
                    int nStartIdx = p * nNumLocClasses * 4;

                    for (int c = 0; c < nNumLocClasses; c++)
                    {
                        int nLabel = (bShareLocation) ? -1 : c;
                        labelBbox[nLabel].Add(new NormalizedBBox(rgLocData[nStartIdx + nOffset + c * 4 + 0],
                                                                 rgLocData[nStartIdx + nOffset + c * 4 + 1],
                                                                 rgLocData[nStartIdx + nOffset + c * 4 + 2],
                                                                 rgLocData[nStartIdx + nOffset + c * 4 + 3]));
                    }
                }

                nOffset += nNumPredsPerClass * nNumLocClasses * 4;
                rgLocPreds.Add(labelBbox);
            }

            return rgLocPreds;
        }

        /// <summary>
        /// Create a set of ground truth bounding boxes from the rgGtData.
        /// </summary>
        /// <param name="rgGtData">Specifies the 1 x 1 x nNumGt x 7 blob with ground truth initialization data.</param>
        /// <param name="nNumGt">Specifies the number of ground truths.</param>
        /// <param name="nBackgroundLabelId">Specifies the background label.</param>
        /// <param name="bUseDifficultGt">Specifies whether or not to use the difficult ground truth.</param>
        /// <returns>A dictionary containing the ground truth's (one per image) is returned with the label of each bbox stored in the NormalizedBBox.</returns>
        public Dictionary<int, List<NormalizedBBox>> GetGroundTruth(float[] rgGtData, int nNumGt, int nBackgroundLabelId, bool bUseDifficultGt)
        {
            Dictionary<int, List<NormalizedBBox>> rgAllGt = new Dictionary<int, List<NormalizedBBox>>();

            for (int i = 0; i < nNumGt; i++)
            {
                int nStartIdx = i * 8;
                int nItemId = (int)rgGtData[nStartIdx];
                if (nItemId == -1)
                    continue;

                int nLabel = (int)rgGtData[nStartIdx + 1];
                m_log.CHECK_NE(nBackgroundLabelId, nLabel, "Found the background label in the dataset!");

                bool bDifficult = (rgGtData[nStartIdx + 7] == 0) ? false : true;
                // Skip reading the difficult ground truth.
                if (!bUseDifficultGt && bDifficult)
                    continue;

                NormalizedBBox bbox = new NormalizedBBox(rgGtData[nStartIdx + 3],
                                                         rgGtData[nStartIdx + 4],
                                                         rgGtData[nStartIdx + 5],
                                                         rgGtData[nStartIdx + 6],
                                                         nLabel,
                                                         bDifficult);
                bbox.size = Size(bbox);

                if (!rgAllGt.ContainsKey(nItemId))
                    rgAllGt.Add(nItemId, new List<NormalizedBBox>());

                rgAllGt[nItemId].Add(bbox);
            }

            return rgAllGt;
        }

        /// <summary>
        /// Find matches between a list of two bounding boxes.
        /// </summary>
        /// <param name="rgGtBboxes">Specifies a list of ground truth bounding boxes.</param>
        /// <param name="rgPredBboxes">Specifies a list of predicted bounding boxes.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="match_type">Specifies the matching type.</param>
        /// <param name="fOverlapThreshold">Specifies the overlap.</param>
        /// <param name="bIgnoreCrossBoundaryBbox">Specifies whether or not to ignore corss boundary bounding boxes.</param>
        /// <param name="rgMatchIndices">Specifies the list where the indexes of matches are placed.</param>
        /// <param name="rgMatchOverlaps">Specifies the list where the overlaps of matches are placed.</param>
        public void Match(List<NormalizedBBox> rgGtBboxes, List<NormalizedBBox> rgPredBboxes, int nLabel, MultiBoxLossParameter.MatchType match_type, float fOverlapThreshold, bool bIgnoreCrossBoundaryBbox, out List<int> rgMatchIndices, out List<float> rgMatchOverlaps)
        {
            int nNumPred = rgPredBboxes.Count;
            rgMatchIndices = Utility.Create<int>(nNumPred, -1);
            rgMatchOverlaps = Utility.Create<float>(nNumPred, 0);

            int nNumGt = 0;
            List<int> rgGtIndices = new List<int>();

            // label -1 means comparing against all ground truth.
            if (nLabel == -1)
            {
                nNumGt = rgGtBboxes.Count;
                for (int i = 0; i < nNumGt; i++)
                {
                    rgGtIndices.Add(i);
                }
            }

            // Otherwise match gt boxes with the specified label.
            else
            {
                for (int i = 0; i < rgGtBboxes.Count; i++)
                {
                    if (rgGtBboxes[i].label == nLabel)
                    {
                        nNumGt++;
                        rgGtIndices.Add(i);
                    }
                }
            }

            if (nNumGt == 0)
                return;

            // Store the positive overlap between predictions and ground truth.
            Dictionary<int, Dictionary<int, float>> rgOverlaps = new Dictionary<int, Dictionary<int, float>>();
            for (int i = 0; i < nNumPred; i++)
            {
                rgOverlaps.Add(i, new Dictionary<int, float>());

                if (bIgnoreCrossBoundaryBbox && IsCrossBoundary(rgPredBboxes[i]))
                {
                    rgMatchIndices.Add(-2);
                    continue;
                }

                for (int j = 0; j < nNumGt; j++)
                {
                    float fOverlap = JaccardOverlap(rgPredBboxes[i], rgGtBboxes[rgGtIndices[j]]);
                    if (fOverlap > 1e-6f)
                    {
                        rgMatchOverlaps[i] = Math.Max(rgMatchOverlaps[i], fOverlap);
                        rgOverlaps[i].Add(j, fOverlap);
                    }
                }
            }

            // Bipartite matching.
            List<int> rgGtPool = new List<int>();
            for (int i = 0; i < nNumGt; i++)
            {
                rgGtPool.Add(i);
            }

            // Find the most overlapped gt and corresponding predictions.
            while (rgGtPool.Count > 0)
            {
                int nMaxIdx = -1;
                int nMaxGtIdx = -1;
                float fMaxOverlap = -1;

                foreach (KeyValuePair<int, Dictionary<int, float>> kv in rgOverlaps)
                {
                    int i = kv.Key;

                    // The prediction already has match ground truth or is ignored.
                    if (rgMatchIndices[i] != -1)
                        continue;

                    for (int p = 0; p < rgGtPool.Count; p++)
                    {
                        int j = rgGtPool[p];

                        // No overlap between the i'th prediction and j'th ground truth.
                        if (!kv.Value.ContainsKey(j))
                            continue;

                        // Find the maximum overlap pair.
                        if (kv.Value[j] > fMaxOverlap)
                        {
                            // If the prediction has not been matched to any ground truth,
                            // and the overlap is larger than the maximum overlap, update.
                            nMaxIdx = i;
                            nMaxGtIdx = j;
                            fMaxOverlap = kv.Value[j];
                        }
                    }
                }

                // Cannot find a good match.
                if (nMaxIdx == -1)
                {
                    break;
                }
                else
                {
                    m_log.CHECK_EQ(rgMatchIndices[nMaxIdx], -1, "The match index at index=" + nMaxIdx.ToString() + " should be -1.");
                    rgMatchIndices[nMaxIdx] = rgGtIndices[nMaxGtIdx];
                    rgMatchOverlaps[nMaxIdx] = fMaxOverlap;

                    // Remove the ground truth.
                    rgGtPool.Remove(nMaxGtIdx);
                }
            }

            // Do the matching
            switch (match_type)
            {
                case MultiBoxLossParameter.MatchType.BIPARTITE:
                    // Already done.
                    break;

                case MultiBoxLossParameter.MatchType.PER_PREDICTION:
                    // Get most overlapped for the rest of the prediction bboxes.
                    foreach (KeyValuePair<int, Dictionary<int, float>> kv in rgOverlaps)
                    {
                        int i = kv.Key;

                        // The prediction already has matched ground truth or is ignored.
                        if (rgMatchIndices[i] != -1)
                            continue;

                        int nMaxGtIdx = -1;
                        float fMaxOverlap = -1;

                        for (int j = 0; j < nNumGt; j++)
                        {
                            // No overlap between the i'th prediction and j'th ground truth.
                            if (!kv.Value.ContainsKey(j))
                                continue;

                            // Find the maximum overlapped pair.
                            float fOverlap = kv.Value[j];

                            // If the prediction has not been matched on any ground truth,
                            // and the overlap is larger than the maximum overlap, update.
                            if (fOverlap >= fOverlapThreshold && fOverlap > fMaxOverlap)
                            {
                                nMaxGtIdx = j;
                                fMaxOverlap = fOverlap;
                            }
                        }

                        // Found a matched ground truth.
                        if (nMaxGtIdx != -1)
                        {
                            m_log.CHECK_EQ(rgMatchIndices[i], -1, "The match index at index=" + i.ToString() + " should be -1.");
                            rgMatchIndices[i] = rgGtIndices[nMaxGtIdx];
                            rgMatchOverlaps[i] = fMaxOverlap;
                        }
                    }
                    break;

                default:
                    m_log.FAIL("Unknown matching type '" + match_type.ToString() + "'!");
                    break;
            }
        }

        /// <summary>
        /// Returns whether or not the bbox is overlaps outside the range [0,1]
        /// </summary>
        /// <param name="bbox">Specifies the bounding box to test.</param>
        /// <returns>If the bbox overlaps, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool IsCrossBoundary(NormalizedBBox bbox)
        {
            if (bbox.xmin < 0 || bbox.xmin > 1)
                return true;

            if (bbox.ymin < 0 || bbox.ymin > 1)
                return true;

            if (bbox.xmax < 0 || bbox.xmax > 1)
                return true;

            if (bbox.ymax < 0 || bbox.ymax > 1)
                return true;

            return false;
        }

        /// <summary>
        /// Decode a set of bounding box.
        /// </summary>
        /// <param name="rgPriorBbox">Specifies an list of prior bounding boxs.</param>
        /// <param name="rgrgfPriorVariance">Specifies the list of prior variance (must have 4 elements each of which are > 0).</param>
        /// <param name="code_type">Specifies the code type.</param>
        /// <param name="bEncodeVarianceInTarget">Specifies whether or not to encode the variance in the target.</param>
        /// <param name="rgBbox">Specifies a list of bounding boxs.</param>
        /// <returns>A list of decoded bounding box is returned.</returns>
        public List<NormalizedBBox> Decode(List<NormalizedBBox> rgPriorBbox, List<List<float>> rgrgfPriorVariance, PriorBoxParameter.CodeType code_type, bool bEncodeVarianceInTarget, bool bClip, List<NormalizedBBox> rgBbox)
        {
            m_log.CHECK_EQ(rgPriorBbox.Count, rgrgfPriorVariance.Count, "The number of prior boxes must match the number of variance lists.");
            m_log.CHECK_EQ(rgPriorBbox.Count, rgBbox.Count, "The number of prior boxes must match the number of boxes.");
            int nNumBoxes = rgPriorBbox.Count;

            if (nNumBoxes >= 1)
                m_log.CHECK_EQ(rgrgfPriorVariance[0].Count, 4, "The variance lists must have 4 items.");

            List<NormalizedBBox> rgDecodeBoxes = new List<NormalizedBBox>();

            for (int i = 0; i < nNumBoxes; i++)
            {
                NormalizedBBox decode_box = Decode(rgPriorBbox[i], rgrgfPriorVariance[i], code_type, bEncodeVarianceInTarget, bClip, rgBbox[i]);
                rgDecodeBoxes.Add(decode_box);
            }

            return rgDecodeBoxes;
        }

        /// <summary>
        /// Decode a bounding box.
        /// </summary>
        /// <param name="prior_bbox">Specifies the prior bounding box.</param>
        /// <param name="rgfPriorVariance">Specifies the prior variance (must have 4 elements each of which are > 0).</param>
        /// <param name="code_type">Specifies the code type.</param>
        /// <param name="bEncodeVarianceInTarget">Specifies whether or not to encode the variance in the target.</param>
        /// <param name="bClip">Specifies whether or not to enable clip or not.</param>
        /// <param name="bbox">Specifies the bounding box.</param>
        /// <returns>The decoded bounding box is returned.</returns>
        public NormalizedBBox Decode(NormalizedBBox prior_bbox, List<float> rgfPriorVariance, PriorBoxParameter.CodeType code_type, bool bEncodeVarianceInTarget, bool bClip, NormalizedBBox bbox)
        {
            NormalizedBBox decode_bbox;

            switch (code_type)
            {
                case PriorBoxParameter.CodeType.CORNER:
                    if (bEncodeVarianceInTarget)
                    {
                        // Variance is encoded in target, we simply need to add the offset predictions.
                        decode_bbox = new NormalizedBBox(prior_bbox.xmin + bbox.xmin,
                                                         prior_bbox.ymin + bbox.ymin,
                                                         prior_bbox.xmax + bbox.xmax,
                                                         prior_bbox.ymax + bbox.ymax);
                    }
                    else
                    {
                        // Variance is encoded in the bbox, we need to scale the offset accordingly.
                        m_log.CHECK_EQ(rgfPriorVariance.Count, 4, "The variance must have 4 values!");
                        foreach (float fVar in rgfPriorVariance)
                        {
                            m_log.CHECK_GT(fVar, 0, "Each variance must be greater than 0.");
                        }

                        decode_bbox = new NormalizedBBox(prior_bbox.xmin + rgfPriorVariance[0] * bbox.xmin,
                                                         prior_bbox.ymin + rgfPriorVariance[1] * bbox.ymin,
                                                         prior_bbox.xmax + rgfPriorVariance[2] * bbox.xmax,
                                                         prior_bbox.ymax + rgfPriorVariance[3] * bbox.ymax);
                    }
                    break;

                case PriorBoxParameter.CodeType.CENTER_SIZE:
                    {
                        float fPriorWidth = prior_bbox.xmax - prior_bbox.xmin;
                        m_log.CHECK_GT(fPriorWidth, 0, "The prior width must be greater than zero.");
                        float fPriorHeight = prior_bbox.ymax - prior_bbox.ymin;
                        m_log.CHECK_GT(fPriorHeight, 0, "The prior height must be greater than zero.");
                        float fPriorCenterX = (prior_bbox.xmin + prior_bbox.xmax) / 2;
                        float fPriorCenterY = (prior_bbox.ymin + prior_bbox.ymax) / 2;

                        float fDecodeBboxCenterX;
                        float fDecodeBboxCenterY;
                        float fDecodeBboxWidth;
                        float fDecodeBboxHeight;

                        if (bEncodeVarianceInTarget)
                        {
                            // Variance is encoded in target, we simply need to resote the offset prdedictions.
                            fDecodeBboxCenterX = bbox.xmin * fPriorWidth + fPriorCenterX;
                            fDecodeBboxCenterY = bbox.ymin * fPriorHeight + fPriorCenterY;
                            fDecodeBboxWidth = (float)Math.Exp(bbox.xmax) * fPriorWidth;
                            fDecodeBboxHeight = (float)Math.Exp(bbox.ymax) * fPriorHeight;
                        }
                        else
                        {
                            // Variance is encoded in the bbox, we need to scale the offset accordingly.
                            fDecodeBboxCenterX = rgfPriorVariance[0] * bbox.xmin * fPriorWidth + fPriorCenterX;
                            fDecodeBboxCenterY = rgfPriorVariance[1] * bbox.ymin * fPriorHeight + fPriorCenterY;
                            fDecodeBboxWidth = (float)Math.Exp(rgfPriorVariance[2] * bbox.xmax) * fPriorWidth;
                            fDecodeBboxHeight = (float)Math.Exp(rgfPriorVariance[3] * bbox.ymax) * fPriorHeight;
                        }

                        decode_bbox = new NormalizedBBox(fDecodeBboxCenterX - fDecodeBboxWidth / 2,
                                                         fDecodeBboxCenterY - fDecodeBboxHeight / 2,
                                                         fDecodeBboxCenterX + fDecodeBboxWidth / 2,
                                                         fDecodeBboxCenterY + fDecodeBboxHeight / 2);
                    }
                    break;

                case PriorBoxParameter.CodeType.CORNER_SIZE:
                    {
                        float fPriorWidth = prior_bbox.xmax - prior_bbox.xmin;
                        m_log.CHECK_GT(fPriorWidth, 0, "The prior width must be greater than zero.");
                        float fPriorHeight = prior_bbox.ymax - prior_bbox.ymin;
                        m_log.CHECK_GT(fPriorHeight, 0, "The prior height must be greater than zero.");

                        if (bEncodeVarianceInTarget)
                        {
                            // Variance is encoded in target, we simply need to add the offset predictions.
                            decode_bbox = new NormalizedBBox(prior_bbox.xmin + bbox.xmin * fPriorWidth,
                                                             prior_bbox.ymin + bbox.ymin * fPriorHeight,
                                                             prior_bbox.xmax + bbox.xmax * fPriorWidth,
                                                             prior_bbox.ymax + bbox.ymax * fPriorHeight);
                        }
                        else
                        {
                            // Encode variance in bbox.
                            m_log.CHECK_EQ(rgfPriorVariance.Count, 4, "The variance must have 4 values!");
                            foreach (float fVar in rgfPriorVariance)
                            {
                                m_log.CHECK_GT(fVar, 0, "Each variance must be greater than 0.");
                            }

                            decode_bbox = new NormalizedBBox(prior_bbox.xmin + rgfPriorVariance[0] * bbox.xmin * fPriorWidth,
                                                             prior_bbox.ymin + rgfPriorVariance[1] * bbox.ymin * fPriorHeight,
                                                             prior_bbox.xmax + rgfPriorVariance[2] * bbox.xmax * fPriorWidth,
                                                             prior_bbox.ymax + rgfPriorVariance[3] * bbox.ymax * fPriorHeight);
                        }
                    }
                    break;

                default:
                    m_log.FAIL("Unknown code type '" + code_type.ToString());
                    return null;
            }

            decode_bbox.size = Size(decode_bbox);
            if (bClip)
                decode_bbox = Clip(decode_bbox);

            return decode_bbox;
        }

        /// <summary>
        /// Encode a bounding box.
        /// </summary>
        /// <param name="prior_bbox">Specifies the prior bounding box.</param>
        /// <param name="rgfPriorVariance">Specifies the prior variance (must have 4 elements each of which are > 0).</param>
        /// <param name="code_type">Specifies the code type.</param>
        /// <param name="bEncodeVarianceInTarget">Specifies whether or not to encode the variance in the target.</param>
        /// <param name="bbox">Specifies the bounding box.</param>
        /// <returns>The encoded bounding box is returned.</returns>
        public NormalizedBBox Encode(NormalizedBBox prior_bbox, List<float> rgfPriorVariance, PriorBoxParameter.CodeType code_type, bool bEncodeVarianceInTarget, NormalizedBBox bbox)
        {
            NormalizedBBox encode_bbox;

            switch (code_type)
            {
                case PriorBoxParameter.CodeType.CORNER:
                    if (bEncodeVarianceInTarget)
                    {
                        encode_bbox = new NormalizedBBox(bbox.xmin - prior_bbox.xmin,
                                                         bbox.ymin - prior_bbox.ymin,
                                                         bbox.xmax - prior_bbox.xmax,
                                                         bbox.ymax - prior_bbox.ymax);
                    }
                    else
                    {
                        // Encode variance in bbox.
                        m_log.CHECK_EQ(rgfPriorVariance.Count, 4, "The variance must have 4 values!");
                        foreach (float fVar in rgfPriorVariance)
                        {
                            m_log.CHECK_GT(fVar, 0, "Each variance must be greater than 0.");
                        }

                        encode_bbox = new NormalizedBBox((bbox.xmin - prior_bbox.xmin) / rgfPriorVariance[0],
                                                         (bbox.ymin - prior_bbox.ymin) / rgfPriorVariance[1],
                                                         (bbox.xmax - prior_bbox.xmax) / rgfPriorVariance[2],
                                                         (bbox.ymax - prior_bbox.ymax) / rgfPriorVariance[3]);
                    }
                    break;

                case PriorBoxParameter.CodeType.CENTER_SIZE:
                    {
                        float fPriorWidth = prior_bbox.xmax - prior_bbox.xmin;
                        m_log.CHECK_GT(fPriorWidth, 0, "The prior width must be greater than zero.");
                        float fPriorHeight = prior_bbox.ymax - prior_bbox.ymin;
                        m_log.CHECK_GT(fPriorHeight, 0, "The prior height must be greater than zero.");
                        float fPriorCenterX = (prior_bbox.xmin + prior_bbox.xmax) / 2;
                        float fPriorCenterY = (prior_bbox.ymin + prior_bbox.ymax) / 2;

                        float fBboxWidth = bbox.xmax - bbox.xmin;
                        m_log.CHECK_GT(fBboxWidth, 0, "The bbox width must be greater than zero.");
                        float fBboxHeight = bbox.ymax - bbox.ymin;
                        m_log.CHECK_GT(fBboxHeight, 0, "The bbox height must be greater than zero.");
                        float fBboxCenterX = (bbox.xmin + bbox.xmax) / 2;
                        float fBboxCenterY = (bbox.ymin + bbox.ymax) / 2;

                        if (bEncodeVarianceInTarget)
                        {
                            encode_bbox = new NormalizedBBox((fBboxCenterX - fPriorCenterX) / fPriorWidth,
                                                             (fBboxCenterY - fPriorCenterY) / fPriorHeight,
                                                             (float)Math.Log(fBboxWidth / fPriorWidth),
                                                             (float)Math.Log(fBboxHeight / fPriorHeight));
                        }
                        else
                        {
                            // Encode variance in bbox.
                            m_log.CHECK_EQ(rgfPriorVariance.Count, 4, "The variance must have 4 values!");
                            foreach (float fVar in rgfPriorVariance)
                            {
                                m_log.CHECK_GT(fVar, 0, "Each variance must be greater than 0.");
                            }

                            encode_bbox = new NormalizedBBox((fBboxCenterX - fPriorCenterX) / fPriorWidth / rgfPriorVariance[0],
                                                             (fBboxCenterY - fPriorCenterY) / fPriorHeight / rgfPriorVariance[1],
                                                             (float)Math.Log(fBboxWidth / fPriorWidth) / rgfPriorVariance[2],
                                                             (float)Math.Log(fBboxHeight / fPriorHeight) / rgfPriorVariance[3]);
                        }
                    }
                    break;

                case PriorBoxParameter.CodeType.CORNER_SIZE:
                    {
                        float fPriorWidth = prior_bbox.xmax - prior_bbox.xmin;
                        m_log.CHECK_GT(fPriorWidth, 0, "The prior width must be greater than zero.");
                        float fPriorHeight = prior_bbox.ymax - prior_bbox.ymin;
                        m_log.CHECK_GT(fPriorHeight, 0, "The prior height must be greater than zero.");
                        float fPriorCenterX = (prior_bbox.xmin + prior_bbox.xmax) / 2;
                        float fPriorCenterY = (prior_bbox.ymin + prior_bbox.ymax) / 2;

                        if (bEncodeVarianceInTarget)
                        {
                            encode_bbox = new NormalizedBBox((bbox.xmin - prior_bbox.xmin) / fPriorWidth,
                                                             (bbox.ymin - prior_bbox.ymin) / fPriorHeight,
                                                             (bbox.xmax - prior_bbox.xmax) / fPriorWidth,
                                                             (bbox.ymax - prior_bbox.ymax) / fPriorHeight);
                        }
                        else
                        {
                            // Encode variance in bbox.
                            m_log.CHECK_EQ(rgfPriorVariance.Count, 4, "The variance must have 4 values!");
                            foreach (float fVar in rgfPriorVariance)
                            {
                                m_log.CHECK_GT(fVar, 0, "Each variance must be greater than 0.");
                            }

                            encode_bbox = new NormalizedBBox((bbox.xmin - prior_bbox.xmin) / fPriorWidth / rgfPriorVariance[0],
                                                             (bbox.ymin - prior_bbox.ymin) / fPriorHeight / rgfPriorVariance[1],
                                                             (bbox.xmax - prior_bbox.xmax) / fPriorWidth / rgfPriorVariance[2],
                                                             (bbox.ymax - prior_bbox.ymax) / fPriorHeight / rgfPriorVariance[3]);
                        }
                    }
                    break;

                default:
                    m_log.FAIL("Unknown code type '" + code_type.ToString());
                    return null;
            }

            return encode_bbox;
        }

        /// <summary>
        /// Calculates the Jaccard overlap between two bounding boxes.
        /// </summary>
        /// <param name="bbox1">Specifies the first bounding box.</param>
        /// <param name="bbox2">Specifies the second bounding box.</param>
        /// <param name="bNormalized">Specifies whether or not the bboxes are normalized or not.</param>
        /// <returns>The Jaccard overlap is returned.</returns>
        public float JaccardOverlap(NormalizedBBox bbox1, NormalizedBBox bbox2, bool bNormalized = true)
        {
            NormalizedBBox intersect_bbox = Intersect(bbox1, bbox2);
            float fIntersectWidth = intersect_bbox.xmax - intersect_bbox.xmin;
            float fIntersectHeight = intersect_bbox.ymax - intersect_bbox.ymin;

            if (!bNormalized)
            {
                fIntersectWidth += 1;
                fIntersectHeight += 1;
            }

            if (fIntersectWidth > 0 && fIntersectHeight > 0)
            {
                float fIntersectSize = fIntersectWidth * fIntersectHeight;
                float fBbox1Size = Size(bbox1);
                float fBbox2Size = Size(bbox2);
                return fIntersectSize / (fBbox1Size + fBbox2Size - fIntersectSize);
            }

            return 0;
        }

        public NormalizedBBox Output(NormalizedBBox bbox, SizeF szImg, ResizeParameter p)
        {
            int height = (int)szImg.Height;
            int width = (int)szImg.Width;
            NormalizedBBox temp_bbox = bbox.Clone();

            if (p != null)
            {
                float fResizeHeight = p.height;
                float fResizeWidth = p.width;
                float fResizeAspect = fResizeWidth / fResizeHeight;
                int nHeightScale = (int)p.height_scale;
                int nWidthScale = (int)p.width_scale;
                float fAspect = (float)width / (float)height;
                float fPadding;

                switch (p.resize_mode)
                {
                    case ResizeParameter.ResizeMode.WARP:
                        temp_bbox = Clip(temp_bbox);
                        return Scale(temp_bbox, height, width);

                    case ResizeParameter.ResizeMode.FIT_LARGE_SIZE_AND_PAD:
                        float fxmin = 0.0f;
                        float fymin = 0.0f;
                        float fxmax = 1.0f;
                        float fymax = 1.0f;

                        if (fAspect > fResizeAspect)
                        {
                            fPadding = (fResizeHeight - fResizeWidth / fAspect) / 2;
                            fymin = fPadding / fResizeHeight;
                            fymax = 1.0f - fPadding / fResizeHeight;
                        }
                        else
                        {
                            fPadding = (fResizeWidth - fResizeHeight * fAspect) / 2;
                            fxmin = fPadding / fResizeWidth;
                            fxmax = 1.0f - fPadding / fResizeWidth;
                        }

                        Project(new NormalizedBBox(fxmin, fymin, fxmax, fymax), bbox, out temp_bbox);
                        temp_bbox = Clip(temp_bbox);
                        return Scale(temp_bbox, height, width);

                    case ResizeParameter.ResizeMode.FIT_SMALL_SIZE:
                        if (nHeightScale == 0 || nWidthScale == 0)
                        {
                            temp_bbox = Clip(temp_bbox);
                            return Scale(temp_bbox, height, width);
                        }
                        else
                        {
                            temp_bbox = Scale(temp_bbox, nHeightScale, nWidthScale);
                            return Clip(temp_bbox, height, width);
                        }

                    default:
                        m_log.FAIL("Unknown resize mode '" + p.resize_mode.ToString() + "'!");
                        return null;
                }
            }
            else
            {
                // Clip the normalized bbox first.
                temp_bbox = Clip(temp_bbox);
                // Scale the bbox according to the original image size.
                return Scale(temp_bbox, height, width);
            }
        }

        /// <summary>
        /// Project one bbox onto another.
        /// </summary>
        /// <param name="src">Specifies the source bbox.</param>
        /// <param name="bbox">Specifies the second bbox.</param>
        /// <param name="proj_bbox">Returns the projected bbox here.</param>
        /// <returns>The new project bbox is returned if a projection was made, otherwise the original bbox is returned.</returns>
        public bool Project(NormalizedBBox src, NormalizedBBox bbox, out NormalizedBBox proj_bbox)
        {
            proj_bbox = bbox.Clone();

            if (bbox.xmin >= src.xmax || bbox.xmax <= src.xmin ||
                bbox.ymin >= src.ymax || bbox.ymax <= src.ymin)
                return false;

            float src_width = src.xmax - src.xmin;
            float src_height = src.ymax - src.ymin;
            proj_bbox = new NormalizedBBox((bbox.xmin - src.xmin) / src_width,
                                           (bbox.ymin - src.ymin) / src_height,
                                           (bbox.xmax - src.xmin) / src_width,
                                           (bbox.ymax - src.ymin) / src_height,
                                           bbox.label, bbox.difficult);
            proj_bbox = Clip(proj_bbox);

            float fSize = Size(proj_bbox);
            if (fSize > 0)
                return true;

            return false;
        }

        /// <summary>
        /// Create the intersection of two bounding boxes.
        /// </summary>
        /// <param name="bbox1">Specifies the first bounding box.</param>
        /// <param name="bbox2">Specifies the second bounding box.</param>
        /// <returns>The intersection of the two bounding boxes is returned.</returns>
        public NormalizedBBox Intersect(NormalizedBBox bbox1, NormalizedBBox bbox2)
        {
            // Return [0,0,0,0] if there is no intersection.
            if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
                bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin)
                return new NormalizedBBox(0.0f, 0.0f, 0.0f, 0.0f);

            return new NormalizedBBox(Math.Max(bbox1.xmin, bbox2.xmin),
                                      Math.Max(bbox1.ymin, bbox2.ymin),
                                      Math.Min(bbox1.xmax, bbox2.xmax),
                                      Math.Min(bbox1.ymax, bbox2.ymax));
        }

        /// <summary>
        /// Clip the BBox to a set range.
        /// </summary>
        /// <param name="bbox">Specifies the input bounding box.</param>
        /// <param name="fHeight">Specifies the clipping height.</param>
        /// <param name="fWidth">Specifies the clipping width.</param>
        /// <returns>A new, clipped NormalizedBBox is returned.</returns>
        public NormalizedBBox Clip(NormalizedBBox bbox, float fHeight = 1.0f, float fWidth = 1.0f)
        {
            NormalizedBBox clipped = bbox.Clone();
            clipped.xmin = Math.Max(Math.Min(bbox.xmin, fWidth), 0.0f);
            clipped.ymin = Math.Max(Math.Min(bbox.ymin, fHeight), 0.0f);
            clipped.xmax = Math.Max(Math.Min(bbox.xmax, fWidth), 0.0f);
            clipped.ymax = Math.Max(Math.Min(bbox.ymax, fHeight), 0.0f);
            clipped.size = Size(clipped);
            return clipped;
        }

        /// <summary>
        /// Scale the BBox to a set range.
        /// </summary>
        /// <param name="bbox">Specifies the input bounding box.</param>
        /// <param name="nHeight">Specifies the scaling height.</param>
        /// <param name="nWidth">Specifies the scaling width.</param>
        /// <returns>A new, scaled NormalizedBBox is returned.</returns>
        public NormalizedBBox Scale(NormalizedBBox bbox, int nHeight, int nWidth)
        {
            NormalizedBBox scaled = bbox.Clone();
            scaled.xmin = bbox.xmin * nWidth;
            scaled.ymin = bbox.ymin * nHeight;
            scaled.xmax = bbox.xmax * nWidth;
            scaled.ymax = bbox.ymax * nHeight;
            bool bNormalized = !(nWidth > 1 || nHeight > 1);
            scaled.size = Size(scaled, bNormalized);
            return scaled;
        }

        /// <summary>
        /// Calculate the size of a BBox.
        /// </summary>
        /// <param name="bbox">Specifies the input bounding box.</param>
        /// <param name="bNormalized">Specifies whether or not the bounding box falls within the range [0,1].</param>
        /// <returns>The size of the bounding box is returned.</returns>
        public float Size(NormalizedBBox bbox, bool bNormalized = true)
        {
            if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
            {
                // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0
                return 0f;
            }

            float fWidth = bbox.xmax - bbox.xmin;
            float fHeight = bbox.ymax - bbox.ymin;

            if (bNormalized)
                return fWidth * fHeight;
            else // bbox is not in range [0,1]
                return (fWidth + 1) * (fHeight + 1);
        }
    }
}
