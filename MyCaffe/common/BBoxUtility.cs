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
    public class BBoxUtility<T> : IDisposable
    {
        Blob<T> m_blobDiff;
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
            m_blobDiff = new Blob<T>(cuda, log);
        }

        /// <summary>
        /// Clean up all resources.
        /// </summary>
        public void Dispose()
        {
            if (m_blobDiff != null)
            {
                m_blobDiff.Dispose();
                m_blobDiff = null;
            }
        }

        /// <summary>
        /// Compute the average precision given true positive and false positive vectors.
        /// </summary>
        /// <param name="rgTp">Specifies a list of scores and true positive.</param>
        /// <param name="nNumPos">Specifies the number of true positives.</param>
        /// <param name="rgFp">Specifies a list of scores and false positive.</param>
        /// <param name="apVersion">Specifies the different ways of computing the Average Precisions.
        /// @see [Tag: Average Precision](https://sanchom.wordpress.com/tag/average-precision/)
        ///   
        /// Versions:
        /// 11point: The 11-point interpolated average precision, used in VOC2007.
        /// MaxIntegral: maximally interpolated AP. Used in VOC2012/ILSVRC.
        /// Integral: the natrual integral of the precision-recall curve.
        /// <param name="rgPrec">Returns the computed precisions.</param>
        /// <param name="rgRec">Returns the computed recalls.</param>
        /// <returns>The Average Precision value is returned.</returns>
        public float ComputeAP(List<Tuple<float, int>> rgTp, int nNumPos, List<Tuple<float, int>> rgFp, ApVersion apVersion, out List<float> rgPrec, out List<float> rgRec)
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

            switch (apVersion)
            {
                // VOC2007 style for computing AP
                case ApVersion.ELEVENPOINT:
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
                case ApVersion.MAXINTEGRAL:
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
                case ApVersion.INTEGRAL:
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
                    m_log.FAIL("Unknown ap version '" + apVersion.ToString() + "'!");
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

            if (nTopK > -1 && nTopK < rgItems.Count)
                rgItems = rgItems.Take(nTopK).ToList();

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
        /// <returns>The indices of the bboxes after nms are returned.</returns>
        public List<int> ApplyNMS(List<NormalizedBBox> rgBBoxes, List<float> rgScores, float fThreshold, int nTopK)
        {
            Dictionary<int, Dictionary<int, float>> rgOverlaps;
            return ApplyNMS(rgBBoxes, rgScores, fThreshold, nTopK, false, out rgOverlaps);
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
        /// <returns>The indices of the bboxes after nms are returned.</returns>
        public List<int> ApplyNMS(List<NormalizedBBox> rgBBoxes, List<float> rgScores, float fThreshold, int nTopK, bool bReuseOverlaps, out Dictionary<int, Dictionary<int, float>> rgOverlaps)
        {
            List<int> rgIndices = new List<int>();
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
                        if (rgOverlaps.ContainsKey(nBestIdx) &&
                            rgOverlaps[nBestIdx].ContainsKey(nCurIdx))
                            // Use the computed overlap.
                            fCurOverlap = rgOverlaps[nBestIdx][nCurIdx];
                        else if (rgOverlaps.ContainsKey(nCurIdx) &&
                            rgOverlaps[nCurIdx].ContainsKey(nBestIdx))
                            // Use the computed overlap.
                            fCurOverlap = rgOverlaps[nCurIdx][nBestIdx];
                        else
                        {
                            fCurOverlap = JaccardOverlap(best_bbox, cur_bbox);

                            // Store the overlap for future use.
                            if (!rgOverlaps.ContainsKey(nBestIdx))
                                rgOverlaps.Add(nBestIdx, new Dictionary<int, float>());

                            if (!rgOverlaps[nBestIdx].ContainsKey(nCurIdx))
                                rgOverlaps[nBestIdx].Add(nCurIdx, fCurOverlap);
                            else
                                rgOverlaps[nBestIdx][nCurIdx] = fCurOverlap;
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

            return rgIndices;
        }

        /// <summary>
        /// Get detection results from rgData.
        /// </summary>
        /// <param name="rgData">Specifies a 1 x 1 x nNumDet x 7 blob data.</param>
        /// <param name="nNumDet">Specifies the number of detections.</param>
        /// <param name="nBackgroundLabelId">Specifies the label for the background class which is used to do a
        /// sanity check so that no detection contains it.</param>
        /// <returns>The detection results are returned for each class from each image.</returns>
        public Dictionary<int, Dictionary<int, List<NormalizedBBox>>> GetDetectionResults(float[] rgData, int nNumDet, int nBackgroundLabelId)
        {
            Dictionary<int, Dictionary<int, List<NormalizedBBox>>> rgAllDetections = new Dictionary<int, Dictionary<int, List<NormalizedBBox>>>();

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
                    rgAllDetections.Add(nItemId, new Dictionary<int, List<NormalizedBBox>>());

                if (!rgAllDetections[nItemId].ContainsKey(nLabel))
                    rgAllDetections[nItemId].Add(nLabel, new List<NormalizedBBox>());

                rgAllDetections[nItemId][nLabel].Add(bbox);
            }

            return rgAllDetections;
        }

        /// <summary>
        /// Get the prior boundary boxes from the rgPriorData.
        /// </summary>
        /// <param name="rgPriorData">Specifies the prior data as a 1 x 2 x nNumPriors x 4 x 1 blob.</param>
        /// <param name="nNumPriors">Specifies the number of priors.</param>
        /// <param name="rgPriorVariances">Specifies the prior variances need by prior bboxes.</param>
        /// <returns>The prior bbox list is returned.</returns>
        public List<NormalizedBBox> GetPrior(float[] rgPriorData, int nNumPriors, out List<List<float>> rgPriorVariances)
        {
            List<NormalizedBBox> rgPriorBboxes = new List<NormalizedBBox>();
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

            return rgPriorBboxes;
        }

        private int getLabel(int nPredIdx, int nNumPredsPerClass, int nNumClasses, int nBackgroundLabel, DictionaryMap<List<int>> rgMatchIndices, List<NormalizedBBox> rgGtBoxes)
        {
            int nLabel = nBackgroundLabel;

            if (rgMatchIndices != null && rgMatchIndices.Count > 0 && rgGtBoxes != null && rgGtBoxes.Count > 0)
            {
                List<KeyValuePair<int, List<int>>> rgMatches = rgMatchIndices.Map.ToList();

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
        /// <returns>The confidence loss values are returned with confidence loss per location for each image.</returns>
        public List<List<float>> ComputeConfLoss(float[] rgConfData, int nNum, int nNumPredsPerClass, int nNumClasses, int nBackgroundLabelId, MultiBoxLossParameter.ConfLossType loss_type)
        {
            List<List<float>> rgrgAllConfLoss = new List<List<float>>();
            int nOffset = 0;

            for (int i = 0; i < nNum; i++)
            {
                List<float> rgConfLoss = new List<float>();

                for (int p = 0; p < nNumPredsPerClass; p++)
                {
                    int nStartIdx = p * nNumClasses;
                    // Get the label index.
                    int nLabel = nBackgroundLabelId;
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

                rgrgAllConfLoss.Add(rgConfLoss);
                nOffset += nNumPredsPerClass * nNumClasses;
            }

            return rgrgAllConfLoss;
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
        public List<List<float>> ComputeConfLoss(float[] rgConfData, int nNum, int nNumPredsPerClass, int nNumClasses, int nBackgroundLabelId, MultiBoxLossParameter.ConfLossType loss_type, List<DictionaryMap<List<int>>> rgAllMatchIndices, DictionaryMap<List<NormalizedBBox>> rgAllGtBoxes)
        {
            List<Dictionary<int, List<int>>> rgAllMatchIndices1 = new List<Dictionary<int, List<int>>>();
            foreach (DictionaryMap<List<int>> item in rgAllMatchIndices)
            {
                rgAllMatchIndices1.Add(item.Map);
            }

            return ComputeConfLoss(rgConfData, nNum, nNumPredsPerClass, nNumClasses, nBackgroundLabelId, loss_type, rgAllMatchIndices1, rgAllGtBoxes.Map);
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
        public List<List<float>> ComputeConfLoss(float[] rgConfData, int nNum, int nNumPredsPerClass, int nNumClasses, int nBackgroundLabelId, MultiBoxLossParameter.ConfLossType loss_type, List<Dictionary<int, List<int>>> rgAllMatchIndices, Dictionary<int, List<NormalizedBBox>> rgAllGtBoxes)
        {
            m_log.CHECK_LT(nBackgroundLabelId, nNumClasses, "The background id must be less than the number of classes!");
            List<List<float>> rgrgAllConfLoss = new List<List<float>>();
            int nOffset = 0;

            for (int i = 0; i < nNum; i++)
            {
                List<float> rgConfLoss = new List<float>();
                Dictionary<int, List<int>> rgMatchIndices = rgAllMatchIndices[i];

                for (int p = 0; p < nNumPredsPerClass; p++)
                {
                    int nStartIdx = p * nNumClasses;
                    // Get the label index.
                    int nLabel = nBackgroundLabelId;

                    foreach (KeyValuePair<int, List<int>> kv in rgMatchIndices)
                    {
                        List<int> rgMatchIndex = kv.Value;
                        m_log.CHECK_EQ(rgMatchIndex.Count, nNumPredsPerClass, "The number of match indexes must be equal to the NumPredsPerClass!");

                        if (rgMatchIndex[p] > -1)
                        {
                            m_log.CHECK(rgAllGtBoxes.ContainsKey(i), "The AllGtBoxes does not have the label '" + i.ToString() + "'!");
                            List<NormalizedBBox> rgGtBboxes = rgAllGtBoxes[i];

                            m_log.CHECK_LT(rgMatchIndex[p], rgGtBboxes.Count, "The match index at '" + p.ToString() + "' must be less than the number of Gt bboxes at label " + i.ToString() + " (" + rgGtBboxes.Count.ToString() + ")!");

                            nLabel = rgGtBboxes[rgMatchIndex[p]].label;
                            m_log.CHECK_GE(nLabel, 0, "The label must be >= 0.");
                            m_log.CHECK_NE(nLabel, nBackgroundLabelId, "The label cannot be the background label of '" + nBackgroundLabelId.ToString() + "'!");
                            m_log.CHECK_LT(nLabel, nNumClasses, "The label must be < NumClasses (" + nNumClasses.ToString() + ")!");

                            // A prior can only be matched to one gt bbox.
                            break;
                        }
                    }

                    float fLoss = 0;
                    switch (loss_type)
                    {
                        case MultiBoxLossParameter.ConfLossType.SOFTMAX:
                            {
                                m_log.CHECK_GE(nLabel, 0, "The label must be >= 0 for the SOFTMAX loss type.");
                                m_log.CHECK_LT(nLabel, nNumClasses, "The label must be < NumClasses for the SOFTMAX loss type.");
                                // Compute softmax probability.
                                // We need to subtract the max to avoid numerical issues.
                                float fMaxVal = rgConfData[nStartIdx];
                                for (int c = 1; c < nNumClasses; c++)
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

                rgrgAllConfLoss.Add(rgConfLoss);
                nOffset += nNumPredsPerClass * nNumClasses;
            }

            return rgrgAllConfLoss;
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
        public DictionaryMap<List<NormalizedBBox>> GetGroundTruth(float[] rgGtData, int nNumGt, int nBackgroundLabelId, bool bUseDifficultGt)
        {
            DictionaryMap<List<NormalizedBBox>> rgAllGt = new DictionaryMap<List<NormalizedBBox>>(null);

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

                if (rgAllGt[nItemId] == null)
                    rgAllGt[nItemId] = new List<NormalizedBBox>();

                rgAllGt[nItemId].Add(bbox);
            }

            return rgAllGt;
        }

        /// <summary>
        /// Create a set of ground truth bounding boxes from the rgGtData.
        /// </summary>
        /// <param name="rgGtData">Specifies the 1 x 1 x nNumGt x 7 blob with ground truth initialization data.</param>
        /// <param name="nNumGt">Specifies the number of ground truths.</param>
        /// <param name="nBackgroundLabelId">Specifies the background label.</param>
        /// <param name="bUseDifficultGt">Specifies whether or not to use the difficult ground truth.</param>
        /// <returns>A dictionary containing the ground truths per label is returned.</returns>
        public Dictionary<int, LabelBBox> GetGroundTruthEx(float[] rgGtData, int nNumGt, int nBackgroundLabelId, bool bUseDifficultGt)
        {
            Dictionary<int, LabelBBox> rgAllGtBboxes = new Dictionary<int, LabelBBox>();

            for (int i = 0; i < nNumGt; i++)
            {
                int nStartIdx = i * 8;
                int nItemId = (int)rgGtData[nStartIdx];
                if (nItemId == -1)
                    break;

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

                if (!rgAllGtBboxes.ContainsKey(nItemId))
                    rgAllGtBboxes.Add(nItemId, new LabelBBox());

                rgAllGtBboxes[nItemId].Add(nLabel, bbox);
            }

            return rgAllGtBboxes;
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
        /// Decode all bboxes in a batch.
        /// </summary>
        /// <param name="rgAllLocPreds">Specifies the batch of local predictions.</param>
        /// <param name="rgPriorBboxes">Specifies the set of prior bboxes.</param>
        /// <param name="rgrgfPrioVariances">Specifies the prior variances.</param>
        /// <param name="nNum">Specifies the number of items in the batch.</param>
        /// <param name="bShareLocation">Specifies whether or not to share locations.</param>
        /// <param name="nNumLocClasses">Specifies the number of local classes.</param>
        /// <param name="nBackgroundLabelId">Specifies the background label.</param>
        /// <param name="codeType">Specifies the coding type.</param>
        /// <param name="bVarianceEncodedInTarget">Specifies whether or not the variance is encoded in the target or not.</param>
        /// <param name="bClip">Specifies whether or not to clip.</param>
        /// <returns>The decoded Bboxes are returned.</returns>
        public List<LabelBBox> DecodeAll(List<LabelBBox> rgAllLocPreds, List<NormalizedBBox> rgPriorBboxes, List<List<float>> rgrgfPrioVariances, int nNum, bool bShareLocation, int nNumLocClasses, int nBackgroundLabelId, PriorBoxParameter.CodeType codeType, bool bVarianceEncodedInTarget, bool bClip)
        {
            List<LabelBBox> rgAllDecodedBboxes = new List<LabelBBox>();

            m_log.CHECK_EQ(rgAllLocPreds.Count, nNum, "The number of Loc Preds does not equal the expected Num!");

            for (int i = 0; i < nNum; i++)
            {
                // Decode predictions into bboxes.
                LabelBBox decode_bboxes = new LabelBBox();

                for (int c = 0; c < nNumLocClasses; c++)
                {
                    int nLabel = (bShareLocation) ? -1 : c;

                    // Ignore background class.
                    if (nLabel == nBackgroundLabelId)
                        continue;

                    // Something bad happened if there are not predictions for current label.
                    if (!rgAllLocPreds[i].Contains(nLabel))
                        m_log.FAIL("Could not find the location predictions for label '" + nLabel.ToString() + "'!");

                    List<NormalizedBBox> rgLabelLocPreds = rgAllLocPreds[i][nLabel];
                    decode_bboxes[nLabel] = Decode(rgPriorBboxes, rgrgfPrioVariances, codeType, bVarianceEncodedInTarget, bClip, rgLabelLocPreds);
                }

                rgAllDecodedBboxes.Add(decode_bboxes);
            }

            return rgAllDecodedBboxes;
        }

        /// <summary>
        /// Decode a set of bounding box.
        /// </summary>
        /// <param name="rgPriorBbox">Specifies an list of prior bounding boxs.</param>
        /// <param name="rgrgfPriorVariance">Specifies the list of prior variance (must have 4 elements each of which are > 0).</param>
        /// <param name="code_type">Specifies the code type.</param>
        /// <param name="bEncodeVarianceInTarget">Specifies whether or not to encode the variance in the target.</param>
        /// <param name="bClip">Specifies to enable/disable the clipping.</param>
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
        /// Check if a bbox meets the emit constraint w.r.t the src_bbox.
        /// </summary>
        /// <param name="src_bbox">Specifies the source Bbox.</param>
        /// <param name="bbox">Specifies the Bbox to check.</param>
        /// <param name="emit_constraint">Specifies the emit constraint.</param>
        /// <returns>If the emit constraint is met, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool MeetEmitConstraint(NormalizedBBox src_bbox, NormalizedBBox bbox, EmitConstraint emit_constraint)
        {
            if (emit_constraint.emit_type == EmitConstraint.EmitType.CENTER)
            {
                float fXCenter = (bbox.xmin + bbox.xmax) / 2;
                float fYCenter = (bbox.ymin + bbox.ymax) / 2;

                if ((fXCenter >= src_bbox.xmin && fXCenter <= src_bbox.xmax) &&
                    (fYCenter >= src_bbox.ymin && fYCenter <= src_bbox.ymax))
                    return true;
                else
                    return false;
            }
            else if (emit_constraint.emit_type == EmitConstraint.EmitType.MIN_OVERLAP)
            {
                float fBboxCoverage = Coverage(bbox, src_bbox);
                if (fBboxCoverage > emit_constraint.emit_overlap)
                    return true;
                else
                    return false;
            }
            else
            {
                m_log.FAIL("Unknown emit type!");
                return false;
            }
        }

        /// <summary>
        /// Compute the coverage of bbox1 by bbox2.
        /// </summary>
        /// <param name="bbox1">Specifies the first BBox.</param>
        /// <param name="bbox2">Specifies the second BBox.</param>
        /// <returns>The coverage of bbox1 by bbox2 is returned.</returns>
        public float Coverage(NormalizedBBox bbox1, NormalizedBBox bbox2)
        {
            NormalizedBBox intersectBBox = Intersect(bbox1, bbox2);
            float fIntersectSize = Size(intersectBBox);

            if (fIntersectSize > 0)
            {
                float fBbox1Size = Size(bbox1);
                return fBbox1Size / fIntersectSize;
            }

            return 0;
        }

        /// <summary>
        /// Locate bbox in the coordinate system of the source Bbox.
        /// </summary>
        /// <param name="srcBbox">Specifies the source Bbox.</param>
        /// <param name="bbox">Specifies the bbox to locate.</param>
        /// <returns>The bbox located within the source Bbox is returned.</returns>
        public NormalizedBBox Locate(NormalizedBBox srcBbox, NormalizedBBox bbox)
        {
            float fSrcWidth = srcBbox.xmax - srcBbox.xmin;
            float fSrcHeight = srcBbox.ymax - srcBbox.ymin;

            return new NormalizedBBox(srcBbox.xmin + bbox.xmin * fSrcWidth,
                                      srcBbox.ymin + bbox.ymin * fSrcHeight,
                                      srcBbox.xmax + bbox.xmax * fSrcWidth,
                                      srcBbox.ymax + bbox.ymax * fSrcHeight, 0, bbox.difficult);
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

        /// <summary>
        /// Output the predicted bbox on the actual image.
        /// </summary>
        /// <param name="bbox">Specifies the bbox.</param>
        /// <param name="szImg">Specifies the image size.</param>
        /// <param name="p">Specifies the resize parameter.</param>
        /// <returns>The predicted bbox is returned.</returns>
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
        /// Extrapolate the transformed bbox if height_scale and width_scale are explicitly
        /// provied, and the FIT_SMALL_SIZE resize mode is specified.
        /// </summary>
        /// <param name="param">Specifies the resize parameter.</param>
        /// <param name="nHeight">Specifies the height.</param>
        /// <param name="nWidth">Specifies the width.</param>
        /// <param name="crop_bbox">Specifies the crop Bbox.</param>
        /// <param name="bbox">Specifies the Bbox to be updated.</param>
        public void Extrapolate(ResizeParameter param, int nHeight, int nWidth, NormalizedBBox crop_bbox, NormalizedBBox bbox)
        {
            float fHeightScale = param.height_scale;
            float fWidthScale = param.width_scale;

            if (fHeightScale > 0 && fWidthScale > 0 && param.resize_mode == ResizeParameter.ResizeMode.FIT_SMALL_SIZE)
            {
                float fOrigAspect = (float)nWidth / (float)nHeight;
                float fResizeHeight = param.height;
                float fResizeWidth = param.width;
                float fResizeAspect = fResizeWidth / fResizeHeight;

                if (fOrigAspect < fResizeAspect)
                    fResizeHeight = fResizeWidth / fOrigAspect;
                else
                    fResizeWidth = fResizeHeight * fOrigAspect;

                float fCropHeight = fResizeHeight * (crop_bbox.ymax - crop_bbox.ymin);
                float fCropWidth = fResizeWidth * (crop_bbox.xmax - crop_bbox.xmin);
                m_log.CHECK_GE(fCropWidth, fWidthScale, "The crop width must be >= the width scale!");
                m_log.CHECK_GE(fCropHeight, fHeightScale, "The crop height must be >= the height scale!");

                bbox.Set(bbox.xmin * fCropWidth / fWidthScale,
                         bbox.xmax * fCropWidth / fWidthScale,
                         bbox.ymin * fCropHeight / fHeightScale,
                         bbox.ymax * fCropHeight / fHeightScale);
            }
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

        /// <summary>
        /// Find matches between prediction bboxes and ground truth bboxes.
        /// </summary>
        /// <param name="rgAllLocPreds">Specifies the location prediction, where each item contains location prediction for an image.</param>
        /// <param name="rgAllGtBboxes">Specifies the ground truth bboxes for the batch.</param>
        /// <param name="rgPriorBboxes">Specifies all prior bboxes in the format of NormalizedBBox.</param>
        /// <param name="rgrgPriorVariances">Specifies all the variances needed by prior bboxes.</param>
        /// <param name="p">Specifies the parameter for the MultiBoxLossLayer.</param>
        /// <param name="rgAllMatchOverlaps">Returns the jaccard overlaps between predictions and ground truth.</param>
        /// <param name="rgAllMatchIndices">Returns the mapping between predictions and ground truth.</param>
        public void FindMatches(List<LabelBBox> rgAllLocPreds, DictionaryMap<List<NormalizedBBox>> rgAllGtBboxes, List<NormalizedBBox> rgPriorBboxes, List<List<float>> rgrgPriorVariances, MultiBoxLossParameter p, out List<DictionaryMap<List<float>>> rgAllMatchOverlaps, out List<DictionaryMap<List<int>>> rgAllMatchIndices)
        {
            rgAllMatchOverlaps = new List<DictionaryMap<List<float>>>();
            rgAllMatchIndices = new List<DictionaryMap<List<int>>>();

            int nNumClasses = (int)p.num_classes;
            m_log.CHECK_GE(nNumClasses, 1, "The num_classes should not be less than 1.");

            bool bShareLocation = p.share_location;
            int nLocClasses = (bShareLocation) ? 1 : nNumClasses;
            MultiBoxLossParameter.MatchType matchType = p.match_type;
            float fOverlapThreshold = p.overlap_threshold;
            bool bUsePriorForMatching = p.use_prior_for_matching;
            int nBackgroundLabelId = (int)p.background_label_id;
            PriorBoxParameter.CodeType codeType = p.code_type;
            bool bEncodeVarianceInTarget = p.encode_variance_in_target;
            bool bIgnoreCrossBoundaryBbox = p.ignore_cross_boundary_bbox;

            // Find the matches.
            int nNum = rgAllLocPreds.Count;
            for (int i = 0; i < nNum; i++)
            {
                DictionaryMap<List<int>> rgMatchIndices = new DictionaryMap<List<int>>(null);
                DictionaryMap<List<float>> rgMatchOverlaps = new DictionaryMap<List<float>>(null);

                // Check if there is a ground truth for the current image.
                if (!rgAllGtBboxes.Map.ContainsKey(i))
                {
                    // There is no gt for current image.  All predictions are negative.
                    rgAllMatchIndices.Add(rgMatchIndices);
                    rgAllMatchOverlaps.Add(rgMatchOverlaps);
                    continue;
                }

                // Find match between predictions and ground truth.
                List<NormalizedBBox> rgGtBboxes = rgAllGtBboxes[i];
                if (!bUsePriorForMatching)
                {
                    for (int c = 0; c < nLocClasses; c++)
                    {
                        int nLabel = (bShareLocation) ? -1 : c;

                        // Ignore background loc predictions.
                        if (!bShareLocation && nLabel == nBackgroundLabelId)
                            continue;

                        // Decode the prediction into bbox first.
                        bool bClipBbox = false;
                        List<NormalizedBBox> rgLocBBoxes = Decode(rgPriorBboxes, rgrgPriorVariances, codeType, bEncodeVarianceInTarget, bClipBbox, rgAllLocPreds[i][nLabel]);

                        List<int> rgMatchIndices1;
                        List<float> rgMatchOverlaps1;
                        Match(rgGtBboxes, rgLocBBoxes, nLabel, matchType, fOverlapThreshold, bIgnoreCrossBoundaryBbox, out rgMatchIndices1, out rgMatchOverlaps1);

                        rgMatchIndices[nLabel] = rgMatchIndices1;
                        rgMatchOverlaps[nLabel] = rgMatchOverlaps1;
                    }
                }
                else
                {
                    // Use prior bboxes to match against all ground truth.
                    List<int> rgTempMatchIndices = new List<int>();
                    List<float> rgTempMatchOverlaps = new List<float>();
                    int nLabel = -1;

                    Match(rgGtBboxes, rgPriorBboxes, nLabel, matchType, fOverlapThreshold, bIgnoreCrossBoundaryBbox, out rgTempMatchIndices, out rgTempMatchOverlaps);

                    if (bShareLocation)
                    {
                        rgMatchIndices[nLabel] = rgTempMatchIndices;
                        rgMatchOverlaps[nLabel] = rgTempMatchOverlaps;
                    }
                    else
                    {
                        // Get ground truth label for each ground truth bbox.
                        List<int> rgGtLabels = new List<int>();
                        for (int g = 0; g < rgGtBboxes.Count; g++)
                        {
                            rgGtLabels.Add(rgGtBboxes[g].label);
                        }

                        // Distribute the matching results to different loc_class.
                        for (int c = 0; c < nLocClasses; c++)
                        {
                            // Ignore background loc predictions.
                            if (c == nBackgroundLabelId)
                                continue;

                            rgMatchIndices[c] = rgTempMatchIndices;
                            rgMatchOverlaps[c] = rgTempMatchOverlaps;

                            for (int m = 0; m < rgTempMatchIndices.Count; m++)
                            {
                                if (rgTempMatchIndices[m] > -1)
                                {
                                    int nGtIdx = rgTempMatchIndices[m];
                                    m_log.CHECK_LT(nGtIdx, rgGtLabels.Count, "The gt index is larger than the number of gt labels.");
                                    if (c == rgGtLabels[nGtIdx])
                                        rgMatchIndices[c][m] = nGtIdx;
                                }
                            }
                        }
                    }
                }

                rgAllMatchIndices.Add(rgMatchIndices);
                rgAllMatchOverlaps.Add(rgMatchOverlaps);
            }
        }

        /// <summary>
        /// Counts the number of matches in the list of maps.
        /// </summary>
        /// <param name="rgAllMatchIndices">Specifies the list of match indices.</param>
        /// <param name="nNum">Specifies the number of items.</param>
        /// <returns>The total matches found in the number of items is returned.</returns>
        public int CountNumMatches(List<DictionaryMap<List<int>>> rgAllMatchIndices, int nNum)
        {
            int nNumMatches = 0;

            for (int i = 0; i < nNum; i++)
            {
                Dictionary<int, List<int>> rgMatchIndices = rgAllMatchIndices[i].Map;

                foreach (KeyValuePair<int, List<int>> kv in rgMatchIndices)
                {
                    List<int> rgMatchIndex = kv.Value;

                    for (int m = 0; m < rgMatchIndex.Count; m++)
                    {
                        if (rgMatchIndex[m] > -1)
                            nNumMatches++;
                    }
                }
            }

            return nNumMatches;
        }

        /// <summary>
        /// Returns whether or not mining is eligible given the mining type and match index.
        /// </summary>
        /// <param name="miningType">Specifies the mining type.</param>
        /// <param name="nMatchIdx">Specifies the matching index.</param>
        /// <param name="fMatchOverlap">Specifies the matching overlap.</param>
        /// <param name="fNegOverlap">Specifies the negative overlap.</param>
        /// <returns>If mining is allowed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool IsEligibleMining(MultiBoxLossParameter.MiningType miningType, int nMatchIdx, float fMatchOverlap, float fNegOverlap)
        {
            if (miningType == MultiBoxLossParameter.MiningType.MAX_NEGATIVE)
            {
                if (nMatchIdx == -1 && fMatchOverlap < fNegOverlap)
                    return true;
                else
                    return false;
            }
            else if (miningType == MultiBoxLossParameter.MiningType.HARD_EXAMPLE)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Mine the hard examples from the batch.
        /// </summary>
        /// <param name="blobConf">Specifies the confidence prediction.</param>
        /// <param name="rgAllLocPreds">Specifies the location prediction, where each item contains the location prediction for an image.</param>
        /// <param name="rgAllGtBBoxes">Specifies the ground truth bboxes for the batch.</param>
        /// <param name="rgPriorBboxes">Specifies the prior bboxes in the format of NormalizedBBox.</param>
        /// <param name="rgrgPriorVariances">Specifies the variances needed by the prior bboxes.</param>
        /// <param name="rgAllMatchOverlaps">Specifies the jaccard overlap between predictions and the ground truth.</param>
        /// <param name="p">Specifies the parameters for the MultiBoxLossLayer.</param>
        /// <param name="rgAllMatchIndices">Specifies the mapping between predictions and the ground truth.</param>
        /// <param name="rgAllNegIndices">Specifies the indices for negative samples.</param>
        /// <param name="nNumNegs">Specifies the numberof negative indices.</param>
        /// <returns>The number of matches is returned.</returns>
        public int MineHardExamples(Blob<T> blobConf, List<LabelBBox> rgAllLocPreds, DictionaryMap<List<NormalizedBBox>> rgAllGtBBoxes, List<NormalizedBBox> rgPriorBboxes, List<List<float>> rgrgPriorVariances, List<DictionaryMap<List<float>>> rgAllMatchOverlaps, MultiBoxLossParameter p, List<DictionaryMap<List<int>>> rgAllMatchIndices, List<List<int>> rgAllNegIndices, out int nNumNegs)
        {
            int nNum = rgAllLocPreds.Count;
            int nNumMatches = CountNumMatches(rgAllMatchIndices, nNum);

            nNumNegs = 0;

            int nNumPriors = rgPriorBboxes.Count;
            m_log.CHECK_EQ(nNumPriors, rgrgPriorVariances.Count, "The number of priors must be the same as the number of prior variances.");

            // Get parameters.
            int nNumClasses = (int)p.num_classes;
            m_log.CHECK_GE(nNumClasses, 1, "num_classes should be at least 1.");

            int nBackgroundLabelId = (int)p.background_label_id;
            bool bUsePriorForNms = p.use_prior_for_nms;
            MultiBoxLossParameter.ConfLossType confLossType = p.conf_loss_type;
            MultiBoxLossParameter.MiningType miningType = p.mining_type;

            if (miningType == MultiBoxLossParameter.MiningType.NONE)
                return nNumMatches;

            MultiBoxLossParameter.LocLossType locLossType = p.loc_loss_type;
            float fNegPosRatio = p.neg_pos_ratio;
            float fNegOverlap = p.neg_overlap;
            PriorBoxParameter.CodeType codeType = p.code_type;
            bool bEncodeVarianceInTarget = p.encode_variance_in_target;
            float fNmsThreshold = 0;
            int nTopK = -1;

            if (p.nms_param != null)
            {
                fNmsThreshold = p.nms_param.nms_threshold;
                nTopK = p.nms_param.top_k.GetValueOrDefault(1);
            }

            int nSampleSize = p.sample_size;

            // Compute confidence losses based on matching results.
            float[] rgConfData = Utility.ConvertVecF<T>(blobConf.mutable_cpu_data);
            List<List<float>> rgAllConfLoss = ComputeConfLoss(rgConfData, nNum, nNumPriors, nNumClasses, nBackgroundLabelId, confLossType, rgAllMatchIndices, rgAllGtBBoxes);
            List<List<float>> rgAllLocLoss = new List<List<float>>();

            // Compute localization losses based on matching results.
            if (miningType == MultiBoxLossParameter.MiningType.HARD_EXAMPLE)
            {
                Blob<T> blobLocPred = new Blob<T>(m_cuda, m_log);
                Blob<T> blobLocGt = new Blob<T>(m_cuda, m_log);

                if (nNumMatches != 0)
                {
                    List<int> rgLocShape = Utility.Create<int>(2, 1);
                    rgLocShape[1] = nNumMatches * 4;
                    blobLocPred.Reshape(rgLocShape);
                    blobLocGt.Reshape(rgLocShape);
                    EncodeLocPrediction(rgAllLocPreds, rgAllGtBBoxes, rgAllMatchIndices, rgPriorBboxes, rgrgPriorVariances, p, blobLocPred, blobLocGt);
                }

                rgAllLocLoss = ComputeLocLoss(blobLocPred, blobLocGt, rgAllMatchIndices, nNum, nNumPriors, locLossType);
            }
            // No localization loss.
            else
            {
                for (int i = 0; i < nNum; i++)
                {
                    List<float> rgLocLoss = Utility.Create<float>(nNumPriors, 0.0f);
                    rgAllLocLoss.Add(rgLocLoss);
                }
            }

            for (int i = 0; i < nNum; i++)
            {
                DictionaryMap<List<int>> rgMatchIndices = rgAllMatchIndices[i];
                DictionaryMap<List<float>> rgMatchOverlaps = rgAllMatchOverlaps[i];

                // loc + conf loss.
                List<float> rgConfLoss = rgAllConfLoss[i];
                List<float> rgLocLoss = rgAllLocLoss[i];
                List<float> rgLoss = new List<float>();

                for (int j = 0; j < rgConfLoss.Count; j++)
                {
                    rgLoss.Add(rgConfLoss[j] + rgLocLoss[j]);
                }

                // Pick negatives or hard examples based on loss.
                List<int> rgSelIndices = new List<int>();
                List<int> rgNegIndices = new List<int>();

                foreach (KeyValuePair<int, List<int>> kv in rgMatchIndices.Map)
                {
                    int nLabel = kv.Key;
                    int nNumSel = 0;

                    // Get potential indices and loss pairs.
                    List<KeyValuePair<float, int>> rgLossIndices = new List<KeyValuePair<float, int>>();

                    for (int m = 0; m < rgMatchIndices[nLabel].Count; m++)
                    {
                        if (IsEligibleMining(miningType, rgMatchIndices[nLabel][m], rgMatchOverlaps[nLabel][m], fNegOverlap))
                        {
                            rgLossIndices.Add(new KeyValuePair<float, int>(rgLoss[m], m));
                            nNumSel++;
                        }
                    }

                    if (miningType == MultiBoxLossParameter.MiningType.MAX_NEGATIVE)
                    {
                        int nNumPos = 0;

                        for (int m = 0; m < rgMatchIndices[nLabel].Count; m++)
                        {
                            if (rgMatchIndices[nLabel][m] > -1)
                                nNumPos++;
                        }

                        nNumSel = Math.Min((int)(nNumPos * fNegPosRatio), nNumSel);
                    }
                    else if (miningType == MultiBoxLossParameter.MiningType.HARD_EXAMPLE)
                    {
                        m_log.CHECK_GT(nSampleSize, 0, "The sample size must be greater than 0 for HARD_EXAMPLE mining.");
                        nNumSel = Math.Min(nSampleSize, nNumSel);
                    }

                    // Select samples.
                    if (p.nms_param != null && fNmsThreshold > 0)
                    {
                        // Do nms before selecting samples.
                        List<float> rgSelLoss = new List<float>();
                        List<NormalizedBBox> rgSelBoxes = new List<NormalizedBBox>();

                        if (bUsePriorForNms)
                        {
                            for (int m = 0; m < rgMatchIndices[nLabel].Count; m++)
                            {
                                if (IsEligibleMining(miningType, rgMatchIndices[nLabel][m], rgMatchOverlaps[nLabel][m], fNegOverlap))
                                {
                                    rgSelLoss.Add(rgLoss[m]);
                                    rgSelBoxes.Add(rgPriorBboxes[m]);
                                }
                            }
                        }
                        else
                        {
                            // Decode the prediction into bbox first.
                            bool bClipBbox = false;
                            List<NormalizedBBox> rgLocBBoxes = Decode(rgPriorBboxes, rgrgPriorVariances, codeType, bEncodeVarianceInTarget, bClipBbox, rgAllLocPreds[i][nLabel]);

                            for (int m = 0; m < rgMatchIndices[nLabel].Count; m++)
                            {
                                if (IsEligibleMining(miningType, rgMatchIndices[nLabel][m], rgMatchOverlaps[nLabel][m], fNegOverlap))
                                {
                                    rgSelLoss.Add(rgLoss[m]);
                                    rgSelBoxes.Add(rgLocBBoxes[m]);
                                }
                            }
                        }

                        // Do non-maximum suppression based on the loss.
                        List<int> rgNmsIndices = ApplyNMS(rgSelBoxes, rgSelLoss, fNmsThreshold, nTopK);
                        if (rgNmsIndices.Count < nNumSel)
                            m_log.WriteLine("WARNING: Not enough samples after NMS: " + rgNmsIndices.Count.ToString());

                        // Pick top example indices after nms.
                        nNumSel = Math.Min(rgNmsIndices.Count, nNumSel);
                        for (int n = 0; n < nNumSel; n++)
                        {
                            rgSelIndices.Insert(0, rgLossIndices[rgNmsIndices[n]].Value);
                        }
                    }
                    else
                    {
                        // Pick top exampel indices based on loss.
                        rgLossIndices = rgLossIndices.OrderByDescending(p1 => p1.Key).ToList();
                        for (int n = 0; n < nNumSel; n++)
                        {
                            rgSelIndices.Insert(0, rgLossIndices[n].Value);
                        }
                    }

                    // Update the match_indices and select neg_indices.
                    for (int m = 0; m < rgMatchIndices[nLabel].Count; m++)
                    {
                        if (rgMatchIndices[nLabel][m] > -1)
                        {
                            if (miningType == MultiBoxLossParameter.MiningType.HARD_EXAMPLE && !rgSelIndices.Contains(m))
                            {
                                rgMatchIndices[nLabel][m] = -1;
                                nNumMatches -= 1;
                            }
                        }
                        else if (rgMatchIndices[nLabel][m] == -1)
                        {
                            if (!rgSelIndices.Contains(m))
                            {
                                rgNegIndices.Add(m);
                                nNumNegs += 1;
                            }
                        }
                    }
                }

                rgAllNegIndices.Add(rgNegIndices);
            }

            return nNumMatches;
        }

        /// <summary>
        /// Encode the localization prediction and ground truth for each matched prior.
        /// </summary>
        /// <param name="rgAllLocPreds">Specifies the location prediction, where each item contains the location prediction for an image.</param>
        /// <param name="rgAllGtBboxes">Specifies the ground truth bboxes for the batch.</param>
        /// <param name="rgAllMatchIndices">Specifies the mapping between predictions and the ground truth.</param>
        /// <param name="rgPriorBboxes">Specifies the prior bboxes in the format of NormalizedBBox.</param>
        /// <param name="rgrgPriorVariances">Specifies the variances needed by the prior bboxes.</param>
        /// <param name="p">Specifies the parameters for the MultiBoxLossLayer.</param>
        /// <param name="blobLocPred">Specifies the location prediction results.</param>
        /// <param name="blobLocGt">Specifies the encoded location ground truth.</param>
        public void EncodeLocPrediction(List<LabelBBox> rgAllLocPreds, DictionaryMap<List<NormalizedBBox>> rgAllGtBboxes, List<DictionaryMap<List<int>>> rgAllMatchIndices, List<NormalizedBBox> rgPriorBboxes, List<List<float>> rgrgPriorVariances, MultiBoxLossParameter p, Blob<T> blobLocPred, Blob<T> blobLocGt)
        {
            float[] rgLocPredData = Utility.ConvertVecF<T>(blobLocPred.mutable_cpu_data);
            float[] rgLocGtData = Utility.ConvertVecF<T>(blobLocGt.mutable_cpu_data);

            int nNum = rgAllLocPreds.Count;
            // Get parameters.
            PriorBoxParameter.CodeType codeType = p.code_type;
            bool bEncodeVarianceInTarget = p.encode_variance_in_target;
            bool bBpInside = p.bp_inside;
            bool bUsePriorForMatching = p.use_prior_for_matching;
            int nCount = 0;

            for (int i = 0; i < nNum; i++)
            {
                foreach (KeyValuePair<int, List<int>> kv in rgAllMatchIndices[i].Map)
                {
                    int nLabel = kv.Key;
                    List<int> rgMatchIndex = kv.Value;

                    m_log.CHECK(rgAllLocPreds[i].Contains(nLabel), "The all local pred must contain the label '" + nLabel.ToString() + "'!");
                    List<NormalizedBBox> rgLocPred = rgAllLocPreds[i][nLabel];

                    for (int j = 0; j < rgMatchIndex.Count; j++)
                    {
                        if (rgMatchIndex[j] <= -1)
                            continue;

                        // Store encoded ground truth.
                        int nGtIdx = rgMatchIndex[j];
                        m_log.CHECK(rgAllGtBboxes.Map.ContainsKey(i), "All gt bboxes should contain '" + i.ToString() + "'!");
                        m_log.CHECK_LT(nGtIdx, rgAllGtBboxes[i].Count, "The ground truth index should be less than the number of ground truths at '" + i.ToString() + "'!");
                        NormalizedBBox gtBbox = rgAllGtBboxes[i][nGtIdx];
                        m_log.CHECK_LT(j, rgPriorBboxes.Count, "The prior bbox count is too small!");
                        NormalizedBBox gtEncode = Encode(rgPriorBboxes[j], rgrgPriorVariances[j], codeType, bEncodeVarianceInTarget, gtBbox);

                        rgLocGtData[nCount * 4 + 0] = gtEncode.xmin;
                        rgLocGtData[nCount * 4 + 1] = gtEncode.ymin;
                        rgLocGtData[nCount * 4 + 2] = gtEncode.xmax;
                        rgLocGtData[nCount * 4 + 3] = gtEncode.ymax;

                        // Store location prediction.
                        m_log.CHECK_LT(j, rgLocPred.Count, "The loc pred count is too small!");

                        if (bBpInside)
                        {
                            NormalizedBBox matchBbox = rgPriorBboxes[j];

                            if (!bUsePriorForMatching)
                            {
                                bool bClipBbox = false;
                                matchBbox = Decode(rgPriorBboxes[j], rgrgPriorVariances[j], codeType, bEncodeVarianceInTarget, bClipBbox, rgLocPred[j]);
                            }

                            // When a dimension of match_bbox is outside of image region, use
                            // gt_encode to simulate zero gradient.
                            rgLocPredData[nCount * 4 + 0] = (matchBbox.xmin < 0 || matchBbox.xmin > 1) ? gtEncode.xmin : rgLocPred[j].xmin;
                            rgLocPredData[nCount * 4 + 1] = (matchBbox.ymin < 0 || matchBbox.ymin > 1) ? gtEncode.ymin : rgLocPred[j].ymin;
                            rgLocPredData[nCount * 4 + 2] = (matchBbox.xmax < 0 || matchBbox.xmax > 1) ? gtEncode.xmax : rgLocPred[j].xmax;
                            rgLocPredData[nCount * 4 + 3] = (matchBbox.ymax < 0 || matchBbox.ymax > 1) ? gtEncode.ymax : rgLocPred[j].ymax;
                        }
                        else
                        {
                            rgLocPredData[nCount * 4 + 0] = rgLocPred[j].xmin;
                            rgLocPredData[nCount * 4 + 1] = rgLocPred[j].ymin;
                            rgLocPredData[nCount * 4 + 2] = rgLocPred[j].xmax;
                            rgLocPredData[nCount * 4 + 3] = rgLocPred[j].ymax;
                        }

                        if (bEncodeVarianceInTarget)
                        {
                            for (int k = 0; k < 4; k++)
                            {
                                m_log.CHECK_GT(rgrgPriorVariances[j][k], 0, "The variance at " + j.ToString() + ", " + k.ToString() + " must be greater than zero.");
                                rgLocPredData[nCount * 4 + k] /= rgrgPriorVariances[j][k];
                                rgLocGtData[nCount * 4 + k] /= rgrgPriorVariances[j][k];
                            }
                        }

                        nCount++;
                    }
                }
            }

            blobLocPred.mutable_cpu_data = Utility.ConvertVec<T>(rgLocPredData);
            blobLocGt.mutable_cpu_data = Utility.ConvertVec<T>(rgLocGtData);
        }

        /// <summary>
        /// Encode the confidence predictions and ground truth for each matched prior.
        /// </summary>
        /// <param name="rgfConfData">Specifies the num x num_priors * num_classes blob.</param>
        /// <param name="nNum">Specifies the number of images.</param>
        /// <param name="nNumPriors">Specifies the number of priors (predictions) per image.</param>
        /// <param name="p">Specifies the parameters for the MultiBoxLossLayer.</param>
        /// <param name="rgAllMatchIndices">Specifies the mapping between predictions and the ground truth.</param>
        /// <param name="rgAllNegIndices">Specifies the indices for negative samples.</param>
        /// <param name="rgAllGtBBoxes">Specifies the ground truth bboxes for the batch.</param>
        /// <param name="blobConfPred">Specifies the confidence prediction results.</param>
        /// <param name="blobConfGt">Specifies the confidence ground truth.</param>
        public void EncodeConfPrediction(float[] rgfConfData, int nNum, int nNumPriors, MultiBoxLossParameter p, List<DictionaryMap<List<int>>> rgAllMatchIndices, List<List<int>> rgAllNegIndices, DictionaryMap<List<NormalizedBBox>> rgAllGtBBoxes, Blob<T> blobConfPred, Blob<T> blobConfGt)
        {
            float[] rgConfPredData = Utility.ConvertVecF<T>(blobConfPred.mutable_cpu_data);
            float[] rgConfGtData = Utility.ConvertVecF<T>(blobConfGt.mutable_cpu_data);
            int nConfDataOffset = 0;
            int nConfGtDataOffset = 0;

            // Get parameters.
            int nNumClasses = (int)p.num_classes;
            m_log.CHECK_GE(nNumClasses, 1, "The the num_classes should not be less than 1.");
            int nBackgroundLabelId = (int)p.background_label_id;
            bool bMapObjectToAgnostic = p.map_object_to_agnostic;

            if (bMapObjectToAgnostic)
            {
                if (nBackgroundLabelId >= 0)
                    m_log.CHECK_EQ(nNumClasses, 2, "There should be 2 classes when mapping obect to agnostic with a background label.");
                else
                    m_log.CHECK_EQ(nNumClasses, 1, "There should only b 1 class when mapping object to agnostic with no background label.");
            }

            MultiBoxLossParameter.MiningType miningType = p.mining_type;
            bool bDoNegMining;

            if (p.do_neg_mining.HasValue)
            {
                m_log.WriteLine("WARNING: do_neg_mining is depreciated, using mining_type instead.");
                bDoNegMining = p.do_neg_mining.Value;
                m_log.CHECK(bDoNegMining == (miningType != MultiBoxLossParameter.MiningType.NONE), "The mining_type and do_neg_mining settings are inconsistent.");
            }

            bDoNegMining = (miningType != MultiBoxLossParameter.MiningType.NONE) ? true : false;
            MultiBoxLossParameter.ConfLossType confLossType = p.conf_loss_type;
            int nCount = 0;

            for (int i = 0; i < nNum; i++)
            {
                if (rgAllGtBBoxes.Map.ContainsKey(i))
                {
                    // Save matched (positive) bboxes scores and labels.
                    DictionaryMap<List<int>> rgMatchIndicies = rgAllMatchIndices[i];

                    foreach (KeyValuePair<int, List<int>> kv in rgAllMatchIndices[i].Map)
                    {
                        List<int> rgMatchIndex = kv.Value;
                        m_log.CHECK_EQ(rgMatchIndex.Count, nNumPriors, "The match index count should equal the number of priors '" + nNumPriors.ToString() + "'!");

                        for (int j = 0; j < nNumPriors; j++)
                        {
                            if (rgMatchIndex[j] <= -1)
                                continue;

                            int nGtLabel = (bMapObjectToAgnostic) ? nBackgroundLabelId + 1 : rgAllGtBBoxes[i][rgMatchIndex[j]].label;
                            int nIdx = (bDoNegMining) ? nCount : j;

                            switch (confLossType)
                            {
                                case MultiBoxLossParameter.ConfLossType.SOFTMAX:
                                    rgConfGtData[nConfGtDataOffset + nIdx] = nGtLabel;
                                    break;

                                case MultiBoxLossParameter.ConfLossType.LOGISTIC:
                                    rgConfGtData[nConfGtDataOffset + nIdx * nNumClasses + nGtLabel] = 1;
                                    break;

                                default:
                                    m_log.FAIL("Unknown conf loss type.");
                                    break;
                            }

                            if (bDoNegMining)
                            {
                                Array.Copy(rgfConfData, nConfDataOffset + j * nNumClasses, rgConfPredData, nCount * nNumClasses, nNumClasses);
                                nCount++;
                            }
                        }
                    }

                    // Go to next image.
                    if (bDoNegMining)
                    {
                        // Save negative bboxes scores and labels.
                        for (int n = 0; n < rgAllNegIndices[i].Count; n++)
                        {
                            int j = rgAllNegIndices[i][n];
                            m_log.CHECK_LT(j, nNumPriors, "The number of priors is too small!");

                            Array.Copy(rgfConfData, nConfDataOffset + j * nNumClasses, rgConfPredData, nCount * nNumClasses, nNumClasses);

                            switch (confLossType)
                            {
                                case MultiBoxLossParameter.ConfLossType.SOFTMAX:
                                    rgConfGtData[nConfGtDataOffset + nCount] = nBackgroundLabelId;
                                    break;

                                case MultiBoxLossParameter.ConfLossType.LOGISTIC:
                                    if (nBackgroundLabelId >= 0 && nBackgroundLabelId < nNumClasses)
                                        rgConfGtData[nConfGtDataOffset + nCount * nNumClasses + nBackgroundLabelId] = 1;
                                    break;

                                default:
                                    m_log.FAIL("Unknown conf loss type.");
                                    break;
                            }

                            nCount++;
                        }
                    }
                }

                if (bDoNegMining)
                    nConfDataOffset += nNumPriors * nNumClasses;
                else
                    nConfGtDataOffset += nNumPriors;
            }

            blobConfPred.mutable_cpu_data = Utility.ConvertVec<T>(rgConfPredData);
            blobConfGt.mutable_cpu_data = Utility.ConvertVec<T>(rgConfGtData);
        }

        /// <summary>
        /// Compute the localization loss per matched prior.
        /// </summary>
        /// <param name="blobLocPred">Specifies the location prediction results.</param>
        /// <param name="blobLocGt">Specifies the encoded location ground truth.</param>
        /// <param name="rgAllMatchIndices">Specifies the mapping between predictions and the ground truth.</param>
        /// <param name="nNum">Specifies the number of images in the batch.</param>
        /// <param name="nNumPriors">Specifies the total number of priors.</param>
        /// <param name="lossType">Specifies the type of localization loss, Smooth_L1 or L2.</param>
        /// <returns>Returns the localization loss for all priors in the batch.</returns>
        public List<List<float>> ComputeLocLoss(Blob<T> blobLocPred, Blob<T> blobLocGt, List<DictionaryMap<List<int>>> rgAllMatchIndices, int nNum, int nNumPriors, MultiBoxLossParameter.LocLossType lossType)
        {
            List<List<float>> rgLocAllLoss = new List<List<float>>();
            int nLocCount = blobLocPred.count();
            m_log.CHECK_EQ(nLocCount, blobLocGt.count(), "The loc pred and loc gt must have the same count!");
            float[] rgfDiff = null;

            if (nLocCount != 0)
            {
                m_blobDiff.ReshapeLike(blobLocPred);
                m_cuda.sub(nLocCount, blobLocPred.gpu_data, blobLocGt.gpu_data, m_blobDiff.mutable_gpu_data);
                rgfDiff = Utility.ConvertVecF<T>(m_blobDiff.mutable_cpu_data);
            }

            int nCount = 0;

            for (int i = 0; i < nNum; i++)
            {
                List<float> rgLocLoss = Utility.Create<float>(nNumPriors, 0.0f);

                foreach (KeyValuePair<int, List<int>> kv in rgAllMatchIndices[i].Map)
                {
                    List<int> rgMatchIndex = kv.Value;
                    m_log.CHECK_EQ(nNumPriors, rgMatchIndex.Count, "The match index count at " + i.ToString() + " is too small.");

                    for (int j = 0; j < rgMatchIndex.Count; j++)
                    {
                        if (rgMatchIndex[j] <= -1)
                            continue;

                        double dfLoss = 0;

                        for (int k = 0; k < 4; k++)
                        {
                            float fVal = rgfDiff[nCount * 4 + k];

                            if (lossType == MultiBoxLossParameter.LocLossType.SMOOTH_L1)
                            {
                                float fAbsVal = Math.Abs(fVal);

                                if (fAbsVal < 1.0f)
                                    dfLoss += 0.5 * fVal * fVal;
                                else
                                    dfLoss += fAbsVal - 0.5;
                            }
                            else if (lossType == MultiBoxLossParameter.LocLossType.L2)
                            {
                                dfLoss += 0.5 * fVal * fVal;
                            }
                            else
                            {
                                m_log.FAIL("Unknown loc loss type!");
                            }
                        }

                        rgLocLoss[j] = (float)dfLoss;
                        nCount++;
                    }
                }

                rgLocAllLoss.Add(rgLocLoss);
            }

            return rgLocAllLoss;
        }
    }
}
