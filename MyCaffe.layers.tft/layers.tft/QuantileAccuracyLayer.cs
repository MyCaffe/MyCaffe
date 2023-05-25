using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The QuantileAccuracyLayer implements the Quantile Accuracy Layer used in TFT models.
    /// </summary>
    /// <remarks>
    /// Each target value is compared against the predicted center value +/- each quantile accuracy range.  The number of target values falling within the center +/- each quantile accuracy range defines the accuracy.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class QuantileAccuracyLayer<T> : Layer<T>
    {
        float m_fLastAverage = 0;
        Dictionary<float, List<float>> m_rgAccuracies = new Dictionary<float, List<float>>();
        Dictionary<float, List<float>> m_rgAcccuracyAverages = new Dictionary<float, List<float>>();
        Dictionary<float, int> m_rgWithinTargetCounts;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public QuantileAccuracyLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.QUANTILE_ACCURACY;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x, target
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: one per accuracy range, each containing an accuracy % value for the accuracy range.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return m_param.quantile_accuracy_param.accuracy_ranges.Count; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nN = colBottom[0].num;
            int nC = colBottom[0].channels;
            m_log.CHECK_EQ(colBottom[0].height, 3, "Currently, the Quantile Accuracy Layer only supports 3 quantile predictions (upper, center, lower).");

            m_rgWithinTargetCounts = new Dictionary<float, int>(nC);

            foreach (float fRange in m_param.quantile_accuracy_param.accuracy_ranges)
            {
                m_rgAccuracies.Add(fRange, new List<float>(nN));
                m_rgAcccuracyAverages.Add(fRange, new List<float>((int)m_param.quantile_accuracy_param.average_period));
                m_rgWithinTargetCounts.Add(fRange, 0);
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgShape = new List<int>() { 1 };

            foreach (Blob<T> blobTop in colTop)
            {
                blobTop.Reshape(rgShape);
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nN = colBottom[0].num;
            int nC = colBottom[0].channels;
            int nQ = colBottom[0].height;
            float[] rgX = convertF(colBottom[0].update_cpu_data());
            float[] rgTgt = convertF(colBottom[1].update_cpu_data());

            if (nQ != 3)
                throw new Exception("There should only be 3 quantile predictions (upper, center, lower).");

            foreach (KeyValuePair<float, List<float>> kv in m_rgAccuracies)
            {
                kv.Value.Clear();
            }

            for (int i = 0; i < nN; i++)
            {
                foreach (float fRange in m_param.quantile_accuracy_param.accuracy_ranges)
                {
                    m_rgWithinTargetCounts[fRange] = 0;
                }

                for (int c = 0; c < nC; c++)
                {
                    int nIdx = i * nC * nQ + c * nQ;
                    float fUpper = rgX[nIdx + 2];
                    float fCenter = rgX[nIdx + 1];
                    float fLower = rgX[nIdx];

                    float fUpperRange = Math.Abs(fUpper - fCenter);
                    float fLowerRange = Math.Abs(fCenter - fLower);

                    for (int r = 0; r < m_param.quantile_accuracy_param.accuracy_ranges.Count; r++)
                    {
                        float fUpperTarget = fCenter + fUpperRange * m_param.quantile_accuracy_param.accuracy_ranges[r];
                        float fLowerTarget = fCenter - fLowerRange * m_param.quantile_accuracy_param.accuracy_ranges[r];
                        float fTarget = rgTgt[i * nC + c];

                        if (fTarget <= fUpperTarget && fTarget >= fLowerTarget)
                            m_rgWithinTargetCounts[m_param.quantile_accuracy_param.accuracy_ranges[r]]++;
                    }
                }

                foreach (KeyValuePair<float, int> kvp in m_rgWithinTargetCounts)
                {
                    float fAccuracy = (float)kvp.Value / nC;
                    m_rgAccuracies[kvp.Key].Add(fAccuracy);
                }
            }

            int nIdx1 = 0;
            foreach (KeyValuePair<float, List<float>> kvp in m_rgAccuracies)
            {
                float fAccuracy = kvp.Value.Average();

                m_rgAcccuracyAverages[kvp.Key].Add(fAccuracy);
                if (m_rgAcccuracyAverages[kvp.Key].Count > m_param.quantile_accuracy_param.average_period)
                    m_rgAcccuracyAverages[kvp.Key].RemoveAt(0);
            }

            foreach (KeyValuePair<float, List<float>> kvp in m_rgAcccuracyAverages)
            {
                float fAveAccuracy = average(kvp.Value, m_param.quantile_accuracy_param.average_period);
                colTop[nIdx1].SetData(fAveAccuracy);
                nIdx1++;
            }
        }

        private float average(List<float> rg, uint nN)
        {
            if (rg.Count == 0)
                m_fLastAverage = 0;
            else if (rg.Count == 1)
                m_fLastAverage = rg[0];
            else if (rg.Count < nN)
            {
                m_fLastAverage = (((float)nN - 1) / (float)nN) * m_fLastAverage;
                m_fLastAverage += (1.0f / (float)nN) * rg[rg.Count - 1];
            }
            else
            {
                m_fLastAverage = rg.Average();
            }

            return m_fLastAverage;
        }

        /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
