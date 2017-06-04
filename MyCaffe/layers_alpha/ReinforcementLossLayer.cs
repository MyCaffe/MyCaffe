using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// Computes the multinomial logistic loss for a one-of-many
    /// classification task, passing real-valued predictions through a
    /// softmax to get a probability distribution over classes.
    /// 
    /// This layer is initialized with the MyCaffe.param.ReinforcementLossParameter.
    /// </summary>
    /// <remarks>
    /// This layer should be preferred over separate
    /// SofmaxLayer + MultinomialLogisticLossLayer
    /// as its gradient computation is more numerically stable.
    /// At test time, this layer can be replaced simply by a SofmaxLayer.
    /// 
    /// @see [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274v2) by Yuxi Li, 2017.
    /// @see [Reinforcement learning with raw image pixels as input state](http://www.montefiore.ulg.ac.be/services/stochastic/pubs/2006/EMW06/ernst-iwicpas-2006.pdf) by Damien Ernst, Raphaël Marée, and Louis Wehenkel, 2006.
    /// @see [Self-Optimizing Memory Controllers: A Reinforcement Learning Approach](https://www.csl.cornell.edu/~martinez/doc/isca08.pdf) by Engin Ipek, Onur Mutlu, José F. Martínez, and Rich Caruana, 2008.
    /// @see [Deep Auto-Encoder Neural Networks in Reinforcement Learning](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.172.1873&rank=1) by Sascha Lange and Martin Riedmiller, 2010.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ReinforcementLossLayer<T> : Layer<T>
    {
        TransferInput.fnGetInputData m_fnGetInput = null;


        /// <summary>
        /// The ReinforcementLossLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LossParameter of type REINFORCEMENT_LOSS with param reinforcement_loss_param.
        /// </param>
        /// <param name="fnGet">Specifies the function used to retrieve the reinforcement data.</param>
        public ReinforcementLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p, TransferInput.fnGetInputData fnGet)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.REINFORCEMENT_LOSS;
            m_fnGetInput = fnGet;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // LossLayers have non-zero (1) loss by default.
            if (m_param.loss_weight.Count == 0)
                m_param.loss_weight.Add(1.0);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector.</param>
        /// <param name="colTop">bottom output Blob vector.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom[0].count(), colTop[0].count(), "The top and bottom should have the same count.");
            m_cuda.copy(colBottom[0].count(), colBottom[0].gpu_data, colTop[0].gpu_data);
        }

        /// <summary>
        /// Computes the loss error gradient w.r.t the output.
        /// </summary>
        /// <param name="colTop">top output blob vector, 
        /// providing the error gradient with respect to the outputs.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels.</param>
        /// <param name="colBottom">bottom input blob vector
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // Get input which should be a list of int's where
            //  each entry is the input image index for each 
            //  batch item input.
            BatchInput bi = m_fnGetInput();
            List<int> rgLastInput = bi.InputData as List<int>;

            m_log.CHECK(rgLastInput != null, "The last input should be of type List<int> and should not be null!");
            m_log.CHECK_EQ(colTop[0].num, rgLastInput.Count, "The last input should have the same number of items in top[0].");

            int nBatchSize = colTop[0].num;
            int nNumOutput = colTop[0].channels;
            T[] rgTop = colTop[0].update_cpu_data();
            T[] rgBottom = new T[rgTop.Length];

            for (int i = 0; i < nBatchSize; i++)
            {
                // Get the maximum output
                float fPreQ = -float.MaxValue;
                int nPreQIdx = -1;

                for (int j = 0; j < nNumOutput; j++)
                {
                    int nIdx = (i * nNumOutput) + j;
                    float fOutput = (float)Convert.ChangeType(rgTop[nIdx], typeof(float));

                    if (fOutput > fPreQ)
                    {
                        nPreQIdx = nIdx;
                        fPreQ = fOutput;
                    }
                }

                // Get the reinforcement info.
                BatchInformationCollection col = m_param.reinforcement_loss_param.BatchInfoCollection;

                // Set the maximum output to: maxout = (terminal) ? R : R + lambda * qmax1
                // Set all other outputs to zero.

                float fTarget = (float)Convert.ChangeType(rgTop[nPreQIdx], typeof(float));

                if (col != null)
                {
                    BatchInformation batchInfo = col[bi.BatchIndex];
                    BatchItem batchItem = batchInfo[i];

                    // clip to range of -1,1
                    fTarget = clip((float)batchItem.Reward, -0.9f, 0.9f);

                    if (!batchItem.Terminal)
                        fTarget += (float)(m_param.reinforcement_loss_param.discount_rate * batchItem.QMax1);
                }

                float fDiff = fTarget - fPreQ;
                float fDelta = clip(fDiff, -0.9f, 0.9f);
                rgBottom[nPreQIdx] = (T)Convert.ChangeType(fDelta, typeof(T));
            }

            colBottom[0].mutable_cpu_diff = rgBottom;
        }

        float clip(float fVal, float fMin, float fMax)
        {
            if (fVal >= 0 && fVal < 0.01)
                return 0.01f;

            if (fVal <= 0 && fVal > -0.01)
                return -0.01f;

            if (fVal < fMin)
                return fMin;

            if (fVal > fMax)
                return fMax;

            return fVal;
        }
    }
}
