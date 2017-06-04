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
    /// TripletLoss Layer
    /// original C++ code added by Binbin Xu
    /// declanxu@gmail.com or declanxu@126.com
    /// Zhejiang University, State Key Lab of CAD&CG
    /// 
    /// This layer is initialized with the MyCaffe.param.TripletLossSimpleParameter.
    /// </summary>
    /// <remarks>
    /// * Original implemetation at https://github.com/freesouls/caffe
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, and Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TripletLossSimpleLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobDiff;
        Blob<T> m_blobSub;
        int m_nInnerNum;
        int m_nLabelSeparator;
        int m_nBatchSize;
        double m_dfAlpha;
        Blob<T> m_blobMiddle;
        Blob<T> m_blobDeviceScalar;
        Blob<T> m_blobDeviceTmp;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides TripletLossParameter triplet_loss_param</param>
        public TripletLossSimpleLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TRIPLET_LOSS_SIMPLE;
            m_blobDiff = new Blob<T>(m_cuda, m_log);
            m_blobDiff.Name = "diff";
            m_blobSub = new Blob<T>(m_cuda, m_log);
            m_blobSub.Name = "sub";
            m_blobMiddle = new Blob<T>(m_cuda, m_log, false);
            m_blobDeviceScalar = new Blob<T>(m_cuda, m_log, false);
            m_blobDeviceTmp = new Blob<T>(m_cuda, m_log, false);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobDiff.Dispose();
            m_blobSub.Dispose();
            m_blobMiddle.Dispose();
            m_blobDeviceScalar.Dispose();
            m_blobDeviceTmp.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobDiff);
                col.Add(m_blobSub);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data, label
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: loss
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
            base.LayerSetUp(colBottom, colTop);
            // get some parameters
            m_nLabelSeparator = m_param.triplet_loss_simple_param.separate;
            m_dfAlpha = m_param.triplet_loss_simple_param.alpha;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);
            // store (X_i^n - X_i^p)/N which can backpropagate to prev-layer directly.
            m_blobDiff.ReshapeLike(colBottom[0]);   // bottom[0] is batch_size*channels(128)*1*1
            m_blobSub.ReshapeLike(colBottom[0]);
            m_nInnerNum = colBottom[0].count(1);
            m_nBatchSize = colBottom[0].num;

            m_blobMiddle.Reshape(new List<int>() { m_nInnerNum });
            m_blobDeviceScalar.Reshape(new List<int>() { m_nInnerNum });
            m_blobDeviceTmp.Reshape(new List<int>() { m_nBatchSize });
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs.
        /// </param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$ 
        ///     the loss.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            double dfLoss = 0;
            long hBottomData = colBottom[0].gpu_data;
            float[] rgBottomLabel = convertF(colBottom[1].update_cpu_data());
            long hDiff = m_blobDiff.mutable_gpu_data;
            long hSub = m_blobSub.mutable_gpu_data;
            long hDiffDiff = m_blobDiff.mutable_gpu_diff; // store the diff.
            long hDevScalar = m_blobDeviceScalar.gpu_data;
            long hDevTemp = m_blobDeviceTmp.mutable_gpu_data;
            long hMiddle = m_blobMiddle.mutable_gpu_data;

            m_blobDiff.SetData(0);
            m_blobDiff.SetDiff(0);

            List<int> rgLabels = new List<int>();
            for (int i = 0; i < m_nBatchSize; i++)
            {
                rgLabels.Add((int)rgBottomLabel[i]);
            }

            int nCount = m_blobDiff.count();
            float[] rgfVal;

            m_blobDeviceScalar.SetData(1.0);
            int nN = m_nBatchSize * m_nInnerNum;

            for (int i = 0; i < m_nBatchSize; i++)
            {
                int nLabel = rgLabels[i];

                if (nLabel < m_nLabelSeparator)
                {
                    m_cuda.sub_and_dot(nCount, nN, m_nInnerNum, hBottomData, hBottomData, hSub, 0, i * m_nInnerNum, 0);
                    // original code has innerNum first then batchSize, but this appears to be a bug
                    // for columns are not summed properly. Reversing these sums the columns as expected.
                    // m_cuda.gemv(false, m_nInnerNum, m_nBatchSize, m_tOne, hSub, hDevScalar, m_tZero, hDevTemp);
                    m_cuda.gemv(false, m_nBatchSize, m_nInnerNum, m_tOne, hSub, hDevScalar, m_tZero, hDevTemp);
                    rgfVal = convertF(m_blobDeviceTmp.update_cpu_data());

                    double fMargin = 10000.0;
                    int tmp_k = -1;
                    int tmp_j = -1;
                    for (int j=0; j<m_nBatchSize; j++)
                    {
                        if (j != i && rgLabels[j] == nLabel) // j is the positive
                        {
                            for (int k=0; k<m_nBatchSize; k++)
                            {
                                if (rgLabels[k] != nLabel)  // k is the negative
                                {
                                    if (rgfVal[j] >= rgfVal[k])
                                    {
                                        dfLoss += rgfVal[j] + m_dfAlpha - rgfVal[k];
                                        m_cuda.sub(m_nInnerNum, hBottomData, hBottomData, hMiddle, k * m_nInnerNum, j * m_nInnerNum, 0);
                                        m_cuda.axpy(m_nInnerNum, m_tOne, hMiddle, hDiffDiff, 0, i * m_nInnerNum);
                                        break;
                                    }
                                    else
                                    {
                                        if (rgfVal[k] - rgfVal[j] <= 0.2)
                                        {
                                            dfLoss += rgfVal[j] + m_dfAlpha - rgfVal[k];
                                            m_cuda.sub(m_nInnerNum, hBottomData, hBottomData, hMiddle, k * m_nInnerNum, j * m_nInnerNum, 0);
                                            m_cuda.axpy(m_nInnerNum, m_tOne, hMiddle, hDiffDiff, 0, i * m_nInnerNum);
                                            break;
                                        }

                                        if (rgfVal[k] - rgfVal[j] < fMargin)
                                        {
                                            tmp_k = k;
                                            tmp_j = j;
                                            fMargin = rgfVal[k] - rgfVal[j];
                                        }
                                    }
                                }
                            }

                            if (fMargin < m_dfAlpha && tmp_k != -1)
                            {
                                dfLoss += rgfVal[tmp_j] + m_dfAlpha - rgfVal[tmp_k];
                                m_cuda.sub(m_nInnerNum, hBottomData, hBottomData, hMiddle, tmp_k * m_nInnerNum, tmp_j * m_nInnerNum, 0);
                                m_cuda.axpy(m_nInnerNum, m_tOne, hMiddle, hDiffDiff, 0, i * m_nInnerNum);
                            }
                        }
                    }
                }
            }

            dfLoss /= (2.0 * colBottom[0].num);
            colTop[0].SetData(dfLoss, 0);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                double dfTopDiff = convertD(colTop[0].GetDiff(0));
                if (dfTopDiff != 1.0)
                    m_log.WriteLine("The triplet loss top diff[0] is not 1.0, but instead = " + dfTopDiff.ToString());

                double dfScale = 2.0 * dfTopDiff / colBottom[0].num;
                m_cuda.scale(colBottom[0].count(), dfScale, m_blobDiff.gpu_diff, colBottom[0].mutable_gpu_diff);
            }
            else
            {
                m_log.FAIL("This layer should be back propagated to the previous layer.");
            }
        }
    }
}
