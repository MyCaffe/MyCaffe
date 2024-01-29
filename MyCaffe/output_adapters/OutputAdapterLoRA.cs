using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// WORK IN PROGRESS
namespace MyCaffe.output_adapters
{
    /// <summary>
    /// The OutputAdapterLoRA is used to adapt the output of a layer using the LoRA (Low Rank Adaptation) method.
    /// </summary>
    /// <remarks>
    /// @see [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, 2023
    /// </remarks>"
    /// <typeparam name="T"></typeparam>
    public class OutputAdapterLoRA<T> : OutputAdapter<T>
    {
        Layer<T> m_dropout = null;
        Blob<T> m_blobDrop;
        Blob<T> m_blobxA;
        Blob<T> m_blobxAB;
        Blob<T> m_blobWork;
        double m_dfScale = 1.0;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">OutputAdapter parameter that defines the adapter settings.</param>
        public OutputAdapterLoRA(CudaDnn<T> cuda, Log log, OutputAdapterParameter p) : base (cuda, log, p)
        {
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        protected override void dispose()
        {
            if (m_dropout != null)
            {
                m_dropout.Dispose();
                m_dropout = null;
            }

            if (m_blobDrop != null)
            {
                m_blobDrop.Dispose();
                m_blobDrop = null;
            }

            if (m_blobxA != null)
            {
                m_blobxA.Dispose();
                m_blobxA = null;
            }

            if (m_blobxAB != null)
            {
                m_blobxAB.Dispose();
                m_blobxAB = null;
            }

            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Setup the output adapter. This method is called just after the layer Setup method is called.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="colBottom">Specifies the input data.</param>
        /// <param name="colTop">Specifies the output data.</param>
        public override void Setup(LayerParameter p, BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (p.type != LayerParameter.LayerType.INNERPRODUCT)
                throw new Exception("The LoRA output adapter currently only supports the InnerProduct layer type.");

            if (m_param.dropout_ratio > 0)
            {
                LayerParameter pDropout = new LayerParameter(LayerParameter.LayerType.DROPOUT, p.name + ".LoRA.dropout");
                pDropout.dropout_param.dropout_ratio = m_param.dropout_ratio;
                m_dropout = Layer<T>.Create(m_cuda, m_log, pDropout, null);
                m_dropout.Setup(colBottom, colTop);

                m_blobDrop = new Blob<T>(m_cuda, m_log);
                m_blobDrop.Name = p.name + ".LoRA.dropout";
                m_blobDrop.ReshapeLike(colBottom[0]);
            }

            int nNumInput = colBottom[0].count(p.inner_product_param.axis);
            int nNumOutput = (int)p.inner_product_param.num_output;

            // LoRA_a
            Blob<T> blob = new Blob<T>(m_cuda, m_log);
            blob.Name = p.name + ".LoRA_a";

            List<int> rgShapeA = new List<int>() { (int)m_param.rank, nNumInput };
            blob.Reshape(rgShapeA);
            blob.SetData(0.0);
            blob.SetDiff(0.0);
            m_colBlobs.Add(blob);

            // LoRA_b
            blob = new Blob<T>(m_cuda, m_log);
            blob.Name = p.name + ".LoRA_b";

            List<int> rgShapeB = new List<int>() { nNumOutput, (int)m_param.rank };
            blob.Reshape(rgShapeB);
            blob.SetData(0.0);
            blob.SetDiff(0.0);
            m_colBlobs.Add(blob);

            m_blobxA = new Blob<T>(m_cuda, m_log);
            m_blobxA.Name = p.name + ".LoRA.xA";

            m_blobxAB = new Blob<T>(m_cuda, m_log); 
            m_blobxAB.Name = p.name + ".LoRA.xAB";

            m_blobWork = new Blob<T>(m_cuda, m_log);
            m_blobWork.Name = p.name + ".LoRA.work";

            m_dfScale = m_param.alpha / m_param.rank;
        }

        /// <summary>
        /// Reshape the output adapter. This method is called just after the layer's Reshape is called.
        /// </summary>
        /// <param name="colBottom">Specifies the input data.</param>
        /// <param name="colTop">Specifies the output data.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_dropout != null)
            {
                addBtmTop(colBottom[0], m_blobDrop);
                m_dropout.Reshape(colBottom, m_colTop);
            }
        }

        /// <summary>
        /// Forward propagate the output adapter. This method is called just after the layer's Forward is called.
        /// </summary>
        /// <param name="colBottom">Specifies the input data (which is the output of the layer's Forward call).</param>
        /// <param name="colTop">Specifies the output data passed to the next layer.</param>
        /// <remarks>
        /// As per the LoRA algorithm, the output is calculated via the following steps:
        ///   Out Update = (btm @ a^T @ b^T) * scale
        ///   New Result = btm + Out Update
        /// </remarks>
        public override void Forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm = colBottom[0];

            if (m_dropout != null)
            {
                addBtmTop(colBottom[0], m_blobDrop);
                m_dropout.Forward(colBottom, m_colTop);
                blobBtm = m_blobDrop;
            }

            m_blobxA.MatMul(blobBtm, m_colBlobs[0], true, false, true);
            m_blobxAB.MatMul(m_blobxA, m_colBlobs[1], false, false, true);
            m_blobxAB.scale_data(m_dfScale);

            m_cuda.add(colTop[0].count(), colTop[0].gpu_data, m_blobxAB.gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Backward propagate the output adapter. This method is called just before the layer's Backward is called.
        /// </summary>
        /// <param name="colTop">Specifies the input gradients.</param>
        /// <param name="rgbPropagateDown">Specifies what gets propagated.</param>
        /// <param name="colBottom">Specifies the output gradients (which are then the input gradients to the layer's Backward call).</param>
        /// <remarks>
        /// As per the LoRA algorithm, the gradients are calculated and then added to the original input gradients.  Keep in mind that
        /// both colBottom[0] and colTop[0] are of the same size.
        /// </remarks>
        public override void Backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_cuda.scale(colTop[0].count(), m_dfScale, colTop[0].gpu_diff, m_blobxAB.mutable_gpu_diff);

            m_blobxAB.MatMulGrad(m_blobxA, m_colBlobs[1], m_blobWork);
            m_blobxA.MatMulGrad(colBottom[0], m_colBlobs[0], m_blobWork);

            m_cuda.add(colTop[0].count(), colTop[0].gpu_diff, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
