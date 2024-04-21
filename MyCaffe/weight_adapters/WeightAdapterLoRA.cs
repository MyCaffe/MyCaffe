using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.fused_ops;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.weight_adapters
{
    /// <summary>
    /// The WeightAdapterLoRA is used to adapt the weight of a layer using the LoRA (Low Rank Adaptation) method.
    /// </summary>
    /// <remarks>
    /// @see [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, 2023
    /// @see [GitHub: cccntu/minLoRA](https://github.com/cccntu/minLoRA) by Jonathan Chang, 2023, GitHub.
    /// </remarks>"
    /// <typeparam name="T"></typeparam>
    public class WeightAdapterLoRA<T> : WeightAdapter<T>
    {
        Layer<T> m_dropout = null;
        Blob<T> m_blobDrop;
        Blob<T> m_blobC;
        double m_dfScale = 1.0;
        List<bool> m_rgbProp = new List<bool>() { true };

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">OutputAdapter parameter that defines the adapter settings.</param>
        /// <param name="phase">Specifies the phase over which the weight adapter will run.</param>
        public WeightAdapterLoRA(CudaDnn<T> cuda, Log log, WeightAdapterParameter p, Phase phase) : base(cuda, log, p)
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

            dispose(ref m_blobDrop);
            dispose(ref m_blobC);

            base.dispose();
        }

        /// <summary>
        /// Setup the weight adapter. This method is called just after the layer Setup method is called.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="wt">Specifies the input data (which is the output of the layer's Forward call).</param>
        public override void Setup(LayerParameter p, Blob<T> wt)
        {
            if (p.type != LayerParameter.LayerType.INNERPRODUCT && p.type != LayerParameter.LayerType.LINEAR)
                throw new Exception("The LoRA output adapter currently only supports the InnerProduct and Linear layer types.");

            if (m_param.dropout_ratio > 0)
            {
                LayerParameter pDropout = new LayerParameter(LayerParameter.LayerType.DROPOUT, p.name + ".LoRA.dropout");
                pDropout.dropout_param.dropout_ratio = m_param.dropout_ratio;
                m_dropout = Layer<T>.Create(m_cuda, m_log, pDropout, null);
                addBtmTop(wt, wt);
                m_dropout.Setup(m_colBtm, m_colTop);

                m_blobDrop = createIntraLayerBlob("lora_drop", true, true);
                //m_blobDrop = new Blob<T>(m_cuda, m_log);
                //m_blobDrop.Name = p.name + ".LoRA.dropout";
                m_blobDrop.ReshapeLike(wt);
            }

            int nFanIn = wt.shape(0);
            int nFanOut = wt.shape(1);

            // LoRA_a
            Blob<T> blob = new Blob<T>(m_cuda, m_log);
            blob.Name = p.name + ".LoRA_a";
            blob.blob_type = BLOB_TYPE.WEIGHT;

            List<int> rgShapeA = new List<int>() { (int)m_param.rank, nFanIn };

            if (!shareParameter(blob, rgShapeA))
            {
                blob.Reshape(rgShapeA);

                // Mimic kaiming_uniform(a=5) initialization.
                float fRange = (float)Math.Sqrt(6.0 / ((1 * 25) * nFanIn));
                FillerParameter fp = new FillerParameter("uniform", 0, 0, fRange);
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(blob);

                blob.SetDiff(0.0);
            }
            m_colBlobs.Add(blob);

            // LoRA_b
            blob = new Blob<T>(m_cuda, m_log);
            blob.Name = p.name + ".LoRA_b";
            blob.blob_type = BLOB_TYPE.WEIGHT;

            List<int> rgShapeB = new List<int>() { nFanOut, (int)m_param.rank };
            if (!shareParameter(blob, rgShapeB))
            {
                blob.Reshape(rgShapeB);
                blob.SetData(0);
                blob.SetDiff(0.0);
            }
            m_colBlobs.Add(blob);

            m_blobC = createIntraLayerBlob("lora_c");
            //m_blobC = new Blob<T>(m_cuda, m_log);
            //m_blobC.Name = p.name + ".LoRA.C";

            m_dfScale = m_param.alpha / m_param.rank;
        }

        /// <summary>
        /// Reshape the weight adapter. This method is called just after the layer's Reshape is called.
        /// </summary>
        /// <param name="wt">Specifies the input data (which is the output of the layer's Forward call).</param>
        public override void Reshape(Blob<T> wt)
        {
            if (m_dropout != null)
            {
                addBtmTop(wt, m_blobDrop);
                m_dropout.Reshape(m_colBtm, m_colTop);
            }
        }

        /// <summary>
        /// Forward propagate the output adapter. This method is called just after the layer's Forward is called.
        /// </summary>
        /// <param name="wt">Specifies the input weights (which is used by the Forward call after being altered).</param>
        /// <remarks>
        /// As per the LoRA algorithm, the output is calculated via the following steps:
        ///   Out Update = (btm @ a^T @ b^T) * scale
        ///   New Result = btm + Out Update
        /// </remarks>
        public override long Forward(Blob<T> wt)
        {
            Blob<T> blobA = m_colBlobs[0];
            Blob<T> blobB = m_colBlobs[1];
            int nNumAxes = wt.num_axes;

            if (nNumAxes != 2)
                throw new Exception("The LoRA output adapter currently only supports 2D data.");

            if (m_dropout != null)
            {
                addBtmTop(blobA, m_blobDrop);
                m_dropout.Forward(m_colBtm, m_colTop);
                blobA = m_blobDrop;
            }

            blobA.Unsqueeze(0);
            blobB.Unsqueeze(0);
            blobA.Unsqueeze(0);
            blobB.Unsqueeze(0);

            m_blobC.MatMul(blobB, blobA, true);
            m_blobC.scale_data(m_dfScale);
            m_cuda.add(wt.count(), wt.gpu_data, m_blobC.gpu_data, m_blobC.mutable_gpu_data);

            blobA.Squeeze(0);
            blobB.Squeeze(0);
            blobA.Squeeze(0);
            blobB.Squeeze(0);

            return m_blobC.gpu_data;
        }

        /// <summary>
        /// Returns the weight blob.
        /// </summary>
        public override Blob<T> Weight
        {
            get { return m_blobC; }
        }

        /// <summary>
        /// Backward propagate the output adapter. This method is called just before the layer's Backward is called.
        /// </summary>
        /// <param name="colBtm">Specifies the input data (input to the Forward pass).</param>
        /// <param name="colTop">Specifies the output data (output by the Forward pass).</param>
        /// <param name="wt">Specifies the weight blob.</param>
        /// <remarks>
        /// As per the LoRA algorithm, the gradients are calculated and then added to the original input gradients.  
        /// </remarks>
        public override long Backward(BlobCollection<T> colTop, BlobCollection<T> colBtm, Blob<T> wt)
        {
            Blob<T> blobA = m_colBlobs[0];
            Blob<T> blobB = m_colBlobs[1];

            blobA.Unsqueeze(0);
            blobB.Unsqueeze(0);
            blobA.Unsqueeze(0);
            blobB.Unsqueeze(0);

            bool bSqueeze = false;
            if (m_blobC.num_axes == 2)
            {
                bSqueeze = true;
                m_blobC.Unsqueeze(0);
                m_blobC.Unsqueeze(0);
            }

            m_cuda.scale(wt.count(), m_dfScale, wt.gpu_diff, m_blobC.mutable_gpu_diff);
            m_blobC.MatMulGrad(blobB, blobA);

            if (m_blobC.num_axes == 4 && bSqueeze)
            {
                m_blobC.Squeeze(0);
                m_blobC.Squeeze(0);
            }

            blobA.Squeeze(0);
            blobB.Squeeze(0);
            blobA.Squeeze(0);
            blobB.Squeeze(0);

            if (m_dropout != null)
            {
                addBtmTop(blobA, m_blobDrop);
                m_dropout.Backward(m_colTop, m_rgbProp, m_colBtm);
            }

            return wt.gpu_diff;
        }
    }
}
