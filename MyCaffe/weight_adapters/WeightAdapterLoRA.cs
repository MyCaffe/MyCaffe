﻿using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.output_adapters
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
        List<int> m_rgShape = new List<int>() { 1, 1, 1, 1, };

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">OutputAdapter parameter that defines the adapter settings.</param>
        public WeightAdapterLoRA(CudaDnn<T> cuda, Log log, WeightAdapterParameter p) : base(cuda, log, p)
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

            if (m_blobC != null)
            {
                m_blobC.Dispose();
                m_blobC = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Setup the weight adapter. This method is called just after the layer Setup method is called.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="wt">Specifies the input data (which is the output of the layer's Forward call).</param>
        public override void Setup(LayerParameter p, Blob<T> wt)
        {
            if (p.type != LayerParameter.LayerType.INNERPRODUCT)
                throw new Exception("The LoRA output adapter currently only supports the InnerProduct layer type.");

            if (m_param.dropout_ratio > 0)
            {
                LayerParameter pDropout = new LayerParameter(LayerParameter.LayerType.DROPOUT, p.name + ".LoRA.dropout");
                pDropout.dropout_param.dropout_ratio = m_param.dropout_ratio;
                m_dropout = Layer<T>.Create(m_cuda, m_log, pDropout, null);
                addBtmTop(wt, wt);
                m_dropout.Setup(m_colBtm, m_colTop);

                m_blobDrop = new Blob<T>(m_cuda, m_log);
                m_blobDrop.Name = p.name + ".LoRA.dropout";
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

            m_blobC = new Blob<T>(m_cuda, m_log);
            m_blobC.Name = p.name + ".LoRA.C";

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

        private void unsqueeze(Blob<T> b)
        {
            if (b.num_axes == 4)
                return;

            m_rgShape.Clear();
            m_rgShape.Add(1);
            m_rgShape.Add(1);
            m_rgShape.Add(b.shape(0));
            m_rgShape.Add(b.shape(1));
            b.Reshape(m_rgShape);
        }

        private void squeeze(Blob<T> b)
        {
            if (b.num_axes != 4)
                return;

            m_rgShape.Clear();
            m_rgShape.Add(b.shape(2));
            m_rgShape.Add(b.shape(3));
            b.Reshape(m_rgShape);
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

            unsqueeze(blobA);
            unsqueeze(blobB);

            m_blobC.MatMul(blobB, blobA, true);
            m_blobC.scale_data(m_dfScale);

            squeeze(m_blobC);
            squeeze(blobA);
            squeeze(blobB);

            m_cuda.add(wt.count(), wt.gpu_data, m_blobC.gpu_data, m_blobC.mutable_gpu_data);

            return m_blobC.gpu_data;
        }

        /// <summary>
        /// Backward propagate the output adapter. This method is called just before the layer's Backward is called.
        /// </summary>
        /// <param name="colBtm">Specifies the input data (input to the Forward pass).</param>
        /// <param name="colTop">Specifies the output data (output by the Forward pass).</param>
        /// <param name="wt">Specifies the input gradients (which is used by the Backward call after being altered).</param>
        /// <remarks>
        /// As per the LoRA algorithm, the gradients are calculated and then added to the original input gradients.  
        /// </remarks>
        public override long Backward(BlobCollection<T> colTop, BlobCollection<T> colBtm, Blob<T> wt)
        {
            Blob<T> blobA = m_colBlobs[0];
            Blob<T> blobB = m_colBlobs[1];
            int nNumAxes = wt.num_axes;

            if (nNumAxes != 2)
                throw new Exception("The LoRA output adapter currently only supports 2D data.");

            unsqueeze(blobA);
            unsqueeze(blobB);
            unsqueeze(m_blobC);

            m_cuda.scale(wt.count(), m_dfScale, wt.gpu_diff, m_blobC.mutable_gpu_diff);
            m_blobC.MatMulGrad(blobB, blobA);

            squeeze(blobA);
            squeeze(blobB);

            if (m_dropout != null)
            {
                addBtmTop(blobA, m_blobDrop);
                m_dropout.Backward(m_colTop, m_rgbProp, m_colBtm);
            }

            return wt.gpu_diff;
        }
    }
}