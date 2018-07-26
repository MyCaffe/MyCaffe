using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The ArgMaxLayer computes the index of the K max values for each datum across
    /// all dimensions @f$ (C \times H \times W) @f$.
    /// This layer is initialized with the MyCaffe.param.ArgMaxParameter.
    /// </summary>
    /// <remarks>
    /// Intended for use after a classification layer to produce a prediction.
    /// If parameter out_max_val is set to true, output is a vector of pairs
    /// (max_ind, max_val) for each image.  The axis parameter specifies an axis
    /// along which to maximize.
    /// 
    /// @see [Detecting Unexpected Obstacles for Self-Driving Cars: Fusing Deep Learning and Geometric Modeling](https://arxiv.org/abs/1612.06573v1) by Sebastian Ramos, Stefan Gehrig, Peter Pinggera, Uwe Franke, and Carsten Rother, 2016. 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ArgMaxLayer<T> : Layer<T>
    {
        bool m_bOutMaxVal;
        int m_nTopK;
        int? m_nAxis = null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides ArgMaxParameter argmax_param
        /// with ArgMaxLayer options:
        ///  - top_k (optional uint, default 1).
        ///          the number K of maximal items to output.
        ///  - out_max_val (optional bool, default false).
        ///          if set, output a vector of pairs (max_ind, max_val) unless axis is set then
        ///          output max_val along the specified axis.
        ///  - axis (optional int).
        ///          if set, maximise along the specified axis else maximise the flattened
        ///          trailing dimensions for each indes of the first / num dimension.
        /// </param>
        public ArgMaxLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ARGMAX;
        }

        /// <summary>
        /// Returns the exact number of bottom blobs required: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of top blobs required: argmax
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
            ArgMaxParameter p = m_param.argmax_param;
            m_bOutMaxVal = p.out_max_val;
            m_nTopK = (int)p.top_k;
            m_nAxis = null;

            m_log.CHECK_GE(m_nTopK, 1, "Top k must not be less than 1.");

            if (p.axis.HasValue)
            {
                m_nAxis = colBottom[0].CanonicalAxisIndex(p.axis.Value);
                m_log.CHECK_GE(m_nAxis.Value, 0, "axis must not be less than zero.");
                m_log.CHECK_LE(m_nAxis.Value, colBottom[0].num_axes, "axis must be less tahn or equal to the dimension of the axis.");
                m_log.CHECK_LE(m_nTopK, colBottom[0].shape(m_nAxis.Value), "top_k must be less than or equal to the dimension of the axis.");
            }
            else
            {
                m_log.CHECK_LE(m_nTopK, colBottom[0].count(1), "top_k must be less than or equal to the dimension of the flattened bottom blob per instance.");
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumTopAxes = colBottom[0].num_axes;

            if (nNumTopAxes < 3)
                nNumTopAxes = 3;

            List<int> rgShape = Utility.Create<int>(nNumTopAxes, 1);

            if (m_nAxis.HasValue)
            {
                // Produces max_ind or max_val per axis
                rgShape = Utility.Clone<int>(colBottom[0].shape());
                rgShape[m_nAxis.Value] = m_nTopK;
            }
            else
            {
                rgShape[0] = colBottom[0].shape(0);
                // Produces max_ind
                rgShape[2] = m_nTopK;

                if (m_bOutMaxVal)
                {
                    // Produces max_ind and max_val
                    rgShape[1] = 2;
                }
            }

            colTop[0].Reshape(rgShape);
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs.
        /// <param name="colTop">output blob vector (length 1)
        ///  -# @f$ (N \times 1 \times K) @f$ or, if out_max_val
        ///     @f$ (N \times 2 \times K) @f$ unless axis set then e.g.
        ///     @f$ (N \times K \times H \times W) @f$ if axis == 1
        ///     the computed outputs @f$
        ///       y_n = \arg\max\limits_i x_{ni}
        ///       (for k = 1)
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            double[] rgBottomData = convertD(colBottom[0].update_cpu_data());
            double[] rgTopData = convertD(colTop[0].mutable_cpu_data);
            int nDim;
            int nAxisDist;

            if (m_nAxis.HasValue)
            {
                nDim = colBottom[0].shape(m_nAxis.Value);
                // Distance between values of axis in blob.
                nAxisDist = colBottom[0].count(m_nAxis.Value) / nDim;
            }
            else
            {
                nDim = colBottom[0].count(1);
                nAxisDist = 1;
            }

            int nNum = colBottom[0].count() / nDim;

            for (int i = 0; i < nNum; i++)
            {
                List<KeyValuePair<double, int>> rgBottomDataPair = new List<KeyValuePair<double, int>>();

                for (int j = 0; j < nDim; j++)
                {
                    int nIdx = (i / nAxisDist * nDim + j) * nAxisDist + i % nAxisDist;
                    rgBottomDataPair.Add(new KeyValuePair<double, int>(rgBottomData[nIdx], j));
                }

                rgBottomDataPair.Sort(new Comparison<KeyValuePair<double, int>>(sortDataItems));

                for (int j = 0; j < m_nTopK; j++)
                {
                    if (m_bOutMaxVal)
                    {
                        if (m_nAxis.HasValue)
                        {
                            // Produces max_val per axis
                            int nIdx = (i / nAxisDist * m_nTopK + j) * nAxisDist + i % nAxisDist;
                            rgTopData[nAxisDist] = rgBottomDataPair[j].Key;
                        }
                        else
                        {
                            // Produces max_ind and max_val
                            int nIdx1 = 2 * i * m_nTopK + j;
                            rgTopData[nIdx1] = rgBottomDataPair[j].Value;
                            int nIdx2 = 2 * i * m_nTopK + m_nTopK + j;
                            rgTopData[nIdx2] = rgBottomDataPair[j].Key;
                        }
                    }
                    else
                    {
                        // Produces max_ind per axis.
                        int nIdx = (i / nAxisDist * m_nTopK + j) * nAxisDist + i % nAxisDist;
                        rgTopData[nIdx] = rgBottomDataPair[j].Value;
                    }
                }
            }

            colTop[0].mutable_cpu_data = convert(rgTopData);
        }

        private int sortDataItems(KeyValuePair<double, int> a, KeyValuePair<double, int> b)
        {
            if (a.Key < b.Key)
                return 1;

            if (a.Key > b.Key)
                return -1;

            return 0;
        }

        /// @brief Not implemented.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            new NotImplementedException();
        }
    }
}
