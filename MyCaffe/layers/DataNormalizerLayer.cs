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
    /// The DataNormalizerLayer normalizes the input data (and optionally label) based on the normalization operations specified in the layer parameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DataNormalizerLayer<T> : Layer<T>
    {
        BlobCollection<T> m_colWork1;
        BlobCollection<T> m_colWork2;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides DataNormalizerParameter data_normalizer_param
        /// with DataNormalizerLayer options:
        ///  - across_data_and_label (optional bool, default false).
        ///          when <i>true</i> normalizes across data range within both data and label.
        ///  - steps.
        ///          List of normalization steps to take, with each performed in the order
        ///          for which they are listed.  
        ///          Steps include:
        ///            center - subtract the average to center the data.
        ///            stdev - divide the standard deviation of the data.
        ///            range - fit the data into the range specified by 'output_min','output_max'
        ///            additive - add the current value to the previous.
        ///            returns - take the percentage change between the current and one previous.
        ///            log - take the log of the current value.
        ///  - ignore_ch (optional int).
        ///          List of channel indexes to ignore and NOT normalize.
        ///  - input_min (optional double, default 0)
        ///          Specifies the input minimum used with stationary data.  When both 'input_min' 
        ///          and 'input_max' are set to 0, the data min/max used to normalize is determined
        ///          dynamcially from the data itself.
        ///  - input_max (optional double, default 0)
        ///          Specifies the input minimum used with stationary data.  When both 'input_min' 
        ///          and 'input_max' are set to 0, the data min/max used to normalize is determined
        ///          dynamcially from the data itself.
        ///  - input_mean (optional double, default null)
        ///          When specified, used by the 'center' step.  When not specified, the input_mean
        ///          is determined dynamically from the data itself.
        ///  - input_stdev (optional double, default null)
        ///          When specified, used by the 'stdev' step.  When not specified, 
        ///          the input_stdev is determined dynamically from the data itself.
        ///  - output_min (optional double, default 0)
        ///          Specifies the output minimum used by the 'range' step.
        ///  - output_max (optional double, default 0)
        ///          Specifies the output maximum used by the 'range' step.
        /// </param>
        public DataNormalizerLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DATA_NORMALIZER;
            m_colWork1 = new BlobCollection<T>();
            m_colWork2 = new BlobCollection<T>();
        }

        /// <summary>
        /// Clean up any resources used.
        /// </summary>
        protected override void dispose()
        {
            if (m_colWork1 != null)
            {
                m_colWork1.Dispose();
                m_colWork1 = null;
            }

            if (m_colWork2 != null)
            {
                m_colWork2.Dispose();
                m_colWork2 = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Returns the minimum number of bottom blobs required: data
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of bottom blobs required: data, label
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the minimum number of top blobs required: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of top blobs required: data, label
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            DataNormalizerParameter p = m_param.data_normalizer_param;

            if (p.steps.Count == 0)
                m_log.WriteLine("WARNING: No normalization steps are specified, data will just pass through in its normal form.");

            if (p.steps.Contains(DataNormalizerParameter.NORMALIZATION_STEP.RANGE))
            {
                double dfRange = p.output_max - p.output_min;
                m_log.CHECK_GT(dfRange, 0, "The output data range must be greater than 0!");
            }

            if (p.steps.Contains(DataNormalizerParameter.NORMALIZATION_STEP.STDEV))
            {
                if (p.input_stdev.HasValue)
                    m_log.CHECK_NE(p.input_stdev.Value, 0, "The standard deviation cannot be zero!");
            }

            for (int i = 0; i < colBottom.Count; i++)
            {
                m_colWork1.Add(new Blob<T>(m_cuda, m_log, colBottom[i]));
                m_colWork2.Add(new Blob<T>(m_cuda, m_log, colBottom[i]));
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom.Count, colTop.Count, "The top and bottom must have the same number of blobs.");

            for (int i = 0; i < colBottom.Count; i++)
            {
                colTop[i].ReshapeLike(colBottom[i]);
                m_colWork1[i].ReshapeLike(colBottom[i]);
                m_colWork2[i].ReshapeLike(colBottom[i]);
            }
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 1 or 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the data.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the label (optional).
        /// </param>
        /// <param name="colTop">input blob vector (length 1 or 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the normalized data.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the normalized label (optional).
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            for (int i = 0; i < colBottom.Count; i++)
            {
                m_colWork1[i].CopyFrom(colBottom[i]);
                m_colWork2[i].CopyFrom(colBottom[i]);
            }

            foreach (DataNormalizerParameter.NORMALIZATION_STEP step in m_param.data_normalizer_param.steps)
            {
                switch (step)
                {
                    case DataNormalizerParameter.NORMALIZATION_STEP.LOG:
                        run_step_log(m_colWork1);
                        break;

                    case DataNormalizerParameter.NORMALIZATION_STEP.CENTER:
                        run_step_center(m_colWork1, m_colWork2);
                        break;

                    case DataNormalizerParameter.NORMALIZATION_STEP.STDEV:
                        run_step_stdev(m_colWork1, m_colWork2);
                        break;
                }
            }

            for (int i = 0; i < colTop.Count; i++)
            {
                colTop[i].CopyFrom(m_colWork1[i]);
            }
        }

        private void run_step_log(BlobCollection<T> col)
        {
            foreach (Blob<T> b in col)
            {
                m_cuda.log(b.count(), b.gpu_data, b.mutable_gpu_data);              
            }
        }

        private void run_step_center(BlobCollection<T> col1, BlobCollection<T> col2)
        {
            if (m_param.data_normalizer_param.input_mean.HasValue)
            {
                double dfMean = m_param.data_normalizer_param.input_mean.GetValueOrDefault(0);

                if (dfMean != 0)
                {
                    for (int i = 0; i < col1.Count; i++)
                    {
                        col1[i].add_scalar(-dfMean);
                    }
                }
            }
            else
            {
                for (int i = 0; i < col1.Count; i++)
                {
                    Blob<T> b1 = col1[i];
                    Blob<T> b2 = col2[i];
                    int nNum = b1.num;
                    int nSpatialCount = b1.count(2);

                    if (nSpatialCount > 1)
                    {
                        m_cuda.channel_max(b1.count(), b1.num, b1.channels, nSpatialCount, b1.gpu_data, b1.mutable_gpu_diff); // b1 = max
                        m_cuda.channel_min(b1.count(), b1.num, b1.channels, nSpatialCount, b1.gpu_data, b2.mutable_gpu_diff); // b2 = min
                        m_cuda.channel_sub(b1.count(), b1.num, b1.channels, nSpatialCount, b2.gpu_diff, b1.mutable_gpu_diff); // b1 = b1 - b2
                        m_cuda.mul_scalar(b1.count(), 1.0 / (double)nSpatialCount, b1.mutable_gpu_diff); // channel mean.
                        m_cuda.channel_sub(b1.count(), b1.num, b1.channels, nSpatialCount, b1.gpu_diff, b1.mutable_gpu_data); // centered by channel
                    }
                }
            }
        }

        private void run_step_stdev(BlobCollection<T> col1, BlobCollection<T> col2)
        {
            if (m_param.data_normalizer_param.input_stdev.HasValue)
            {
                double dfStdev = m_param.data_normalizer_param.input_stdev.GetValueOrDefault(0);

                if (dfStdev != 0 && dfStdev != 1)
                {
                    for (int i = 0; i < col1.Count; i++)
                    {
                        m_cuda.mul_scalar(col1[i].count(), 1.0 / dfStdev, col1[i].mutable_gpu_data);
                    }
                }
            }
            else
            {
                for (int i = 0; i < col1.Count; i++)
                {
                    Blob<T> b1 = col1[i];
                    Blob<T> b2 = col2[i];
                    int nNum = b1.num;
                    int nSpatialCount = b1.count(2);

                    if (nSpatialCount > 1)
                    {
                        m_cuda.channel_max(b1.count(), b1.num, b1.channels, nSpatialCount, b1.gpu_data, b1.mutable_gpu_diff); // b1 = max
                        m_cuda.channel_min(b1.count(), b1.num, b1.channels, nSpatialCount, b1.gpu_data, b2.mutable_gpu_diff); // b2 = min
                        m_cuda.channel_sub(b1.count(), b1.num, b1.channels, nSpatialCount, b2.gpu_diff, b1.mutable_gpu_diff); // b1 = b1 - b2
                        m_cuda.mul_scalar(b1.count(), 1.0 / (double)nSpatialCount, b1.mutable_gpu_diff);                      // xbar = channel mean.
                        m_cuda.channel_sub(b1.count(), b1.num, b1.channels, nSpatialCount, b1.gpu_diff, b2.mutable_gpu_data); // x - xbar
                        m_cuda.channel_mul(b1.count(), b1.num, b1.channels, nSpatialCount, b2.gpu_data, b2.mutable_gpu_data); // (x - xbar)^2
                        m_cuda.channel_sum(b1.count(), b1.num, b1.channels, nSpatialCount, b2.gpu_data, b2.mutable_gpu_diff); // Sum(x - xbar)^2
                        m_cuda.mul_scalar(b1.count(), 1.0 / (double)nSpatialCount, b2.mutable_gpu_diff);
                        m_cuda.sqrt(b1.count(), b2.gpu_diff, b2.mutable_gpu_data);                                            // Std-dev.

                        m_cuda.channel_div(b1.count(), b1.num, b1.channels, nSpatialCount, b2.gpu_data, b1.mutable_gpu_data);
                    }
                }
            }
        }

        /// @brief Not implemented.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            new NotImplementedException();
        }
    }
}
