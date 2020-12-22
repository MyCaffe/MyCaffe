using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The ConstantLayer provides a layer that just outputs a constant value.
    /// This layer is initialized with the MyCaffe.param.ConstantParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ConstantLayer<T> : Layer<T>
    {
        BlobShape m_shape;
        List<float> m_rgF;

        /// <summary>
        /// The ConstantLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Constant with parameter constant_param</param>
        public ConstantLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CONSTANT;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the input blobs, which are not used.</param>
        /// <param name="colTop">Specifies the output blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_shape = m_param.constant_param.output_shape;
            m_rgF = m_param.constant_param.values_f;

            if (!string.IsNullOrEmpty(m_param.constant_param.binary_data_file))
            {
                m_log.CHECK(File.Exists(m_param.constant_param.binary_data_file), "The 'binary_data_file' specified ('" + m_param.constant_param.binary_data_file + "') could not be found!");

                using (FileStream fs = new FileStream(m_param.constant_param.binary_data_file, FileMode.Open, FileAccess.Read))
                using (BinaryReader br = new BinaryReader(fs))
                {
                    BlobProto proto = BlobProto.Load(br);
                    m_rgF = proto.data;
                }
            }
        }

        /// <summary>
        /// Reshape the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the input blobs, which are not used.</param>
        /// <param name="colTop">Specifies the output blobs which are reshaped to the output_shape.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].Reshape(m_param.constant_param.output_shape);
            T[] rgData = Utility.ConvertVec<T>(m_rgF.ToArray());
            colTop[0].SetData(rgData);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector - not used.</param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     where NCHW are specified by the 'output_shape'.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - constant Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {            
        }
    }
}
