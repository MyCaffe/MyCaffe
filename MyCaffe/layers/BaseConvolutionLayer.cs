using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.common;

namespace MyCaffe.layers
{
    /// <summary>
    /// The BaseConvolutionLayer is an abstract base class that factors out BLAS code common to
    /// ConvolutionLayer and DeconvolutionLayer
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
	public abstract class BaseConvolutionLayer<T> : Layer<T> 
	{
        /// <summary>
        /// The spatial dimensions of the filter kernel.
        /// </summary>
        protected Blob<T> m_blobKernelShape;
        /// <summary>
        /// The spatial dimensions of the stride.
        /// </summary>
        protected Blob<T> m_blobStride;
        /// <summary>
        /// The spatial dimensions of the padding.
        /// </summary>
        protected Blob<T> m_blobPad;
        /// <summary>
        /// The spatial dimentions of the dilation.
        /// </summary>
        protected Blob<T> m_blobDilation;
        /// <summary>
        /// The spatial dimensions of the convolution input.
        /// </summary>
        protected Blob<T> m_blobConvInputShape;
        /// <summary>
        /// The spatial dimensionss of the col_buffer.
        /// </summary>
        protected List<int> m_rgColBufferShape;
        /// <summary>
        /// The spatial dimensions of the output.
        /// </summary>
        protected List<int> m_rgOutputShape = new List<int>();
        /// <summary>
        /// The buttom shape.
        /// </summary>
        protected List<int> m_rgBottomShape = new List<int>();
        /// <summary>
        /// The number of spatial axes.
        /// </summary>
        protected int m_nNumSpatialAxes;
        /// <summary>
        /// The bottom dimension.
        /// </summary>
        protected int m_nBottomDim;
        /// <summary>
        /// The top dimension.
        /// </summary>
        protected int m_nTopDim;
        /// <summary>
        /// The channel axis.
        /// </summary>
        protected int m_nChannelAxis;
        /// <summary>
        /// The number of items in the batch.
        /// </summary>
        protected int m_nNum;
        /// <summary>
        /// The number of channels in each item.
        /// </summary>
        protected int m_nChannels;
        /// <summary>
        /// The group.
        /// </summary>
        protected int m_nGroup;
        /// <summary>
        /// The output spatial dimension.
        /// </summary>
        protected int m_nOutSpatialDim;
        /// <summary>
        /// The weight offset used.
        /// </summary>
        protected int m_nWeightOffset;
        /// <summary>
        /// The number of outputs.
        /// </summary>
        protected int m_nNumOutput;
        /// <summary>
        /// Whether or not to use bias.
        /// </summary>
        protected bool m_bBiasTerm;
        /// <summary>
        /// Whether or not the kernel is 1x1.
        /// </summary>
        protected bool m_bIs1x1;
        /// <summary>
        /// Whether or not to force n-dim 2 column.
        /// </summary>
        protected bool m_bForceNDim2col;

        int m_nNumKernelsIm2col;
        int m_nNumKernelsCol2im;
        int m_nConvOutChannels;
        int m_nConvInChannels;
        int m_nConvOutSpatialDim;
        int m_nKernelDim;
        int m_nColOffset;
        int m_nOutputOffset;

        Blob<T> m_blobColBuffer;
        Blob<T> m_blobBiasMultiplier;

        long m_hWorkspaceData = 0;
        long m_lWorkspaceSize = 0;
        bool m_bWorkspaceOwner = false;

        
        /// <summary>
        /// The BaseConvolutionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter.</param>
        public BaseConvolutionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_blobKernelShape = new Blob<T>(cuda, log);
            m_blobStride = new Blob<T>(cuda, log);
            m_blobPad = new Blob<T>(cuda, log);
            m_blobDilation = new Blob<T>(cuda, log);
            m_blobConvInputShape = new Blob<T>(cuda, log);

            m_blobColBuffer = new Blob<T>(cuda, log);
            m_blobColBuffer.Name = "conv_col_buffer";

            m_blobBiasMultiplier = new Blob<T>(cuda, log);
            m_blobBiasMultiplier.Name = "conv_bias_mult";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobKernelShape.Dispose();
            m_blobStride.Dispose();
            m_blobPad.Dispose();
            m_blobDilation.Dispose();
            m_blobConvInputShape.Dispose();

            m_blobColBuffer.Dispose();
            m_blobBiasMultiplier.Dispose();

            if (m_bWorkspaceOwner && m_hWorkspaceData != 0)
            {
                m_cuda.DisableGhostMemory();
                m_cuda.FreeMemory(m_hWorkspaceData);
                m_cuda.ResetGhostMemory();
                m_hWorkspaceData = 0;
                m_bWorkspaceOwner = false;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes) || reverse_dimensions())
                {
                    col.Add(m_blobColBuffer);
                    col.Add(m_blobBiasMultiplier);
                }

                return col;
            }
        }

        /// <summary>
        /// Retruns the WorkspaceArgs containing the workspace used by this Layer.
        /// </summary>
        /// <returns></returns>
        protected override WorkspaceArgs getWorkspace()
        {
            WorkspaceArgs args = base.getWorkspace();

            if (args != null)
                return args;

            m_bWorkspaceOwner = true;
            return new common.WorkspaceArgs(m_hWorkspaceData, m_lWorkspaceSize);
        }

        /// <summary>
        /// If not already set, allocates the workspace needed in GPU memory.
        /// </summary>
        /// <param name="lSize">Specifies the size (in items) of workspace needed.</param>
        /// <returns>This method always returns <i>true</i>.</returns>
        protected override bool setWorkspace(long lSize)
        {
            if (!m_bWorkspaceOwner && base.setWorkspace(lSize))
                return true;

            m_bWorkspaceOwner = true;

            if (lSize < m_lWorkspaceSize)
                return true;

            m_lWorkspaceSize = lSize;
            m_cuda.DisableGhostMemory();

            if (m_hWorkspaceData != 0)
                m_cuda.FreeMemory(m_hWorkspaceData);

            m_hWorkspaceData = m_cuda.AllocMemory(m_lWorkspaceSize);
            m_cuda.ResetGhostMemory();

            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Configure the kernel size, padding, stride and inputs.
            ConvolutionParameter p = m_param.convolution_param;

            m_bForceNDim2col = p.force_nd_im2col;
            m_nChannelAxis = colBottom[0].CanonicalAxisIndex(p.axis);

            int nFirstSpatialAxis = m_nChannelAxis + 1;
            int nNumAxes = colBottom[0].num_axes;

            m_nNumSpatialAxes = nNumAxes - nFirstSpatialAxis;

            m_log.CHECK_GE(m_nNumSpatialAxes, 0, "The number of spatial axes must be zero or greater.");

            List<int> rgBottomDimBlobShape = new List<int>() { m_nNumSpatialAxes + 1 };
            List<int> rgSpaitalDimBlobShape = new List<int>() { Math.Max(m_nNumSpatialAxes, 1) };

            // Setup filter kernel dimensions (blobKernelShape)
            m_blobKernelShape.Reshape(rgSpaitalDimBlobShape);
            T[] rgKernelShape = m_blobKernelShape.mutable_cpu_data;

            if (p.kernel_h.HasValue || p.kernel_w.HasValue)
            {
                m_log.CHECK_EQ(m_nNumSpatialAxes, 2, "kernel_h & kernel_w can only be used in 2D convolution.");
                m_log.CHECK_EQ(0, p.kernel_size.Count, "Either kernel_size or kernel_h/w should be specified; not both.");
                rgKernelShape[0] = (T)Convert.ChangeType(p.kernel_h.Value, typeof(T));
                rgKernelShape[1] = (T)Convert.ChangeType(p.kernel_w.Value, typeof(T));
            }
            else
            {
                int nNumKernelDims = p.kernel_size.Count;
                m_log.CHECK(nNumKernelDims == 1 || nNumKernelDims == m_nNumSpatialAxes, "Kernel size must be specified once, or once per spatial dimension (kernel_size specified " + nNumKernelDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");

                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    int nIdx = (nNumKernelDims == 1) ? 0 : i;
                    rgKernelShape[i] = (T)Convert.ChangeType(p.kernel_size[nIdx], typeof(T));
                }
            }

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                m_log.CHECK_GT((int)Convert.ChangeType(rgKernelShape[i], typeof(int)), 0, "Filter dimension must be non-zero.");
            }

            m_blobKernelShape.mutable_cpu_data = rgKernelShape;


            // Setup stride dimensions (blobStride)
            m_blobStride.Reshape(rgSpaitalDimBlobShape);
            T[] rgStrideData = m_blobStride.mutable_cpu_data;

            if (p.stride_h.HasValue || p.stride_w.HasValue)
            {
                m_log.CHECK_EQ(m_nNumSpatialAxes, 2, "stride_h & stride_w can only be used in 2D convolution.");
                m_log.CHECK_EQ(0, p.stride.Count, "Either stride_size or stride_h/w should be specified; not both.");
                rgStrideData[0] = (T)Convert.ChangeType(p.stride_h.Value, typeof(T));
                rgStrideData[1] = (T)Convert.ChangeType(p.stride_w.Value, typeof(T));
            }
            else
            {
                int nNumStrideDims = p.stride.Count;
                m_log.CHECK(nNumStrideDims == 0 || nNumStrideDims == 1 || nNumStrideDims == m_nNumSpatialAxes, "Stride size must be specified once, or once per spatial dimension (stride specified " + nNumStrideDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");
                int nDefaultStride = 1;

                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    if (nNumStrideDims == 0)
                    {
                        rgStrideData[i] = (T)Convert.ChangeType(nDefaultStride, typeof(T));
                    }
                    else
                    {
                        int nIdx = (nNumStrideDims == 1) ? 0 : i;
                        rgStrideData[i] = (T)Convert.ChangeType(p.stride[nIdx], typeof(T));
                    }
                    m_log.CHECK_GT((int)Convert.ChangeType(rgStrideData[i], typeof(int)), 0, "Stride dimension must be non-zero.");
                }
            }

            m_blobStride.mutable_cpu_data = rgStrideData;


            // Setup pad dimensions (blobPad)
            m_blobPad.Reshape(rgSpaitalDimBlobShape);
            T[] rgPadData = m_blobPad.mutable_cpu_data;

            if (p.pad_h.HasValue || p.pad_w.HasValue)
            {
                m_log.CHECK_EQ(m_nNumSpatialAxes, 2, "pad_h & pad_w can only be used in 2D convolution.");
                m_log.CHECK_EQ(0, p.pad.Count, "Either pad_size or pad_h/w should be specified; not both.");
                rgPadData[0] = (T)Convert.ChangeType(p.pad_h.Value, typeof(T));
                rgPadData[1] = (T)Convert.ChangeType(p.pad_w.Value, typeof(T));
            }
            else
            {
                int nNumPadDims = p.pad.Count;
                m_log.CHECK(nNumPadDims == 0 || nNumPadDims == 1 || nNumPadDims == m_nNumSpatialAxes, "Pad size must be specified once, or once per spatial dimension (pad specified " + nNumPadDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");
                int nDefaultPad = 0;

                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    if (nNumPadDims == 0)
                    {
                        rgPadData[i] = (T)Convert.ChangeType(nDefaultPad, typeof(T));
                    }
                    else
                    {
                        int nIdx = (nNumPadDims == 1) ? 0 : i;
                        rgPadData[i] = (T)Convert.ChangeType(p.pad[nIdx], typeof(T));
                    }
                }
            }

            m_blobPad.mutable_cpu_data = rgPadData;


            // Setup dilation dimensions (blobDilation)
            m_blobDilation.Reshape(rgSpaitalDimBlobShape);
            T[] rgDilationData = m_blobDilation.mutable_cpu_data;
            int nNumDilationDims = p.dilation.Count;

            m_log.CHECK(nNumDilationDims == 0 || nNumDilationDims == 1 || nNumDilationDims == m_nNumSpatialAxes, "Dilation size must be specified once, or once per spatial dimension (dilation specified " + nNumDilationDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");
            int nDefaultDilation = 1;

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                if (nNumDilationDims == 0)
                {
                    rgDilationData[i] = (T)Convert.ChangeType(nDefaultDilation, typeof(T));
                }
                else
                {
                    int nIdx = (nNumDilationDims == 1) ? 0 : i;
                    rgDilationData[i] = (T)Convert.ChangeType(p.dilation[nIdx], typeof(T));
                }
            }

            m_blobDilation.mutable_cpu_data = rgDilationData;


            // Special case: im2col is the identity for 1x1 convolution with stride 1
            // add no padding, so flag for skipping the buffer and transformation.
            m_bIs1x1 = true;

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                if (!(val_at(rgKernelShape, i) == 1 && 
                      val_at(rgStrideData, i) == 1 && 
                      val_at(rgPadData, i) == 0))
                {
                    m_bIs1x1 = false;
                    break;
                }
            }

            // Configure output channels and groups.
            m_nChannels = colBottom[0].shape(m_nChannelAxis);
            m_nNumOutput = (int)p.num_output;
            m_log.CHECK_GT(m_nNumOutput, 0, "Output count must be greater than zero.");

            m_nGroup = (int)p.group;
            m_log.CHECK_EQ(m_nChannels % m_nGroup, 0, "The channels must span evenly across the groups.");
            m_log.CHECK_EQ(m_nNumOutput % m_nGroup, 0, "The number of output should be a in multiples of group.");

            if (reverse_dimensions())
            {
                m_nConvOutChannels = m_nChannels;
                m_nConvInChannels = m_nNumOutput;
            }
            else
            {
                m_nConvOutChannels = m_nNumOutput;
                m_nConvInChannels = m_nChannels;
            }

            // Handle the parameters: weights and biases
            // - blobs[0] holds the filter weights.
            // - blobs[1] holds the biases (optional)

            List<int> rgWeightShape = new List<int>();
            rgWeightShape.Add(m_nConvOutChannels);
            rgWeightShape.Add(m_nConvInChannels / m_nGroup);

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                rgWeightShape.Add(val_at(rgKernelShape, i));
            }

            m_bBiasTerm = p.bias_term;

            List<int> rgBiasShape = new List<int>() { m_nNumOutput };

            if (m_colBlobs.Count > 0)
            {
                m_log.CHECK_EQ(1 + ((m_bBiasTerm) ? 1 : 0), m_colBlobs.Count, "Incorrect number of weight blobs.");

                if (!Utility.Compare<int>(rgWeightShape, m_colBlobs[0].shape()))
                {
                    Blob<T> b = new Blob<T>(m_cuda, m_log, rgWeightShape);
                    m_log.FAIL("Incorrect weight shape: expected shape " + b.shape_string + "; instead, shape was " + m_colBlobs[0].shape_string);                   
                }

                if (m_bBiasTerm && !Utility.Compare<int>(rgBiasShape, m_colBlobs[1].shape()))
                {
                    Blob<T> b = new Blob<T>(m_cuda, m_log, rgBiasShape);
                    m_log.FAIL("Incorrect bias shape: expected shape " + b.shape_string + "; instead, shape was " + m_colBlobs[1].shape_string);
                }

                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                m_colBlobs.Clear();

                // Initialize and fill the weights:
                // output channels x input channels per-group x kernel height x kernel width.
                Blob<T> blobWts = new Blob<T>(m_cuda, m_log);
                blobWts.Name = colTop[0].Name + " weights";

                if (!shareParameter(blobWts, rgWeightShape))
                {
                    blobWts.Reshape(rgWeightShape);
                    Filler<T> wtFiller = Filler<T>.Create(m_cuda, m_log, p.weight_filler);
                    wtFiller.Fill(blobWts);
                }

                m_colBlobs.Add(blobWts);

                // If necessary, initialize and fill the biases:
                if (m_bBiasTerm)
                {
                    Blob<T> blobBias = new Blob<T>(m_cuda, m_log);
                    blobBias.Name = colTop[0].Name + " bias";

                    if (!shareParameter(blobBias, rgBiasShape))
                    {
                        blobBias.Reshape(rgBiasShape);
                        Filler<T> biasFiller = Filler<T>.Create(m_cuda, m_log, p.bias_filler);
                        biasFiller.Fill(blobBias);
                    }

                    m_colBlobs.Add(blobBias);
                }
            }

            m_nKernelDim = m_colBlobs[0].count(1);
            m_nWeightOffset = m_nConvOutChannels * m_nKernelDim / m_nGroup;

            // Propagate gradients to the parameters (as directed by backward pass).
            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nFirstSpatialAxis = m_nChannelAxis + 1;
            m_log.CHECK_EQ(colBottom[0].num_axes, nFirstSpatialAxis + m_nNumSpatialAxes, "bottom num_axes may not change.");

            m_nNum = colBottom[0].count(0, m_nChannelAxis);
            m_log.CHECK_EQ(colBottom[0].shape(m_nChannelAxis), m_nChannels, "Input size incompatible with convolution kernel.");

            // TODO: generalize to handle inputs of different shapes.
            for (int i = 1; i < colBottom.Count; i++)
            {
                m_log.CHECK(Utility.Compare<int>(colBottom[0].shape(), colBottom[i].shape()), "Shape mismatch - bottom[0]: '" + colBottom[0].shape_string + "' vs. bottom[" + i.ToString() + "]: '" + colBottom[i].shape_string + "'");
            }

            // Shape the tops.
            m_rgBottomShape = Utility.Clone<int>(colBottom[0].shape());
            compute_output_shape();

            List<int> rgTopShape = new List<int>();

            for (int i = 0; i < m_nChannelAxis; i++)
            {
                rgTopShape.Add(colBottom[0].shape(i));
            }

            rgTopShape.Add(m_nNumOutput);

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                rgTopShape.Add(m_rgOutputShape[i]);
            }

            for (int i = 0; i < colTop.Count; i++)
            {
                colTop[i].Reshape(rgTopShape);
            }

            if (reverse_dimensions())
                m_nConvOutSpatialDim = colBottom[0].count(nFirstSpatialAxis);
            else
                m_nConvOutSpatialDim = colTop[0].count(nFirstSpatialAxis);

            m_nColOffset = m_nKernelDim * m_nConvOutSpatialDim;
            m_nOutputOffset = m_nConvOutChannels * m_nConvOutSpatialDim / m_nGroup;

            if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes) || reverse_dimensions())
            {
                // Setup input dimensions (blobConvInputShape)
                List<int> rgBottomDimBlobShape = new List<int>() { m_nNumSpatialAxes + 1 };
                m_blobConvInputShape.Reshape(rgBottomDimBlobShape);

                T[] rgConvInputShapeData = m_blobConvInputShape.mutable_cpu_data;
                for (int i = 0; i < m_nNumSpatialAxes + 1; i++)
                {
                    if (reverse_dimensions())
                        rgConvInputShapeData[i] = (T)Convert.ChangeType(colTop[0].shape(m_nChannelAxis + i), typeof(T));
                    else
                        rgConvInputShapeData[i] = (T)Convert.ChangeType(colBottom[0].shape(m_nChannelAxis + i), typeof(T));
                }
                m_blobConvInputShape.mutable_cpu_data = rgConvInputShapeData;

                // The im2col result buffer will only hold one image at a time to avoid
                // overly large memory usage.  In the special case of 1x1 convolution
                // it goes lazily unused to save memory.
                m_rgColBufferShape = new List<int>();
                m_rgColBufferShape.Add(m_nKernelDim * m_nGroup);

                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    if (reverse_dimensions())
                        m_rgColBufferShape.Add(input_shape(i + 1));
                    else
                        m_rgColBufferShape.Add(m_rgOutputShape[i]);
                }

                shareLayerBlob(m_blobColBuffer, m_rgColBufferShape);
                m_blobColBuffer.Reshape(m_rgColBufferShape);
            }

            m_nBottomDim = colBottom[0].count(m_nChannelAxis);
            m_nTopDim = colTop[0].count(m_nChannelAxis);
            m_nNumKernelsIm2col = m_nConvInChannels * m_nConvOutSpatialDim;
            m_nNumKernelsCol2im = (reverse_dimensions()) ? m_nTopDim : m_nBottomDim;

            // Setup up the all ones 'bias_multiplier' for adding biases by BLAS
            m_nOutSpatialDim = colTop[0].count(nFirstSpatialAxis);

            if (m_bBiasTerm)
            {
                if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes) || reverse_dimensions())
                {
                    List<int> rgBiasMultShape = new List<int>() { m_nOutSpatialDim };
                    shareLayerBlob(m_blobBiasMultiplier, rgBiasMultShape);
                    m_blobBiasMultiplier.Reshape(rgBiasMultShape);
                    m_blobBiasMultiplier.SetData(1.0);
                }
            }
        }

        /// <summary>
        /// Returns the minimum number of required bottom Blobs: input 
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: output
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns that there are an equal number of top and bottom Blobs.
        /// </summary>
        public override bool EqualNumBottomTopBlobs
        {
            get { return true; }
        }

        /// <summary>
        /// Helper function that abstract away the column buffer and gemm arguments.
        /// </summary>
        /// <remarks>
        /// This function is only used when performing the Caffe version of convolution.
        /// </remarks>
        /// <param name="hInput">Specifies a handle to the input data in GPU memory.</param>
        /// <param name="nInputOffset">Specifies an offset (in items) into the input data.</param>
        /// <param name="hWeights">Specifies a handle to the weight data in GPU memory.</param>
        /// <param name="hOutput">Specifies a handle to the output data in GPU memory.</param>
        /// <param name="nOutputOffset">Specifies an offset (in items) into the output data.</param>
        /// <param name="bSkipIm2Col">Specifies whether or not to skip the im2coll function call for it was already computed.</param>
        protected void forward_gemm(long hInput, int nInputOffset, long hWeights, long hOutput, int nOutputOffset, bool bSkipIm2Col = false)
        {
            long hColBuff = hInput;
            int nColBuffOffset = nInputOffset;

            if (!m_bIs1x1)
            {
                if (!bSkipIm2Col)
                    conv_im2col(hInput, nInputOffset, m_blobColBuffer.mutable_gpu_data, 0);

                hColBuff = m_blobColBuffer.gpu_data;
                nColBuffOffset = 0;
            }

            for (int g = 0; g < m_nGroup; g++)
            {
                m_cuda.gemm(false, false, m_nConvOutChannels / m_nGroup, m_nConvOutSpatialDim, m_nKernelDim, m_tOne, hWeights, hColBuff, m_tZero, hOutput, m_nWeightOffset * g, nColBuffOffset + m_nColOffset * g, nOutputOffset + m_nOutputOffset * g); 
            }
        }

        /// <summary>
        /// Helper function that abstracts away the column buffer and gemm arguments.
        /// </summary>
        /// <remarks>
        /// This function is only used when performing the Caffe version of convolution.
        /// </remarks>
        /// <param name="hOutput">Specifies a handle to the output data in GPU memory.</param>
        /// <param name="nOutputOffset">Specifies an offset (in items) into the output data.</param>
        /// <param name="hBias">Specifies a handle to the bias data in GPU memory.</param>
        protected void forward_bias(long hOutput, int nOutputOffset, long hBias)
        {
            m_cuda.gemm(false, false, m_nNumOutput, m_nOutSpatialDim, 1, m_tOne, hBias, m_blobBiasMultiplier.gpu_data, m_tOne, hOutput, 0, 0, nOutputOffset);
        }

        /// <summary>
        /// Helper function that abstract away the column buffer and gemm arguments.
        /// </summary>
        /// <remarks>
        /// This function is only used when performing the Caffe version of convolution.
        /// </remarks>
        /// <param name="hOutput">Specifies a handle to the output data in GPU memory.</param>
        /// <param name="nOutputOffset">Specifies an offset (in items) into the output data.</param>
        /// <param name="hWeights">Specifies a handle to the weight data in GPU memory.</param>
        /// <param name="hInput">Specifies a handle to the input data in GPU memory.</param>
        /// <param name="nInputOffset">Specifies an offset (in items) into the input data.</param>
        protected void backward_gemm(long hOutput, int nOutputOffset, long hWeights, long hInput, int nInputOffset)
        {
            long hColBuff = m_blobColBuffer.mutable_gpu_data;
            int nColBuffOffset = 0;

            if (m_bIs1x1)
            {
                hColBuff = hInput;
                nColBuffOffset = nInputOffset;
            }

            for (int g = 0; g < m_nGroup; g++)
            {
                m_cuda.gemm(true, false, m_nKernelDim, m_nConvOutSpatialDim, m_nConvOutChannels / m_nGroup, m_tOne, hWeights, hOutput, m_tZero, hColBuff, m_nWeightOffset * g, nOutputOffset + m_nOutputOffset * g, nColBuffOffset + m_nColOffset * g);
            }

            if (!m_bIs1x1)
                conv_col2im(hColBuff, nColBuffOffset, hInput, nInputOffset);
        }

        /// <summary>
        /// Helper function that abstract away the column buffer and gemm arguments.
        /// </summary>
        /// <remarks>
        /// This function is only used when performing the Caffe version of convolution.
        /// </remarks>
        /// <param name="hInput">Specifies a handle to the input data in GPU memory.</param>
        /// <param name="nInputOffset">Specifies an offset (in items) into the input data.</param>
        /// <param name="hOutput">Specifies a handle to the output data in GPU memory.</param>
        /// <param name="nOutputOffset">Specifies an offset (in items) into the output data.</param>
        /// <param name="hWeights">Specifies a handle to the weight data in GPU memory.</param>
        protected void weight_gemm(long hInput, int nInputOffset, long hOutput, int nOutputOffset, long hWeights)
        {
            long hColBuff = hInput;
            int nColBuffOffset = nInputOffset;

            if (!m_bIs1x1)
            {
                conv_im2col(hInput, nInputOffset, m_blobColBuffer.mutable_gpu_data, 0);
                hColBuff = m_blobColBuffer.gpu_data;
                nColBuffOffset = 0;
            }

            for (int g = 0; g < m_nGroup; g++)
            {
                m_cuda.gemm(false, true, m_nConvOutChannels / m_nGroup, m_nKernelDim, m_nConvOutSpatialDim, m_tOne, hOutput, hColBuff, m_tOne, hWeights, nOutputOffset + m_nOutputOffset * g, nColBuffOffset + m_nColOffset * g, m_nWeightOffset * g);
            }
        }

        /// <summary>
        /// Helper function that abstracts away the column buffer and gemm arguments.
        /// </summary>
        /// <remarks>
        /// This function is only used when performing the Caffe version of convolution.
        /// </remarks>
        /// <param name="hBias">Specifies a handle to the bias data in GPU memory.</param>
        /// <param name="hInput">Specifies a handle to the input data in GPU memory.</param>
        /// <param name="nInputOffset">Specifies an offset (in items) into the input data.</param>
        protected void backward_bias(long hBias, long hInput, int nInputOffset)
        {
            m_cuda.gemv(false, m_nNumOutput, m_nOutSpatialDim, m_tOne, hInput, m_blobBiasMultiplier.gpu_data, m_tOne, hBias, nInputOffset, 0, 0);
        }

        /// <summary>
        /// Returns the spatial dimensions of the input.
        /// </summary>
        /// <param name="i">Specifies the index to add to the channel index.</param>
        /// <returns>The spatial dimension at the index is returned.</returns>
        protected int input_shape(int i)
        {
            return m_rgBottomShape[m_nChannelAxis + i];
        }

        /// <summary>
        /// reverse_dimensions should return true iff we are implementing deconv, so
        /// that conv helpers know which dimensions to use.
        /// </summary>
        /// <returns>If the dimensions are to be reversed (e.g. when deconvolving), <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        protected abstract bool reverse_dimensions();

        /// <summary>
        /// Compute height_out and width_out from other parameters.
        /// </summary>
        protected abstract void compute_output_shape();

        private void conv_im2col(long hData, int nDataOffset, long hColBuff, int nColBuffOffset)
        {
            if (!m_bForceNDim2col && m_nNumSpatialAxes == 2)
            {
                T[] rgConvInputShape = m_blobConvInputShape.update_cpu_data();
                T[] rgKernelShape = m_blobKernelShape.update_cpu_data();
                T[] rgPad = m_blobPad.update_cpu_data();
                T[] rgStride = m_blobStride.update_cpu_data();
                T[] rgDilation = m_blobDilation.update_cpu_data();

                m_cuda.im2col(hData, 
                              nDataOffset, 
                              m_nConvInChannels,
                              val_at(rgConvInputShape, 1),
                              val_at(rgConvInputShape, 2),
                              val_at(rgKernelShape, 0),
                              val_at(rgKernelShape, 1),
                              val_at(rgPad, 0),
                              val_at(rgPad, 1),
                              val_at(rgStride, 0),
                              val_at(rgStride, 1),
                              val_at(rgDilation, 0),
                              val_at(rgDilation, 1),
                              hColBuff, 
                              nColBuffOffset);
            }
            else
            {
                m_cuda.im2col_nd(hData, 
                              nDataOffset, 
                              m_nNumSpatialAxes, 
                              m_nNumKernelsIm2col,
                              0,
                              m_blobConvInputShape.gpu_data, 
                              m_blobColBuffer.gpu_shape, 
                              m_blobKernelShape.gpu_data, 
                              m_blobPad.gpu_data, 
                              m_blobStride.gpu_data, 
                              m_blobDilation.gpu_data,
                              hColBuff, 
                              nColBuffOffset);
            }
        }

        private void conv_col2im(long hColBuff, int nColBuffOffset, long hData, int nDataOffset)
        {
            if (!m_bForceNDim2col && m_nNumSpatialAxes == 2)
            {
                T[] rgConvInputShape = m_blobConvInputShape.update_cpu_data();
                T[] rgKernelShape = m_blobKernelShape.update_cpu_data();
                T[] rgPad = m_blobPad.update_cpu_data();
                T[] rgStride = m_blobStride.update_cpu_data();
                T[] rgDilation = m_blobDilation.update_cpu_data();

                m_cuda.col2im(hColBuff,
                              nColBuffOffset, 
                              m_nConvInChannels, 
                              val_at(rgConvInputShape, 1),
                              val_at(rgConvInputShape, 2),
                              val_at(rgKernelShape, 0),
                              val_at(rgKernelShape, 1),
                              val_at(rgPad, 0),
                              val_at(rgPad, 1),
                              val_at(rgStride, 0),
                              val_at(rgStride, 1),
                              val_at(rgDilation, 0),
                              val_at(rgDilation, 1),
                              hData, 
                              nDataOffset);
            }
            else
            {
                m_cuda.im2col_nd(hColBuff, 
                              nColBuffOffset, 
                              m_nNumSpatialAxes, 
                              m_nNumKernelsCol2im,
                              0,
                              m_blobConvInputShape.gpu_data, 
                              m_blobColBuffer.gpu_shape, 
                              m_blobKernelShape.gpu_data, 
                              m_blobPad.gpu_data, 
                              m_blobStride.gpu_data, 
                              m_blobDilation.gpu_data,
                              hData, 
                              nDataOffset);
            }
        }
    }
}
