using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Drawing;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;
using System.IO;
using System.Reflection;

/// <summary>
/// The MyCaffe.layers namespace contains all layers that have a solidified code base, including the Layer class.
/// </summary>
namespace MyCaffe.layers
{
    /// <summary>
    /// An interface for the units of computation which can be composed into a Net.
    /// </summary>
    /// <remarks>
    /// Layer%s must implement an override to the forward function, in which they take their input (bottom) Blob%s
    /// (if any) and compute their output Blob%s (if any).  They may also implement aan override to the backward function,
    /// in which they compute the error gradients with respect to their input Blob's, given the error 
    /// gradients with their output Blob%s.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public abstract class Layer<T> : IDisposable
    {
        /// <summary>
        /// Specifies the Layer type.
        /// </summary>
        protected LayerParameter.LayerType m_type = LayerParameter.LayerType._MAX;
        /// <summary>
        /// Specifies the CudaDnn connection to Cuda.
        /// </summary>
        protected CudaDnn<T> m_cuda;
        /// <summary>
        /// Specifies the Log for output.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// Specifies the LayerParameter describing the Layer.
        /// </summary>
        protected LayerParameter m_param;
        /// <summary>
        /// Specifies the Phase under which the Layer is run.
        /// </summary>
        protected Phase m_phase;
        /// <summary>
        /// Specifies the learnable parameter Blobs of the Layer.
        /// </summary>
        protected BlobCollection<T> m_colBlobs;
        /// <summary>
        /// Specifies whether or not to compute the learnable diff of each parameter Blob.
        /// </summary>
        protected DictionaryMap<bool> m_rgbParamPropagateDown;
        /// <summary>
        /// Specifies the loss values that indeicate whether each top (output) Blob has a non-zero
        /// weight in the objective function..
        /// </summary>
        protected DictionaryMap<double> m_rgLoss;
        /// <summary>
        /// Specifies a generic type equal to 1.0.
        /// </summary>
        protected T m_tOne;
        /// <summary>
        /// Specifies a generic type equal to 0.0.
        /// </summary>
        protected T m_tZero;
        /// <summary>
        /// Enables/disables the pass-through mode for the layer.  Default = <i>false</i>.
        /// </summary>
        protected bool m_bEnablePassthrough = false;
        /// <summary>
        /// Specifies that the half size of the top (if any) should be converted to the base size.
        /// </summary>
        protected bool m_bUseHalfSize = false;
        /// <summary>
        /// Specifies whether or not the layer should convert the top on the forward pass when using half sized memory (typically only done with input data).
        /// </summary>
        protected bool m_bConvertTopOnFwd = false;
        /// <summary>
        /// Specifies whether or not to convert the top on the backward pass when using half sized memory (typically not done on loss layers).
        /// </summary>
        protected bool m_bConvertTopOnBwd = true;
        /// <summary>
        /// Specifies whether or not the layer should convert the bottom when using half sized memory.
        /// </summary>
        protected bool m_bConvertBottom = true;
        /// <summary>
        /// Specifies whether or not the reshape on forward is needed or not.
        /// </summary>
        protected bool m_bReshapeOnForwardNeeded = true;

        private List<List<int>> m_rgrgLastBottomShape = new List<List<int>>();
        private List<List<int>> m_rgrgLastTopShape = new List<List<int>>();

        private double m_dfForwardTiming = 0;
        private double m_dfForwardAverageTiming = 0;
        private double m_dfBackwardTiming = 0;
        private double m_dfBackwardAverageTiming = 0;
        private double m_dfAverageInterval = 20.0;
        private Stopwatch m_swTiming = new Stopwatch();
        private WorkspaceArgs m_argsWs = new WorkspaceArgs(0, 0);

        /// <summary>
        /// Specifies the OnGetWorkspace event that fires when the getWorkspace() function is called by a layer to get a shareable workspace to conserve GPU memory.
        /// </summary>
        public event EventHandler<WorkspaceArgs> OnGetWorkspace;
        /// <summary>
        /// Specifies the OnSetWorkspace event that fires when the setWorkspace() function is called by a layer to get a shareable workspace to conserve GPU memory.
        /// </summary>
        public event EventHandler<WorkspaceArgs> OnSetWorkspace;
        /// <summary>
        /// Specifies the OnGetIteration event that fires when a layer needs to get the current iteration from the solver.
        /// </summary>
        public event EventHandler<GetIterationArgs> OnGetIteration;
        /// <summary>
        /// Specifies the OnGetWorkBlob event that is only supported when debugging to get a work
        /// blob from the primary Net holding this layer.
        /// </summary>
        /// <remarks>
        /// When implemented, this event causes a nan/inf check at the end of each forward and backward pass
        /// and is only recommended use during debugging.</remarks>
        public event EventHandler<GetWorkBlobArgs<T>> OnDebug;

        enum CONVERT_TYPE
        {
            BASE_TO_HALF,
            HALF_TO_BASE
        }

        /// <summary>
        /// The Layer constructor.
        /// </summary>
        /// <remarks>
        /// Setup code for derivative classes should go into an override of the LayerSetup function where the 
        /// dimensionsn of the Blob%s are provided to the Layer.
        /// </remarks>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter that contains the settings of the Layer.</param>
        public Layer(CudaDnn<T> cuda, Log log, LayerParameter p)
        {
            m_cuda = cuda;
            m_log = log;
            m_param = p.Clone(true);
            m_phase = p.phase;
            m_rgbParamPropagateDown = new DictionaryMap<bool>(false);
            m_rgLoss = new DictionaryMap<double>(0.0);
            m_colBlobs = new BlobCollection<T>();

            for (int i = 0; i < p.blobs.Count; i++)
            {
                m_colBlobs.Add(new Blob<T>(cuda, log, p.blobs[i]));
            }

            m_tOne = (T)Convert.ChangeType(1, typeof(T));
            m_tZero = (T)Convert.ChangeType(0, typeof(T));
        }

        /// <summary>
        /// Releases all GPU and host resources used by the Layer.
        /// </summary>
        public void Dispose()
        {
            dispose();
        }

        /// <summary>
        /// Releases all GPU and host resources used by the Layer.
        /// </summary>
        protected virtual void dispose()
        {
        }

        /// <summary>
        /// Re-initialize the parameters of the layer.
        /// </summary>
        /// <param name="target">Specifies the weights to target (e.g. weights, bias or both).</param>
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public virtual bool ReInitializeParameters(WEIGHT_TARGET target)
        {
            return true;
        }

        /// <summary>
        /// Fires the OnGetIteration event to query the current iteration.
        /// </summary>
        /// <returns>The GetIterationArgs is returned if the event is connected, otherwise <i>null</i> is returned.</returns>
        protected GetIterationArgs getCurrentIteration()
        {
            GetIterationArgs args = null;

            if (OnGetIteration != null)
            {
                args = new GetIterationArgs();
                OnGetIteration(this, args);               
            }

            return args;
        }

        /// <summary>
        /// Changes the layer's Phase to the one specified.
        /// </summary>
        /// <param name="phase">Specifies the new Phase for the layer.</param>
        public void SetPhase(Phase phase)
        {
            m_phase = phase;
            m_param.phase = phase;
        }

        /// <summary>
        /// Implements common Layer setup functionality.
        /// </summary>
        /// <remarks>
        /// Checks that the number of bottom and top blobs are correct.  Calls LayerSetup to do Layer specific
        /// setup for each layer type, followed by Reshape to setup the sizes of the top Blobs and internal
        /// buffers.  Shes up the loss weight multiplier blobs for any non-zero loss weights.
        /// </remarks>
        /// <param name="colBottom">Specifies the collection of preshaped bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of allocated but unshaped top (output) Blobs.</param>
        public void Setup(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            try
            {
                m_rgrgLastBottomShape = new List<List<int>>();
                m_rgrgLastTopShape = new List<List<int>>();

                CheckBlobCounts(colBottom, colTop);
                LayerSetUp(colBottom, colTop);
                Reshape(colBottom, colTop);
                setShapes(colBottom, colTop);
                SetLossWeights(colTop);
            }
            catch (Exception excpt)
            {
                if (m_param != null)
                    throw new Exception("Layer: '" + m_param.name + "' (" + m_param.type.ToString() + ") Error: " + excpt.Message, excpt);
                else
                    throw excpt;
            }
        }

        /// <summary>
        /// Performs Layer specific setup.  Derived layers should override this function as well
        /// as the Reshape function.
        /// </summary>
        /// <remarks>
        /// This method should perform one-time Layer specific setup.  This may include reading
        /// and processing relevant parameters from teh <code>layer_param</code>.
        /// Setting up the shapes of top (output) blobs and internal buffers should be done in the
        /// <code>Reshape</code> function, which will be called before the Forward pass to 
        /// adjust the top (input) Blob sizes.
        /// </remarks>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs to this Layer.</param>
        /// <param name="colTop">Specifies the collection of allocated but unshaped top (output) Blobs.</param>
        public abstract void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop);

        /// <summary>
        /// This function allows other layers to gather needed information from the NetParameters if any, and is called when initialzing the Net.
        /// </summary>
        /// <param name="np">Specifies the NetParameter.</param>
        public virtual void SetNetParameterUsed(NetParameter np)
        {
        }

        /// <summary>
        /// Adjust the shapes of top blobs and internal buffers to accomodate the shapes
        /// of the bottom blobs.
        /// </summary>
        /// <remarks>
        /// This method should reshape top blobs as needed according to the shapes of the bottom (input) Blob%s,
        /// as well as reshaping any internal buffers and making any other necessary adjustments so that the layer
        /// can accomodate the bottom (input) Blobs.</remarks>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, with requested input shapes.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs, which should be reshaped as needed by the Layer.</param>
        public abstract void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop);

        /// <summary>
        /// Convert half memory to full memory.
        /// </summary>
        /// <param name="nCount">Specifies the number of items.</param>
        /// <param name="hMem">Specifies the memory to convert.</param>
        /// <returns>A handle to the converted memory is returned.</returns>
        protected long convert_to_full(int nCount, long hMem)
        {
            if (OnGetWorkspace == null || OnSetWorkspace == null)
                throw new Exception("The OnGetWorkSpace and OnSetWorkspace events must be connected!");

            ulong lSize = (ulong)nCount * CudaDnn<T>.basetype_size(false);
            WorkspaceArgs args = getWorkspace();
            if (args.Size < lSize)
            {
                setWorkspace(lSize);
                args = getWorkspace();
            }

            m_cuda.copy(nCount, hMem, args.Data, 0, 0, -1, null, false);
            return args.Data;
        }

        /// <summary>
        /// Convert a collection of blobs from / to half size.
        /// </summary>
        /// <param name="col">Specifies the collection to convert.</param>
        protected void convert(BlobCollection<T> col)
        {
            ulong lMaxSize = 0;
            bool bConversionNeeded = false;

            foreach (Blob<T> b in col)
            {
                if ((m_bUseHalfSize && !b.HalfSize) ||
                    (!m_bUseHalfSize && b.HalfSize))
                {
                    ulong lSize = b.GetConversionWorkSize(m_bUseHalfSize);
                    if (lMaxSize < lSize)
                        lMaxSize = lSize;

                    bConversionNeeded = true;
                }
            }

            if (!bConversionNeeded)
                return;

            if (OnGetWorkspace == null || OnSetWorkspace == null)
                throw new Exception("The OnGetWorkSpace and OnSetWorkspace events must be connected!");

            WorkspaceArgs args = getWorkspace();
            if (args.Size < lMaxSize)
            {
                setWorkspace(lMaxSize);
                args = getWorkspace();
            }

            foreach (Blob<T> b in col)
            {
                if (m_bUseHalfSize && !b.HalfSize)
                    b.ConvertToHalf(args.Data, args.Size, true, true);
                else if (!m_bUseHalfSize && b.HalfSize)
                    b.ConvertToBase(args.Data, args.Size, true, true);
            }
        }

        /// <summary>
        /// ConvertToBase converts any blobs in a collection that are in half size to the base size.
        /// </summary>
        /// <param name="col">Specifies the blob collection to convert.</param>
        public void ConvertToBase(BlobCollection<T> col)
        {
            if (OnGetWorkspace == null || OnSetWorkspace == null)
                return;

            ulong lMaxSize = 0;
            bool bConversionNeeded = false;

            foreach (Blob<T> b in col)
            {
                if (b.HalfSize)
                {
                    ulong lSize = b.GetConversionWorkSize(false);
                    if (lMaxSize < lSize)
                        lMaxSize = lSize;

                    bConversionNeeded = true;
                }
            }

            if (!bConversionNeeded)
                return;

            WorkspaceArgs args = getWorkspace();
            if (args.Size < lMaxSize)
            {
                setWorkspace(lMaxSize);
                args = getWorkspace();
            }

            foreach (Blob<T> b in col)
            {
                b.ConvertToBase(args.Data, args.Size, true, true);
            }
        }

        /// <summary>
        /// Tests the shapes of both the bottom and top blobs and if they are the same as the previous sizing, returns <i>false</i> indicating that no reshape is needed.
        /// </summary>
        /// <param name="colBottom">Specifies the bottom blobs.</param>
        /// <param name="colTop">Specifies the top blobs.</param>
        /// <param name="bReset">Specifies to reset the test (set to <i>false</i> when using in second derivative classes, e.g. set to true in BaseConvolutionLayer, and false in ConvolutionLayer).</param>
        /// <returns>If a reshape is needed, returns <i>true</i> otherwise returns <i>fasle</i>.</returns>
        protected bool reshapeNeeded(BlobCollection<T> colBottom, BlobCollection<T> colTop, bool bReset = true)
        {
            if (!bReset)
                return m_bReshapeOnForwardNeeded;

            if (!compareShapes(colBottom, colTop))
            {
                m_bReshapeOnForwardNeeded = true;
                return true;
            }
            else
            {
                m_bReshapeOnForwardNeeded = false;
                return false;
            }
        }

        /// <summary>
        /// Compare the shapes of the top and bottom and if the same, return true, otherwise false.
        /// </summary>
        /// <param name="colBottom">Specifies the bottom blobs.</param>
        /// <param name="colTop">Specifies the top blobs.</param>
        /// <returns>If the top and bottom blobs have not changed shape, true is returned, otherwise false.</returns>
        protected bool compareShapes(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!compareShapes(colBottom, m_rgrgLastBottomShape))
                return false;

            if (!compareShapes(colTop, m_rgrgLastTopShape))
                return false;

            return true;
        }

        private bool compareShapes(BlobCollection<T> col, List<List<int>> rgrg)
        {
            if (rgrg.Count != col.Count)
                return false;

            for (int i = 0; i < col.Count; i++)
            {
                int nCount = col[i].shape().Count;
                if (rgrg[i].Count != nCount)
                    return false;

                for (int j = 0; j < nCount; j++)
                {
                    if (col[i].shape()[j] != rgrg[i][j])
                        return false;
                }
            }

            return true;
        }

        private void setShapes(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            setShapes(colBottom, ref m_rgrgLastBottomShape);
            setShapes(colTop, ref m_rgrgLastTopShape);
        }

        private void setShapes(BlobCollection<T> col, ref List<List<int>> rgrg)
        {
            if (rgrg.Count != col.Count)
                rgrg = new List<List<int>>(col.Count);

            for (int i = 0; i < col.Count; i++)
            {
                int nCount = col[i].shape().Count;
                if (rgrg.Count < col.Count)
                    rgrg.Add(new List<int>());
                else if (rgrg[i].Count != nCount)
                    rgrg[i] = new List<int>(nCount);

                for (int j = 0; j < nCount; j++)
                {
                    nCount = col[i].shape().Count;
                    if (rgrg[i].Count < nCount)
                        rgrg[i].Add(col[i].shape()[j]);
                    else
                        rgrg[i][j] = col[i].shape()[j];
                }
            }
        }

        /// <summary>
        /// Given the bottom (input) Blobs, this function computes the top (output) Blobs and the loss.
        /// </summary>
        /// <remarks>
        /// The Forward function calls the overriden forward function implemented by each specific Layer derivative 
        /// to compute the top (output) Blob's values given the bottom (input) Blobs.  If the layer has any non-zero
        /// <code>loss_weights</code> this function then computes and returns the loss.
        /// </remarks>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, whos data fields
        /// store the input data for this layers' outputs.</param>
        /// <param name="colTop">Specifies the collection of preshaped top (output) Blobs, whos data fields
        /// will store this layers' outputs.</param>
        /// <returns>Returns the total loss from the Layer.</returns>
        public double Forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            try
            {
                m_swTiming.Restart();
                double dfLoss = 0;

                if (m_bConvertBottom)
                    convert(colBottom);

                Reshape(colBottom, colTop);

                if (m_bConvertTopOnFwd)
                    convert(colTop);

                forward(colBottom, colTop);

                for (int i = 0; i < colTop.Count; i++)
                {
                    if (loss(i) == 0)
                        continue;

                    int nCount = colTop[i].count();
                    long hData = colTop[i].gpu_data;
                    long hDiff = colTop[i].gpu_diff;
                    double dfBlobLoss = m_cuda.dot_double(nCount, hData, hDiff);

                    dfLoss += dfBlobLoss;
                }

                m_swTiming.Stop();
                m_dfForwardTiming = m_swTiming.Elapsed.TotalMilliseconds;
                m_dfForwardAverageTiming = getAveTiming(m_dfAverageInterval, m_dfForwardTiming, m_dfForwardAverageTiming);

                if (OnDebug != null)
                {
                    GetWorkBlobArgs<T> args = new GetWorkBlobArgs<T>();
                    OnDebug(this, args);

                    foreach (Blob<T> b in colTop)
                    {
                        Tuple<double, double, double, double> mm_data = b.minmax_data(args.Blob, true);
                        Tuple<double, double, double, double> mm_diff = b.minmax_diff(args.Blob, true);

                        if (mm_data.Item3 > 0 || mm_data.Item4 > 0)
                            throw new Exception("NAN or INF detected in the TOP '" + b.Name + "' Data for layer '" + m_param.name + "' on the forward pass.");

                        if (mm_diff.Item3 > 0 || mm_diff.Item4 > 0)
                            throw new Exception("NAN or INF detected in TOP '" + b.Name + "' Diff for layer '" + m_param.name + "' on the forward pass.");
                    }
                }

                return dfLoss;
            }
            catch (Exception excpt)
            {
                if (m_param != null)
                    throw new Exception("Layer: '" + m_param.name + "' (" + m_param.type.ToString() + ") Error: " + excpt.Message, excpt);
                else
                    throw excpt;
            }
        }

        /// <summary>
        /// This forward abstract function must be overriden by each derived Layer class to compute the 
        /// top (output) Blobs for this layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, whos data fields
        /// store the input data for this layers' outputs.</param>
        /// <param name="colTop">Specifies the collection of preshaped top (output) Blobs, whos data fields
        /// will store this layers' outputs.</param>
        protected abstract void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop);

        /// <summary>
        /// Given the top Blob error gradients, compute the bottom Blob error gradients.
        /// </summary>
        /// <remarks>
        /// The Backward function calls the overriden backward function implemented by each specific Layer derivative,
        /// to compute the bottom (input) Blob diffs given the top (output) Blob diffs.
        /// </remarks>
        /// <param name="colTop">Specifies a collection of top (output) Blobs, whos diff fields store the gradient of the
        /// error with respect to themselves.</param>
        /// <param name="rgbPropagateDown">Specifies a List with equal length to the bottom, with each element
        /// indicating whether or not to propagate the error gradients down to the bottom Blob at the corresponding
        /// index.</param>
        /// <param name="colBottom">Specifies a collection of bottom (input) Blobs, whos diff fields are filled with
        /// the gradient of the error with respect to themselves after the Backward function is run.</param>
        public void Backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            try
            {
                m_swTiming.Restart();

                if (m_bConvertTopOnBwd)
                    convert(colTop);

                convert(colBottom);

                backward(colTop, rgbPropagateDown, colBottom);
                m_swTiming.Stop();
                m_dfBackwardTiming = m_swTiming.Elapsed.TotalMilliseconds;
                m_dfBackwardAverageTiming = getAveTiming(m_dfAverageInterval, m_dfBackwardTiming, m_dfBackwardAverageTiming);

                if (OnDebug != null)
                {
                    GetWorkBlobArgs<T> args = new GetWorkBlobArgs<T>();
                    OnDebug(this, args);

                    foreach (Blob<T> b in colBottom)
                    {
                        Tuple<double, double, double, double> mm_data = b.minmax_data(args.Blob, true);
                        Tuple<double, double, double, double> mm_diff = b.minmax_diff(args.Blob, true);

                        if (mm_data.Item3 > 0 || mm_data.Item4 > 0)
                            throw new Exception("NAN or INF detected in the BOTTOM '" + b.Name + "' Data for layer '" + m_param.name + "' on the backward pass.");

                        if (mm_diff.Item3 > 0 || mm_diff.Item4 > 0)
                            throw new Exception("NAN or INF detected in the BOTTOM '" + b.Name + "' Diff for layer '" + m_param.name + "' on the backward pass.");
                    }
                }
            }
            catch (Exception excpt)
            {
                if (m_param != null)
                    throw new Exception("Layer: '" + m_param.name + "' (" + m_param.type.ToString() + ") Error: " + excpt.Message, excpt);
                else
                    throw excpt;
            }
        }

        /// <summary>
        /// This backward abstract function must be overriden by each derived Layer class to compute the 
        /// bottom (intput) Blob diffs for this Layer.
        /// </summary>
        /// <param name="colTop">Specifies a collection of top (output) Blobs, whos diff fields store the gradient of the
        /// error with respect to themselves.</param>
        /// <param name="rgbPropagateDown">Specifies a List with equal length to the bottom, with each element
        /// indicating whether or not to propagate the error gradients down to the bottom Blob at the corresponding
        /// index.</param>
        /// <param name="colBottom">Specifies a collection of bottom (input) Blobs, whos diff fields are filled with
        /// the gradient of the error with respect to themselves after the Backward function is run.</param>
        protected abstract void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom);

        /// <summary>
        /// Returns the collection of learnable parameter Blobs for the Layer.
        /// </summary>
        public BlobCollection<T> blobs
        {
            get { return m_colBlobs; }
        }

        /// <summary>
        /// Returns the collection of internal Blobs used by the Layer.
        /// </summary>
        public virtual BlobCollection<T> internal_blobs
        {
            get { return new BlobCollection<T>(); }
        }

        /// <summary>
        /// Returns the LayerParameter for this Layer.
        /// </summary>
        public LayerParameter layer_param
        {
            get { return m_param; }
        }

        /// <summary>
        /// Returns the scalar loss associated with the top Blob at a given index.
        /// </summary>
        /// <param name="nTopIdx">Specifies the index.</param>
        /// <returns>The loss value is returned.</returns>
        public double loss(int nTopIdx)
        {
            return m_rgLoss[nTopIdx];
        }

        /// <summary>
        /// Sets the loss associated with a top Blob at a given index.
        /// </summary>
        /// <param name="nTopIdx">Specifies the index.</param>
        /// <param name="dfLoss">Specifies the loss value.</param>
        public void set_loss(int nTopIdx, double dfLoss)
        {
            m_rgLoss[nTopIdx] = dfLoss;
        }

        /// <summary>
        /// Returns the LayerType of this Layer.
        /// </summary>
        public LayerParameter.LayerType type
        {
            get { return m_type; }
        }

        /// <summary>
        /// Returns the exact number of bottom (input) Blobs required by the Layer, 
        /// or -1 if no exact number is required.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return a non-negative value if your Layer
        /// expects an exact number of bottom (input) Blobs.
        /// </remarks>
        public virtual int ExactNumBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of bottom (input) Blobs required by the Layer,
        /// or -1 if no minimum number is required.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return a non-negative value if your Layer
        /// expects a minimum number of bottom (input) Blobs.
        /// </remarks>
        public virtual int MinBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the maximum number of bottom (input) Blobs required by the Layer,
        /// or -1 if no maximum number is required.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return a non-negative value if your Layer
        /// expects a maximum number of bottom (input) Blobs.
        /// </remarks>
        public virtual int MaxBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the exact number of top (output) Blobs required by the Layer, 
        /// or -1 if no exact number is required.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return a non-negative value if your Layer
        /// expects an exact number of top (output) Blobs.
        /// </remarks>
        public virtual int ExactNumTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of top (output) Blobs required by the Layer,
        /// or -1 if no minimum number is required.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return a non-negative value if your Layer
        /// expects a minimum number of top (output) Blobs.
        /// </remarks>
        public virtual int MinTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the maximum number of top (output) Blobs required by the Layer,
        /// or -1 if no maximum number is required.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return a non-negative value if your Layer
        /// expects a maximum number of top (output) Blobs.
        /// </remarks>
        public virtual int MaxTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns <i>true</i> if the Layer requires and equal number of bottom (input) and
        /// top (output) Blobs.
        /// </summary>
        /// <remarks>
        /// This method should be overriden to return <i>ture</i> if your Layer expects an
        /// equal number of bottom and top Blobs.
        /// </remarks>
        public virtual bool EqualNumBottomTopBlobs
        {
            get { return false; }
        }

        /// <summary>
        /// Return whether "anonymous" top (output) Blobs are created automatically by the
        /// Layer.
        /// </summary>
        /// <remarks>
        /// If this method returns <i>true</i>, Net::Init will create enough "anonymous" top
        /// Blobs to fulfill the requirement specified by ExactNumTopBlobs() or MinTopBlobs().
        /// </remarks>
        public virtual bool AutoTopBlobs
        {
            get { return false; }
        }

        /// <summary>
        /// Return whether to allow <code>force_backward</code> for a given bottom (input) Blob
        /// index.
        /// </summary>
        /// <remarks>
        /// If AllowForceBackward(i) == <i>false</i>, the <code>force_backward</code> setting will
        /// be ignored and backpropagate to Blob i only if it needs gradient information. (as is done
        /// when <code>force_backward == false</code>
        /// </remarks>
        /// <param name="nBottomIdx">Specifies the index of the bottom (input) item to force.</param>
        /// <returns></returns>
        public virtual bool AllowForceBackward(int nBottomIdx)
        {
            return true;
        }

        /// <summary>
        /// Returns whether or not the Layer should compute gradients w.r.t. a
        /// parameter at a particular index given by a parameter index.
        /// </summary>
        /// <param name="nParamIdx">Specifies the parameter index.</param>
        /// <returns></returns>
        public bool param_propagate_down(int nParamIdx)
        {
            return m_rgbParamPropagateDown[nParamIdx];
        }

        /// <summary>
        /// Sets whether or not the Layer should compute gradients w.r.t. a
        /// parameter at a particular index given by a parameter index.
        /// </summary>
        /// <param name="nParamIdx">Specifies the index.</param>
        /// <param name="bPropagate">Specifies whether or not to progagate down the parameter.</param>
        public void set_param_propagate_down(int nParamIdx, bool bPropagate)
        {
            m_rgbParamPropagateDown[nParamIdx] = bPropagate;
        }

        /// <summary>
        /// Called by the Layer::Setup function to check the number of bottom (input) and top (output) Blobs
        /// provided match  the expected number of blobs expected via the {EactNum,Min,Max}{Bottom,Top}Blobs
        /// functions.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected void CheckBlobCounts(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (ExactNumBottomBlobs >= 0)
                m_log.CHECK_EQ(ExactNumBottomBlobs, colBottom.Count, type.ToString() + " Layer takes " + ExactNumBottomBlobs.ToString() + " bottom blob(s) as input.");

            if (MinBottomBlobs >= 0)
                m_log.CHECK_LE(MinBottomBlobs, colBottom.Count, type.ToString() + " Layer takes at least " + MinBottomBlobs.ToString() + " bottom blob(s) as input.");

            if (MaxBottomBlobs >= 0)
                m_log.CHECK_GE(MaxBottomBlobs, colBottom.Count, type.ToString() + " Layer takes at most " + MaxBottomBlobs.ToString() + " bottom blob(s) as input.");

            if (ExactNumTopBlobs >= 0)
                m_log.CHECK_EQ(ExactNumTopBlobs, colTop.Count, type.ToString() + " Layer takes " + ExactNumTopBlobs.ToString() + " top blob(s) as output.");

            if (MinTopBlobs >= 0)
                m_log.CHECK_LE(MinTopBlobs, colTop.Count, type.ToString() + " Layer takes at least " + MinTopBlobs.ToString() + " top blob(s) as input.");

            if (MaxTopBlobs >= 0)
                m_log.CHECK_GE(MaxTopBlobs, colTop.Count, type.ToString() + " Layer takes at most " + MaxTopBlobs.ToString() + " bottom blob(s) as input.");

            if (EqualNumBottomTopBlobs)
                m_log.CHECK_EQ(colBottom.Count, colTop.Count, type.ToString() + " Layer produces one top blob as output for each bottom blob input.");
        }

        /// <summary>
        /// Called by Layer::Setup to initialize the weights associated with any top (output) Blobs
        /// in the loss function ans store non-zero loss weights in the diff Blob.
        /// </summary>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected void SetLossWeights(BlobCollection<T> colTop)
        {
            if (m_param.loss_weight.Count > 0)
            {
                m_log.CHECK_EQ(colTop.Count, m_param.loss_weight.Count, "loss_weight must be unspecified or specified once per top blob.");

                for (int i = 0; i < colTop.Count; i++)
                {
                    double dfLossWeight = m_param.loss_weight[i];

                    if (dfLossWeight == 0)
                        continue;

                    set_loss(i, dfLossWeight);
                    colTop[i].SetDiff(dfLossWeight);
                }
            }
        }

        /// <summary>
        /// Attempts to share a parameter Blob if another parameter Blob with the same name and accpetable size is found.
        /// </summary>
        /// <param name="b">Specifies the Blob to share.</param>
        /// <param name="rgMinShape">Specifies the minimum shape requried to share.</param>
        /// <returns>If the Blob is shared, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        protected bool shareParameter(Blob<T> b, List<int> rgMinShape)
        {
            LayerParameterEx<T> paramEx = m_param as LayerParameterEx<T>;
            if (paramEx == null)
                return false;

            if (paramEx.SharedBlobs == null)
                return false;

            return paramEx.SharedBlobs.Share(b, rgMinShape, false);
        }

        /// <summary>
        /// Attempts to share a Layer Blob if another parameter Blob with the same name and acceptable size is found.
        /// </summary>
        /// <param name="b">Specifies the Blob to share.</param>
        /// <param name="rgMinShape">Specifies the minimum shape requried to share.</param>
        /// <returns>If the Blob is shared, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        protected bool shareLayerBlob(Blob<T> b, List<int> rgMinShape)
        {
            LayerParameterEx<T> paramEx = m_param as LayerParameterEx<T>;
            if (paramEx == null)
                return false;

            if (paramEx.SharedLayerBlobs == null)
                return false;

            return paramEx.SharedLayerBlobs.Share(b, rgMinShape, false);
        }

        /// <summary>
        /// Attempts to share the Layer blobs and internal_blobs with matching names and sizes with those in another matching layer.
        /// </summary>
        /// <param name="layer">Specifies the layer who will use the shared blobs and internal blobs from the shared layer.</param>
        /// <returns>If the layer blobs and internal blobs are shared successfully <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        protected bool shareLayerBlobs(Layer<T> layer)
        {
            LayerParameterEx<T> paramEx = m_param as LayerParameterEx<T>;
            if (paramEx == null)
                return false;

            if (paramEx.SharedLayer == null)
                return false;

            if (paramEx.SharedLayer.blobs.Count != layer.blobs.Count)
                return false;

            for (int i = 0; i < paramEx.SharedLayer.blobs.Count; i++)
            {
                Blob<T> bSrc = paramEx.SharedLayer.blobs[i];
                Blob<T> bDst = layer.blobs[i];

                string strSrc = bSrc.shape_string;
                string strDst = bDst.shape_string;
                if (strSrc != strDst)
                {
                    m_log.WriteLine("WARNING: Cannot share blob '" + bSrc.Name + "'(" + strSrc + ") with blob '" + bDst.Name + "'(" + strDst + ") because the sizes differ!");
                    return false;
                }

                bSrc.Share(bDst);
            }

            if (paramEx.SharedLayer.internal_blobs.Count != layer.internal_blobs.Count)
                return false;

            for (int i = 0; i < paramEx.SharedLayer.internal_blobs.Count; i++)
            {
                Blob<T> bSrc = paramEx.SharedLayer.internal_blobs[i];
                Blob<T> bDst = layer.internal_blobs[i];

                string strSrc = bSrc.shape_string;
                string strDst = bDst.shape_string;
                if (strSrc != strDst)
                {
                    m_log.WriteLine("WARNING: Cannot share internal blob '" + bSrc.Name + "'(" + strSrc + ") with internal blob '" + bDst.Name + "'(" + strDst + ") because the sizes differ!");
                    return false;
                }

                bSrc.Share(bDst);
            }

            return true;
        }

        /// <summary>
        /// Returns the timing of the last forward pass in milliseconds.
        /// </summary>
        public double forward_timing
        {
            get { return m_dfForwardTiming; }
        }

        /// <summary>
        /// Returns the average timing of the forward passes in milliseconds.
        /// </summary>
        public double forward_timing_average
        {
            get { return m_dfForwardAverageTiming; }
        }

        /// <summary>
        /// Returns the timing of the last backward pass in milliseconds.
        /// </summary>
        public double backward_timing
        {
            get { return m_dfBackwardTiming; }
        }

        /// <summary>
        /// Returns the average timing of the backward passes in milliseconds.
        /// </summary>
        public double backward_timing_average
        {
            get { return m_dfBackwardAverageTiming; }
        }

        /// <summary>
        /// Enables/disables the pass-through mode.
        /// </summary>
        /// <remarks>
        /// When enabled, the forward pass merely compies the bottom inputs to the top outputs and returns.
        /// </remarks>
        /// <param name="bEnable">Enable/disable the pass-through mode.</param>
        public void SetEnablePassthrough(bool bEnable)
        {
            m_bEnablePassthrough = bEnable;
        }

        /// <summary>
        /// Returns the WorkspaceArgs used to share a workspace between Layers.
        /// </summary>
        /// <returns>The WorkspaceArgs are returned.</returns>
        protected virtual WorkspaceArgs getWorkspace()
        {
            if (OnGetWorkspace == null)
                return null;

            OnGetWorkspace(this, m_argsWs);

            return m_argsWs;
        }

        /// <summary>
        /// Sets the workspace size (in items) and returns <i>true</i> if set, <i>false</i> otherwise.
        /// </summary>
        /// <param name="lSize"></param>
        /// <returns></returns>
        protected virtual bool setWorkspace(ulong lSize)
        {
            if (OnSetWorkspace == null)
                return false;

            OnSetWorkspace(this, new WorkspaceArgs(0, lSize));
            return true;
        }

        /// <summary>
        /// Checks a Blob for NaNs and throws an exception if found.
        /// </summary>
        /// <param name="b">Specifies the Blob to check.</param>
        protected void check_nan(Blob<T> b)
        {
            double[] rg = convertD(b.update_cpu_data());

            for (int i = 0; i < rg.Length; i++)
            {
                if (double.IsNaN(rg[i]))
                    throw new Exception("NAN FOUND!");
            }
        }

        /// <summary>
        /// Converts a <i>double</i> to a generic.
        /// </summary>
        /// <param name="df">Specifies the <i>double</i> value.</param>
        /// <returns>Returns the generic value.</returns>
        protected T convert(double df)
        {
            return (T)Convert.ChangeType(df, typeof(T));
        }

        /// <summary>
        /// Converts a generic to a <i>double</i> value.
        /// </summary>
        /// <param name="df">Specifies the generic value.</param>
        /// <returns>The <i>double</i> value is returned.</returns>
        protected double convertD(T df)
        {
            return (double)Convert.ChangeType(df, typeof(double));
        }

        /// <summary>
        /// Converts a generic to a <i>float</i> value.
        /// </summary>
        /// <param name="df">Specifies the generic value.</param>
        /// <returns>The <i>float</i> value is returned.</returns>
        protected float convertF(T df)
        {
            return (float)Convert.ChangeType(df, typeof(float));
        }

        /// <summary>
        /// Converts an array of generic values into an array of <i>double</i> values.
        /// </summary>
        /// <param name="rg">Specifies the array of generic values.</param>
        /// <returns>The array of <i>double</i> values is returned.</returns>
        protected double[] convertD(T[] rg)
        {
            if (typeof(T) == typeof(double))
                return (double[])Convert.ChangeType(rg, typeof(double[]));

            double[] rgdf = new double[rg.Length];
            Array.Copy(rg, rgdf, rg.Length);

            return rgdf;
        }

        /// <summary>
        /// Converts an array of <i>double</i> values into an array of generic values.
        /// </summary>
        /// <param name="rg">Specifies the array of <i>double</i> values.</param>
        /// <returns>Returns an array of generic values.</returns>
        protected T[] convert(double[] rg)
        {
            if (typeof(T) == typeof(double))
                return (T[])Convert.ChangeType(rg, typeof(T[]));

            float[] rgf = new float[rg.Length];
            Array.Copy(Array.ConvertAll(rg, p => Convert.ToSingle(p)), rgf, rg.Length);

            return (T[])Convert.ChangeType(rgf, typeof(T[]));
        }

        /// <summary>
        /// Converts an array of <i>float</i> values into an array of generic values.
        /// </summary>
        /// <param name="rg">Specifies the array of <i>float</i> values.</param>
        /// <returns>Returns an array of generic values.</returns>
        protected float[] convertF(T[] rg)
        {
            if (typeof(T) == typeof(float))
                return (float[])Convert.ChangeType(rg, typeof(float[]));

            float[] rgf = new float[rg.Length];
            Array.Copy(Array.ConvertAll(rg, p => Convert.ToSingle(p)), rgf, rg.Length);

            return rgf;
        }

        /// <summary>
        /// Converts an array of <i>float</i> values into an array of generic values.
        /// </summary>
        /// <param name="rg">Specifies the array of <i>float</i> values.</param>
        /// <returns>Returns an array of generic values.</returns>
        protected T[] convert(float[] rg)
        {
            if (typeof(T) == typeof(float))
                return (T[])Convert.ChangeType(rg, typeof(T[]));

            T[] rgt = new T[rg.Length];
            Array.Copy(rg, rgt, rg.Length);

            return rgt;
        }

        /// <summary>
        /// Returns the integer value at a given index in a generic array.
        /// </summary>
        /// <param name="rg">Specifies the generic array.</param>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The value at the index is returned as an integer.</returns>
        protected int val_at(T[] rg, int nIdx)
        {
            return (int)Convert.ChangeType(rg[nIdx], typeof(int));
        }

        /// <summary>
        /// Returns the Size of a given two element Blob, such as one that stores Blob size information.
        /// </summary>
        /// <param name="b">Specifies the Blob.</param>
        /// <returns>The height and width are returned in a Size object.</returns>
        protected Size size_at(Blob<T> b)
        {
            T[] rg = b.update_cpu_data();
            int nHeight = val_at(rg, 0);
            int nWidth = (rg.Length > 1) ? val_at(rg, 1) : nHeight;
            return new Size(nWidth, nHeight);
        }

        private double getAveTiming(double dfInterval, double dfTiming, double dfAveTiming)
        {
            double dfRatio = 1.0 / m_dfAverageInterval;
            return (dfAveTiming * (1.0 - dfRatio)) + (dfTiming * dfRatio);
        }

        /// <summary>
        /// Create a new Layer based on the LayerParameter.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter that contains the LayerType to create.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used by some Layers when created.</param>
        /// <param name="imgDb">Optionally, specifies the MyCaffeImageDatabase used by data Layers.</param>
        /// <param name="trxinput">Optionally, specifies the transfer input object used by some of the data Layers.</param>
        /// <returns></returns>
        public static Layer<T> Create(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXImageDatabaseBase imgDb = null, TransferInput trxinput = null)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.ABSVAL:
                    return new AbsValLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY:
                    return new AccuracyLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.ARGMAX:
                    return new ArgMaxLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.BATCHNORM:
                    return new BatchNormLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.BATCHREINDEX:
                    return new BatchReindexLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.BNLL:
                    return new BNLLLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.BIAS:
                    return new BiasLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.CLIP:
                    return new ClipLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.CONCAT:
                    return new ConcatLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.CONSTANT:
                    return new ConstantLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.CONTRASTIVE_LOSS:
                    return new ContrastiveLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.CONVOLUTION:
                    return new ConvolutionLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.CROP:
                    return new CropLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.DECONVOLUTION:
                    return new DeconvolutionLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.IM2COL:
                    return new Im2colLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.DATA:
                    return new DataLayer<T>(cuda, log, p, imgDb, evtCancel);

                case LayerParameter.LayerType.DATA_NORMALIZER:
                    return new DataNormalizerLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.DEBUG:
                    return new DebugLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.DROPOUT:
                    return new DropoutLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.DUMMYDATA:
                    return new DummyDataLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.ELTWISE:
                    return new EltwiseLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.ELU:
                    return new ELULayer<T>(cuda, log, p);

                case LayerParameter.LayerType.EMBED:
                    return new EmbedLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.EUCLIDEAN_LOSS:
                    return new EuclideanLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.EXP:
                    return new ExpLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.FILTER:
                    return new FilterLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.FLATTEN:
                    return new FlattenLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.GRADIENTSCALER:
                    return new GradientScaleLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.HINGE_LOSS:
                    return new HingeLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.IMAGE_DATA:
                    return new ImageDataLayer<T>(cuda, log, p, evtCancel);

                case LayerParameter.LayerType.INFOGAIN_LOSS:
                    return new InfogainLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.INNERPRODUCT:
                    return new InnerProductLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.INPUT:
                    return new InputLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.LOG:
                    return new LogLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.LABELMAPPING:
                    return new LabelMappingLayer<T>(cuda, log, p, imgDb);

                case LayerParameter.LayerType.LRN:
                    return new LRNLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MAE_LOSS:
                    return new MAELossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MATH:
                    return new MathLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MEMORYDATA:
                    return new MemoryDataLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MEMORY_LOSS:
                    return new MemoryLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MISH:
                    return new MishLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MULTINOMIALLOGISTIC_LOSS:
                    return new MultinomialLogisticLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.MVN:
                    return new MVNLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.PARAMETER:
                    return new ParameterLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.POOLING:
                    return new PoolingLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.POWER:
                    return new PowerLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.PRELU:
                    return new PReLULayer<T>(cuda, log, p);

                case LayerParameter.LayerType.REDUCTION:
                    return new ReductionLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.RELU:
                    return new ReLULayer<T>(cuda, log, p);

                case LayerParameter.LayerType.RESHAPE:
                    return new ReshapeLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SCALE:
                    return new ScaleLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SIGMOID:
                    return new SigmoidLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SIGMOIDCROSSENTROPY_LOSS:
                    return new SigmoidCrossEntropyLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS:
                    return new SoftmaxCrossEntropyLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SILENCE:
                    return new SilenceLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SLICE:
                    return new SliceLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SOFTMAX:
                    return new SoftmaxLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SOFTMAXWITH_LOSS:
                    return new SoftmaxLossLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SPLIT:
                    return new SplitLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SPP:
                    return new SPPLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.SWISH:
                    return new SwishLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.TANH:
                    return new TanhLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.THRESHOLD:
                    return new ThresholdLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.TILE:
                    return new TileLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.LSTM_SIMPLE:
                    return new LSTMSimpleLayer<T>(cuda, log, p);

                case LayerParameter.LayerType.RNN:
                    return new RNNLayer<T>(cuda, log, p, evtCancel);

                case LayerParameter.LayerType.LSTM:
                    return new LSTMLayer<T>(cuda, log, p, evtCancel);

                case LayerParameter.LayerType.LSTM_UNIT:
                    return new LSTMUnitLayer<T>(cuda, log, p);

                default:
                    Layer<T> layer = createDynamicLayer(cuda, log, p, imgDb, evtCancel);
                    if (layer != null)
                        return layer;

                    log.FAIL("Unknown layer type: " + p.type.ToString());
                    break;
            }

            throw new NotImplementedException("The layer type: " + p.type.ToString() + " is not implemented yet.");
        }

        private static Layer<T> createDynamicLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabaseBase imgDb, CancelEvent evtCancel)
        {
            string strDir = System.IO.Path.GetDirectoryName(new System.Uri(System.Reflection.Assembly.GetExecutingAssembly().CodeBase).LocalPath);
            string[] rgstrFiles = Directory.GetFiles(strDir);

            foreach (string strFile in rgstrFiles)
            {
                FileInfo fi = new FileInfo(strFile);
                if (fi.Name.ToLower().IndexOf("mycaffe.layers.") == 0 && fi.Extension.ToLower() == ".dll")
                {
                    ILayerCreator icreator = loadCreator(strFile);
                    if (icreator != null)
                    {
                        Layer<T> layer;

                        if (typeof(T) == typeof(double))
                            layer = icreator.CreateDouble(cuda as CudaDnn<double>, log, p, evtCancel, imgDb) as Layer<T>;
                        else
                            layer = icreator.CreateSingle(cuda as CudaDnn<float>, log, p, evtCancel, imgDb) as Layer<T>;

                        if (layer != null)
                            return layer;
                    }
                }
            }

            return null;
        }

        private static ILayerCreator loadCreator(string strPath)
        {
            try
            {
                Assembly a = Assembly.LoadFile(strPath);

                foreach (Type t in a.GetTypes())
                {
                    if (t.IsPublic)
                    {
                        Type iface = t.GetInterface("ILayerCreator");
                        if (iface != null)
                        {
                            object obj = Activator.CreateInstance(t);
                            return (ILayerCreator)obj;
                        }
                    }
                }

                return null;
            }
            catch (Exception excpt)
            {
                return null;
            }
        }
    }

    /// <summary>
    /// The LayerParameterEx class is used when sharing another Net to conserve GPU memory and
    /// extends the LayerParameter with shared Blobs for this purpose.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LayerParameterEx<T> : LayerParameter
    {
        BlobCollection<T> m_colSharedBlobs = null;
        BlobCollection<T> m_colLayerBlobs = new BlobCollection<T>();
        Layer<T> m_layer;

        /// <summary>
        /// The LayerParameterEx constructor.
        /// </summary>
        /// <param name="p">Specifies the original LayerParameter that is wrapped.</param>
        /// <param name="colBlobs">Specifies the Net parameter Blobs to share.</param>
        /// <param name="colLayerBlobs">Specifies the Net layer Blobs to share.</param>
        /// <param name="sharedLayer">Specifies the shared Net layer matching this one that we are creating.</param>
        public LayerParameterEx(LayerParameter p, BlobCollection<T> colBlobs, BlobCollection<T> colLayerBlobs, Layer<T> sharedLayer)
            : base(p)
        {
            m_colSharedBlobs = colBlobs;
            m_colLayerBlobs = colLayerBlobs;
            m_layer = sharedLayer;
        }

        /// <summary>
        /// Returns the layer in the shared Net that matches this one.
        /// </summary>
        public Layer<T> SharedLayer
        {
            get { return m_layer; }
        }

        /// <summary>
        /// Returns the shared parameter Blobs.
        /// </summary>
        public BlobCollection<T> SharedBlobs
        {
            get { return m_colSharedBlobs; }
        }

        /// <summary>
        /// Returns the shared Layer Blobs.
        /// </summary>
        public BlobCollection<T> SharedLayerBlobs
        {
            get { return m_colLayerBlobs; }
        }

        /// <summary>
        /// Creates and returns a new copy of this instance.
        /// </summary>
        /// <param name="bCloneBlobs">Specifies whether or not to clone (or just share) the shared Blobs.</param>
        /// <returns>The cloned LayerParameter is returned.</returns>
        public override LayerParameter Clone(bool bCloneBlobs)
        {
            return new LayerParameterEx<T>(base.Clone(bCloneBlobs), m_colSharedBlobs, m_colLayerBlobs, m_layer);
        }
    }
}
