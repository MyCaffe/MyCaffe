using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The ComputeGraph class provides a simple computation graph of operations used in a forward pass
    /// that are stored in an array on each call and then unwound with calls that calculate the gradients
    /// on the backward pass.
    /// </summary>
    /// <remarks>
    /// This class is a re-write of the ComputeGraph originally created by Mohamed Ashmawy in the
    /// open-source project [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn) distributed under the MIT license.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type.</typeparam>
    public class ComputeGraph<T> : IDisposable
    {
        Blob<T> m_blobWork;
        CudaDnn<T> m_cuda;
        Log m_log;
        List<Tuple<string, Action>> m_rgBackprop = new List<Tuple<string, Action>>();
        Dictionary<string, Blob<T>[]> m_rgDebug = new Dictionary<string, Blob<T>[]>();
        bool m_bNeedsBackprop = true;
        bool m_bCheckForNans = false;
        bool m_bClipGradients = false;
        bool m_bAddDebug = false;
        int m_nAxis = 0;
        string m_strMarker = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to CUDA.</param>
        /// <param name="log">Specifies the general output log.</param>
        /// <param name="nAxis">Specifies the axis under which to perform all of the actions in the graph.</param>
        /// <param name="bNeedsBackprop">Optionally, specifies whether or not to back propagate (default = true).</param>
        /// <param name="bClipGradients">Optionally, specifies whether or not to clip the gradients (default = false, Debug only).</param>
        /// <param name="bCheckNans">Optionally, specifies whether or not to check for nans (default = false, Debug only).</param>
        /// <param name="bAddDebug">Optionally, specifies to add debug information (default = false, Debug only).</param>
        public ComputeGraph(CudaDnn<T> cuda, Log log, int nAxis, bool bNeedsBackprop = true, bool bClipGradients = false, bool bCheckNans = false, bool bAddDebug = false)
        {
            m_cuda = cuda;
            m_log = log;
            m_blobWork = new Blob<T>(cuda, log);
            m_bNeedsBackprop = bNeedsBackprop;
            m_bCheckForNans = bCheckNans;
            m_bClipGradients = bClipGradients;
            m_bAddDebug = bAddDebug;
            m_nAxis = nAxis;            
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }
        }

        private void add_debug(string str, params Blob<T>[] rg)
        {
            string strName = m_rgDebug.Count.ToString() + "_" + str;
            m_rgDebug.Add(strName, rg);
        }

        /// <summary>
        /// Returns a dictionary of Blobs used during each operation, only filled when 'bAddDebug' = true in the constructor.
        /// </summary>
        public Dictionary<string, Blob<T>[]> Debug
        {
            get { return m_rgDebug; }
        }

        /// <summary>
        /// Get/set a string marker added to the debug information and used to indicate where in the code a given operation takes place.
        /// </summary>
        public string marker
        {
            get { return m_strMarker; }
            set { m_strMarker = value; }
        }

        /// <summary>
        /// Get/set whether or not to back propagate.
        /// </summary>
        public bool needs_backprop
        {
            get { return m_bNeedsBackprop; }
            set { m_bNeedsBackprop = value; }
        }

        /// <summary>
        /// Returns the axis on which all operations are performed.
        /// </summary>
        public int axis
        {
            get { return m_nAxis; }
        }
        
        private Blob<T> work
        {
            get { return m_blobWork; }
        }

        private int input_count(Blob<T> b)
        {
            if (b.num_axes <= m_nAxis + 2)
                return 1;

            return b.count(m_nAxis + 2);
        }

        private void clip_gradient1(Blob<T> b)
        {
            float[] rg = Utility.ConvertVecF<T>(b.mutable_cpu_diff);
            
            for (int i = 0; i < rg.Length; i++)
            {
                if (Math.Abs(rg[i]) < 0.000001)
                    rg[i] = 0;
                else
                    rg[i] = (float)Math.Round(rg[i], 7);
            }

            b.mutable_cpu_diff = Utility.ConvertVec<T>(rg);
        }

        private void clip_gradient(params Blob<T>[] rg)
        {
            foreach (Blob<T> b in rg)
            {
                clip_gradient1(b);
            }
        }

        private T[] round(T[] rgData1, int nDecimals)
        {
            float[] rgData = Utility.ConvertVecF<T>(rgData1);

            for (int i = 0; i < rgData.Length; i++)
            {
                rgData[i] = (float)Math.Round(rgData[i], nDecimals);
            }

            return Utility.ConvertVec<T>(rgData);
        }

        private void check_nan(params Blob<T>[] rg)
        {
            for (int i = 0; i < rg.Length; i++)
            {
                work.ReshapeLike(rg[i]);
                Tuple<double, double, double, double> data = rg[i].minmax_data(work, true);
                Tuple<double, double, double, double> diff = rg[i].minmax_diff(work, true);

                double dfDataNanCount = data.Item3;
                double dfDataInfCount = data.Item4;
                double dfDiffNanCount = diff.Item3;
                double dfDiffInfCount = diff.Item4;

                if (dfDataNanCount > 0 || dfDataInfCount > 0)
                    throw new Exception("NAN or INF detected in " + rg[i].Name + " data!");

                if (dfDataNanCount > 0 || dfDataInfCount > 0)
                    throw new Exception("NAN or INF detected in " + rg[i].Name + " diff!");
            }
        }

        private void apply(Blob<T> work, Blob<T> btm)
        {
            m_cuda.add(btm.count(), work.gpu_diff, btm.gpu_diff, btm.mutable_gpu_diff);
        }

        /// <summary>
        /// DebugOp operation places a debug stub in the backpropagation chain for debugging only.
        /// </summary>
        public void DebugOp(params Blob<T>[] rgB)
        {
            string strMarker = marker;
            Action backward = () =>
            {
                string str = "";
                for (int i = 0; i < rgB.Length; i++)
                {
                    str += rgB[i].Name + ",";
                }
                str = str.TrimEnd(',');

                Trace.WriteLine("Debugging at " + strMarker + " blobs: " + str);
            };
            m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
        }

        /// <summary>
        /// Round operation, rounds the values to the nearest specified decimal.
        /// </summary>
        /// <param name="b">Specifies the blob to round.</param>
        /// <param name="nDecimals">Optionally, specifies the decimals (defautl = 6).</param>
        /// <returns>Returns the blob.</returns>
        public Blob<T> Round(Blob<T> b, int nDecimals = 6)
        {
            b.mutable_cpu_data = round(b.mutable_cpu_data, nDecimals);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    b.mutable_cpu_diff = round(b.mutable_cpu_diff, nDecimals);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return b;
        }

        /// <summary>
        /// PeeKRow operation copies data and diffs from one row from within the bottom matrix and places it in the top vector.
        /// </summary>
        /// <param name="btm">Specifies the input matrix.</param>
        /// <param name="top">Specifies the output vector.</param>
        /// <param name="ix">Specifies the row within the input matrix to copy.</param>
        /// <returns>The top blob is returned.</returns>
        public Blob<T> PeekRow(Blob<T> btm, Blob<T> top, int ix)
        {
            string strMarker = marker;
            List<int> rgShape = new List<int>() { 1, 1 };
            rgShape[1] = btm.count(2);
            top.Reshape(rgShape);

            int nSpatialDim = btm.count(2);
            m_cuda.copy(nSpatialDim, btm.gpu_data, top.mutable_gpu_data, nSpatialDim * ix, 0);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    m_cuda.copy(nSpatialDim, top.gpu_diff, btm.mutable_gpu_diff, 0, nSpatialDim * ix);
                    m_cuda.copy(nSpatialDim, top.gpu_data, btm.mutable_gpu_data, 0, nSpatialDim * ix);

                    if (m_bCheckForNans)
                        check_nan(btm);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// PeekItem operation copies a single item from the bottom to the top.
        /// </summary>
        /// <param name="btm">Specifies the input vector.</param>
        /// <param name="top">Specifies the output matrix.</param>
        /// <param name="ix">Specifies the row into which the data is copied within the top.</param>
        public void PeekItem(Blob<T> btm, Blob<T> top, int ix)
        {
            string strMarker = marker;
            int nSpatialDim = btm.count(m_nAxis);

            m_cuda.copy(nSpatialDim, btm.gpu_data, top.mutable_gpu_data, 0, nSpatialDim * ix);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    btm.Reshape(1, 1, 1, 1);
                    m_cuda.copy(nSpatialDim, top.gpu_diff, btm.mutable_gpu_diff, nSpatialDim * ix, 0);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }
        }

        /// <summary>
        /// CopyToRow operation copies the bottom vector into the top matrix.
        /// </summary>
        /// <param name="btm">Specifies the input vector.</param>
        /// <param name="top">Specifies the output matrix.</param>
        /// <param name="ix">Specifies the row into which the data is copied within the top.</param>
        /// <param name="bCopyDiff">Optionally, specifies to copy the diff and the data.</param>
        /// <returns>The top blob is returned.</returns>
        public Blob<T> CopyToRow(Blob<T> btm, Blob<T> top, int ix, bool bCopyDiff = false)
        {
            string strMarker = marker;
            int nSpatialDim = btm.count(m_nAxis);

            if (btm.count() == 0)
                top.SetData(0, nSpatialDim * ix, nSpatialDim);
            else
                m_cuda.copy(nSpatialDim, btm.gpu_data, top.mutable_gpu_data, 0, nSpatialDim * ix);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    if (bCopyDiff)
                        m_cuda.copy(nSpatialDim, top.gpu_diff, btm.mutable_gpu_diff, nSpatialDim * ix, 0);
                    m_cuda.copy(nSpatialDim, top.gpu_data, btm.mutable_gpu_data, nSpatialDim * ix, 0);

                    if (m_bCheckForNans)
                        check_nan(btm);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// CopyToCache operation copies the blob into the cache.
        /// </summary>
        /// <param name="btm">Specifies the input vector to copy.</param>
        /// <param name="cache">Specifies the cache.</param>
        public void CopyToCache(Blob<T> btm, Cache<T> cache)
        {
            string strMarker = marker;
            cache.CopyToCache(btm, m_nAxis);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    cache.CopyFromCache(btm, m_nAxis);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }
        }

        /// <summary>
        /// 'tanh' operation runs the tanh on each item in the btm and places the results in the top.
        /// </summary>
        /// <param name="btm">Specifies the input data.</param>
        /// <param name="top">Specifies the output data.</param>
        /// <returns>The top blob is returned.</returns>
        public Blob<T> tanh(Blob<T> btm, Blob<T> top)
        {
            string strMarker = marker;
            top.ReshapeLike(btm);

            m_cuda.tanh_fwd(btm.count(), btm.gpu_data, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    work.ReshapeLike(btm);
                    m_cuda.tanh_bwd(top.count(), top.gpu_diff, top.gpu_data, work.mutable_gpu_diff);
                    apply(work, btm);

                    if (m_bClipGradients)
                        clip_gradient(btm);
                    if (m_bCheckForNans)
                        check_nan(btm);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - tanh", btm, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// 'sigmoid' operation runs the sigmoid on each item in the btm and places the results in the top.
        /// </summary>
        /// <param name="btm">Specifies the input data.</param>
        /// <param name="top">Specifies the output data.</param>
        /// <returns>The top blob is returned.</returns>
        public Blob<T> sigmoid(Blob<T> btm, Blob<T> top)
        {
            string strMarker = marker;
            top.ReshapeLike(btm);

            m_cuda.sigmoid_fwd(btm.count(), btm.gpu_data, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    work.ReshapeLike(btm);
                    m_cuda.sigmoid_bwd(top.count(), top.gpu_diff, top.gpu_data, work.mutable_gpu_diff);
                    apply(work, btm);

                    if (m_bClipGradients)
                        clip_gradient(btm);
                    if (m_bCheckForNans)
                        check_nan(btm);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - sigmoid", btm, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// 'elthmul' operation mutliplies each element of the 'btm1' with the 'btm2' and places the results in 'top'.
        /// </summary>
        /// <param name="btm1">Specifies the first input.</param>
        /// <param name="btm2">Specifies the second input.</param>
        /// <param name="top">Specifies the output.</param>
        /// <returns>The 'top' Blob is returned.</returns>
        public Blob<T> eltmul(Blob<T> btm1, Blob<T> btm2, Blob<T> top)
        {
            string strMarker = marker;
            top.ReshapeLike(btm1);

            m_cuda.mul(top.count(), btm1.gpu_data, btm2.gpu_data, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    work.ReshapeLike(btm1);
                    m_cuda.mul(btm2.count(), btm2.gpu_data, top.gpu_diff, work.mutable_gpu_diff);
                    apply(work, btm1);
                    work.ReshapeLike(btm2);
                    m_cuda.mul(btm1.count(), btm1.gpu_data, top.gpu_diff, work.mutable_gpu_diff);
                    apply(work, btm2);

                    if (m_bClipGradients)
                        clip_gradient(btm1, btm2);
                    if (m_bCheckForNans)
                        check_nan(btm1, btm2);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - eltmul", btm1, btm2, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// 'scalemul' operation mutliplies each element of the 'btm1' with the first item within 'btm2' and places the results in 'top'.
        /// </summary>
        /// <param name="btm1">Specifies the first input.</param>
        /// <param name="btm2">Specifies the second input.</param>
        /// <param name="top">Specifies the output.</param>
        /// <param name="nIdx">Specifies the index of btm2 to use for scaling.</param>
        /// <returns>The 'top' Blob is returned.</returns>
        public Blob<T> scalemul(Blob<T> btm1, Blob<T> btm2, Blob<T> top, int nIdx = 0)
        {
            string strMarker = marker;
            top.ReshapeLike(btm1);

            T fScale = btm2.GetData(nIdx);
            m_cuda.scale(top.count(), fScale, btm1.gpu_data, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    work.ReshapeLike(btm1);
                    m_cuda.scale(top.count(), fScale, top.gpu_diff, work.mutable_gpu_diff);
                    apply(work, btm1);

                    work.ReshapeLike(btm2);
                    float fDot = m_cuda.dot_float(btm1.count(), btm1.gpu_data, top.gpu_diff);
                    work.SetDiff(0);
                    work.SetDiff(fDot, nIdx);
                    apply(work, btm2);

                    if (m_bClipGradients)
                        clip_gradient(btm1, btm2);
                    if (m_bCheckForNans)
                        check_nan(btm1, btm2);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - scalemul", btm1, btm2, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// 'mul' operation performs a blas gemm operation on the 'btm1' matrix with the 'btm2' matrix and places the results in 'top'.
        /// </summary>
        /// <param name="btm1">Specifies the first input.</param>
        /// <param name="btm2">Specifies the second input.</param>
        /// <param name="top">Specifies the output.</param>
        /// <param name="bAccumulateGrad">Optionally, specifies to accumulate the gradient (default = true).</param>
        /// <returns>The 'top' Blob is returned.</returns>
        public Blob<T> mul(Blob<T> btm1, Blob<T> btm2, Blob<T> top, bool bAccumulateGrad = true)
        {
            string strMarker = marker;
            int nM = btm1.shape(m_nAxis);
            int nN = btm2.count(m_nAxis + 1);
            int nK = btm1.count(m_nAxis + 1);

            List<int> rgShape = Utility.Create<int>(m_nAxis, 1);
            rgShape.Add(nM);
            rgShape.Add(nN);

            top.Reshape(rgShape);

            m_cuda.gemm(false, false, nM, nN, nK, Blob<T>.One, btm1.gpu_data, btm2.gpu_data, Blob<T>.Zero, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    T fBeta = (bAccumulateGrad) ? Blob<T>.One : Blob<T>.Zero;
                    m_cuda.gemm(false, true, nM, nK, nN, Blob<T>.One, top.gpu_diff, btm2.gpu_data, fBeta, btm1.mutable_gpu_diff);
                    m_cuda.gemm(true, false, nK, nN, nM, Blob<T>.One, btm1.gpu_data, top.gpu_diff, fBeta, btm2.mutable_gpu_diff);

                    if (m_bClipGradients)
                        clip_gradient(btm1, btm2);
                    if (m_bCheckForNans)
                        check_nan(btm1, btm2);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - mul", btm1, btm2, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// 'elthmul' operation adds each element of the 'btm1' with the 'btm2' and places the results in 'top'.
        /// </summary>
        /// <param name="btm1">Specifies the first input.</param>
        /// <param name="btm2">Specifies the second input.</param>
        /// <param name="top">Specifies the output.</param>
        /// <param name="bAccumulateGrad">Optionally, specifies whether or not to acumulate the gradients (default = true).</param>
        /// <returns>The 'top' Blob is returned.</returns>
        public Blob<T> add(Blob<T> btm1, Blob<T> btm2, Blob<T> top, bool bAccumulateGrad = true)
        {
            string strMarker = marker;
            top.ReshapeLike(btm1);

            m_cuda.add(top.count(), btm1.gpu_data, btm2.gpu_data, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    if (!bAccumulateGrad)
                    {
                        btm1.SetDiff(0);
                        btm2.SetDiff(0);
                    }

                    m_cuda.add(btm1.count(), btm1.gpu_diff, top.gpu_diff, btm1.mutable_gpu_diff);
                    m_cuda.add(btm2.count(), btm2.gpu_diff, top.gpu_diff, btm2.mutable_gpu_diff);

                    if (m_bClipGradients)
                        clip_gradient(btm1, btm2);
                    if (m_bCheckForNans)
                        check_nan(btm1, btm2);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - add", btm1, btm2, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// 'clear_grad' operation only runs on the backward pass and zeros out the gradients on an input.
        /// </summary>
        /// <param name="b">Specifies the input.</param>
        /// <returns>The input Blob is returned.</returns>
        public Blob<T> clear_grad(Blob<T> b)
        {
            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    b.SetDiff(0);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return b;
        }

        /// <summary>
        /// 'clear_grad' operation only runs on the backward pass and zeros out the gradients of the inputs.
        /// </summary>
        /// <param name="rg">Specifies an array of inputs.</param>
        public void clear_grad(BlobCollection<T> rg)
        {
            foreach (Blob<T> b in rg)
            {
                clear_grad(b);
            }
        }

        /// <summary>
        /// 'softmax' operation runs the softmax on each item in the btm and places the results in the top.
        /// </summary>
        /// <param name="btm">Specifies the input data.</param>
        /// <param name="top">Specifies the output data.</param>
        /// <returns>The top blob is returned.</returns>
        public Blob<T> softmax(Blob<T> btm, Blob<T> top)
        {
            string strMarker = marker;
            top.ReshapeLike(btm);

            int nOuterNum = btm.count(0, m_nAxis);
            int nInnerNum = btm.count(m_nAxis + 1);
            int nChannels = top.shape(m_nAxis);
            int nCount = btm.count();

            work.ReshapeLike(top);

            m_cuda.copy(nCount, btm.gpu_data, top.mutable_gpu_data);

            // We need to subtract the max to avoid numerical issues, compute the exp
            // and then normalize.
            // compute max.
            m_cuda.channel_max(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, top.gpu_data, work.mutable_gpu_data);
            // subtract
            m_cuda.channel_sub(nCount, nOuterNum, nChannels, nInnerNum, work.gpu_data, top.mutable_gpu_data);
            // exponentiate
            m_cuda.exp(nCount, top.gpu_data, top.mutable_gpu_data);
            // Sum after exp
            m_cuda.channel_sum(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, top.gpu_data, work.mutable_gpu_data);
            // divide
            m_cuda.channel_div(nCount, nOuterNum, nChannels, nInnerNum, work.gpu_data, top.mutable_gpu_data);

            if (m_bNeedsBackprop)
            {
                Action backward = () =>
                {
                    work.ReshapeLike(top);
                    m_cuda.copy(nCount, top.gpu_diff, work.mutable_gpu_diff);

                    // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
                    m_cuda.channel_dot(nOuterNum * nInnerNum, nOuterNum, nChannels, nInnerNum, top.gpu_diff, top.gpu_data, work.mutable_gpu_data);
                    m_cuda.channel_sub(nCount, nOuterNum, nChannels, nInnerNum, work.gpu_data, work.mutable_gpu_diff);

                    // elementwise multiplication
                    m_cuda.mul(nCount, work.gpu_diff, top.gpu_data, work.mutable_gpu_diff);
                    apply(work, btm);

                    if (m_bClipGradients)
                        clip_gradient(btm);
                    if (m_bCheckForNans)
                        check_nan(btm);
                    if (m_bAddDebug)
                        add_debug(strMarker + " - softmax", btm, top);
                };
                m_rgBackprop.Add(new Tuple<string, Action>(m_strMarker, backward));
            }

            return top;
        }

        /// <summary>
        /// Returns the backward operation count.
        /// </summary>
        public int BackwardCount
        {
            get { return m_rgBackprop.Count; }
        }

        /// <summary>
        /// Runs a backward operation at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the backward operation to run.</param>
        public void BackwardOne(int nIdx)
        {
            m_rgBackprop[nIdx].Item2();
        }

        /// <summary>
        /// Runs a backward operation on all items starting from the last and running through the first.
        /// </summary>
        /// <param name="bClear">Optionally, specifies to clear the list of operations upon completing (default = false).</param>
        public void Backward(bool bClear = false)
        {
            for (int i = m_rgBackprop.Count - 1; i >= 0; i--)
            {
                m_rgBackprop[i].Item2();
            }

            if (bClear)
                m_rgBackprop.Clear();
        }

        /// <summary>
        /// Clears all backward operations from the list.
        /// </summary>
        public void Clear()
        {
            m_rgBackprop.Clear();
        }
    }

    /// <summary>
    /// The Cache class is used to cache blobs over time.
    /// </summary>
    /// <typeparam name="T">Specifies the base type.</typeparam>
    public class Cache<T> : IDisposable
    {
        CudaDnn<T> m_cuda;
        Blob<T> m_blobCache;
        int m_nCacheIdx = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to Cuda.</param>
        /// <param name="log">Specifies the log for output.</param>
        public Cache(CudaDnn<T> cuda, Log log)
        {
            m_cuda = cuda;
            m_blobCache = new Blob<T>(cuda, log, false);
            m_blobCache.Name = "cache";
        }

        /// <summary>
        /// Release any resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobCache != null)
            {
                m_blobCache.Dispose();
                m_blobCache = null;
            }
        }

        /// <summary>
        /// Create the cache memory.
        /// </summary>
        /// <param name="nCount">Specifies the number of items in the cache.</param>
        /// <param name="rgItemShape">Specifies the shape of each item within the cache.</param>
        public void Create(int nCount, List<int> rgItemShape)
        {
            List<int> rgShape = new List<int>(rgItemShape);
            rgShape.Insert(0, nCount);
            m_blobCache.Reshape(rgShape);
        }

        /// <summary>
        /// Resets the cache.
        /// </summary>
        public void Reset()
        {
            m_nCacheIdx = 0;
            m_blobCache.SetData(0);
        }

        /// <summary>
        /// Copies a blob to the current location in the cache.
        /// </summary>
        /// <param name="b">Specifies the blob to copy.</param>
        /// <param name="nAxis">Specifies the axis where all data to the right is copied.</param>
        public void CopyToCache(Blob<T> b, int nAxis)
        {
            int nSpatialDim = b.count(nAxis);

            if (m_nCacheIdx >= m_blobCache.num)
                throw new Exception("The cache is full!");

            m_cuda.copy(nSpatialDim, b.gpu_data, m_blobCache.mutable_gpu_data, 0, nSpatialDim * m_nCacheIdx);
            m_nCacheIdx++;
        }

        /// <summary>
        /// Copies a value from the current location in the cache to the blob.
        /// </summary>
        /// <param name="b">Specifies the blob where the data is copied.</param>
        /// <param name="nAxis">Specifies the axis where all data to the right is copied.</param>
        public void CopyFromCache(Blob<T> b, int nAxis)
        {
            int nSpatialDim = b.count(nAxis);

            m_nCacheIdx--;
            if (m_nCacheIdx < 0)
                throw new Exception("The cache is empty!");

            m_cuda.copy(nSpatialDim, m_blobCache.gpu_data, b.mutable_gpu_data, nSpatialDim * m_nCacheIdx, 0);
        }
    }
}
