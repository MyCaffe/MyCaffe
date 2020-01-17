using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.common;
using MyCaffe.param;
using System.Threading;
using System.Diagnostics;

namespace MyCaffe.layers
{
    /// <summary>
    /// The BasePrefetchingDataLayer is the base class for data Layers that pre-fetch data before feeding the Blobs of data into the Net.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public abstract class BasePrefetchingDataLayer<T> : BaseDataLayer<T>
    {
        /// <summary>
        /// Specifies the pre-fetch cache.
        /// </summary>
        protected Batch<T>[] m_rgPrefetch;
        /// <summary>
        /// Specifies the cancellation event for the internal thread.
        /// </summary>
        protected CancelEvent m_evtCancel;
        BlockingQueue<Batch<T>> m_rgPrefetchFree;
        BlockingQueue<Batch<T>> m_rgPrefetchFull;
        Batch<T> m_prefetch_current = null;
        InternalThread<T> m_internalThread;
        Exception m_err = null;

        /// <summary>
        /// The BaseDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public BasePrefetchingDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabaseBase db, CancelEvent evtCancel)
            : base(cuda, log, p, db)
        {
            m_evtCancel = evtCancel;

            m_internalThread = new InternalThread<T>();
            m_internalThread.DoWork += new EventHandler<ActionStateArgs<T>>(m_internalThread_DoWork);
            m_internalThread.OnPreStop += internalThread_OnPreStop;
            m_internalThread.OnPreStart += internalThread_OnPreStart;

            if (m_evtCancel != null)
                m_internalThread.CancelEvent.AddCancelOverride(m_evtCancel);

            m_rgPrefetch = new Batch<T>[p.data_param.prefetch];
            m_rgPrefetchFree = new BlockingQueue<Batch<T>>(m_evtCancel);
            m_rgPrefetchFull = new BlockingQueue<Batch<T>>(m_evtCancel);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i] = new Batch<T>(cuda, log);
                m_rgPrefetchFree.Push(m_rgPrefetch[i]);
            }
        }

        private void internalThread_OnPreStart(object sender, EventArgs e)
        {
            m_rgPrefetchFree.Reset();
            m_rgPrefetchFull.Reset();
        }

        private void internalThread_OnPreStop(object sender, EventArgs e)
        {
            m_rgPrefetchFree.Abort();
            m_rgPrefetchFull.Abort();
            preStop();
        }

        protected virtual void preStop()
        {
        }

        /** @copydoc BaseDataLayer::dispose */
        protected override void dispose()
        {
            m_internalThread.StopInternalThread();

            if (m_rgPrefetchFull != null)
            {
                m_rgPrefetchFull.Dispose();
                m_rgPrefetchFull = null;
            }

            if (m_rgPrefetchFree != null)
            {
                m_rgPrefetchFree.Dispose();
                m_rgPrefetchFree = null;
            }

            if (m_evtCancel != null)
                m_internalThread.CancelEvent.RemoveCancelOverride(m_evtCancel);

            base.dispose();
        }

        /// <summary>
        /// LayerSetUp implements common data layer setup functonality, and calls
        /// DatLayerSetUp to do special data layer setup for individual layer types.
        /// </summary>
        /// <remarks>
        /// This method should not be overriden.
        /// </remarks>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Data.update_cpu_data();

                if (m_bOutputLabels)
                    m_rgPrefetch[i].Label.update_cpu_data();
            }

            m_transformer.InitRand();

            if (!delayPrefetch)
            {
                m_log.WriteLine("Initializing prefetch for '" + m_param.name + "'...");
                statupPrefetch();
            }
            else
            {
                m_log.WriteLine("Delaying prefetch for '" + m_param.name + "'...");
            }
        }

        /// <summary>
        /// Specifies whether or not to delay the prefetch.
        /// </summary>
        protected virtual bool delayPrefetch
        {
            get { return false; }
        }

        /// <summary>
        /// Starts the prefetch thread.
        /// </summary>
        protected void statupPrefetch()
        {
            m_err = null;
            m_internalThread.StartInternalThread(m_cuda, m_log, m_cuda.GetDeviceID());
            m_log.WriteLine("Prefetch initialized for '" + m_param.name + "'.");
        }

        void m_internalThread_DoWork(object sender, ActionStateArgs<T> e)
        {
            CudaDnn<T> cuda = m_cuda;
            Log log = m_log;
            long hStream = 0; // cuda.CreateStream(false);

            try
            {
                while (!m_internalThread.CancellationPending)
                {
                    Batch<T> batch = new Batch<T>(cuda, log);

                    if (m_rgPrefetchFree.Pop(ref batch))
                    {
                        load_batch(batch);

                        batch.Data.AsyncGpuPush(hStream);
                        if (hStream != 0)
                            m_cuda.SynchronizeStream(hStream);

                        if (m_bOutputLabels)
                        {
                            batch.Label.AsyncGpuPush(hStream);
                            if (hStream != 0)
                                m_cuda.SynchronizeStream(hStream);
                        }

                        m_rgPrefetchFull.Push(batch);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            catch (Exception excpt)
            {
                m_err = excpt;
                m_rgPrefetchFull.Abort();
                m_rgPrefetchFree.Abort();
                throw excpt;
            }
            finally
            {
                if (hStream != 0)
                    cuda.FreeStream(hStream);
            }
        }

        /// <summary>
        /// Provides a final processing step that may be utilized by derivative classes.
        /// </summary>
        /// <param name="blobTop">Specifies the top blob just about to be set out the forward operation as the Top[0] blob.</param>
        protected virtual void final_process(Blob<T> blobTop)
        {
            return;
        }

        /// <summary>
        /// The forward override implements the functionality to load pre-fetch data and feed it into the 
        /// top (output) Blobs.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_prefetch_current != null)
                m_rgPrefetchFree.Push(m_prefetch_current);

            if (m_rgPrefetchFull.Pop(ref m_prefetch_current))
            {
                // Reshape to loaded data.
                colTop[0].ReshapeLike(m_prefetch_current.Data);

                // Copy the data.
                m_cuda.copy(m_prefetch_current.Data.count(), m_prefetch_current.Data.gpu_data, colTop[0].mutable_gpu_data);
                final_process(colTop[0]);

                //-----------------------------------------
                // If the blob has a fixed range set, set 
                //  the range in the top data.
                //-----------------------------------------
                m_transformer.SetRange(colTop[0]);

                if (m_bOutputLabels)
                {
                    // Reshape to loaded labels.
                    colTop[1].ReshapeLike(m_prefetch_current.Label);

                    // Copy the labels.
                    m_cuda.copy(m_prefetch_current.Label.count(), m_prefetch_current.Label.gpu_data, colTop[1].mutable_gpu_data);
                }
            }
            else if (m_err != null)
            {
                throw m_err;
            }
        }

        /// <summary>
        /// The load_batch abstract function should be overriden by each derivative data Layer to 
        /// load a batch of data.
        /// </summary>
        /// <param name="batch">Specifies the Batch of data loaded.</param>
        protected abstract void load_batch(Batch<T> batch);
    }

    /// <summary>
    /// The Batch contains both the data and label Blobs of the batch.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Batch<T> : IDisposable 
    {
        Blob<T> m_data;
        Blob<T> m_label;

        /// <summary>
        /// The Batch constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        public Batch(CudaDnn<T> cuda, Log log)
        {
            m_data = new Blob<T>(cuda, log);
            m_label = new Blob<T>(cuda, log);
        }

        /// <summary>
        /// Release all GPU and host resources used (if any).
        /// </summary>
        public void Dispose()
        {
        }

        /// <summary>
        /// Returns the data Blob of the batch.
        /// </summary>
        public Blob<T> Data
        {
            get { return m_data; }
        }

        /// <summary>
        /// Returns the label Blob of the batch.
        /// </summary>
        public Blob<T> Label
        {
            get { return m_label; }
        }
    }
}
