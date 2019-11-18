using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using MyCaffe.basecode;

namespace MyCaffe.common
{
    /// <summary>
    /// The InternalThread manages an internal thread used for Parallel and data collection operations.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class InternalThread<T> : IDisposable
    {
        Task m_task = null;
        Thread m_thread = null;
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        AutoResetEvent m_evtDone = new AutoResetEvent(false);
        bool m_bUseThread = true;

        /// <summary>
        /// The DoWork event is the working thread function.
        /// </summary>
        public event EventHandler<ActionStateArgs<T>> DoWork;

        /// <summary>
        /// The InternalThread constructor.
        /// </summary>
        /// <param name="bUseThreadVsTask">Optionally, specifies to use a Thread vs a Task (default = false).</param>
        public InternalThread(bool bUseThreadVsTask = false)
        {
            m_bUseThread = bUseThreadVsTask;
        }

        /// <summary>
        /// Releases all resources used by the InernalThread.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called from Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            if (m_evtCancel != null)
                m_evtCancel.Set();

            StopInternalThread();

            if (m_evtCancel != null)
            {
                m_evtCancel.Dispose();
                m_evtCancel = null;
            }
        }

        /// <summary>
        /// Releases all resources used by the InernalThread.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// Starts running the internal thread function which then calls the DoWork event.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda placed in the ActionStartArgs passed along to DoWork.</param>
        /// <param name="log">Specifies the Log for output, placed in the ActionStartArgs passed along to DoWork.</param>
        /// <param name="nDeviceID">Optionally, specifies the DeviceID placed in the ActionStartArgs passed along to DoWork.</param>
        /// <param name="arg">Optionally, specifies an argument defined by the caller.</param>
        public void StartInternalThread(CudaDnn<T> cuda, Log log, int nDeviceID = 0, object arg = null)
        {
            m_evtCancel.Reset();

            if (m_bUseThread)
            {
                if (m_thread == null)
                {
                    m_thread = new Thread(new ParameterizedThreadStart(InternalThreadEntry));
                    m_thread.Start(new ActionStateArgs<T>(cuda, log, m_evtCancel, nDeviceID, arg));
                }
            }
            else
            {
                if (m_task == null)
                {
                    Action<object> action = new Action<object>(InternalThreadEntry);
                    m_task = Task.Factory.StartNew(action, new ActionStateArgs<T>(cuda, log, m_evtCancel, nDeviceID, arg), TaskCreationOptions.LongRunning);
                }
            }
        }

        /// <summary>
        /// Stops the internal thread.
        /// </summary>
        public void StopInternalThread()
        {
            if (m_thread != null)
            {
                m_evtCancel.Set();
                if (!m_evtDone.WaitOne(10000))
                    throw new Exception("Failed to stop the internal thread!");
                m_thread = null;
            }

            if (m_task != null)
            {
                m_evtCancel.Set();
                if (!m_evtDone.WaitOne(10000))
                    throw new Exception("Failed to stop the internal thread!");
                m_task = null;
            }
        }

        /// <summary>
        /// Specifies the internal thread entry.
        /// </summary>
        /// <param name="obj"></param>
        protected void InternalThreadEntry(object obj)
        {
            ActionStateArgs<T> state = obj as ActionStateArgs<T>;

            m_evtCancel.Reset();

            try
            {
                if (DoWork != null)
                    DoWork(this, state);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_evtDone.Set();
            }
        }

        /// <summary>
        /// Returns whether or not a cancellation is pending.
        /// </summary>
        public bool CancellationPending
        {
            get
            {
                if (m_evtCancel == null)
                    return true;

                return m_evtCancel.WaitOne(0);
            }
        }

        /// <summary>
        /// Returns the CancelEvent that terminates the thread when Set.
        /// </summary>
        public ManualResetEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        /// <summary>
        /// Returns whether or not the internal thread has been started.
        /// </summary>
        public bool IsStarted
        {
            get
            {
                if (m_task == null)
                    return false;

                return true;
            }
        }
    }

    /// <summary>
    /// The ActionStateArgs are sent to the DoWork event when fired from the InternalThreadEntry.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ActionStateArgs<T> : EventArgs
    {
        CudaDnn<T> m_cuda = null;
        Log m_log = null;
        ManualResetEvent m_evtCancel;
        int m_nDeviceID = 0;
        object m_arg = null;

        /// <summary>
        /// The ActionStateArgs constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="evtCancel">Specifies the CancelEvent that when Set signals to DoWork that it should terminate.</param>
        /// <param name="nDeviceID">Optionally, specifies the DeviceID.</param>
        /// <param name="arg">Optionally, specifies an argument defined by the caller.</param>
        public ActionStateArgs(CudaDnn<T> cuda, Log log, ManualResetEvent evtCancel, int nDeviceID = 0, object arg = null)
            : base()
        {
            m_log = log;
            m_cuda = cuda;
            m_evtCancel = evtCancel;
            m_nDeviceID = nDeviceID;
            m_arg = arg;
        }

        /// <summary>
        /// Get/set the Cuda Dnn connection to Cuda.
        /// </summary>
        public CudaDnn<T> cuda
        {
            get { return m_cuda; }
            set { m_cuda = value; }
        }

        /// <summary>
        /// Returns the Log used for output.
        /// </summary>
        public Log log
        {
            get { return m_log; }
        }

        /// <summary>
        /// Returns the CancelEvent used to cancel the thread.
        /// </summary>
        public ManualResetEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        /// <summary>
        /// Returns the Device ID of the device to use in the thread.
        /// </summary>
        public int DeviceID
        {
            get { return m_nDeviceID; }
        }

        /// <summary>
        /// Returns the user supplied argument.
        /// </summary>
        public object Arg
        {
            get { return m_arg; }
        }
    }
}
