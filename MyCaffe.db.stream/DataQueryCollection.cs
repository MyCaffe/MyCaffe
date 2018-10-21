using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    /// <summary>
    /// The DataQueryCollection manages all active data queries.
    /// </summary>
    class DataQueryCollection : GenericList<DataQuery>, IDisposable
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public DataQueryCollection()
        {
        }

        /// <summary>
        /// Release all data queries.
        /// </summary>
        public void Dispose()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.Dispose();
            }
        }

        /// <summary>
        /// Enable all data queries allowing each to actively query data, filling their internal queues.
        /// </summary>
        public void Start()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.EnableQueueThread = true;
            }
        }

        /// <summary>
        /// Stop all data queries from actively querying data.
        /// </summary>
        public void Stop()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.EnableQueueThread = false;
            }
        }

        /// <summary>
        /// Shutdown all data queries, stopping their internal query threads.
        /// </summary>
        public void Shutdown()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.Shutdown();
            }
        }
    }
}
