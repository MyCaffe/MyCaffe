using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    class DataQueryCollection : GenericList<DataQuery>, IDisposable
    {
        public DataQueryCollection()
        {
        }

        public void Dispose()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.Dispose();
            }
        }

        public void Start()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.EnableQueueThread = true;
            }
        }

        public void Stop()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.EnableQueueThread = false;
            }
        }

        public void Shutdown()
        {
            foreach (DataQuery dq in m_rgItems)
            {
                dq.Shutdown();
            }
        }
    }
}
