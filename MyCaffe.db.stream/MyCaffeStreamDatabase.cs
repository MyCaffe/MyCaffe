using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    public partial class MyCaffeStreamDatabase : Component, IXStreamDatabase
    {
        Log m_log;
        MgrQuery m_qryMgr = new MgrQuery();

        public MyCaffeStreamDatabase(Log log)
        {
            m_log = log;
            InitializeComponent();
        }

        public MyCaffeStreamDatabase(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        private void dispose()
        {
            m_qryMgr.Shutdown();
        }

        public void Initialize(int nQueryCount, DateTime dtStart, int nTimeSpanInMs, int nSegmentSize, int nMaxCount, string strSchema)
        {
            m_qryMgr.Initialize(nQueryCount, dtStart, nTimeSpanInMs, nSegmentSize, nMaxCount, strSchema);
        }

        public void Shutdown()
        {
            m_qryMgr.Shutdown();
        }

        public void AddDirectQuery(IXCustomQuery iqry)
        {
            m_qryMgr.AddDirectQuery(iqry);
        }

        public SimpleDatum Query(int nWait)
        {
            return m_qryMgr.Query(nWait);
        }

        public int[] QuerySize()
        {
            List<int> rg = m_qryMgr.GetQuerySize();

            if (rg == null)
                return null;

            return rg.ToArray();
        }

        public void Reset(int nStartOffset = 0)
        {
            m_qryMgr.Reset(nStartOffset);
        }
    }
}
