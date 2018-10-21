using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    public class MgrQuery
    {
        CustomQueryCollection m_colCustomQuery = new CustomQueryCollection();
        DataQueryCollection m_colDataQuery = new DataQueryCollection();
        DataItemCollection m_colData;
        PropertySet m_schema;
        CancelEvent m_evtCancel = new CancelEvent();
        ManualResetEvent m_evtEnabled = new ManualResetEvent(false);
        ManualResetEvent m_evtPaused = new ManualResetEvent(false);
        Task m_taskConsolidate;
        int m_nQueryCount = 0;
        int m_nSegmentSize;
        int m_nFieldCount = 0;

        public MgrQuery()
        {
        }

        public void Initialize(int nQueryCount, DateTime dtStart, int nTimeSpanInMs, int nSegmentSize, int nMaxCount, string strSchema)
        {
            m_colCustomQuery.Load();
            m_colData = new DataItemCollection(nQueryCount);
            m_nQueryCount = nQueryCount;
            m_nSegmentSize = nSegmentSize;

            m_schema = new PropertySet(strSchema);

            int nConnections = m_schema.GetPropertyAsInt("ConnectionCount");
            for (int i = 0; i < nConnections; i++)
            {
                string strConTag = "Connection" + i.ToString();
                string strCustomQuery = m_schema.GetProperty(strConTag + "_CustomQueryName");
                string strCustomQueryParam = m_schema.GetProperty(strConTag + "_CustomQueryParam");

                IXCustomQuery iqry = m_colCustomQuery.Find(strCustomQuery);
                if (iqry == null)
                    throw new Exception("Could not find the custom query '" + strCustomQuery + "'!");

                DataQuery dq = new DataQuery(iqry.Clone(strCustomQueryParam), dtStart, TimeSpan.FromMilliseconds(nTimeSpanInMs), nSegmentSize, nMaxCount);
                m_colDataQuery.Add(dq);

                m_nFieldCount += (dq.FieldCount - 1);  // subtract each sync field.
            }

            m_nFieldCount += 1; // add the sync field
            m_colDataQuery.Start();

            m_evtCancel.Reset();
            m_taskConsolidate = Task.Factory.StartNew(new Action(consolidateThread));
            m_evtEnabled.Set();
            m_colData.WaitData(10000);
        }

        private void consolidateThread()
        {
            int nAllDataReady = ((int)Math.Pow(2, m_colDataQuery.Count)) - 1;
            int nWait = 0;

            while (!m_evtCancel.WaitOne(nWait))
            {
                if (!m_evtEnabled.WaitOne(0))
                {
                    nWait = 250;
                    m_evtPaused.Set();
                    continue;
                }

                nWait = 0;
                m_evtPaused.Reset();

                int nDataReady = 0;
                int nDataDone = 0;

                for (int i = 0; i < m_colDataQuery.Count; i++)
                {
                    DataQuery dq = m_colDataQuery[i];
                    if (dq.DataReady(1))
                        nDataReady |= (0x0001 << i);

                    if (dq.DataDone())
                        nDataDone |= (0x0001 << i);
                }

                if (nDataDone != 0)
                    m_colData.QueryEnd.Set();

                if (nDataReady != nAllDataReady)
                    continue;

                DataItem di = new DataItem(m_nFieldCount);
                int nLocalFieldCount = m_colDataQuery[0].FieldCount;
                double[] rg = m_colDataQuery[0].GetNextData();

                if (rg == null)
                    continue;

                DateTime dtSync = Utility.ConvertTimeFromMinutes(rg[0]);
                bool bSkip = false;

                for (int i = 0; i < m_nSegmentSize; i++)
                {
                    int nFieldIdx = di.Add(0, i, rg, nLocalFieldCount);

                    for (int j = 1; j < m_colDataQuery.Count; j++)
                    {
                        double[] rg1 = m_colDataQuery[j].GetNextData();
                        if (rg1 == null)
                        {
                            bSkip = true;
                            break;
                        }

                        DateTime dtSync1 = Utility.ConvertTimeFromMinutes(rg1[0]);

                        while (dtSync1 < dtSync)
                        {
                            rg1 = m_colDataQuery[j].GetNextData();
                            if (rg1 == null)
                            {
                                bSkip = true;
                                break;
                            }

                            dtSync1 = Utility.ConvertTimeFromMinutes(rg1[0]);
                        }

                        if (bSkip)
                            break;

                        nLocalFieldCount = m_colDataQuery[j].FieldCount;
                        nFieldIdx = di.Add(nFieldIdx, i, rg1, nLocalFieldCount);
                    }

                    if (bSkip)
                        break;
                }

                if (!bSkip)
                    m_colData.Add(di);
            }
        }

        public void AddDirectQuery(IXCustomQuery iqry)
        {
            m_colCustomQuery.Add(iqry);
        }

        public void Reset(int nStartOffset)
        {
            m_evtEnabled.Reset();
            m_evtPaused.WaitOne();

            foreach (DataQuery dq in m_colDataQuery)
            {
                dq.Reset(nStartOffset);
            }

            m_colData.Clear();
            m_evtEnabled.Set();
            m_colData.WaitData(10000);
        }

        public void Shutdown()
        {
            m_evtCancel.Set();
            m_colDataQuery.Shutdown();
            m_colData.Cancel.Set();
        }

        public List<int> GetQuerySize()
        {
            List<int> rg = new List<int>();

            if (m_colDataQuery.Count == 0)
                return rg;

            rg.Add(1);
            rg.Add(m_nFieldCount);
            rg.Add(m_nQueryCount);

            return rg;
        }

        public SimpleDatum Query(int nWait)
        {
            if (!m_colData.WaitForCount(nWait))
                return null;

            int nCount = m_nQueryCount * m_nFieldCount;
            Valuemap vals = new Valuemap(1, m_nFieldCount, m_nQueryCount);
            double[] rgLast = null;

            for (int i = 0; i < m_nQueryCount; i++)
            {
                DataItem di = m_colData.GetData(nWait);
                if (di == null)
                    throw new Exception("The data item should not be null!");

                double[] rgItem = di.GetData();

                for (int j = 0; j < rgItem.Length; j++)
                {
                    vals.SetPixel(i, j, rgItem[j]);
                }

                rgLast = rgItem;
            }

            SimpleDatum sd = new SimpleDatum(vals);
            sd.TimeStamp = Utility.ConvertTimeFromMinutes(rgLast[0]);

            return sd;
        }
    }
}
