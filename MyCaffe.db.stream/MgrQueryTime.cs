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
    /// <summary>
    /// The MgrQueryTime class manages the collection of data queries, and the internal data queue that contains all synchronized data items from
    /// the data queries, all fused together.
    /// </summary>
    public class MgrQueryTime : IXQuery
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

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nQueryCount">Specifies the size of each query.</param>
        /// <param name="dtStart">Specifies the state date used for data collection.</param>
        /// <param name="nTimeSpanInMs">Specifies the time increment used between each data item.</param>
        /// <param name="nSegmentSize">Specifies the amount of data to query on the back-end from each custom query.</param>
        /// <param name="nMaxCount">Specifies the maximum number of items to allow in memory.</param>
        /// <param name="strSchema">Specifies the database schema.</param>
        /// <param name="rgCustomQueries">Optionally, specifies any custom queries to add directly.</param>
        /// <remarks>
        /// The database schema defines the number of custom queries to use along with their names.  A simple key=value; list
        /// defines the streaming database schema using the following format:
        /// \code{.cpp}
        ///  "ConnectionCount=2;
        ///   Connection0_CustomQueryName=Test1;
        ///   Connection0_CustomQueryParam=param_string1
        ///   Connection1_CustomQueryName=Test2;
        ///   Connection1_CustomQueryParam=param_string2"
        /// \endcode
        /// Each param_string specifies the parameters of the custom query and may include the database connection string, database
        /// table, and database fields to query.
        /// </remarks>
        public MgrQueryTime(int nQueryCount, DateTime dtStart, int nTimeSpanInMs, int nSegmentSize, int nMaxCount, string strSchema, List<IXCustomQuery> rgCustomQueries)
        {
            m_colCustomQuery.Load();
            m_colData = new DataItemCollection(nQueryCount);
            m_nQueryCount = nQueryCount;
            m_nSegmentSize = nSegmentSize;

            m_schema = new PropertySet(strSchema);

            foreach (IXCustomQuery icustomquery in rgCustomQueries)
            {
                m_colCustomQuery.Add(icustomquery);
            }

            int nConnections = m_schema.GetPropertyAsInt("ConnectionCount");
            for (int i = 0; i < nConnections; i++)
            {
                string strConTag = "Connection" + i.ToString();
                string strCustomQuery = m_schema.GetProperty(strConTag + "_CustomQueryName");
                string strCustomQueryParam = m_schema.GetProperty(strConTag + "_CustomQueryParam");

                IXCustomQuery iqry = m_colCustomQuery.Find(strCustomQuery);
                if (iqry == null)
                    throw new Exception("Could not find the custom query '" + strCustomQuery + "'!");

                if (iqry.QueryType != CUSTOM_QUERY_TYPE.TIME)
                    throw new Exception("The custom query '" + iqry.Name + "' does not support the 'CUSTOM_QUERY_TYPE.TIME'!");

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

        /// <summary>
        /// The consoldiate thread synchronized all data queries using their synchronization field (field #0) to make sure 
        /// that all data items line up.
        /// </summary>
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

        /// <summary>
        /// Add a custom query directly to the streaming database.
        /// </summary>
        /// <remarks>
        /// By default, the streaming database looks in the \code{.cpp}'./CustomQuery'\endcode folder relative
        /// to the streaming database assembly to look for CustomQuery DLL's that implement
        /// the IXCustomQuery interface.  When found, these assemblies are added to the list
        /// accessible via the schema.  Alternatively, custom queries may be added directly
        /// using this method.
        /// </remarks>
        /// <param name="iqry">Specifies the custom query to add.</param>
        public void AddDirectQuery(IXCustomQuery iqry)
        {
            m_colCustomQuery.Add(iqry);
        }

        /// <summary>
        /// Reset the query to the start date used in Initialize, optionally with an offset from the start.
        /// </summary>
        /// <param name="nStartOffset">Optionally, specifies the offset from the start to use (default = 0).</param>
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

        /// <summary>
        /// Shutdown the data queries and consolidation thread.
        /// </summary>
        public void Shutdown()
        {
            m_evtCancel.Set();
            m_colDataQuery.Shutdown();
            m_colData.Cancel.Set();
        }

        /// <summary>
        /// Returns the query size of the data in the form:
        /// [0] = channels
        /// [1] = height
        /// [2] = width.
        /// </summary>
        /// <returns>The query size is returned.</returns>
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

        /// <summary>
        /// Query the next data in the streaming database.
        /// </summary>
        /// <param name="nWait">Specfies the maximum amount of time (in ms.) to wait for data.</param>
        /// <returns>A simple datum containing the data is returned.</returns>
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
