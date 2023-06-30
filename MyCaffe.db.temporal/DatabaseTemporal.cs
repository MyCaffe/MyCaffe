using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using SimpleGraphing;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms.VisualStyles;
using static MyCaffe.basecode.descriptors.ValueStreamDescriptor;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The DatabaseTemporal is used to manage all temporal specific database objects.
    /// </summary>
    public class DatabaseTemporal : Database
    {
        ConnectInfo m_ci;
        DNNEntitiesTemporal m_entitiesTemporal = null;
        Dictionary<int, string> m_rgstrValueItems = new Dictionary<int, string>();
        Dictionary<int, string> m_rgstrValueItemsByIndex = new Dictionary<int, string>();
        DataTable m_rgRawValueCache = new DataTable();
        SqlBulkCopy m_sqlBulkCopy = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public DatabaseTemporal()
        {
            m_rgRawValueCache.Columns.Add("SourceID", typeof(int));
            m_rgRawValueCache.Columns.Add("ItemID", typeof(int));
            m_rgRawValueCache.Columns.Add("TimeStamp", typeof(DateTime));
            m_rgRawValueCache.Columns.Add("RawData", typeof(byte[]));
            m_rgRawValueCache.Columns.Add("Active", typeof(bool));
        }

        /// <summary>
        /// Opens a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source to open.</param>
        /// <param name="nForceLoad">Optionally, specifies how to force load the data (default = NONE).</param>
        /// <param name="ci">Optionally, specifies a specific connection to use (default = null).</param>
        public override void Open(int nSrcId, FORCE_LOAD nForceLoad = FORCE_LOAD.NONE, ConnectInfo ci = null)
        {
            base.Open(nSrcId, nForceLoad, ci);
            m_ci = ci;
            m_entitiesTemporal = EntitiesConnectionTemporal.CreateEntities(ci);
        }

        /// <summary>
        /// Enable bulk inserts.
        /// </summary>
        /// <param name="bEnable">Enables the bulk mode.</param>
        public void EnableBulk(bool bEnable)
        {
            if (m_entitiesTemporal == null)
                return;

            m_sqlBulkCopy = new SqlBulkCopy(m_entitiesTemporal.Database.Connection.ConnectionString, SqlBulkCopyOptions.TableLock);
            m_sqlBulkCopy.DestinationTableName = "dbo.RawValues";
            m_sqlBulkCopy.BulkCopyTimeout = 600;
            m_sqlBulkCopy.ColumnMappings.Add("SourceID", "SourceID");
            m_sqlBulkCopy.ColumnMappings.Add("ItemID", "ItemID");
            m_sqlBulkCopy.ColumnMappings.Add("TimeStamp", "TimeStamp");
            m_sqlBulkCopy.ColumnMappings.Add("RawData", "RawData");
            m_sqlBulkCopy.ColumnMappings.Add("Active", "Active");

            m_entitiesTemporal.Configuration.AutoDetectChangesEnabled = false;
            m_entitiesTemporal.Configuration.ValidateOnSaveEnabled = false;
        }

        /// <summary>
        /// Close the current data source.
        /// </summary>
        public override void Close()
        {
            if (m_sqlBulkCopy != null)
            {
                m_sqlBulkCopy.Close();
                m_sqlBulkCopy = null;
            }

            if (m_entitiesTemporal != null)
            {
                m_entitiesTemporal.Dispose();
                m_entitiesTemporal = null;
            }

            m_ci = null;

            base.Close();
        }

        /// <summary>
        /// Delete a data source from the database.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>Returns true if the source was found.</returns>
        public override bool DeleteSource(int nSrcId = 0)
        {
            string strCmd;

            if (!base.DeleteSource(nSrcId))
                return false;

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                strCmd = "IF OBJECT_ID (N'dbo.ValueItems', N'U') IS NOT NULL DELETE FROM ValueItems WHERE (SourceID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);

                strCmd = "IF OBJECT_ID (N'dbo.RawValues', N'U') IS NOT NULL DELETE FROM RawValues WHERE (SourceID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);
            }

            DeleteSourceData(nSrcId);

            return true;
        }

        /// <summary>
        /// Delete the data source data (images, means, results and parameters) from the database.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        /// <returns>Returns true if the source was found.</returns>
        public override bool DeleteSourceData(int nSrcId = 0)
        {
            if (!base.DeleteSourceData(nSrcId))
                return false;

            DeleteValueItems(nSrcId);
            DeleteRawValues(nSrcId);

            return true;
        }

        /// <summary>
        /// Delete a dataset.
        /// </summary>
        /// <param name="strDsName">Specifies the dataset name.</param>
        /// <param name="bDeleteRelatedProjects">Specifies whether or not to also delete all projects using the dataset.  <b>WARNING!</b> Use this with caution for it will permenantly delete the projects and their results.</param>
        /// <param name="log">Specifies the Log object for status output.</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the delete.</param>
        public override void DeleteDataset(string strDsName, bool bDeleteRelatedProjects, Log log, CancelEvent evtCancel)
        {
            Dataset ds = GetDataset(strDsName);
            if (ds == null)
                return;

            Source srcTraining = GetSource(ds.TrainingSourceID.GetValueOrDefault());
            Source srcTesting = GetSource(ds.TestingSourceID.GetValueOrDefault());

            DeleteDatasetTemporalTables(srcTraining.ID, srcTesting.ID, log, evtCancel);

            base.DeleteDataset(strDsName, bDeleteRelatedProjects, log, evtCancel);
        }

        /// <summary>
        /// Delete a dataset temporal tables..
        /// </summary>
        /// <param name="nSrcIDTrain">Specifies the train source ID.</param>
        /// <param name="nSrcIDTest">Specifies the test source ID.</param>
        /// <param name="log">Specifies the Log object for status output.</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the delete.</param>
        public static void DeleteDatasetTemporalTables(int nSrcIDTrain, int nSrcIDTest, Log log, CancelEvent evtCancel)
        {
            string strCmd;

            if (!DeleteRawValuesEx(nSrcIDTest, log, evtCancel))
                return;

            if (!DeleteRawValuesEx(nSrcIDTrain, log, evtCancel))
                return;

            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                entities.Database.CommandTimeout = 180;

                strCmd = "DELETE ValueItems WHERE (SourceID = " + nSrcIDTrain.ToString() + ") OR (SourceID = " + nSrcIDTest.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Delete all RawValues in a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteRawValues(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "IF OBJECT_ID (N'dbo.RawValues', N'U') IS NOT NULL DELETE FROM RawValues WHERE (SourceID = " + nSrcId.ToString() + ")";

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }

        /// <summary>
        /// Delete all RawValues in a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event used to abort the operation.</param>
        public static bool DeleteRawValuesEx(int nSrcId, Log log, CancelEvent evtCancel)
        {
            string strCmd = "IF OBJECT_ID (N'dbo.RawValues', N'U') IS NOT NULL SELECT COUNT(ID) FROM RawValues WHERE (SourceID = " + nSrcId.ToString() + ") ELSE SELECT 0";
            Stopwatch sw = new Stopwatch();

            sw.Start();

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                entities.Database.CommandTimeout = 180;
                int lCount = entities.Database.SqlQuery<int>(strCmd).FirstOrDefault();
                long lBlock = 10000;
                long lStep = lCount / lBlock;
                long lIdx = 0;

                while (lIdx < lCount)
                {
                    strCmd = "DELETE TOP (" + lBlock.ToString() + ") RawValues WHERE (SourceID = " + nSrcId.ToString() + ")";
                    entities.Database.ExecuteSqlCommand(strCmd);

                    lIdx += lBlock;

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        if (evtCancel.WaitOne(0))
                            return false;

                        double dfPct = (double)lIdx / (double)lCount;
                        log.WriteLine("Deleting RawValues at " + dfPct.ToString("P4"));
                    }
                }

                strCmd = "DELETE RawValues WHERE (SourceID = " + nSrcId.ToString() + ")";
                entities.Database.ExecuteSqlCommand(strCmd);
            }

            return true;
        }

        /// <summary>
        /// Delete all ValueItems in a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteValueItems(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "IF OBJECT_ID (N'dbo.ValueItems', N'U') IS NOT NULL DELETE FROM ValueItems WHERE (SourceID = " + nSrcId.ToString() + ")";

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                entities.Database.ExecuteSqlCommand(strCmd);
            }
        }


        //---------------------------------------------------------------------
        //  Raw Values
        //---------------------------------------------------------------------
        #region Raw Values

        /// <summary>
        /// Add static raw values for a data stream.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="data">Specifies the raw data values.</param>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public void PutRawValue(int nSrcID, int nItemID, RawValueDataCollection data)
        {
            DataRow dr = m_rgRawValueCache.NewRow();

            dr["SourceID"] = nSrcID;
            dr["ItemID"] = nItemID;
            if (data.TimeStamp.HasValue)
                dr["TimeStamp"] = data.TimeStamp.Value;
            dr["RawData"] = data.ToBytes();
            dr["Active"] = true;

            m_rgRawValueCache.Rows.Add(dr);
        }

        /// <summary>
        /// Add raw values for a data stream.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="dt">Specifies the date/time associated with the value.</param>
        /// <param name="data">Specifies the raw data values.</param>
        /// <param name="bActive">Specifies the active state of the record.</param>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public void PutRawValue(int nSrcID, int nItemID, DateTime dt, RawValueDataCollection data, bool bActive)
        {
            DataRow dr = m_rgRawValueCache.NewRow();

            dr["SourceID"] = nSrcID;
            dr["ItemID"] = nItemID;
            dr["TimeStamp"] = dt;
            dr["RawData"] = data.ToBytes();
            dr["Active"] = bActive;

            m_rgRawValueCache.Rows.Add(dr);
        }

        /// <summary>
        /// Save the raw values.
        /// </summary>
        public void SaveRawValues()
        {
            if (m_rgRawValueCache.Rows.Count > 0)
            {
                try
                {
                    m_sqlBulkCopy.WriteToServer(m_rgRawValueCache);
                }
                catch (Exception excpt)
                {
                    throw excpt;
                }
                finally
                {
                    ClearRawValues();
                }
            }
        }

        /// <summary>
        /// Clear the raw values.   
        /// </summary>
        public void ClearRawValues()
        {
            m_rgRawValueCache.Clear();
        }

        /// <summary>
        /// Load the static value stream categorical values for a given source and item.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <returns>A list of the static raw values is returned.</returns>
        public RawValueSet GetValues(int nSrcID, int nItemID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<RawValue> rgData = entities.RawValues.AsNoTracking().Where(p => p.SourceID == nSrcID && p.ItemID == nItemID).OrderBy(p => p.TimeStamp).ToList();
                return RawValueSet.FromData(nSrcID, nItemID, rgData);
            }
        }

        #endregion


        //---------------------------------------------------------------------
        //  Value Items
        //---------------------------------------------------------------------
        #region Value Items

        /// <summary>
        /// Add a new value item to the database.
        /// </summary>
        /// <param name="nSrcId">Specifies the source ID of the value item.</param>
        /// <param name="nItemIdx">Specifies the index of the item.</param>
        /// <param name="strName">Specifies the name of the value item.</param>
        /// <returns>The value item ID is returned.</returns>
        public int AddValueItem(int nSrcId, int nItemIdx, string strName)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueItem> rgItems = entities.ValueItems.Where(p=>p.SourceID == nSrcId && p.Name == strName).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].ID;

                ValueItem item = new ValueItem();
                item.Name = strName;
                item.Idx = nItemIdx;
                item.SourceID = nSrcId;
                entities.ValueItems.Add(item);
                entities.SaveChanges();

                return item.ID;
            }
        }

        /// <summary>
        /// Returns the value item ID given the value item name if found, or 0.
        /// </summary>
        /// <param name="strName">Specifies the value item to find.</param>
        /// <returns>The value item ID is returned or 0 if not found.</returns>
        public int GetValueItemID(string strName)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueItem> rgItems = entities.ValueItems.Where(p => p.Name == strName).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].ID;

                return 0;
            }
        }

        /// <summary>
        /// Returns the value item ID given the value item name if found, or 0.
        /// </summary>
        /// <param name="nID">Specifies the value item ID to find.</param>
        /// <returns>The value item name is returned or null if not found.</returns>
        public string GetValueItemName(int nID)
        {
            if (m_rgstrValueItems.ContainsKey(nID))
                return m_rgstrValueItems[nID];  

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueItem> rgItems = entities.ValueItems.Where(p => p.ID == nID).ToList();

                if (rgItems.Count > 0)
                {
                    m_rgstrValueItems.Add(nID, rgItems[0].Name);
                    m_rgstrValueItemsByIndex.Add(rgItems[0].Idx.GetValueOrDefault(0), rgItems[0].Name);
                    return rgItems[0].Name;
                }

                return null;
            }
        }

        /// <summary>
        /// Returns a list of all value item IDs associated with a SourceID.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <returns>The list of item ID's is returned.</returns>
        public List<int> GetAllItemIDs(int nSrcID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                return entities.ValueItems.Where(p => p.SourceID == nSrcID).Select(p => p.ID).ToList();
            }
        }

        /// <summary>
        /// Returns the value item ID given the value item name if found, or 0.
        /// </summary>
        /// <param name="strName">Specifies the value item to find.</param>
        /// <returns>The value item ID is returned or 0 if not found.</returns>
        public int GetValueItemIndex(string strName)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueItem> rgItems = entities.ValueItems.Where(p => p.Name == strName).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].Idx.GetValueOrDefault(0);

                return 0;
            }
        }

        /// <summary>
        /// Returns the value item name given the value item index if found, or 0.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the value item to find.</param>
        /// <returns>The value item name is returned or null if not found.</returns>
        public string GetValueItemNamByIndex(int nIdx)
        {
            if (m_rgstrValueItemsByIndex.ContainsKey(nIdx))
                return m_rgstrValueItemsByIndex[nIdx];

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueItem> rgItems = entities.ValueItems.Where(p => p.Idx == nIdx).ToList();

                if (rgItems.Count > 0)
                {
                    m_rgstrValueItemsByIndex.Add(nIdx, rgItems[0].Name);
                    m_rgstrValueItems.Add(rgItems[0].ID, rgItems[0].Name);
                    return rgItems[0].Name;
                }

                return null;
            }
        }

        /// <summary>
        /// Returns a list of all value item Indices associated with a SourceID.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <returns>The list of item Inidices is returned.</returns>
        public List<int> GetAllItemIndices(int nSrcID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                return entities.ValueItems.Where(p => p.SourceID == nSrcID).Select(p => p.Idx.GetValueOrDefault(0)).ToList();
            }
        }

        /// <summary>
        /// Returns a list of all value items associated with a SourceID.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <returns>The list of all value items is returned.</returns>
        public List<ValueItem> GetAllValueItems(int nSrcID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                return entities.ValueItems.Where(p => p.SourceID == nSrcID).ToList();
            }
        }

        #endregion

        #region Value Streams

        /// <summary>
        /// Add a new value stream to the source ID.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="strName">Specifies the value stream name.</param>
        /// <param name="nOrdering">Specifies the value stream ordering.</param>
        /// <param name="classType">Specifies the value stream class type (STATIC, OBSERVED or KNOWN)</param>
        /// <param name="valType">Specifies the value stream type (NUMERIC or CATEGORICAL).</param>
        /// <param name="dtStart">Optionally, specifies the value stream start date (null for STATIC).</param>
        /// <param name="dtEnd">Optionally, specifies the value stream end date (null for STATIC).</param>
        /// <param name="nSecPerStep">Optionally, specifies the value stream seconds per step (null for STATIC).</param>
        /// <param name="nTotalSteps">Optionally, specifies the total number of steps (1 for STATIC).</param>
        /// <returns>The ID of the value stream added is returned.</returns>
        public int AddValueStream(int nSrcID, string strName, int nOrdering, STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valType, DateTime? dtStart = null, DateTime? dtEnd = null, int? nSecPerStep = null, int nTotalSteps = 1)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                ValueStream vs = null;

                List<ValueStream> rg = entities.ValueStreams.Where(p => p.SourceID == nSrcID && p.Name == strName).ToList();
                if (rg.Count == 0)
                    vs = new ValueStream();
                else
                    vs = rg[0];

                vs.SourceID = nSrcID;
                vs.Name = strName;
                vs.Ordering = (short)nOrdering;
                vs.ClassTypeID = (byte)classType;
                vs.ValueTypeID = (byte)valType;
                vs.StartTime = dtStart;
                vs.EndTime = dtEnd;
                vs.SecondsPerStep = nSecPerStep;
                vs.TotalSteps = nTotalSteps;

                if (rg.Count == 0)
                    entities.ValueStreams.Add(vs);

                entities.SaveChanges();

                return vs.ID;
            }
        }

        /// <summary>
        /// Returns a list of all value streams associated with a SourceID.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <returns>The list of all value streams is returned.</returns>
        public List<ValueStream> GetAllValueStreams(int nSrcID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                return entities.ValueStreams.Where(p => p.SourceID == nSrcID).ToList();
            }
        }

        #endregion
    }

    /// <summary>
    /// The RawValueData class is used to hold the data values for an ItemID.
    /// </summary>
    public class RawValueSet
    {
        int m_nSourceID;
        int m_nItemID;
        RawValueDataCollection m_staticValues = null;
        List<RawValueDataCollection> m_rgValues = new List<RawValueDataCollection>();
        bool m_bSorted = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        public RawValueSet(int nSrcID, int nItemID)
        {
            m_nSourceID = nSrcID;
            m_nItemID = nItemID;
        }

        /// <summary>
        /// Creates a RawValueSet from a list of RawValues.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="rg">Specifies the raw values.</param>
        /// <returns></returns>
        public static RawValueSet FromData(int nSrcID, int nItemID, List<RawValue> rg)
        {
            RawValueSet set = new RawValueSet(nSrcID, nItemID);

            foreach (RawValue val in rg)
            {
                set.Add(val);
            }

            return set;
        }

        /// <summary>
        /// Add a raw value to the set.
        /// </summary>
        /// <param name="val">Specifies the raw value loaded from the dataset.</param>
        /// <exception cref="Exception">An exception is thrown if more than one </exception>
        public void Add(RawValue val)
        {
            if (!val.TimeStamp.HasValue)
            {
                if (m_staticValues != null)
                    throw new Exception("There should only be one static value set for an item!");

                m_staticValues = RawValueDataCollection.LoadFromBytes(null, val.RawData);
            }
            else
            {
                m_rgValues.Add(RawValueDataCollection.LoadFromBytes(val.TimeStamp, val.RawData));
            }
        }

        /// <summary>
        /// Returns the row count of observed and known items.
        /// </summary>
        public int ColCount
        {
            get
            {
                if (m_rgValues.Count == 0)
                    return 0;

                return m_rgValues.Count;
            }
        }

        /// <summary>
        /// Returns the end time of the data values.
        /// </summary>
        public DateTime EndTime
        {
            get
            {
                if (m_rgValues.Count == 0)
                    return DateTime.MinValue;

                return m_rgValues[m_rgValues.Count - 1].TimeStamp.Value;
            }
        }

        /// <summary>
        /// Returns the source ID.
        /// </summary>
        public int SourceID
        {
            get { return m_nSourceID; }
        }

        /// <summary>
        /// Returns the item ID.
        /// </summary>
        public int ItemID
        {
            get { return m_nItemID; }
        }

        /// <summary>
        /// Return the static values.
        /// </summary>
        /// <returns>The a tuple of the numeric, categorical static values is returned.</returns>
        public Tuple<float[], float[]> GetStaticValues()
        {
            List<float> rgNum = new List<float>();
            List<float> rgCat = new List<float>();

            foreach (RawValueData data in m_staticValues)
            {
                if (data.ValueType == STREAM_VALUE_TYPE.NUMERIC)
                    rgNum.Add(data.Value);
                else
                    rgCat.Add(data.Value);
            }

            return new Tuple<float[], float[]>(rgNum.ToArray(), rgCat.ToArray());
        }

        /// <summary>
        /// Returns the observed values.
        /// </summary>
        /// <param name="nIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of items to collect.</param>
        /// <returns>A tuple of the observed numeric and categorical values is returned, or null if not enough items exists from 'nIdx'.</returns>
        public Tuple<float[], float[]> GetObservedValues(int nIdx, int nCount)
        {
            if (nIdx + nCount >= m_rgValues.Count)
                return null;

            List<float> rgNum = new List<float>();
            List<float> rgCat = new List<float>();

            if (!m_bSorted)
            {
                m_rgValues = m_rgValues.OrderBy(p => p.TimeStamp).ToList();
                m_bSorted = true;
            }
 
            for (int i=nIdx; i<nIdx + nCount; i++)
            {
                foreach (RawValueData data in m_rgValues[i])
                {
                    if (data.ClassType == STREAM_CLASS_TYPE.OBSERVED)
                    {
                        if (data.ValueType == STREAM_VALUE_TYPE.NUMERIC)
                            rgNum.Add(data.ValueNormalized.GetValueOrDefault(data.Value));
                        else
                            rgCat.Add(data.Value);
                    }
                }
            }

            return new Tuple<float[], float[]>(rgNum.ToArray(), rgCat.ToArray());
        }

        /// <summary>
        /// Returns the observed values.
        /// </summary>
        /// <param name="nIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of items to collect.</param>
        /// <param name="nValIdx">Specifies the index of the target value.</param>
        /// <returns>A tuple of the observed numeric and categorical values is returned, or null if not enough items exists from 'nIdx'.</returns>
        public float[] GetObservedNumValues(int nIdx, int nCount, int nValIdx)
        {
            if (nIdx + nCount >= m_rgValues.Count)
                return null;

            List<float> rgNum = new List<float>();

            if (!m_bSorted)
            {
                m_rgValues = m_rgValues.OrderBy(p => p.TimeStamp).ToList();
                m_bSorted = true;
            }

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                RawValueDataCollection col = m_rgValues[i];
                RawValueData val = col[nValIdx];

                if (val.ClassType != STREAM_CLASS_TYPE.OBSERVED && val.ValueType != STREAM_VALUE_TYPE.NUMERIC)
                    throw new Exception("The value at index " + nValIdx.ToString() + " is not a numeric observed value!");

                rgNum.Add(val.ValueNormalized.GetValueOrDefault(val.Value));
            }

            return rgNum.ToArray();
        }

        /// <summary>
        /// Returns the observed values.
        /// </summary>
        /// <param name="nIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of items to collect.</param>
        /// <returns>A tuple of the known numeric and categorical values is returned, or null if not enough items exists from 'nIdx'.</returns>
        public Tuple<float[], float[]> GetKnownValues(int nIdx, int nCount)
        {
            if (nIdx + nCount >= m_rgValues.Count)
                return null;

            List<float> rgNum = new List<float>();
            List<float> rgCat = new List<float>();

            if (!m_bSorted)
            {
                m_rgValues = m_rgValues.OrderBy(p => p.TimeStamp).ToList();
                m_bSorted = true;
            }

            if (nIdx + nCount >= m_rgValues.Count)
                return null;

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                foreach (RawValueData data in m_rgValues[i])
                {
                    if (data.ClassType == STREAM_CLASS_TYPE.KNOWN)
                    {
                        if (data.ValueType == STREAM_VALUE_TYPE.NUMERIC)
                            rgNum.Add(data.ValueNormalized.GetValueOrDefault(data.Value));
                        else
                            rgCat.Add(data.Value);
                    }
                }
            }

            return new Tuple<float[], float[]>(rgNum.ToArray(), rgCat.ToArray());
        }

        /// <summary>
        /// Returns the time sync values.
        /// </summary>
        /// <param name="nIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of items to collect.</param>
        /// <returns>An array of the time values is returned.</returns>
        public DateTime[] GetTimeSyncValues(int nIdx, int nCount)
        {
            List<DateTime> rgTime = new List<DateTime>();

            for (int i = nIdx; i < nIdx + nCount; i++)
            {
                RawValueDataCollection col = m_rgValues[i];
                if (col.TimeStamp.HasValue)
                    rgTime.Add(col.TimeStamp.Value);
            }

            return rgTime.ToArray();
        }
    }

    /// <summary>
    /// The RawValueDataCollection class is used to hold a collection of RawValueData items.
    /// </summary>
    public class RawValueDataCollection : IEnumerable<RawValueData>
    {
        DateTime? m_dt;
        List<RawValueData> m_rgData = new List<RawValueData>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public RawValueDataCollection(DateTime? dt)
        {
            m_dt = dt;
        }

        /// <summary>
        /// Returns the time stamp of the raw value data items.
        /// </summary>
        public DateTime? TimeStamp
        {
            get { return m_dt; }
        }

        /// <summary>
        /// Returns the number of raw value items.
        /// </summary>
        public int Count
        {
            get { return m_rgData.Count; }
        }

        /// <summary>
        /// Returns a raw value item at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The raw value item is returned.</returns>
        public RawValueData this[int nIdx]
        {
            get { return m_rgData[nIdx]; }
        }

        /// <summary>
        /// Manually add a new raw value data item.
        /// </summary>
        /// <param name="data">Specifies the data item to add.</param>
        public void Add(RawValueData data)
        {
            m_rgData.Add(data);
        }

        /// <summary>
        /// Set the data values in the collection with the data values provided.
        /// </summary>
        /// <param name="rgData">Specifies the data values.</param>
        /// <exception cref="Exception">An exception is thrown if the array of data values is not the same length as the list of data values.</exception>
        public void SetData(float[] rgData)
        {
            if (m_rgData.Count != rgData.Length)
                throw new Exception("The number of data items must match the number of raw value data items!");

            for (int i = 0; i < rgData.Length; i++)
            {
                m_rgData[i].Value = rgData[i];
            }
        }

        /// <summary>
        /// Set the data values in the collection with the data values provided.
        /// </summary>
        /// <param name="dt">Specifies the time stamp for the data values.</param>
        /// <param name="rgData">Specifies the data values.</param>
        /// <exception cref="Exception">An exception is thrown if the array of data values is not the same length as the list of data values.</exception>
        public void SetData(DateTime dt, float[] rgData)
        {
            m_dt = dt;
            SetData(rgData);
        }

        /// <summary>
        /// Converts the raw value data items to a byte aray.
        /// </summary>
        /// <returns>The serialized byte array is returned.</returns>
        public byte[] ToBytes()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write(m_rgData.Count);

                for (int i = 0; i < m_rgData.Count; i++)
                {
                    m_rgData[i].Save(bw);
                }

                ms.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Loads a raw value data collection from a byte array.
        /// </summary>
        /// <param name="rg">Specifies the serialized byte array.</param>
        /// <returns>The raw data value collection is returned.</returns>
        public static RawValueDataCollection LoadFromBytes(DateTime? dt, byte[] rg)
        {
            RawValueDataCollection col = new RawValueDataCollection(dt);

            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                int nCount = br.ReadInt32();

                for (int i = 0; i < nCount; i++)
                {
                    col.m_rgData.Add(RawValueData.Load(br));
                }
            }

            return col;
        }

        /// <summary>
        /// Returns an enumerator for the raw value data items.
        /// </summary>
        /// <returns>Returns the enumerator.</returns>
        public IEnumerator<RawValueData> GetEnumerator()
        {
            return ((IEnumerable<RawValueData>)m_rgData).GetEnumerator();
        }

        /// <summary>
        /// Returns an enumerator for the raw value data items.
        /// </summary>
        /// <returns>Returns the enumerator.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)m_rgData).GetEnumerator();
        }
    }

    /// <summary>
    /// The RawValueData class contains a single raw value item.
    /// </summary>
    public class RawValueData
    {
        int m_nStrmID = 0;
        STREAM_CLASS_TYPE m_classType = STREAM_CLASS_TYPE.STATIC;
        STREAM_VALUE_TYPE m_valueType = STREAM_VALUE_TYPE.NUMERIC;
        float m_fVal;
        float? m_fValNorm;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="classType">Specifies the raw value item class.</param>
        /// <param name="valueType">Specifies the raw value item value type.</param>
        /// <param name="fVal">Specifies the raw value item data.</param>
        /// <param name="fValNorm">Optionally, specifies the raw value item normalized data.</param>
        public RawValueData(STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valueType, float fVal, float? fValNorm = null)
        {
            m_classType = classType;
            m_valueType = valueType;
            m_fVal = fVal;
            m_fValNorm = fValNorm;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="classType">Specifies the raw value item class.</param>
        /// <param name="valueType">Specifies the raw value item value type.</param>
        /// <param name="nStrmId">Specifies the stream ID.</param>
        public RawValueData(STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valueType, int nStrmId)
        {
            m_nStrmID = nStrmId;
            m_classType = classType;
            m_valueType = valueType;
        }


        /// <summary>
        /// Save the raw value item to a binary writer.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write((byte)m_classType);
            bw.Write((byte)m_valueType);
            bw.Write(m_fVal);

            bw.Write(m_fValNorm.HasValue);
            if (m_fValNorm.HasValue)
                bw.Write(m_fValNorm.Value);
        }

        /// <summary>
        /// Load a raw value item from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The new RawValueData is returned.</returns>
        public static RawValueData Load(BinaryReader br)
        {
            STREAM_CLASS_TYPE classType = (STREAM_CLASS_TYPE)br.ReadByte();
            STREAM_VALUE_TYPE valueType = (STREAM_VALUE_TYPE)br.ReadByte();
            float fVal = br.ReadSingle();

            float? fValNorm = null;
            if (br.ReadBoolean())
                fValNorm = br.ReadSingle();

            return new RawValueData(classType, valueType, fVal, fValNorm);
        }

        /// <summary>
        /// Returns the raw value item class.
        /// </summary>
        public STREAM_CLASS_TYPE ClassType
        {
            get { return m_classType; }
        }

        /// <summary>
        /// Returns the raw value item value type.
        /// </summary>
        public STREAM_VALUE_TYPE ValueType
        {
            get { return m_valueType; }
        }

        /// <summary>
        /// Returns the raw value item data.
        /// </summary>
        public float Value
        {
            get { return m_fVal; }
            set { m_fVal = value; }
        }

        /// <summary>
        /// Returns the raw value item normalized data if it exists.
        /// </summary>
        public float? ValueNormalized
        {
            get { return m_fValNorm; }
            set { m_fValNorm = value; }
        }
    }
}
