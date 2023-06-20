using MyCaffe.basecode;
using MyCaffe.db.image;
using SimpleGraphing;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        Dictionary<int, string> m_rgstrValueStreams = new Dictionary<int, string>();

        /// <summary>
        /// Defines the stream value type.
        /// </summary>
        public enum STREAM_VALUE_TYPE
        {
            /// <summary>
            /// Specifies that the value stream hold numeric data.
            /// </summary>
            NUMERIC = 0x01,
            /// <summary>
            /// Specifies that the value stream holds categorical data.
            /// </summary>
            CATEGORICAL = 0x02
        }

        /// <summary>
        /// Defines the stream class type.
        /// </summary>
        public enum STREAM_CLASS_TYPE
        {
            /// <summary>
            /// Specifies static values that are not time related.  The DateTime in each static value is set to NULL.
            /// </summary>
            STATIC = 0x01,
            /// <summary>
            /// Specifies raw values that are only known up to the present time.
            /// </summary>
            OBSERVED = 0x02,
            /// <summary>
            /// Specifies raw values that are known in both the past and future.
            /// </summary>
            KNOWN = 0x04
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public DatabaseTemporal()
        {
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
        /// Close the current data source.
        /// </summary>
        public override void Close()
        {
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

                strCmd = "IF OBJECT_ID (N'dbo.ValueStreams', N'U') IS NOT NULL DELETE FROM ValueStreams WHERE (SourceID = " + nSrcId.ToString() + ")";
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
            DeleteValueStreams(nSrcId);
            DeleteRawValues(nSrcId);

            return true;
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

        /// <summary>
        /// Delete all ValueStreams in a data source.
        /// </summary>
        /// <param name="nSrcId">Optionally, specifies the ID of the data source (default = 0, which then uses the open data source ID).</param>
        public void DeleteValueStreams(int nSrcId = 0)
        {
            if (nSrcId == 0)
                nSrcId = m_src.ID;

            string strCmd = "IF OBJECT_ID (N'dbo.ValueStreams', N'U') IS NOT NULL DELETE FROM ValueStreams WHERE (SourceID = " + nSrcId.ToString() + ")";

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
        /// Add all raw values from each data stream within a PlotCollection.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="plots">Specifies the plots containing the value streams.</param>
        /// <param name="bActive">Optionally, specifies the active state of each value (default = true).</param>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        /// <remarks>
        /// Each value stream is stored in each element of the Y values of each plot.  The X value of each plot is used as the timestamp.  
        /// The PlotCollection must contain a set of parameters where one parameter exists for each Y values stream with the name and
        /// stream ID for the associated value stream.
        /// 
        /// For example, if the Y_values contain the following value streams:
        ///   Y_values[0] = log value
        ///   Y_values[1] = hours_from_start
        ///   Y_values[2] = time_on_day
        ///   
        /// then, the PlotCollection parameters must be set to:
        ///   Param0 = "log value", StreamID = 1
        ///   Param1 = "hours_from_start", StreamID = 2
        ///   Param2 = "time_on_day", StreamID = 3
        ///   
        /// NOTE: All times saved in the database are in UTC.
        /// </remarks>
        public void PutRawValues(int nSrcID, int nItemID, PlotCollection plots, bool bActive = true)
        {
            if (plots == null || plots.Count == 0)
                return;

            if (plots.Parameters.Count != plots[0].Y_values.Length)
                throw new Exception("There must be a parameters for each Y value containing the stream ID for each.");

            List<KeyValuePair<string, double>> rgStreams = plots.Parameters.ToList();

            for (int i = 0; i < plots.Count; i++)
            {
                for (int j = 0; j < plots[i].Y_values.Length; j++)
                {
                    RawValue val = new RawValue();
                    val.StreamID = (int)rgStreams[j].Value;
                    val.RawData = (decimal)plots[i].Y_values[j];
                    val.Active = bActive;
                    val.TimeStamp = DateTime.FromFileTimeUtc((long)plots[i].X);
                    val.ItemID = nItemID;
                    val.SourceID = nSrcID;

                    m_entitiesTemporal.RawValues.Add(val);
                }
            }

            List<ValueStream> rgStrm = m_entitiesTemporal.ValueStreams.Where(p => p.ValueItemID == nItemID).ToList();
            foreach (ValueStream strm in rgStrm)
            {
                int nCount = strm.ItemCount.GetValueOrDefault(0);
                nCount += plots.Count;

                strm.ItemCount = nCount;
            }

            m_entitiesTemporal.SaveChanges();
            m_entitiesTemporal.Dispose();
            m_entitiesTemporal = EntitiesConnectionTemporal.CreateEntities(m_ci);
        }

        /// <summary>
        /// Add static raw values for a data stream.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="nStreamID">Specifies the static stream ID.</param>
        /// <param name="fVal">Specifies the static value.</param>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public void PutRawValues(int nSrcID, int nItemID, int nStreamID, float fVal)
        {
            RawValue val = new RawValue();
            val.StreamID = nStreamID;
            val.RawData = (decimal)fVal;
            val.Active = true;
            val.TimeStamp = null; // Static values are not bound by time.
            val.ItemID = nItemID;
            val.SourceID = nSrcID;

            m_entitiesTemporal.RawValues.Add(val);

            List<ValueStream> rgStrm = m_entitiesTemporal.ValueStreams.Where(p => p.ValueItemID == nItemID && p.ID == nStreamID).ToList();
            foreach (ValueStream strm in rgStrm)
            {
                int nCount = strm.ItemCount.GetValueOrDefault(0);
                nCount++;
                strm.ItemCount = nCount;
            }

            m_entitiesTemporal.SaveChanges();
            m_entitiesTemporal.Dispose();
            m_entitiesTemporal = EntitiesConnectionTemporal.CreateEntities(m_ci);
        }

        private bool compareTime(List<DateTime> rg1, List<RawValue> rg2)
        {
            if (rg1.Count != rg2.Count)
                return false;

            for (int i = 0; i < rg1.Count; i++)
            {
                if (rg1[i] != rg2[i].TimeStamp)
                    return false;
            }

            return true;
        }

        private PlotCollection getRawValues(int nSrcID, int nItemID, STREAM_CLASS_TYPE classType, DateTime dtStart, int nCount, bool bNormalizedValue, out DateTime dtEnd, out bool bEOD)
        {
            List<ValueStream> rgStreams = null;
            List<DateTime> rgTime = new List<DateTime>();
            IQueryable<RawValue> iqry = null;
            List<float[]> rgRawData = new List<float[]>();
            int nItemCount = 0;

            dtEnd = DateTime.MinValue;

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                rgStreams = entities.ValueStreams.AsNoTracking().Where(p => p.ValueItemID == nItemID && p.ClassTypeID == (int)classType).OrderBy(p => p.Ordering).ToList();
                iqry = entities.RawValues.AsNoTracking().Where(p => p.SourceID == nSrcID && p.ItemID == nItemID && p.TimeStamp >= dtStart);

                // Collect the data.
                for (int i = 0; i < rgStreams.Count; i++)
                {
                    int nID = rgStreams[i].ID;
                    List<RawValue> rgVal = iqry.Where(p => p.StreamID == nID).ToList();
                    float[] rgRawData1 = null;

                    if (bNormalizedValue)
                        rgRawData1 = rgVal.Select(p => (float)p.NormalizedData).ToArray();
                    else
                        rgRawData1 = rgVal.Select(p => (float)p.RawData).ToArray();

                    if (nItemCount == 0)
                        nItemCount = rgRawData1.Length;
                    else if (nItemCount != rgRawData1.Length)
                        throw new Exception("The number of values in each stream must be the same.");

                    if (dtEnd == DateTime.MinValue)
                        dtEnd = rgVal[rgVal.Count - 1].TimeStamp.Value;
                    else if (dtEnd != rgVal[rgVal.Count - 1].TimeStamp.Value)
                        throw new Exception("The end time must be the same for all streams.");

                    rgRawData.Add(rgRawData1);

                    if (i == 0)
                        rgTime = rgVal.Select(p => p.TimeStamp.Value).ToList();
                    else if (!compareTime(rgTime, rgVal))
                        throw new Exception("The time vectors must be the same across all streams.");
                }
            }

            PlotCollection plots = new PlotCollection();

            for (int i = 0; i < nItemCount; i++)
            {
                DateTime dt = rgTime[i];
                long lTime = dt.ToFileTimeUtc();
                float[] rgYval = new float[rgStreams.Count];

                for (int j = 0; j < rgStreams.Count; j++)
                {
                    rgYval[j] = rgRawData[j][i];
                }

                Plot plot = new Plot(lTime, rgYval);
                plot.Tag = dt;
                plots.Add(plot);
            }

            for (int i = 0; i < rgStreams.Count; i++)
            {
                plots.Parameters.Add(rgStreams[i].Name, rgStreams[i].ID);
                plots.ParametersEx.Add(rgStreams[i].Name, rgStreams[i]);
            }

            plots.ParametersEx.Add("SourceID", nSrcID);
            plots.ParametersEx.Add("ItemID", nItemID);

            // If we cannot advance another block before exceeding the end time, we are at end of data.
            TimeSpan ts = TimeSpan.FromSeconds(rgStreams[0].SecondsPerStep.Value * rgStreams[0].ItemCount.Value);
            if (dtEnd + ts > rgStreams[0].EndTime.Value)
                bEOD = true;
            else
                bEOD = false;

            return plots;
        }

        /// <summary>
        /// Load the observed value stream values between a start and end time.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="dtStart">Specifies the start time in UTC.</param>
        /// <param name="nCount">Specifies the number of items to load.</param>
        /// <param name="bNormalizedValue">Specifies to get the normalized value.</param>
        /// <param name="dtEnd">Returns the end time in UTC.</param>
        /// <param name="bEOD">Returns true when the end of data is hit.</param>
        /// <returns>The PlotCollection </returns>
        public PlotCollection GetRawValuesObserved(int nSrcID, int nItemID, DateTime dtStart, int nCount, bool bNormalizedValue, out DateTime dtEnd, out bool bEOD)
        {
            return getRawValues(nSrcID, nItemID, STREAM_CLASS_TYPE.OBSERVED, dtStart, nCount, bNormalizedValue, out dtEnd, out bEOD);
        }

        /// <summary>
        /// Load the observed value stream values between a start and end time.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <param name="dtStart">Specifies the start time in UTC.</param>
        /// <param name="nCount">Specifies the number of items to load.</param>
        /// <param name="bNormalizedValue">Specifies to get the normalized value.</param>
        /// <param name="dtEnd">Returns the end time in UTC.</param>
        /// <param name="bEOD">Returns true when the end of data is hit.</param>
        /// <returns>The PlotCollection </returns>
        public PlotCollection GetRawValuesKnown(int nSrcID, int nItemID, DateTime dtStart, int nCount, bool bNormalizedValue, out DateTime dtEnd, out bool bEOD)
        {
            return getRawValues(nSrcID, nItemID, STREAM_CLASS_TYPE.KNOWN, dtStart, nCount, bNormalizedValue, out dtEnd, out bEOD);
        }

        /// <summary>
        /// Load the static value stream values for a given source and item.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID.</param>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <returns>A list of the static raw values is returned.</returns>
        public List<RawValue> GetStaticValues(int nSrcID, int nItemID)
        {
            STREAM_CLASS_TYPE classType = STREAM_CLASS_TYPE.STATIC;
            List<int> rgStreams = null;

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                rgStreams = entities.ValueStreams.AsNoTracking().Where(p => p.ValueItemID == nItemID && p.ClassTypeID == (int)classType).OrderBy(p => p.Ordering).Select(p => p.ID).ToList();
                return entities.RawValues.AsNoTracking().Where(p => p.SourceID == nSrcID && p.ItemID == nItemID && rgStreams.Contains(p.StreamID.Value)).ToList();
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
        /// <param name="strName">Specifies the name of the value item.</param>
        /// <returns>The value item ID is returned.</returns>
        public int AddValueItem(int nSrcId, string strName)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueItem> rgItems = entities.ValueItems.Where(p=>p.SourceID == nSrcId && p.Name == strName).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].ID;

                ValueItem item = new ValueItem();
                item.Name = strName;
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


        //---------------------------------------------------------------------
        //  Value Streams
        //---------------------------------------------------------------------
        #region Value Streams

        /// <summary>
        /// Add a new known value stream to the database.
        /// </summary>
        /// <param name="nSourceID">Specifies the data source associated with the value item.</param>
        /// <param name="nValueItemID">Specifies the value item associated with the stream.</param>
        /// <param name="strName">Specifies the name of the value stream.</param>
        /// <param name="valueType">Specifies the value stream type (NUMERIC or CATEGORICAL)</param>
        /// <param name="nOrdering">Specifies the ordering of the data.</param>
        /// <param name="dtStart">Specifies the start date of the data.</param>
        /// <param name="dtEnd">Specifies the end date of the data.</param>
        /// <param name="nSecPerStep">Specifies the seconds per time step.</param>
        /// <returns>The value item ID is returned.</returns>
        /// <remarks>Known values are known both in the past and future (e.g., time from start, hour of day, day of week and holidays are known values.</remarks>
        public int AddKnownValueStream(int nSourceID, int nValueItemID, string strName, STREAM_VALUE_TYPE valueType, int nOrdering, DateTime dtStart, DateTime dtEnd, int nSecPerStep)
        {
            STREAM_CLASS_TYPE classType = STREAM_CLASS_TYPE.KNOWN;

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueStream> rgItems = entities.ValueStreams.Where(p => p.SourceID == nSourceID && p.ValueItemID == nValueItemID && p.Name == strName && p.ClassTypeID == (byte)classType && p.ValueTypeID == (byte)valueType).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].ID;

                ValueStream item = new ValueStream();
                item.SourceID = nSourceID;
                item.ValueItemID = nValueItemID;
                item.Name = strName;
                item.ClassTypeID = (byte)classType;
                item.ValueTypeID = (byte)valueType;
                item.Ordering = nOrdering;
                item.StartTime = dtStart;
                item.EndTime = dtEnd;
                item.SecondsPerStep = nSecPerStep;
                entities.ValueStreams.Add(item);
                entities.SaveChanges();

                return item.ID;
            }
        }

        /// <summary>
        /// Add a new observed value stream to the database.
        /// </summary>
        /// <param name="nSourceID">Specifies the data source associated with the value item.</param>
        /// <param name="nValueItemID">Specifies the value item associated with the stream.</param>
        /// <param name="strName">Specifies the name of the value stream.</param>
        /// <param name="valueType">Specifies the value stream type (NUMERIC or CATEGORICAL)</param>
        /// <param name="nOrdering">Specifies the ordering of the data.</param>
        /// <param name="dtStart">Specifies the start date of the data.</param>
        /// <param name="dtEnd">Specifies the end date of the data.</param>
        /// <param name="nSecPerStep">Specifies the seconds per time step.</param>
        /// <returns>The value item ID is returned.</returns>
        /// <remarks>Observed values are only known up to the present time (e.g., log power usage and traffic flow are examples of observed values.)</remarks>
        public int AddObservedValueStream(int nSourceID, int nValueItemID, string strName, STREAM_VALUE_TYPE valueType, int nOrdering, DateTime dtStart, DateTime dtEnd, int nSecPerStep)
        {
            STREAM_CLASS_TYPE classType = STREAM_CLASS_TYPE.OBSERVED;

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueStream> rgItems = entities.ValueStreams.Where(p => p.SourceID == nSourceID && p.ValueItemID == nValueItemID && p.Name == strName && p.ClassTypeID == (byte)classType && p.ValueTypeID == (byte)valueType).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].ID;

                ValueStream item = new ValueStream();
                item.SourceID = nSourceID;
                item.ValueItemID = nValueItemID;
                item.Name = strName;
                item.ClassTypeID = (byte)classType;
                item.ValueTypeID = (byte)valueType;
                item.Ordering = nOrdering;
                item.StartTime = dtStart;
                item.EndTime = dtEnd;
                item.SecondsPerStep = nSecPerStep;
                entities.ValueStreams.Add(item);
                entities.SaveChanges();

                return item.ID;
            }
        }

        /// <summary>
        /// Add a new static value stream to the database.
        /// </summary>
        /// <param name="nSourceID">Specifies the data source associated with the value item.</param>
        /// <param name="nValueItemID">Specifies the value item associated with the stream.</param>
        /// <param name="strName">Specifies the name of the value stream.</param>
        /// <param name="valueType">Specifies the value stream type (NUMERIC or CATEGORICAL)</param>
        /// <param name="nOrdering">Specifies the ordering of the data.</param>
        /// <returns>The value item ID is returned.</returns>
        /// <remarks>Static values are values that are not bound by time (e.g., store location, store type, item type, and customer id are examples of static values.)</remarks>
        public int AddStaticValueStream(int nSourceID, int nValueItemID, string strName, STREAM_VALUE_TYPE valueType, int nOrdering)
        {
            STREAM_CLASS_TYPE classType = STREAM_CLASS_TYPE.STATIC;

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueStream> rgItems = entities.ValueStreams.Where(p => p.SourceID == nSourceID && p.ValueItemID == nValueItemID && p.Name == strName && p.ClassTypeID == (byte)classType && p.ValueTypeID == (byte)valueType).ToList();

                if (rgItems.Count > 0)
                    return rgItems[0].ID;

                ValueStream item = new ValueStream();
                item.SourceID = nSourceID;
                item.ValueItemID = nValueItemID;
                item.Name = strName;
                item.ClassTypeID = (byte)classType;
                item.ValueTypeID = (byte)valueType;
                item.Ordering = nOrdering;
                item.StartTime = null;
                item.EndTime = null;
                item.SecondsPerStep = null;
                entities.ValueStreams.Add(item);
                entities.SaveChanges();

                return item.ID;
            }
        }

        /// <summary>
        /// Returns the value stream ID given the value stream name if found, or 0.
        /// </summary>
        /// <param name="strName">Specifies the value stream to find.</param>
        /// <param name="classType">Specifies the value strea class (STATIC, OBSERVED or KNOWN)</param>
        /// <param name="valueType">Specifies the value stream type (NUMERIC or CATEGORICAL)</param>
        /// <returns>The value stream ID is returned or 0 if not found.</returns>
        public int GetValueStreamID(string strName, STREAM_CLASS_TYPE classType, STREAM_VALUE_TYPE valueType)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueStream> rgItems = entities.ValueStreams.Where(p => p.Name == strName && p.ClassTypeID == (byte)classType && p.ValueTypeID == (byte)valueType).ToList();

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
        public string GetValueStreamName(int nID)
        {
            if (m_rgstrValueStreams.ContainsKey(nID))
                return m_rgstrValueStreams[nID];

            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                List<ValueStream> rgItems = entities.ValueStreams.Where(p => p.ID == nID).ToList();

                if (rgItems.Count > 0)
                {
                    m_rgstrValueStreams.Add(nID, rgItems[0].Name);
                    return rgItems[0].Name;
                }

                return null;
            }
        }

        /// <summary>
        /// Returns a list of all stream ID's associated with an item ID.
        /// </summary>
        /// <param name="nItemID"></param>
        /// <returns></returns>
        public List<int> GetAllStreams(int nItemID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                return entities.ValueStreams.Where(p => p.ValueItemID == nItemID).Select(p => p.ID).ToList();
            }
        }

        /// <summary>
        /// Returns a list of all value streams associated with an ItemID.
        /// </summary>
        /// <param name="nItemID">Specifies the item ID.</param>
        /// <returns>The list of all value streams is returned.</returns>
        public List<ValueStream> GetAllValueStreams(int nItemID)
        {
            using (DNNEntitiesTemporal entities = EntitiesConnectionTemporal.CreateEntities())
            {
                return entities.ValueStreams.Where(p => p.ValueItemID == nItemID).ToList();
            }
        }

        #endregion
    }
}
