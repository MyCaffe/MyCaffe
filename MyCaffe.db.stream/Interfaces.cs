using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    [ServiceContract]
    public interface IXStreamDatabaseEvent /** @private */
    {
        [OperationContract(IsOneWay = false)]
        void OnResult(string strMsg, double dfProgress);

        [OperationContract(IsOneWay = false)]
        void OnError(StreamDatabaseErrorData err);
    }

    /// <summary>
    /// The IXStreamDatabase interface is the main interface to the MyCaffe Streaing Database.
    /// </summary>
    [ServiceContract(CallbackContract = typeof(IXStreamDatabaseEvent), SessionMode = SessionMode.Required)]
    public interface IXStreamDatabase
    {
        /// <summary>
        /// Initialize the streaming database by loading the initial queues.
        /// </summary>
        /// <param name="nQueryCount">Specifies the number of items to include in each query.</param>
        /// <param name="dtStart">Specifies the start date of the stream.</param>
        /// <param name="nTimeSpanInMs">Specifies the time increment between data items in the stream in milliseconds.</param>
        /// <param name="nSegmentSize">Specifies the segment size of data queried from the database.</param>
        /// <param name="nMaxCount">Specifies the maximum number of items to load into memory for each custom query.</param>
        /// <param name="strSchema">Specifies the query schema to use.</param>
        [OperationContract(IsOneWay = false)]
        void Initialize(int nQueryCount, DateTime dtStart, int nTimeSpanInMs, int nSegmentSize, int nMaxCount, string strSchema);

        /// <summary>
        /// Shutdown the database.
        /// </summary>
        [OperationContract(IsOneWay = false)]
        void Shutdown();

        /// <summary>
        /// Reset the query postion to the start established during Initialize.
        /// </summary>
        /// <param name="nStartOffset">Optionally, specifies an offset from the start to use (default = 0).</param>
        [OperationContract(IsOneWay = false)]
        void Reset(int nStartOffset = 0);

        /// <summary>
        /// Query a setgment of data from the internal queueus.
        /// </summary>
        /// <param name="nWait">Optionally, specifies the amount of time to wait for data in ms. (default = 1000ms).</param>
        /// <returns></returns>
        [OperationContract(IsOneWay = false)]
        SimpleDatum Query(int nWait = 1000);

        /// <summary>
        /// Returns the Query size using the Blob sizing methodology.
        /// </summary>
        /// <returns>The data size is returned.</returns>
        [OperationContract(IsOneWay = false)]
        int[] QuerySize();
    }

    /// <summary>
    /// The custom query interface defines the functions implemented by each Custom Query object used
    /// to specifically query the tables of the underlying database.
    /// </summary>
    /// <remarks>
    /// Each Custom Query implementation DLL must be placed within the '.\CustomQuery' directory that
    /// is relative to the MyCaffe.db.stream.dll file location. For example, see the following directory
    /// structure:
    /// 
    /// c:\temp\MyCaffe.db.stream.dll
    /// c:\temp\CustomQuery\mycustomquery.dll  - implements the IXCustomQuery interface.
    /// </remarks>
    public interface IXCustomQuery
    {
        /// <summary>
        /// Returns the name of the Custom Query.
        /// </summary>
        string Name { get; }
        /// <summary>
        /// Returns the field count for this query.
        /// </summary>
        int FieldCount { get; }
        /// <summary>
        /// Open a connection to the underlying database using the connection string specified.
        /// </summary>
        void Open();
        /// <summary>
        /// Close a currently open connection.
        /// </summary>
        void Close();
        /// <summary>
        /// Query the fields specified (in the Open function) starting from the date-time specified.
        /// </summary>
        /// <remarks>Items are returned in column-major format (e.g. datetime, val1, val2, datetime, val1, val2...)</remarks>
        /// <param name="dt">Specifies the start date-time where the query should start.  Note, using ID based querying assumes that all other Custom Queries used have synchronized date-time fields.</param>
        /// <param name="ts">Specifies the timespan between data items.</param>
        /// <param name="nCount">Specifies the number of items to query.</param>
        /// <returns>A two dimensional array is returned containing the items for each field queried.</returns>
        double[] QueryByTime(DateTime dt, TimeSpan ts, int nCount);
        /// <summary>
        /// Return a new instance of the custom query.
        /// </summary>
        /// <param name="strParam">Specifies the custom query parameters.</param>
        /// <returns>A new instance of the custom query is returned.</returns>
        IXCustomQuery Clone(string strParam);
        /// <summary>
        /// Reset the custom query.
        /// </summary>
        void Reset();
    }

    public class ParamPacker
    {
        public static string Pack(string str)
        {
            str = Utility.Replace(str, ';', '|');
            return Utility.Replace(str, '=', '~');
        }

        public static string UnPack(string str)
        {
            str = Utility.Replace(str, '|', ';');
            return Utility.Replace(str, '~', '=');
        }
    }

    [DataContract]
    public class StreamDatabaseErrorData /** @private */
    {
        [DataMember]
        public bool Result { get; set; }
        [DataMember]
        public string ErrorMessage { get; set; }
        [DataMember]
        public string ErrorDetails { get; set; }
    }
}
