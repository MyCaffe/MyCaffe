using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.db.stream namespace contains all data streaming related classes.
/// </summary>
namespace MyCaffe.db.stream
{
#pragma warning disable 1591

    [ServiceContract]
    public interface IXStreamDatabaseEvent /** @private */
    {
        [OperationContract(IsOneWay = false)]
        void OnResult(string strMsg, double dfProgress);

        [OperationContract(IsOneWay = false)]
        void OnError(StreamDatabaseErrorData err);
    }

#pragma warning restore 1591

    /// <summary>
    /// Defines the query type to use.
    /// </summary>
    [Serializable]
    [DataContract]
    public enum QUERY_TYPE
    {
        /// <summary>
        /// Specifies to use a general query.
        /// </summary>
        GENERAL,
        /// <summary>
        /// Specifies to use a synchronized query.
        /// </summary>
        SYNCHRONIZED
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
        /// <param name="qt">Specifies the query type to use (see remarks).</param>
        /// <param name="strSchema">Specifies the query schema to use.</param>
        /// <remarks>
        /// Additional settings for each query type are specified in the 'strSchema' parameter as a set
        /// of key=value pairs for each of the settings.  The following are the query specific settings
        /// that are expected for each QUERY_TYPE.
        /// 
        /// qt = TIME:
        ///    'QueryCount' - Specifies the number of items to include in each query.
        ///    'Start' - Specifies the start date of the stream.
        ///    'TimeSpanInMs' - Specifies the time increment between data items in the stream in milliseconds.
        ///    'SegmentSize' - Specifies the segment size of data queried from the database.
        ///    'MaxCount' - Specifies the maximum number of items to load into memory for each custom query.
        ///    
        /// qt = GENERAL:
        ///    none at this time.
        /// </remarks>
        [OperationContract(IsOneWay = false)]
        void Initialize(QUERY_TYPE qt, string strSchema);

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

        /// <summary>
        /// The Query information returns information about the data queried such as header information.
        /// </summary>
        /// <returns>The information about the data is returned.</returns>
        [OperationContract(IsOneWay = false)]
        Dictionary<string, float> QueryInfo();
    }

#pragma warning disable 1591

    /// <summary>
    /// The IXQuery interface is implemented by each MgrQuery within the MyCaffeStreamDatabase.
    /// </summary>
    public interface IXQuery /** @private */
    {
        void AddDirectQuery(IXCustomQuery iqry);
        void Reset(int nStartOffset);
        void Shutdown();
        List<int> GetQuerySize();
        SimpleDatum Query(int nWait);
        byte[] ConvertOutput(float[] rg, out string type);
        Dictionary<string, float> QueryInfo();
    }

#pragma warning restore 1591

    /// <summary>
    /// Defines the custom query type to use.
    /// </summary>
    public enum CUSTOM_QUERY_TYPE
    {
        /// <summary>
        /// Each custom query supporting the BYTE query, must implement the QueryByte function.
        /// </summary>
        BYTE,
        /// <summary>
        /// Each custom query supporting the REAL query, where the base type is <i>float</i>, must implement the QueryReal function.
        /// </summary>
        REAL_FLOAT,
        /// <summary>
        /// Each custom query supporting the REAL query, where the base type is <i>double</i>, must implement the QueryReal function.
        /// </summary>
        REAL_DOUBLE,
        /// <summary>
        /// Each custom query supporting the TIME query, must implement the QueryByTime funtion.
        /// </summary>
        TIME
    }


    /// <summary>
    /// The custom query interface defines the functions implemented by each Custom Query object used
    /// to specifically query the tables of the underlying database.
    /// </summary>
    /// <remarks>
    /// Each Custom Query implementation DLL must be placed within the \code{.cpp}'./CustomQuery'\endcode directory that
    /// is relative to the MyCaffe.db.stream.dll file location. For example, see the following directory
    /// structure:
    /// \code{.cpp}
    /// c:/temp/MyCaffe.db.stream.dll
    /// c:/temp/CustomQuery/mycustomquery.dll  - implements the IXCustomQuery interface.
    /// \endcode
    /// </remarks>
    public interface IXCustomQuery
    {
        /// <summary>
        /// Returns the custom query type supported by the custom query.
        /// </summary>
        CUSTOM_QUERY_TYPE QueryType { get; }
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
        /// Returns the query count for the current query.
        /// </summary>
        /// <returns>The query size is returned..</returns>
        List<int> GetQuerySize();
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
        /// Query the raw bytes.
        /// </summary>
        /// <returns></returns>
        byte[] QueryBytes();
        /// <summary>
        /// Query the data as a set one or more double arrays.
        /// </summary>
        /// <returns></returns>
        List<double[]> QueryRealD();
        /// <summary>
        /// Query the data as a set one or more float arrays.
        /// </summary>
        /// <returns></returns>
        List<float[]> QueryRealF();
        /// <summary>
        /// The Query information returns information about the data queried such as header information.
        /// </summary>
        /// <returns>The information about the data is returned.</returns>
        Dictionary<string, float> QueryInfo();
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
        /// <summary>
        /// Converts the output values into the native type used by the CustomQuery.
        /// </summary>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="strType">Returns the output type.</param>
        /// <returns>The converted output data is returned as a byte stream.</returns>
        byte[] ConvertOutput(float[] rg, out string strType);
    }

    /// <summary>
    /// The ParamPacker is use to pack and unpack parameters sent to each custom query.
    /// </summary>
    public class ParamPacker
    {
        /// <summary>
        /// Pack the custom query parameters.
        /// </summary>
        /// <param name="str">Specifies the parameters.</param>
        /// <returns>The packed parameters are returned.</returns>
        public static string Pack(string str)
        {
            str = Utility.Replace(str, ';', '|');
            return Utility.Replace(str, '=', '~');
        }

        /// <summary>
        /// Unpack the custom query parameters.
        /// </summary>
        /// <param name="str">Specifies the parameters.</param>
        /// <returns>The unpacked parameters are returned.</returns>
        public static string UnPack(string str)
        {
            str = Utility.Replace(str, '|', ';');
            return Utility.Replace(str, '~', '=');
        }
    }

#pragma warning disable 1591

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

#pragma warning restore 1591
}
