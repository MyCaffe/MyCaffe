using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.db.stream namespace contains the main body of MyCaffeStreamDatabase used to stream data.  
/// </summary>
namespace MyCaffe.db.stream
{
    /// <summary>
    /// The MyCaffeStreamDatabase provides a streaming data input source to MyCaffe gyms used as input for dynamic, reinforcement learning.
    /// </summary>
    public partial class MyCaffeStreamDatabase : Component, IXStreamDatabase
    {
        Log m_log;
        MgrQuery m_qryMgr = new MgrQuery();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">Specifies the output log.</param>
        public MyCaffeStreamDatabase(Log log)
        {
            m_log = log;
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">Specifies the container.</param>
        public MyCaffeStreamDatabase(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        private void dispose()
        {
            m_qryMgr.Shutdown();
        }

        /// <summary>
        /// The Initialize method initializes the streaming database component, preparing it for data queries.
        /// </summary>
        /// <param name="nQueryCount">Specifies the size of each query.</param>
        /// <param name="dtStart">Specifies the state date used for data collection.</param>
        /// <param name="nTimeSpanInMs">Specifies the time increment used between each data item.</param>
        /// <param name="nSegmentSize">Specifies the amount of data to query on the back-end from each custom query.</param>
        /// <param name="nMaxCount">Specifies the maximum number of items to allow in memory.</param>
        /// <param name="strSchema">Specifies the database schema.</param>
        /// <remarks>
        /// The database schema defines the number of custom queries to use along with their names.  A simple key=value; list
        /// defines the streaming database schema using the following format:
        /// 
        /// "ConnectionCount=2;
        ///  Connection0_CustomQueryName=Test1;
        ///  Connection0_CustomQueryParam=param_string1
        ///  Connection1_CustomQueryName=Test2;
        ///  Connection1_CustomQueryParam=param_string2"
        ///  
        /// Each param_string specifies the parameters of the custom query and may include the database connection string, database
        /// table, and database fields to query.
        /// </remarks>
        public void Initialize(int nQueryCount, DateTime dtStart, int nTimeSpanInMs, int nSegmentSize, int nMaxCount, string strSchema)
        {
            m_qryMgr.Initialize(nQueryCount, dtStart, nTimeSpanInMs, nSegmentSize, nMaxCount, strSchema);
        }

        /// <summary>
        /// Shutdonw the streaming database.
        /// </summary>
        public void Shutdown()
        {
            m_qryMgr.Shutdown();
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
            m_qryMgr.AddDirectQuery(iqry);
        }

        /// <summary>
        /// Query the next data in the streaming database.
        /// </summary>
        /// <param name="nWait">Specfies the maximum amount of time (in ms.) to wait for data.</param>
        /// <returns>A simple datum containing the data is returned.</returns>
        public SimpleDatum Query(int nWait)
        {
            return m_qryMgr.Query(nWait);
        }

        /// <summary>
        /// Returns the query size of the data in the form:
        /// [0] = channels
        /// [1] = height
        /// [2] = width.
        /// </summary>
        /// <returns>The query size is returned.</returns>
        public int[] QuerySize()
        {
            List<int> rg = m_qryMgr.GetQuerySize();

            if (rg == null)
                return null;

            return rg.ToArray();
        }

        /// <summary>
        /// Reset the query to the start date used in Initialize, optionally with an offset from the start.
        /// </summary>
        /// <param name="nStartOffset">Optionally, specifies the offset from the start to use (default = 0).</param>
        public void Reset(int nStartOffset = 0)
        {
            m_qryMgr.Reset(nStartOffset);
        }
    }
}
