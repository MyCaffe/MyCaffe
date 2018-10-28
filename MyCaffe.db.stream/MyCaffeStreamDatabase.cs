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
        IXQuery m_iquery;
        List<IXCustomQuery> m_rgCustomQueryToAdd = new List<IXCustomQuery>();

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
            m_iquery.Shutdown();
        }

        /// <summary>
        /// The Initialize method initializes the streaming database component, preparing it for data queries.
        /// </summary>
        /// <param name="qt">Specifies the query type to use (see remarks).</param>
        /// <param name="strSettings">Specifies the query settings to use (see remarks.)</param>
        /// <param name="strSchema">Specifies the query schema to use.</param>
        /// <remarks>
        /// Additional settings for each query type are specified in the 'strSettings' parameter as a set
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
        ///    
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
        public void Initialize(QUERY_TYPE qt, string strSchema)
        {
            if (qt == QUERY_TYPE.SYNCHRONIZED)
            {
                PropertySet ps = new PropertySet(strSchema);
                int nQueryCount = ps.GetPropertyAsInt("QueryCount", 0);
                DateTime dtStart = ps.GetPropertyAsDateTime("Start");
                int nTimeSpanInMs = ps.GetPropertyAsInt("TimeSpanInMs");
                int nSegmentSize = ps.GetPropertyAsInt("SegmentSize");
                int nMaxCount = ps.GetPropertyAsInt("MaxCount");
                
                m_iquery = new MgrQueryTime(nQueryCount, dtStart, nTimeSpanInMs, nSegmentSize, nMaxCount, strSchema, m_rgCustomQueryToAdd);
            }
            else
            {
                m_iquery = new MgrQueryGeneral(strSchema, m_rgCustomQueryToAdd);
            }
        }

        /// <summary>
        /// Shutdonw the streaming database.
        /// </summary>
        public void Shutdown()
        {
            m_iquery.Shutdown();
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
            m_rgCustomQueryToAdd.Add(iqry);
        }

        /// <summary>
        /// Query the next data in the streaming database.
        /// </summary>
        /// <param name="nWait">Specfies the maximum amount of time (in ms.) to wait for data.</param>
        /// <returns>A simple datum containing the data is returned.</returns>
        public SimpleDatum Query(int nWait)
        {
            return m_iquery.Query(nWait);
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
            List<int> rg = m_iquery.GetQuerySize();

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
            m_iquery.Reset(nStartOffset);
        }
    }
}
