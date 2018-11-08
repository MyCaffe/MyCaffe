using MyCaffe.basecode;
using MyCaffe.db.stream.stdqueries;
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
    public class MgrQueryGeneral : IXQuery
    {
        CustomQueryCollection m_colCustomQuery = new CustomQueryCollection();
        PropertySet m_schema;
        CancelEvent m_evtCancel = new CancelEvent();
        IXCustomQuery m_iquery;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strSchema">Specifies the database schema.</param>
        /// <param name="rgCustomQueries">Optionally, specifies any custom queries to diretly add.</param>
        /// <remarks>
        /// The database schema defines the number of custom queries to use along with their names.  A simple key=value; list
        /// defines the streaming database schema using the following format:
        /// \code{.cpp}
        ///  "ConnectionCount=1;
        ///   Connection0_CustomQueryName=Test1;
        ///   Connection0_CustomQueryParam=param_string1
        /// \endcode
        /// Each param_string specifies the parameters of the custom query and may include the database connection string, database
        /// table, and database fields to query.
        /// </remarks>
        public MgrQueryGeneral(string strSchema, List<IXCustomQuery> rgCustomQueries)
        {
            m_colCustomQuery.Load();
            m_schema = new PropertySet(strSchema);

            m_colCustomQuery.Add(new StandardQueryTextFile());
            m_colCustomQuery.Add(new StandardQueryWAVFile());

            foreach (IXCustomQuery icustomquery in rgCustomQueries)
            {
                m_colCustomQuery.Add(icustomquery);
            }

            int nConnections = m_schema.GetPropertyAsInt("ConnectionCount");
            if (nConnections != 1)
                throw new Exception("Currently, the general query type only supports 1 connection.");

            string strConTag = "Connection0";
            string strCustomQuery = m_schema.GetProperty(strConTag + "_CustomQueryName");
            string strCustomQueryParam = m_schema.GetProperty(strConTag + "_CustomQueryParam");
           
            IXCustomQuery iqry = m_colCustomQuery.Find(strCustomQuery);
            if (iqry == null)
                throw new Exception("Could not find the custom query '" + strCustomQuery + "'!");

            if (iqry.QueryType != CUSTOM_QUERY_TYPE.BYTE && iqry.QueryType != CUSTOM_QUERY_TYPE.REAL)
                throw new Exception("The custom query '" + iqry.Name + "' must support the 'CUSTOM_QUERY_TYPE.BYTE' or 'CUSTOM_QUERY_TYPE.REAL' query type!");

            string strParam = ParamPacker.UnPack(strCustomQueryParam);
            m_iquery = iqry.Clone(strParam);
            m_iquery.Open();
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
            m_iquery.Reset();
        }

        /// <summary>
        /// Shutdown the data queries and consolidation thread.
        /// </summary>
        public void Shutdown()
        {
            m_evtCancel.Set();
            m_iquery.Close();
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
            return m_iquery.GetQuerySize();
        }

        /// <summary>
        /// Query the next data in the streaming database.
        /// </summary>
        /// <param name="nWait">Specfies the maximum amount of time (in ms.) to wait for data.</param>
        /// <returns>A simple datum containing the data is returned.</returns>
        public SimpleDatum Query(int nWait)
        {
            if (m_iquery.QueryType == CUSTOM_QUERY_TYPE.REAL)
            {
                List<double[]> rgData = m_iquery.QueryReal();

                if (rgData == null || rgData.Count == 0 || rgData[0].Length == 0)
                    return null;

                double[] rgFullData = new double[rgData[0].Length * rgData.Count];
                int nOffset = 0;

                for (int i = 0; i < rgData.Count; i++)
                {
                    Array.Copy(rgData[i], 0, rgFullData, nOffset, rgData[i].Length);
                    nOffset += rgData[i].Length;
                }

                return new SimpleDatum(true, rgData.Count, 1, rgData[0].Length, -1, DateTime.MinValue, null,  rgFullData.ToList(), 0, false, -1);
            }
            else
            {
                byte[] rgData = m_iquery.QueryBytes();

                if (rgData == null || rgData.Length == 0)
                    return null;

                return new SimpleDatum(false, 1, 1, rgData.Length, -1, DateTime.MinValue, rgData.ToList(), null, 0, false, -1);
            }
        }

        /// <summary>
        /// Converts the output values into the native type used by the CustomQuery.
        /// </summary>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="type">Returns the output type.</param>
        /// <returns>The converted output data is returned as a byte stream.</returns>
        public byte[] ConvertOutput(float[] rg, out string type)
        {
            return m_iquery.ConvertOutput(rg, out type);
        }

        /// The Query information returns information about the data queried such as header information.
        /// </summary>
        /// <returns>The information about the data is returned.</returns>
        public Dictionary<string, float> QueryInfo()
        {
            return m_iquery.QueryInfo();
        }
    }
}
