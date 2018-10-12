using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Data.SqlClient;
using System.Data.Entity.Core.EntityClient;
using System.Data.Entity;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The DNNEntities class defines the entities used to connecto the database via Entity Frameworks.
    /// </summary>
    public partial class DNNEntities
    {
        /// <summary>
        /// The DNNEntities constructor.
        /// </summary>
        /// <param name="strConnectionString">Specifies the connection string.</param>
        public DNNEntities(string strConnectionString)
            : base(strConnectionString)
        {
        }
    }

    /// <summary>
    /// The EntitiesConnection class defines how to connect to the database via Entity Frameworks.
    /// </summary>
    public class EntitiesConnection
    {
        static string g_strServerName = ".";
        static Dictionary<int, string> m_rgstrConnections = new Dictionary<int, string>();

        /// <summary>
        /// The EntitiesConnection constructor.
        /// </summary>
        public EntitiesConnection()
        {
        }

        /// <summary>
        /// Get/set the global database server name.
        /// </summary>
        public static string GlobalDatabaseServerName
        {
            get { return g_strServerName; }
            set { g_strServerName = value; }
        }

        /// <summary>
        /// Creates the connection string used.
        /// </summary>
        /// <param name="strDb">Specifies the database name (default = "DNN")</param>
        /// <param name="strServerName">Specifies the server instance (default = ".")</param>
        /// <returns></returns>
        public static string CreateConnectionString(string strDb = "DNN", string strServerName = ".")
        {
            string strKey = strDb + strServerName;
            int nKey = strKey.GetHashCode();

            if (m_rgstrConnections.ContainsKey(nKey))
                return m_rgstrConnections[nKey];

            string strProviderName = "System.Data.SqlClient";
            string strDatabaseName = strDb;
            SqlConnectionStringBuilder sqlBuilder = new SqlConnectionStringBuilder();
            EntityConnectionStringBuilder builder = new EntityConnectionStringBuilder();

            sqlBuilder.DataSource = strServerName;
            sqlBuilder.InitialCatalog = strDatabaseName;
            sqlBuilder.IntegratedSecurity = true;

            string strProviderString = sqlBuilder.ToString();

            builder.Provider = strProviderName;
            builder.ProviderConnectionString = strProviderString;
            builder.Metadata = @"res://*/" + strDb + "Model.csdl|" + 
                                "res://*/" + strDb + "Model.ssdl|" +
                                "res://*/" + strDb + "Model.msl";

            string strConnection = builder.ToString();

            m_rgstrConnections.Add(nKey, strConnection);

            return strConnection;
        }

        /// <summary>
        /// Returns the DNNEntities to use.
        /// </summary>
        /// <param name="strDb">Specifies the database name (default = "DNN")</param>
        /// <returns></returns>
        public static DNNEntities CreateEntities(string strDb = "DNN")
        {
            return new DNNEntities(CreateConnectionString(strDb, g_strServerName));
        }
    }
}
