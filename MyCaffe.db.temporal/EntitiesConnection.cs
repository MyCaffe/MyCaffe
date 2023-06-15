using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Data.SqlClient;
using System.Data.Entity.Core.EntityClient;
using System.Data.Entity;
using MyCaffe.basecode;
using System.Data.Entity.SqlServer;
using System.Data.Entity.Infrastructure;
using MyCaffe.db.image;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The DNNEntities class defines the entities used to connecto the database via Entity Frameworks.
    /// </summary>
    public partial class DNNEntitiesTemporal
    {
        /// <summary>
        /// The DNNEntities constructor.
        /// </summary>
        /// <param name="strConnectionString">Specifies the connection string.</param>
        public DNNEntitiesTemporal(string strConnectionString)
            : base(strConnectionString)
        {
        }
    }

    /// <summary>
    /// The EntitiesConnection class defines how to connect to the database via Entity Frameworks.
    /// </summary>
    public class EntitiesConnectionTemporal : EntitiesConnection
    {
        static Dictionary<int, string> m_rgstrConnections = new Dictionary<int, string>();
        static Dictionary<int, ConnectInfo> m_rgciConnections = new Dictionary<int, ConnectInfo>();

        /// <summary>
        /// The EntitiesConnection constructor.
        /// </summary>
        public EntitiesConnectionTemporal()
        {
        }

        /// <summary>
        /// Creates the connection string used.
        /// </summary>
        /// <param name="strDb">Specifies the database to use (default = 'DNN').</param>
        /// <returns>The connection string is returned.</returns>
        /// <remarks>Note, this uses the server specified in the GlobalDatabaseConnectionInfo.</remarks>
        public static new string CreateConnectionString(string strDb)
        {
            return CreateConnectionString(new ConnectInfo(null, strDb));
        }

        /// <summary>
        /// Creates the connection string used.
        /// </summary>
        /// <param name="ci">Specifies the database connection info (default = 'DNN' db and '.' server).</param>
        /// <returns>The connection string is returned.</returns>
        public static new string CreateConnectionString(ConnectInfo ci)
        {
            if (ci == null)
                ci = g_connectInfo;

            string strDb = ci.Database;
            string strServerName = g_connectInfo.Server;

            if (ci.Server != null)
                strServerName = ci.Server;

            if (strServerName == "NONE" || strServerName == "DEFAULT")
                strServerName = ".";

            string strKey = strDb + strServerName;
            int nKey = strKey.GetHashCode();

            if (m_rgstrConnections.ContainsKey(nKey) && m_rgciConnections.ContainsKey(nKey) && m_rgciConnections[nKey].Compare(ci))
                return m_rgstrConnections[nKey];

            string strProviderName = "System.Data.SqlClient";
            string strDatabaseName = strDb;
            SqlConnectionStringBuilder sqlBuilder = new SqlConnectionStringBuilder();
            EntityConnectionStringBuilder builder = new EntityConnectionStringBuilder();

            sqlBuilder.DataSource = strServerName;
            sqlBuilder.InitialCatalog = strDatabaseName;

            if (string.IsNullOrEmpty(ci.Password))
            {
                sqlBuilder.IntegratedSecurity = true;
            }
            else
            {
                sqlBuilder.PersistSecurityInfo = false;
                sqlBuilder.UserID = ci.Username;
                sqlBuilder.Password = ci.Password;
                sqlBuilder.MultipleActiveResultSets = false;
                sqlBuilder.Encrypt = true;
                sqlBuilder.TrustServerCertificate = true;
                sqlBuilder.ConnectTimeout = 180;
            }

            string strProviderString = sqlBuilder.ToString();

            builder.Provider = strProviderName;
            builder.ProviderConnectionString = strProviderString;
            builder.Metadata = @"res://*/" + strDb + "ModelTemporal.csdl|" + 
                                "res://*/" + strDb + "ModelTemporal.ssdl|" +
                                "res://*/" + strDb + "ModelTemporal.msl";

            string strConnection = builder.ToString();

            if (!m_rgstrConnections.ContainsKey(nKey))
                m_rgstrConnections.Add(nKey, strConnection);
            else
                m_rgstrConnections[nKey] = strConnection;

            if (!m_rgciConnections.ContainsKey(nKey))
                m_rgciConnections.Add(nKey, ci);
            else
                m_rgciConnections[nKey] = ci;

            return strConnection;
        }

        /// <summary>
        /// Returns the DNNEntitiesTemporal to use.
        /// </summary>
        /// <param name="ci">Optionally, specifies the database connection info (default = null, which defaults to 'DNN' db and '.' server).</param>
        /// <returns></returns>
        public static new DNNEntitiesTemporal CreateEntities(ConnectInfo ci = null)
        {
            return new DNNEntitiesTemporal(CreateConnectionString(ci));
        }
    }
}
