using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Data.SqlClient;
using MyCaffe.basecode;
using System.Diagnostics;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The DatabaseManagement class is used to create the image database.
    /// </summary>
    public class DatabaseManagement
    {
        ConnectInfo m_ci;
        string m_strPath;

        /// <summary>
        /// Specifies whether or not the database is just being updated or not.
        /// </summary>
        protected bool m_bUpdateDatabase = true;

        /// <summary>
        /// The DatabaseManagement constructor.
        /// </summary>
        /// <param name="ci">Specifies the connection info (recommended values: db = 'DNN', server = '.')</param>
        /// <param name="strPath">Specifies the file path where the database should be created.</param>
        public DatabaseManagement(ConnectInfo ci, string strPath = "")
        {
            m_ci = ci;
            m_strPath = strPath;
        }

        /// <summary>
        /// Returns the connection information of the database.
        /// </summary>
        public ConnectInfo ConnectInfo
        {
            get { return m_ci; }
        }

        /// <summary>
        /// Returns the connection string used to connect to the database named 'strName'.
        /// </summary>
        /// <param name="strName">Specifies the database name of the database to connect to.</param>
        /// <returns>The connection string is returned.</returns>
        protected string GetConnectionString(string strName)
        {
            return "Data Source=" + m_ci.Server + ";Initial Catalog=" + strName + ";Integrated Security=True; MultipleActiveResultSets=True;";
        }

        /// <summary>
        /// Queries whether or not the database exists.
        /// </summary>
        /// <param name="bExists">Returns <i>true</i> if the database exists, <i>false</i> otherwise.</param>
        /// <returns>Returns <i>null</i> on success, an Exception on error.</returns>
        public Exception DatabaseExists(out bool bExists)
        {
            int nResult = 0;

            bExists = false;

            try
            {
                SqlConnection connection = new SqlConnection(GetConnectionString("master"));
                SqlCommand cmdQuery = new SqlCommand(getQueryDatabaseCmd(m_ci.Database), connection);

                connection.Open();
                SqlDataReader reader = cmdQuery.ExecuteReader();

                while (reader.Read())
                {
                    nResult = reader.GetInt32(0);
                }

                connection.Close();
                cmdQuery.Dispose();
            }
            catch (Exception excpt)
            {
                return excpt;
            }

            bExists = (nResult == 1) ? true : false;

            return null;
        }

        /// <summary>
        /// The PurgeDatabase function delete the data from the database.
        /// </summary>
        /// <returns>Returns <i>null</i> on success, an Exception on error.</returns>
        public Exception PurgeDatabase()
        {
            try
            {
                SqlConnection connection = new SqlConnection(GetConnectionString(m_ci.Database));

                connection.Open();
                deleteTables(connection);
                createTables(connection, false, false);
                connection.Close();
            }
            catch (Exception excpt)
            {
                return excpt;
            }

            return null;
        }

        /// <summary>
        /// The CreateDatabae creates a new instance of the database in Microsoft SQL.
        /// </summary>
        /// <param name="bUpdateDatabase">Specifies to update an existing database.</param>
        /// <returns>Returns <i>null</i> on success, an Exception on error.</returns>
        public Exception CreateDatabase(bool bUpdateDatabase = false)
        {
            try
            {
                SqlConnection connection;

                m_bUpdateDatabase = bUpdateDatabase;

                if (!m_bUpdateDatabase)
                {
                    bool bExists;
                    Exception err = DatabaseExists(out bExists);

                    if (err != null)
                        throw err;

                    if (bExists)
                        throw new Exception("Database already exists!");

                    connection = new SqlConnection(GetConnectionString("master"));
                    SqlCommand cmdCreate;
                    string strCmd = getCreateDatabaseCmd(m_ci.Database, m_strPath);

                    connection.Open();
                    cmdCreate = new SqlCommand(strCmd, connection);
                    cmdCreate.CommandTimeout = 120;
                    cmdCreate.ExecuteNonQuery();
                    cmdCreate.Dispose();
                    connection.Close();
                }

                connection = new SqlConnection(GetConnectionString(m_ci.Database));
                connection.Open();
                createTables(connection, true, m_bUpdateDatabase);
                connection.Close();
            }
            catch (Exception excpt)
            {
                return excpt;
            }

            return null;
        }

        /// <summary>
        /// Create the temporal tables.
        /// </summary>
        public void CreateTemporalTables()
        {
            try
            {
                SqlConnection connection;

                connection = new SqlConnection(GetConnectionString(m_ci.Database));
                connection.Open();
                createTables(connection, true, true, true);
                connection.Close();
            }
            catch (Exception excpt)
            {
                Trace.WriteLine("ERROR: Creating temporal tables - " + excpt.Message);
            }
        }

        /// <summary>
        /// The createTables function creates the tables of the database.
        /// </summary>
        /// <param name="connection">Specifies the SQL connection.</param>
        /// <param name="bFullCreate">When <i>true</i> the full database is created, otherwise all tables are created except the DatasetCreators table.</param>
        /// <param name="bUpdateOnly">When <i>true</i> an existing database is being updated.</param>
        /// <param name="bTemporalOnly">When <i>true</i> only the temporal tables are created.</param>
        protected virtual void createTables(SqlConnection connection, bool bFullCreate, bool bUpdateOnly, bool bTemporalOnly = false)
        {
            SqlCommand cmdCreate;

            if (!bUpdateOnly)
            {
                cmdCreate = new SqlCommand(Properties.Resources.CreateDatasetGroupsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateDatasetsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateDatasetParametersTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateLabelsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateLabelBoostsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImageGroupsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImageMeansTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImageResultsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImageParametersTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImagesTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateSourcesTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateSourceParametersTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateModelGroupsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                if (bFullCreate)
                {
                    cmdCreate = new SqlCommand(Properties.Resources.CreateDatasetCreatorsTable, connection);
                    cmdCreate.ExecuteNonQuery();
                    cmdCreate.Dispose();
                }
            }

            cmdCreate = new SqlCommand(Properties.Resources.CreateRawValuesTable, connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand(Properties.Resources.CreateValueStreamsTable, connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand(Properties.Resources.CreateValueItemsTable, connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            if (!bTemporalOnly)
            {
                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImageIndex, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawImageIndex2, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.CreateRawValuesIndex, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.UpdateRawImageResultsTable, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();

                cmdCreate = new SqlCommand(Properties.Resources.UpdateRawImageResultsTable2, connection);
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();
            }
        }

        /// <summary>
        /// The deleteTables function deletes all tables except for the DatasetCreators table.
        /// </summary>
        /// <param name="connection">Specifies the SQL connection.</param>
        protected virtual void deleteTables(SqlConnection connection)
        {
            SqlCommand cmdCreate;

            cmdCreate = new SqlCommand("DROP TABLE DatasetGroups", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE Datasets", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE DatasetParameters", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE Labels", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE LabelBoosts", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE RawImageGroups", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE RawImageMeans", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE RawImages", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE RawImageParameters", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE Sources", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE SourceParameters", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE ModelGroups", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE RawValues", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE ValueStreams", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE Items", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();
        }

        /// <summary>
        /// Returns the query database command.
        /// </summary>
        /// <param name="strName">Specifies the database name.</param>
        /// <returns>The SQL command is returned.</returns>
        protected string getQueryDatabaseCmd(string strName)
        {
            string strCmd = Properties.Resources.QueryDatabaseExists;

            strCmd = strCmd.Replace("%DBNAME%", strName);

            return strCmd;
        }

        /// <summary>
        /// Returns the create database command.
        /// </summary>
        /// <param name="strName">Specifies the database name.</param>
        /// <param name="strPath">Specifies the file path where the database is to be created.</param>
        /// <returns>The SQL command is returned.</returns>
        protected string getCreateDatabaseCmd(string strName, string strPath)
        {
            string strCmd = Properties.Resources.CreateDatabase;

            while (strCmd.Contains("%DBNAME%"))
            {
                strCmd = strCmd.Replace("%DBNAME%", strName);
            }

            while (strCmd.Contains("%PATH%"))
            {
                strCmd = strCmd.Replace("%PATH%", strPath);
            }

            return strCmd;
        }
    }
}
