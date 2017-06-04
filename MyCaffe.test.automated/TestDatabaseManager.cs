using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.test.automated
{
    public class TestDatabaseManager
    {
        string m_strName = "Testing";
        string m_strInstance = ".";

        public TestDatabaseManager()
        {
        }

        public string DatabaseName
        {
            get { return m_strName; }
        }

        private string GetConnectionString(string strName)
        {
            return "Data Source=" + m_strInstance + ";Initial Catalog=" + strName + ";Integrated Security=True; MultipleActiveResultSets=True;";
        }

        public Exception DatabaseExists(out bool bExists)
        {
            int nResult = 0;

            bExists = false;

            try
            {
                SqlConnection connection = new SqlConnection(GetConnectionString("master"));
                SqlCommand cmdQuery = new SqlCommand(getQueryDatabaseCmd(m_strName), connection);

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

        public Exception CreateDatabase(string strPath)
        {
            try
            {
                bool bExists;
                Exception err = DatabaseExists(out bExists);

                if (err != null)
                    throw err;

                if (bExists)
                    throw new Exception("Database already exists!");

                SqlConnection connection = new SqlConnection(GetConnectionString("master"));
                SqlCommand cmdCreate;
                string strCmd = getCreateDatabaseCmd(m_strName, strPath);

                connection.Open();
                cmdCreate = new SqlCommand(strCmd, connection);
                cmdCreate.CommandTimeout = 120;
                cmdCreate.ExecuteNonQuery();
                cmdCreate.Dispose();
                connection.Close();

                connection = new SqlConnection(GetConnectionString(m_strName));
                connection.Open();
                createTables(connection, true);
                connection.Close();
            }
            catch (Exception excpt)
            {
                return excpt;
            }

            return null;
        }

        protected virtual void createTables(SqlConnection connection, bool bFullCreate)
        {
            SqlCommand cmdCreate;

            cmdCreate = new SqlCommand(Properties.Resources.CreateSessionsTable, connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand(Properties.Resources.CreateTestsTable, connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();
        }

        protected virtual void deleteTables(SqlConnection connection)
        {
            SqlCommand cmdCreate;

            cmdCreate = new SqlCommand("DROP TABLE Sessions", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();

            cmdCreate = new SqlCommand("DROP TABLE Tests", connection);
            cmdCreate.ExecuteNonQuery();
            cmdCreate.Dispose();
        }

        protected string getQueryDatabaseCmd(string strName)
        {
            string strCmd = Properties.Resources.QueryDatabaseExists;

            strCmd = strCmd.Replace("%DBNAME%", strName);

            return strCmd;
        }

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
