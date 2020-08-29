using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ConnectInfo class specifies the server, database and username/password used to connect to a database.
    /// </summary>
    public class ConnectInfo
    {
        string m_strServer;
        string m_strUn;
        string m_strPw;
        string m_strDb;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strServer">Specifies the server (default = null)</param>
        /// <param name="strDb">Specifies the database (default = 'DNN')</param>
        /// <param name="strUn">Specifies the username (default = null)</param>
        /// <param name="strPw">Specifies the password (default = null)</param>
        public ConnectInfo(string strServer = null, string strDb = "DNN", string strUn = null, string strPw = null)
        {
            m_strServer = strServer;
            m_strDb = strDb;
            m_strUn = strUn;
            m_strPw = strPw;
        }

        /// <summary>
        /// Compare another ConnectInfo with this one.
        /// </summary>
        /// <param name="ci">Specifies the other ConnectInfo to compare to.</param>
        /// <returns>If the two ConnectInfo settings are the same, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool Compare(ConnectInfo ci)
        {
            if (ci.Server != m_strServer)
                return false;

            if (ci.Database != m_strDb)
                return false;

            if (ci.Username != m_strUn)
                return false;

            if (ci.Password != m_strPw)
                return false;

            return true;
        }

        /// <summary>
        /// Returns a string representation of the connection.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            if (m_strServer == null)
                return m_strDb;

            return m_strServer + "\\" + m_strDb;
        }

        /// <summary>
        /// Returns the server.
        /// </summary>
        public string Server
        {
            get { return m_strServer; }
            set { m_strServer = value; }
        }

        /// <summary>
        /// Returns the database.
        /// </summary>
        public string Database
        {
            get { return m_strDb; }
            set { m_strDb = value; }
        }

        /// <summary>
        /// Returns the username.
        /// </summary>
        public string Username
        {
            get { return m_strUn; }
        }

        /// <summary>
        /// Returns the password.
        /// </summary>
        public string Password
        {
            get { return m_strPw; }
        }
    }
}
