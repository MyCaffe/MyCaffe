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
        TYPE m_location = TYPE.NONE;
        string m_strServer;
        string m_strUn;
        string m_strPw;
        string m_strDb;

        /// <summary>
        /// Defines the generic connection location
        /// </summary>
        public enum TYPE
        {
            /// <summary>
            /// Specifies no location.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies a local connection.
            /// </summary>
            LOCAL,
            /// <summary>
            /// Specifies a remote connection.
            /// </summary>
            REMOTE,
            /// <summary>
            /// Specifies a connection to Azure.
            /// </summary>
            AZURE
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strServer">Specifies the server (default = null)</param>
        /// <param name="strDb">Specifies the database (default = 'DNN')</param>
        /// <param name="strUn">Specifies the username (default = null)</param>
        /// <param name="strPw">Specifies the password (default = null)</param>
        /// <param name="location">Specifies a generic location type.</param>
        public ConnectInfo(string strServer = null, string strDb = "DNN", string strUn = null, string strPw = null, TYPE location = TYPE.NONE)
        {
            m_strServer = strServer;
            m_strDb = strDb;
            m_strUn = strUn;
            m_strPw = strPw;
            m_location = location;
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

            if (ci.Location != m_location)
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

            if (m_strServer == "AZURE")
                return m_strServer;

            return m_strServer + "\\" + m_strDb;
        }

        /// <summary>
        /// Returns a string representation of the connection.
        /// </summary>
        /// <param name="strAzure">Specifies the Azure server.</param>
        /// <returns>The string representation is returned.</returns>
        public string ToString(string strAzure)
        {
            if (m_strServer.Contains(strAzure))
                return "AZURE";

            return ToString();
        }

        /// <summary>
        /// Get/set the generic location type.
        /// </summary>
        public TYPE Location
        {
            get { return m_location; }
            set { m_location = value; }
        }

        /// <summary>
        /// Get/set the server.
        /// </summary>
        public string Server
        {
            get { return m_strServer; }
            set { m_strServer = value; }
        }

        /// <summary>
        /// Get/set the database.
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

        /// <summary>
        /// Returns a string representation of the connection information.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public string ToInfoString()
        {
            if (Server == "DEFAULT" || Server == "AZURE")
                return Server;

            return Server + ";" + Database + ";" + Username + ";" + Password;
        }

        /// <summary>
        /// Returns the connection information as a set of key-value pairs.
        /// </summary>
        /// <returns>The key-value pair string is returned.</returns>
        public string ToKeyValuePairs()
        {
            string str = "";

            str += "Type\t" + Location.ToString() + ";";
            str += "Server\t" + Server + ";";
            str += "Database\t" + Database + ";";
            str += "Username\t" + Username + ";";
            str += "Password\t" + Password + ";";

            return str;
        }

        /// <summary>
        /// Parses a key-value pair string containing the connection information and returns a ConnectInfo.
        /// </summary>
        /// <param name="strKeyValPairs">Specifies the key-value pairs string.</param>
        /// <returns>The new ConnectInfo object is returned with the connection information.</returns>
        public static ConnectInfo ParseKeyValuePairs(string strKeyValPairs)
        {
            string[] rgstr = strKeyValPairs.Split(';');
            string strServer = null;
            string strDatabase = null;
            string strUsername = null;
            string strPassword = null;
            string strType = null;
            TYPE location = TYPE.NONE;

            foreach (string str1 in rgstr)
            {
                if (!string.IsNullOrEmpty(str1))
                {
                    string[] rgstr1 = str1.Split('\t');
                    if (rgstr1.Length != 2)
                        throw new Exception("ConnectInfo: Invalid key-value pair!");

                    if (rgstr1[0] == "Type")
                        strType = rgstr1[1];

                    else if (rgstr1[0] == "Server")
                        strServer = rgstr1[1];

                    else if (rgstr1[0] == "Database")
                        strDatabase = rgstr1[1];

                    else if (rgstr1[0] == "Username")
                        strUsername = rgstr1[1];

                    else if (rgstr1[0] == "Password")
                        strPassword = rgstr1[1];
                }
            }

            if (strType == TYPE.AZURE.ToString())
                location = TYPE.AZURE;
            else if (strType == TYPE.REMOTE.ToString())
                location = TYPE.REMOTE;
            else if (strType == TYPE.LOCAL.ToString())
                location = TYPE.LOCAL;

            if (strServer.Length == 0)
                strServer = null;

            if (strDatabase.Length == 0)
                strDatabase = null;

            if (strUsername.Length == 0)
                strUsername = null;

            if (strPassword.Length == 0)
                strPassword = null;

            return new ConnectInfo(strServer, strDatabase, strUsername, strPassword, location);
        }
    }
}
