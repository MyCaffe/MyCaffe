using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.image
{
    /// <summary>
    /// The DatabaseInstanceQuery class is used to find all installed instances of SQL on a given machine.
    /// </summary>
    public class DatabaseInstanceQuery
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public DatabaseInstanceQuery()
        {
        }

        /// <summary>
        /// Returns a list of the SQL instances as a string.
        /// </summary>
        /// <returns>The list of SQL instances is returned.</returns>
        public static string GetInstancesAsText()
        {
            List<string> rgstr = GetInstances();
            string strOut = "";

            foreach (string str in rgstr)
            {
                strOut += str + "\n";
            }

            return strOut;
        }

        /// <summary>
        /// Returns a list of the SQL instances as a list of strings.
        /// </summary>
        /// <returns>A list of SQL instance strings is returned.</returns>
        public static List<string> GetInstances()
        {
            RegistryKey baseKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, RegistryView.Registry64);
            RegistryKey key = baseKey.OpenSubKey(@"SOFTWARE\Microsoft\Microsoft SQL Server\Instance Names\SQL");
            List<string> rgstr = new List<string>();

            if (key == null)
                return rgstr;

            foreach (string str in key.GetValueNames())
            {
                rgstr.Add(".\\" + str);
            }

            key.Close();
            baseKey.Close();

            return rgstr;
        }
    }
}
