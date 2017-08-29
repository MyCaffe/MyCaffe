using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.imagedb
{
    public class DatabaseInstanceQuery
    {
        public DatabaseInstanceQuery()
        {
        }

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
