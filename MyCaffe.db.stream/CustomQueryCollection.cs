using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    public class CustomQueryCollection
    {
        GenericList<IXCustomQuery> m_rgCustomQueries = new GenericList<IXCustomQuery>();

        public CustomQueryCollection()
        {
        }

        public void Load()
        {
            addCustomQueries();
        }

        public void Add(IXCustomQuery iqry)
        {
            m_rgCustomQueries.Add(iqry);
        }

        public IXCustomQuery Find(string strName)
        {
            foreach (IXCustomQuery iqry in m_rgCustomQueries)
            {
                if (strName == iqry.Name)
                    return iqry;
            }

            return null;
        }

        private void addCustomQueries()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string path = Uri.UnescapeDataString(uri.Path);
            string strPath = Path.GetDirectoryName(path);

            strPath += "\\CustomQuery";

            if (Directory.Exists(strPath))
            {
                string[] rgstrFiles = Directory.GetFiles(strPath);

                foreach (string strFile in rgstrFiles)
                {
                    FileInfo fi = new FileInfo(strFile);

                    if (fi.Extension.ToLower() == ".dll")
                    {
                        IXCustomQuery iqry = loadCustomQuery(strFile);
                        if (iqry != null)
                            m_rgCustomQueries.Add(iqry);
                    }
                }
            }
        }

        private IXCustomQuery loadCustomQuery(string strFile)
        {
            try
            {
                Assembly a = Assembly.LoadFile(strFile);
                AssemblyName aName = a.GetName();

                foreach (Type t in a.GetTypes())
                {
                    Type[] rgT = t.GetInterfaces();

                    foreach (Type iface in rgT)
                    {
                        string strIface = iface.ToString();
                        if (strIface.Contains("IXCustomQuery"))
                        {
                            object obj = Activator.CreateInstance(t);
                            return obj as IXCustomQuery;
                        }
                    }
                }

                return null;
            }
            catch (Exception)
            {
                return null;
            }
        }
    }
}
