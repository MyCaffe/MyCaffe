using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    public class GymCollection : IEnumerable<IXMyCaffeGym>
    {
        List<IXMyCaffeGym> m_rgGym = new List<IXMyCaffeGym>();

        public GymCollection()
        {
        }

        public void Load()
        {
            m_rgGym.Add(new CartPoleGym());
            m_rgGym.Add(new AtariGym());

            addCustomGyms();
        }

        private void addCustomGyms()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string path = Uri.UnescapeDataString(uri.Path);
            string strPath = Path.GetDirectoryName(path);

            strPath += "\\CustomGyms";

            if (Directory.Exists(strPath))
            {
                string[] rgstrFiles = Directory.GetFiles(strPath);

                foreach (string strFile in rgstrFiles)
                {
                    FileInfo fi = new FileInfo(strFile);

                    if (fi.Extension.ToLower() == ".dll")
                    {
                        IXMyCaffeGym igym = loadCustomGym(strFile);
                        if (igym != null)
                            m_rgGym.Add(igym);
                    }
                }
            }
        }

        private IXMyCaffeGym loadCustomGym(string strFile)
        {
            try
            {
                Assembly a = Assembly.LoadFile(strFile);
                AssemblyName aName = a.GetName();
                IXMyCaffeGym igym = null;

                foreach (Type t in a.GetTypes())
                {
                    Type[] rgT = t.GetInterfaces();

                    foreach (Type iface in rgT)
                    {
                        string strIface = iface.ToString();
                        if (strIface.Contains("IXMyCaffeGym"))
                        {
                            object obj = Activator.CreateInstance(t);
                            return obj as IXMyCaffeGym;
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

        public IXMyCaffeGym Find(string strName)
        {
            foreach (IXMyCaffeGym igym in m_rgGym)
            {
                if (igym.Name == strName)
                    return igym;
            }

            return null;
        }

        public IEnumerator<IXMyCaffeGym> GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }
    }
}
