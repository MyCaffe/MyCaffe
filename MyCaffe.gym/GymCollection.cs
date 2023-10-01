using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.gym
{
    /// <summary>
    /// The GymCollection contains the available Gyms.
    /// </summary>
    public class GymCollection : IEnumerable<IXMyCaffeGym>
    {
        List<IXMyCaffeGym> m_rgGym = new List<IXMyCaffeGym>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public GymCollection()
        {
        }

        /// <summary>
        /// Loads the default and dynamic gyms.
        /// </summary>
        /// <remarks>
        /// Each dynamic Gym must implement a DLL with the IXMyCaffeGym interface implemented.  When loading dynamic
        /// Gym's, this class looks for these DLL's in the \code ./CustomGyms \endcode directory relative to the
        /// location of the MyCaffe.gym assembly.</remarks>
        /// <returns>A list of errors occuring while loading Gyms is returned if any occur.</returns>
        public List<Exception> Load()
        {
            m_rgGym.Add(new CartPoleGym());
            m_rgGym.Add(new AtariGym());
            m_rgGym.Add(new CurveGym());
            m_rgGym.Add(new DataGeneralGym());
            m_rgGym.Add(new ModelGym());

            return addCustomGyms();
        }

        private List<Exception> addCustomGyms()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string path = Uri.UnescapeDataString(uri.Path);
            string strPath = Path.GetDirectoryName(path);
            List<Exception> rgErr = new List<Exception>();

            strPath += "\\CustomGyms";

            if (Directory.Exists(strPath))
            {
                string[] rgstrFiles = Directory.GetFiles(strPath);

                foreach (string strFile in rgstrFiles)
                {
                    FileInfo fi = new FileInfo(strFile);

                    if (fi.Extension.ToLower() == ".dll")
                    {
                        Exception excpt;
                        IXMyCaffeGym igym = loadCustomGym(strFile, out excpt);
                        if (igym != null)
                        {
                            m_rgGym.Add(igym);
                        }
                        else if (excpt != null)
                        {
                            rgErr.Add(excpt);
                        }
                    }
                }
            }

            return rgErr;
        }

        private IXMyCaffeGym loadCustomGym(string strFile, out Exception err)
        {
            err = null;

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
                        if (strIface.Contains("IXMyCaffeGym"))
                        {
                            object obj = Activator.CreateInstance(t);
                            return obj as IXMyCaffeGym;
                        }
                    }
                }

                return null;
            }
            catch (Exception excpt)
            {
                if (excpt is System.Reflection.ReflectionTypeLoadException)
                {
                    var typeLoadException = excpt as ReflectionTypeLoadException;
                    var loaderExceptions = typeLoadException.LoaderExceptions;

                    if (loaderExceptions != null && loaderExceptions.Length > 0)
                        excpt = new Exception("Gym '" + strFile + "' failed to load!", loaderExceptions[0]);
                }

                err = excpt;

                return null;
            }
        }

        /// <summary>
        /// Search for a given Gym by its name.
        /// </summary>
        /// <param name="strName">Specifies the name of the Gym to look for.</param>
        /// <returns>If found the Gym IXMyCaffeGym interface is returned, otherwise <i>null</i> is returned.</returns>
        public IXMyCaffeGym Find(string strName)
        {
            int nPos = strName.IndexOf(':');
            if (nPos > 0)
                strName = strName.Substring(0, nPos);

            foreach (IXMyCaffeGym igym in m_rgGym)
            {
                if (igym.Name == strName)
                    return igym;
            }

            return null;
        }

        /// <summary>
        /// Find the first Gym that has all names within the list of names contained within its name.
        /// </summary>
        /// <param name="rgstrNames">Specifies the list of names to look for.</param>
        /// <returns>If found the Gym is returned, otherwise null.</returns>
        public IXMyCaffeGym Find(string[] rgstrNames)
        {
            List<IXMyCaffeGym> rgGym = new List<IXMyCaffeGym>(m_rgGym);

            foreach (string strName in rgstrNames)
            {
                if (string.IsNullOrEmpty(strName))
                    continue;

                if (strName == "MyCaffe")
                    continue;

                for (int i = rgGym.Count - 1; i >= 0; i--)
                {
                    if (!rgGym[i].Name.Contains(strName))
                        rgGym.RemoveAt(i);
                }
            }

            if (rgGym.Count == 0)
                return null;

            return rgGym[0];
        }

        /// <summary>
        /// Returns the collections enumerator.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<IXMyCaffeGym> GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }

        /// <summary>
        /// Returns the collections enumerator.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgGym.GetEnumerator();
        }
    }
}
