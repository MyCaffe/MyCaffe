using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param
{
    /// <summary>
    /// The DataDebugParameter is used by the DataParameter when the 'enable_debugging' = True.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DataDebugParameter
    {
        int m_nIterations = 1;
        string m_strDebugDataSavePath = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public DataDebugParameter()
        {
        }

        /// <summary>
        /// (/b optional, default = 1) Specifies the number of iterations to output debug information.
        /// </summary>
        [Description("Optionally, specifies the number of iterations for which to output debug information (default = 1)")]
        public int iterations
        {
            get { return m_nIterations; }
            set { m_nIterations = value; }
        }

        /// <summary>
        /// (/b optional, default = null) Specifies the path where the debug data images are saved, otherwise is ignored when null.  This setting is only used for debugging.
        /// </summary>
        [Description("Specifies the path where the debug data images are saved (default = null, which ignores this setting).")]
        public string debug_save_path
        {
            get { return m_strDebugDataSavePath; }
            set { m_strDebugDataSavePath = value; }
        }

        private string debug_save_path_persist
        {
            get
            {
                string strPath = Utility.Replace(m_strDebugDataSavePath, ':', ';');
                return Utility.Replace(strPath, ' ', '~');
            }

            set
            {
                string strPath = Utility.Replace(value, ';', ':');
                m_strDebugDataSavePath = Utility.Replace(strPath, '~', ' ');
            }
        }

        /// <summary>
        /// Copies the specified source data noise parameter to this one.
        /// </summary>
        /// <param name="src">Specifies the source data noise parameter.</param>
        public void Copy(DataDebugParameter src)
        {
            if (src == null)
                return;

            m_nIterations = src.m_nIterations;
            m_strDebugDataSavePath = src.m_strDebugDataSavePath;
        }

        /// <summary>
        /// Convert the DataDebugParameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the RawProto name.</param>
        /// <returns>The RawProto containing the settings is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("iterations", m_nIterations.ToString());
            rgChildren.Add("debug_data_path", debug_save_path_persist);

            return new RawProto(strName, "", rgChildren);
        }


        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DataDebugParameter FromProto(RawProto rp, DataDebugParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new DataDebugParameter();

            if ((strVal = rp.FindValue("iterations")) != null)
                p.iterations = int.Parse(strVal);

            if ((strVal = rp.FindValue("debug_data_path")) != null)
                p.debug_save_path_persist = strVal;

            return p;
        }
    }
}
