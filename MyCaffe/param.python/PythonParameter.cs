using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.python
{
    /// <summary>
    /// Specifies the parameters for the PythonLayer.
    /// </summary>
    /// <remarks>
    /// @see [Example calling Python from C#](https://stackoverflow.com/questions/138918/calling-python-from-c-sharp)
    /// @see [doWork](https://github.com/MyCaffe/MyCaffe/blob/7462fcb217b8247912d0beb0ccce7088125c4bdb/MyCaffe.app/FormGptTest.cs#L85)
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class PythonParameter : LayerParameterBase
    {
        string m_strPythonPath = "$Default$";
        
        /** @copydoc LayerParameterBase */
        public PythonParameter()
        {
        }

        /// <summary>
        /// Specifies the path to the Python runtime.
        /// </summary>
        /// <remarks>
        /// Specifying the default value of '$Default$' will use the default Python runtime path at:
        /// <code>
        /// string strPythonPath = @"C:\Users\" + strUserName + @"\AppData\Local\Programs\Python\Python39\python39.dll"
        /// </code>
        /// </remarks>
        [Description("Specifies the path to the Python runtime.")]
        public string python_path
        {
            get { return m_strPythonPath; }
            set { m_strPythonPath = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PythonParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PythonParameter p = (PythonParameter)src;
            m_strPythonPath = p.m_strPythonPath;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PythonParameter p = new PythonParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("python_path", python_path);

            return new RawProto(strName, "", rgChildren);
        }
        
        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PythonParameter FromProto(RawProto rp)
        {
            string strVal;
            PythonParameter p = new PythonParameter();

            if ((strVal = rp.FindValue("python_path")) != null)
                p.python_path = strVal;

            return p;
        }
    }
}
