using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using MyCaffe.basecode;

/// <summary>
/// The MyCaffe.param namespace contains all parameter objects that correspond to the native C++ %Caffe prototxt file.
/// </summary>
namespace MyCaffe.basecode
{
    /// <summary>
    /// The BaseParameter class is the base class for all other parameter classes.
    /// </summary>
    public abstract class BaseParameter
    {
        /// <summary>
        /// Constructor for the parameter.
        /// </summary>
        public BaseParameter()
        {
            // For international versions of Windows, force decimal to '.' instead of ','
            // for parsing prototxt.
            if (Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator != ".")
            {
                string strCultureName = Thread.CurrentThread.CurrentCulture.Name;
                CultureInfo cinfo = new CultureInfo(strCultureName);
                cinfo.NumberFormat.NumberDecimalSeparator = ".";
                Thread.CurrentThread.CurrentCulture = cinfo;
            }
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public abstract RawProto ToProto(string strName);

        /// <summary>
        /// Compare this parameter to another parameter.
        /// </summary>
        /// <param name="p">Specifies the other parameter to compare with this one.</param>
        /// <returns>Returns <i>true</i> if the two parameters are the same, <i>false</i> otherwise.</returns>
        public virtual bool Compare(BaseParameter p)
        {
            RawProto p1 = ToProto("foo");
            RawProto p2 = p.ToProto("foo");
            string str1 = p1.ToString();
            string str2 = p2.ToString();

            return str1 == str2;
        }
    }
}
