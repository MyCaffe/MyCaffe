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
        static CultureInfo cultureUS = null;

        /// <summary>
        /// Constructor for the parameter.
        /// </summary>
        public BaseParameter()
        {
        }

        /// <summary>
        /// Parse double values using the US culture if the decimal separator = '.', then using the native culture, and if then 
        /// lastly trying the US culture to handle prototypes containing '.' as the separator, yet parsed in a culture that does
        /// not use '.' as a decimal.
        /// </summary>
        /// <param name="strVal">Specifies the value to parse.</param>
        /// <returns>The double value is returned.</returns>
        public static double ParseDouble(string strVal)
        {
            if (Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator == "." || string.IsNullOrEmpty(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return double.Parse(strVal);

            if (strVal.Contains(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return double.Parse(strVal);

            if (cultureUS == null)
                cultureUS = CultureInfo.CreateSpecificCulture("en-US");

            return double.Parse(strVal, cultureUS);
        }

        /// <summary>
        /// Parse double values using the US culture if the decimal separator = '.', then using the native culture, and if then 
        /// lastly trying the US culture to handle prototypes containing '.' as the separator, yet parsed in a culture that does
        /// not use '.' as a decimal.
        /// </summary>
        /// <param name="strVal">Specifies the value to parse.</param>
        /// <param name="df">Returns the double value parsed.</param>
        /// <returns>Returns <i>true</i> on a successful parse.</returns>
        public static bool TryParse(string strVal, out double df)
        {
            if (Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator == "." || string.IsNullOrEmpty(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return double.TryParse(strVal, out df);

            if (strVal.Contains(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return double.TryParse(strVal, out df);

            if (cultureUS == null)
                cultureUS = CultureInfo.CreateSpecificCulture("en-US");

            return double.TryParse(strVal, NumberStyles.Any, cultureUS, out df);
        }

        /// <summary>
        /// Parse float values using the US culture if the decimal separator = '.', then using the native culture, and if then 
        /// lastly trying the US culture to handle prototypes containing '.' as the separator, yet parsed in a culture that does
        /// not use '.' as a decimal.
        /// </summary>
        /// <param name="strVal">Specifies the value to parse.</param>
        /// <returns>The float value is returned.</returns>
        public static float ParseFloat(string strVal)
        {
            if (Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator == "." || string.IsNullOrEmpty(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return float.Parse(strVal);

            if (strVal.Contains(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return float.Parse(strVal);

            if (cultureUS == null)
                cultureUS = CultureInfo.CreateSpecificCulture("en-US");

            return float.Parse(strVal, cultureUS);
        }

        /// <summary>
        /// Parse doufloatble values using the US culture if the decimal separator = '.', then using the native culture, and if then 
        /// lastly trying the US culture to handle prototypes containing '.' as the separator, yet parsed in a culture that does
        /// not use '.' as a decimal.
        /// </summary>
        /// <param name="strVal">Specifies the value to parse.</param>
        /// <param name="f">Returns the float value parsed.</param>
        /// <returns>Returns <i>true</i> on a successful parse.</returns>
        public static bool TryParse(string strVal, out float f)
        {
            if (Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator == "." || string.IsNullOrEmpty(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return float.TryParse(strVal, out f);

            if (strVal.Contains(Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator))
                return float.TryParse(strVal, out f);

            if (cultureUS == null)
                cultureUS = CultureInfo.CreateSpecificCulture("en-US");

            return float.TryParse(strVal, NumberStyles.Any, cultureUS, out f);
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
