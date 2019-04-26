using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param.ui
{
    /// <summary>
    /// The CsvConverter is used to display string lists on a single line in a property grid.
    /// </summary>
    public class CsvConverter : TypeConverter
    {
        /// <summary>
        /// Overrides the ConvertTo method of TypeConverter.
        /// </summary>
        /// <param name="context">Specifies the context.</param>
        /// <param name="culture">Specifies the culture.</param>
        /// <param name="value">Specifies the value.</param>
        /// <param name="destinationType">Specifies the distination type.</param>
        /// <returns>The converted type is returned.</returns>
        public override object ConvertTo(ITypeDescriptorContext context,
           CultureInfo culture, object value, Type destinationType)
        {
            List<String> v = value as List<String>;

            if (destinationType == typeof(string) && v != null)
            {
                return String.Join(",", v.ToArray());
            }

            return base.ConvertTo(context, culture, value, destinationType);
        }
    }
}
