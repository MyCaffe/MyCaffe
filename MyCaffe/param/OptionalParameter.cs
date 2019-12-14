using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param
{
    /// <summary>
    /// The OptionalParameter is the base class for parameters that are optional such as the MaskParameter, DistorationParameter, ExpansionParameter, NoiseParameter, and ResizeParameter.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class OptionalParameter
    {
        bool m_bActive = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active (default = false).</param>
        public OptionalParameter(bool bActive = false)
        {
            m_bActive = bActive;
        }

        /// <summary>
        /// When active, the parameter is used, otherwise it is ignored.
        /// </summary>
        public bool Active
        {
            get { return m_bActive; }
            set { m_bActive = value; }
        }
    }
}
