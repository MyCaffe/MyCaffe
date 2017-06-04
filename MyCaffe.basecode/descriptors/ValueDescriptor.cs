using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ValueDescriptor class contains the description of a single value.
    /// </summary>
    [Serializable]
    public class ValueDescriptor : BaseDescriptor
    {
        double m_dfValue;

        /// <summary>
        /// The ValueDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="dfVal">Specifies the value of the item.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        public ValueDescriptor(int nID, string strName, double dfVal, string strOwner)
            : base(nID, strName, strOwner)
        {
            m_dfValue = dfVal;
        }

        /// <summary>
        /// The ValueDescriptor constructor.
        /// </summary>
        /// <param name="v">Specifies another ValueDescriptor used to create this one.</param>
        public ValueDescriptor(ValueDescriptor v)
            : this(v.ID, v.Name, v.Value, v.Owner)
        {
        }

        /// <summary>
        /// Returns the value of the item.
        /// </summary>
        public double Value
        {
            get { return m_dfValue; }
        }
    }
}
