using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The GroupDescriptor class defines a group.
    /// </summary>
    [Serializable]
    public class GroupDescriptor : BaseDescriptor
    {
        /// <summary>
        /// The GroupDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        public GroupDescriptor(int nID, string strName, string strOwner)
            : base(nID, strName, strOwner)
        {
        }

        /// <summary>
        /// The GroupDescriptor constructor.
        /// </summary>
        /// <param name="g">Specifies another GroupDescriptor used to create this one.</param>
        public GroupDescriptor(GroupDescriptor g)
            : this(g.ID, g.Name, g.Owner)
        {
        }

        /// <summary>
        /// Returns the string representation of the GroupDescriptor.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            if (String.IsNullOrEmpty(Name))
                return null;

            return "Group " + Name;
        }
    }
}
