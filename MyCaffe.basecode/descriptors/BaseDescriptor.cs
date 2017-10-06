using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The descriptors namespace contains all descriptor used to describe various items stored within the database.
/// </summary>
namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The BaseDescriptor is the base class for all other descriptors, where descriptors are used to describe various items stored
    /// within the database.
    /// </summary>
    [Serializable]
    public class BaseDescriptor
    {
        int m_nID;
        string m_strName;
        string m_strOwner;

        /// <summary>
        /// The BaseDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        public BaseDescriptor(int nID, string strName, string strOwner)
        {
            m_nID = nID;
            m_strName = strName;
        }

        /// <summary>
        /// The BaseDescriptor constructor.
        /// </summary>
        /// <param name="b">Specifies another BaseDescriptor used to create this one.</param>
        public BaseDescriptor(BaseDescriptor b)
            : this(b.ID, b.Name, b.Owner)
        {
        }

        /// <summary>
        /// Copy another BaseDescriptor into this one.
        /// </summary>
        /// <param name="b">Specifies the BaseDescriptor to copy.</param>
        public void Copy(BaseDescriptor b)
        {
            m_nID = b.ID;
            m_strName = b.Name;
            m_strOwner = b.Owner;
        }

        /// <summary>
        /// Get/set the database ID of the item.
        /// </summary>
        [ReadOnly(true)]
        public int ID
        {
            get { return m_nID; }
            set { m_nID = value; }
        }

        /// <summary>
        /// Get/set the name of the item.
        /// </summary>
        [ReadOnly(true)]
        public string Name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        /// <summary>
        /// Get/set the owner of the item.
        /// </summary>
        [ReadOnly(true)]
        [Browsable(false)]
        public string Owner
        {
            get { return m_strOwner; }
            set { m_strOwner = value; }
        }
    }
}
