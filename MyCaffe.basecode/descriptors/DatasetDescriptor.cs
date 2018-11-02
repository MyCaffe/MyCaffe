using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The DatasetDescriptor class describes a dataset which contains both a training data source and testing data source.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DatasetDescriptor : BaseDescriptor
    {
        SourceDescriptor m_srcTest;
        SourceDescriptor m_srcTrain;
        GroupDescriptor m_groupDataset;
        GroupDescriptor m_groupModel;
        ParameterDescriptorCollection m_colParameters = new ParameterDescriptorCollection();
        string m_strCreatorName;
        string m_strDescription;
        GYM_TYPE m_gymType = GYM_TYPE.NONE;

        /// <summary>
        /// The DatasetDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="grpModel">Specifies the group of the model.</param>
        /// <param name="grpDs">Specifies the group of the dataset.</param>
        /// <param name="srcTrain">Specifies the data source for training.</param>
        /// <param name="srcTest">Specifies the data source for testing.</param>
        /// <param name="strCreatorName">Specifies the dataset creator name.</param>
        /// <param name="strDescription">Specifies a description of the dataset.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        /// <param name="gymType">Optionally, specifies the gym type to use if any (default = NONE).</param>
        public DatasetDescriptor(int nID, string strName, GroupDescriptor grpModel, GroupDescriptor grpDs, SourceDescriptor srcTrain, SourceDescriptor srcTest, string strCreatorName, string strDescription, string strOwner = null, GYM_TYPE gym = GYM_TYPE.NONE)
            : base(nID, strName, strOwner)
        {
            m_gymType = gym;

            if (grpModel != null)
                m_groupModel = new descriptors.GroupDescriptor(grpModel);
            else
                m_groupModel = new descriptors.GroupDescriptor(0, null, null);

            if (grpDs != null)
                m_groupDataset = new descriptors.GroupDescriptor(grpDs);
            else
                m_groupDataset = new descriptors.GroupDescriptor(0, null, null);

            if (srcTest != null)
                m_srcTest = new SourceDescriptor(srcTest);

            if (srcTrain != null)
                m_srcTrain = new SourceDescriptor(srcTrain);

            m_strDescription = strDescription;
            m_strCreatorName = strCreatorName;
        }

        /// <summary>
        /// The DatasetDescriptor constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the item.</param>
        public DatasetDescriptor(string strName)
            : this(0, strName, null, null, null, null, null, "")
        {
        }

        /// <summary>
        /// The DatasetDescriptor constructor.
        /// </summary>
        /// <param name="d">Specifies another DatasetDesciptor used to create this one.</param>
        public DatasetDescriptor(DatasetDescriptor d)
            : this(d.ID, d.Name, d.ModelGroup, d.DatasetGroup, d.TrainingSource, d.TestingSource, d.CreatorName, d.Description, d.Owner, d.GymType)
        {
            m_colParameters = new ParameterDescriptorCollection(d.Parameters);
        }

        /// <summary>
        /// Resize the testing and training data sources.
        /// </summary>
        /// <param name="nChannels">Specifies the new channel size.</param>
        /// <param name="nHeight">Specifies the new height size.</param>
        /// <param name="nWidth">Specifies the new width size.</param>
        public void Resize(int nChannels, int nHeight, int nWidth)
        {
            m_srcTest.Resize(nChannels, nHeight, nWidth);
            m_srcTrain.Resize(nChannels, nHeight, nWidth);
        }

        /// <summary>
        /// Copy another DatasetDesciptor into this one.
        /// </summary>
        /// <param name="ds">Specifies the DatasetDesciptor to copy.</param>
        public void Copy(DatasetDescriptor ds)
        {
            base.Copy(ds);

            m_gymType = ds.m_gymType;

            if (ds.m_srcTest != null)
                m_srcTest = new SourceDescriptor(ds.m_srcTest);
            else
                m_srcTest = null;

            if (ds.m_srcTrain != null)
                m_srcTrain = new SourceDescriptor(ds.m_srcTrain);
            else
                m_srcTrain = null;

            if (ds.m_groupDataset != null)
                m_groupDataset = new GroupDescriptor(ds.m_groupDataset);
            else
                m_groupDataset = null;

            if (ds.m_groupModel != null)
                m_groupModel = new GroupDescriptor(ds.m_groupModel);
            else
                m_groupModel = null;

            m_colParameters = new descriptors.ParameterDescriptorCollection();
            foreach (ParameterDescriptor p in ds.m_colParameters)
            {
                m_colParameters.Add(new ParameterDescriptor(p));
            }

            m_strCreatorName = ds.m_strCreatorName;
            m_strDescription = ds.m_strDescription;
        }

        /// <summary>
        /// Returns whether or not this dataset is from a Gym.
        /// </summary>
        public bool IsGym
        {
            get { return (m_gymType == GYM_TYPE.NONE) ? false : true; }
        }

        /// <summary>
        /// Returns the Gym type, if any.
        /// </summary>
        public GYM_TYPE GymType
        {
            get { return m_gymType; }
        }

        /// <summary>
        /// Returns the full name which returns 'GYM:Name:Type' when using a gym based dataset, otherwise just 'Name' is returned.
        /// </summary>
        public string FullName
        {
            get { return (IsGym) ? "GYM:" + Name : Name; }
        }

        /// <summary>
        /// Returns whether or not the name is from a gym.
        /// </summary>
        /// <param name="str">Specifies the name.</param>
        /// <returns>If the name is from a gym, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public static bool IsGymName(string str)
        {
            if (str.IndexOf("GYM:") == 0)
                return true;

            return false;
        }

        /// <summary>
        /// Returns the actual gym name by parsing off the 'GYM:' if it exists.
        /// </summary>
        /// <param name="str">Specifies the name.</param>
        /// <param name="strType">Specifies the type.</param>
        /// <returns>The actual gym name is returned.</returns>
        public static string GetGymName(string str, out string strType)
        {
            strType = null;

            int nPos = str.IndexOf("GYM:");
            if (nPos < 0)
                return str;

            str = str.Substring(4);

            nPos = str.IndexOf(':');
            if (nPos < 0)
                return str;

            strType = str.Substring(nPos + 1);
            return str.Substring(0, nPos);
        }

        /// <summary>
        /// Returns the dataset group.
        /// </summary>
        [Category("Groups"), Description("Specifies the dataset group (if any).")]
        public GroupDescriptor DatasetGroup
        {
            get { return m_groupDataset; }
        }

        /// <summary>
        /// Get/set the dataset model group.
        /// </summary>
        [Category("Groups"), Description("Specifies the model group (if any).")]
        public GroupDescriptor ModelGroup
        {
            get { return m_groupModel; }
            set { m_groupModel = value; }
        }

        /// <summary>
        /// Get/set the training data source.
        /// </summary>
        [Category("Sources"), Description("Specifies the data source used when training.")]
        public SourceDescriptor TrainingSource
        {
            get { return m_srcTrain; }
            set { m_srcTrain = value; }
        }

        /// <summary>
        /// Get/set the testing data source.
        /// </summary>
        [Category("Sources"), Description("Specifies the data source used when testing.")]
        public SourceDescriptor TestingSource
        {
            get { return m_srcTest; }
            set { m_srcTest = value; }
        }

        /// <summary>
        /// Returns the training source name, or <i>null</i> if not specifies.
        /// </summary>
        [Browsable(false)]
        public string TrainingSourceName
        {
            get { return (m_srcTrain == null) ? null : m_srcTrain.Name; }
        }

        /// <summary>
        /// Returns the testing source name or <i>null</i> if not specified.
        /// </summary>
        [Browsable(false)]
        public string TestingSourceName
        {
            get { return (m_srcTest == null) ? null : m_srcTest.Name; }
        }

        /// <summary>
        /// Returns the dataset creator name.
        /// </summary>
        [Description("Specifies the name of the creator used to create this dataset.")]
        public string CreatorName
        {
            get { return m_strCreatorName; }
        }

        /// <summary>
        /// Get/set the description of the Dataset.
        /// </summary>
        [Description("Specifies the description of this dataset.")]
        public string Description
        {
            get { return m_strDescription; }
            set { m_strDescription = value; }
        }

        /// <summary>
        /// Get/set the dataset parameters (if any).
        /// </summary>
        [Description("Specifies the parameters of the data set (if any).")]
        public ParameterDescriptorCollection Parameters
        {
            get { return m_colParameters; }
            set { m_colParameters = value; }
        }

        /// <summary>
        /// Serialize a dataset descriptor to a byte array.
        /// </summary>
        /// <param name="ds">Specifies the dataset descriptor to serialize.</param>
        /// <returns>A byte array containing the serialized dataset is returned.</returns>
        public static byte[] Serialize(DatasetDescriptor ds)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                BinaryFormatter bf = new BinaryFormatter();
                bf.Serialize(ms, ds);
                ms.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Deserialize a dataset descriptor from a byte array.
        /// </summary>
        /// <param name="rg">Specifies the byte array.</param>
        /// <returns>The deserialized dataset descriptor is returned.</returns>
        public static DatasetDescriptor Deserialize(byte[] rg)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            {
                BinaryFormatter bf = new BinaryFormatter();
                return bf.Deserialize(ms) as DatasetDescriptor;
            }
        }
    }
}
