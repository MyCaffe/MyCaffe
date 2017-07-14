using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
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
        public DatasetDescriptor(int nID, string strName, GroupDescriptor grpModel, GroupDescriptor grpDs, SourceDescriptor srcTrain, SourceDescriptor srcTest, string strCreatorName, string strDescription, string strOwner = null)
            : base(nID, strName, strOwner)
        {
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
            : this(d.ID, d.Name, d.ModelGroup, d.DatasetGroup, d.TrainingSource, d.TestingSource, d.CreatorName, d.Description, d.Owner)
        {
        }

        /// <summary>
        /// Copy another DatasetDesciptor into this one.
        /// </summary>
        /// <param name="ds">Specifies the DatasetDesciptor to copy.</param>
        public void Copy(DatasetDescriptor ds)
        {
            base.Copy(ds);

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
    }
}
