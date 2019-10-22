using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ProjectDescriptor class contains all information describing a project, such as its: dataset, group, settings, solver description, model description, and parmeters.
    /// </summary>
    [Serializable]
    public class ProjectDescriptor : BaseDescriptor
    {
        DatasetDescriptor m_dataset;
        DatasetDescriptor m_datasetTarget = null;
        GroupDescriptor m_group;
        SettingsCaffe m_settings = new SettingsCaffe();
        string m_strSolverName;
        string m_strSolverDescription;
        string m_strModelName;
        string m_strModelDescription;
        bool m_bActive = true;
        int m_nTotalIterations;
        string m_strGpuOverride;
        ParameterDescriptorCollection m_rgParameters = new ParameterDescriptorCollection();
        ValueDescriptorCollection m_rgAnalysisItems = new ValueDescriptorCollection();

        /// <summary>
        /// The ProjectDescriptor constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the item.</param>
        public ProjectDescriptor(string strName)
            : base(0, strName, null)
        {
        }

        /// <summary>
        /// The ProjectDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="ds">Specifies the dataset used.</param>
        /// <param name="grp">Specifies the project group.</param>
        /// <param name="strSolverName">Specifies the solver name.</param>
        /// <param name="strSolverDesc">Specifies the solver description script.</param>
        /// <param name="strModelName">Specifies the model name.</param>
        /// <param name="strModelDesc">Specifies the model description script.</param>
        /// <param name="strGpuOverride">Specifies the GPU ID's to use as an override.</param>
        /// <param name="nTotalIterations">Specifies the total number of iterations.</param>
        /// <param name="bActive">Specifies whether or not the project is active.</param>
        /// <param name="settings">Specifies the settings to use.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        public ProjectDescriptor(int nID, string strName, DatasetDescriptor ds, GroupDescriptor grp, string strSolverName, string strSolverDesc, string strModelName, string strModelDesc, string strGpuOverride, int nTotalIterations, bool bActive, SettingsCaffe settings, string strOwner)
            : base(nID, strName, strOwner)
        {
            if (settings != null)
                m_settings = settings.Clone();

            if (ds != null)
                m_dataset = new DatasetDescriptor(ds);

            if (grp != null)
                m_group = new GroupDescriptor(grp);
            else
                m_group = new GroupDescriptor(0, null, null);

            m_strSolverName = strSolverName;
            m_strSolverDescription = strSolverDesc;
            m_strModelName = strModelName;
            m_strModelDescription = strModelDesc;
            m_bActive = bActive;
            m_nTotalIterations = nTotalIterations;
            m_strGpuOverride = strGpuOverride;
        }

        /// <summary>
        /// The ProjectDescriptor constructor.
        /// </summary>
        /// <param name="p">Specifies another ProjectDescriptor used to create this one.</param>
        public ProjectDescriptor(ProjectDescriptor p)
            : this(p.ID, p.Name, p.Dataset, p.Group, p.m_strSolverName, p.SolverDescription, p.m_strModelName, p.ModelDescription, p.GpuOverride, p.TotalIterations, p.Active, p.Settings, p.Owner)
        {
            m_datasetTarget = p.DatasetTarget;
        }

        /// <summary>
        /// Get/set the GPU ID's to use as an override.
        /// </summary>
        public virtual string GpuOverride
        {
            get { return m_strGpuOverride; }
            set { m_strGpuOverride = value; }
        }

        /// <summary>
        /// Get/set the dataset used.
        /// </summary>
        [Browsable(false)]
        public DatasetDescriptor Dataset
        {
            get { return m_dataset; }
            set { m_dataset = value; }
        }


        /// <summary>
        /// Get/set the secondary 'target' dataset (if used).
        /// </summary>
        [Browsable(false)]
        public DatasetDescriptor DatasetTarget
        {
            get { return m_datasetTarget; }
            set { m_datasetTarget = value; }
        }

        /// <summary>
        /// Get/set the project group.
        /// </summary>
        [Description("Specifies the group of the project.")]
        public GroupDescriptor Group
        {
            get { return m_group; }
            set { m_group = value; }
        }

        /// <summary>
        /// Get/set the solver name.
        /// </summary>
        [Description("Specifies the name of the solver used by the project.")]
        [ReadOnly(true)]
        public string SolverName
        {
            get { return m_strSolverName; }
            set { m_strSolverName = value; }
        }

        /// <summary>
        /// Get/set the solver description script.
        /// </summary>
        [Browsable(false)]
        [ReadOnly(true)]
        public string SolverDescription
        {
            get { return m_strSolverDescription; }
            set { m_strSolverDescription = value; }
        }

        /// <summary>
        /// Get/set the model name.
        /// </summary>
        [Description("Specifies the name of the model used by the project.")]
        [ReadOnly(true)]
        public string ModelName
        {
            get { return m_strModelName; }
            set { m_strModelName = value; }
        }

        /// <summary>
        /// Get/set the model description script.
        /// </summary>
        [Browsable(false)]
        [ReadOnly(true)]
        public string ModelDescription
        {
            get { return m_strModelDescription; }
            set { m_strModelDescription = value; }
        }

        /// <summary>
        /// Get/set the total iterations.
        /// </summary>
        [ReadOnly(true)]
        [Category("Performance"), Description("Specifies the total number of iterations run on this project.")]
        public int TotalIterations
        {
            get { return m_nTotalIterations; }
            set { m_nTotalIterations = value; }
        }

        /// <summary>
        /// Returns whether or not the project is active.
        /// </summary>
        [Description("Specifies whether or not the project is active.")]
        [ReadOnly(true)]
        public bool Active
        {
            get { return m_bActive; }
        }

        /// <summary>
        /// Returns the collection of parameters of the Project.
        /// </summary>
        [Description("Specifies parameters associated with the project.")]
        public ParameterDescriptorCollection Parameters
        {
            get { return m_rgParameters; }
        }

        /// <summary>
        /// Returns the collection of analysis ValueDescriptors of the Project.
        /// </summary>
        [Category("Performance"), Description("Contains a list of analysis items for the project.")]
        public ValueDescriptorCollection AnalysisItems
        {
            get { return m_rgAnalysisItems; }
        }

        /// <summary>
        /// Get/set the settings of the Project.
        /// </summary>
        [Browsable(false)]
        public SettingsCaffe Settings
        {
            get { return m_settings; }
            set { m_settings = value; }
        }

        /// <summary>
        /// Creates the string representation of the descriptor.
        /// </summary>
        /// <returns>The string representation of the descriptor is returned.</returns>
        public override string ToString()
        {
            return Name + ": model = " + m_strModelName + " solver = " + m_strSolverName;
        }
    }
}
