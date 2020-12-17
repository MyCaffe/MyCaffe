using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ProjectEx class manages a project containing the solver description, model description, data set (with training data source and testing data source) and
    /// project results.
    /// </summary>
    public class ProjectEx
    {
        ProjectDescriptor m_project;
        StateDescriptor m_state;
        RawProto m_protoModel = null;
        RawProto m_protoSolver = null;
        bool m_bExistTest = false;
        bool m_bExistTrain = false;
        bool m_bDatasetAdjusted = false;
        bool m_bDefaultSaveImagesToFile = true;
        Stage m_stage = Stage.NONE;
        int m_nOriginalProjectID = 0;

        /// <summary>
        /// The OverrrideModel event fires each time the SetDataset function is called.
        /// </summary>
        public event EventHandler<OverrideProjectArgs> OnOverrideModel;
        /// <summary>
        /// The OverrideSolver event fires each time the SetDataset function is called.
        /// </summary>
        public event EventHandler<OverrideProjectArgs> OnOverrideSolver;

        /// <summary>
        /// The ProjectEx constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the project.</param>
        /// <param name="strDsName">Optionally, specifies the name of the dataset used by the project.</param>
        public ProjectEx(string strName, string strDsName = null)
        {
            m_project = new ProjectDescriptor(strName);
            m_project.Dataset = new descriptors.DatasetDescriptor(strDsName);
            m_state = new StateDescriptor(0, strName + " results", m_project.Owner);
        }

        /// <summary>
        /// The ProjectEx constructor.
        /// </summary>
        /// <param name="prj">Specifies the project descriptor for the project.</param>
        /// <param name="state">Specifies the state descriptor for the project.</param>
        /// <param name="bExistTrain">Specifies whether or not training results exist for the proejct.</param>
        /// <param name="bExistTest">Specifies whether or not testing results exist for the project.</param>
        /// <param name="bQueryModel">Optionally, specifies whether or not to set (and parse) the model.</param>
        /// <param name="bQuerySolver">Optionally, specifies whether or not to set (and parse) the solver.</param>
        public ProjectEx(ProjectDescriptor prj, StateDescriptor state = null, bool bExistTrain = false, bool bExistTest = false, bool bQueryModel = true, bool bQuerySolver = true)
        {
            m_project = prj;

            if (state == null)
                state = new StateDescriptor(0, prj.Name + " results", m_project.Owner);

            m_state = state;

            if (bQueryModel)    
                ModelDescription = prj.ModelDescription;
            else
                m_project.ModelName = getModelName(prj.ModelDescription);

            if (bQuerySolver)
                SolverDescription = prj.SolverDescription;
            else
                m_project.SolverName = getSolverType(prj.SolverDescription);

            m_bExistTest = bExistTest;
            m_bExistTrain = bExistTrain;
        }

        /// <summary>
        /// Returns whether or not the data criteria is required by the current project model (e.g. the model contains an AnnotatedData layer).
        /// </summary>
        /// <returns>If the model requires the data criteria be loaded, then <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool RequiresDataCriteria()
        {
            RawProtoCollection col = m_protoModel.FindChildren("layer");
            foreach (RawProto layer in col)
            {
                RawProto type = layer.FindChild("type");
                if (type.Value.ToLower() == "annotateddata")
                    return true;
            }

            return false;
        }

        private string parse(string str, string strTarget, string strDefault = "UNKNOWN")
        {
            if (str == null)
                return strDefault;

            int nPos1 = 0;

            while (nPos1 < str.Length)
            {
                nPos1 = str.IndexOf(strTarget, nPos1);
                if (nPos1 < 0)
                    return strDefault;

                if (nPos1 == 0 || char.IsWhiteSpace(str[nPos1 - 1]) || str[nPos1 - 1] == '\n' || str[nPos1 - 1] == '\r')
                    break;

                nPos1++;
            }

            if (nPos1 >= str.Length)
                return strDefault;

            nPos1 += strTarget.Length;

            while (nPos1 < str.Length && (char.IsWhiteSpace(str[nPos1]) || str[nPos1] == '\"'))
            {
                nPos1++;
            }

            string strName;

            int nPos2 = str.IndexOfAny(new char[] { ' ', '\n', '\r', '\"', '\t' }, nPos1);
            if (nPos2 < 0)
                strName = str.Substring(nPos1);
            else
                strName = str.Substring(nPos1, nPos2 - nPos1).Trim(' ', '\"');

            return strName;
        }

        private string getModelName(string strDesc)
        {
            string strName = parse(strDesc, "name:");
            return strName;
        }

        private string getSolverType(string strDesc)
        {
            string strName = parse(strDesc, "type:", "SGD");
            return strName;
        }

        private void setDatasetFromProto(RawProto proto)
        {
            RawProtoCollection col = proto.FindChildren("layer");
            string strSrcTest = null;
            string strSrcTrain = null;
            string strSrcTest2 = null;
            string strSrcTrain2 = null;

            foreach (RawProto rp in col)
            {
                RawProto protoType = rp.FindChild("type");
                if (protoType != null && protoType.Value == "Data")
                {
                    RawProto protoParam = rp.FindChild("data_param");
                    if (protoParam != null)
                    {
                        bool bPrimary = true;

                        RawProto protoPrimary = protoParam.FindChild("primary_data");
                        if (protoPrimary != null)
                            bPrimary = bool.Parse(protoPrimary.Value);

                        RawProto protoSrc = protoParam.FindChild("source");
                        if (protoSrc != null)
                        {
                            RawProto protoInclude = rp.FindChild("include");
                            if (protoInclude != null)
                            {
                                RawProto protoPhase = protoInclude.FindChild("phase");
                                if (protoPhase != null)
                                {
                                    if (bPrimary)
                                    {
                                        if (protoPhase.Value == "TRAIN")
                                            strSrcTrain = protoSrc.Value;
                                        else if (protoPhase.Value == "TEST")
                                            strSrcTest = protoSrc.Value;
                                    }
                                    else
                                    {
                                        if (protoPhase.Value == "TRAIN")
                                            strSrcTrain2 = protoSrc.Value;
                                        else if (protoPhase.Value == "TEST")
                                            strSrcTest2 = protoSrc.Value;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (strSrcTest != null)
            {
                bool bSaveImagesToFile = (m_project.Dataset.TestingSource != null) ? m_project.Dataset.TestingSource.SaveImagesToFile : m_bDefaultSaveImagesToFile;
                m_project.Dataset.TestingSource = new SourceDescriptor(strSrcTest, bSaveImagesToFile);
            }

            if (strSrcTrain != null)
            {
                bool bSaveImagesToFile = (m_project.Dataset.TrainingSource != null) ? m_project.Dataset.TrainingSource.SaveImagesToFile : m_bDefaultSaveImagesToFile;
                m_project.Dataset.TrainingSource = new SourceDescriptor(strSrcTrain, bSaveImagesToFile);
            }

            if (strSrcTest2 != null || strSrcTrain2 != null)
            {
                if (m_project.DatasetTarget == null)
                    m_project.DatasetTarget = new DatasetDescriptor(m_project.Dataset.Name + "_tgt");

                if (strSrcTest2 != null)
                {
                    bool bSaveImagesToFile = (m_project.DatasetTarget.TestingSource != null) ? m_project.DatasetTarget.TestingSource.SaveImagesToFile : m_bDefaultSaveImagesToFile;
                    m_project.Dataset.TestingSource = new SourceDescriptor(strSrcTest2, bSaveImagesToFile);
                }

                if (strSrcTrain2 != null)
                {
                    bool bSaveImagesToFile = (m_project.DatasetTarget.TrainingSource != null) ? m_project.DatasetTarget.TrainingSource.SaveImagesToFile : m_bDefaultSaveImagesToFile;
                    m_project.Dataset.TrainingSource = new SourceDescriptor(strSrcTrain2, bSaveImagesToFile);
                }
            }
        }

        private void setDatasetToProto(RawProto proto)
        {
            RawProtoCollection col = proto.FindChildren("layer");
            string strSrcTest = m_project.Dataset.TestingSourceName;
            string strSrcTrain = m_project.Dataset.TrainingSourceName;
            string strSrcTest2 = (m_project.DatasetTarget != null) ? m_project.DatasetTarget.TestingSourceName : null;
            string strSrcTrain2 = (m_project.DatasetTarget != null ) ? m_project.DatasetTarget.TrainingSourceName : null;

            foreach (RawProto rp in col)
            {
                RawProto protoType = rp.FindChild("type");
                if (protoType != null && (protoType.Value == "Data" || protoType.Value == "AnnotatedData"))
                {
                    RawProto protoParam = rp.FindChild("data_param");
                    if (protoParam != null)
                    {
                        bool bPrimary = true;

                        RawProto protoPrimary = protoParam.FindChild("primary_data");
                        if (protoPrimary != null)
                            bPrimary = bool.Parse(protoPrimary.Value);

                        RawProto protoSrc = protoParam.FindChild("source");
                        if (protoSrc != null)
                        {
                            RawProto protoInclude = rp.FindChild("include");
                            if (protoInclude != null)
                            {
                                RawProto protoPhase = protoInclude.FindChild("phase");
                                if (protoPhase != null)
                                {
                                    if (bPrimary)
                                    {
                                        if (protoPhase.Value == "TRAIN")
                                        {
                                            if (strSrcTrain != null)
                                                protoSrc.Value = strSrcTrain;
                                        }
                                        else if (protoPhase.Value == "TEST")
                                        {
                                            if (strSrcTest != null)
                                                protoSrc.Value = strSrcTest;
                                        }
                                    }
                                    else
                                    {
                                        if (protoPhase.Value == "TRAIN")
                                        {
                                            if (strSrcTrain2 != null)
                                                protoSrc.Value = strSrcTrain2;
                                        }
                                        else if (protoPhase.Value == "TEST")
                                        {
                                            if (strSrcTest2 != null)
                                                protoSrc.Value = strSrcTest2;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Get/set whether or not the dataset for the project has been changed.
        /// </summary>
        public bool DatasetAdjusted
        {
            get { return m_bDatasetAdjusted; }
            set { m_bDatasetAdjusted = value; }
        }

        /// <summary>
        /// Returns the custom trainer and properties used by the project (if any).
        /// </summary>
        /// <param name="strProperties">Specifies the properties associated with the custom trainer.  The properties are stored in the solver parameter field 'custom_trainer_propeties' as a list of comma ('=') separated key value pairs each separated by ';'</param>
        /// <remarks>An example set of properties uses the following format: key1=val1;key2=val2;...</remarks>
        /// <returns>The custom trainer name is returned.</returns>
        public string GetCustomTrainer(out string strProperties)
        {
            if (m_protoSolver == null)
                SolverDescription = m_project.SolverDescription;

            strProperties = "";

            RawProto rp = m_protoSolver.FindChild("custom_trainer");
            if (rp == null)
                return null;

            if (rp.Value == null || rp.Value.Length == 0)
                return null;

            RawProto rprop = m_protoSolver.FindChild("custom_trainer_properties");
            if (rprop != null)
                strProperties = rprop.Value;

            return rp.Value;
        }

        private Phase getPhase(RawProto rp)
        {
            RawProto rpInc = rp.FindChild("include");
            if (rpInc == null)
                return Phase.NONE;

            RawProto rpPhase = rpInc.FindChild("phase");
            if (rpPhase == null)
                return Phase.NONE;

            string strPhase = rpPhase.Value.ToUpper();

            if (strPhase == Phase.TEST.ToString())
                return Phase.TEST;

            if (strPhase == Phase.TRAIN.ToString())
                return Phase.TRAIN;

            return Phase.NONE;
        }

        /// <summary>
        /// Returns the batch size of the project used in a given Phase.
        /// </summary>
        /// <param name="phase">Specifies the Phase to use.</param>
        /// <returns>The batch size is returned.</returns>
        public int GetBatchSize(Phase phase)
        {
            if (m_protoModel == null)
                ModelDescription = m_project.ModelDescription;

            RawProtoCollection col = m_protoModel.FindChildren("layer");

            foreach (RawProto rp1 in col)
            {
                Phase p = getPhase(rp1);

                if (p == phase || phase == Phase.NONE)
                {
                    RawProto rp = rp1.FindChild("batch_data_param");

                    if (rp == null)
                        rp = rp1.FindChild("data_param");

                    if (rp == null)
                        rp = rp1.FindChild("memory_data_param");

                    if (rp != null)
                    {
                        rp = rp.FindChild("batch_size");

                        if (rp == null)
                            return 0;

                        return int.Parse(rp.Value);
                    }
                }
            }

            return 0;
        }

        /// <summary>
        /// Returns the setting of a Layer (if found).
        /// </summary>
        /// <param name="phase">Specifies the Phase to use.</param>
        /// <param name="strLayer">Specifies the Layer name.</param>
        /// <param name="strParam">Specifies the Layer setting name to look for.</param>
        /// <returns>If found the setting value is returned, otherwise <i>null</i> is returned.</returns>
        public double? GetLayerSetting(Phase phase, string strLayer, string strParam)
        {
            if (m_protoModel == null)
                ModelDescription = m_project.ModelDescription;

            RawProtoCollection col = m_protoModel.FindChildren("layer");

            foreach (RawProto rp1 in col)
            {
                Phase p = getPhase(rp1);

                if (p == phase || phase == Phase.NONE)
                {
                    RawProto rp = rp1.FindChild(strLayer);

                    if (rp != null)
                    {
                        rp = rp.FindChild(strParam);

                        if (rp == null)
                            return null;

                        return BaseParameter.ParseDouble(rp.Value);
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Get a setting from the solver descriptor.
        /// </summary>
        /// <param name="strParam">Specifies the setting to retrieve.</param>
        /// <returns>The setting is returned if found, otherwise <i>null</i> is returned.</returns>
        public string GetSolverSetting(string strParam)
        {
            if (m_protoSolver == null)
                SolverDescription = m_project.SolverDescription;

            RawProto proto = m_protoSolver.FindChild(strParam);
            if (proto == null)
                return null;

            return proto.Value;
        }

        /// <summary>
        /// Get a setting from the solver descriptor as a double value.
        /// </summary>
        /// <param name="strParam">Specifies the setting to retrieve.</param>
        /// <returns>The setting is returned as a double if found, otherwise <i>null</i> is returned.</returns>
        public double? GetSolverSettingAsNumeric(string strParam)
        {
            string strVal = GetSolverSetting(strParam);
            if (strVal == null)
                return null;

            double dfVal;
            if (!BaseParameter.TryParse(strVal, out dfVal))
                return null;

            return dfVal;
        }

        /// <summary>
        /// Get a setting from the solver descriptor as an integer value.
        /// </summary>
        /// <param name="strParam">Specifies the setting to retrieve.</param>
        /// <returns>The setting is returned as a int if found, otherwise <i>null</i> is returned.</returns>
        public int? GetSolverSettingAsInt(string strParam)
        {
            double? dfVal = GetSolverSettingAsNumeric(strParam);
            if (!dfVal.HasValue)
                return null;

            return (int)dfVal.Value;
        }

        /// <summary>
        /// Get a setting from the solver descriptor as a boolean value.
        /// </summary>
        /// <param name="strParam">Specifies the setting to retrieve.</param>
        /// <returns>The setting is returned as a bool if found, otherwise <i>null</i> is returned.</returns>
        public bool? GetSolverSettingAsBool(string strParam)
        {
            string strVal = GetSolverSetting(strParam);
            if (strVal == null)
                return null;

            return bool.Parse(strVal);
        }

        /// <summary>
        /// Get/set the Caffe setting to use with the Project.
        /// </summary>
        public SettingsCaffe Settings
        {
            get { return m_project.Settings; }
            set { m_project.Settings = value; }
        }

        /// <summary>
        /// Get/set the name of the Project.
        /// </summary>
        public string Name
        {
            get { return m_project.Name; }
            set { m_project.Name = value; }
        }

        /// <summary>
        /// Returns the ID of the Project in the database.
        /// </summary>
        public int ID
        {
            get { return m_project.ID; }
        }

        /// <summary>
        /// Get/set the original project ID.
        /// </summary>
        public int OriginalID
        {
            get
            {
                if (m_nOriginalProjectID > 0)
                    return m_nOriginalProjectID;

                return ID;
            }
            set
            {
                m_nOriginalProjectID = value;
            }
        }

        /// <summary>
        /// Get/set the ID of the Project owner.
        /// </summary>
        public string Owner
        {
            get { return m_project.Owner; }
            set { m_project.Owner = value; }
        }

        /// <summary>
        /// Returns whether or not the Project is active.
        /// </summary>
        public bool Active
        {
            get { return m_project.Active; }
        }

        /// <summary>
        /// Returns the training category of the project, or NONE if no custom trainer is used.
        /// </summary>
        public TRAINING_CATEGORY TrainingCategory
        {
            get
            {
                if (m_protoSolver == null)
                    return TRAINING_CATEGORY.NONE;

                string strCustomTrainer = GetSolverSetting("custom_trainer");
                if (string.IsNullOrEmpty(strCustomTrainer))
                    return TRAINING_CATEGORY.NONE;

                if (strCustomTrainer == "RL.Trainer")
                    return TRAINING_CATEGORY.REINFORCEMENT;

                if (strCustomTrainer == "RNN.Trainer")
                    return TRAINING_CATEGORY.RECURRENT;

                if (strCustomTrainer == "Dual.Trainer")
                    return TRAINING_CATEGORY.DUAL;

                return TRAINING_CATEGORY.CUSTOM;
            }
        }

        /// <summary>
        /// Return the stage under which the project was opened.
        /// </summary>
        public Stage Stage
        {
            get { return m_stage; }
            set { m_stage = value; }
        }

        /// <summary>
        /// Get/set the super boost probability used by the Project.
        /// </summary>
        public double SuperBoostProbability
        {
            get { return (double)m_project.Settings.SuperBoostProbability; }
            set { m_project.Settings.SuperBoostProbability = value; }
        }

        /// <summary>
        /// Returns whether or not the Project uses the training data source when testing (default = <i>false</i>).
        /// </summary>
        public bool UseTrainingSourceForTesting
        {
            get { return m_project.Parameters.Find("UseTrainingSourceForTesting", false); }
        }

        /// <summary>
        /// Returns whether or not label balancing is enabled.  When enabled, first the label set is randomly selected and then the image
        /// is selected from the label set using the image selection criteria (e.g. Random).
        /// </summary>
        public bool EnableLabelBalancing
        {
            get { return m_project.Settings.EnableLabelBalancing; }
        }

        /// <summary>
        /// Returns whether or not label boosting is enabled.  When using Label boosting, images are selected from boosted labels with 
        /// a higher probability that images from other label sets.
        /// </summary>
        public bool EnableLabelBoosting
        {
            get { return m_project.Settings.EnableLabelBoosting; }
        }

        /// <summary>
        /// Returns whether or not random image selection is enabled.  When enabled, images are randomly selected from the entire set, or 
        /// randomly from a label set when label balancing is in effect.
        /// </summary>
        public bool EnableRandomSelection
        {
            get { return m_project.Settings.EnableRandomInputSelection; }
        }

        /// <summary>
        /// Returns whether or not pair selection is enabled.  When using pair selection, images are queried in pairs where the first query selects
        /// the image based on the image selection criteria (e.g. Random), and then the second image query returns the image just following the 
        /// first image in the database.
        /// </summary>
        public bool EnablePairSelection
        {
            get { return m_project.Settings.EnablePairInputSelection; }
        }

        /// <summary>
        /// Returns the list of comma separated GPU ID's that are to be used when training this Project.
        /// </summary>
        public string GpuOverride
        {
            get { return m_project.GpuOverride; }
        }

        /// <summary>
        /// Returns the method used to load the images into memory.  Loading all images into memory has the highest training performance for 
        /// memory access is much faster than disk acces (even with an SSD).
        /// </summary>
        public IMAGEDB_LOAD_METHOD ImageLoadMethod
        {
            get { return m_project.Settings.ImageDbLoadMethod; }
        }

        /// <summary>
        /// Returns the image load limit.
        /// </summary>
        public int ImageLoadLimit
        {
            get { return m_project.Settings.ImageDbLoadLimit; }
        }

        /// <summary>
        /// Returns the image load limit refresh period in milliseconds.
        /// </summary>
        public int ImageLoadLimitRefreshPeriod
        {
            get { return m_project.Settings.ImageDbAutoRefreshScheduledUpdateInMs; }
        }

        /// <summary>
        /// Returns the image load limit refresh percentage (to update).
        /// </summary>
        public double ImageLoadLimitRefreshPercent
        {
            get { return m_project.Settings.ImageDbAutoRefreshScheduledReplacementPercent; }
        }

        /// <summary>
        /// Returns the snapshot weight update favor.  The snapshot can favor an improving accuracy, decreasing error, or both when saving weights.
        /// </summary>
        /// <remarks>
        /// Note, weights updates are saved separately from the entire solver state that is snapshot on regular intervals.
        /// </remarks>
        public SNAPSHOT_WEIGHT_UPDATE_METHOD SnapshotWeightUpdateMethod
        {
            get { return m_project.Settings.SnapshotWeightUpdateMethod; }
        }

        /// <summary>
        /// Returns the snapshot load method.  When loading the best error or accuracy, the snapshot loaded may not be the last one taken.
        /// </summary>
        public SNAPSHOT_LOAD_METHOD SnapshotLoadMethod
        {
            get { return m_project.Settings.SnapshotLoadMethod; }
        }

        /// <summary>
        /// Get/set the solver description script used by the Project.
        /// </summary>
        public string SolverDescription
        {
            get { return (m_protoSolver == null) ? null : m_protoSolver.ToString(); }
            set
            {
                m_project.SolverName = getSolverType(value);
                m_project.SolverDescription = value;
                m_protoSolver = null;

                if (value != null && value.Length > 0)
                {
                    m_protoSolver = RawProto.Parse(value);

                    if (m_project.Dataset != null)
                    {
                        if (string.IsNullOrEmpty(m_project.Dataset.Name))
                            setDatasetFromProto(m_protoSolver);
                        else
                            setDatasetToProto(m_protoSolver);
                    }

                    RawProto rpType = m_protoSolver.FindChild("type");
                    if (rpType != null)
                        m_project.SolverName = rpType.Value;
                }
            }
        }

        /// <summary>
        /// Get/set the model description script used by the Project.
        /// </summary>
        public string ModelDescription
        {
            get { return (m_protoModel == null) ? null : m_protoModel.ToString(); }
            set
            {
                m_project.ModelName = getModelName(value);
                m_project.ModelDescription = value;
                m_protoModel = null;

                if (value != null && value.Length > 0)
                {
                    m_protoModel = RawProto.Parse(value);

                    if (m_project.Dataset != null)
                    {
                        if (string.IsNullOrEmpty(m_project.Dataset.Name))
                            setDatasetFromProto(m_protoModel);
                        else
                            setDatasetToProto(m_protoModel);
                    }

                    RawProto rpName = m_protoModel.FindChild("name");
                    if (rpName != null)
                        m_project.ModelName = rpName.Value;
                }
            }
        }

        /// <summary>
        /// Return the project group descriptor of the group that the Project resides (if any).
        /// </summary>
        public GroupDescriptor ProjectGroup
        {
            get { return m_project.Group; }
        }

        /// <summary>
        /// Return the model group descriptor of the group that the Project participates in (if any).
        /// </summary>
        public GroupDescriptor ModelGroup
        {
            get { return m_project.Dataset.ModelGroup; }
        }

        /// <summary>
        /// Return the dataset group descriptor of the group that the Project participates in (if any).
        /// </summary>
        public GroupDescriptor DatasetGroup
        {
            get { return m_project.Dataset.DatasetGroup; }
        }

        /// <summary>
        /// Returns any project parameters that may exist (if any).
        /// </summary>
        public ParameterDescriptorCollection Parameters
        {
            get { return m_project.Parameters; }
        }

        /// <summary>
        /// Get/set the total number of iterations that the Project has been trained.
        /// </summary>
        public int TotalIterations
        {
            get { return m_project.TotalIterations; }
            set { m_project.TotalIterations = value; }
        }

        /// <summary>
        /// Return whether or not the project has results from a training session.
        /// </summary>
        public bool HasResults
        {
            get { return m_state.HasResults; }
        }

        /// <summary>
        /// Get/set the current number of iterations that the Project has been trained.
        /// </summary>
        public int Iterations
        {
            get { return m_state.Iterations; }
            set { m_state.Iterations = value; }
        }

        /// <summary>
        /// Get/set the best accuracy observed while testing the Project.
        /// </summary>
        public double BestAccuracy
        {
            get { return m_state.Accuracy; }
            set { m_state.Accuracy = value; }
        }

        /// <summary>
        /// Get/set the best error observed while training the Project.
        /// </summary>
        public double BestError
        {
            get { return m_state.Error; }
            set { m_state.Error = value; }
        }

        /// <summary>
        /// Get/set the solver state.
        /// </summary>
        public byte[] SolverState
        {
            get { return m_state.State; }
            set { m_state.State = value; }
        }

        /// <summary>
        /// Get/set the weight state.
        /// </summary>
        public byte[] WeightsState
        {
            get { return m_state.Weights; }
            set { m_state.Weights = value; }
        }

        /// <summary>
        /// Return the name of the dataset used.
        /// </summary>
        public string DatasetName
        {
            get
            {
                if (m_project.Dataset != null)
                    return m_project.Dataset.Name;

                return null;
            }
        }

        /// <summary>
        /// Return the descriptor of the dataset used.
        /// </summary>
        public DatasetDescriptor Dataset
        {
            get { return m_project.Dataset; }
        }

        /// <summary>
        /// Returns the target dataset (if exists) or <i>null</i> if it does not.
        /// </summary>
        /// <remarks>
        /// The target dataset only applies when using both a source and target dataset.
        /// </remarks>
        public DatasetDescriptor DatasetTarget
        {
            get { return m_project.DatasetTarget; }
        }

        /// <summary>
        /// Get/set the dataset ID of the target dataset (if exists), otherwise return 0.
        /// </summary>
        public int TargetDatasetID
        {
            get
            {
                ParameterDescriptor p = m_project.Parameters.Find("TargetDatasetID");
                if (p == null)
                    return 0;

                int nID;
                if (!int.TryParse(p.Value, out nID))
                    return 0;

                return nID;
            }

            set
            {
                ParameterDescriptor p = m_project.Parameters.Find("TargetDatasetID");
                if (p == null)
                    m_project.Parameters.Add(new ParameterDescriptor(0, "TargetDatasetID", value.ToString()));
                else
                    p.Value = value.ToString();
            }
        }

        /// <summary>
        /// Return whether or not testing results exist.
        /// </summary>
        public bool ExistTestResults
        {
            get { return m_bExistTest; }
        }

        /// <summary>
        /// Return whether or not training results exist.
        /// </summary>
        public bool ExistTrainResults
        {
            get { return m_bExistTrain; }
        }

        /// <summary>
        /// Return Project performance metrics.
        /// </summary>
        public ValueDescriptorCollection ProjectPerformanceItems
        {
            get { return m_project.AnalysisItems; }
        }

        /// <summary>
        /// Return the name of the model used by the Project.
        /// </summary>
        public string ModelName
        {
            get { return m_project.ModelName; }
        }

        /// <summary>
        /// Return the type of the Solver used by the Project.
        /// </summary>
        public string SolverType
        {
            get { return m_project.SolverName; }
        }

        /// <summary>
        /// Set a given Solver variable in the solver description script.
        /// </summary>
        /// <param name="strVar">Specifies the variable name.</param>
        /// <param name="strVal">Specifies the variable value.</param>
        /// <returns>If the variable is found and set, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool SetSolverVariable(string strVar, string strVal)
        {
            if (m_protoSolver != null)
            {
                RawProto protoVar = m_protoSolver.FindChild(strVar);

                if (protoVar != null)
                    protoVar.Value = strVal;
                else
                    m_protoSolver.Children.Add(new RawProto(strVar, strVal));

                m_project.SolverDescription = m_protoSolver.ToString();

                return true;
            }

            return false;
        }

        /// <summary>
        /// Load the solver description from a file.
        /// </summary>
        /// <param name="strFile">Specifies the solver file.</param>
        public void LoadSolverFile(string strFile)
        {
            using (StreamReader sr = new StreamReader(strFile))
            {
                SolverDescription = sr.ReadToEnd();
            }
        }

        /// <summary>
        /// Load the model description from a file.
        /// </summary>
        /// <param name="strFile">Specifies the model file.</param>
        public void LoadModelFile(string strFile)
        {
            using (StreamReader sr = new StreamReader(strFile))
            {
                ModelDescription = sr.ReadToEnd();
            }
        }

        /// <summary>
        /// Create a model description as a RawProto for running the Project.
        /// </summary>
        /// <param name="strName">Specifies the model name.</param>
        /// <param name="nNum">Specifies the batch size to use.</param>
        /// <param name="nChannels">Specifies the number of channels of each item in the batch.</param>
        /// <param name="nHeight">Specifies the height of each item in the batch.</param>
        /// <param name="nWidth">Specifies the width of each item in the batch.</param>
        /// <param name="protoTransform">Returns a RawProto describing the Data Transformation parameters to use.</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <param name="bSkipLossLayer">Optionally, specifies to skip the loss layer and not output a converted layer to replace it (default = false).</param>
        /// <returns>The RawProto of the model description is returned.</returns>
        public RawProto CreateModelForRunning(string strName, int nNum, int nChannels, int nHeight, int nWidth, out RawProto protoTransform, Stage stage = Stage.NONE, bool bSkipLossLayer = false)
        {
            return CreateModelForRunning(m_project.ModelDescription, strName, nNum, nChannels, nHeight, nWidth, out protoTransform, stage, bSkipLossLayer);
        }

        /// <summary>
        /// Create a model description as a RawProto for training the Project.
        /// </summary>
        /// <param name="strModelDescription">Specifies the model description.</param>
        /// <param name="strName">Specifies the model name.</param>
        /// <param name="bCaffeFormat">Specifies whether or not the model description should use the native C++ caffe format where coloring is ordered in BGR, or use the MyCaffe format where coloring is ordered in RGB.</param>
        /// <returns>The RawProto of the model description is returned.</returns>
        public static RawProto CreateModelForTraining(string strModelDescription, string strName, bool bCaffeFormat = false)
        {
            RawProto proto = RawProto.Parse(strModelDescription);

            string strLayers = "layer";
            RawProtoCollection rgLayers = proto.FindChildren("layer");
            if (rgLayers.Count == 0)
            {
                rgLayers = proto.FindChildren("layers");
                strLayers = "layers";
            }

            bool bDirty = false;
            RawProtoCollection rgRemove = new RawProtoCollection();
            RawProto protoSoftmax = null;
            RawProto protoName = proto.FindChild("name");
            int nTrainDataLayerIdx = -1;
            int nTestDataLayerIdx = -1;
            int nSoftmaxLossLayerIdx = -1;
            int nAccuracyLayerIdx = -1;
            int nIdx = 0;

            foreach (RawProto layer in rgLayers)
            {
                bool bRemove = false;
                RawProto type = layer.FindChild("type");
                RawProto include = layer.FindChild("include");
                RawProto exclude = layer.FindChild("exclude");

                string strType = type.Value.ToLower();

                if (strType == "softmax")
                    protoSoftmax = layer;

                if (include != null)
                {
                    RawProto phase = include.FindChild("phase");
                    if (phase != null)
                    {
                        if (phase.Value != "TEST" && phase.Value != "TRAIN")
                            bRemove = true;
                        else
                        {
                            if (strType == "data")
                            {
                                if (phase.Value == "TRAIN")
                                    nTrainDataLayerIdx = nIdx;
                                else
                                    nTestDataLayerIdx = nIdx;
                            }
                            else if (strType == "accuracy")
                            {
                                nAccuracyLayerIdx = nIdx;
                            }
                            else if (strType == "softmaxwithloss")
                            {
                                nSoftmaxLossLayerIdx = nIdx;
                            }
                        }
                    }
                }

                if (!bRemove && exclude != null)
                {
                    RawProto phase = exclude.FindChild("phase");
                    if (phase != null)
                    {
                        if (phase.Value == "TEST" || phase.Value == "TRAIN")
                            bRemove = true;
                    }
                }

                if (bRemove)
                {
                    rgRemove.Add(layer);
                }

                nIdx++;
            }

            if (nTestDataLayerIdx < 0)
            {
                string strProto = getDataLayerProto(strLayers, strName, bCaffeFormat, 16, "", Phase.TEST);
                RawProto protoData = RawProto.Parse(strProto).Children[0];

                if (nTrainDataLayerIdx > 0)
                    rgLayers.Insert(nTrainDataLayerIdx + 1, protoData);
                else
                    rgLayers.Insert(0, protoData);

                bDirty = true;
            }

            if (nTrainDataLayerIdx < 0)
            {
                string strProto = getDataLayerProto(strLayers, strName, bCaffeFormat, 16, "", Phase.TRAIN);
                RawProto protoData = RawProto.Parse(strProto).Children[0];
                rgLayers.Insert(0, protoData);
                bDirty = true;
            }

            foreach (RawProto layer in rgRemove)
            {
                proto.RemoveChild(layer);
            }

            if (protoSoftmax != null)
            {
                RawProto type = protoSoftmax.FindChild("type");
                if (type != null)
                    type.Value = "SoftmaxWithLoss";

                protoSoftmax.Children.Add(new RawProto("bottom", "label"));
                protoSoftmax.Children.Add(new RawProto("loss_weight", "1", null, RawProto.TYPE.NUMERIC));

                string strInclude = "include { phase: TRAIN }";
                protoSoftmax.Children.Add(RawProto.Parse(strInclude).Children[0]);

                string strLoss = "loss_param { normalization: VALID }";
                protoSoftmax.Children.Add(RawProto.Parse(strLoss).Children[0]);
                bDirty = true;
            }

            if (nAccuracyLayerIdx < 0)
            {
                string strBottom = null;
                if (rgLayers.Count > 0)
                {
                    RawProto last = rgLayers[rgLayers.Count - 1];
                    RawProtoCollection colBtm = last.FindChildren("bottom");

                    if (colBtm.Count > 0)
                        strBottom = colBtm[0].Value;
                }

                if (strBottom != null)
                {
                    string strProto = getAccuracyLayerProto(strLayers, strBottom);
                    RawProto protoData = RawProto.Parse(strProto).Children[0];
                    rgLayers.Add(protoData);
                    bDirty = true;
                }
            }

            if (bDirty || proto.FindChildren("input_dim").Count > 0)
            {
                rgLayers.Insert(0, protoName);
                proto = new RawProto("root", null, rgLayers);
            }

            return proto;
        }

        private static string getDataLayerProto(string strLayer, string strName, bool bCaffeFormat, int nBatchSize, string strSrc, Phase phase)
        {
            string strRgb = (bCaffeFormat) ? "BGR" : "RGB";
            string strPhase = phase.ToString();
            return strLayer + " { name: \"" + strName + "\" type: \"Data\" top: \"data\" top: \"label\" include { phase: " + strPhase + " } transform_param { scale: 1 mirror: True use_imagedb_mean: True color_order: " + strRgb + " } data_param { source: \"" + strSrc + "\" batch_size: " + nBatchSize.ToString() + " backend: IMAGEDB enable_random_selection: True } }";
        }

        private static string getAccuracyLayerProto(string strLayer, string strBottom)
        {
            return strLayer + " { name: \"accuracy\" type: \"Accuracy\" bottom: \"" + strBottom + "\" bottom: \"label\" top: \"accuracy\" include { phase: TEST } accuracy_param { top_k: 1 } }";
        }

        private static PhaseStageCollection getPhases(RawProto proto, string strType)
        {
            PhaseStageCollection psCol = new PhaseStageCollection();

            RawProtoCollection type = proto.FindChildren(strType);
            if (type == null || type.Count == 0)
                return psCol;

            return getPhases(type);
        }

        private static PhaseStageCollection getPhases(RawProtoCollection col)
        {
            PhaseStageCollection psCol = new PhaseStageCollection();

            foreach (RawProto proto1 in col)
            {
                RawProto protoPhase = proto1.FindChild("phase");
                if (protoPhase == null)
                    continue;

                Stage stage = Stage.NONE;
                RawProto protoStage = proto1.FindChild("stage");
                if (protoStage != null)
                {
                    if (protoStage.Value == Stage.RL.ToString())
                        stage = Stage.RL;

                    else if (protoStage.Value == Stage.RNN.ToString())
                        stage = Stage.RNN;
                }

                Phase phase = Phase.NONE;
                if (protoPhase != null)
                {
                    if (protoPhase.Value == Phase.ALL.ToString())
                        phase = Phase.ALL;

                    else if (protoPhase.Value == Phase.RUN.ToString())
                        phase = Phase.RUN;

                    else if (protoPhase.Value == Phase.TEST.ToString())
                        phase = Phase.TEST;

                    else if (protoPhase.Value == Phase.TRAIN.ToString())
                        phase = Phase.TRAIN;
                }

                psCol.Add(phase, stage);
            }

            return psCol;
        }

        private static bool includeLayer(RawProto layer, Stage stage, out PhaseStageCollection psInclude, out PhaseStageCollection psExclude)
        {
            psInclude = getPhases(layer, "include");
            psExclude = getPhases(layer, "exlcude").FindAllWith(stage);

            PhaseStageCollection psInclude1 = psInclude.FindAllWith(Stage.NONE);
            PhaseStageCollection psInclude2 = psInclude.FindAllWith(stage);
            PhaseStageCollection psInclude3 = psInclude.FindAllWith(Phase.NONE, Phase.ALL, Phase.RUN);
            psExclude = psExclude.FindAllWith(Phase.RUN);

            if (psExclude.Count > 0)
                return false;

            if (psInclude.Count > 0)
            {
                if (psInclude3.Count == 0 || (psInclude1.Count == 0 && psInclude2.Count == 0))
                    return false;
            }

            return true;
        }


        /// <summary>
        /// Create a model description as a RawProto for running the Project.
        /// </summary>
        /// <param name="strModelDescription">Specifies the model description to use.</param>
        /// <param name="strName">Specifies the model name.</param>
        /// <param name="nNum">Specifies the batch size to use.</param>
        /// <param name="nChannels">Specifies the number of channels of each item in the batch.</param>
        /// <param name="nHeight">Specifies the height of each item in the batch.</param>
        /// <param name="nWidth">Specifies the width of each item in the batch.</param>
        /// <param name="protoTransform">Returns a RawProto describing the Data Transformation parameters to use.</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <param name="bSkipLossLayer">Optionally, specifies to skip the loss layer and not output a converted layer to replace it (default = false).</param>
        /// <param name="stage">Optionally, specifies the stage to create the run network on.</param>
        /// <returns>The RawProto of the model description is returned.</returns>
        public static RawProto CreateModelForRunning(string strModelDescription, string strName, int nNum, int nChannels, int nHeight, int nWidth, out RawProto protoTransform, Stage stage = Stage.NONE, bool bSkipLossLayer = false)
        {
            PhaseStageCollection psInclude;
            PhaseStageCollection psExclude;
            RawProto proto = RawProto.Parse(strModelDescription);
            int nNameIdx = proto.FindChildIndex("name");
            int nInputInsertIdx = -1;
            int nInputShapeInsertIdx = -1;
            bool bNoInput = false;

            protoTransform = null;

            nNameIdx++;
            if (nNameIdx < 0)
                nNameIdx = 0;

            List<Tuple<string, int, int, int, int>> rgInputs = new List<Tuple<string, int, int, int, int>>();
            rgInputs.Add(new Tuple<string, int, int, int, int>(strName, nNum, nChannels, nHeight, nWidth));

            RawProtoCollection rgLayers = proto.FindChildren("layer");
            bool bUsesLstm = false;

            foreach (RawProto layer in rgLayers)
            {
                RawProto type = layer.FindChild("type");
                if (type != null)
                {
                    string strType = type.Value.ToLower();
                    if (strType == "lstm")
                    {
                        bUsesLstm = true;
                        break;
                    }
                }
            }

            bool bFoundInput = false;
            bool bFoundMemoryData = false;

            foreach (RawProto layer in rgLayers)
            {
                RawProto type = layer.FindChild("type");
                if (type != null)
                {
                    string strType = type.Value.ToLower();
                    if (strType == "input")
                    {
                        bFoundInput = true;

                        if (includeLayer(layer, stage, out psInclude, out psExclude))
                        {
                            rgInputs.Clear();

                            RawProtoCollection rgTop = layer.FindChildren("top");
                            RawProto input_param = layer.FindChild("input_param");
                            if (input_param != null)
                            {
                                RawProtoCollection rgShape = input_param.FindChildren("shape");

                                if (rgTop.Count == rgShape.Count)
                                {
                                    for (int i = 0; i < rgTop.Count; i++)
                                    {
                                        if (bUsesLstm && i < 2)
                                        {
                                            RawProtoCollection rgDim = rgShape[i].FindChildren("dim");
                                            if (rgDim.Count > 1)
                                            {
                                                rgDim[1].Value = "1";
                                            }
                                        }

                                        if (rgTop[i].Value.ToLower() != "label")
                                        {
                                            List<int> rgVal = new List<int>();
                                            RawProtoCollection rgDim = rgShape[i].FindChildren("dim");
                                            foreach (RawProto dim in rgDim)
                                            {
                                                rgVal.Add(int.Parse(dim.Value));
                                            }

                                            nNum = (rgVal.Count > 0) ? rgVal[0] : 1;
                                            nChannels = (rgVal.Count > 1) ? rgVal[1] : 1;
                                            nHeight = (rgVal.Count > 2) ? rgVal[2] : 1;
                                            nWidth = (rgVal.Count > 3) ? rgVal[3] : 1;

                                            rgInputs.Add(new Tuple<string, int, int, int, int>(rgTop[i].Value, nNum, nChannels, nHeight, nWidth));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (strType == "memorydata")
                    {
                        bFoundMemoryData = true;

                        if (includeLayer(layer, stage, out psInclude, out psExclude))
                        {
                            bNoInput = true;
                            rgInputs.Clear();
                        }
                    }
                    else if (strType == "data")
                    {
                        if (rgInputs.Count > 0)
                        {
                            RawProtoCollection colTop = layer.FindChildren("top");
                            if (colTop.Count > 0)
                                rgInputs[0] = new Tuple<string, int, int, int, int>(colTop[0].Value, rgInputs[0].Item2, rgInputs[0].Item3, rgInputs[0].Item4, rgInputs[0].Item5);
                        }
                    }

                    if (bFoundInput && bFoundMemoryData)
                        break;
                }
            }

            RawProto input = null;
            RawProto input_shape = null;
            RawProtoCollection rgInput = null;
            RawProtoCollection rgInputShape = null;

            if (!bNoInput)
            {
                rgInput = new RawProtoCollection();
                rgInputShape = new RawProtoCollection();

                input = proto.FindChild("input");
                if (input != null)
                {
                    input.Value = rgInputs[0].Item1;
                }
                else
                {
                    for (int i = 0; i < rgInputs.Count; i++)
                    {
                        input = new RawProto("input", rgInputs[i].Item1, null, RawProto.TYPE.STRING);
                        rgInput.Add(input);
                        nInputInsertIdx = nNameIdx;
                        nNameIdx++;
                    }
                }

                input_shape = proto.FindChild("input_shape");
                if (input_shape != null)
                {
                    RawProtoCollection colDim = input_shape.FindChildren("dim");

                    if (colDim.Count > 0)
                        colDim[0].Value = rgInputs[0].Item2.ToString();

                    if (colDim.Count > 1)
                        colDim[1].Value = rgInputs[0].Item3.ToString();

                    if (colDim.Count > 2)
                        colDim[2].Value = rgInputs[0].Item4.ToString();

                    if (colDim.Count > 3)
                        colDim[3].Value = rgInputs[0].Item5.ToString();
                }
                else
                {
                    for (int i = 0; i < rgInputs.Count; i++)
                    {
                        input_shape = new RawProto("input_shape", "");

                        nNum = rgInputs[i].Item2;
                        nChannels = rgInputs[i].Item3;
                        nHeight = rgInputs[i].Item4;
                        nWidth = rgInputs[i].Item5;

                        input_shape.Children.Add(new RawProto("dim", nNum.ToString()));
                        input_shape.Children.Add(new RawProto("dim", nChannels.ToString()));

                        if (nHeight > 1 || nWidth > 1)
                        {
                            input_shape.Children.Add(new RawProto("dim", nHeight.ToString()));
                            input_shape.Children.Add(new RawProto("dim", nWidth.ToString()));
                        }

                        rgInputShape.Add(input_shape);
                        nInputShapeInsertIdx = nNameIdx;
                    }
                }
            }

            RawProto net_name = proto.FindChild("name");
            if (net_name != null)
                net_name.Value += "-Live";

            RawProtoCollection rgRemove = new RawProtoCollection();

            List<RawProto> rgProtoSoftMaxLoss = new List<basecode.RawProto>();
            RawProto protoSoftMax = null;

            foreach (RawProto layer in rgLayers)
            {
                RawProto type = layer.FindChild("type");
                if (type != null)
                {
                    string strType = type.Value.ToLower();
                    bool bKeepLayer = false;

                    bool bInclude = includeLayer(layer, stage, out psInclude, out psExclude);

                    if (strType == "data" || strType == "annotateddata" || strType == "batchdata")
                    {
                        if (psInclude.Find(Phase.TEST, stage) != null)
                            protoTransform = layer.FindChild("transform_param");
                    }
                    else if (strType == "decode")
                    {
                        List<RawProto> rgBtm = new List<RawProto>();

                        foreach (RawProto child in layer.Children)
                        {
                            if (child.Name == "bottom")
                                rgBtm.Add(child);
                        }

                        if (rgBtm.Count > 0)
                            rgBtm.RemoveAt(0);

                        foreach (RawProto btm in rgBtm)
                        {
                            layer.Children.Remove(btm);
                        }
                    }

                    if (!bInclude)
                    {
                        rgRemove.Add(layer);
                    }
                    else if (psExclude.Find(Phase.RUN, stage) != null)
                    {
                        rgRemove.Add(layer);
                    }
                    else if (strType == "input")
                    {
                        rgRemove.Add(layer);
                    }
                    else if (strType == "softmaxwithloss")
                    {
                        if (!bSkipLossLayer)
                        {
                            rgProtoSoftMaxLoss.Add(layer);
                            bKeepLayer = true;
                        }
                        else
                        {
                            rgRemove.Add(layer);
                        }
                    }
                    else if (strType == "memoryloss" ||
                             strType == "contrastive_loss" ||
                             strType == "contrastiveloss" ||
                             strType == "euclidean_loss" ||
                             strType == "euclideanloss" ||
                             strType == "hinge_loss" ||
                             strType == "hingeloss" ||
                             strType == "infogain_loss" ||
                             strType == "infogainloss" ||
                             strType == "multinomiallogistic_loss" ||
                             strType == "multinomiallogisticloss" ||
                             strType == "sigmoidcrossentropy_loss" ||
                             strType == "sigmoidcrossentropyloss" ||
                             strType == "softmaxcrossentropy_loss" ||
                             strType == "softmaxcrossentropyloss" ||
                             strType == "triplet_loss" ||
                             strType == "tripletloss" ||
                             strType == "triplet_loss_simple" ||
                             strType == "tripletlosssimple")
                    {
                        rgRemove.Add(layer);
                    }
                    else if (strType == "softmax")
                    {
                        protoSoftMax = layer;
                    }
                    else if (strType == "labelmapping")
                    {
                        rgRemove.Add(layer);
                    }
                    else if (strType == "binaryhash")
                    {
                        RawProtoCollection col = layer.FindChildren("bottom");
                        if (col.Count > 0)
                            layer.RemoveChild(col[col.Count - 1]);
                    }
                    else if (strType == "debug")
                    {
                        rgRemove.Add(layer);
                    }

                    if (!bKeepLayer && psInclude.FindAllWith(Phase.TEST, Phase.TRAIN).Count > 0 && psInclude.FindAllWith(Phase.RUN).Count == 0)
                    {
                        rgRemove.Add(layer);
                    }
                    else
                    {
                        RawProto max_btm = layer.FindChild("max_bottom_count");
                        if (max_btm != null)
                        {
                            RawProto phase1 = max_btm.FindChild("phase");
                            RawProto stage1 = max_btm.FindChild("stage");

                            if (phase1 != null && phase1.Value == "RUN" && (stage1 == null || stage1.Value == stage.ToString() || stage1.Value == Stage.NONE.ToString()))
                            {
                                RawProto count = max_btm.FindChild("count");
                                int nCount = int.Parse(count.Value);

                                int nBtmIdx = layer.FindChildIndex("bottom");
                                int nBtmEnd = layer.Children.Count;
                                List<int> rgRemoveIdx = new List<int>();

                                for (int i = nBtmIdx; i < layer.Children.Count; i++)
                                {
                                    if (layer.Children[i].Name != "bottom")
                                    {
                                        nBtmEnd = i;
                                        break;
                                    }
                                }

                                for (int i = nBtmEnd - 1; i >= nBtmIdx + nCount; i--)
                                {
                                    layer.Children.RemoveAt(i);
                                }
                            }
                        }
                    }
                }

                RawProto exclude = layer.FindChild("exclude");
                if (exclude != null)
                {
                    RawProto phase = exclude.FindChild("phase");
                    if (phase != null)
                    {
                        if (phase.Value == "RUN")
                        {
                            if (!rgRemove.Contains(layer))
                                rgRemove.Add(layer);
                        }
                    }
                }
            }

            foreach (RawProto protoSoftMaxLoss in rgProtoSoftMaxLoss)
            {
                if (protoSoftMax != null)
                {
                    rgRemove.Add(protoSoftMaxLoss);
                }
                else
                {
                    RawProto type = protoSoftMaxLoss.FindChild("type");
                    if (type != null)
                        type.Value = "Softmax";

                    protoSoftMaxLoss.RemoveChild("bottom", "label", true);
                }
            }

            foreach (RawProto layer in rgRemove)
            {
                proto.RemoveChild(layer);
            }

            if (input != null && input_shape != null)
            {
                if (protoTransform != null)
                {
                    RawProto resize = protoTransform.FindChild("resize_param");
                    
                    if (resize != null)
                    {
                        bool bActive = (bool)resize.FindValue("active", typeof(bool));
                        if (bActive)
                        {
                            int nNewHeight = (int)resize.FindValue("height", typeof(int));
                            int nNewWidth = (int)resize.FindValue("width", typeof(int));

                            if (rgInputShape[0].Children.Count < 1)
                                rgInputShape[0].Children.Add(new RawProto("dim", "1"));

                            if (rgInputShape[0].Children.Count < 2)
                                rgInputShape[0].Children.Add(new RawProto("dim", "1"));

                            if (rgInputShape[0].Children.Count < 3)
                                rgInputShape[0].Children.Add(new RawProto("dim", nNewHeight.ToString()));
                            else
                                rgInputShape[0].Children[2] = new RawProto("dim", nNewHeight.ToString());

                            if (rgInputShape[0].Children.Count < 4)
                                rgInputShape[0].Children.Add(new RawProto("dim", nNewWidth.ToString()));
                            else
                                rgInputShape[0].Children[3] = new RawProto("dim", nNewWidth.ToString());
                        }
                    }
                }

                for (int i = rgInputShape.Count - 1; i >= 0; i--)
                {
                    proto.Children.Insert(0, rgInputShape[i]);
                }

                for (int i = rgInput.Count - 1; i >= 0; i--)
                {
                    proto.Children.Insert(0, rgInput[i]);
                }
            }

            return proto;
        }

        /// <summary>
        /// Sets the dataset used by the Project, overriding the current dataset used.
        /// </summary>
        /// <remarks>
        /// Note, this function 'fixes' up the model used by the Project to use the new dataset.
        /// </remarks>
        /// <param name="dataset">Specifies the new dataset to use.</param>
        public void SetDataset(DatasetDescriptor dataset)
        {
            if (dataset == null)
                return;

            m_project.Dataset = dataset;

            if (m_project.ModelDescription != null && m_project.ModelDescription.Length > 0)
            {
                bool bResized = false;
                string strProto = SetDataset(m_project.ModelDescription, dataset, out bResized);
                RawProto proto = RawProto.Parse(strProto);

                if (OnOverrideModel != null)
                {
                    OverrideProjectArgs args = new OverrideProjectArgs(proto);
                    OnOverrideModel(this, args);
                    proto = args.Proto;
                }

                ModelDescription = proto.ToString();
            }

            if (m_project.SolverDescription != null && m_project.SolverDescription.Length > 0)
            {
                if (OnOverrideSolver != null)
                {
                    RawProto proto = RawProto.Parse(m_project.SolverDescription);
                    OverrideProjectArgs args = new OverrideProjectArgs(proto);
                    OnOverrideSolver(this, args);
                    proto = args.Proto;

                    SolverDescription = proto.ToString();
                }
            }
        }

        /// <summary>
        /// Sets the dataset of a model, overriding the current dataset used.
        /// </summary>
        /// <remarks>
        /// Note, this function 'fixes' up the model used by the Project to use the new dataset.
        /// </remarks>
        /// <param name="strModelDesc">Specifies the model description to update.</param>
        /// <param name="dataset">Specifies the new dataset to use.</param>
        /// <param name="bResized">Returns whether or not the model was resized with a different output size.</param>
        /// <param name="bUpdateOutputs">Optionally, specifies whether or not to update the number of outputs in the last layer (e.g. the number of classes in the dataset).</param>
        public static string SetDataset(string strModelDesc, DatasetDescriptor dataset, out bool bResized, bool bUpdateOutputs = false)
        {
            bResized = false;

            if (dataset == null)
                return null;

            string strTypeLast = null;
            RawProto protoLast = null;
            RawProto proto = RawProto.Parse(strModelDesc);
            List<RawProto> rgLastIp = new List<RawProto>();
            RawProtoCollection colLayers = proto.FindChildren("layer");
            RawProto protoDataTrainBatch = null;
            RawProto protoDataTestBatch = null;

            if (colLayers.Count == 0)
                colLayers = proto.FindChildren("layers");

            RawProtoCollection colIP = new RawProtoCollection();

            foreach (RawProto protoChild in colLayers)
            {
                RawProto type = protoChild.FindChild("type");
                RawProto name = protoChild.FindChild("name");

                string strType = type.Value.ToLower();

                if (strType == "data")
                {
                    int nCropSize = 0;

                    RawProto data_param = protoChild.FindChild("data_param");
                    if (data_param != null)
                    {
                        RawProto batchProto = data_param.FindChild("batch_size");

                        RawProto include = protoChild.FindChild("include");
                        if (include != null)
                        {
                            RawProto phase = include.FindChild("phase");
                            if (phase != null)
                            {
                                RawProto source = data_param.FindChild("source");
                                if (phase.Value == "TEST")
                                {
                                    protoDataTestBatch = batchProto;

                                    if (source != null)
                                    {
                                        source.Value = dataset.TestingSource.Name;
                                        nCropSize = dataset.TestingSource.ImageHeight;
                                    }
                                    else
                                    {
                                        data_param.Children.Add(new RawProto("source", dataset.TestingSource.Name, null, RawProto.TYPE.STRING));
                                    }
                                }
                                else
                                {
                                    protoDataTrainBatch = batchProto;

                                    if (source != null)
                                    {
                                        source.Value = dataset.TrainingSource.Name;
                                        nCropSize = dataset.TrainingSource.ImageHeight;
                                    }
                                    else
                                    {
                                        data_param.Children.Add(new RawProto("source", dataset.TrainingSource.Name, null, RawProto.TYPE.STRING));
                                    }
                                }
                            }
                        }
                    }

                    RawProto transform_param = protoChild.FindChild("transform_param");
                    if (transform_param != null)
                    {
                        RawProto crop_size = transform_param.FindChild("crop_size");
                        if (crop_size != null)
                        {
                            int nSize = int.Parse(crop_size.Value);

                            if (nCropSize != nSize)
                                crop_size.Value = nCropSize.ToString();
                        }
                    }
                }
                else if (strType.Contains("loss"))
                {
                    if (colIP.Count > 0)
                    {
                        rgLastIp.Add(colIP[0]);
                        colIP.Clear();
                    }
                }
                else if (strType == "inner_product" || strType == "innerproduct")
                {
                    colIP.Insert(0, protoChild);
                }

                protoLast = protoChild;
                strTypeLast = strType;
            }

            if (protoDataTestBatch != null && protoDataTrainBatch != null)
            {
                int nTestSize = int.Parse(protoDataTestBatch.Value);
                int nTrainSize = int.Parse(protoDataTrainBatch.Value);

                if (nTrainSize < nTestSize)
                    protoDataTrainBatch.Value = nTestSize.ToString();
            }

            if (bUpdateOutputs)
            {
                foreach (RawProto lastIp in rgLastIp)
                {
                    RawProto protoParam = lastIp.FindChild("inner_product_param");
                    if (protoParam != null)
                    {
                        RawProto protoNumOut = protoParam.FindChild("num_output");
                        if (protoNumOut != null)
                        {
                            int nNumOut = dataset.TrainingSource.Labels.Count;

                            if (nNumOut > 0)
                            {
                                protoNumOut.Value = nNumOut.ToString();
                                bResized = true;
                            }
                        }
                    }
                }
            }

            return proto.ToString();
        }

        /// <summary>
        /// This method searches for a given parameter within a given layer, optionally for a certain Phase.
        /// </summary>
        /// <remarks>
        /// An example usage may be: layer = 'data', param = 'data_param', field = 'source'
        /// </remarks>
        /// <param name="strModelDescription">Specifies the model description to search.</param>
        /// <param name="strLayerName">Specifies the name of the layer, when <i>null</i> only the layer type is used..</param>
        /// <param name="strLayerType">Specifies the type of the layer.</param>
        /// <param name="strParam">Specifies the name of the parameter, such as 'data_param'.</param>
        /// <param name="strField">Specifies the field of the parameter, such as 'source'.</param>
        /// <param name="phaseMatch">Optionally, specifies the phase.</param>
        /// <returns>If found, the parameter value is returned, otherwise <i>null</i> is returned.</returns>
        public static string FindLayerParameter(string strModelDescription, string strLayerName, string strLayerType, string strParam, string strField, Phase phaseMatch = Phase.NONE)
        {
            RawProto proto = RawProto.Parse(strModelDescription);

            RawProtoCollection rgLayers = proto.FindChildren("layer");
            RawProto firstFound = null;

            foreach (RawProto layer in rgLayers)
            {
                RawProto type = layer.FindChild("type");
                RawProto name = layer.FindChild("name");

                if (strLayerType == type.Value.ToString() && (strLayerName == null || name.Value.ToString() == strLayerName))
                {
                    if (phaseMatch != Phase.NONE)
                    {
                        RawProto include = layer.FindChild("include");

                        if (include != null)
                        {
                            RawProto phase = include.FindChild("phase");
                            if (phase != null)
                            {
                                if (phase.Value == phaseMatch.ToString())
                                {
                                    firstFound = layer;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            if (firstFound == null)
                                firstFound = layer;
                        }
                    }
                    else
                    {
                        if (firstFound == null)
                            firstFound = layer;
                    }
                }
            }

            if (firstFound == null)
                return null;

            RawProto child = null;

            if (strParam != null)
                child = firstFound.FindChild(strParam);

            if (child != null)
                firstFound = child;

            return firstFound.FindValue(strField);
        }

        /// <summary>
        /// Disables the testing interval so that no test passes are run.
        /// </summary>
        /// <returns>Returns <i>true</i> if the testing was disabled, false if it was not setup in the first place.</returns>
        public bool DisableTesting()
        {
            // Force parse the proto if not already parsed.
            if (m_protoSolver == null)
            {
                string strProto = SolverDescription;
                SolverDescription = strProto;
            }

            bool bSet = false;
            RawProto protoTestIter = m_protoSolver.FindChild("test_iter");
            RawProto protoTestInterval = m_protoSolver.FindChild("test_interval");
            RawProto protoTestInit = m_protoSolver.FindChild("test_initialization");

            if (protoTestInterval != null)
            {
                if (protoTestInterval.Value != "0")
                {
                    protoTestInterval.Value = "0";
                    bSet = true;
                }
            }

            if (protoTestInit != null)
            {
                if (protoTestInit.Value != "False")
                {
                    protoTestInit.Value = "False";
                    bSet = true;
                }
            }

            if (protoTestIter != null)
            {
                m_protoSolver.RemoveChild(protoTestIter);
                bSet = true;
            }

            if (bSet)
                SolverDescription = m_protoSolver.ToString();

            return bSet;
        }

        /// <summary>
        /// Returns a string representation of the Project.
        /// </summary>
        /// <returns>The string describing the Project is returned.</returns>
        public override string ToString()
        {
            string strName = Name;

            if (strName == null || strName.Length == 0)
            {
                string strModelDesc = ModelDescription;

                if (strModelDesc != null && strModelDesc.Length > 0)
                {
                    int nPos = strModelDesc.IndexOf("name:");

                    if (nPos < 0)
                        nPos = strModelDesc.IndexOf("Name:");

                    if (nPos >= 0)
                    {
                        nPos += 5;
                        int nPos2 = strModelDesc.IndexOfAny(new char[] { ' ', '\n', '\r' }, nPos);

                        if (nPos2 > 0)
                            strName = strModelDesc.Substring(nPos + 5, nPos2).Trim();
                    }
                }

                if (strName.Length == 0)
                    strName = "(ID = " + m_project.ID.ToString() + ")";
            }

            return "Project: " + strName + " -> Dataset: " + m_project.Dataset.Name;
        }
    }

    class PhaseStageCollection /** @private */
    {
        List<PhaseStage> m_rgItems = new List<PhaseStage>();

        public PhaseStageCollection()
        {
        }

        public int Count
        {
            get { return m_rgItems.Count; }
        }

        public bool Add(Phase p, Stage s)
        {
            PhaseStage ps = Find(p, s);
            if (ps == null)
            {
                m_rgItems.Add(new PhaseStage(p, s));
                return true;
            }

            return false;
        }

        public PhaseStage Find(Phase p, Stage s)
        {
            foreach (PhaseStage ps in m_rgItems)
            {
                if (ps.Phase == p && ps.Stage == s)
                    return ps;
            }

            return null;
        }

        public PhaseStageCollection FindAllWith(Stage stage)
        {
            PhaseStageCollection psCol = new PhaseStageCollection();

            foreach (PhaseStage ps in m_rgItems)
            {
                if (ps.Stage == stage)
                    psCol.Add(ps.Phase, ps.Stage);
            }

            return psCol;
        }
        public PhaseStageCollection FindAllWith(params Phase[] phase)
        {
            PhaseStageCollection psCol = new PhaseStageCollection();

            foreach (PhaseStage ps in m_rgItems)
            {
                if (phase.Contains(ps.Phase))
                    psCol.Add(ps.Phase, ps.Stage);
            }

            return psCol;
        }
    }

    class PhaseStage /** @private */
    {
        Phase m_phase = Phase.NONE;
        Stage m_stage = Stage.NONE;

        public PhaseStage(Phase p, Stage s)
        {
            m_phase = p;
            m_stage = s;
        }

        public Phase Phase
        {
            get { return m_phase; }
        }

        public Stage Stage
        {
            get { return m_stage; }
        }
    }
}
