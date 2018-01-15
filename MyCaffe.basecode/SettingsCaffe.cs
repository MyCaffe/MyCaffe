using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Serialization;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The SettingsCaffe defines the settings used by the MyCaffe CaffeControl.
    /// </summary>
    [Serializable]
    public class SettingsCaffe : ISerializable
    {
        int m_nMaskAllButLastColumns = 0;
        bool m_bEnableLabelBalancing = false;
        bool m_bEnableLabelBoosting = false;
        bool m_bEnableRandomInputSelection = true;
        bool m_bEnablePairInputSelection = false;
        bool m_bUseTrainingSourceForTesting = false;
        double m_dfSuperBoostProbability = 0.0;
        int m_nMaximumIterationOverride = -1;
        int m_nTestingIterationOverride = -1;
        string m_strDefaultModelGroup = "";
        string m_strGpuIds = "0";
        IMAGEDB_LOAD_METHOD m_imageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND;
        int m_nImageDbLoadLimit = 0;
        SNAPSHOT_WEIGHT_UPDATE_METHOD m_snapshotWeightUpdateMethod = SNAPSHOT_WEIGHT_UPDATE_METHOD.FAVOR_ACCURACY;
        SNAPSHOT_LOAD_METHOD m_snapshotLoadMethod = SNAPSHOT_LOAD_METHOD.STATE_BEST_ACCURACY;

        /// <summary>
        /// The SettingsCaffe constructor.
        /// </summary>
        public SettingsCaffe()
        {
        }

        /// <summary>
        /// The SettingsCaffe constructor that copies another SettingsCaffe instance.
        /// </summary>
        public SettingsCaffe(SettingsCaffe s)
        {
            m_nMaskAllButLastColumns = s.m_nMaskAllButLastColumns;
            m_bEnableLabelBalancing = s.m_bEnableLabelBalancing;
            m_bEnableLabelBoosting = s.m_bEnableLabelBoosting;
            m_bEnableRandomInputSelection = s.m_bEnableRandomInputSelection;
            m_bEnablePairInputSelection = s.m_bEnablePairInputSelection;
            m_bUseTrainingSourceForTesting = s.m_bUseTrainingSourceForTesting;
            m_dfSuperBoostProbability = s.m_dfSuperBoostProbability;
            m_nMaximumIterationOverride = s.m_nMaximumIterationOverride;
            m_nTestingIterationOverride = s.m_nTestingIterationOverride;
            m_strDefaultModelGroup = s.m_strDefaultModelGroup;
            m_strGpuIds = s.m_strGpuIds;
            m_imageDbLoadMethod = s.m_imageDbLoadMethod;
            m_nImageDbLoadLimit = s.m_nImageDbLoadLimit;
            m_snapshotWeightUpdateMethod = s.m_snapshotWeightUpdateMethod;
            m_snapshotLoadMethod = s.m_snapshotLoadMethod;
        }

        /// <summary>
        /// The SettingsCaffe constructor used during deserialization.
        /// </summary>
        /// <param name="info">Specifies the serialization information.</param>
        /// <param name="context">Specifies the serialization context.</param>
        public SettingsCaffe(SerializationInfo info, StreamingContext context)
        {
            m_bEnableLabelBalancing = getBool(info, "bEnableLabelBalancing", m_bEnableLabelBalancing);
            m_bEnableLabelBoosting = getBool(info, "bEnableLabelBoosting", m_bEnableLabelBoosting);
            m_bEnableRandomInputSelection = getBool(info, "bEnableRandomInputSelection", m_bEnableRandomInputSelection);
            m_bEnablePairInputSelection = getBool(info, "bEnablePairInputSelection", m_bEnablePairInputSelection);
            m_bUseTrainingSourceForTesting = getBool(info, "bUseTrainingSourceForTesting", m_bUseTrainingSourceForTesting);
            m_dfSuperBoostProbability = getDouble(info, "dfSuperBoostProbability", m_dfSuperBoostProbability);
            m_nMaximumIterationOverride = getInt(info, "nMaximumIterationOverride", m_nMaximumIterationOverride);
            m_nTestingIterationOverride = getInt(info, "nTestingIterationOverride", m_nTestingIterationOverride);
            m_strDefaultModelGroup = info.GetString("strDefaultModelGroup");
            m_strGpuIds = getString(info, "strGpuIds", m_strGpuIds);
            m_nMaskAllButLastColumns = getInt(info, "nMaskAllButLastColumns", m_nMaskAllButLastColumns);
            m_imageDbLoadMethod = (IMAGEDB_LOAD_METHOD)getInt(info, "ImageDbLoadMethod", (int)m_imageDbLoadMethod);
            m_nImageDbLoadLimit = getInt(info, "ImageDbLoadLimit", m_nImageDbLoadLimit);
            m_snapshotWeightUpdateMethod = (SNAPSHOT_WEIGHT_UPDATE_METHOD)getInt(info, "SnapshotWeightUpdateMethod", (int)m_snapshotWeightUpdateMethod);
            m_snapshotLoadMethod = (SNAPSHOT_LOAD_METHOD)getInt(info, "SnapshotLoadMethod", (int)m_snapshotLoadMethod);
        }

        private bool getBool(SerializationInfo info, string str, bool bDefault)
        {
            try
            {
                return info.GetBoolean(str);
            }
            catch
            {
                return bDefault;
            }
        }

        private double getDouble(SerializationInfo info, string str, double dfDefault)
        {
            try
            {
                return info.GetDouble(str);
            }
            catch
            {
                return dfDefault;
            }
        }

        private string getString(SerializationInfo info, string str, string strDefault)
        {
            try
            {
                return info.GetString(str);
            }
            catch
            {
                return strDefault;
            }
        }

        private int getInt(SerializationInfo info, string str, int nDefault)
        {
            try
            {
                return info.GetInt32(str);
            }
            catch
            {
                return nDefault;
            }
        }

        /// <summary>
        /// The GetObjectData is used during serialization.
        /// </summary>
        /// <param name="info">Specifies the serialization information.</param>
        /// <param name="context">Specifies the serialization context.</param>
        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("bEnableLabelBalancing", m_bEnableLabelBalancing);
            info.AddValue("bEnableLabelBoosting", m_bEnableLabelBoosting);
            info.AddValue("bEnableRandomInputSelection", m_bEnableRandomInputSelection);
            info.AddValue("bEnablePairInputSelection", m_bEnablePairInputSelection);
            info.AddValue("bUseTrainingSourceForTesting", m_bUseTrainingSourceForTesting);
            info.AddValue("dfSuperBoostProbability", m_dfSuperBoostProbability);
            info.AddValue("nMaximumIterationOverride", m_nMaximumIterationOverride);
            info.AddValue("nTestingIterationOverride", m_nTestingIterationOverride);
            info.AddValue("strDefaultModelGroup", m_strDefaultModelGroup);
            info.AddValue("strGpuIds", m_strGpuIds);
            info.AddValue("nMaskAllButLastColumns", m_nMaskAllButLastColumns);
            info.AddValue("ImageDbLoadMethod", (int)m_imageDbLoadMethod);
            info.AddValue("ImageDbLoadLimit", m_nImageDbLoadLimit);
            info.AddValue("SnapshotWeightUpdateMethod", (int)m_snapshotWeightUpdateMethod);
            info.AddValue("SnapshotLoadMethod", (int)m_snapshotLoadMethod);
        }

        /// <summary>
        /// Returns a copy of the SettingsCaffe object.
        /// </summary>
        /// <returns>The copy of the SettingsCaffe object is returned.</returns>
        public SettingsCaffe Clone()
        {
            SettingsCaffe s = new SettingsCaffe();

            s.m_bEnableLabelBalancing = m_bEnableLabelBalancing;
            s.m_bEnableLabelBoosting = m_bEnableLabelBoosting;
            s.m_bEnableRandomInputSelection = m_bEnableRandomInputSelection;
            s.m_bEnablePairInputSelection = m_bEnablePairInputSelection;
            s.m_bUseTrainingSourceForTesting = m_bUseTrainingSourceForTesting;
            s.m_dfSuperBoostProbability = m_dfSuperBoostProbability;
            s.m_nMaximumIterationOverride = m_nMaximumIterationOverride;
            s.m_nTestingIterationOverride = m_nTestingIterationOverride;
            s.m_strDefaultModelGroup = m_strDefaultModelGroup;
            s.m_strGpuIds = m_strGpuIds;
            s.m_nMaskAllButLastColumns = m_nMaskAllButLastColumns;
            s.m_imageDbLoadMethod = m_imageDbLoadMethod;
            s.m_nImageDbLoadLimit = m_nImageDbLoadLimit;
            s.m_snapshotWeightUpdateMethod = m_snapshotWeightUpdateMethod;
            s.m_snapshotLoadMethod = m_snapshotLoadMethod;

            return s;
        }

        /// <summary>
        /// Get/set label balancing.  When enabled, first the label set is randomly selected and then the image
        /// is selected from the label set using the image selection criteria (e.g. Random).
        /// </summary>
        public bool EnableLabelBalancing
        {
            get { return m_bEnableLabelBalancing; }
            set { m_bEnableLabelBalancing = value; }
        }

        /// <summary>
        /// Get/set label boosting.  When using Label boosting, images are selected from boosted labels with 
        /// a higher probability that images from other label sets.
        /// </summary>
        public bool EnableLabelBoosting
        {
            get { return m_bEnableLabelBoosting; }
            set { m_bEnableLabelBoosting = value; }
        }

        /// <summary>
        /// Get/set random image selection.  When enabled, images are randomly selected from the entire set, or 
        /// randomly from a label set when label balancing is in effect.
        /// </summary>
        public bool EnableRandomInputSelection
        {
            get { return m_bEnableRandomInputSelection; }
            set { m_bEnableRandomInputSelection = value; }
        }

        /// <summary>
        /// Get/set pair image selection.  When using pair selection, images are queried in pairs where the first query selects
        /// the image based on the image selection criteria (e.g. Random), and then the second image query returns the image just following the 
        /// first image in the database.
        /// </summary>
        public bool EnablePairInputSelection
        {
            get { return m_bEnablePairInputSelection; }
            set { m_bEnablePairInputSelection = value; }
        }

        /// <summary>
        /// Get/set whether or not to use the training datasource when testing.
        /// </summary>
        public bool UseTrainingSourceForTesting
        {
            get { return m_bUseTrainingSourceForTesting; }
            set { m_bUseTrainingSourceForTesting = value; }
        }

        /// <summary>
        /// Get/set the superboost probability used when selecting boosted images.
        /// </summary>
        public double SuperBoostProbability
        {
            get { return m_dfSuperBoostProbability; }
            set { m_dfSuperBoostProbability = value; }
        }

        /// <summary>
        /// Get/set the maximum iteration override.  When set, this overrides the training iterations specified in the solver description.
        /// </summary>
        public int MaximumIterationOverride
        {
            get { return m_nMaximumIterationOverride; }
            set { m_nMaximumIterationOverride = value; }
        }

        /// <summary>
        /// Get/set the testing iteration override.  When set, this overrides the testing iterations specified in the solver description.
        /// </summary>
        public int TestingIterationOverride
        {
            get { return m_nTestingIterationOverride; }
            set { m_nTestingIterationOverride = value; }
        }

        /// <summary>
        /// Get/set the default model group to use.
        /// </summary>
        public string DefaultModelGroup
        {
            get { return m_strDefaultModelGroup; }
            set { m_strDefaultModelGroup = value; }
        }

        /// <summary>
        /// Get/set the default GPU ID's to use when training.
        /// </summary>
        /// <remarks>
        /// When using multi-GPU training, it is highly recommended to only train on TCC enabled drivers, otherwise driver timeouts may occur on large models.
        /// @see [NVIDIA Tesla Compute Cluster (TCC) Help](http://docs.nvidia.com/gameworks/content/developertools/desktop/tesla_compute_cluster.htm)
        /// </remarks>
        public string GpuIds
        {
            get { return m_strGpuIds; }
            set { m_strGpuIds = value; }
        }

        /// <summary>
        /// Get/set the number of columns to leave un-masked (by default this is disabled).
        /// </summary>
        public int MaskAllButLastColumns
        {
            get { return m_nMaskAllButLastColumns; }
            set { m_nMaskAllButLastColumns = value; }
        }

        /// <summary>
        /// Get/set the image database loading method.
        /// </summary>
        public IMAGEDB_LOAD_METHOD ImageDbLoadMethod
        {
            get { return m_imageDbLoadMethod; }
            set { m_imageDbLoadMethod = value; }
        }

        /// <summary>
        /// Get/set the image database load limit.
        /// </summary>
        public int ImageDbLoadLimit
        {
            get { return m_nImageDbLoadLimit; }
            set { m_nImageDbLoadLimit = value; }
        }

        /// <summary>
        /// Get/set the snapshot update method.
        /// </summary>
        public SNAPSHOT_WEIGHT_UPDATE_METHOD SnapshotWeightUpdateMethod
        {
            get { return m_snapshotWeightUpdateMethod; }
            set { m_snapshotWeightUpdateMethod = value; }
        }

        /// <summary>
        /// Get/set the snapshot load method.
        /// </summary>
        public SNAPSHOT_LOAD_METHOD SnapshotLoadMethod
        {
            get { return m_snapshotLoadMethod; }
            set { m_snapshotLoadMethod = value; }
        }
    }
}
