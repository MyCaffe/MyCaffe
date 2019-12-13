using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param
{
    /// <summary>
    /// The DataNoiseParameter is used by the DataParameter when the 'use_data_for_nonmatch' = True, which is used when 'images_per_blob' > 1.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DataNoiseParameter
    {
        int m_nNoiseDataLabel = -1;
        bool m_bUseNoisyMean = true;
        string m_strNoisyDataSavePath = null;
        FillerParameter m_noiseFiller = new FillerParameter("constant", 1);

        /// <summary>
        /// The constructor.
        /// </summary>
        public DataNoiseParameter()
        {
        }

        /// <summary>
        /// (\b optional, default = true) When <i>true</i> the noise is applied to the mean and used as the noisy data.  NOTE: When using this setting, the filler should be configured to produce values in the range [0,1].
        /// </summary>
        public bool use_noisy_mean
        {
            get { return m_bUseNoisyMean; }
            set { m_bUseNoisyMean = value; }
        }

        /// <summary>
        /// (\b optional, default = -1) Specifies the label used with each noise filled data used when 'use_noise_for_nonmatch' = <i>true</i>.
        /// </summary>
        public int noise_data_label
        {
            get { return m_nNoiseDataLabel; }
            set { m_nNoiseDataLabel = value; }
        }

        /// <summary>
        /// Specifies the noise filler used when 'use_noise_for_nonmatch' = <i>true</i>.  By default the 'noise_filter' is set to CONSTANT(1) which, when used with the 'use_noisy_mean' = True, uses the mean image as the data noise.
        /// </summary>
        public FillerParameter noise_filler
        {
            get { return m_noiseFiller; }
            set { m_noiseFiller = value; }
        }

        /// <summary>
        /// (/b optional, default = null) Specifies the path where the noisy data image is saved, otherwise is ignored when null.  This setting is only used for debugging.
        /// </summary>
        public string noisy_save_path
        {
            get { return m_strNoisyDataSavePath; }
            set { m_strNoisyDataSavePath = value; }
        }

        private string noisy_save_path_persist
        {
            get
            {
                string strPath = Utility.Replace(m_strNoisyDataSavePath, ':', ';');
                return Utility.Replace(strPath, ' ', '~');
            }

            set
            {
                string strPath = Utility.Replace(value, ';', ':');
                m_strNoisyDataSavePath = Utility.Replace(strPath, '~', ' ');
            }
        }

        /// <summary>
        /// Copies the specified source data noise parameter to this one.
        /// </summary>
        /// <param name="pSrc">Specifies the source data noise parameter.</param>
        public void Copy(DataNoiseParameter pSrc)
        {
            if (pSrc == null)
                return;

            m_bUseNoisyMean = pSrc.m_bUseNoisyMean;
            m_nNoiseDataLabel = pSrc.m_nNoiseDataLabel;
            m_strNoisyDataSavePath = pSrc.m_strNoisyDataSavePath;

            if (pSrc.m_noiseFiller != null)
                m_noiseFiller = pSrc.m_noiseFiller.Clone();
        }

        /// <summary>
        /// Convert the DataNoiseParameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the RawProto name.</param>
        /// <returns>The RawProto containing the settings is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("use_noisy_mean", m_bUseNoisyMean.ToString());
            rgChildren.Add("noise_data_label", m_nNoiseDataLabel.ToString());
            rgChildren.Add("noisy_data_path", noisy_save_path_persist);

            if (noise_filler != null)
                rgChildren.Add(noise_filler.ToProto("noise_filler"));

            return new RawProto(strName, "", rgChildren);
        }


        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DataNoiseParameter FromProto(RawProto rp, DataNoiseParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new DataNoiseParameter();

            if ((strVal = rp.FindValue("use_noisy_mean")) != null)
                p.use_noisy_mean = bool.Parse(strVal);

            if ((strVal = rp.FindValue("noise_data_label")) != null)
                p.noise_data_label = int.Parse(strVal);

            if ((strVal = rp.FindValue("noisy_data_path")) != null)
                p.noisy_save_path_persist = strVal;

            RawProto rpNoiseFiller = rp.FindChild("noise_filler");
            if (rpNoiseFiller != null)
                p.noise_filler = FillerParameter.FromProto(rpNoiseFiller);

            return p;
        }
    }
}
