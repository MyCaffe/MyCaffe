using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the DataTemporalLayer (used in TFT models).  
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// </remarks>
    public class DataTemporalParameter : LayerParameterBase
    {
        string m_strSource = null;
        uint m_nBatchSize;
        uint m_nNumHistoricalSteps;
        uint m_nNumFutureSteps;
        SOURCE_TYPE m_srcType = SOURCE_TYPE.PATH_NPY_FILE;
        Phase? m_forcedPhase = null;
        double m_dfMaxLoadPercent = 1;
        uint m_nChunkCount = 1024;
        int m_nDripRefreshRate = 0;
        uint? m_nSeed = null;
        bool m_bShuffleData = true;
        bool m_bOutputTargetHistorical = false;

        /// <summary>
        /// Defines the type of source data.
        /// </summary>
        public enum SOURCE_TYPE
        {
            /// <summary>
            /// Specifies the source is a path to a set of *.npy files where a npy file exists for the following:
            /// name_[type]_schema.xml                  - schema data for the data set  files.
            /// [type]_known_num.npy                    - known numeric data (used in past and future)
            /// [type]_known_cat.npy                    - known categorical data (used in past and future)
            /// [type]_observed_num.npy                 - observed numeric data (used in past and one used for target)
            /// [type]_observed_cat.npy                 - observed categorical data (used in past)
            /// [type]_static_num.npy                   - static numeric data (used in static)
            /// [type]_static_cat.npy                   - static categorical data (used in static)
            /// Where 'type' = 'test', 'train' or 'validation'
            /// 
            /// All data files contain data streams for time and category ID and all data streams are in the same 
            /// order: category ID, field, time and contain 'float' types.  The schema.xml file defines the fields
            /// contained within each file as well as the target field within the observed_num.npy file.  In addition,
            /// the schema.xml file contains lookup tables for all categorical data and for the the category ID.  
            /// For the category ID, the lookup table also contains the start and end index of valid data in each
            /// data stream.  If one of the npy files above does not exist, not data for that class of data exists.
            /// NOTE: TIME and ID fields are only used for reference and debugging and are not used as input data.
            /// </summary>
            PATH_NPY_FILE,
            /// <summary>
            /// Specifies the source path is the name of a data source within the SQL (or SQL Express) database.
            /// </summary>
            SQL_DB
        }

        /** @copydoc LayerParameterBase */
        public DataTemporalParameter()
        {
        }

        /// <summary>
        /// Optionally, specifies to output a top containing the target historical data.
        /// </summary>
        [Description("Optionally, specifies to output a top containing the target historical data.")]
        public bool output_target_historical
        {
            get { return m_bOutputTargetHistorical; }
            set { m_bOutputTargetHistorical = value; }
        }

        /// <summary>
        /// Optionally, specifies the phase to use when loading data.
        /// </summary>
        [Description("Optionally, specifies the phase to use when loading data.")]
        public Phase? forced_phase
        {
            get { return m_forcedPhase; }
            set { m_forcedPhase = value; }
        }

        /// <summary>
        /// Specifies to randomly select from the data (default = true).
        /// </summary>
        [Description("Specifies to randomly select from the data (default = true).")]
        public bool shuffle_data
        {
            get { return m_bShuffleData; }
            set { m_bShuffleData = value; }
        }

        /// <summary>
        /// Specifies the number of items to load per cycle when background loading (default = 1024).
        /// </summary>
        /// <remarks>
        /// Note the chunk count must be larger than the batch size.
        /// </remarks>
        [Description("Specifies the number of items to load per cycle when background loading (default = 1024).")]
        public uint chunk_count
        {
            get { return m_nChunkCount; }
            set { m_nChunkCount = value; }
        }

        /// <summary>
        /// Specifies the random seed used to shuffle the data.  When not specified, the default seed is used.
        /// </summary>
        [Description("Specifies the random seed used to shuffle the data.  When not specified, the default seed is used.")]
        public uint? seed
        {
            get { return m_nSeed; }
            set { m_nSeed = value; }
        }

        /// <summary>
        /// Specifies the maximum percent of data rows to load (default = 1.0 = 100%).
        /// </summary>
        [Description("Specifies the maximum percent of data rows to load (default = 1.0 = 100%).")]
        public double max_load_percent
        {
            get { return m_dfMaxLoadPercent; }
            set { m_dfMaxLoadPercent = value; }
        }

        /// <summary>
        /// Specifies rate the drip refresh occurs in seconds (default = 0, disabled).
        /// </summary>
        [Description("Specifies rate the drip refresh occurs in seconds (default = 0, disabled).")]
        public int drip_refresh_rate_in_sec
        {
            get { return m_nDripRefreshRate; }
            set { m_nDripRefreshRate= value; }
        }

        /// <summary>
        /// Specifies the type of source data.
        /// </summary>
        [Description("Specifies the type of source data.")]
        public SOURCE_TYPE source_type
        {
            get { return m_srcType; }
            set { m_srcType = value; }
        }

        /// <summary>
        /// Specifies the data source.  
        /// </summary>
        /// <remarks>
        /// When the source type is equal to PATH_NPY_FILE, the 'source' value is a path pointing the data *.npy data files.
        /// </remarks>
        [Description("Specifies the data source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// Specifies the batch size of the data.
        /// </summary>
        [Description("Specifies the batch size the data.")]
        public virtual uint batch_size
        {
            get { return m_nBatchSize; }
            set { m_nBatchSize = value; }
        }

        /// <summary>
        /// Specifies the number of historical steps
        /// </summary>
        [Description("Specifies the number of historical steps.")]
        public uint num_historical_steps
        {
            get { return m_nNumHistoricalSteps; }
            set { m_nNumHistoricalSteps = value; }
        }

        /// <summary>
        /// Specifies the number of future steps
        /// </summary>
        [Description("Specifies the number of future steps.")]
        public uint num_future_steps
        {
            get { return m_nNumFutureSteps; }
            set { m_nNumFutureSteps = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DataTemporalParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            DataTemporalParameter p = (DataTemporalParameter)src;

            m_srcType = p.source_type;
            m_strSource = p.source;
            m_nBatchSize = p.batch_size;

            m_nNumHistoricalSteps = p.num_historical_steps;
            m_nNumFutureSteps = p.num_future_steps;

            m_dfMaxLoadPercent = p.max_load_percent;
            m_nDripRefreshRate = p.drip_refresh_rate_in_sec;
            m_nSeed = p.seed;
            m_nChunkCount = p.chunk_count;
            m_bShuffleData = p.shuffle_data;
            m_forcedPhase = p.forced_phase;
            m_bOutputTargetHistorical = p.output_target_historical;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            DataTemporalParameter p = new DataTemporalParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("source", source);
            rgChildren.Add("source_type", source_type.ToString());

            rgChildren.Add("num_historical_steps", num_historical_steps.ToString());
            rgChildren.Add("num_future_steps", num_future_steps.ToString());

            rgChildren.Add("max_load_percent", max_load_percent.ToString());
            rgChildren.Add("drip_refresh_rate_in_sec", drip_refresh_rate_in_sec.ToString());
            rgChildren.Add("chunk_count", chunk_count.ToString());
            rgChildren.Add("shuffle_data", shuffle_data.ToString());
            rgChildren.Add("output_target_historical", output_target_historical.ToString());

            if (seed.HasValue)
                rgChildren.Add("seed", seed.Value.ToString());

            if (forced_phase.HasValue)
                rgChildren.Add("forced_phase", forced_phase.Value.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DataTemporalParameter FromProto(RawProto rp)
        {
            string strVal;
            DataTemporalParameter p = new DataTemporalParameter();

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal;

            if ((strVal = rp.FindValue("source_type")) != null)
            {
                if (strVal == SOURCE_TYPE.PATH_NPY_FILE.ToString())
                    p.source_type = SOURCE_TYPE.PATH_NPY_FILE;
                else if (strVal == SOURCE_TYPE.SQL_DB.ToString())
                    p.source_type = SOURCE_TYPE.SQL_DB;
                else
                    throw new Exception("Unknown source_type '" + strVal + "'!");
            }

            if ((strVal = rp.FindValue("num_historical_steps")) != null)
                p.num_historical_steps = uint.Parse(strVal);

            if ((strVal = rp.FindValue("num_future_steps")) != null)
                p.num_future_steps = uint.Parse(strVal);

            if ((strVal = rp.FindValue("max_load_percent")) != null)
                p.max_load_percent = double.Parse(strVal);

            if ((strVal = rp.FindValue("drip_refresh_rate_in_sec")) != null)
                p.drip_refresh_rate_in_sec = int.Parse(strVal);

            if ((strVal = rp.FindValue("chunk_count")) != null)
                p.chunk_count = uint.Parse(strVal);

            if (p.chunk_count == 0)
                p.chunk_count = 1;

            if ((strVal = rp.FindValue("seed")) != null)
                p.seed = uint.Parse(strVal);

            if ((strVal = rp.FindValue("shuffle_data")) != null)
                p.shuffle_data = bool.Parse(strVal);

            if ((strVal = rp.FindValue("forced_phase")) != null)
            {
                if (strVal == Phase.TRAIN.ToString())
                    p.forced_phase = Phase.TRAIN;
                else if (strVal == Phase.TEST.ToString())
                    p.forced_phase = Phase.TEST;
                else if (strVal == Phase.RUN.ToString())
                    p.forced_phase = Phase.RUN;
                else
                    throw new Exception("Unknown forced_phase '" + strVal + "'!");
            }

            if ((strVal = rp.FindValue("output_target_historical")) != null)
                p.output_target_historical = bool.Parse(strVal);

            return p;
        }
    }
}
