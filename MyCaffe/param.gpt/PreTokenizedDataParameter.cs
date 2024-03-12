using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.param.python;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the PreTokenizedDataLayer.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class PreTokenizedDataParameter : LayerParameterBase
    {
        uint m_nBatchSize;
        uint m_nBlockSize;
        string m_strSource;
        int? m_nSeed = null;
        bool m_bShuffle = true;
        int? m_nPadToken = null;
        VOCABULARY_TYPE m_vocabType = VOCABULARY_TYPE.LLAMA2;
        SAMPLE_METHOD m_sampleMethod = SAMPLE_METHOD.PROBABILITY;

        /// <summary>
        /// Defines the vocabulary type to use.
        /// </summary>
        public enum VOCABULARY_TYPE
        {
            /// <summary>
            /// Specifies to use the Byte-Pair Encoding (BPE) vocabulary supported by the LlaMA2 models.
            /// </summary>
            LLAMA2,
        }

        /// <summary>
        /// Defines the sampling method used.
        /// </summary>
        public enum SAMPLE_METHOD
        {
            /// <summary>
            /// Specifies to use the argmax method.
            /// </summary>
            ARGMAX,
            /// <summary>
            /// Specifies to use the probability sampling method where the probabilities
            /// are totaled until the sum >= a random number.
            /// </summary>
            PROBABILITY
        }

        /** @copydoc LayerParameterBase */
        public PreTokenizedDataParameter()
        {            
        }

        /// <summary>
        /// Optionally, specifies the pad token, or null to ignore (default = null).
        /// </summary>
        [Description("Optionally, specifies the pad token, or null to ignore (default = null).")]
        public int? pad_token
        {
            get { return m_nPadToken; }
            set { m_nPadToken = value; }
        }

        /// <summary>
        /// Specifies the seed used to initialize the random number generator (normally only for testing).
        /// </summary>
        [Description("Specifies the seed used to initialize the random number generator (normally only for testing).")]
        public int? seed
        {
            get { return m_nSeed; }
            set { m_nSeed = value; }
        }

        /// <summary>
        /// Specifies to shuffle the data (default = true).
        /// </summary>
        [Description("Specifies to shuffle the data (default = true).")]
        public bool shuffle
        {
            get { return m_bShuffle; }
            set { m_bShuffle = value; }
        }

        /// <summary>
        /// Specifies the vocabulary type to use.
        /// </summary>
        [Description("Specifies the vocabulary type to use.")]
        public VOCABULARY_TYPE vocabulary_type
        {
            get { return m_vocabType; }
            set { m_vocabType = value; }
        }

        /// <summary>
        /// Specifies the sampling method used when post processing logits (default = ARGMAX).
        /// </summary>
        [Description("Specifies the sampling method used when post processing logits (default = ARGMAX).")]
        public SAMPLE_METHOD sample_method
        {
            get { return m_sampleMethod; }
            set { m_sampleMethod = value; }
        }

        /// <summary>
        /// Specifies the data source based on the INPUT_TYPE used.  Each dataset has both a training and testing data source.
        /// </summary>
        [Description("Specifies the data source based on the INPUT_TYPE used.  Each dataset has both a training and testing data source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// The number of heads used.
        /// </summary>
        [Description("Specifies batch size.")]
        public uint batch_size
        {
            get { return m_nBatchSize; }
            set { m_nBatchSize = value; }
        }

        /// <summary>
        /// Specifies size of the block.
        /// </summary>
        public uint block_size
        {
            get { return m_nBlockSize; }
            set { m_nBlockSize = value; }
        }
        
        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PreTokenizedDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PreTokenizedDataParameter p = (PreTokenizedDataParameter)src;

            m_strSource = p.source;
            m_nBatchSize = p.batch_size;
            m_nBlockSize = p.block_size;
            m_nSeed = p.seed;
            m_vocabType = p.vocabulary_type;       
            m_sampleMethod = p.sample_method;
            m_bShuffle = p.shuffle;
            m_nPadToken = p.pad_token;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PreTokenizedDataParameter p = new PreTokenizedDataParameter();
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

            rgChildren.Add("vocabulary_type", vocabulary_type.ToString());
            rgChildren.Add("sample_method", sample_method.ToString());
            rgChildren.Add("source", "\"" + source + "\"");
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("block_size", block_size.ToString());
            rgChildren.Add("shuffle", shuffle.ToString().ToLower());

            if (pad_token != null)
                rgChildren.Add("pad_token", pad_token.ToString());

            if (seed != null)
                rgChildren.Add("seed", seed.ToString());
            
            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PreTokenizedDataParameter FromProto(RawProto rp)
        {
            string strVal;
            PreTokenizedDataParameter p = new PreTokenizedDataParameter();

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);
            
            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("seed")) != null)
                p.seed = int.Parse(strVal);

            if ((strVal = rp.FindValue("pad_token")) != null)
                p.pad_token = int.Parse(strVal);

            if ((strVal = rp.FindValue("shuffle")) != null)
                p.shuffle = bool.Parse(strVal);

            if ((strVal = rp.FindValue("vocabulary_type")) != null)
            {
                if (strVal == VOCABULARY_TYPE.LLAMA2.ToString())
                    p.vocabulary_type = VOCABULARY_TYPE.LLAMA2;
                else
                    throw new Exception("Unknown vocabulary type '" + strVal + "'");
            }

            if ((strVal = rp.FindValue("sample_method")) != null)
            {
                if (strVal == SAMPLE_METHOD.ARGMAX.ToString())
                    p.sample_method = SAMPLE_METHOD.ARGMAX;
                else if (strVal == SAMPLE_METHOD.PROBABILITY.ToString())
                    p.sample_method = SAMPLE_METHOD.PROBABILITY;
                else
                    throw new Exception("Unknown sample method '" + strVal + "'");
            }

            return p;
        }
    }
}
