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
    /// Specifies the parameters for the TokenizedDataLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class TokenizedDataParameter : LayerParameterBase
    {
        /// <summary>
        /// Python layer implementations use this parameter for Python specific settings such as the location of the runtime.
        /// </summary>
        protected PythonParameter m_pythonParam = new PythonParameter();
        
        uint m_nBatchSize;
        uint m_nBlockSize;
        INPUT_TYPE m_inputType;
        string m_strSource;
        int? m_nSeed = null;
        string m_strDbgIdxFile;
        VOCABULARY_TYPE m_vocabType = VOCABULARY_TYPE.CHARACTER;
        SAMPLE_METHOD m_sampleMethod = SAMPLE_METHOD.ARGMAX;

        /// <summary>
        /// Defines the vocabulary type to use.
        /// </summary>
        public enum VOCABULARY_TYPE
        {
            /// <summary>
            /// Specifies to use a character based vocabulary.
            /// </summary>
            CHARACTER,
            /// <summary>
            /// Specifies to use a word based vocabulary.
            /// </summary>
            WORD,
            /// <summary>
            /// Specifies to use pre-generated SentencePiece vocabulary.
            /// </summary>
            SENTENCEPIECE
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

        /// <summary>
        /// Defines the input type used.
        /// </summary>
        public enum INPUT_TYPE
        {
            /// <summary>
            /// Specifies that the source is a text filename.
            /// </summary>
            TEXT_FILE
        }

        /** @copydoc LayerParameterBase */
        public TokenizedDataParameter()
        {            
        }

        /// <summary>
        /// Specifies the PythonParameter used by the python implementation of the TokenizedDataPairsLayer, otherwise this is null.
        /// </summary>
        public PythonParameter python_param
        {
            get { return m_pythonParam; }
            set { m_pythonParam = value; }
        }

        /// <summary>
        /// Specifies the seed used to initialize the random number generator (normally only for testing).
        /// </summary>
        public int? seed
        {
            get { return m_nSeed; }
            set { m_nSeed = value; }
        }

        /// <summary>
        /// Specifies data source input type.
        /// </summary>
        [Description("Specifies data source input type.")]
        public INPUT_TYPE input_type
        {
            get { return m_inputType; }
            set { m_inputType = value; }
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
        /// Specifies an optional data index file used for debugging only.
        /// </summary>
        [Description("Specifies an optional data index file used for debuging only.")]
        public string debug_index_file
        {
            get { return m_strDbgIdxFile; }
            set { m_strDbgIdxFile = value; }
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
            TokenizedDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TokenizedDataParameter p = (TokenizedDataParameter)src;

            m_pythonParam = p.python_param;
            m_inputType = p.input_type;
            m_strSource = p.source;
            m_nBatchSize = p.batch_size;
            m_nBlockSize = p.block_size;
            m_nSeed = p.seed;
            m_strDbgIdxFile = p.debug_index_file;
            m_vocabType = p.vocabulary_type;       
            m_sampleMethod = p.sample_method;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TokenizedDataParameter p = new TokenizedDataParameter();
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

            if (m_pythonParam != null)
                rgChildren.Add(m_pythonParam.ToProto("python_param"));

            rgChildren.Add("input_type", input_type.ToString());
            rgChildren.Add("vocabulary_type", vocabulary_type.ToString());
            rgChildren.Add("sample_method", sample_method.ToString());
            rgChildren.Add("source", "\"" + source + "\"");
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("block_size", block_size.ToString());

            if (!string.IsNullOrEmpty(debug_index_file))
                rgChildren.Add("debug_index_file", debug_index_file);

            if (seed != null)
                rgChildren.Add("seed", seed.ToString());
            
            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TokenizedDataParameter FromProto(RawProto rp)
        {
            string strVal;
            TokenizedDataParameter p = new TokenizedDataParameter();

            RawProto rpPython = rp.FindChild("python_param");
            if (rpPython != null)
                p.python_param = PythonParameter.FromProto(rpPython);

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);
            
            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("seed")) != null)
                p.seed = int.Parse(strVal);

            if ((strVal = rp.FindValue("debug_index_file")) != null)
                p.debug_index_file = strVal;

            if ((strVal = rp.FindValue("input_type")) != null)
            {
                if (strVal == INPUT_TYPE.TEXT_FILE.ToString())
                    p.input_type = INPUT_TYPE.TEXT_FILE;
                else
                    throw new Exception("Unknown input type '" + strVal + "'");
            }

            if ((strVal = rp.FindValue("vocabulary_type")) != null)
            {
                if (strVal == VOCABULARY_TYPE.CHARACTER.ToString())
                    p.vocabulary_type = VOCABULARY_TYPE.CHARACTER;
                else if (strVal == VOCABULARY_TYPE.WORD.ToString())
                    p.vocabulary_type = VOCABULARY_TYPE.WORD;
                else if (strVal == VOCABULARY_TYPE.SENTENCEPIECE.ToString())
                    p.vocabulary_type = VOCABULARY_TYPE.SENTENCEPIECE;
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
