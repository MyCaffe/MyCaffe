using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.model
{
    /// <summary>
    /// The Config structure defines the configuration of the model, which is also the header in Llama based models.
    /// </summary>
    struct Config
    {
        /// <summary>
        /// Specifies the embed dimension of the model, for Llama7B this is 4096.
        /// </summary>
        public int dim;
        /// <summary>
        /// Specifies the hidden dimension of the model, for Llama7B this is 11008.
        /// </summary>
        public int hidden_dim;
        /// <summary>
        /// Specifies the number of layers in the model, for Llama7B this is 32.
        /// </summary>
        public int n_layers;
        /// <summary>
        /// Specifies the number of heads in the model, for Llama7B this is 32.
        /// </summary>
        public int n_heads;
        /// <summary>
        /// Specifies the number of key-value heads in the model, for Llama7B this is 32.
        /// </summary>
        public int n_kv_heads;
        /// <summary>
        /// Specifies the vocabulary size of the model, for Llama7B this is 32000.
        /// </summary>
        public int vocab_size;
        /// <summary>
        /// Specifies the sequence length of the model, for Llama7B this is 2048.
        /// </summary>
        public int seq_len;
        /// <summary>
        /// Specifies whether or not the classifier is shared, for Llama7B this is 0.
        /// </summary>
        public byte shared_classifier;
    }
}
