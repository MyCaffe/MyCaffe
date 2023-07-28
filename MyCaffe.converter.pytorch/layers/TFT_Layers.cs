using MyCaffe.basecode;
using MyCaffe.fillers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class DataTemporalLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCount = 0;

        public DataTemporalLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            if (m_layer.include.Count == 0 || m_layer.include[0].phase != Phase.TRAIN)
                return "";

            string strCode = "";
            if (gen == GENERATE.CLASSES)
            {
                if (m_nGenerationCount == 0)
                    strCode += generateDataTemporalClass(m_layer);
                m_nGenerationCount++;
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = DataTemporal()" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        private string generateDataTemporalClass(LayerParameter p)
        {
            string strCode = "";

            strCode += "class DataTemporal(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self):" + Environment.NewLine;
            strCode += "        super(DataTemporal, self).__init__()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        return None" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }
    }

    class ChannelEmbeddingLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCreditCount = 0;
        static int m_nGenerationCount = 0;

        public ChannelEmbeddingLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CREDITS)
            {
                strCode += generateCredits();
            }
            else if (gen == GENERATE.CLASSES)
            {
                if (m_nGenerationCount == 0)
                {
                    strCode += generateNullTransformationClass(m_layer);
                    strCode += generateTimeDistributedClass(m_layer);
                    strCode += generateNumericalTransformationClass(m_layer);
                    strCode += generateCategoricalTransformationClass(m_layer);
                    strCode += generateChannelEmbeddingClass(m_layer);
                }

                m_nGenerationCount++;
            }
            else if (gen == GENERATE.DEFINITION)
            {
                bool bTimeDistributed = false;
                string strCardinalities = Utility.ToString<int>(m_layer.categorical_trans_param.cardinalities, -1, -1, "[", "]");

                strCode += "        self.time_distributed = " + bTimeDistributed.ToString() + Environment.NewLine;
                strCode += "        self.categorical_cardinalities = " + strCardinalities + Environment.NewLine;
                strCode += "        self." + m_layer.name + " = nn.ChannelEmbedding(state_size=" + m_layer.numeric_trans_param.state_size.ToString() + ",num_numeric=" + m_layer.numeric_trans_param.num_input.ToString() + ",num_categorical=" + m_layer.categorical_trans_param.num_input.ToString() + ",categorical_cardinalities=self.categorical_cardinalities,time_distributed=self.time_distributed)" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
                strCode += "        self." + m_layer.name + ".init_weights()" + Environment.NewLine;
            }
            else if (gen == GENERATE.FORWARD)
            {
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;
            }

            return strCode;
        }

        public static string generateCredits()
        {
            string strCode = "";

            if (m_nGenerationCreditCount > 0)
                return strCode;

            strCode += "# NullTransform, TimeDistributed, NumericalTransformation, CategoricalTransformation, ChannelEmbedding Layers:" + Environment.NewLine;
            strCode += "#   original code <https://github.com/PlaytikaOSS/tft-torch/tree/main>" + Environment.NewLine;
            strCode += "#   license: (MIT) <https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE>" + Environment.NewLine;

            m_nGenerationCreditCount++;

            return strCode;
        }

        private string generateNullTransformationClass(LayerParameter p)
        {
            string strCode = "";

            strCode += "class NullTransformation(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self):" + Environment.NewLine;
            strCode += "        super(NullTransformation, self).__init__()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x: torch.Tensor):" + Environment.NewLine;
            strCode += "        return []" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string generateTimeDistributedClass(LayerParameter p)
        {
            string strCode = "";

            strCode += "class TimeDistributed(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, module: nn.Module, batch_first: bool = True, return_reshaped: bool = True):" + Environment.NewLine;
            strCode += "        super(TimeDistributed, self).__init__()" + Environment.NewLine;
            strCode += "        self.module = module" + Environment.NewLine;
            strCode += "        self.batch_first = batch_first" + Environment.NewLine;
            strCode += "        self.return_reshaped = return_reshaped" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x: torch.Tensor):" + Environment.NewLine;
            strCode += "        if len(x.shape) <= 2:" + Environment.NewLine;
            strCode += "            return self.module(x)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        x_reshape = x.contiguous().view(-1, x.shape[-1])" + Environment.NewLine;
            strCode += "        y = self.module(x_reshape)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if self.return_reshaped:" + Environment.NewLine;
            strCode += "            y = y.contiguous().view(x.shape[0], -1, y.shape[-1])" + Environment.NewLine;
            strCode += "        else:" + Environment.NewLine;
            strCode += "            y = y.contiguous().view(-1, x.shape[1], y.shape[-1])" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        return y" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string generateNumericalTransformationClass(LayerParameter p)
        {
            FillerParameter wtfiller = new FillerParameter("xavier");
            FillerParameter biasfiller = new FillerParameter("constant");
            string strCode = "";

            if (m_bAddComments)
                strCode += generateNumericalTransformationClassComments();

            strCode += "class NumericalTransformation(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, num_inputs: int, state_size: int):" + Environment.NewLine;
            strCode += "        super(NumericalTransformation, self).__init__()" + Environment.NewLine;
            strCode += "        self.num_inputs = num_inputs" + Environment.NewLine;
            strCode += "        self.state_size = state_size" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.numeric_projection_layers = nn.ModuleList()" + Environment.NewLine;
            strCode += "        for i in range(num_inputs):" + Environment.NewLine;
            strCode += "            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += "        for i in range(self.num_inputs):" + Environment.NewLine;
            strCode += initWeights("    ", "self.numeric_projection_layers[i]", true, wtfiller, biasfiller);
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x: torch.Tensor):" + Environment.NewLine;
            strCode += "        projections = []" + Environment.NewLine;
            strCode += "        for i in range(self.num_inputs):" + Environment.NewLine;
            strCode += "            x1 = x[:,[i]]" + Environment.NewLine;
            strCode += "            x2 = self.numeric_projection_layers[i](x1)" + Environment.NewLine;
            strCode += "            projections.append(x2)" + Environment.NewLine;
            strCode += "        return projections" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string generateNumericalTransformationClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The NumericalTransformation class transforms the numerical input into a set of projections for each input." + Environment.NewLine;
            strCode += "# Each input is projected using a dedicated linear layer to a vector within the state_size, which is." + Environment.NewLine;
            strCode += "# output as a list of length num_inputs that contains each embedding." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# num_input : int" + Environment.NewLine;
            strCode += "#    The number of numerical inputs." + Environment.NewLine;
            strCode += "# state_size : int" + Environment.NewLine;
            strCode += "#    The state size of the model, which determines the embedding dimension/width for each input variable." + Environment.NewLine;

            return strCode;
        }

        private string generateCategoricalTransformationClass(LayerParameter p)
        {
            FillerParameter wtfiller = new FillerParameter("xavier");
            string strCode = "";

            if (m_bAddComments)
                strCode += generateCategoricalTransformationClassComments();

            strCode += "class CategoricalTransformation(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, num_inputs: int, state_size: int, cardinalities: List[int]):" + Environment.NewLine;
            strCode += "        super(CategoricalTransformation, self).__init__()" + Environment.NewLine;
            strCode += "        self.num_inputs = num_inputs" + Environment.NewLine;
            strCode += "        self.state_size = state_size" + Environment.NewLine;
            strCode += "        self.cardinalities = cardinalities" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.categorical_embedding_layers = nn.ModuleList()" + Environment.NewLine;
            strCode += "        for i, cardinality in enumerate(self.cardinalities):" + Environment.NewLine;
            strCode += "            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += "        for i, cardinality in enumerate(self.cardinalities):" + Environment.NewLine;
            strCode += initWeights("    ", "self.categorical_embedding_layers[i]", false, wtfiller, null);
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x: torch.Tensor):" + Environment.NewLine;
            strCode += "        embeddings = []" + Environment.NewLine;
            strCode += "        for i in range(self.num_inputs):" + Environment.NewLine;
            strCode += "            x1 = x[:,i]" + Environment.NewLine;
            strCode += "            x2 = self.categorical_embedding_layers[i](x1)" + Environment.NewLine;
            strCode += "            embeddings.append(x2)" + Environment.NewLine;
            strCode += "        return embeddings" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string generateCategoricalTransformationClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The CategoricalTransformation class transforms the categorical input into a set of embeddings for each input." + Environment.NewLine;
            strCode += "# Each input is projected using a dedicated embedding layer to a vector within the state_size, which is." + Environment.NewLine;
            strCode += "# output as a list of length num_inputs that contains each embedding." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# num_input : int" + Environment.NewLine;
            strCode += "#    The number of categorical inputs." + Environment.NewLine;
            strCode += "# state_size : int" + Environment.NewLine;
            strCode += "#    The state size of the model, which determines the embedding dimension/width for each input variable." + Environment.NewLine;
            strCode += "# cadinalities : List[int]" + Environment.NewLine;
            strCode += "#    The cardinality of each categorical input." + Environment.NewLine;

            return strCode;
        }

        private string generateChannelEmbeddingClass(LayerParameter p)
        {
            string strCode = "";

            if (m_bAddComments)
                strCode += generateChannelEmbeddingClassComments();

            strCode += "class ChannelEmbedding(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, state_size: int, num_numeric: int, num_categorical: int, categorical_cardinalities: List[int], time_distribute: Optional[bool] = False):" + Environment.NewLine;
            strCode += "        super(ChannelEmbedding, self).__init__()" + Environment.NewLine;
            strCode += "        self.state_size = state_size" + Environment.NewLine;
            strCode += "        self.num_numeric = num_numeric" + Environment.NewLine;
            strCode += "        self.num_categorical = num_categorical" + Environment.NewLine;
            strCode += "        self.categorical_cardinalities = categorical_cardinalities" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if num_numeric > 0:" + Environment.NewLine;
            strCode += "            if time_distribute:" + Environment.NewLine;
            strCode += "                self.numerical_transformation = TimeDistributed(NumericalTransformation(num_inputs=self.num_numeric, state_size=self.state_size))" + Environment.NewLine;
            strCode += "            else:" + Environment.NewLine;
            strCode += "                self.numerical_transformation = NumericalTransformation(num_inputs=self.num_numeric, state_size=self.state_size)" + Environment.NewLine;
            strCode += "        else:" + Environment.NewLine;
            strCode += "            self.numerical_transformation = NullTransformation()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if num_categorical > 0:" + Environment.NewLine;
            strCode += "            if time_distribute:" + Environment.NewLine;
            strCode += "                self.categorical_transformation = TimeDistributed(CategoricalTransformation(num_inputs=self.num_categorical, state_size=self.state_size, cardinalities=self.categorical_cardinalities))" + Environment.NewLine;
            strCode += "            else:" + Environment.NewLine;
            strCode += "                self.categorical_transformation = CategoricalTransformation(num_inputs=self.num_categorical, state_size=self.state_size, cardinalities=self.categorical_cardinalities)" + Environment.NewLine;
            strCode += "        else:" + Environment.NewLine;
            strCode += "           self.categorical_transformation = NullTransformation()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += "        self.numeric_transformation.init_weights()" + Environment.NewLine;
            strCode += "        self.categorical_transformation.init_weights()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        batch_shape = x_num.shape if x_num.nelement() > 0 else x_cat.shape" + Environment.NewLine;
            strCode += "        processed_num = self.numerical_transformation(x_num)" + Environment.NewLine;
            strCode += "        processed_cat = self.categorical_transformation(x_cat)" + Environment.NewLine;
            strCode += "        merged_transformations = torch.cat(processed_num + processed_cat, dim=1)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        return merged_transformations" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string generateChannelEmbeddingClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The ChannelEmbedding class handles the transformation/embedding of the input channel of numeric and categorical tensors." + Environment.NewLine;
            strCode += "# A NumericalTransformation class is used to process the numerical inputs, and" + Environment.NewLine;
            strCode += "# a CategoricalTransformation class is used to process the categorical inputs." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# state_size : int" + Environment.NewLine;
            strCode += "#    The state size of the model, which determines the embedding dimension/width for each input variable." + Environment.NewLine;
            strCode += "# num_numeric : int" + Environment.NewLine;
            strCode += "#    The number of numeric inputs." + Environment.NewLine;
            strCode += "# num_categorical : int" + Environment.NewLine;
            strCode += "#    The number of categorical inputs." + Environment.NewLine;
            strCode += "# categorical_cardinalities : List[int]" + Environment.NewLine;
            strCode += "#    The cardinality of each categorical input." + Environment.NewLine;
            strCode += "# time_distribute : Optional[bool] = False" + Environment.NewLine;
            strCode += "#    When True, the TimeDistributed transformation is applied to each time step of the input tensor." + Environment.NewLine;

            return strCode;
        }
    }

    class GLULayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCreditCount = 0;
        static int m_nGenerationCount = 0;

        public GLULayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CREDITS)
            {
                strCode += generateGluCredits();
            }
            else if (gen == GENERATE.CLASSES)
            {
                strCode += generateGluClass(m_layer.glu_param.bias_term, m_layer.glu_param.weight_filler, m_layer.glu_param.bias_filler, m_bAddComments);
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = GLU(input_dim=" + m_layer.glu_param.input_dim.ToString() + ")" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
                strCode += "        self." + m_layer.name + ".init_weights()" + Environment.NewLine;
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        public static string generateGluCredits()
        {
            string strCode = "";

            if (m_nGenerationCreditCount > 0)
                return strCode;

            strCode += "# GLU Layer:" + Environment.NewLine;
            strCode += "#   original code <https://github.com/PlaytikaOSS/tft-torch/tree/main>" + Environment.NewLine;
            strCode += "#   license: (MIT) <https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE>" + Environment.NewLine;

            m_nGenerationCreditCount++;

            return strCode;
        }

        public static string generateGluClass(bool bBiasTerm, FillerParameter wtFiller, FillerParameter biasFiller, bool bAddComments)
        {
            string strCode = "";

            if (m_nGenerationCount == 0)
            {
                if (bAddComments)
                    strCode += generateGluClassComments();
                strCode += "class GLU(nn.Module):" + Environment.NewLine;
                strCode += "    def __init__(self, input_dim: int):" + Environment.NewLine;
                strCode += "        super(GLU, self).__init__()" + Environment.NewLine;
                strCode += "        self.input_dim = input_dim" + Environment.NewLine;
                strCode += "        self.fc1 = nn.Linear(self.input_dim, self.input_dim)" + Environment.NewLine;
                strCode += "        self.fc2 = nn.Linear(self.input_dim, self.input_dim)" + Environment.NewLine;
                strCode += "        self.sigmoid = nn.Sigmoid()" + Environment.NewLine;
                strCode += Environment.NewLine;
                strCode += "    def init_weights(self):" + Environment.NewLine;
                strCode += initWeights("    ", "self.fc1", bBiasTerm, wtFiller, biasFiller);
                strCode += initWeights("    ", "self.fc2", bBiasTerm, wtFiller, biasFiller);
                strCode += Environment.NewLine;
                strCode += "    def forward(self, x: torch.Tensor) -> torch.Tensor:" + Environment.NewLine;
                strCode += "        x = self.fc1(x)" + Environment.NewLine;
                strCode += "        sig = self.sigmoid(x)" + Environment.NewLine;
                strCode += "        x = self.fc2(x)" + Environment.NewLine;
                strCode += "        return torch.mul(sig, x)" + Environment.NewLine;
                strCode += Environment.NewLine;
            }

            m_nGenerationCount++;

            return strCode;
        }

        private static string generateGluClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The GLU class defines the Gated Linear Unit (GLU) layer, as described" + Environment.NewLine;
            strCode += "# in Dauphin, Yann N., et al. \"Language modeling with gated convolutional networks.\" arXiv preprint arXiv:1612.08083 (2016)." + Environment.NewLine;
            strCode += "# <https://arxiv.org/pdf/1612.08083.pdf>" + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# The output of this layer is a linear projection (X * W + b) modulated by the gates Sigmoid(X * V + c)." + Environment.NewLine;
            strCode += "# These gates multiply each element of the matrix X * W and control the information passed on in the heirarchy." + Environment.NewLine;
            strCode += "# This unit is simplified gating mechanism for non-deterministic gates that reduce the vanishing gradient problem," + Environment.NewLine;
            strCode += "# by having linear units coupled to the gates. This retains the non-linear capabilities of the network while allowing" + Environment.NewLine;
            strCode += "# the gradient to propagate through the linear unit without scaling." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# input_dim : int" + Environment.NewLine;
            strCode += "#    The embedding width/dimension of the input." + Environment.NewLine;

            return strCode;
        }
    }

    class GRNLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCreditCount = 0;
        static int m_nGenerationCount = 0;

        public GRNLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CREDITS)
            {
                strCode += generateCredits();
            }
            else if (gen == GENERATE.CLASSES)
            {
                strCode += generateGrnClass(m_layer);
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = GRN(input_dim=" + m_layer.grn_param.input_dim.ToString() + ", hidden_dim=" + m_layer.grn_param.hidden_dim.ToString() + ", output_dim=" + m_layer.grn_param.output_dim.ToString() + ", dropout=" + m_layer.grn_param.dropout_ratio.ToString() + ", context_dim=" + m_layer.grn_param.context_dim.ToString() + ", batch_first=" + m_layer.grn_param.batch_first.ToString() + ", activation=\'" + m_layer.grn_param.activation.ToString() + "\')" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
                strCode += "        self." + m_layer.name + ".init_weights()" + Environment.NewLine;
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        public static string generateCredits()
        {
            string strCode = "";

            if (m_nGenerationCreditCount++ > 0)
                return strCode;

            strCode += "# GLU, GRN Layers:" + Environment.NewLine;
            strCode += "#   original code <https://github.com/PlaytikaOSS/tft-torch/tree/main>" + Environment.NewLine;
            strCode += "#   license: (MIT) <https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE>" + Environment.NewLine;

            m_nGenerationCreditCount++;

            return strCode;
        }

        private string generateGrnClass(LayerParameter p)
        {
            string strCode = "";

            if (m_nGenerationCount > 0)
                return strCode;

            strCode += GLULayerInfo.generateGluClass(true, p.grn_param.weight_filler, p.grn_param.bias_filler, m_bAddComments);

            if (m_bAddComments)
                strCode += generateClassComments(p);

            strCode += "class GRN(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: Optional[float] = 0.05, context_dim: Optional[int] = None, batch_first: Optional[bool] = True, activation: Optional[string] = 'ELU'):" + Environment.NewLine;
            strCode += "        super(GRN, self).__init__()" + Environment.NewLine;
            strCode += "        self.input_dim = input_dim" + Environment.NewLine;
            strCode += "        self.hidden_dim = hidden_dim" + Environment.NewLine;
            strCode += "        self.output_dim = output_dim" + Environment.NewLine;
            strCode += "        self.context_dim = context_dim" + Environment.NewLine;
            strCode += "        self.batch_first = batch_first" + Environment.NewLine;
            strCode += "        self.dropout = dropout" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.project_residual: bool = self.input_dim != self.output_dim" + Environment.NewLine;
            strCode += "        if self.project_residual:" + Environment.NewLine;
            strCode += "            self.skip_layer = TimeDistributed(nn.Linear(self.input_dim, self.output_dim))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.fc1 = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim), batch_first=batch_first)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if self.context_dim is not None:" + Environment.NewLine;
            strCode += "            self.context_projection = TimeDistributed(nn.Linear(self.context_dim, self.hidden_dim, bias=False), batch_first=batch_first)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if activation == 'RELU':" + Environment.NewLine;
            strCode += "            self.activation = nn.ReLU()" + Environment.NewLine;
            strCode += "        else:" + Environment.NewLine;
            strCode += "            self.activation = nn.ELU()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.fc2 = TimeDistributed(nn.Linear(self.hidden_dim, self.output_dim), batch_first=batch_first)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.dropout = nn.Dropout(self.dropout)" + Environment.NewLine;
            strCode += "        self.gate = TimeDistributed(GLU(self.output_dim), batch_first=batch_first)" + Environment.NewLine;
            strCode += "        self.layernorm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += initWeights("", "self.skip_layer", true, m_layer.grn_param.weight_filler, m_layer.grn_param.bias_filler);
            strCode += initWeights("", "self.fc1", true, m_layer.grn_param.weight_filler, m_layer.grn_param.bias_filler);
            strCode += initWeights("", "self.context_projection", true, m_layer.grn_param.weight_filler, m_layer.grn_param.bias_filler);
            strCode += initWeights("", "self.fc2", true, m_layer.grn_param.weight_filler, m_layer.grn_param.bias_filler);
            strCode += "        self.gate.init_weights()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        if self.project_residual:" + Environment.NewLine;
            strCode += "            residual = self.skip_layer(x)" + Environment.NewLine;
            strCode += "        else:" + Environment.NewLine;
            strCode += "            residual = x" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        x = self.fc1(x)" + Environment.NewLine;
            strCode += "        if context is not None:" + Environment.NewLine;
            strCode += "            context = self.context_projection(context)" + Environment.NewLine;
            strCode += "            x = x + context" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        x = self.activation(x)" + Environment.NewLine;
            strCode += "        x = self.fc2(x)" + Environment.NewLine;
            strCode += "        x = self.dropout(x)" + Environment.NewLine;
            strCode += "        x = self.gate(x)" + Environment.NewLine;
            strCode += "        x = x + residual" + Environment.NewLine;
            strCode += "        x = self.layernorm(x)" + Environment.NewLine;
            strCode += "        return x" + Environment.NewLine;
            strCode += Environment.NewLine;

            m_nGenerationCount++;

            return strCode;
        }

        private string generateClassComments(LayerParameter p)
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The GRN class defines the Gated Residual Network layer." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# The primary consists of the input (x) and an optional context vector (c)." + Environment.NewLine;
            strCode += "# A GLU is used for controlling the extent to which the module contributes to the original input (x)," + Environment.NewLine;
            strCode += "# potentially skipping over the layer entirely as the GLU outputs could be close to zero, therefore" + Environment.NewLine;
            strCode += "# suppressing the non-linear contribution.  When no context vector is used, the GRN treats the context" + Environment.NewLine;
            strCode += "# input as zero.  During training, dropout is applied before the gating layer." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# input_dim : int" + Environment.NewLine;
            strCode += "#    The embedding width/dimension of the input." + Environment.NewLine;
            strCode += "# hidden_dim : int" + Environment.NewLine;
            strCode += "#    The itermediate embedding width." + Environment.NewLine;
            strCode += "# output_dim : int" + Environment.NewLine;
            strCode += "#    The embedding width of the output." + Environment.NewLine;
            strCode += "# dropout : Optional[float]" + Environment.NewLine;
            strCode += "#    The dropout ratio to use." + Environment.NewLine;
            strCode += "# context_dim : int" + Environment.NewLine;
            strCode += "#    The embedding width/dimension of the context siganl expected to be fed as an auxiliary input." + Environment.NewLine;
            strCode += "# batch_first : bool" + Environment.NewLine;
            strCode += "#    When true, the first dimension of the input and output is the batch size." + Environment.NewLine;
            return strCode;
        }
    }

    class VarSelNetLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCreditCount = 0;
        static int m_nGenerationCount = 0;

        public VarSelNetLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CREDITS)
            {
                strCode += generateCredits();
            }
            else if (gen == GENERATE.CLASSES)
            {
                strCode += generateVarSelNetClass(m_layer);
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = VarSelNet(input_dim=" + m_layer.varselnet_param.input_dim.ToString() + ", num_inputs=" + m_layer.varselnet_param.num_inputs.ToString() + ", hidden_dim=" + m_layer.varselnet_param.hidden_dim.ToString() + ", dropout=" + m_layer.varselnet_param.dropout_ratio.ToString() + ", context_dim=" + m_layer.varselnet_param.context_dim.ToString() + ", batch_first=" + m_layer.varselnet_param.batch_first.ToString() + ")" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
                strCode += "        self." + m_layer.name + ".init_weights()" + Environment.NewLine;
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        public static string generateCredits()
        {
            string strCode = "";

            if (m_nGenerationCreditCount > 0)
                return strCode;

            strCode += "# VarSetNet Layer:" + Environment.NewLine;
            strCode += "#   original code <https://github.com/PlaytikaOSS/tft-torch/tree/main>" + Environment.NewLine;
            strCode += "#   license: (MIT) <https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE>" + Environment.NewLine;

            m_nGenerationCreditCount++;

            return strCode;
        }

        private string generateVarSelNetClass(LayerParameter p)
        {
            string strCode = "";

            if (m_nGenerationCount > 0)
                return strCode;

            if (m_bAddComments)
                strCode += generateClassComments();

            strCode += "class VarSelNet(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float, context_dim: Optional[int] = None, batch_first: Optional[bool] = True):" + Environment.NewLine;
            strCode += "        super(VarSelNet, self).__init__()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.hidden_dim = hidden_dim" + Environment.NewLine;
            strCode += "        self.input_dim = input_dim" + Environment.NewLine;
            strCode += "        self.num_inputs = num_inputs" + Environment.NewLine;
            strCode += "        self.dropout = dropout" + Environment.NewLine;
            strCode += "        self.context_dim = context_dim" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.flattened_grn = GRN(input_dim=self.num_inputs * self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.num_inputs, dropout=self.dropout, context_dim=self.context_dim, batch_first=batch_first)" + Environment.NewLine;
            strCode += "        self.softmax = nn.Softmax(dim=1)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.single_variable_grns = nn.ModuleList()" + Environment.NewLine;
            strCode += "        for i in range(self.num_inputs):" + Environment.NewLine;
            strCode += "            self.single_variable_grns.append(GRN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, dropout=self.dropout, batch_first=batch_first))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += "        self.flattened_grn.init_weights()" + Environment.NewLine;
            strCode += "        for i in range(self.num_inputs):" + Environment.NewLine;
            strCode += "            self.single_variable_grns[i].init_weights()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, flattened_embedding: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        sparse_weights = self.flattened_grn(flattened_embedding, context)" + Environment.NewLine;
            strCode += "        sparse_weights = self.softmax(sparse_weights)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        processed_inputs = []" + Environment.NewLine;
            strCode += "        for i in range(self.num_inputs):" + Environment.NewLine;
            strCode += "#            processed_inputs.append(self.single_variable_grns[i](flattened_embedding[..., i * self.input_dim:(i + 1) * self.input_dim]))" + Environment.NewLine;
            strCode += "            processed_inputs.append(self.single_variable_grns[i](flattened_embedding[:, i * self.input_dim:(i + 1) * self.input_dim]))" + Environment.NewLine;
            strCode += "        processed_inputs = torch.stack(processed_inputs, dim=1)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        outputs = processed_inputs * sparse_weights.transpose(1, 2)" + Environment.NewLine;
            strCode += "        outputs = outputs.sum(axis=-1)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        return outputs, sparse_weights" + Environment.NewLine;
            strCode += Environment.NewLine;

            m_nGenerationCount++;

            return strCode;
        }

        private string generateClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The VarSelNet class handles the fact that the relevant and specific contribution of each input." + Environment.NewLine;
            strCode += "# variable to the output is unknown.  This class enables instance-wise variable selection, and is applied." + Environment.NewLine;
            strCode += "# to both the static covariates and time-dependent covariates.  In addition to providing insights int which" + Environment.NewLine;
            strCode += "# variables are most important, this class also enables the model remove any unecessary noisy inputs that" + Environment.NewLine;
            strCode += "# could negatively impact performance." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# input_dim : int" + Environment.NewLine;
            strCode += "#    The attribute/embedding dimension of the input, associated with the 'state_size' of the model.." + Environment.NewLine;
            strCode += "# num_input : int" + Environment.NewLine;
            strCode += "#    The number of input variables, including both numeric and categorical inputs." + Environment.NewLine;
            strCode += "# hidden_dim : int" + Environment.NewLine;
            strCode += "#    The embedding width of the output." + Environment.NewLine;
            strCode += "# dropout : float" + Environment.NewLine;
            strCode += "#    The dropout rate associated with the 'GRN' classes." + Environment.NewLine;
            strCode += "# context_dim : int" + Environment.NewLine;
            strCode += "#    The embedding width of the context signal expected to be fed as an auxilary input." + Environment.NewLine;
            strCode += "# batch_first : bool" + Environment.NewLine;
            strCode += "#    When True, the first dimension of the input and output tensors represent the batch size." + Environment.NewLine;

            return strCode;
        }
    }

    class GateAddNormLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCreditCount = 0;
        static int m_nGenerationCount = 0;

        public GateAddNormLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CREDITS)
            {
                strCode += generateCredits();
            }
            else if (gen == GENERATE.CLASSES)
            {
                strCode += generateGateAddNormClass(m_layer);
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = GLU(input_dim=" + m_layer.glu_param.input_dim.ToString() + ")" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
                strCode += "        self." + m_layer.name + ".init_weights()" + Environment.NewLine;
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        public static string generateCredits()
        {
            string strCode = "";

            if (m_nGenerationCreditCount > 0)
                return strCode;

            strCode += "# GateAddNorm Layer:" + Environment.NewLine;
            strCode += "#   original code <https://github.com/PlaytikaOSS/tft-torch/tree/main>" + Environment.NewLine;
            strCode += "#   license: (MIT) <https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE>" + Environment.NewLine;
            m_nGenerationCreditCount++;

            return strCode;
        }

        private string generateGateAddNormClass(LayerParameter p)
        {
            string strCode = "";

            if (m_nGenerationCount > 0)
                return strCode;

            if (m_bAddComments)
                strCode += generateClassComments();

            strCode += "class GateAddNorm(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, input_dim: int, dropout: Optional[float] = None):" + Environment.NewLine;
            strCode += "        super(GateAddNorm, self).__init__()" + Environment.NewLine;
            strCode += "        self.input_dim = input_dim" + Environment.NewLine;
            strCode += "        self.dropout = dropout" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if self.dropout is not None:" + Environment.NewLine;
            strCode += "            self.dropout_layer = nn.Dropout(p=self.dropout)" + Environment.NewLine;
            strCode += "        self.gate = TimeDistributed(GLU(self.input_dim), batch_first=True)" + Environment.NewLine;
            strCode += "        self.layernorm = TimeDistributed(nn.LayerNorm(self.input_dim), batch_first=True)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += "       self.gate.init_weights()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        if self.dropout is not None:" + Environment.NewLine;
            strCode += "            x = self.dropout_layer(x)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        x = self.gate(x)" + Environment.NewLine;
            strCode += "        if residual is not None:" + Environment.NewLine;
            strCode += "            x = x + residual" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        x = self.layernorm(x)" + Environment.NewLine;
            strCode += "        return x" + Environment.NewLine;
            strCode += Environment.NewLine;

            m_nGenerationCount++;

            return strCode;
        }

        private string generateClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The GateAddNorm class performs a dropout, residual connection and layer normalization." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# input_dim : int" + Environment.NewLine;
            strCode += "#    The attribute/embedding dimension of the input, associated with the 'state_size' of the model.." + Environment.NewLine;
            strCode += "# dropout : float" + Environment.NewLine;
            strCode += "#    The dropout rate associated with the 'GRN' classes." + Environment.NewLine;

            return strCode;
        }
    }

    class MultiheadAttentionInterpLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCreditCount = 0;
        static int m_nGenerationCount = 0;

        public MultiheadAttentionInterpLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CREDITS)
            {
                strCode += generateCredits();
            }
            else if (gen == GENERATE.CLASSES)
            {
                strCode += generateMultiheadAttentionInterpClass(m_layer);
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = MultiheadAttentionInterp(embed_dim=" + m_layer.multihead_attention_interp_param.embed_dim.ToString() + ", num_heads=" + m_layer.multihead_attention_interp_param.num_heads.ToString() + ")" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
                strCode += "        self." + m_layer.name + ".init_weights()" + Environment.NewLine;
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        public static string generateCredits()
        {
            string strCode = "";

            if (m_nGenerationCreditCount > 0)
                return strCode;

            strCode += "# MultiheadAttentionInterp Layer:" + Environment.NewLine;
            strCode += "#   original code <https://github.com/PlaytikaOSS/tft-torch/tree/main>" + Environment.NewLine;
            strCode += "#   license: (MIT) <https://github.com/PlaytikaOSS/tft-torch/blob/main/LICENSE>" + Environment.NewLine;

            m_nGenerationCreditCount++;

            return strCode;
        }

        private string generateMultiheadAttentionInterpClass(LayerParameter p)
        {
            string strCode = "";

            if (m_nGenerationCount > 0)
                return strCode;

            if (m_bAddComments)
                strCode += generateClassComments();

            strCode += "class MultiheadAttentionInterp(nn.Module):" + Environment.NewLine;
            strCode += "    def __init__(self, embed_dim: int, num_heads: int):" + Environment.NewLine;
            strCode += "        super(MultiheadAttentionInterp, self).__init__()" + Environment.NewLine;
            strCode += "        self.d_model = embed_dim" + Environment.NewLine;
            strCode += "        self.num_heads = num_heads" + Environment.NewLine;
            strCode += "        self.all_heads_dim = embed_dim * num_heads" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.w_q = nn.Linear(self.embed_dim, self.all_heads_dim)" + Environment.NewLine;
            strCode += "        self.w_k = nn.Linear(self.embed_dim, self.all_heads_dim)" + Environment.NewLine;
            strCode += "        self.w_v = nn.Linear(self.embed_dim, self.ebmed_dim)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        self.out = nn.Linear(self.d_model, self.d_model)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += initWeights("    ", "self.w_q", true, m_layer.multihead_attention_interp_param.weight_filler, m_layer.multihead_attention_interp_param.bias_filler);
            strCode += initWeights("    ", "self.w_k", true, m_layer.multihead_attention_interp_param.weight_filler, m_layer.multihead_attention_interp_param.bias_filler);
            strCode += initWeights("    ", "self.w_v", true, m_layer.multihead_attention_interp_param.weight_filler, m_layer.multihead_attention_interp_param.bias_filler);
            strCode += initWeights("    ", "self.out", true, m_layer.multihead_attention_interp_param.weight_filler, m_layer.multihead_attention_interp_param.bias_filler);
            strCode += Environment.NewLine;
            strCode += "    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        num_samples = q.size(0)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        q_proj = self.w_q(q).view(num_samples, -1, self.num_heads, self.d_model)" + Environment.NewLine;
            strCode += "        k_proj = self.w_k(k).view(num_samples, -1, self.num_heads, self.d_model)" + Environment.NewLine;
            strCode += "        v_proj = self.w_v(v).repeat(1, 1, self.num_heads).view(num_samples, -1, self.num_heads, self.d_model)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        q_proj = q_proj.transpose(1, 2)" + Environment.NewLine;
            strCode += "        k_proj = k_proj.transpose(1, 2)" + Environment.NewLine;
            strCode += "        v_proj = v_proj.transpose(1, 2)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        attn_outputs_all_heads, attn_scores_all_heads = self.attention(q_proj, k_proj, v_proj, mask)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        attn_scores = attn_scores_all_heads.mean(dim=1)" + Environment.NewLine;
            strCode += "        attn_outputs = attn_outputs_all_heads.mean(dim=1)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        attn_outputs = self.out(attn_outputs)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        return attn_outputs, attn_scores" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:" + Environment.NewLine;
            strCode += "        scores = torch.matmul(q, k.transpose(-2, -1))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if mask is not None:" + Environment.NewLine;
            strCode += "            scores = scores.masked_fill(mask, -1e9)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        scores = F.softmax(scores, dim=-1)" + Environment.NewLine;
            strCode += "        outputs = torch.matmul(scores, v)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        return outputs, scores" + Environment.NewLine;
            strCode += Environment.NewLine;

            m_nGenerationCount++;

            return strCode;
        }

        private string generateClassComments()
        {
            string strCode = "";

            strCode += LayerInfo.generateBar();
            strCode += "# The MultiheadAttentionInterp class learns long-term relationsips across different time-steps." + Environment.NewLine;
            strCode += "# A multi-head attention is modified to enhance explainability.  With traditional multi-head attention" + Environment.NewLine;
            strCode += "# the 'values' signal is shared for all heads an additive aggregation is employed across all heads." + Environment.NewLine;
            strCode += "# However, according to the paper, each head can learn different temporal patterns, while attending to a common set" + Environment.NewLine;
            strCode += "# of input features which can be interpreted as a simple ensemble over attention weights in a combined matrix, which" + Environment.NewLine;
            strCode += "# compared to the original multi-head attention matrix, yields an increased representational capacity." + Environment.NewLine;
            strCode += "#" + Environment.NewLine;
            strCode += "# Parameters" + Environment.NewLine;
            strCode += "# ----------" + Environment.NewLine;
            strCode += "# embed_dim : int" + Environment.NewLine;
            strCode += "#    The dimensions associated with the 'state_size' of the model, corresponding to the input and output." + Environment.NewLine;
            strCode += "# num_heads : float" + Environment.NewLine;
            strCode += "#    The number of heads used by the multi-head attention component." + Environment.NewLine;

            return strCode;
        }
    }

    class ReshapeTemporalLayerInfo : LayerInfo /** @private */
    {
        static int m_nGenerationCountBefore = 0;
        static int m_nGenerationCountAfter = 0;

        public ReshapeTemporalLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CLASSES)
            {
                strCode += generateReshapeTemporalClass(m_layer);
            }
            if (gen == GENERATE.DEFINITION)
            {
                if (m_layer.reshape_temporal_param.mode == param.tft.ReshapeTemporalParameter.MODE.BEFORE)
                    strCode += "        self." + m_layer.name + " = ReshapeTemporalBefore()" + Environment.NewLine;
                else if (m_layer.reshape_temporal_param.mode == param.tft.ReshapeTemporalParameter.MODE.AFTER)
                    strCode += "        self." + m_layer.name + " = ReshapeTemporalAfter()" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }

        private string generateReshapeTemporalClass(LayerParameter p)
        {
            string strCode = "";

            if (p.reshape_temporal_param.mode == param.tft.ReshapeTemporalParameter.MODE.BEFORE)
            {
                if (m_nGenerationCountBefore == 0)
                {
                    strCode += "class ReshapeTemporalBefore(nn.Module):" + Environment.NewLine;
                    strCode += "    def __init__(self):" + Environment.NewLine;
                    strCode += "        super(ReshapeTemporalBefore, self).__init__()" + Environment.NewLine;
                    strCode += Environment.NewLine;
                    strCode += "    def forward(self, x: torch.Tensor) -> torch.Tensor:" + Environment.NewLine;
                    strCode += "        pass" + Environment.NewLine;
                    strCode += Environment.NewLine;
                }
                m_nGenerationCountBefore++;
            }
            else
            {
                if (m_nGenerationCountAfter == 0)
                {
                    strCode += "class ReshapeTemporalAfter(nn.Module):" + Environment.NewLine;
                    strCode += "    def __init__(self):" + Environment.NewLine;
                    strCode += "        super(ReshapeTemporalAfter, self).__init__()" + Environment.NewLine;
                    strCode += Environment.NewLine;
                    strCode += "    def forward(self, x: torch.Tensor) -> torch.Tensor:" + Environment.NewLine;
                    strCode += "        pass" + Environment.NewLine;
                    strCode += Environment.NewLine;
                }
                m_nGenerationCountAfter++;
            }

            return strCode;
        }
    }
}
