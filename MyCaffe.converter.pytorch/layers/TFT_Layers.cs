using MyCaffe.basecode;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class ChannelEmbeddingLayerInfo : LayerInfo /** @private */
    {
        public ChannelEmbeddingLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CLASSES)
            {
                strCode += generateNumericalTransformationClass(m_layer);
                strCode += generateCategoricalTransformationClass(m_layer);
                strCode += generateNullTransformationClass(m_layer);
                strCode += generateChannelEmbeddingClass(m_layer);
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
                // TODO: Implement.
            }
            else if (gen == GENERATE.FORWARD)
            {
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;
            }

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
            string strCode = "";

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

        private string generateCategoricalTransformationClass(LayerParameter p)
        {
            string strCode = "";

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

        private string generateChannelEmbeddingClass(LayerParameter p)
        {
            string strCode = "";

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
            strCode += "            self.numerical_transformation = NullTransform()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        if num_categorical > 0:" + Environment.NewLine;
            strCode += "            if time_distribute:" + Environment.NewLine;
            strCode += "                self.categorical_transformation = TimeDistributed(CategoricalTransformation(num_inputs=self.num_categorical, state_size=self.state_size, cardinalities=self.categorical_cardinalities))" + Environment.NewLine;
            strCode += "            else:" + Environment.NewLine;
            strCode += "                self.categorical_transformation = CategoricalTransformation(num_inputs=self.num_categorical, state_size=self.state_size, cardinalities=self.categorical_cardinalities)" + Environment.NewLine;
            strCode += "        else:" + Environment.NewLine;
            strCode += "           self.categorical_transformation = NullTransform()" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:" + Environment.NewLine;
            strCode += "        batch_shape = x_num.shape if x_num.nelement() > 0 else x_cat.shape" + Environment.NewLine;
            strCode += "        processed_num = self.numerical_transformation(x_num)" + Environment.NewLine;
            strCode += "        processed_cat = self.categorical_transformation(x_cat)" + Environment.NewLine;
            strCode += "        merged_trasformations = torch.cat(processed_num + processed_cat, dim=1)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        return merged_transformations" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }
    }

    class GrnLayerInfo : LayerInfo /** @private */
    {
        public GrnLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.CLASSES)
            {
            }
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self." + m_layer.name + " = nn.Softmax(dim=" + m_layer.softmax_param.axis.ToString() + ")" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }
}
