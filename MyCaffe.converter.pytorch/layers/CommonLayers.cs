using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class InnerProductLayerInfo : LayerInfo /** @private */
    {
        public InnerProductLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_outputs[0].Shape[1] = (int)layer.inner_product_param.num_output;
            m_outputs[0].Shape[2] = 1;
            m_outputs[0].Shape[3] = 1;
        }

        public override string Generate(GENERATE gen)
        {
            int nInFeatures = m_inputs[0].getCount(m_layer.inner_product_param.axis);
            int nOutFeatures = (int)m_layer.inner_product_param.num_output;

            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Linear(in_features=" + nInFeatures + ", out_features=" + nOutFeatures + ", bias=" + m_layer.inner_product_param.bias_term.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
                strCode += initWeights(m_layer.name, m_layer.inner_product_param.bias_term, m_layer.inner_product_param.weight_filler, m_layer.inner_product_param.bias_filler);
            else if (gen == GENERATE.FORWARD)
            {
                strCode += "        " + m_inputs.AsText + " = " + m_inputs.AsText + ".view(" + m_inputs.AsText + ".size(0), -1)" + Environment.NewLine;
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;
            }

            return strCode;
        }
    }

    class ConcatLayerInfo : LayerInfo /** @private */
    {
        public ConcatLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_outputs = m_inputs.Clone(1);
            m_outputs[0].Name = layer.top[0];

            int nCount = 0;

            for (int i = 0; i < m_inputs.Count; i++)
            {
                nCount += m_inputs[i].Shape[layer.concat_param.axis];
            }

            m_outputs[0].Shape[layer.concat_param.axis] = nCount;
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "#       self." + m_layer.name + " = Concat(" + m_inputs.AsText + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = torch.concat(" + m_inputs.AsText + ", dim=0)" + Environment.NewLine;

            return strCode;
        }
    }

    class DropoutLayerInfo : LayerInfo /** @private */
    {
        public DropoutLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Dropout(p=" + m_layer.dropout_param.dropout_ratio.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class SoftmaxLayerInfo : LayerInfo /** @private */
    {
        public SoftmaxLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Softmax(dim=" + m_layer.softmax_param.axis.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class AccuracyLayerInfo : LayerInfo /** @private */
    {
        public AccuracyLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_rgstrReturnValues.Add("self.accuracy", 2);
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
            {
                strCode += "#        self." + m_layer.name + " = Accuracy(" + m_inputs.AsText + ")" + Environment.NewLine;
                strCode += "        self.accuracy_sum = 0" + Environment.NewLine;
                strCode += "        self.accuracy_count = 0" + Environment.NewLine;
                strCode += "        self.accuracy = 0" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
            {
                strCode += "        x1 = torch.argmax(" + m_inputs.AsText + ", dim=1)" + Environment.NewLine;
                strCode += "        self.accuracy_sum += torch.sum(x1 == " + m_layer.bottom[1] + ")" + Environment.NewLine;
                strCode += "        self.accuracy_count += len(x1)" + Environment.NewLine;
                strCode += "        self.accuracy = self.accuracy_sum / self.accuracy_count" + Environment.NewLine;
            }

            return strCode;
        }
    }
}
