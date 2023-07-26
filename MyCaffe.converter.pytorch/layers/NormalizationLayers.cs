using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class LRNLayerInfo : LayerInfo /** @private */
    {
        public LRNLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.LocalResponseNorm(size=" + m_layer.lrn_param.local_size.ToString() + ", alpha=" + m_layer.lrn_param.alpha.ToString() + ", beta=" + m_layer.lrn_param.beta.ToString() + ", k=" + m_layer.lrn_param.k.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class BatchNormLayerInfo : LayerInfo /** @private */
    {
        public BatchNormLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.BatchNorm2d(num_features=" + m_inputs[0].getCount(1).ToString() + ", eps=" + m_layer.batch_norm_param.eps.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class LayerNormLayerInfo : LayerInfo /** @private */
    {
        public LayerNormLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.LayerNorm2d(num_features=" + m_inputs[0].getCount(2).ToString() + ", eps=" + m_layer.layer_norm_param.epsilon.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }
}
