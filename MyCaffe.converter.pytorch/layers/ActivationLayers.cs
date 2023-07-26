using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class ReluLayerInfo : LayerInfo /** @private */
    {
        public ReluLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.ReLU()" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class ELULayerInfo : LayerInfo /** @private */
    {
        public ELULayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.ELU()" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class SigmoidLayerInfo : LayerInfo /** @private */
    {
        public SigmoidLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Sigmoid()" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class TanhLayerInfo : LayerInfo /** @private */
    {
        public TanhLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Tanh()" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }
}
