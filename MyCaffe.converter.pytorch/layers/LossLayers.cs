using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class SoftmaxLossLayerInfo : SoftmaxLayerInfo /** @private */
    {
        public SoftmaxLossLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_rgstrReturnValues.Add("loss", 1);
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";

            if (gen == GENERATE.DEFINITION)
            {
                strCode += "        self.smx = nn.Softmax(dim=" + m_layer.softmax_param.axis.ToString() + ")" + Environment.NewLine;
                strCode += "        self." + m_layer.name + " = nn.CrossEntropyLoss()" + Environment.NewLine;
            }
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
            {
                strCode += "        smx1 = self.smx(" + m_inputs.AsText + ")" + Environment.NewLine;
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(smx1, " + m_layer.bottom[1] + ")" + Environment.NewLine;
            }

            return strCode;
        }
    }

}
