using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class DataLayerInfo : LayerInfo /** @private */
    {
        public DataLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_rgParameters.Add("batch_size", layer.data_param.batch_size);
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = "";
            return strCode;
        }
    }
}
