using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.pytorch.layers
{
    class ConvolutionLayerInfo : LayerInfo /** @private */
    {
        public ConvolutionLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            int nPad = (layer.convolution_param.pad != null && layer.convolution_param.pad.Count > 0) ? (int)layer.convolution_param.pad[0] : 0;
            int nKernel = (layer.convolution_param.kernel_size != null && layer.convolution_param.kernel_size.Count > 0) ? (int)layer.convolution_param.kernel_size[0] : 1;
            int nStride = (layer.convolution_param.stride != null && layer.convolution_param.stride.Count > 0) ? (int)layer.convolution_param.stride[0] : 1;

            m_outputs[0].Shape[1] = (int)layer.convolution_param.num_output;
            m_outputs[0].Shape[2] = (int)Math.Floor((double)(m_inputs[0].Shape[2] + 2 * nPad - nKernel) / nStride) + 1;
            m_outputs[0].Shape[3] = (int)Math.Floor((double)(m_inputs[0].Shape[3] + 2 * nPad - nKernel) / nStride) + 1;
        }

        public override string Generate(GENERATE gen)
        {
            int nPad = (m_layer.convolution_param.pad != null && m_layer.convolution_param.pad.Count > 0) ? (int)m_layer.convolution_param.pad[0] : 0;
            int nKernel = (m_layer.convolution_param.kernel_size != null && m_layer.convolution_param.kernel_size.Count > 0) ? (int)m_layer.convolution_param.kernel_size[0] : 1;
            int nStride = (m_layer.convolution_param.stride != null && m_layer.convolution_param.stride.Count > 0) ? (int)m_layer.convolution_param.stride[0] : 1;
            int nDilation = (m_layer.convolution_param.dilation != null && m_layer.convolution_param.dilation.Count > 0) ? (int)m_layer.convolution_param.dilation[0] : 1;

            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Conv2d(in_channels=" + m_inputs[0].Shape[1] + ", out_channels=" + m_layer.convolution_param.num_output.ToString() + ", kernel_size=" + nKernel.ToString() + ", stride=" + nStride.ToString() + ", padding=" + nPad.ToString() + ", dilation=" + nDilation.ToString() + ", groups=" + m_layer.convolution_param.group.ToString() + ", bias=" + m_layer.convolution_param.bias_term.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
                strCode += initWeights(m_layer.name, m_layer.convolution_param.bias_term, m_layer.convolution_param.weight_filler, m_layer.convolution_param.bias_filler);
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class PoolingLayerInfo : LayerInfo /** @private */
    {
        public PoolingLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            int nPad = (layer.pooling_param.pad != null && layer.pooling_param.pad.Count > 0) ? (int)layer.pooling_param.pad[0] : 0;
            int nKernel = (layer.pooling_param.kernel_size != null && layer.pooling_param.kernel_size.Count > 0) ? (int)layer.pooling_param.kernel_size[0] : 1;
            int nStride = (layer.pooling_param.stride != null && layer.pooling_param.stride.Count > 0) ? (int)layer.pooling_param.stride[0] : 1;

            m_outputs[0].Shape[2] = (int)Math.Floor((double)(m_inputs[0].Shape[2] + 2 * nPad - nKernel) / nStride) + 1;
            m_outputs[0].Shape[3] = (int)Math.Floor((double)(m_inputs[0].Shape[3] + 2 * nPad - nKernel) / nStride) + 1;
        }

        public override string Generate(GENERATE gen)
        {
            int nPad = (m_layer.pooling_param.pad != null && m_layer.pooling_param.pad.Count > 0) ? (int)m_layer.pooling_param.pad[0] : 0;
            int nKernel = (m_layer.pooling_param.kernel_size != null && m_layer.pooling_param.kernel_size.Count > 0) ? (int)m_layer.pooling_param.kernel_size[0] : 1;
            int nStride = (m_layer.pooling_param.stride != null && m_layer.pooling_param.stride.Count > 0) ? (int)m_layer.pooling_param.stride[0] : 1;
            int nDilation = (m_layer.pooling_param.dilation != null && m_layer.pooling_param.dilation.Count > 0) ? (int)m_layer.pooling_param.dilation[0] : 1;

            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.MaxPool2d(kernel_size=" + nKernel.ToString() + ", stride=" + nStride.ToString() + ", padding=" + nPad.ToString() + ", dilation=" + nDilation.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else if (gen == GENERATE.FORWARD)
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }
}
