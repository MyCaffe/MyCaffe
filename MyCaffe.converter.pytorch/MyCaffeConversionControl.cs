using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.CompilerServices.RuntimeHelpers;

/// <summary>
/// The MyCaffe.converter.pytorch namespace contains the objects used to convert from the MyCaffe and CAFFE model formats to PyTorch *.py models.
/// </summary>
namespace MyCaffe.converter.pytorch
{
    /// <summary>
    /// The MyCaffeConversionControl is used to convert a MyCaffe model to a PyTorch model.
    /// </summary>
    /// <remarks>
    /// WORK IN PROGRESS
    /// </remarks>
    public partial class MyCaffeConversionControl<T> : Component
    {
        string m_strOriginalPath = null;
        NetworkInfo m_net;
        SolverInfo m_solver;


        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeConversionControl()
        {
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">Specifies the container.</param>
        public MyCaffeConversionControl(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        /// <summary>
        /// Convert a MyCaffe model description, weights and optionally mean image from the MyCaffe model format to a Pytorch *.py model file.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="data">Specifies the MyCaffe model data including the model description, the weights and optionally, the image mean.</param>
        /// <param name="strModelFile">Specifies the .py output file for the model.</param>
        /// <param name="strOptimizerFile">Specifies the .py output file for the optimizer.</param>
        public void ConvertMyCaffeToPyTorch(CudaDnn<T> cuda, Log log, MyCaffeModelData data, string strModelFile, string strOptimizerFile)
        {
            m_strOriginalPath = Path.GetDirectoryName(strModelFile);

            m_net = new NetworkInfo(NetParameter.FromProto(RawProto.Parse(data.ModelDescription)), data.InputShape);
            
            if (!string.IsNullOrEmpty(data.SolverDescription))
                m_solver = new SolverInfo(SolverParameter.FromProto(RawProto.Parse(data.SolverDescription)), m_net);

            using (StreamWriter sw = new StreamWriter(strModelFile))
            {
                sw.Write(m_net.Generate());
            }

            using (StreamWriter sw = new StreamWriter(strOptimizerFile))
            {
                sw.Write(m_solver.Generate(strModelFile));
            }
        }
    }

    class SolverInfo /** @private */
    {
        SolverParameter m_solver;
        NetworkInfo m_net;

        public SolverInfo(SolverParameter solver, NetworkInfo net)
        {
            m_solver = solver;
            m_net = net;
        }

        public SolverParameter Solver
        {
            get { return m_solver; }
        }

        public string Generate(string strModelFile)
        {
            string strCode = "";

            strCode += addImports(strModelFile);
            strCode += addGpu();
            strCode += addClass();

            return strCode;
        }

        private string addImports(string strModelFile)
        {
            string strCode = "";

            FileInfo fi = new FileInfo(strModelFile);

            strCode += "import torch" + Environment.NewLine;
            strCode += "from torch import optim" + Environment.NewLine;
            strCode += "from " + fi.Name + " import " + m_net.Net.name + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addGpu()
        {
            string strCode = "";

            strCode += "# Use GPU if available, otherwise use CPU" + Environment.NewLine;
            strCode += "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addClass()
        {
            string strCode = "";

            strCode += "# Trainer class used to train the network" + Environment.NewLine;
            strCode += "class Trainer:" + Environment.NewLine;
            strCode += "    def __init__(self, net, opt, device):" + Environment.NewLine;
            strCode += "        self.net = net" + Environment.NewLine;
            strCode += "        self.solver = opt" + Environment.NewLine;
            strCode += "        self.device = device" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def train(self, inputs, labels):" + Environment.NewLine;
            strCode += "        self.net.train()" + Environment.NewLine;
            strCode += "        inputs = inputs.to(self.device)" + Environment.NewLine;
            strCode += "        labels = labels.to(self.device)" + Environment.NewLine;
            strCode += "        outputs = self.net(inputs)" + Environment.NewLine;
            strCode += "        loss = self.solver.loss(outputs, labels)" + Environment.NewLine;
            strCode += "        loss.backward()" + Environment.NewLine;
            strCode += "        self.solver.step()" + Environment.NewLine;
            strCode += "        return loss" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def test(self, inputs, labels):" + Environment.NewLine;
            strCode += "        self.net.eval()" + Environment.NewLine;
            strCode += "        inputs = inputs.to(self.device)" + Environment.NewLine;
            strCode += "        labels = labels.to(self.device)" + Environment.NewLine;
            strCode += "        outputs = self.net(inputs)" + Environment.NewLine;
            strCode += "        loss = self.solver.loss(outputs, labels)" + Environment.NewLine;
            strCode += "        return loss" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def save(self, strFile):" + Environment.NewLine;
            strCode += "        torch.save(self.net.state_dict(), strFile)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def load(self, strFile):" + Environment.NewLine;
            strCode += "        self.net.load_state_dict(torch.load(strFile))" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Create the model, optimizer and trainer" + Environment.NewLine;
            strCode += "model = " + m_net.Net.name + "()" + Environment.NewLine;
            strCode += "model.to(device)" + Environment.NewLine;
            strCode += "model.init_weights()" + Environment.NewLine;
            strCode += "opt = optim." + getSolver() + "(filter(lambda p: p.requires_grad, list(model.parameters())), lr=" + m_solver.base_lr.ToString() + ")" + Environment.NewLine;
            strCode += "trainer = Trainer(model, opt, device)" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Training" + Environment.NewLine;
            strCode += "for epoch in range(" + m_solver.max_iter.ToString() + "):" + Environment.NewLine;
            strCode += "    for i, (inputs, labels) in enumerate(train_loader):" + Environment.NewLine;
            strCode += "        loss = trainer.train(inputs, labels)" + Environment.NewLine;
            strCode += "        print(\"Epoch: %d, Iter: %d, Loss: %f\" % (epoch, i, loss))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    for i, (inputs, labels) in enumerate(test_loader):" + Environment.NewLine;
            strCode += "        loss = trainer.test(inputs, labels)" + Environment.NewLine;
            strCode += "        print(\"Epoch: %d, Iter: %d, Loss: %f\" % (epoch, i, loss))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "solver.save(\"" + m_solver.snapshot_prefix + "\")" + Environment.NewLine;

            return strCode;
        }

        private string getSolver()
        {
            switch (m_solver.type)
            {
                case SolverParameter.SolverType.SGD:
                    return "SGD";

                case SolverParameter.SolverType.ADAM:
                    return "Adam";

                case SolverParameter.SolverType.ADAMW:
                    return "AdamW";

                case SolverParameter.SolverType.RMSPROP:
                    return "RMSprop";

                case SolverParameter.SolverType.ADAGRAD:
                    return "Adagrad";

                default:
                    return "# ERROR: Unknown solver type '" + m_solver.type.ToString() + "' found!";
            }
        }
    }

    class NetworkInfo /** @private */
    {
        NetParameter m_net;
        LayerInfoCollection m_layers = new LayerInfoCollection();

        public NetworkInfo(NetParameter net, List<int> rgInputShape)
        {
            m_net = net;

            VariableCollection inputs = new VariableCollection();
            inputs.Add(net.layer[0].top[0], rgInputShape);

            foreach (LayerParameter layer in net.layer)
            {
                inputs = Add(layer, inputs);
            }
        }

        public string Generate()
        {
            string strCode = "";

            strCode += addImports();
            strCode += addClass();

            return strCode;
        }

        private string addImports()
        {
            string strCode = "";

            strCode += "import torch" + Environment.NewLine;
            strCode += "import torch.nn as nn" + Environment.NewLine;
            strCode += "import torch.nn.functional as F" + Environment.NewLine;
            strCode += "import torch.nn.init as init" + Environment.NewLine;
            strCode += "import numpy as np" + Environment.NewLine;
            strCode += "import math" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addClass()
        {
            string strCode = "";

            strCode += "# Define the model class." + Environment.NewLine;
            strCode += "class " + m_net.name + "(nn.Module):" + Environment.NewLine;
            strCode += addConstructor();
            strCode += addInitWeights();
            strCode += addForward();
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addConstructor()
        {
            string strCode = "# The constructor." + Environment.NewLine;

            strCode += "    def __init__(self):" + Environment.NewLine;
            strCode += "        super(" + m_net.name + ", self).__init__()" + Environment.NewLine;
            strCode += m_layers.Generate(LayerInfo.GENERATE.DEFINITION);
            strCode += Environment.NewLine;
            return strCode;
        }

        private string addInitWeights()
        {
            string strCode = "# Initialize the weights." + Environment.NewLine;

            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += m_layers.Generate(LayerInfo.GENERATE.INITWEIGHTS);
            strCode += Environment.NewLine;
            return strCode;
        }

        private string addForward()
        {
            string strCode = "# The forward method." + Environment.NewLine;

            strCode += "    def forward(self, x):" + Environment.NewLine;
            strCode += m_layers.Generate(LayerInfo.GENERATE.FORWARD);
            strCode += "        return " + m_layers[m_layers.Count - 1].Outputs[0].Name + Environment.NewLine;
            strCode += Environment.NewLine;
            return strCode;
        }

        public NetParameter Net
        {
            get { return m_net; }
        }

        public LayerInfoCollection Layers
        {
            get { return m_layers; }
        }

        public VariableCollection Add(LayerParameter layer, VariableCollection inputs)
        {
            return m_layers.Add(LayerInfo.Create(layer, inputs));
        }
    }

    class LayerInfoCollection /** @private */
    {
        List<LayerInfo> m_rgLayers = new List<LayerInfo>();

        public LayerInfoCollection()
        {
        }

        public string Generate(LayerInfo.GENERATE gen)
        {
            string strCode = "";
            for (int i = 0; i < m_rgLayers.Count; i++)
            {
                LayerInfo layer = m_rgLayers[i];
                strCode += layer.Generate(gen);
            }
            return strCode;
        }

        public List<LayerInfo> Layers
        {
            get { return m_rgLayers; }
        }

        public VariableCollection Add(LayerInfo layer)
        {
            m_rgLayers.Add(layer);
            return layer.Outputs;
        }

        public int Count
        {
            get { return m_rgLayers.Count; }
        }

        public LayerInfo this[int nIdx]
        {
            get { return m_rgLayers[nIdx]; }
        }
    }

    class ConvolutionLayerInfo : LayerInfo /** @private */
    {
        public ConvolutionLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_outputs[0].Shape[1] = (int)layer.convolution_param.num_output;
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
            else
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
            else
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class InnerProductLayerInfo : LayerInfo /** @private */
    {
        public InnerProductLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
            m_outputs[0].Shape[1] = (int)layer.inner_product_param.num_output;
        }

        public override string Generate(GENERATE gen)
        {
            int nInFeatures = m_inputs[0].getCount(m_layer.inner_product_param.axis);
            int nOutFeatures = (int)m_layer.inner_product_param.num_output;

            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.Linear(in_features=" + nInFeatures + ", out_features=" + nOutFeatures + ", bias=" + m_layer.inner_product_param.bias_term.ToString() + ")" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
                initWeights(m_layer.name, m_layer.inner_product_param.bias_term, m_layer.inner_product_param.weight_filler, m_layer.inner_product_param.bias_filler);
            else
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

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
            else
                strCode += "        " + m_outputs.AsText + " = torch.concat(" + m_inputs.AsText + ", dim=0)" + Environment.NewLine;

            return strCode;
        }
    }

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
            else
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
            else
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
            else
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

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
            else
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
            else
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
            else
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class SoftmaxLossLayerInfo : SoftmaxLayerInfo /** @private */
    {
        public SoftmaxLossLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
        }

        public override string Generate(GENERATE gen)
        {
            string strCode = base.Generate(gen);

            if (gen == GENERATE.DEFINITION)
                strCode += "        self." + m_layer.name + " = nn.CrossEntropyLoss()" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
            {
            }
            else
                strCode += "        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")" + Environment.NewLine;

            return strCode;
        }
    }

    class AccuracyLayerInfo : LayerInfo /** @private */
    {
        public AccuracyLayerInfo(LayerParameter layer, VariableCollection inputs) : base(layer, inputs)
        {
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
            else
            {
                strCode += "        self.accuracy_sum += torch.sum(" + m_inputs.AsText + ")" + Environment.NewLine;
                strCode += "        self.accuracy_count += 1" + Environment.NewLine;
                strCode += "        self.accuracy = self.accuracy_sum / self.accuracy_count" + Environment.NewLine;
            }

            return strCode;
        }
    }


    class LayerInfo /** @private */
    {
        protected LayerParameter m_layer;
        protected VariableCollection m_inputs = new VariableCollection();
        protected VariableCollection m_outputs = new VariableCollection();

        public enum GENERATE
        {
            DEFINITION,
            INITWEIGHTS,
            FORWARD
        }

        public LayerInfo(LayerParameter layer, VariableCollection inputs)
        {
            m_layer = layer;
            m_inputs = inputs;
            m_outputs = m_inputs.Clone();
            m_outputs[0].Name = layer.top[0];
        }

        public static LayerInfo Create(LayerParameter layer, VariableCollection inputs)
        {
            switch (layer.type)
            {
                case LayerParameter.LayerType.CONVOLUTION:
                    return new ConvolutionLayerInfo(layer, inputs);

                case LayerParameter.LayerType.POOLING:
                    return new PoolingLayerInfo(layer, inputs);

                case LayerParameter.LayerType.INNERPRODUCT:
                    return new InnerProductLayerInfo(layer, inputs);

                case LayerParameter.LayerType.CONCAT:
                    return new ConcatLayerInfo(layer, inputs);

                case LayerParameter.LayerType.LRN:
                    return new LRNLayerInfo(layer, inputs);

                case LayerParameter.LayerType.RELU:
                    return new ReluLayerInfo(layer, inputs);

                case LayerParameter.LayerType.SIGMOID:
                    return new SigmoidLayerInfo(layer, inputs);

                case LayerParameter.LayerType.TANH:
                    return new TanhLayerInfo(layer, inputs);

                case LayerParameter.LayerType.DROPOUT:
                    return new DropoutLayerInfo(layer, inputs);

                case LayerParameter.LayerType.BATCHNORM:
                    return new BatchNormLayerInfo(layer, inputs);

                case LayerParameter.LayerType.LAYERNORM:
                    return new LayerNormLayerInfo(layer, inputs);

                case LayerParameter.LayerType.SOFTMAX:
                    return new SoftmaxLayerInfo(layer, inputs);

                case LayerParameter.LayerType.SOFTMAXWITH_LOSS:
                    return new SoftmaxLossLayerInfo(layer, inputs);

                case LayerParameter.LayerType.ACCURACY:
                    return new AccuracyLayerInfo(layer, inputs);

                default:
                    return new LayerInfo(layer, inputs);
            }
        }

        public LayerParameter Layer
        {
            get { return m_layer; }
        }

        public VariableCollection Inputs
        {
            get { return m_inputs; }
        }

        public VariableCollection Outputs
        {
            get { return m_outputs; }
        }

        public virtual string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "#        self." + m_layer.name + " = nn." + m_layer.type.ToString() + "()   # Not Supported" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
                strCode += "#        self." + m_layer.name + ".apply(weights_init)" + Environment.NewLine;
            else
                strCode += "#        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")  # Not Supported" + Environment.NewLine;

            return strCode;
        }

        protected string initWeights(string strName, bool bBias, FillerParameter weightFiller, FillerParameter biasFiller)
        {
            string strCode = "";

            if (weightFiller != null)
                strCode += initWeights(strName, "weight", weightFiller);

            if (bBias && biasFiller != null)
                strCode += initWeights(strName, "bias", biasFiller);

            return strCode;
        }

        protected string initWeights(string strName, string strItem, FillerParameter filler)
        {
            string strCode = "";
            string strType = getFiller(filler, strName + "." + strItem + ".data");

            if (filler != null)
                strCode += "        init." + strType + Environment.NewLine;

            return strCode;
        }

        protected string getFiller(FillerParameter filler, string strName)
        {
            if (filler == null)
                return "uniform_(" + strName + ", a=" + filler.min.ToString() + ", b=" + filler.max.ToString() + ")";

            switch (filler.FillerTypeMethod)
            {
                case FillerParameter.FillerType.CONSTANT:
                    return "constant_(" + strName + ", " + filler.value.ToString() + ")";

                case FillerParameter.FillerType.GAUSSIAN:
                    return "normal_(" + strName + ", mean=" + filler.mean.ToString() + ", std=" + filler.std.ToString() + ")";

                case FillerParameter.FillerType.XAVIER:
                    return "xavier_normal_(" + strName + ")";

                case FillerParameter.FillerType.MSRA:
                    if (filler.variance_norm == FillerParameter.VarianceNorm.FAN_IN)
                        return "kaiming_uniform_(" + strName + ", mode: str='fan_in')";
                    else if (filler.variance_norm == FillerParameter.VarianceNorm.FAN_OUT)
                        return "kaiming_uniform_(" + strName + ", mode: str='fan_out')";
                    else
                        return "kaiming_uniform_(" + strName + ")";

                case FillerParameter.FillerType.POSITIVEUNITBALL:
                    return "uniform_(" + strName + ", a=" + filler.min.ToString() + ", b=" + filler.max.ToString() + ")";

                case FillerParameter.FillerType.UNIFORM:
                    return "uniform_(" + strName + ", a=" + filler.min.ToString() + ", b=" + filler.max.ToString() + ")";

                default:
                    return "uniform_(" + strName + ", a=" + filler.min.ToString() + ", b=" + filler.max.ToString() + ")";
            }
        }
    }

    class VariableCollection /** @private */
    {
        List<VariableInfo> m_rgVariables = new List<VariableInfo>();
        public VariableCollection()
        {
        }

        public void Add(string strName, List<int> rgShape)
        {
            m_rgVariables.Add(new VariableInfo(strName, rgShape));
        }

        public VariableCollection Clone(int nMax = int.MaxValue)
        {
            VariableCollection col = new VariableCollection();
            int nIdx = 0;

            foreach (VariableInfo var in m_rgVariables)
            {
                col.Add(var.Name, var.Shape);
                nIdx++;

                if (nIdx >= nMax)
                    break;
            }

            return col;
        }

        public string AsText
        {
            get
            {
                string strOut = "";

                foreach (VariableInfo var in m_rgVariables)
                {
                    strOut += var.Name + ",";
                }

                return strOut.TrimEnd(',');
            }
        }

        public int Count
        {
            get { return m_rgVariables.Count; }
        }

        public VariableInfo this[int nIdx]
        {
            get { return m_rgVariables[nIdx]; }
        }

        public VariableInfo Find(string strName)
        {
            foreach (VariableInfo var in m_rgVariables)
            {
                if (var.Name == strName)
                    return var;
            }

            return null;
        }
    }

    class VariableInfo /** @private */
    {
        string m_strName;
        List<int> m_rgShape;

        public VariableInfo(string strName, List<int> rgShape)
        {
            m_strName = strName;
            m_rgShape = rgShape;
        }

        public string Name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        public List<int> Shape
        {
            get { return m_rgShape; }
        }

        public int getCount(int nAxis)
        {
            int nCount = 1;

            for (int i = nAxis; i < m_rgShape.Count; i++)
            {
                nCount *= m_rgShape[i];
            }

            return nCount;
        }
    }
}
