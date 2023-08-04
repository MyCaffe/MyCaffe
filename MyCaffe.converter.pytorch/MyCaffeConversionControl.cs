using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.converter.pytorch.layers;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.param.gpt;
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
        /// <param name="bAddComments">Specifies to add comments to generated code.</param>
        public void ConvertMyCaffeToPyTorch(CudaDnn<T> cuda, Log log, MyCaffeModelData data, string strModelFile, string strOptimizerFile, bool bAddComments)
        {
            m_strOriginalPath = Path.GetDirectoryName(strModelFile);

            m_net = new NetworkInfo(NetParameter.FromProto(RawProto.Parse(data.ModelDescription)), data.InputShape);
            
            if (!string.IsNullOrEmpty(data.SolverDescription))
                m_solver = new SolverInfo(SolverParameter.FromProto(RawProto.Parse(data.SolverDescription)), m_net);

            using (StreamWriter sw = new StreamWriter(strModelFile))
            {
                sw.Write(m_net.Generate(bAddComments));
            }

            using (StreamWriter sw = new StreamWriter(strOptimizerFile))
            {
                sw.Write(m_solver.Generate(strModelFile, bAddComments));
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

        public string Generate(string strModelFile, bool bAddComments)
        {
            string strCode = "";

            strCode += addImports(strModelFile);
            strCode += addGpu();
            strCode += addLoadDataFunction();
            strCode += addDataSet();
            strCode += addTrainerClass();
            strCode += addTraining();

            return strCode;
        }

        private string addImports(string strModelFile)
        {
            string strCode = "";

            FileInfo fi = new FileInfo(strModelFile);

            strCode += "import os" + Environment.NewLine;
            strCode += "import imageio.v2 as iio" + Environment.NewLine;
            strCode += "import pandas as pd" + Environment.NewLine;
            strCode += "import numpy as np" + Environment.NewLine;
            strCode += "import random" + Environment.NewLine;
            strCode += "import torch" + Environment.NewLine;
            strCode += "import albumentations" + Environment.NewLine;
            strCode += "from torch import optim" + Environment.NewLine;
            strCode += "from " + Path.GetFileNameWithoutExtension(fi.Name) + " import " + m_net.Net.name + Environment.NewLine;
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

        private string addLoadDataFunction()
        {
            string strCode = "";

            strCode += "# load the data files" + Environment.NewLine;
            strCode += "def load_data(fullfile):" + Environment.NewLine;
            strCode += "    # load the image paths" + Environment.NewLine;
            strCode += "    img_paths = []" + Environment.NewLine;
            strCode += "    labels = []" + Environment.NewLine;
            strCode += "    for line in open(fullfile):" + Environment.NewLine;
            strCode += "        line = line.split(' ')" + Environment.NewLine;
            strCode += "        img_paths.append(line[0].strip(\'\\n\'))" + Environment.NewLine;
            strCode += "        labels.append(int(line[1].strip('\\n\')))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    return img_paths, labels" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addDataSet()
        {
            string strCode = "";

            strCode += "# Dataset class used to load the data" + Environment.NewLine;
            strCode += "class ImageDataset(torch.utils.data.Dataset):" + Environment.NewLine;
            strCode += "    def __init__(self, path, labels, tfms=None):" + Environment.NewLine;
            strCode += "        self.X = path" + Environment.NewLine;
            strCode += "        self.y = labels" + Environment.NewLine;
            strCode += "        self.img_list = []" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        for i in range(len(self.X)):" + Environment.NewLine;
            strCode += "            self.img_list.append(None)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "        # apply augmentations to the image" + Environment.NewLine;
            strCode += "        if tfms == 0: # if validating" + Environment.NewLine;
            strCode += "           self.transform = albumentations.Compose([" + Environment.NewLine;

            string strMean = getMean();
            if (!string.IsNullOrEmpty(strMean))
                strCode += "               albumentations.Normalize(mean=[" + strMean + "], always_apply=True)" + Environment.NewLine;

            strCode += "           ])" + Environment.NewLine;
            strCode += "        else: # if training" + Environment.NewLine;
            strCode += "           self.transform = albumentations.Compose([" + Environment.NewLine;

            if (m_net.Net.layer[0].type == LayerParameter.LayerType.DATA &&
                m_net.Net.layer[0].transform_param.mirror)
                strCode += "               albumentations.HorizontalFlip(p=1.0)," + Environment.NewLine;

            if (!string.IsNullOrEmpty(strMean))
                strCode += "               albumentations.Normalize(mean=[" + strMean + "], always_apply=True)" + Environment.NewLine;

            strCode += "           ])" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def __len__(self):" + Environment.NewLine;
            strCode += "        return len(self.X)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def __getitem__(self, idx):" + Environment.NewLine;
            strCode += "        if self.img_list[idx] is None:" + Environment.NewLine;
            strCode += "            img = iio.imread(self.X[idx],pilmode=\'RGB\')" + Environment.NewLine;
            strCode += "            img = np.array(img).astype(float)" + Environment.NewLine;

            if (m_net.Net.layer[0].type == LayerParameter.LayerType.DATA &&
                m_net.Net.layer[0].transform_param.color_order == TransformationParameter.COLOR_ORDER.RGB)
                strCode += "            img = np.transpose(img, (2, 0, 1)).astype(np.float32)" + Environment.NewLine;

            if (m_net.Net.layer[0].type == LayerParameter.LayerType.DATA &&
                m_net.Net.layer[0].transform_param.scale != 1.0)
                strCode += "            img = img * " + m_net.Net.layer[0].transform_param.scale + Environment.NewLine;

            strCode += "            img = self.transform(image=img)['image']" + Environment.NewLine;
            strCode += "            self.img_list[idx] = img" + Environment.NewLine;

            strCode += "        img = self.img_list[idx]" + Environment.NewLine;
            strCode += "        label = self.y[idx]" + Environment.NewLine;
            strCode += "        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.long)" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string getMean()
        {
            string strMean = "";
            if (m_net.Net.layer[0].type == LayerParameter.LayerType.DATA && m_net.Net.layer[0].transform_param != null && m_net.Net.layer[0].transform_param.mean_value.Count > 0)
            {
                strMean = m_net.Net.layer[0].transform_param.mean_value[0].ToString();
                for (int i = 1; i < m_net.Net.layer[0].transform_param.mean_value.Count; i++)
                {
                    strMean += "," + m_net.Net.layer[0].transform_param.mean_value[i].ToString();
                }
            }

            return strMean;
        }

        private string addTrainerClass()
        {
            string strCode = "";

            strCode += "# Trainer class used to train the network" + Environment.NewLine;
            strCode += "class Trainer:" + Environment.NewLine;
            strCode += "    def __init__(self, net, opt, device):" + Environment.NewLine;
            strCode += "        self.net = net" + Environment.NewLine;
            strCode += "        self.solver = opt" + Environment.NewLine;
            strCode += "        self.device = device" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def seed_everything(self, SEED=42):" + Environment.NewLine;
            strCode += "        torch.manual_seed(SEED)" + Environment.NewLine;
            strCode += "        torch.cuda.manual_seed(SEED)" + Environment.NewLine;
            strCode += "        torch.cuda.manual_seed_all(SEED)" + Environment.NewLine;
            strCode += "        torch.backends.cudnn.deterministic = True" + Environment.NewLine;
            strCode += "        torch.backends.cudnn.benchmark = True" + Environment.NewLine;
            strCode += "        np.random.seed(SEED)" + Environment.NewLine;
            strCode += "        random.seed(SEED)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def train(self, inputs, labels):" + Environment.NewLine;
            strCode += "        self.net.train()" + Environment.NewLine;
            strCode += "        self.solver.zero_grad()" + Environment.NewLine;
            strCode += "        inputs = inputs.to(self.device)" + Environment.NewLine;
            strCode += "        labels = labels.to(self.device)" + Environment.NewLine;
            strCode += "        loss, acc = self.net(inputs, labels)" + Environment.NewLine;
            strCode += "        loss.backward()" + Environment.NewLine;
            strCode += "        self.solver.step()" + Environment.NewLine;
            strCode += "        return loss, acc" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def test(self, inputs, labels):" + Environment.NewLine;
            strCode += "        self.net.eval()" + Environment.NewLine;
            strCode += "        inputs = inputs.to(self.device)" + Environment.NewLine;
            strCode += "        labels = labels.to(self.device)" + Environment.NewLine;
            strCode += "        loss, acc = self.net(inputs, labels)" + Environment.NewLine;
            strCode += "        return loss, acc" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def save(self, strFile):" + Environment.NewLine;
            strCode += "        torch.save(self.net.state_dict(), strFile)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    def load(self, strFile):" + Environment.NewLine;
            strCode += "        self.net.load_state_dict(torch.load(strFile))" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addTraining()
        {
            string strCode = "";

            strCode += "train_file = 'c:/ProgramData/MyCaffe/test_data/mnist/training/file_list.txt'" + Environment.NewLine;
            strCode += "test_file = 'c:/ProgramData/MyCaffe/test_data/mnist/testing/file_list.txt'" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Load the training and testing data files" + Environment.NewLine;
            strCode += "train_X, train_y = load_data(train_file)" + Environment.NewLine;
            strCode += "test_X, test_y = load_data(test_file)" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Load the training and testing data" + Environment.NewLine;
            strCode += "train_dataset = ImageDataset(train_X, train_y, 1)" + Environment.NewLine;
            strCode += "test_dataset = ImageDataset(test_X, test_y, 0)" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Create the training and testing data loaders" + Environment.NewLine;
            strCode += "train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=" + m_net.batch_size.ToString() + ", shuffle=True)" + Environment.NewLine;
            strCode += "test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=" + m_net.batch_size.ToString() + ", shuffle=True)" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Create the model, optimizer and trainer" + Environment.NewLine;
            strCode += "model = " + m_net.Net.name + "()" + Environment.NewLine;
            strCode += "model.to(device)" + Environment.NewLine;
            strCode += "model.init_weights()" + Environment.NewLine;
            strCode += "opt = optim." + getSolver() + "(filter(lambda p: p.requires_grad, list(model.parameters())), " + getSolverParam() + ")" + Environment.NewLine;
            strCode += "trainer = Trainer(model, opt, device)" + Environment.NewLine;
            strCode += "trainer.seed_everything()" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Training" + Environment.NewLine;
            strCode += "for epoch in range(1, 10):" + Environment.NewLine;
            strCode += "    for i, (inputs, labels) in enumerate(train_loader):" + Environment.NewLine;
            strCode += "        loss, acc = trainer.train(inputs, labels)" + Environment.NewLine;
            strCode += "        print(\"Epoch: %d, Iter: %d, Loss: %f, Accuracy: %f\" % (epoch, i, loss, acc))" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "    for i, (inputs, labels) in enumerate(test_loader):" + Environment.NewLine;
            strCode += "        loss, acc = trainer.test(inputs, labels)" + Environment.NewLine;
            strCode += "        print(\"Epoch: %d, Iter: %d, Loss: %f, Accuracy: %f\" % (epoch, i, loss, acc))" + Environment.NewLine;
            strCode += Environment.NewLine;

            strCode += "# Save the weights" + Environment.NewLine;
            strCode += "current_directory = os.getcwd()" + Environment.NewLine;
            strCode += "final_directory = os.path.join(current_directory, r'weights')" + Environment.NewLine;
            strCode += "if not os.path.exists(final_directory):" + Environment.NewLine;
            strCode += "   os.makedirs(final_directory)" + Environment.NewLine;

            strCode += "wt_file = os.path.join(final_directory,'mnist_weights.wts')" + Environment.NewLine; ;
            strCode += "trainer.save(wt_file)" + Environment.NewLine;
            strCode += Environment.NewLine;
            strCode += "print(\"Training Done!\")" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        private string getSolverParam()
        {
            string strParam = "lr = " + m_solver.base_lr.ToString();

            if (m_solver.momentum != 0)
                strParam += ", momentum = " + m_solver.momentum.ToString();

            if (m_solver.weight_decay != 0)
                strParam += ", weight_decay = " + m_solver.weight_decay.ToString();

            return strParam;
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

        public int batch_size
        {
            get
            {
                double? dfParam = m_layers.GetParameter("batch_size");
                if (dfParam.HasValue)
                    return (int)dfParam.Value;

                return 0;
            }
        }

        public string Generate(bool bAddComments)
        {
            string strCode = "";

            strCode += addCredits();
            strCode += addImports(bAddComments);
            strCode += addClasses(bAddComments);
            strCode += addClass(bAddComments);

            return strCode;
        }

        private string addImports(bool bAddComments)
        {
            string strCode = "";

            strCode += "import random" + Environment.NewLine;
            strCode += "import albumentations" + Environment.NewLine;
            strCode += "import torch" + Environment.NewLine;
            strCode += "import torch.nn as nn" + Environment.NewLine;
            strCode += "import torch.nn.functional as F" + Environment.NewLine;
            strCode += "import torch.nn.init as init" + Environment.NewLine;
            strCode += "import numpy as np" + Environment.NewLine;
            strCode += "import math" + Environment.NewLine;
            strCode += "from typing import List, Dict, Tuple, Optional" + Environment.NewLine;
            strCode += Environment.NewLine;

            return strCode;
        }

        public string addCredits()
        {
            string strCode = "";
            string strCredits = m_layers.Generate(LayerInfo.GENERATE.CREDITS, true);

            if (!string.IsNullOrEmpty(strCredits))
            {
                strCode += "# -----------------------------------------------------------------------------" + Environment.NewLine;
                strCode += "# " + strCredits;

                if (!strCode.EndsWith(Environment.NewLine))
                    strCode += Environment.NewLine;

                strCode += "# -----------------------------------------------------------------------------" + Environment.NewLine;
                strCode += Environment.NewLine;
                strCode += Environment.NewLine;
            }

            return strCode;
        }

        private string addClasses(bool bAddComments)
        {
            string strCode = m_layers.Generate(LayerInfo.GENERATE.CLASSES, bAddComments);

            if (!string.IsNullOrEmpty(strCode))
                strCode += Environment.NewLine;

            return strCode;
        }

        private string addClass(bool bAddComments)
        {
            string strCode = "";

            if (bAddComments)
                strCode += "# Define the model class." + Environment.NewLine;

            strCode += "class " + m_net.name + "(nn.Module):" + Environment.NewLine;
            strCode += addConstructor(bAddComments);
            strCode += addInitWeights(bAddComments);
            strCode += addForward(bAddComments);
            strCode += Environment.NewLine;

            return strCode;
        }

        private string addConstructor(bool bAddComments)
        {
            string strCode = "";

            if (bAddComments)
                strCode += "    # The constructor." + Environment.NewLine;

            strCode += "    def __init__(self):" + Environment.NewLine;
            strCode += "        super(" + m_net.name + ", self).__init__()" + Environment.NewLine;
            strCode += m_layers.Generate(LayerInfo.GENERATE.DEFINITION, bAddComments);
            strCode += Environment.NewLine;
            return strCode;
        }

        private string addInitWeights(bool bAddComments)
        {
            string strCode = "";
            
            if (bAddComments)
                strCode += "    # Initialize the weights." + Environment.NewLine;

            strCode += "    def init_weights(self):" + Environment.NewLine;
            strCode += m_layers.Generate(LayerInfo.GENERATE.INITWEIGHTS, bAddComments);
            strCode += Environment.NewLine;
            return strCode;
        }

        private string addForward(bool bAddComments)
        {
            string strCode = "";
            
            if (bAddComments)
                strCode += "    # The forward method." + Environment.NewLine;

            string strX = m_net.layer[0].top[0];
            string strY = m_net.layer[0].top[1];

            strCode += "    def forward(self, " + strX + ", " + strY + "):" + Environment.NewLine;
            strCode += m_layers.Generate(LayerInfo.GENERATE.FORWARD, bAddComments);
            strCode += "        return " + m_layers.GetReturnValues() + Environment.NewLine;
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

        public string Generate(LayerInfo.GENERATE gen, bool bAddComments)
        {
            string strCode = "";
            for (int i = 0; i < m_rgLayers.Count; i++)
            {
                LayerInfo layer = m_rgLayers[i];
                layer.AddComments = bAddComments;
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

        public double? GetParameter(string strName)
        {
            for (int i = 0; i < m_rgLayers.Count; i++)
            {
                double? dfParam = m_rgLayers[i].GetParameter(strName);
                if (dfParam.HasValue)
                    return dfParam;
            }

            return null;
        }

        public string GetReturnValues()
        {
            Dictionary<string, int> rgRv = new Dictionary<string, int>();

            foreach(LayerInfo layer in m_rgLayers)
            {
                foreach (KeyValuePair<string, int> kv in layer.ReturnValues)
                {
                    rgRv.Add(kv.Key, kv.Value);
                }
            }

            List<KeyValuePair<string, int>> rgRv1 = rgRv.OrderBy(p => p.Value).ToList();

            if (rgRv1.Count == 0)
                return "# Missing Return Value";

            string strReturn = rgRv1[0].Key;

            for (int i=1; i<rgRv1.Count; i++)
            {
                strReturn += ", " + rgRv1[i].Key;
            }

            return strReturn;
        }
    }

    class LayerInfo /** @private */
    {
        protected Dictionary<string, int> m_rgstrReturnValues = new Dictionary<string, int>();
        protected Dictionary<string, double> m_rgParameters = new Dictionary<string, double>();
        protected LayerParameter m_layer;
        protected VariableCollection m_inputs = new VariableCollection();
        protected VariableCollection m_outputs = new VariableCollection();
        protected bool m_bAddComments = false;

        public enum GENERATE
        {
            CREDITS,
            CLASSES,
            DEFINITION,
            INITWEIGHTS,
            FORWARD
        }

        public LayerInfo(LayerParameter layer, VariableCollection inputs)
        {
            m_layer = layer;
            m_inputs = inputs.Clone();

            for (int i=0; i<layer.bottom.Count && i<m_inputs.Count; i++)
            {
                m_inputs[i].Name = layer.bottom[i];
            }   

            m_outputs = m_inputs.Clone();

            for (int i = 0; i < layer.top.Count && i < m_outputs.Count; i++)
            {
                m_outputs[i].Name = layer.top[i];
            }
        }

        public static LayerInfo Create(LayerParameter layer, VariableCollection inputs)
        {
            switch (layer.type)
            {
                /// Data Layes
                /// 
                case LayerParameter.LayerType.DATA:
                    return new DataLayerInfo(layer, inputs);


                /// Common Layers
                /// 

                case LayerParameter.LayerType.CONCAT:
                    return new ConcatLayerInfo(layer, inputs);

                case LayerParameter.LayerType.SPLIT:
                    return new SplitLayerInfo(layer, inputs);

                case LayerParameter.LayerType.INNERPRODUCT:
                    return new InnerProductLayerInfo(layer, inputs);

                case LayerParameter.LayerType.LSTM:
                    return new LSTMLayerInfo(layer, inputs);


                /// Vision Layers
                /// 

                case LayerParameter.LayerType.CONVOLUTION:
                    return new ConvolutionLayerInfo(layer, inputs);

                case LayerParameter.LayerType.POOLING:
                    return new PoolingLayerInfo(layer, inputs);


                /// Activation Layers
                /// 

                case LayerParameter.LayerType.RELU:
                    return new ReluLayerInfo(layer, inputs);

                case LayerParameter.LayerType.ELU:
                    return new ELULayerInfo(layer, inputs);

                case LayerParameter.LayerType.SIGMOID:
                    return new SigmoidLayerInfo(layer, inputs);

                case LayerParameter.LayerType.TANH:
                    return new TanhLayerInfo(layer, inputs);

                case LayerParameter.LayerType.DROPOUT:
                    return new DropoutLayerInfo(layer, inputs);

                case LayerParameter.LayerType.SOFTMAX:
                    return new SoftmaxLayerInfo(layer, inputs);


                /// Normalization Layers
                /// 

                case LayerParameter.LayerType.LRN:
                    return new LRNLayerInfo(layer, inputs);

                case LayerParameter.LayerType.BATCHNORM:
                    return new BatchNormLayerInfo(layer, inputs);

                case LayerParameter.LayerType.LAYERNORM:
                    return new LayerNormLayerInfo(layer, inputs);


                /// Loss Layers
                /// 

                case LayerParameter.LayerType.SOFTMAXWITH_LOSS:
                    return new SoftmaxLossLayerInfo(layer, inputs);

                case LayerParameter.LayerType.ACCURACY:
                    return new AccuracyLayerInfo(layer, inputs);


                /// TFT Layers

                case LayerParameter.LayerType.DATA_TEMPORAL:
                    return new DataTemporalLayerInfo(layer, inputs);

                case LayerParameter.LayerType.CHANNEL_EMBEDDING:
                    return new ChannelEmbeddingLayerInfo(layer, inputs);

                case LayerParameter.LayerType.GLU:
                    return new GLULayerInfo(layer, inputs);

                case LayerParameter.LayerType.GRN:
                    return new GRNLayerInfo(layer, inputs);

                case LayerParameter.LayerType.VARSELNET:
                    return new VarSelNetLayerInfo(layer, inputs);

                case LayerParameter.LayerType.GATEADDNORM:
                    return new GateAddNormLayerInfo(layer, inputs);

                case LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP:
                    return new MultiheadAttentionInterpLayerInfo(layer, inputs);

                case LayerParameter.LayerType.RESHAPE_TEMPORAL:
                    return new ReshapeTemporalLayerInfo(layer, inputs);

                default:
                    return new LayerInfo(layer, inputs);
            }
        }

        protected static string generateBar()
        {
            return "# -----------------------------------------------------------------------------" + Environment.NewLine;
        }

        public Dictionary<string, int> ReturnValues
        {
            get { return m_rgstrReturnValues; }
        }

        public double? GetParameter(string strName)
        {
            if (!m_rgParameters.ContainsKey(strName))
                return null;

            return m_rgParameters[strName];
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

        public bool AddComments
        {
            get { return m_bAddComments; }
            set { m_bAddComments = value; }
        }

        public virtual string Generate(GENERATE gen)
        {
            string strCode = "";
            if (gen == GENERATE.DEFINITION)
                strCode += "#        self." + m_layer.name + " = nn." + m_layer.type.ToString() + "()   # Not Supported" + Environment.NewLine;
            else if (gen == GENERATE.INITWEIGHTS)
                strCode += "#        self." + m_layer.name + ".apply(weights_init)" + Environment.NewLine;
            else if (gen == GENERATE.FORWARD)
                strCode += "#        " + m_outputs.AsText + " = self." + m_layer.name + "(" + m_inputs.AsText + ")  # Not Supported" + Environment.NewLine;

            return strCode;
        }

        protected static string initWeights(string strIndent, string strName, bool bBias, FillerParameter weightFiller, FillerParameter biasFiller)
        {
            string strCode = "";

            if (weightFiller != null)
                strCode += initWeights(strIndent, strName, "weight", weightFiller);

            if (bBias && biasFiller != null)
                strCode += initWeights(strIndent, strName, "bias", biasFiller);

            return strCode;
        }

        protected static string initWeights(string strIndent, string strName, string strItem, FillerParameter filler)
        {
            string strCode = "";
            string strType = getFiller(filler, "self." + strName + "." + strItem + ".data");

            if (filler != null)
                strCode += strIndent + "        init." + strType + Environment.NewLine;

            return strCode;
        }

        protected static string getFiller(FillerParameter filler, string strName)
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
                col.Add(var.Name, new List<int>(var.Shape));
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

        public VariableInfo Clone()
        {
            return new VariableInfo(m_strName, new List<int>(m_rgShape));
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
