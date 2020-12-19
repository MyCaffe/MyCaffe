using Google.Protobuf;
using Google.Protobuf.Collections;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.param.beta;
using Onnx;
using OnnxControl;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.converter.onnx namespace contains the objects used to convert to/from the MyCaffe and CAFFE model formats from/to the ONNX model format.  
/// </summary>
/// <remarks>
/// @see [ONNX Syntax](https://github.com/onnx/onnx/blob/master/docs/IR.md) for more information on the ONNX model format.
/// </remarks>
namespace MyCaffe.converter.onnx
{
    /// <summary>
    /// The MyCaffeConversionControl handles converting between MyCaffe and ONNX formats.  The OnnxControl is used to read and write ONNX files.
    /// </summary>
    /// <typeparam name="T">Specifies the base type used by MyCaffe which is either <i>float</i> or <i>double</i>.</typeparam>
    public partial class MyCaffeConversionControl<T> : Component
    {
        string m_strReport = "";
        string m_strOriginalPath = null;
        bool m_bEnableBackward = false;
        double? m_dfWtScaleMin = null;
        double? m_dfWtScaleMax = null;
        List<string> m_rgstrIgnoreLayerNames = new List<string>();
        int m_nReshapeCount = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeConversionControl()
        {
            InitializeComponent();
            m_strOriginalPath = AssemblyDirectory;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">A container holding the component.</param>
        public MyCaffeConversionControl(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
            m_strOriginalPath = AssemblyDirectory;
        }

        private static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }

        /// <summary>
        /// Set the scaling factors applied to the weights.
        /// </summary>
        /// <param name="dfMin">Specifies the minimum of the range.</param>
        /// <param name="dfMax">Specifies the maximum of the range.</param>
        public void SetWeightScaling(double dfMin, double dfMax)
        {
            m_dfWtScaleMax = dfMax;
            m_dfWtScaleMin = dfMin;
        }

        /// <summary>
        /// Get/set the list of layer names to ignore (layers are ignored when they contain the text from one of these items).
        /// </summary>
        public List<string> IgnoreLayerNames
        {
            get { return m_rgstrIgnoreLayerNames; }
            set { m_rgstrIgnoreLayerNames = value; }
        }

        /// <summary>
        /// Returns the report from the conversion.
        /// </summary>
        public string ReportString
        {
            get { return m_strReport; }
        }

        /// <summary>
        /// Convert a MyCaffe model description, weights and optionally mean image from the MyCaffe model format to the ONNX format and save the result as 
        /// a .onnx model file.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="data">Specifies the MyCaffe model data including the model description, the weights and optionally, the image mean.</param>
        /// <param name="strOutputFile">Specifies the .onnx output file.</param>
        /// <param name="nOpSetVersion">Specifies the Operation set version (default = 9).</param>
        /// <param name="bUseRawData">Optionally, specifies whether or not to store tensor data as RawData or as the native FloatData or DoubleData (default = true).</param>
        /// <param name="dstDataType">Optionally, specifies the output data type, which currently can be either FLOAT or DOUBLE (default = FLOAT).</param>
        public void ConvertMyCaffeToOnnxFile(CudaDnn<T> cuda, Log log, MyCaffeModelData data, string strOutputFile, int nOpSetVersion = 9, bool bUseRawData = true, OnnxDefinitions.DataType dstDataType = OnnxDefinitions.DataType.FLOAT)
        {
            m_strOriginalPath = Path.GetDirectoryName(strOutputFile);
            ModelProto protoOnnx = ConvertMyCaffeToOnnx(cuda, log, data, nOpSetVersion, bUseRawData, dstDataType);
            PersistOnnx persist = new PersistOnnx();

            // Save the new model
            persist.Save(protoOnnx, strOutputFile);
        }

        /// <summary>
        /// Convert a MyCaffe model description, weights and optionally mean image from the MyCaffe model format to the ONNX format and return the 
        /// ONNX ModelProto object containing the model in the ONNX format.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="data">Specifies the MyCaffe model data including the model description, the weights and optionally, the image mean.</param>
        /// <param name="nOpSetVersion">Specifies the Operation set version (default = 9).</param>
        /// <param name="bUseRawData">Optionally, specifies whether or not to store tensor data as RawData or as the native FloatData or DoubleData (default = true).</param>
        /// <param name="dstDataType">Optionally, specifies the output data type, which currently can be either FLOAT or DOUBLE (default = FLOAT).</param>
        /// <returns>The model is returned in the ONNX format as a ModelProto (defined within the OnnxControl)</returns>
        public ModelProto ConvertMyCaffeToOnnx(CudaDnn<T> cuda, Log log, MyCaffeModelData data, int nOpSetVersion = 9, bool bUseRawData = true, OnnxDefinitions.DataType dstDataType = OnnxDefinitions.DataType.FLOAT)
        {
            // Parse the Caffe Model Description;
            RawProto protoMyCaffe = RawProto.Parse(data.ModelDescription);
            NetParameter netParam = NetParameter.FromProto(protoMyCaffe);

            Net<T> net = new Net<T>(cuda, log, netParam, new CancelEvent(), null);

            // Load the weights
            if (data.Weights != null)
            {
                PersistCaffe<T> persistCaffe = new PersistCaffe<T>(log, false);
                net.LoadWeights(data.Weights, persistCaffe);
            }

            // Convert the MyCaffe net to an Onnx model.
            ModelProto protoOnnx = convertToOnnx(log, net, nOpSetVersion, bUseRawData, dstDataType);

            // Cleanup
            if (net != null)
                net.Dispose();

            return protoOnnx;
        }

        /// <summary>
        /// Convert a model currently loaded into the MyCaffeControl to an ONNX ModelProto.
        /// </summary>
        /// <param name="ctrl">Specifies the MyCaffeControl object.</param>
        /// <param name="nOpSetVersion">Specifies the Operation set version (default = 9).</param>
        /// <param name="bUseRawData">Optionally, specifies whether or not to store tensor data as RawData or as the native FloatData or DoubleData (default = true).</param>
        /// <param name="dstDataType">Optionally, specifies the output data type, which currently can be either FLOAT or DOUBLE (default = FLOAT).</param>
        /// <param name="phase">Optionally, specifies the phase (which netork) to convert (default = RUN).</param>
        /// <returns>The ONNX model proto is returns that matches the network converted.</returns>
        public ModelProto ConvertMyCaffeToOnnx(MyCaffeControl<T> ctrl, int nOpSetVersion = 9, bool bUseRawData = true, OnnxDefinitions.DataType dstDataType = OnnxDefinitions.DataType.FLOAT, Phase phase = Phase.RUN)
        {
            Net<T> net = ctrl.GetInternalNet(phase);
            return convertToOnnx(ctrl.Log, net, nOpSetVersion, bUseRawData, dstDataType);
        }

        /// <summary>
        /// Convert a model currently loaded into the MyCaffeControl to an ONNX .onnx model file.
        /// </summary>
        /// <param name="ctrl">Specifies the MyCaffeControl object.</param>
        /// <param name="strOnnxFile">Specifies the output .onnx file.</param>
        /// <param name="nOpSetVersion">Specifies the Operation set version (default = 9).</param>
        /// <param name="bUseRawData">Optionally, specifies whether or not to store tensor data as RawData or as the native FloatData or DoubleData (default = true).</param>
        /// <param name="dstDataType">Optionally, specifies the output data type, which currently can be either FLOAT or DOUBLE (default = FLOAT).</param>
        /// <param name="phase">Optionally, specifies the phase (which netork) to convert (default = RUN).</param>
        public void ConvertMyCaffeToOnnxFile(MyCaffeControl<T> ctrl, string strOnnxFile, int nOpSetVersion = 9, bool bUseRawData = true, OnnxDefinitions.DataType dstDataType = OnnxDefinitions.DataType.FLOAT, Phase phase = Phase.RUN)
        {
            m_strOriginalPath = Path.GetDirectoryName(strOnnxFile);
            ModelProto proto = ConvertMyCaffeToOnnx(ctrl, nOpSetVersion, bUseRawData, dstDataType, phase);
            PersistOnnx persist = new PersistOnnx();
            persist.Save(proto, strOnnxFile);
        }

        /// <summary>
        /// Convert an ONNX .onnx model file to the MyCaffe model description, weights and optionally mean image.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="strOnnxFile">Specifies the ONNX .onnx file.</param>
        /// <param name="bFixlupNeuronNodes">Optionally, specifies to fixup the neuron nodes (e.g. Relu, Prelu, Elu, Sigmoid, Tahn, etc.) by connecting them to inline nodes by connnecting them back to their parent which is common in Caffe type models (default = true).</param>
        /// <param name="bIncludeLastLayerWeights">Optionally, specifies to include the weights for the last layer (default = false, usually not included for transfer learning).</param>
        /// <param name="dsTraining">Optionally, specifies a training dataset which when supplied converts the model to a training model where inputs 
        /// are replaced with data layers, and outputs (e.g. softmax) with loss and accuracy layers (default = false).</param>
        /// <returns>The MyCaffe model description, model weights and image mean are returned as a MyCaffeModelData object.</returns>
        public MyCaffeModelData ConvertOnnxToMyCaffeFromFile(CudaDnn<T> cuda, Log log, string strOnnxFile, bool bFixlupNeuronNodes = true, bool bIncludeLastLayerWeights = false, DatasetDescriptor dsTraining = null)
        {
            m_strOriginalPath = Path.GetDirectoryName(strOnnxFile);
            PersistOnnx persist = new PersistOnnx();
            ModelProto proto = persist.Load(strOnnxFile);
            MyCaffeModelData data = ConvertOnnxToMyCaffe(cuda, log, proto, bFixlupNeuronNodes, bIncludeLastLayerWeights, dsTraining);
            data.OriginalDownloadFile = strOnnxFile;

            return data;
        }

        /// <summary>
        /// Convert an ONNX ModelProto to the MyCaffe model description, weights and optionally mean image.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="onnxModel">Specifies the ONNX model.</param>
        /// <param name="bFixupNeuronNodes">Optionally, specifies to fixup the neuron nodes (e.g. Relu, Prelu, Elu, Sigmoid, Tahn, etc.) by connecting them to inline nodes by connnecting them back to their parent which is common in Caffe type models (default = true).</param>
        /// <param name="bIncludeLastLayerWeights">Optionally, specifies to include the weights for the last layer (default = false, usually not included for transfer learning).</param>
        /// <param name="dsTraining">Optionally, specifies a training dataset which when supplied converts the model to a training model where inputs 
        /// are replaced with data layers, and outputs (e.g. softmax) with loss and accuracy layers (default = false).</param>
        /// <returns>The MyCaffe model description, model weights and image mean are returned as a MyCaffeModelData object.</returns>
        public MyCaffeModelData ConvertOnnxToMyCaffe(CudaDnn<T> cuda, Log log, ModelProto onnxModel, bool bFixupNeuronNodes = true, bool bIncludeLastLayerWeights = false, DatasetDescriptor dsTraining = null)
        {
            Tuple<NetParameter, BlobCollection<T>> data = convertToMyCaffe(cuda, log, onnxModel, bFixupNeuronNodes, dsTraining);

            NetParameter netParam = data.Item1;
            RawProto protoMyCaffe = netParam.ToProto("root");

            if (!bIncludeLastLayerWeights && data.Item2.Count > 0)
                data.Item2.RemoveAt(data.Item2.Count - 1);

            PersistCaffe<T> persist = new PersistCaffe<T>(log, false);
            byte[] rgWeights = persist.SaveWeights(data.Item2, false);

            return new MyCaffeModelData(protoMyCaffe.ToString(), rgWeights);
        }

        private ModelProto convertToOnnx(Log log, Net<T> net, int nOpSetVersion = 9, bool bUseRawData = true, OnnxDefinitions.DataType dstDataType = OnnxDefinitions.DataType.FLOAT)
        {
            ModelProto proto = new ModelProto();
            NetParameter netParam = net.net_param;

            Assembly assembly = Assembly.GetExecutingAssembly();
            FileVersionInfo ver = FileVersionInfo.GetVersionInfo(assembly.Location);

            proto.IrVersion = 1;
            proto.ProducerName = "MyCaffe Converter for ONNX";
            proto.ProducerVersion = ver.FileVersion;
            proto.Domain = "org.mycaffe";
            proto.ModelVersion = 1;

            OperatorSetIdProto opset = new OperatorSetIdProto();
            opset.Version = nOpSetVersion;
            opset.Domain = "";
            proto.OpsetImport.Add(opset);

            StringStringEntryProto author = new StringStringEntryProto();
            author.Key = "model_author";
            author.Value = "SignalPop LLC";
            proto.MetadataProps.Add(author);

            StringStringEntryProto license = new StringStringEntryProto();
            license.Key = "model_license";
            license.Value = "https://github.com/MyCaffe/MyCaffe/blob/master/LICENSE";
            proto.MetadataProps.Add(license);

            proto.Graph = new GraphProto();
            proto.Graph.Name = netParam.name;
            addValueInfo(proto.Graph.Input, net.input_blobs);
            addValueInfo(proto.Graph.Output, net.output_blobs);
            addNodes(log, proto.Graph.Node, net.layers, proto.Graph.Initializer);
            addTensors(proto.Graph.Initializer, net.learnable_parameters, bUseRawData, dstDataType);

            return proto;
        }

        private void addValueInfo(RepeatedField<ValueInfoProto> rg, BlobCollection<T> blobs)
        {
            foreach (Blob<T> blob in blobs)
            {
                ValueInfoProto val = new ValueInfoProto();
                val.Name = blob.Name;

                TypeProto type = new TypeProto();
                type.TensorType = new TypeProto.Types.Tensor();
                type.TensorType.ElemType = (int)OnnxDefinitions.DataType.FLOAT;
                type.TensorType.Shape = new TensorShapeProto();

                foreach (int nShape in blob.shape())
                {
                    TensorShapeProto.Types.Dimension dim = new TensorShapeProto.Types.Dimension();
                    dim.DimValue = nShape;
                    type.TensorType.Shape.Dim.Add(dim);
                }

                val.Type = type;
                rg.Add(val);
            }
        }

        private string removeWs(string str, char ch)
        {
            string strOut = "";

            foreach (char ch1 in str)
            {
                if (char.IsWhiteSpace(ch1))
                    strOut += ch;
                else
                    strOut += ch1;
            }

            return strOut;
        }

        private void addShapeTensor(RepeatedField<TensorProto> rg, string strName, List<int> rgShape)
        {
            TensorProto tensor = new TensorProto();
            tensor.Name = strName;
            tensor.DataType = (int)OnnxDefinitions.DataType.INT64;
            tensor.Dims.Add(rgShape.Count);

            for (int i = 0; i < rgShape.Count; i++)
            {
                tensor.Int64Data.Add(rgShape[i]);
            }

            rg.Add(tensor);
        }

        private void addTensors(RepeatedField<TensorProto> rg, BlobCollection<T> blobs, bool bUseRawData = true, OnnxDefinitions.DataType dstDataType = OnnxDefinitions.DataType.FLOAT)
        {
            foreach (Blob<T> blob in blobs)
            {
                TensorProto tensor = new TensorProto();
                tensor.Name = removeWs(blob.Name, '_');
                tensor.DataType = (int)dstDataType;

                foreach (int nShape in blob.shape())
                {
                    tensor.Dims.Add(nShape);
                }

                T[] rgData = blob.mutable_cpu_data;

                if (bUseRawData)
                {
                    if (dstDataType == OnnxDefinitions.DataType.FLOAT)
                    {
                        if (typeof(T) == typeof(float))
                        {
                            float[] rgfData = Utility.ConvertVecF<T>(rgData);
                            byte[] rgByte = new byte[rgfData.Length * sizeof(float)];
                            Buffer.BlockCopy(rgfData, 0, rgByte, 0, rgByte.Length);
                            tensor.RawData = ByteString.CopyFrom(rgByte);
                        }
                        else
                        {
                            double[] rgfData = Utility.ConvertVec<T>(rgData);
                            float[] rgfData2 = rgfData.Select(p => (float)p).ToArray();
                            byte[] rgByte = new byte[rgfData2.Length * sizeof(float)];
                            Buffer.BlockCopy(rgfData2, 0, rgByte, 0, rgByte.Length);
                            tensor.RawData = ByteString.CopyFrom(rgByte);
                        }
                    }
                    else if (dstDataType == OnnxDefinitions.DataType.DOUBLE)
                    {
                        if (typeof(T) == typeof(float))
                        {
                            float[] rgfData = Utility.ConvertVecF<T>(rgData);
                            double[] rgfData2 = rgfData.Select(p => (double)p).ToArray();
                            byte[] rgByte = new byte[rgfData2.Length * sizeof(double)];
                            Buffer.BlockCopy(rgfData2, 0, rgByte, 0, rgByte.Length);
                            tensor.RawData = ByteString.CopyFrom(rgByte);
                        }
                        else
                        {
                            double[] rgfData = Utility.ConvertVec<T>(rgData);
                            byte[] rgByte = new byte[rgfData.Length * sizeof(double)];
                            Buffer.BlockCopy(rgfData, 0, rgByte, 0, rgByte.Length);
                            tensor.RawData = ByteString.CopyFrom(rgByte);
                        }
                    }
                    else
                        throw new Exception("Currently only the FLOAT and DOUBLE data types are supported when exporting.");
                }
                else
                {
                    if (dstDataType == OnnxDefinitions.DataType.FLOAT)
                    {
                        if (typeof(T) == typeof(float))
                        {
                            float[] rgfData = Utility.ConvertVecF<T>(rgData);

                            foreach (float val in rgfData)
                            {
                                tensor.FloatData.Add(val);
                            }
                        }
                        else
                        {
                            double[] rgfData = Utility.ConvertVec<T>(rgData);

                            foreach (double val in rgfData)
                            {
                                tensor.FloatData.Add(Convert.ToSingle(val));
                            }
                        }
                    }
                    else if (dstDataType == OnnxDefinitions.DataType.DOUBLE)
                    {
                        if (typeof(T) == typeof(float))
                        {
                            float[] rgfData = Utility.ConvertVecF<T>(rgData);

                            foreach (float val in rgfData)
                            {
                                tensor.DoubleData.Add(Convert.ToDouble(val));
                            }
                        }
                        else
                        {
                            double[] rgfData = Utility.ConvertVec<T>(rgData);

                            foreach (double val in rgfData)
                            {
                                tensor.DoubleData.Add(val);
                            }
                        }
                    }
                    else
                        throw new Exception("Currently only the FLOAT and DOUBLE data types are supported when exporting.");
                }

                rg.Add(tensor);
            }
        }

        private void addNodes(Log log, RepeatedField<NodeProto> rg, List<Layer<T>> rgLayers, RepeatedField<TensorProto> rgTensors)
        {
            Dictionary<string, List<string>> rgTopCounts = new Dictionary<string, List<string>>();
            LayerParameter.LayerType m_lastType = LayerParameter.LayerType.DATA;
            bool m_bReshapeBeforeInnerProductNeeded = true;

            foreach (Layer<T> layer in rgLayers)
            {
                NodeProto node = new NodeProto();

                node.Name = layer.layer_param.name;

                foreach (string strBottom in layer.layer_param.bottom)
                {
                    string strBtm1 = strBottom;

                    if (rgTopCounts.ContainsKey(strBottom))
                        strBtm1 = rgTopCounts[strBottom].Last();

                    node.Input.Add(strBottom);
                }

                foreach (string strTop in layer.layer_param.top)
                {
                    if (!rgTopCounts.ContainsKey(strTop))
                        rgTopCounts.Add(strTop, new List<string>() { strTop });
                    else
                        rgTopCounts[strTop].Add(strTop + "_" + rgTopCounts[strTop].Count.ToString());

                    string strTop1 = rgTopCounts[strTop].Last();
                    node.Output.Add(strTop1);
                }

                BlobCollection<T> colParams = new BlobCollection<T>();

                switch (layer.type)
                {
                    case LayerParameter.LayerType.ABSVAL:
                        node.OpType = OnnxDefinitions.OPERATORS.Abs.ToString();
                        break;

                    case LayerParameter.LayerType.ARGMAX:
                        if (layer.layer_param.argmax_param.operation == ArgMaxParameter.COMPARE_OPERATOR.MIN)
                            node.OpType = OnnxDefinitions.OPERATORS.ArgMin.ToString();
                        else
                            node.OpType = OnnxDefinitions.OPERATORS.ArgMax.ToString();
                        break;

                    case LayerParameter.LayerType.ELTWISE:
                        if (layer.layer_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.SUM)
                            node.OpType = OnnxDefinitions.OPERATORS.Add.ToString();
                        if (layer.layer_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.SUB)
                            node.OpType = OnnxDefinitions.OPERATORS.Sub.ToString();
                        else if (layer.layer_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.PROD)
                            node.OpType = OnnxDefinitions.OPERATORS.Mul.ToString();
                        else if (layer.layer_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.DIV)
                            node.OpType = OnnxDefinitions.OPERATORS.Div.ToString();
                        else if (layer.layer_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.MAX)
                            node.OpType = OnnxDefinitions.OPERATORS.Max.ToString();
                        else if (layer.layer_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.MIN)
                            node.OpType = OnnxDefinitions.OPERATORS.Min.ToString();
                        break;

                    case LayerParameter.LayerType.BATCHNORM:
                        node.OpType = OnnxDefinitions.OPERATORS.BatchNormalization.ToString();
                        addAttributes(node.Attribute, layer.layer_param.batch_norm_param);
                        break;

                    case LayerParameter.LayerType.CLIP:
                        node.OpType = OnnxDefinitions.OPERATORS.Clip.ToString();
                        addAttributes(node.Attribute, layer.layer_param.clip_param);
                        break;

                    case LayerParameter.LayerType.CONCAT:
                        node.OpType = OnnxDefinitions.OPERATORS.Concat.ToString();
                        addAttributes(node.Attribute, layer.layer_param.concat_param);
                        break;

                    case LayerParameter.LayerType.CONSTANT:
                        node.OpType = OnnxDefinitions.OPERATORS.Constant.ToString();
                        addAttributes(node.Attribute, layer.layer_param.constant_param);
                        break;

                    case LayerParameter.LayerType.CONVOLUTION:
                        node.OpType = OnnxDefinitions.OPERATORS.Conv.ToString();
                        addAttributes(node.Attribute, layer.layer_param.convolution_param);
                        colParams.Add(layer.blobs[0]);
                        if (layer.layer_param.convolution_param.bias_term)
                            colParams.Add(layer.blobs[1]);
                        break;

                    case LayerParameter.LayerType.DROPOUT:
                        node.OpType = OnnxDefinitions.OPERATORS.Dropout.ToString();
                        addAttributes(node.Attribute, layer.layer_param.dropout_param);
                        break;

                    case LayerParameter.LayerType.ELU:
                        node.OpType = OnnxDefinitions.OPERATORS.Elu.ToString();
                        addAttributes(node.Attribute, layer.layer_param.elu_param);
                        break;

                    case LayerParameter.LayerType.EXP:
                        node.OpType = OnnxDefinitions.OPERATORS.Exp.ToString();
                        addAttributes(node.Attribute, layer.layer_param.exp_param);
                        break;

                    case LayerParameter.LayerType.FLATTEN:
                        node.OpType = OnnxDefinitions.OPERATORS.Flatten.ToString();
                        addAttributes(node.Attribute, layer.layer_param.flatten_param);
                        break;

                    case LayerParameter.LayerType.GATHER:
                        node.OpType = OnnxDefinitions.OPERATORS.Gather.ToString();
                        addAttributes(node.Attribute, layer.layer_param.gather_param);
                        break;

                    case LayerParameter.LayerType.INNERPRODUCT:
                        if (m_lastType != LayerParameter.LayerType.RESHAPE && m_bReshapeBeforeInnerProductNeeded)
                        {
                            NodeProto node1 = new NodeProto();
                            node1.OpType = OnnxDefinitions.OPERATORS.Reshape.ToString();
                            node1.Name = "reshape" + m_nReshapeCount.ToString();
                            node1.Input.Add(node.Input[0]);
                            node1.Input.Add(node1.Name);
                            string strOutput = node1.Name + "_out";
                            node1.Output.Add(strOutput);
                            m_nReshapeCount++;

                            ReshapeParameter reshape_param = new ReshapeParameter();
                            reshape_param.axis = 1;
                            reshape_param.num_axes = -1;
                            reshape_param.shape.dim.Add(-1);
                            addAttributes(node1.Attribute, rgTensors, node1.Name, reshape_param, true);

                            rg.Add(node1);
                            node.Input[0] = strOutput;
                        }

                        m_bReshapeBeforeInnerProductNeeded = false;
                        node.OpType = OnnxDefinitions.OPERATORS.Gemm.ToString();
                        addAttributes(node.Attribute, layer.layer_param.inner_product_param);
                        colParams.Add(layer.blobs[0]);
                        if (layer.layer_param.inner_product_param.bias_term)
                            colParams.Add(layer.blobs[1]);
                        break;

                    case LayerParameter.LayerType.LRN:
                        node.OpType = OnnxDefinitions.OPERATORS.LRN.ToString();
                        addAttributes(node.Attribute, layer.layer_param.lrn_param);
                        break;

                    case LayerParameter.LayerType.LOG:
                        node.OpType = OnnxDefinitions.OPERATORS.Log.ToString();
                        addAttributes(node.Attribute, layer.layer_param.log_param);
                        break;

                    case LayerParameter.LayerType.MATH:
                        if (layer.layer_param.math_param.function == MATH_FUNCTION.ACOS)
                            node.OpType = OnnxDefinitions.OPERATORS.Acos.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.ACOSH)
                            node.OpType = OnnxDefinitions.OPERATORS.Acosh.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.COS)
                            node.OpType = OnnxDefinitions.OPERATORS.Cos.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.COSH)
                            node.OpType = OnnxDefinitions.OPERATORS.Cosh.ToString();

                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.ASIN)
                            node.OpType = OnnxDefinitions.OPERATORS.Asin.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.ASINH)
                            node.OpType = OnnxDefinitions.OPERATORS.Asinh.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.SIN)
                            node.OpType = OnnxDefinitions.OPERATORS.Sin.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.SINH)
                            node.OpType = OnnxDefinitions.OPERATORS.Sinh.ToString();

                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.ATAN)
                            node.OpType = OnnxDefinitions.OPERATORS.Atan.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.ATANH)
                            node.OpType = OnnxDefinitions.OPERATORS.Atanh.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.TAN)
                            node.OpType = OnnxDefinitions.OPERATORS.Tan.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.TANH)
                            node.OpType = OnnxDefinitions.OPERATORS.Tanh.ToString();

                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.CEIL)
                            node.OpType = OnnxDefinitions.OPERATORS.Ceil.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.FLOOR)
                            node.OpType = OnnxDefinitions.OPERATORS.Floor.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.NEG)
                            node.OpType = OnnxDefinitions.OPERATORS.Neg.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.SIGN)
                            node.OpType = OnnxDefinitions.OPERATORS.Sign.ToString();
                        else if (layer.layer_param.math_param.function == MATH_FUNCTION.SQRT)
                            node.OpType = OnnxDefinitions.OPERATORS.Sqrt.ToString();
                        break;

                    case LayerParameter.LayerType.POOLING:
                        if (layer.layer_param.pooling_param.global_pooling)
                        {
                            if (layer.layer_param.pooling_param.pool == PoolingParameter.PoolingMethod.AVE)
                                node.OpType = OnnxDefinitions.OPERATORS.GlobalAveragePool.ToString();
                            else if (layer.layer_param.pooling_param.pool == PoolingParameter.PoolingMethod.MAX)
                                node.OpType = OnnxDefinitions.OPERATORS.GlobalMaxPool.ToString();
                            else if (layer.layer_param.pooling_param.pool == PoolingParameter.PoolingMethod.STOCHASTIC)
                                throw new Exception("Currently global STOCHASTIC pooling is not supported for ONNX conversion.");
                        }
                        else
                        {
                            if (layer.layer_param.pooling_param.pool == PoolingParameter.PoolingMethod.AVE)
                                node.OpType = OnnxDefinitions.OPERATORS.AveragePool.ToString();
                            else if (layer.layer_param.pooling_param.pool == PoolingParameter.PoolingMethod.MAX)
                                node.OpType = OnnxDefinitions.OPERATORS.MaxPool.ToString();
                            else if (layer.layer_param.pooling_param.pool == PoolingParameter.PoolingMethod.STOCHASTIC)
                                throw new Exception("Currently STOCHASTIC pooling is not supported for ONNX conversion.");
                        }
                        addAttributes(node.Attribute, layer.layer_param.pooling_param);
                        break;

                    case LayerParameter.LayerType.PRELU:
                        node.OpType = OnnxDefinitions.OPERATORS.PRelu.ToString();
                        colParams = layer.blobs;
                        break;

                    case LayerParameter.LayerType.POWER:
                        node.OpType = OnnxDefinitions.OPERATORS.Pow.ToString();
                        break;

                    case LayerParameter.LayerType.REDUCTION:
                        if (layer.layer_param.reduction_param.operation == ReductionParameter.ReductionOp.MAX)
                            node.OpType = OnnxDefinitions.OPERATORS.ReduceMax.ToString();
                        else if (layer.layer_param.reduction_param.operation == ReductionParameter.ReductionOp.MEAN)
                            node.OpType = OnnxDefinitions.OPERATORS.ReduceMean.ToString();
                        else if (layer.layer_param.reduction_param.operation == ReductionParameter.ReductionOp.MIN)
                            node.OpType = OnnxDefinitions.OPERATORS.ReduceMin.ToString();
                        else if (layer.layer_param.reduction_param.operation == ReductionParameter.ReductionOp.SUM)
                            node.OpType = OnnxDefinitions.OPERATORS.ReduceSum.ToString();
                        else if (layer.layer_param.reduction_param.operation == ReductionParameter.ReductionOp.SUMSQ)
                            node.OpType = OnnxDefinitions.OPERATORS.ReduceSumSquare.ToString();
                        addAttributes(node.Attribute, layer.layer_param.reduction_param);
                        break;

                    case LayerParameter.LayerType.RELU:
                        if (layer.layer_param.relu_param.negative_slope != 0)
                            node.OpType = OnnxDefinitions.OPERATORS.LeakyRelu.ToString();
                        else
                            node.OpType = OnnxDefinitions.OPERATORS.Relu.ToString();
                        addAttributes(node.Attribute, layer.layer_param.relu_param);
                        break;

                    case LayerParameter.LayerType.RESHAPE:
                        node.OpType = OnnxDefinitions.OPERATORS.Reshape.ToString();
                        string strName = "reshape" + m_nReshapeCount.ToString();
                        node.Input.Add(strName);
                        m_nReshapeCount++;
                        addAttributes(node.Attribute, rgTensors, strName, layer.layer_param.reshape_param, true);
                        break;

                    case LayerParameter.LayerType.SOFTMAX:
                        node.OpType = OnnxDefinitions.OPERATORS.Softmax.ToString();
                        addAttributes(node.Attribute, layer.layer_param.softmax_param);
                        break;

                    case LayerParameter.LayerType.SPLIT:
                        node.OpType = OnnxDefinitions.OPERATORS.Split.ToString();
                        break;

                    case LayerParameter.LayerType.SQUEEZE:
                        node.OpType = OnnxDefinitions.OPERATORS.Squeeze.ToString();
                        break;

                    case LayerParameter.LayerType.UNSQUEEZE:
                        node.OpType = OnnxDefinitions.OPERATORS.Unsqueeze.ToString();
                        break;

                    case LayerParameter.LayerType.TRANSPOSE:
                        node.OpType = OnnxDefinitions.OPERATORS.Transpose.ToString();
                        addAttributes(node.Attribute, layer.layer_param.transpose_param);
                        break;

                    default:
                        bool bTraceEnabled = log.EnableTrace;
                        log.EnableTrace = true;
                        log.WriteLine("Ignoring layer '" + layer.layer_param.name + "'(" + layer.type.ToString() + ")");
                        log.EnableTrace = bTraceEnabled;
                        node = null;
                        break;
                }

                if (node != null)
                {
                    foreach (Blob<T> blob in colParams)
                    {
                        node.Input.Add(removeWs(blob.Name, '_'));
                    }

                    rg.Add(node);
                }

                m_lastType = layer.type;
            }
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, BatchNormParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "epsilon";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.eps;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "momentum";
            attrib.Type = AttributeProto.Types.AttributeType.Ints;
            attrib.F = (float)p.moving_average_fraction;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ClipParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "min";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.min;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "max";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.max;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ConcatParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "axis";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = p.axis;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ConstantParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "value";
            attrib.Type = AttributeProto.Types.AttributeType.Tensor;
            attrib.T = new TensorProto();
            attrib.T.DataType = (int)OnnxDefinitions.DataType.FLOAT;

            for (int i = 0; i < p.values_f.Count; i++)
            {
                attrib.T.FloatData.Add(p.values_f[i]);
            }

            foreach (int nDim in p.output_shape.dim)
            {
                attrib.T.Dims.Add(nDim);
            }

            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ConvolutionParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "kernel_shape";
            attrib.Type = AttributeProto.Types.AttributeType.Ints;
            uint h = (p.kernel_h.HasValue) ? p.kernel_h.Value : (p.kernel_size.Count > 0) ? p.kernel_size[0] : 3;
            attrib.Ints.Add(h);
            uint w = (p.kernel_w.HasValue) ? p.kernel_w.Value : (p.kernel_size.Count > 0) ? p.kernel_size[0] : 3;
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "strides";
            attrib.Type = AttributeProto.Types.AttributeType.Ints;
            h = (p.stride_h.HasValue) ? p.stride_h.Value : (p.stride.Count > 0) ? p.stride[0] : 1;
            attrib.Ints.Add(h);
            w = (p.stride_w.HasValue) ? p.stride_w.Value : (p.stride.Count > 0) ? p.stride[0] : 1;
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            if ((p.pad_h.HasValue && p.pad_w.HasValue && p.pad_h != 0 && p.pad_w != 0) || (p.pad.Count > 0 && p.pad[0] != 0))
            {
                attrib = new AttributeProto();
                attrib.Name = "pads";
                attrib.Type = AttributeProto.Types.AttributeType.Ints;
                h = (p.pad_h.HasValue) ? p.pad_h.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
                attrib.Ints.Add(h);
                w = (p.pad_w.HasValue) ? p.pad_w.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
                attrib.Ints.Add(w);
                rgA.Add(attrib);
            }

            if (p.dilation.Count > 0)
            {
                attrib = new AttributeProto();
                attrib.Name = "dilations";
                attrib.Type = AttributeProto.Types.AttributeType.Ints;
                h = (p.dilation.Count > 0) ? p.dilation[0] : 1;
                attrib.Ints.Add(h);
                w = (p.dilation.Count > 0) ? (p.dilation.Count > 1) ? p.dilation[1] : p.dilation[0] : 1;
                attrib.Ints.Add(w);
                rgA.Add(attrib);
            }

            attrib = new AttributeProto();
            attrib.Name = "group";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = p.group;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, DropoutParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "ratio";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.dropout_ratio;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "seed";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = p.seed;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "training_mode";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = (p.active) ? 1 : 0;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, FlattenParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "axis";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = p.axis;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, GatherParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "axis";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = p.axis;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, InnerProductParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "alpha";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = 1.0f;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "beta";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = 1.0f; // ONNX requires this to be non zero.
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "transA";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = 0;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "transB";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = (p.transpose) ? 0 : 1; // see line 381 InnerProductLayer.cs (value is opposite of p.transpose)
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, LRNParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "alpha";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.alpha;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "beta";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.beta;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "bias";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.k;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "size";
            attrib.Type = AttributeProto.Types.AttributeType.Float;
            attrib.F = (float)p.local_size;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, LogParameter p)
        {
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, PoolingParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "kernel_shape";
            attrib.Type = AttributeProto.Types.AttributeType.Ints;
            uint h = (p.kernel_h.HasValue) ? p.kernel_h.Value : (p.kernel_size.Count > 0) ? p.kernel_size[0] : 3;
            attrib.Ints.Add(h);
            uint w = (p.kernel_w.HasValue) ? p.kernel_w.Value : (p.kernel_size.Count > 0) ? p.kernel_size[0] : 3;
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "strides";
            attrib.Type = AttributeProto.Types.AttributeType.Ints;
            h = (p.stride_h.HasValue) ? p.stride_h.Value : (p.stride.Count > 0) ? p.stride[0] : 1;
            attrib.Ints.Add(h);
            w = (p.stride_w.HasValue) ? p.stride_w.Value : (p.stride.Count > 0) ? p.stride[0] : 1;
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            if ((p.pad_h.HasValue && p.pad_w.HasValue && p.pad_h != 0 && p.pad_w != 0) || (p.pad.Count > 0 && p.pad[0] != 0))
            {
                attrib = new AttributeProto();
                attrib.Name = "pads";
                attrib.Type = AttributeProto.Types.AttributeType.Ints;
                h = (p.pad_h.HasValue) ? p.pad_h.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
                attrib.Ints.Add(h);
                w = (p.pad_w.HasValue) ? p.pad_w.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
                attrib.Ints.Add(w);
                rgA.Add(attrib);
            }
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, EluParameter p)
        {
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ExpParameter p)
        {
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ReductionParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "axes";
            attrib.Type = AttributeProto.Types.AttributeType.Ints;
            attrib.Ints.Add(p.axis);

            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ReLUParameter p)
        {
            if (p.negative_slope != 0)
            {
                AttributeProto attrib = new AttributeProto();
                attrib.Name = "alpha";
                attrib.Type = AttributeProto.Types.AttributeType.Float;
                attrib.F = (float)p.negative_slope;
                rgA.Add(attrib);
            }
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, RepeatedField<TensorProto> rgTensors, string strName, ReshapeParameter p, bool bRemoveTrailingOnes)
        {
            List<int> rgShape = new List<int>();

            for (int i = 0; i < p.axis; i++)
            {
                rgShape.Add(-1);
            }

            for (int i = 0; i < p.shape.dim.Count; i++)
            {
                rgShape.Add(p.shape.dim[i]);
            }

            if (bRemoveTrailingOnes)
            {
                while (rgShape.Count > 2 && rgShape[rgShape.Count - 1] == 1)
                {
                    rgShape.RemoveAt(rgShape.Count - 1);
                }
            }

            if (rgShape[0] == -1)
                rgShape[0] = 0;

            addShapeTensor(rgTensors, strName, rgShape);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, SoftmaxParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "axis";
            attrib.Type = AttributeProto.Types.AttributeType.Int;
            attrib.I = p.axis;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, TransposeParameter p)
        {
            foreach (int nDim in p.dim)
            {
                AttributeProto attrib = new AttributeProto();
                attrib.Name = "dim";
                attrib.Type = AttributeProto.Types.AttributeType.Int;
                attrib.I = nDim;
                rgA.Add(attrib);
            }
        }

        private string clean(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (!char.IsWhiteSpace(ch))
                {
                    strOut += ch;
                }
            }

            return strOut;
        }

        private Tuple<NetParameter, BlobCollection<T>> convertToMyCaffe(CudaDnn<T> cuda, Log log, ModelProto proto, bool bFixupNeuronNodes, DatasetDescriptor dsTraining = null)
        {
            try
            {
                NetParameter netParam = new NetParameter();
                BlobCollection<T> colTensors = new BlobCollection<T>();
                BlobCollection<T> colLearnableBlobs = new BlobCollection<T>();
                OnnxDefinitions onnx = new OnnxDefinitions();

                m_strReport = "";

                netParam.name = clean(proto.Graph.Name);
                Tuple<List<string>, List<string>> rgInputs = addInputs(proto.Graph.Input, netParam, false);
                addTensors(proto.Graph.Initializer, colTensors, cuda, log);
                colLearnableBlobs = addLayers(proto.Graph.Node, netParam, colTensors, onnx, rgInputs.Item1, cuda, log, false);
                addInputs(proto.Graph.Input, netParam, true, rgInputs.Item1, rgInputs.Item2);

                NetParameter netParamFixed = fixupModel(netParam, colLearnableBlobs, rgInputs.Item2);

                if (bFixupNeuronNodes)
                    netParamFixed = fixupModelNeuronNodes(netParamFixed);

                if (dsTraining != null)
                    netParamFixed = fixupModelForTraining(netParamFixed, dsTraining);

                netParamFixed = removeLayersWithOrphanedBottoms(netParamFixed);

                return new Tuple<NetParameter, BlobCollection<T>>(netParamFixed, colLearnableBlobs);
            }
            catch (Exception excpt)
            {
                m_strReport += "ERROR: " + excpt.Message + Environment.NewLine;
                throw excpt;
            }
        }

        private NetParameter removeLayersWithOrphanedBottoms(NetParameter net)
        {
            List<int> rgRemoveIdx = new List<int>();
            Dictionary<string, Tuple<int, int>> rgTopToLayerIdx = new Dictionary<string, Tuple<int, int>>();

            // Find all tops and their associated layer index and top index.
            for (int i = 0; i < net.layer.Count; i++)
            {
                for (int j=0; j<net.layer[i].top.Count; j++)
                {
                    if (!rgTopToLayerIdx.ContainsKey(net.layer[i].top[j]))
                        rgTopToLayerIdx.Add(net.layer[i].top[j], new Tuple<int, int>(i, j));
                }
            }

            // Replace the parent top with the bottom of the layer to be removed.
            for (int i=1; i<net.layer.Count; i++)
            {
                Dictionary<string, Tuple<int, int>> rgBtmToParentTop = new Dictionary<string, Tuple<int, int>>();
                bool bMissingBtmFound = false;

                foreach (string strBtm in net.layer[i].bottom)
                {
                    if (!rgTopToLayerIdx.ContainsKey(strBtm))
                    {
                        rgRemoveIdx.Add(i);
                        bMissingBtmFound = true;
                    }
                    else
                    {
                        rgBtmToParentTop.Add(strBtm, rgTopToLayerIdx[strBtm]);
                    }
                }

                if (!bMissingBtmFound && net.layer[i].bottom.Count >= net.layer[i].expected_bottom.Count)
                    rgBtmToParentTop.Clear();
                else if (!rgRemoveIdx.Contains(i))
                    rgRemoveIdx.Add(i);

                foreach (KeyValuePair<string, Tuple<int, int>> kvTopInParent in rgBtmToParentTop)
                {
                    if (net.layer[i].top.Count > 0)
                        net.layer[kvTopInParent.Value.Item1].top[kvTopInParent.Value.Item2] = net.layer[i].top[0];
                }
            }

            // Remove the layer.
            for (int i = rgRemoveIdx.Count - 1; i >= 0; i--)
            {
                net.layer.RemoveAt(rgRemoveIdx[i]);
            }

            return net;
        }

        private NetParameter fixupModelForTraining(NetParameter netParam, DatasetDescriptor ds)
        {
            string strName = (netParam.input.Count > 0) ? netParam.input[0] : "data";

            // Replace the inputs with the data layers.
            LayerParameter dataLayerTrain = new LayerParameter(LayerParameter.LayerType.DATA, strName);
            dataLayerTrain.include.Add(new NetStateRule(Phase.TRAIN));
            dataLayerTrain.transform_param.color_order = TransformationParameter.COLOR_ORDER.BGR;
            dataLayerTrain.transform_param.scale = 1.0;
            dataLayerTrain.data_param.batch_size = 16;
            dataLayerTrain.data_param.source = ds.TrainingSource.Name;
            dataLayerTrain.top.Add(strName);
            dataLayerTrain.top.Add("label");

            LayerParameter dataLayerTest = new LayerParameter(LayerParameter.LayerType.DATA, strName);
            dataLayerTest.include.Add(new NetStateRule(Phase.TEST));
            dataLayerTest.transform_param.color_order = TransformationParameter.COLOR_ORDER.BGR;
            dataLayerTest.transform_param.scale = 1.0;
            dataLayerTest.data_param.batch_size = 16;
            dataLayerTest.data_param.source = ds.TestingSource.Name;
            dataLayerTest.top.Add(strName);
            dataLayerTest.top.Add("label");

            if (netParam.input.Count == 0)
            {
                if (netParam.layer[0].bottom.Count > 0)
                    netParam.layer[0].bottom[0] = strName;
                else
                    netParam.layer[0].bottom.Add(strName);
            }

            m_strReport += "Removed inputs " + Utility.ToString<string>(netParam.input) + Environment.NewLine;
            netParam.input.Clear();
            netParam.input_shape.Clear();

            m_strReport += "Added DATA layer '" + dataLayerTest.name + "' with phase TEST..." + Environment.NewLine;
            netParam.layer.Insert(0, dataLayerTest);
            m_strReport += "Added DATA layer '" + dataLayerTest.name + "' with phase TRAIN..." + Environment.NewLine;
            netParam.layer.Insert(0, dataLayerTrain);

            LayerParameter lastLayer = netParam.layer[netParam.layer.Count - 1];
            if (lastLayer.type == LayerParameter.LayerType.SOFTMAX || lastLayer.type == LayerParameter.LayerType.INNERPRODUCT)
            {
                List<string> rgstrLossBottom;
                List<string> rgstrAccuracyBottom;
                int nAxis = 1;
                if (lastLayer.type == LayerParameter.LayerType.SOFTMAX)
                {
                    m_strReport += "Removing last layer SOFTMAX..." + Environment.NewLine;
                    netParam.layer.Remove(lastLayer);
                    rgstrLossBottom = Utility.Clone<string>(lastLayer.bottom);
                    rgstrAccuracyBottom = Utility.Clone<string>(lastLayer.bottom);
                }
                else
                {
                    rgstrLossBottom = Utility.Clone<string>(lastLayer.top);
                    rgstrAccuracyBottom = Utility.Clone<string>(lastLayer.top);
                }

                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS, "loss");
                loss.top.Add("loss");
                loss.bottom = rgstrLossBottom;
                loss.bottom.Add(dataLayerTrain.top[1]);

                if (lastLayer.softmax_param != null)
                {
                    nAxis = lastLayer.softmax_param.axis;
                    loss.softmax_param = lastLayer.softmax_param;
                }

                m_strReport += "Added new last layer SOFTMAXWITH_LOSS '" + loss.name + "'..." + Environment.NewLine;
                netParam.layer.Add(loss);

                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY, "accuracy");
                accuracy.top.Add("accuracy");
                accuracy.bottom = rgstrAccuracyBottom;
                accuracy.bottom.Add(dataLayerTest.top[1]);
                accuracy.accuracy_param.axis = nAxis;
                accuracy.include.Add(new NetStateRule(Phase.TEST));
                m_strReport += "Added new last layer ACCURACY '" + accuracy.name + "'..." + Environment.NewLine;
                netParam.layer.Add(accuracy);
            }

            return netParam;
        }

        private bool isNeuron(LayerParameter.LayerType type)
        {
            if (type == LayerParameter.LayerType.RELU ||
                type == LayerParameter.LayerType.TANH ||
                type == LayerParameter.LayerType.MATH ||
                type == LayerParameter.LayerType.SIGMOID ||
                type == LayerParameter.LayerType.ELU ||
                type == LayerParameter.LayerType.PRELU ||
                type == LayerParameter.LayerType.DROPOUT)
                return true;

            return false;
        }

        private bool replaceTop(LayerParameter p, string strTopToReplace, string strNewTop)
        {
            for (int i=0; i<p.top.Count; i++)
            {
                if (p.top[i] == strTopToReplace)
                {
                    p.top[i] = strNewTop;
                    return true;
                }
            }

            return false;
        }

        private bool replaceBtm(LayerParameter p, string strBtmToReplace, string strNewBtm)
        {
            for (int i = 0; i < p.bottom.Count; i++)
            {
                if (p.bottom[i] == strBtmToReplace)
                {
                    p.bottom[i] = strNewBtm;
                    return true;
                }
            }

            return false;
        }

        private NetParameter fixupModelNeuronNodes(NetParameter netParam)
        {
            foreach (LayerParameter layer in netParam.layer)
            {
                if (isNeuron(layer.type))
                {
                    string strBtm = layer.bottom[0];

                    if (layer.top.Count == 0)
                    {
                        layer.top.Add(strBtm);
                    }
                    else
                    {
                        string strTop = layer.top[0];

                        LayerParameter layerPrev = findLayerWithTop(netParam, strBtm);
                        List<LayerParameter> rgLayerNext = findLayersWithBtm(netParam, strTop);

                        // MyCaffe neural nodes pass data right back to the prev node.
                        layer.top[0] = layer.bottom[0];

                        // Connect the prev node top with the next node btm.
                        foreach (LayerParameter layerNext in rgLayerNext)
                        {
                            replaceBtm(layerNext, strTop, strBtm);
                        }
                    }
                }
            }

            return netParam;
        }

        private NetParameter fixupModel(NetParameter netParam, BlobCollection<T> col, List<string> rgstrInvalidInput)
        {
            NetParameter p = netParam.Clone();

            // Find the data input.
            int nDataInputIdx = 0;
            for (int i = 0; i < p.input.Count; i++)
            {
                if (p.input[i].Contains("data"))
                {
                    nDataInputIdx = i;
                    break;
                }
            }

            // Change input name to 'data'
            List<LayerParameter> rgLayer1 = findLayersWithBtm(p, p.input[nDataInputIdx]);
            foreach (LayerParameter layer1 in rgLayer1)
            {
                replaceBtm(layer1, p.input[nDataInputIdx], "data");
                m_strReport += "Changed layer '" + layer1.name + " (" + layer1.type.ToString() + ")' input from '" + p.input[nDataInputIdx] + "' to 'data'";
            }

            m_strReport += "Changed data input[" + nDataInputIdx.ToString() + "] from '" + p.input[nDataInputIdx] + "' to 'data'";
            p.input[nDataInputIdx] = "data";


            // Remove all input orphans.
            List<string> rgInputs = Utility.Clone<string>(p.input);
            foreach (LayerParameter layer1 in p.layer)
            {
                foreach (string strBtm in layer1.bottom)
                {
                    rgInputs.Remove(strBtm);
                }

                if (rgInputs.Count == 0)
                    break;
            }

            foreach (string strOrphanInput in rgInputs)
            {
                p.input.Remove(strOrphanInput);
            }

            // Find all orphan bottoms
            LayerDataCollection rgBtmOrphans = new LayerDataCollection(LayerData.TYPE.BTM);
            for (int i = p.layer.Count-1; i>=0; i--)
            {
                LayerParameter layer = p.layer[i];

                rgBtmOrphans.Add(layer.bottom, i, layer);
                rgBtmOrphans.Remove(layer.top);
            }

            rgBtmOrphans.Remove(p.input);

            // Find all orphan tops
            LayerDataCollection rgTopOrphans = new LayerDataCollection(LayerData.TYPE.TOP);
            for (int i=0; i<p.layer.Count; i++)
            {
                LayerParameter layer = p.layer[i];

                if (i < p.layer.Count-1)
                    rgTopOrphans.Add(layer.top, i, layer);

                rgTopOrphans.Remove(layer.bottom);
            }

            // Fixup - remove all top orphans.
            if (rgTopOrphans.Count > 0)
            {
                m_strReport += "[Found '" + rgTopOrphans.Count.ToString() + " Top Orphans]" + Environment.NewLine;
                foreach (LayerData top in rgTopOrphans)
                {
                    m_strReport += "  " + top.Name + " found in layer " + top.Layer.ToString() + Environment.NewLine;
                }

                for (int i = 0; i < p.layer.Count; i++)
                {
                    LayerParameter layer = p.layer[i];
                    m_strReport += rgTopOrphans.FixupTops(i, layer);
                }
            }

            // Fixup - fixup bottom orphans.
            if (rgBtmOrphans.Count > 0)
            {
                m_strReport += "[Found '" + rgBtmOrphans.Count.ToString() + " Bottom Orphans]" + Environment.NewLine;
                foreach (LayerData btm in rgBtmOrphans)
                {
                    m_strReport += "  " + btm.Name + " found in layer " + btm.Layer.ToString() + Environment.NewLine;
                }

                foreach (LayerData item in rgBtmOrphans)
                {
                    // Remove reshape layers with external input sizings.
                    if (item.Layer.type == LayerParameter.LayerType.RESHAPE)
                    {
                        if (item.Layer.bottom.Count > 1 && rgstrInvalidInput.Contains(item.Layer.bottom[1]))
                        {
                            foreach (string strBtm1 in item.Layer.bottom)
                            {
                                p.input.Remove(strBtm1);
                            }

                            string strBtm = item.Layer.bottom[0];
                            string strTop = (item.Layer.top.Count > 0) ? item.Layer.top[0] : null;

                            LayerParameter layerPrev = findLayerWithTop(p, strBtm);
                            List<LayerParameter> rgLayerNext = findLayersWithBtm(p, strTop);

                            // Remove the reshape layer.
                            m_strReport += "Removed RESHAPE layer..." + Environment.NewLine;
                            p.layer.Remove(item.Layer);

                            if (layerPrev != null && rgLayerNext.Count > 0)
                            {
                                // Remove any blobs used by the layer.
                                Blob<T> blob = col.FindBlob(item.Layer.bottom[1]);
                                if (blob != null)
                                {
                                    col.Remove(blob);
                                    m_strReport += "Removed blob '" + blob.Name + "' with shape '" + blob.shape_string + "'..." + Environment.NewLine;
                                    blob.Dispose();
                                }

                                string strTop1 = layerPrev.top[0];

                                // Replace the bottom of next layer (that the reshape outputs to)
                                // with the top of the layer previous to the reshape layer.
                                foreach (LayerParameter layerNext in rgLayerNext)
                                {
                                    for (int i = 0; i < layerNext.bottom.Count; i++)
                                    {
                                        if (layerNext.bottom[i] == strTop)
                                        {
                                            layerNext.bottom[i] = strTop1;
                                            m_strReport += "connected joining layers with '" + strTop1 + "'...";

                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return p;
        }

        private LayerParameter findLayerWithTop(NetParameter net, string strTop)
        {
            foreach (LayerParameter layer in net.layer)
            {
                if (layer.top.Contains(strTop))
                    return layer;
            }

            return null;
        }

        private List<LayerParameter> findLayersWithBtm(NetParameter net, string strBtm)
        {
            List<LayerParameter> rgLayers = new List<LayerParameter>();

            foreach (LayerParameter layer in net.layer)
            {
                if (layer.bottom.Contains(strBtm))
                    rgLayers.Add(layer);
            }

            return rgLayers;
        }

        private Tuple<List<string>, List<string>> addInputs(RepeatedField<ValueInfoProto> rg, NetParameter p, bool bAdd, List<string> rgstrAvailable = null, List<string> rgstrInvalid = null)
        {
            List<string> rgstrInput = new List<string>();
            List<string> rgstrInvalid1 = new List<string>();

            foreach (ValueInfoProto val in rg)
            {
                if ((rgstrAvailable == null || rgstrAvailable.Contains(val.Name)) && (rgstrInvalid == null || !rgstrInvalid.Contains(val.Name)))
                {
                    if (val.Type.TensorType == null)
                        throw new Exception("Currenly only Tensor input types are supported.");

                    if (val.Type.TensorType.ElemType != (int)OnnxDefinitions.DataType.FLOAT &&
                        val.Type.TensorType.ElemType != (int)OnnxDefinitions.DataType.DOUBLE)
                        rgstrInvalid1.Add(val.Name);
                    else
                        rgstrInput.Add(convertWs(val.Name));

                    if (bAdd)
                    {
                        p.input.Add(convertWs(val.Name));
                        List<int> rgShape = new List<int>();

                        TypeProto type = val.Type;
                        if (type.TensorType == null)
                            throw new Exception("Currenly only Tensor input types are supported.");

                        TensorShapeProto shape = type.TensorType.Shape;
                        foreach (TensorShapeProto.Types.Dimension dim in shape.Dim)
                        {
                            rgShape.Add((int)dim.DimValue);
                        }

                        p.input_shape.Add(new BlobShape(rgShape));

                        m_strReport += "Adding input '" + val.Name + " with shape " + rgShape.ToString() + Environment.NewLine;
                    }
                }
            }

            return new Tuple<List<string>, List<string>>(rgstrInput, rgstrInvalid1);
        }

        private void addTensors(RepeatedField<TensorProto> rg, BlobCollection<T> col, CudaDnn<T> cuda, Log log)
        {
            foreach (TensorProto tensor in rg)
            {
                List<int> rgShape = new List<int>();

                foreach (long lDim in tensor.Dims)
                {
                    rgShape.Add((int)lDim);
                }

                Blob<T> blob = new Blob<T>(cuda, log, rgShape);
                blob.Name = convertWs(tensor.Name);

                if (typeof(T) == typeof(float))
                {
                    float[] rgData = getDataAsFloat(tensor);
                    blob.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                }
                else
                {
                    double[] rgData = getDataAsDouble(tensor);
                    blob.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                }

                m_strReport += "Adding tensor '" + blob.Name + "' " + blob.shape_string + Environment.NewLine;

                col.Add(blob);
            }
        }

        /// <summary>
        /// Converts the tensor data into an array of <i>float</i>.
        /// </summary>
        /// <param name="tensor">Specifies the tensor to convert.</param>
        /// <returns>An array of <i>float</i> values is returned.</returns>
        public static float[] getDataAsFloat(TensorProto tensor)
        {
            float[] rgData = null;

            if (tensor.DataType == (int)OnnxDefinitions.DataType.FLOAT)
            {
                if (tensor.FloatData.Count > 0)
                {
                    rgData = new float[tensor.FloatData.Count];
                    for (int i = 0; i < tensor.FloatData.Count; i++)
                    {
                        rgData[i] = tensor.FloatData[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(float);
                    rgData = new float[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData, 0, rgRaw.Length);
                }
            }
            else if (tensor.DataType == (int)OnnxDefinitions.DataType.DOUBLE)
            {
                if (tensor.DoubleData.Count > 0)
                {
                    rgData = new float[tensor.DoubleData.Count];
                    for (int i = 0; i < tensor.DoubleData.Count; i++)
                    {
                        rgData[i] = (float)tensor.DoubleData[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(double);
                    double[] rgData2 = new double[nLen];
                    rgData = new float[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData2, 0, rgRaw.Length);
                    Array.Copy(rgData2, rgData, nLen);
                }
            }
            else if (tensor.DataType == (int)OnnxDefinitions.DataType.INT32)
            {
                if (tensor.Int32Data.Count > 0)
                {
                    rgData = new float[tensor.Int32Data.Count];
                    for (int i = 0; i < tensor.Int32Data.Count; i++)
                    {
                        rgData[i] = (float)tensor.Int32Data[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(int);
                    int[] rgData2 = new int[nLen];
                    rgData = new float[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData2, 0, rgRaw.Length);
                    Array.Copy(rgData2, rgData, nLen);
                }
            }
            else if (tensor.DataType == (int)OnnxDefinitions.DataType.INT64)
            {
                if (tensor.Int64Data.Count > 0)
                {
                    rgData = new float[tensor.Int64Data.Count];
                    for (int i = 0; i < tensor.Int64Data.Count; i++)
                    {
                        rgData[i] = (float)tensor.Int64Data[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(long);
                    long[] rgData2 = new long[nLen];
                    rgData = new float[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData2, 0, rgRaw.Length);
                    Array.Copy(rgData2, rgData, nLen);
                }
            }
            else
            {
                throw new Exception("Currently only the 'DataType.FLOAT' and 'DataType.DOUBLE' are supported for conversions to MyCaffe.");
            }

            return rgData;
        }

        /// <summary>
        /// Converts the tensor data into an array of <i>double</i>.
        /// </summary>
        /// <param name="tensor">Specifies the tensor to convert.</param>
        /// <returns>An array of <i>double</i> values is returned.</returns>
        public static double[] getDataAsDouble(TensorProto tensor)
        {
            double[] rgData = null;

            if (tensor.DataType == (int)OnnxDefinitions.DataType.FLOAT)
            {
                if (tensor.FloatData.Count > 0)
                {
                    rgData = new double[tensor.FloatData.Count];
                    for (int i = 0; i < tensor.FloatData.Count; i++)
                    {
                        rgData[i] = tensor.FloatData[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(float);
                    float[] rgData2 = new float[nLen];
                    rgData = new double[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData2, 0, rgRaw.Length);
                    Array.Copy(rgData2, rgData, nLen);
                }
            }
            else if (tensor.DataType == (int)OnnxDefinitions.DataType.DOUBLE)
            {
                if (tensor.DoubleData.Count > 0)
                {
                    rgData = new double[tensor.DoubleData.Count];
                    for (int i = 0; i < tensor.DoubleData.Count; i++)
                    {
                        rgData[i] = (float)tensor.DoubleData[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(double);
                    double[] rgData2 = new double[nLen];
                    rgData = new double[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData, 0, rgRaw.Length);
                }
            }
            else if (tensor.DataType == (int)OnnxDefinitions.DataType.INT32)
            {
                if (tensor.Int32Data.Count > 0)
                {
                    rgData = new double[tensor.Int32Data.Count];
                    for (int i = 0; i < tensor.Int32Data.Count; i++)
                    {
                        rgData[i] = (float)tensor.Int32Data[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(int);
                    int[] rgData2 = new int[nLen];
                    rgData = new double[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData2, 0, rgRaw.Length);
                    Array.Copy(rgData2, rgData, nLen);
                }
            }
            else if (tensor.DataType == (int)OnnxDefinitions.DataType.INT64)
            {
                if (tensor.Int64Data.Count > 0)
                {
                    rgData = new double[tensor.Int64Data.Count];
                    for (int i = 0; i < tensor.Int64Data.Count; i++)
                    {
                        rgData[i] = (float)tensor.Int64Data[i];
                    }
                }
                else
                {
                    byte[] rgRaw = tensor.RawData.ToByteArray();
                    int nLen = rgRaw.Length / sizeof(long);
                    long[] rgData2 = new long[nLen];
                    rgData = new double[nLen];

                    Buffer.BlockCopy(rgRaw, 0, rgData2, 0, rgRaw.Length);
                    Array.Copy(rgData2, rgData, nLen);
                }
            }
            else
            {
                throw new Exception("Currently only the 'DataType.FLOAT' and 'DataType.DOUBLE' are supported for conversions to MyCaffe.");
            }

            return rgData;
        }

        private int getOutputs(string strLayerName, BlobCollection<T> col, string strWt, string strBias, out bool bBiasTerm, int nAxis = 0)
        {
            int? nWtOutputs = null;
            int? nBiasOutputs = null;

            bBiasTerm = false;

            foreach (Blob<T> blob in col)
            {
                if (blob.Name == strWt)
                {
                    blob.Tag = strLayerName;
                    nWtOutputs = blob.shape()[nAxis];
                }
                else if (blob.Name == strBias && strBias != null)
                {
                    blob.Tag = strLayerName;
                    bBiasTerm = true;
                    nBiasOutputs = blob.shape()[0];
                }

                if (nWtOutputs.HasValue && bBiasTerm)
                    break;
            }

            if (!nWtOutputs.HasValue && !nBiasOutputs.HasValue)
                throw new Exception("Could not find the blob '" + strWt + "' or the blob '" + strBias + "'!");

            if (nWtOutputs.HasValue)
                return nWtOutputs.Value;

            return nBiasOutputs.Value;
        }

        private string getOperator(OnnxDefinitions onnx, OnnxDefinitions.OPERATORS op)
        {
            string str = onnx.GetString(op);
            if (str == null)
                str = op.ToString();

            return str;
        }

        private string convertWs(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (char.IsWhiteSpace(ch))
                    strOut += "_";
                else
                    strOut += ch;
            }

            return strOut;
        }

        private void scale(Blob<T> blob)
        {
            if (!m_dfWtScaleMin.HasValue || double.IsNaN(m_dfWtScaleMin.Value) || !m_dfWtScaleMax.HasValue || double.IsNaN(m_dfWtScaleMax.Value))
                return;

            blob.scale_to_range(m_dfWtScaleMin.Value, m_dfWtScaleMax.Value);
        }

        private BlobCollection<T> addLayers(RepeatedField<NodeProto> rg, NetParameter p, BlobCollection<T> col, OnnxDefinitions onnx, List<string> rgstrInputs, CudaDnn<T> cuda, Log log, bool bIncludeConstants)
        {
            BlobCollection<T> colLearnable = new BlobCollection<T>();
            Dictionary<string, ConstantParameter> rgConstants = new Dictionary<string, ConstantParameter>();
            List<string> rgUsedConstants = new List<string>();
            LayerParameter lastLayer = null;
            LayerParameter lastLayerAdded = null;

            for (int nNodeIdx=0; nNodeIdx<rg.Count; nNodeIdx++)
            {
                NodeProto node = rg[nNodeIdx];
                List<string> rgstrLearnableBlobs = new List<string>();
                LayerParameter layer = null;
                bool bSkipLayer = false;
                string strNodeName = convertWs(node.Name);

                if (string.IsNullOrEmpty(strNodeName))
                    strNodeName = convertWs(node.Output[0]);

                List<string> rgstrExcludedInputs = new List<string>();

                if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Abs))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ABSVAL);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Acos))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.ACOS);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Acosh))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.ACOSH);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Add))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELTWISE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.eltwise_param, EltwiseParameter.EltwiseOp.SUM);

                    // Skip this layer if adding in an external variable.
                    foreach (string strInput in node.Input)
                    {
                        if (!isInputUsed(p.layer, strInput))
                        {
                            bSkipLayer = true;
                            break;
                        }
                    }
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ArgMin))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ARGMAX);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.argmax_param, ArgMaxParameter.COMPARE_OPERATOR.MIN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ArgMax))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ARGMAX);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.argmax_param, ArgMaxParameter.COMPARE_OPERATOR.MAX);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Asin))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.ASIN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Asinh))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.ASINH);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Atan))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.ATAN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Atanh))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.ATANH);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.AveragePool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.AVE, false);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.BatchNormalization))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.BATCHNORM);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.batch_norm_param);

                    BlobCollection<T> colBlobs = new BlobCollection<T>();

                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        string strInput = convertWs(node.Input[i]);

                        Blob<T> blob = col.FindBlob(strInput);
                        if (blob == null && rgConstants.ContainsKey(strInput))
                        {
                            ConstantParameter constParam = rgConstants[strInput];
                            blob = new Blob<T>(cuda, log, constParam.output_shape.dim);
                            blob.Name = strInput;

                            if (constParam.values_f.Count > 1)
                            {
                                T[] rgData = Utility.ConvertVec<T>(constParam.values_f.ToArray());
                                blob.SetData(rgData);
                            }
                            else if (constParam.values_f.Count == 1)
                            {
                                blob.SetData(constParam.values_f[0]);
                            }
                        }

                        blob.Tag = strNodeName;
                        scale(blob);
                        colBlobs.Add(blob);
                    }

                    colLearnable.Add(colBlobs[2]); // mean
                    colLearnable.Add(colBlobs[3]); // var

                    Blob<T> blobVarCor = new Blob<T>(cuda, log);
                    blobVarCor.ReshapeLike(colBlobs[0]);
                    blobVarCor.SetData(1.0);
                    blobVarCor.Tag = strNodeName;

                    colLearnable.Add(blobVarCor); // varcor.

                    layer.batch_norm_param.scale_bias = true;
                    colLearnable.Add(colBlobs[0]); // scale
                    colLearnable.Add(colBlobs[1]); // bias;

                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        string strInput = convertWs(node.Input[i]);
                        rgstrExcludedInputs.Add(strInput);
                    }
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Clip))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.CLIP);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.clip_param);

                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        Blob<T> input = col.FindBlob(node.Input[i]);
                        if (input != null)
                        {
                            float[] rgF = Utility.ConvertVecF<T>(input.update_cpu_data());
                            if (rgF.Length == 1)
                            {
                                if (i == 1)
                                    layer.clip_param.min = rgF[0];
                                else if (i == 2)
                                    layer.clip_param.max = rgF[0];

                                rgUsedConstants.Add(node.Input[i]);
                            }
                        }
                    }
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Concat))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.CONCAT);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.concat_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Constant))
                {
                    string strOutput = convertWs(node.Output[0]);

                    if (!rgConstants.ContainsKey(strOutput))
                    {
                        layer = new LayerParameter(LayerParameter.LayerType.CONSTANT);
                        layer.name = strNodeName;
                        fillParameter(node.Attribute, layer.name, layer.constant_param);
                        rgConstants.Add(strOutput, layer.constant_param);
                    }
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ConstantOfShape))
                {
                    string strOutput = convertWs(node.Output[0]);

                    layer = new LayerParameter(LayerParameter.LayerType.CONSTANT);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, strOutput, layer.constant_param);
                    rgConstants.Add(strOutput, layer.constant_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Conv))
                {
                    int nGroupReductionFactor = 1;
                    int nFirstLearnableIdx = -1;
                    int nAddLearnableCount = 0;

                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        string strInput = convertWs(node.Input[i]);

                        Blob<T> blob = col.FindBlob(strInput);
                        if (blob == null && rgConstants.ContainsKey(strInput))
                        {
                            ConstantParameter constParam = rgConstants[strInput];
                            blob = new Blob<T>(cuda, log, constParam.output_shape.dim);
                            blob.Name = strInput;
                            blob.Tag = strNodeName;

                            if (constParam.values_f.Count > 1)
                            {
                                T[] rgData = Utility.ConvertVec<T>(constParam.values_f.ToArray());
                                blob.SetData(rgData);
                            }
                            else if (constParam.values_f.Count == 1)
                            {
                                blob.SetData(constParam.values_f[0]);
                            }
                        }

                        rgstrLearnableBlobs.Add(convertWs(node.Input[i]));
                        scale(blob);
                        colLearnable.Add(blob);
                        nAddLearnableCount++;

                        if (i == 1)
                            nFirstLearnableIdx = colLearnable.Count - 1;
                    }

                    layer = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.convolution_param, out nGroupReductionFactor);
                    bool bBiasTerm;
                    string strWt = rgstrLearnableBlobs[0];
                    string strBias = (rgstrLearnableBlobs.Count > 1) ? rgstrLearnableBlobs[1] : null;
                    layer.convolution_param.num_output = (uint)getOutputs(layer.name, colLearnable, strWt, strBias, out bBiasTerm);                   
                    layer.convolution_param.bias_term = true;

                    Filler<T> filler = Filler<T>.Create(cuda, log, layer.convolution_param.bias_filler);
                    List<int> rgBiasShape = new List<int>() { colLearnable[colLearnable.Count - 1].num, 1, 1, 1 };

                    if (nAddLearnableCount == 1)
                    {
                        Blob<T> blobBias = new Blob<T>(cuda, log, rgBiasShape);
                        filler.Fill(blobBias);
                        blobBias.Tag = strNodeName;
                        colLearnable.Add(blobBias);
                    }

                    // If the group was reduced, we must expand and duplicate the weights
                    // by the same ratio.
                    if (nFirstLearnableIdx >= 0 && nGroupReductionFactor > 1)
                    {
                        Blob<T> blob = colLearnable[nFirstLearnableIdx];
                        Blob<T> blobNew = new Blob<T>(cuda, log);
                        blobNew.Tag = blob.Tag;
                        blobNew.Reshape(blob.num, blob.channels * nGroupReductionFactor, blob.height, blob.width);

                        for (int c = 0; c < nGroupReductionFactor; c++)
                        {
                            blobNew.CopyFrom(blob, 0, c);
                        }

                        blob.Dispose();
                        colLearnable[nFirstLearnableIdx] = blobNew;

                        Blob<T> blobBias = new Blob<T>(cuda, log, rgBiasShape);
                        filler.Fill(blobBias);
                        blobBias.Tag = strNodeName;
                        colLearnable.Add(blobBias);
                    }

                    m_bEnableBackward = true;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Cos))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.COS);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Cosh))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.COSH);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Div))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELTWISE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.eltwise_param, EltwiseParameter.EltwiseOp.DIV);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Dropout))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.dropout_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Elu))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELU);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Exp))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.EXP);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Flatten))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.FLATTEN);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.flatten_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Gather))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.GATHER);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.gather_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Gemm))
                {
                    int nAddLearnableCount = 0;

                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        string strInput = convertWs(node.Input[i]);
                        rgstrLearnableBlobs.Add(strInput);
                        Blob<T> blob = col.FindBlob(strInput);
                        scale(blob);
                        colLearnable.Add(blob);
                        nAddLearnableCount++;
                    }

                    layer = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.inner_product_param);
                    bool bBiasTerm;
                    string strWt = rgstrLearnableBlobs[0];
                    string strBias = (rgstrLearnableBlobs.Count > 1) ? rgstrLearnableBlobs[1] : null;
                    layer.inner_product_param.num_output = (uint)getOutputs(layer.name, col, strWt, strBias, out bBiasTerm, 1);
                    layer.inner_product_param.bias_term = true;

                    if (nAddLearnableCount == 1)
                    {
                        Filler<T> filler = Filler<T>.Create(cuda, log, layer.inner_product_param.bias_filler);
                        List<int> rgBiasShape = new List<int>() { colLearnable[colLearnable.Count - 1].num, 1, 1, 1 };

                        Blob<T> blobBias = new Blob<T>(cuda, log, rgBiasShape);
                        filler.Fill(blobBias);
                        blobBias.Tag = strNodeName;
                        colLearnable.Add(blobBias);
                    }

                    m_bEnableBackward = true;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.LRN))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.LRN);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.lrn_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Log))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.LOG);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.LeakyRelu))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.RELU);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.relu_param, true);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.MaxPool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.MAX, false);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Min))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELTWISE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.eltwise_param, EltwiseParameter.EltwiseOp.MIN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Max))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELTWISE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.eltwise_param, EltwiseParameter.EltwiseOp.MAX);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.MatMul))
                {
                    int nAddLearnableCount = 0;

                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        string strInput = convertWs(node.Input[i]);
                        rgstrLearnableBlobs.Add(strInput);
                        Blob<T> blob = col.FindBlob(strInput);
                        scale(blob);
                        colLearnable.Add(blob);
                        nAddLearnableCount++;
                    }

                    layer = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.inner_product_param);
                    bool bBiasTerm;
                    string strWt = rgstrLearnableBlobs[0];
                    string strBias = (rgstrLearnableBlobs.Count > 1) ? rgstrLearnableBlobs[1] : null;
                    layer.inner_product_param.num_output = (uint)getOutputs(layer.name, col, strWt, strBias, out bBiasTerm, 1);
                    layer.inner_product_param.bias_term = true;

                    if (nAddLearnableCount == 1)
                    {
                        Filler<T> filler = Filler<T>.Create(cuda, log, layer.inner_product_param.bias_filler);
                        List<int> rgBiasShape = new List<int>() { colLearnable[colLearnable.Count - 1].num, 1, 1, 1 };

                        Blob<T> blobBias = new Blob<T>(cuda, log, rgBiasShape);
                        filler.Fill(blobBias);
                        blobBias.Tag = strNodeName;
                        colLearnable.Add(blobBias);
                    }

                    m_bEnableBackward = true;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Mul))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELTWISE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.eltwise_param, EltwiseParameter.EltwiseOp.PROD);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Neg))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.NEG);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.GlobalAveragePool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.AVE, true);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.GlobalMaxPool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.MAX, true);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Pow))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POWER);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.PRelu))
                {
                    for (int i = 1; i < node.Input.Count; i++)
                    {
                        string strInput = convertWs(node.Input[i]);
                        rgstrLearnableBlobs.Add(strInput);
                        colLearnable.Add(col.FindBlob(strInput));
                    }

                    layer = new LayerParameter(LayerParameter.LayerType.PRELU);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.prelu_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Relu))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.RELU);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.relu_param, false);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Reshape))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.RESHAPE);
                    layer.name = strNodeName;
                    if (!fillParameter(node.Attribute, layer.reshape_param, col, node.Input, node.Output[0], cuda, log))
                        bSkipLayer = true;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Sin))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.SIN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Sinh))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.SINH);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Sign))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.SIGN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Softmax))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.softmax_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Split))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.SPLIT);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Slice))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.SLICE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.slice_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Sqrt))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.SQRT);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Sub))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.ELTWISE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.eltwise_param, EltwiseParameter.EltwiseOp.SUB);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Tan))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.MATH);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.math_param, MyCaffe.common.MATH_FUNCTION.TAN);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Tanh))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.TANH);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Transpose))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.transpose_param);

                    // Skip the initial transpose if one exists for the data layers default to BGR which already does the transpose.
                    if (p.layer.Count == 0 || p.layer[p.layer.Count - 1].type != LayerParameter.LayerType.DATA)
                        bSkipLayer = true;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Shape))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.RESHAPE);
                    layer.name = strNodeName;
                    fillParameter(node.Attribute, layer.reshape_param, col, node.Input, node.Output[0], cuda, log);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Squeeze))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.SQUEEZE);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Unsqueeze))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.UNSQUEEZE);
                    layer.name = strNodeName;
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ReduceMin))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.REDUCTION);
                    layer.name = strNodeName;
                    layer.reduction_param.operation = ReductionParameter.ReductionOp.MIN;
                    fillParameter(node.Attribute, layer.reduction_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ReduceMax))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.REDUCTION);
                    layer.name = strNodeName;
                    layer.reduction_param.operation = ReductionParameter.ReductionOp.MAX;
                    fillParameter(node.Attribute, layer.reduction_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ReduceSum))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.REDUCTION);
                    layer.name = strNodeName;
                    layer.reduction_param.operation = ReductionParameter.ReductionOp.SUM;
                    fillParameter(node.Attribute, layer.reduction_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ReduceMean))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.REDUCTION);
                    layer.name = strNodeName;
                    layer.reduction_param.operation = ReductionParameter.ReductionOp.MEAN;
                    fillParameter(node.Attribute, layer.reduction_param);
                }

                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.ReduceSumSquare))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.REDUCTION);
                    layer.name = strNodeName;
                    layer.reduction_param.operation = ReductionParameter.ReductionOp.SUMSQ;
                    fillParameter(node.Attribute, layer.reduction_param);
                }

                // For now skipping these layers.
                else if (node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.NonMaxSuppression) ||
                         node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.Cast) ||
                         node.OpType == getOperator(onnx, OnnxDefinitions.OPERATORS.TopK))
                {
                layer = new LayerParameter(LayerParameter.LayerType.CONSTANT);
                    bSkipLayer = true;
                }

                if (layer == null)
                    throw new Exception("Currently the node OpType '" + node.OpType + "' is not supported!");

                foreach (string strInput in node.Input)
                {
                    string strInput1 = convertWs(strInput);
                    if (!rgstrLearnableBlobs.Contains(strInput1) && !rgstrExcludedInputs.Contains(strInput1) && !rgUsedConstants.Contains(strInput1))
                        layer.bottom.Add(strInput1);

                    foreach (string strLearnable in rgstrLearnableBlobs)
                    {
                        rgstrInputs.Remove(strLearnable);
                    }
                }

                foreach (string strOutput in node.Output)
                {
                    layer.top.Add(convertWs(strOutput));
                }

                // Add any constant layers for constant inputs.
                for (int i = 1; i < node.Input.Count; i++)
                {
                    if (!rgConstants.ContainsKey(node.Input[i]) && !rgUsedConstants.Contains(node.Input[i]) && !rgstrLearnableBlobs.Contains(node.Input[i]))
                    {
                        Blob<T> input = col.FindBlob(node.Input[i]);
                        if (input != null)
                        {
                            string strName = convertWs(node.Input[i]);

                            if (bIncludeConstants && isLayerUsed(rg, strName, strName))
                            {
                                LayerParameter layerConst = new LayerParameter(LayerParameter.LayerType.CONSTANT);
                                layerConst.name = strName;
                                layerConst.top.Add(layerConst.name);
                                fillParameter(input, layerConst.constant_param);

                                m_strReport += "Adding constant layer '" + layerConst.ToString() + "'" + Environment.NewLine;
                                p.layer.Add(layerConst);
                            }
                        }
                    }
                }

                if (!bSkipLayer && !ignoreLayer(layer))
                {
                    // If we skip a layer, use the top of the last layer as input.
                    if (lastLayer != null && lastLayerAdded != null && lastLayer.name != lastLayerAdded.name)
                        layer.bottom = new List<string>(lastLayerAdded.top);

                    m_strReport += "Adding layer '" + layer.ToString() + "'" + Environment.NewLine;
                    layer.freeze_learning = !m_bEnableBackward;
                    p.layer.Add(layer);
                    lastLayerAdded = layer;
                }

                lastLayer = layer;
            }

            return colLearnable;
        }

        private bool ignoreLayer(LayerParameter layer)
        {
            foreach (string strIgnore in m_rgstrIgnoreLayerNames)
            {
                if (layer.name.ToUpper().Contains(strIgnore.ToUpper()))
                    return true;
            }

            return false;
        }

        private bool isInputUsed(List<LayerParameter> rgLayers, string strBtm)
        {
            foreach (LayerParameter layer in rgLayers)
            {
                if (layer.top.Contains(strBtm))
                    return true;
            }

            return false;
        }

        private bool isLayerUsed(RepeatedField<NodeProto> rgNodes, string strName, string strTop)
        {
            foreach (NodeProto node in rgNodes)
            {
                if (node.Name != strName)
                {
                    foreach (string strBtm in node.Input)
                    {
                        if (strBtm == strTop)
                            return true;
                    }
                }
            }

            return false;
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ArgMaxParameter p, ArgMaxParameter.COMPARE_OPERATOR op)
        {
            p.operation = op;
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, EltwiseParameter p, EltwiseParameter.EltwiseOp op)
        {
            p.operation = op;
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, BatchNormParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "epsilon")
                {
                    p.eps = attrib.F;
                    break;
                }
                else if (attrib.Name == "momentum")
                {
                    p.moving_average_fraction = attrib.F;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ClipParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "min")
                {
                    p.min = attrib.F;
                    break;
                }
                else if (attrib.Name == "max")
                {
                    p.max = attrib.F;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ConcatParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "axis")
                {
                    p.axis = (int)attrib.I;
                    break;
                }
            }
        }

        private string cleanFileName(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (ch == '\\' || ch == '/' || ch == ':')
                    strOut += "_";
                else
                    strOut += ch;
            }

            return strOut;
        }

        private void saveBinaryData(ConstantParameter p, string strName, float[] rgData)
        {
            string strFile = m_strOriginalPath + "\\" + cleanFileName(strName) + ".bin";

            if (File.Exists(strFile))
                File.Delete(strFile);

            using (FileStream fs = new FileStream(strFile, FileMode.CreateNew, FileAccess.Write))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                BlobProto proto = new BlobProto();
                proto.data = new List<float>(rgData);
                proto.Save(bw);
            }

            p.binary_data_file = strFile;
        }

        private void saveBinaryData(ConstantParameter p, string strName, RepeatedField<float> rgData)
        {
            string strFile = m_strOriginalPath + "\\" + strName + ".bin";

            if (File.Exists(strFile))
                File.Delete(strFile);

            using (FileStream fs = new FileStream(strFile, FileMode.CreateNew, FileAccess.Write))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                BlobProto proto = new BlobProto();
                proto.data = rgData.ToList();
                proto.Save(bw);
            }

            p.binary_data_file = strFile;
        }

        private void fillParameter(Blob<T> input, ConstantParameter p)
        {
            foreach (int nDim in input.shape())
            {
                p.output_shape.dim.Add(nDim);
            }

            float[] rgData = Utility.ConvertVecF<T>(input.update_cpu_data());

            if (rgData.Length <= 32)
                p.values_f = new List<float>(rgData);
            else
                saveBinaryData(p, input.Name, rgData);
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, string strName, ConstantParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "value_float")
                {
                    p.values_f.Add(attrib.F);
                    break;
                }
                else if (attrib.Name == "value_floats")
                {
                    foreach (float f in attrib.Floats)
                    {
                        p.values_f.Add(f);
                    }
                    break;
                }
                if (attrib.Name == "value_int")
                {
                    p.values_f.Add(attrib.I);
                    break;
                }
                else if (attrib.Name == "value_ints")
                {
                    foreach (int i in attrib.Ints)
                    {
                        p.values_f.Add(i);
                    }
                    break;
                }
                else if (attrib.Name == "value")
                {
                    if (attrib.T.DataType == (int)OnnxDefinitions.DataType.FLOAT)
                    {
                        if (attrib.T.FloatData.Count <= 32)
                        {
                            foreach (float f in attrib.T.FloatData)
                            {
                                p.values_f.Add(f);
                            }
                        }
                        else
                        {
                            saveBinaryData(p, strName, attrib.T.FloatData);
                        }
                    }
                    else if (attrib.T.DataType == (int)OnnxDefinitions.DataType.INT64)
                    {
                        if (attrib.T.Int64Data.Count > 0 && attrib.T.Int64Data.Count <= 32)
                        {
                            foreach (float f in attrib.T.FloatData)
                            {
                                p.values_f.Add(f);
                            }
                        }
                        else
                        {
                            byte[] rgRaw = attrib.T.RawData.ToByteArray();
                            int nLen = rgRaw.Length / sizeof(long);
                            long[] rgData1 = new long[nLen];

                            Buffer.BlockCopy(rgRaw, 0, rgData1, 0, rgRaw.Length);

                            foreach (long lVal in rgData1)
                            {
                                p.values_f.Add(lVal);
                            }
                        }
                    }
                    else
                    {
                        throw new Exception("The datatype '" + ((OnnxDefinitions.DataType)attrib.T.DataType).ToString() + "' is not yet supported.");
                    }

                    p.output_shape = new BlobShape();

                    foreach (long nDim in attrib.T.Dims)
                    {
                        p.output_shape.dim.Add((int)nDim);
                    }
                }
                else
                {
                    throw new Exception("The attribute name '" + attrib.Name + "' is not yet supported!");
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ConvolutionParameter p, out int nGroupReductionFactor)
        {
            nGroupReductionFactor = 1;

            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "kernel_shape")
                {
                    long h = attrib.Ints[0];
                    if (h <= 0)
                        throw new Exception("Kernel height shape must be > 0!");

                    long w = attrib.Ints[1];
                    if (w <= 0)
                        throw new Exception("Kernel width shape must be > 0!");

                    if (h == w)
                    {
                        p.kernel_size = new List<uint>() { (uint)w };
                    }
                    else
                    {
                        p.kernel_size = new List<uint>();
                        p.kernel_h = (uint)h;
                        p.kernel_w = (uint)w;
                    }
                }

                else if (attrib.Name == "strides")
                {
                    long h = attrib.Ints[0];
                    if (h <= 0)
                        throw new Exception("stride height shape must be > 0!");

                    long w = attrib.Ints[1];
                    if (w <= 0)
                        throw new Exception("stride width shape must be > 0!");

                    if (h == w)
                    {
                        p.stride = new List<uint>() { (uint)w };
                    }
                    else
                    {
                        p.stride = new List<uint>();
                        p.stride_h = (uint)h;
                        p.stride_w = (uint)w;
                    }
                }

                else if (attrib.Name == "pads")
                {
                    long h = attrib.Ints[0];
                    if (h < 0)
                        throw new Exception("pad height shape must be >= 0!");

                    long w = attrib.Ints[1];
                    if (w < 0)
                        throw new Exception("pad width shape must be >= 0!");

                    if (h == w)
                    {
                        p.pad = new List<uint>() { (uint)w };
                    }
                    else
                    {
                        p.pad = new List<uint>();
                        p.pad_h = (uint)h;
                        p.pad_w = (uint)w;
                    }
                }

                else if (attrib.Name == "dilations")
                {
                    long d = attrib.Ints[0];
                    if (d < 0)
                        throw new Exception("dilation shape must be >= 0!");

                    p.dilation = new List<uint>() { (uint)d };
                }

                else if (attrib.Name == "group")
                {
                    p.group = (uint)attrib.I;

                    if (p.group > 144)
                    {
                        nGroupReductionFactor = 3;
                        p.group /= (uint)nGroupReductionFactor;
                    }
                }
            }

            // Add defaults.
            if (p.kernel_size.Count == 0 && !p.kernel_h.HasValue && !p.kernel_w.HasValue)
                p.kernel_size.Add(3);

            if (p.pad.Count == 0 && !p.pad_h.HasValue && !p.pad_w.HasValue)
                p.pad.Add(0);

            if (p.dilation.Count == 0)
                p.dilation.Add(1);

            if (p.stride.Count == 0 && !p.stride_h.HasValue && !p.stride_w.HasValue)
                p.stride.Add(1);
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, DropoutParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "ratio")
                {
                    p.dropout_ratio = attrib.F;
                }
                else if (attrib.Name == "training_mode")
                {
                    p.active = (attrib.I == 0) ? false : true;
                }
                else if (attrib.Name == "seed")
                {
                    p.seed = attrib.I;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, FlattenParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "axis")
                {
                    p.axis = (int)attrib.I;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, GatherParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "axis")
                {
                    p.axis = (int)attrib.I;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, PoolingParameter p, PoolingParameter.PoolingMethod pool, bool bGlobal)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "kernel_shape")
                {
                    long h = attrib.Ints[0];
                    if (h <= 0)
                        throw new Exception("Kernel height shape must be > 0!");

                    long w = attrib.Ints[1];
                    if (w <= 0)
                        throw new Exception("Kernel width shape must be > 0!");

                    if (h == w)
                    {
                        p.kernel_size = new List<uint>() { (uint)w };
                    }
                    else
                    {
                        p.kernel_size = new List<uint>();
                        p.kernel_h = (uint)h;
                        p.kernel_w = (uint)w;
                    }
                }

                else if (attrib.Name == "strides")
                {
                    long h = attrib.Ints[0];
                    if (h <= 0)
                        throw new Exception("stride height shape must be > 0!");

                    long w = attrib.Ints[1];
                    if (w <= 0)
                        throw new Exception("stride width shape must be > 0!");

                    if (h == w)
                    {
                        p.stride = new List<uint>() { (uint)w };
                    }
                    else
                    {
                        p.stride = new List<uint>();
                        p.stride_h = (uint)h;
                        p.stride_w = (uint)w;
                    }
                }

                else if (attrib.Name == "pads")
                {
                    long h = attrib.Ints[0];
                    if (h < 0)
                        throw new Exception("pad height shape must be >= 0!");

                    long w = attrib.Ints[1];
                    if (w < 0)
                        throw new Exception("pad width shape must be >= 0!");

                    if (h == w)
                    {
                        p.pad = new List<uint>() { (uint)w };
                    }
                    else
                    {
                        p.pad = new List<uint>();
                        p.pad_h = (uint)h;
                        p.pad_w = (uint)w;
                    }
                }
            }

            if (p.kernel_size.Count == 0 && !p.kernel_h.HasValue && !p.kernel_w.HasValue)
                p.kernel_size.Add(3);

            if (p.pad.Count == 0 && !p.pad_h.HasValue && !p.pad_w.HasValue)
                p.pad.Add(0);

            if (p.stride.Count == 0 && !p.stride_h.HasValue && !p.stride_w.HasValue)
                p.stride.Add(1);

            p.global_pooling = bGlobal;
            p.pool = pool;
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, InnerProductParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "transB")
                {
                    if (attrib.I != 0)
                        p.transpose = true;
                    else
                        p.transpose = false;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, LRNParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "alpha")
                {
                    p.alpha = attrib.F;
                }
                else if (attrib.Name == "beta")
                {
                    p.beta = attrib.F;
                }
                else if (attrib.Name == "bias")
                {
                    p.k = attrib.F;
                }
                else if (attrib.Name == "size")
                {
                    if (attrib.I != 0)
                        p.local_size = (uint)attrib.I;
                }
            }
        }

        private bool fillParameter(RepeatedField<AttributeProto> rg, ReshapeParameter p, BlobCollection<T> col, RepeatedField<string> rgInputs, string strOutputBlob, CudaDnn<T> cuda, Log log)
        {
            List<float> rgShape = new List<float>();

            p.shape = new BlobShape();

            if (rgInputs.Count > 1)
            {
                string strInput1 = convertWs(rgInputs[1]);
                Blob<T> shape = col.FindBlob(strInput1);
                if (shape == null)
                    return false;

                float[] rgData = Utility.ConvertVecF<T>(shape.mutable_cpu_data);
                rgShape = new List<float>(rgData);

                foreach (float fDim in rgData)
                {
                    p.shape.dim.Add((int)fDim);
                }

                col.Remove(shape);
            }
            else
            {
                foreach (AttributeProto attrib in rg)
                {
                    if (attrib.Name == "shape")
                    {
                        foreach (long lDim in attrib.Ints)
                        {
                            p.shape.dim.Add((int)lDim);
                        }
                    }
                }
            }

            string strInput = convertWs(rgInputs[0]);
            Blob<T> input = col.FindBlob(strInput);
            if (input != null)
            {
                input.Reshape(p.shape);
                input.Name = convertWs(strOutputBlob);
            }
            else
            {
                Blob<T> output = new Blob<T>(cuda, log);
                output.Reshape(p.shape);
                output.Name = convertWs(strOutputBlob);
                col.Add(output);
            }

            return true;
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, PReLUParameter p)
        {
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ReductionParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "keepdims")
                {
                }
                else if (attrib.Name == "axes")
                {                
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ReLUParameter p, bool bLeaky)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "alpha")
                {
                    p.negative_slope = attrib.F;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, SliceParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "axis")
                {
                    p.axis = (int)attrib.I;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, SoftmaxParameter p)
        {
            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "axis")
                {
                    p.axis = (int)attrib.I;
                    break;
                }
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, MathParameter p, MyCaffe.common.MATH_FUNCTION fn)
        {
            p.function = fn;
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, TransposeParameter p)
        {
            List<int> rgDim = new List<int>();

            foreach (AttributeProto attrib in rg)
            {
                if (attrib.Name == "dim")
                {
                    rgDim.Add((int)attrib.I);
                }
                else if (attrib.Name == "perm")
                {
                    foreach (long val in attrib.Ints)
                    {
                        rgDim.Add((int)val);
                    }
                }
            }

            if (rgDim.Count > 0)
                p.dim = rgDim;
        }
    }

#pragma warning disable 1591

    class LayerDataCollection : IEnumerable<LayerData> /** @private */
    {
        LayerData.TYPE m_type;
        List<LayerData> m_rgItems = new List<LayerData>();

        public LayerDataCollection(LayerData.TYPE type)
        {
            m_type = type;
        }

        public LayerData.TYPE Type
        {
            get { return m_type; }
        }

        public bool Contains(string strName)
        {
            foreach (LayerData item in m_rgItems)
            {
                if (item.Name == strName)
                    return true;
            }

            return false;
        }

        public string FixupTops(int nLayerIdx, LayerParameter p)
        {
            List<LayerData> rgItems = FindAll(nLayerIdx);
            string strReport = "";

            foreach (LayerData item in rgItems)
            {
                if (item.Layer.type != LayerParameter.LayerType.CONSTANT)
                {
                    p.top.Remove(item.Name);
                    strReport += "Removed top '" + item.Name + "' from layer '" + item.Layer.name + "(" + item.Layer.type.ToString() + ") at layer index = " + item.LayerIndex.ToString() + Environment.NewLine;
                }
            }

            return strReport;
        }

        public List<LayerData> FindAll(int nLayerIdx)
        {
            return m_rgItems.Where(p => p.LayerIndex == nLayerIdx).ToList();
        }

        public int Count
        {
            get { return m_rgItems.Count; }
        }

        public void Add(LayerData item)
        {
            m_rgItems.Add(item);
        }

        public void Add(List<string> rg, int nIdx, LayerParameter layer)
        {
            foreach (string str in rg)
            {
                m_rgItems.Add(new LayerData(str, nIdx, layer, m_type));
            }
        }

        public void Remove(List<string> rgstr)
        {
            List<int> rgDelete = new List<int>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                for (int j = 0; j < rgstr.Count; j++)
                {
                    if (m_rgItems[i].Name == rgstr[j])
                        rgDelete.Add(i);
                }
            }

            for (int i = rgDelete.Count - 1; i >= 0; i--)
            {
                m_rgItems.RemoveAt(rgDelete[i]);
            }
        }

        public IEnumerator<LayerData> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        public LayerData this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
        }
    }

    class LayerData /** @private */
    {
        string m_strName;
        int m_nLayerIdx;
        LayerParameter m_layer;
        TYPE m_type;

        public enum TYPE
        {
            TOP,
            BTM
        }

        public LayerData(string strName, int nLayerIdx, LayerParameter layer, TYPE type)
        {
            m_strName = strName;
            m_nLayerIdx = nLayerIdx;
            m_layer = layer;
            m_type = type;
        }

        public string Name
        {
            get { return m_strName; }
        }

        public int LayerIndex
        {
            get { return m_nLayerIdx; }
        }

        public LayerParameter Layer
        {
            get { return m_layer; }
        }

        public TYPE Type
        {
            get { return m_type; }
        }

        public override string ToString()
        {
            return m_strName + "(" + m_type.ToString() + ") at layer '" + m_layer.ToString() + "'(idx = " + m_nLayerIdx.ToString() + ")";
        }
    }

#pragma warning restore 1591
}
