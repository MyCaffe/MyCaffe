using Google.Protobuf.Collections;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;
using Onnx;
using OnnxControl;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
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
        /// <param name="container">A container holding the component.</param>
        public MyCaffeConversionControl(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        /// <summary>
        /// Convert a MyCaffe model description, weights and optionally mean image from the MyCaffe model format to the ONNX format and save the result as 
        /// a .onnx model file.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="data">Specifies the MyCaffe model data including the model description, the weights and optionally, the image mean.</param>
        /// <param name="strOutputFile">Specifies the .onnx output file.</param>
        public void ConvertMyCaffeToOnnxFile(CudaDnn<T> cuda, Log log, MyCaffeModelData data, string strOutputFile)
        {
            ModelProto protoOnnx = ConvertMyCaffeToOnnx(cuda, log, data);
            PersistOnnx persist = new PersistOnnx();

            // Save the new 
            persist.Save(protoOnnx, strOutputFile);

        }

        /// <summary>
        /// Convert a MyCaffe model description, weights and optionally mean image from the MyCaffe model format to the ONNX format and return the 
        /// ONNX ModelProto object containing the model in the ONNX format.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="data">Specifies the MyCaffe model data including the model description, the weights and optionally, the image mean.</param>
        /// <returns>The model is returned in the ONNX format as a ModelProto (defined within the OnnxControl)</returns>
        public ModelProto ConvertMyCaffeToOnnx(CudaDnn<T> cuda, Log log, MyCaffeModelData data)
        {
            // Parse the Caffe Model Description;
            RawProto protoMyCaffe = RawProto.Parse(data.ModelDescription);
            NetParameter netParam = NetParameter.FromProto(protoMyCaffe);
            Net<T> net = new Net<T>(cuda, log, netParam, new CancelEvent(), null);

            // Load the weights
            if (data.Weights != null)
            {
                net = new Net<T>(cuda, log, netParam, new CancelEvent(), null);
                PersistCaffe<T> persistCaffe = new PersistCaffe<T>(log, false);
                net.LoadWeights(data.Weights, persistCaffe);
            }

            // Convert the MyCaffe net to an Onnx model.
            ModelProto protoOnnx = convertToOnnx(net);

            // Cleanup
            if (net != null)
                net.Dispose();

            return protoOnnx;
        }

        /// <summary>
        /// Convert a model currently loaded into the MyCaffeControl to an ONNX ModelProto.
        /// </summary>
        /// <param name="ctrl">Specifies the MyCaffeControl object.</param>
        /// <param name="phase">Optionally, specifies the phase (which netork) to convert (default = RUN).</param>
        /// <returns>The ONNX model proto is returns that matches the network converted.</returns>
        public ModelProto ConvertMyCaffeToOnnx(MyCaffeControl<T> ctrl, Phase phase = Phase.RUN)
        {
            Net<T> net = ctrl.GetInternalNet(phase);
            return convertToOnnx(net);
        }

        /// <summary>
        /// Convert a model currently loaded into the MyCaffeControl to an ONNX .onnx model file.
        /// </summary>
        /// <param name="ctrl">Specifies the MyCaffeControl object.</param>
        /// <param name="strOnnxFile">Specifies the output .onnx file.</param>
        /// <param name="phase">Optionally, specifies the phase (which netork) to convert (default = RUN).</param>
        public void ConvertMyCaffeToOnnxFile(MyCaffeControl<T> ctrl, string strOnnxFile, Phase phase = Phase.RUN)
        {
            ModelProto proto = ConvertMyCaffeToOnnx(ctrl, phase);
            PersistOnnx persist = new PersistOnnx();
            persist.Save(proto, strOnnxFile);
        }

        /// <summary>
        /// Convert an ONNX .onnx model file to the MyCaffe model description, weights and optionally mean image.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="strOnnxFile">Specifies the ONNX .onnx file.</param>
        /// <returns>The MyCaffe model description, model weights and image mean are returned as a MyCaffeModelData object.</returns>
        public MyCaffeModelData ConvertOnnxToMyCaffeFromFile(CudaDnn<T> cuda, Log log, string strOnnxFile)
        {
            PersistOnnx persist = new PersistOnnx();
            ModelProto proto = persist.Load(strOnnxFile);
            return ConvertOnnxToMyCaffe(cuda, log, proto);
        }

        /// <summary>
        /// Convert an ONNX ModelProto to the MyCaffe model description, weights and optionally mean image.
        /// </summary>
        /// <param name="cuda">Specifies the connection to cuda uses to interact with the GPU.</param>
        /// <param name="log">Specifies the output log used to show progress.</param>
        /// <param name="onnxModel">Specifies the ONNX model.</param>
        /// <returns>The MyCaffe model description, model weights and image mean are returned as a MyCaffeModelData object.</returns>
        public MyCaffeModelData ConvertOnnxToMyCaffe(CudaDnn<T> cuda, Log log, ModelProto onnxModel)
        {
            Net<T> net = convertToMyCaffe(cuda, log, onnxModel);

            NetParameter netParam = net.net_param;
            RawProto protoMyCaffe = netParam.ToProto("root");

            PersistCaffe<T> persist = new PersistCaffe<T>(log, false);
            byte[] rgWeights = net.SaveWeights(persist, false);

            return new MyCaffeModelData(protoMyCaffe.ToString(), rgWeights);
        }

        private ModelProto convertToOnnx(Net<T> net)
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
            addNodes(proto.Graph.Node, net.layers);
            addTensors(proto.Graph.Initializer, net.learnable_parameters);

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

        private void addTensors(RepeatedField<TensorProto> rg, BlobCollection<T> blobs)
        {
            foreach (Blob<T> blob in blobs)
            {
                TensorProto tensor = new TensorProto();
                tensor.Name = blob.Name;
                tensor.DataType = (int)OnnxDefinitions.DataType.FLOAT;

                foreach (int nShape in blob.shape())
                {
                    tensor.Dims.Add(nShape);
                }

                T[] rgData = blob.mutable_cpu_data;
                foreach (T val in rgData)
                {
                    tensor.FloatData.Add(Convert.ToSingle(val));
                }

                rg.Add(tensor);
            }
        }

        private void addNodes(RepeatedField<NodeProto> rg, List<Layer<T>> rgLayers)
        {
            foreach (Layer<T> layer in rgLayers)
            {
                NodeProto node = new NodeProto();

                node.Name = layer.layer_param.name;

                foreach (string strBottom in layer.layer_param.bottom)
                {
                    node.Input.Add(strBottom);
                }

                foreach (Blob<T> blob in layer.blobs)
                {
                    node.Input.Add(blob.Name);
                }

                foreach (string strTop in layer.layer_param.top)
                {
                    node.Output.Add(strTop);
                }

                switch (layer.type)
                {
                    case LayerParameter.LayerType.CONVOLUTION:
                        node.OpType = OnnxDefinitions.OPERATORS.Conv.ToString();
                        addAttributes(node.Attribute, layer.layer_param.convolution_param);
                        break;

                    case LayerParameter.LayerType.INNERPRODUCT:
                        node.OpType = OnnxDefinitions.OPERATORS.Gemm.ToString();
                        addAttributes(node.Attribute, layer.layer_param.inner_product_param);
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
                        break;

                    case LayerParameter.LayerType.RELU:
                        if (layer.layer_param.relu_param.negative_slope != 0)
                            node.OpType = OnnxDefinitions.OPERATORS.LeakyRelu.ToString();
                        else
                            node.OpType = OnnxDefinitions.OPERATORS.Relu.ToString();
                        addAttributes(node.Attribute, layer.layer_param.relu_param);
                        break;

                    case LayerParameter.LayerType.SOFTMAX:
                        node.OpType = OnnxDefinitions.OPERATORS.Softmax.ToString();
                        addAttributes(node.Attribute, layer.layer_param.softmax_param);
                        break;
                }

                rg.Add(node);
            }
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ConvolutionParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "kernel_shape";
            uint h = (p.kernel_h.HasValue) ? p.kernel_h.Value : p.kernel_size[0];
            attrib.Ints.Add(h);
            uint w = (p.kernel_w.HasValue) ? p.kernel_w.Value : p.kernel_size[0];
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "strides";
            h = (p.stride_h.HasValue) ? p.stride_h.Value : p.stride[0];
            attrib.Ints.Add(h);
            w = (p.stride_w.HasValue) ? p.stride_w.Value : p.stride[0];
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "pad";
            h = (p.pad_h.HasValue) ? p.pad_h.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
            attrib.Ints.Add(h);
            w = (p.pad_w.HasValue) ? p.pad_w.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            if (p.dilation.Count > 0)
            {
                attrib = new AttributeProto();
                attrib.Name = "dilations";
                h = p.dilation[0];
                attrib.Ints.Add(h);
                w = p.dilation[1];
                attrib.Ints.Add(w);
                rgA.Add(attrib);
            }

            attrib = new AttributeProto();
            attrib.Name = "group";
            attrib.I = p.group;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, InnerProductParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "alpha";
            attrib.F = 1.0f;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "beta";
            attrib.F = 0.0f; // see InnerProductLayer.cs line 375
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "transA";
            attrib.I = 0;
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "transB";
            attrib.I = (p.transpose) ? 1 : 0;
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, PoolingParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "kernel_shape";
            uint h = (p.kernel_h.HasValue) ? p.kernel_h.Value : p.kernel_size[0];
            attrib.Ints.Add(h);
            uint w = (p.kernel_w.HasValue) ? p.kernel_w.Value : p.kernel_size[0];
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "strides";
            h = (p.stride_h.HasValue) ? p.stride_h.Value : p.stride[0];
            attrib.Ints.Add(h);
            w = (p.stride_w.HasValue) ? p.stride_w.Value : p.stride[0];
            attrib.Ints.Add(w);
            rgA.Add(attrib);

            attrib = new AttributeProto();
            attrib.Name = "pad";
            h = (p.pad_h.HasValue) ? p.pad_h.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
            attrib.Ints.Add(h);
            w = (p.pad_w.HasValue) ? p.pad_w.Value : (p.pad.Count > 0) ? p.pad[0] : 0;
            attrib.Ints.Add(w);
            rgA.Add(attrib);
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, ReLUParameter p)
        {
            if (p.negative_slope != 0)
            {
                AttributeProto attrib = new AttributeProto();
                attrib.Name = "alpha";
                attrib.F = (float)p.negative_slope;

                rgA.Add(attrib);
            }
        }

        private void addAttributes(RepeatedField<AttributeProto> rgA, SoftmaxParameter p)
        {
            AttributeProto attrib = new AttributeProto();
            attrib.Name = "axis";
            attrib.I = p.axis;
            rgA.Add(attrib);
        }

        private Net<T> convertToMyCaffe(CudaDnn<T> cuda, Log log, ModelProto proto)
        {
            NetParameter netParam = new NetParameter();
            BlobCollection<T> colLearnableBlobs = new BlobCollection<T>();
            OnnxDefinitions onnx = new OnnxDefinitions();

            netParam.name = proto.Graph.Name;
            addInputs(proto.Graph.Input, netParam);
            addTensors(proto.Graph.Initializer, colLearnableBlobs, cuda, log);
            addLayers(proto.Graph.Node, netParam, colLearnableBlobs, onnx);

            RawProto rp = netParam.ToProto("root");
            string str = rp.ToString();

            Net<T> net = new Net<T>(cuda, log, netParam, new CancelEvent(), null);
            net.SetLearnedParameters(colLearnableBlobs);

            return net;
        }

        private void addInputs(RepeatedField<ValueInfoProto> rg, NetParameter p)
        {
            foreach (ValueInfoProto val in rg)
            {
                p.input.Add(val.Name);
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
            }
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
                blob.Name = tensor.Name;

                if (tensor.DataType == (int)OnnxDefinitions.DataType.FLOAT)
                {
                    if (typeof(T) == typeof(float))
                    {
                        float[] rgData = new float[tensor.FloatData.Count];
                        for (int i = 0; i < tensor.FloatData.Count; i++)
                        {
                            rgData[i] = tensor.FloatData[i];
                        }
                        blob.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                    }
                    else
                    {
                        double[] rgData = new double[tensor.FloatData.Count];
                        for (int i = 0; i < tensor.FloatData.Count; i++)
                        {
                            rgData[i] = tensor.FloatData[i];
                        }
                        blob.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                    }
                }
                else if (tensor.DataType == (int)OnnxDefinitions.DataType.DOUBLE)
                {
                    if (typeof(T) == typeof(float))
                    {
                        float[] rgData = new float[tensor.DoubleData.Count];
                        for (int i = 0; i < tensor.DoubleData.Count; i++)
                        {
                            rgData[i] = (float)tensor.DoubleData[i];
                        }
                        blob.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                    }
                    else
                    {
                        double[] rgData = new double[tensor.FloatData.Count];
                        for (int i = 0; i < tensor.DoubleData.Count; i++)
                        {
                            rgData[i] = tensor.DoubleData[i];
                        }
                        blob.mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                    }
                }
                else
                {
                    throw new Exception("Currently only the 'DataType.FLOAT' and 'DataType.DOUBLE' are supported for conversions to MyCaffe.");
                }

                col.Add(blob);
            }
        }

        /// <summary>
        /// Get the number of outputs and whether or not a bias term is set.
        /// </summary>
        /// <remarks>
        /// NOTE: This is a very simplistic method of determining the output size and whether or not a bias exists, 
        /// but this method is entirely dependend on how bias blobs are named by MyCaffe which will certainly not be
        /// the method used by other model builders - for this reason, this method will need to be changed
        /// to support other non-MyCaffe models.
        /// </remarks>
        /// <param name="strLayerName">Specifies the layer name.</param>
        /// <param name="col">Specifies the set of learnable blobs.</param>
        /// <param name="bBiasTerm">If the bias exists, true is returned, otherwise false.</param>
        /// <returns>The number of outputs is returned based on the first rank of the weight blob.</returns>
        private int getOutputs(string strLayerName, BlobCollection<T> col, out bool bBiasTerm)
        {
            string strWt = strLayerName + " weights";
            string strBias = strLayerName + " bias";
            int? nOutputs = null;

            bBiasTerm = false;

            foreach (Blob<T> blob in col)
            {
                if (blob.Name == strWt)
                {
                    blob.Tag = strLayerName;
                    nOutputs = blob.shape()[0];
                }
                else if (blob.Name == strBias)
                {
                    blob.Tag = strLayerName;
                    bBiasTerm = true;
                }

                if (nOutputs.HasValue && bBiasTerm)
                    break;
            }

            if (!nOutputs.HasValue)
                throw new Exception("Could not find the blob '" + strWt + "'!");

            return nOutputs.Value;
        }

        private void addLayers(RepeatedField<NodeProto> rg, NetParameter p, BlobCollection<T> col, OnnxDefinitions onnx)
        {
            List<string> rgstrLearnableBlobs = col.Select(p1 => p1.Name).ToList();

            foreach (NodeProto node in rg)
            {
                LayerParameter layer = null;

                if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.Conv))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.CONVOLUTION);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.convolution_param);
                    bool bBiasTerm;
                    layer.convolution_param.num_output = (uint)getOutputs(layer.name, col, out bBiasTerm);
                    layer.convolution_param.bias_term = bBiasTerm;
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.Gemm))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.inner_product_param);
                    bool bBiasTerm;
                    layer.inner_product_param.num_output = (uint)getOutputs(layer.name, col, out bBiasTerm);
                    layer.inner_product_param.bias_term = bBiasTerm;
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.AveragePool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.AVE, false);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.MaxPool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.MAX, false);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.GlobalAveragePool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.AVE, true);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.GlobalMaxPool))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.POOLING);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.pooling_param, PoolingParameter.PoolingMethod.MAX, true);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.PRelu))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.PRELU);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.prelu_param);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.Relu))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.RELU);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.relu_param, false);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.LeakyRelu))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.RELU);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.relu_param, true);
                }

                else if (node.OpType == onnx.GetString(OnnxDefinitions.OPERATORS.Softmax))
                {
                    layer = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                    layer.name = node.Name;
                    fillParameter(node.Attribute, layer.softmax_param);
                }

                if (layer == null)
                    throw new Exception("Currently the node OpType '" + node.OpType + "' is not supported!");

                foreach (string strInput in node.Input)
                {
                    if (!rgstrLearnableBlobs.Contains(strInput))
                        layer.bottom.Add(strInput);
                }

                foreach (string strOutput in node.Output)
                {
                    layer.top.Add(strOutput);
                }

                p.layer.Add(layer);
            }
        }

        private void fillParameter(RepeatedField<AttributeProto> rg, ConvolutionParameter p)
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

                else if (attrib.Name == "pad")
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

                else if (attrib.Name == "pad")
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

        private void fillParameter(RepeatedField<AttributeProto> rg, PReLUParameter p)
        {
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
    }
}
