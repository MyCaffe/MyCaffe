using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestParameters
    {
        [TestMethod]
        public void TestAccuracyParameter()
        {
            string str = "accuracy_param { axis: -1 ignore_label: 2 }";
            RawProto proto = RawProto.Parse(str).FindChild("accuracy_param");
            AccuracyParameter p = AccuracyParameter.FromProto(proto);

            Assert.AreEqual(p.top_k, (uint)1);
            Assert.AreEqual(p.axis, -1);
            Assert.AreEqual(p.ignore_label.HasValue, true);
            Assert.AreEqual(p.ignore_label.Value, 2);

            RawProto proto2 = p.ToProto("accuracy_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestArgMaxParameter()
        {
            string str = "argmax_param { out_max_val: True axis: -1 }";
            RawProto proto = RawProto.Parse(str).FindChild("argmax_param");
            ArgMaxParameter p = ArgMaxParameter.FromProto(proto);

            Assert.AreEqual(p.out_max_val, true);
            Assert.AreEqual(p.top_k, (uint)1);
            Assert.AreEqual(p.axis, -1);

            RawProto proto2 = p.ToProto("argmax_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestBatchNormParameter()
        {
            string str = "batch_norm_param { moving_average_fraction: 0.888 eps: 1E-08 }";
            RawProto proto = RawProto.Parse(str).FindChild("batch_norm_param");
            BatchNormParameter p = BatchNormParameter.FromProto(proto);

            Assert.AreEqual(p.use_global_stats.GetValueOrDefault(true), true);
            Assert.AreEqual(p.moving_average_fraction, 0.888);
            Assert.AreEqual(p.eps, 1e-8);

            RawProto proto2 = p.ToProto("batch_norm_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestBiasParameter()
        {
            string str = "bias_param { axis: -2 num_axes: 2 filler { type: \"constant\" value: 2 } }";
            RawProto proto = RawProto.Parse(str).FindChild("bias_param");
            BiasParameter p = BiasParameter.FromProto(proto);

            Assert.AreEqual(p.axis, -2);
            Assert.AreEqual(p.num_axes, 2);
            Assert.AreEqual(p.filler.type, "constant");
            Assert.AreEqual(p.filler.value, 2);

            RawProto proto2 = p.ToProto("bias_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestBlobProto()
        {
            string str = "blobproto { shape { dim: 1 dim: 2 dim: 3 dim: 4 } data: 1.1 data: 2.2 data: 3.3 diff: 10.1 diff: 20.2 diff: 30.3 }";
            RawProto proto = RawProto.Parse(str).FindChild("blobproto");
            BlobProto p = BlobProto.FromProto(proto);

            Assert.AreEqual(p.shape.dim.Count, 4);
            Assert.AreEqual(p.shape.dim[0], 1);
            Assert.AreEqual(p.shape.dim[1], 2);
            Assert.AreEqual(p.shape.dim[2], 3);
            Assert.AreEqual(p.shape.dim[3], 4);
            Assert.AreEqual(p.data.Count, 3);
            Assert.AreEqual(p.data[0], 1.1f);
            Assert.AreEqual(p.data[1], 2.2f);
            Assert.AreEqual(p.data[2], 3.3f);
            Assert.AreEqual(p.diff.Count, 3);
            Assert.AreEqual(p.diff[0], 10.1f);
            Assert.AreEqual(p.diff[1], 20.2f);
            Assert.AreEqual(p.diff[2], 30.3f);

            RawProto proto2 = p.ToProto("blobproto");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestBlobProto2()
        {
            string str = "blobproto { num: 1 channels: 2 height: 3 width: 4 double_data: 1.1 double_data: 2.2 double_data: 3.3 double_diff: 10.1 double_diff: 20.2 double_diff: 30.3 }";
            RawProto proto = RawProto.Parse(str).FindChild("blobproto");
            BlobProto p = BlobProto.FromProto(proto);

            Assert.AreEqual(p.shape, null);
            Assert.AreEqual(p.num, 1);
            Assert.AreEqual(p.channels, 2);
            Assert.AreEqual(p.height, 3);
            Assert.AreEqual(p.width, 4);
            Assert.AreEqual(p.double_data.Count, 3);
            Assert.AreEqual(p.double_data[0], 1.1);
            Assert.AreEqual(p.double_data[1], 2.2);
            Assert.AreEqual(p.double_data[2], 3.3);
            Assert.AreEqual(p.double_diff.Count, 3);
            Assert.AreEqual(p.double_diff[0], 10.1);
            Assert.AreEqual(p.double_diff[1], 20.2);
            Assert.AreEqual(p.double_diff[2], 30.3);

            RawProto proto2 = p.ToProto("blobproto");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestConcatParameter()
        {
            string str = "concat_param { axis: -4 concat_dim: 3 }";
            RawProto proto = RawProto.Parse(str).FindChild("concat_param");
            ConcatParameter p = ConcatParameter.FromProto(proto);

            Assert.AreEqual(p.axis, -4);
            Assert.AreEqual(p.concat_dim, (uint)3);

            RawProto proto2 = p.ToProto("concat_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestConvolutionParameter1()
        {
            string str = "convolution_param { engine: CUDNN kernel_size: 4 stride: 2 pad: 1 dilation: 7 num_output: 6 group: 2 weight_filler { type: \"gaussian\" mean: 0.33 std: 0.54 } bias_filler { type: \"xavier\" variance_norm: FAN_OUT } axis: -3 force_nd_im2col: True }";
            RawProto proto = RawProto.Parse(str).FindChild("convolution_param");
            ConvolutionParameter p = ConvolutionParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CUDNN);
            Assert.AreEqual(p.pad.Count, 1);
            Assert.AreEqual(p.pad[0], (uint)1);
            Assert.AreEqual(p.stride.Count, 1);
            Assert.AreEqual(p.stride[0], (uint)2);
            Assert.AreEqual(p.kernel_size.Count, 1);
            Assert.AreEqual(p.kernel_size[0], (uint)4);
            Assert.AreEqual(p.dilation.Count, 1);
            Assert.AreEqual(p.dilation[0], (uint)7);
            Assert.AreEqual(p.num_output, (uint)6);
            Assert.AreEqual(p.bias_term, true);
            Assert.AreEqual(p.group, (uint)2);
            Assert.AreNotEqual(p.weight_filler, null);
            Assert.AreNotEqual(p.bias_filler, null);
            Assert.AreEqual(p.weight_filler.type, "gaussian");
            Assert.AreEqual(p.weight_filler.mean, 0.33);
            Assert.AreEqual(p.weight_filler.std, 0.54);
            Assert.AreEqual(p.bias_filler.type, "xavier");
            Assert.AreEqual(p.bias_filler.variance_norm, FillerParameter.VarianceNorm.FAN_OUT);
            Assert.AreEqual(p.axis, -3);
            Assert.AreEqual(p.force_nd_im2col, true);

            RawProto proto2 = p.ToProto("convolution_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestConvolutionParameter2()
        {
            string str = "convolution_param { engine: CUDNN dilation: 7 kernel_h: 5 kernel_w: 6 stride_h: 3 stride_w: 4 pad_h: 1 pad_w: 2 num_output: 8 bias_term: False group: 9 weight_filler { type: \"uniform\" min: 0.32 max: 2.89 } axis: 3 }";
            RawProto proto = RawProto.Parse(str).FindChild("convolution_param");
            ConvolutionParameter p = ConvolutionParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CUDNN);
            Assert.AreEqual(p.pad.Count, 0);
            Assert.AreEqual(p.stride.Count, 0);
            Assert.AreEqual(p.kernel_size.Count, 0);
            Assert.AreEqual(p.pad_h.HasValue, true);
            Assert.AreEqual(p.pad_h.Value, (uint)1);
            Assert.AreEqual(p.pad_w.HasValue, true);
            Assert.AreEqual(p.pad_w.Value, (uint)2);
            Assert.AreEqual(p.stride_h.HasValue, true);
            Assert.AreEqual(p.stride_h.Value, (uint)3);
            Assert.AreEqual(p.stride_w.HasValue, true);
            Assert.AreEqual(p.stride_w.Value, (uint)4);
            Assert.AreEqual(p.kernel_h.HasValue, true);
            Assert.AreEqual(p.kernel_h.Value, (uint)5);
            Assert.AreEqual(p.kernel_w.HasValue, true);
            Assert.AreEqual(p.kernel_w.Value, (uint)6);
            Assert.AreEqual(p.dilation.Count, 1);
            Assert.AreEqual(p.dilation[0], (uint)7);
            Assert.AreEqual(p.num_output, (uint)8);
            Assert.AreEqual(p.bias_term, false);
            Assert.AreEqual(p.group, (uint)9);
            Assert.AreNotEqual(p.weight_filler, null);
            Assert.AreNotEqual(p.bias_filler, null);
            Assert.AreEqual(p.weight_filler.type, "uniform");
            Assert.AreEqual(p.weight_filler.min, 0.32);
            Assert.AreEqual(p.weight_filler.max, 2.89);
            Assert.AreEqual(p.axis, 3);
            Assert.AreEqual(p.force_nd_im2col, false);

            p.bias_filler = null;

            RawProto proto2 = p.ToProto("convolution_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestDataParameter()
        {
            string str = "data_param { source: \"test\\boo\" batch_size: 63 backend: IMAGEDB prefetch: 6 enable_random_selection: False enable_pair_selection: True }";
            RawProto proto = RawProto.Parse(str).FindChild("data_param");
            DataParameter p = DataParameter.FromProto(proto);

            Assert.AreEqual(p.source, "test\\boo");
            Assert.AreEqual(p.batch_size, (uint)63);
            Assert.AreEqual(p.backend, DataParameter.DB.IMAGEDB);
            Assert.AreEqual(p.prefetch, (uint)6);
            Assert.AreEqual(p.enable_random_selection, false);
            Assert.AreEqual(p.enable_pair_selection, true);

            RawProto proto2 = p.ToProto("data_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestDropoutParameter()
        {
            string str = "dropout_param { dropout_ratio: 0.667 }";
            RawProto proto = RawProto.Parse(str).FindChild("dropout_param");
            DropoutParameter p = DropoutParameter.FromProto(proto);

            Assert.AreEqual(p.dropout_ratio, 0.667);

            RawProto proto2 = p.ToProto("dropout_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestDummyDataParameter1()
        {
            string str = "dummydata_param { data_filler { type: \"uniform\" min: -2.2 max: 3.4 } data_filler { type: \"gaussian\" mean: 0.25 std: 2.34 } shape { dim: 1 dim: 2 } shape { dim: 3 dim: 4 dim: 5 } force_refill: True }";
            RawProto proto = RawProto.Parse(str).FindChild("dummydata_param");
            DummyDataParameter p = DummyDataParameter.FromProto(proto);

            Assert.AreEqual(p.data_filler.Count, 2);
            Assert.AreEqual(p.data_filler[0].type, "uniform");
            Assert.AreEqual(p.data_filler[0].min, -2.2);
            Assert.AreEqual(p.data_filler[0].max, 3.4);
            Assert.AreEqual(p.data_filler[1].type, "gaussian");
            Assert.AreEqual(p.data_filler[1].mean, 0.25);
            Assert.AreEqual(p.data_filler[1].std, 2.34);
            Assert.AreEqual(p.shape.Count, 2);
            Assert.AreEqual(p.shape[0].dim.Count, 2);
            Assert.AreEqual(p.shape[0].dim[0], 1);
            Assert.AreEqual(p.shape[0].dim[1], 2);
            Assert.AreEqual(p.shape[1].dim.Count, 3);
            Assert.AreEqual(p.shape[1].dim[0], 3);
            Assert.AreEqual(p.shape[1].dim[1], 4);
            Assert.AreEqual(p.shape[1].dim[2], 5);
            Assert.AreEqual(p.force_refill, true); // we now refill on each forward pass because of layer memory sharing.

            RawProto proto2 = p.ToProto("dummydata_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestDummyDataParameter2()
        {
            string str = "dummydata_param { data_filler { type: \"uniform\" min: -2.2 max: 3.4 } data_filler { type: \"gaussian\" mean: 0.25 std: 2.34 } num: 1 num: 2 channels: 3 channels: 4 height: 5 height: 6 width: 7 width: 8 force_refill: True }";
            RawProto proto = RawProto.Parse(str).FindChild("dummydata_param");
            DummyDataParameter p = DummyDataParameter.FromProto(proto);

            Assert.AreEqual(p.data_filler.Count, 2);
            Assert.AreEqual(p.data_filler[0].type, "uniform");
            Assert.AreEqual(p.data_filler[0].min, -2.2);
            Assert.AreEqual(p.data_filler[0].max, 3.4);
            Assert.AreEqual(p.data_filler[1].type, "gaussian");
            Assert.AreEqual(p.data_filler[1].mean, 0.25);
            Assert.AreEqual(p.data_filler[1].std, 2.34);
            Assert.AreEqual(p.shape.Count, 0);
            Assert.AreEqual(p.num.Count, 2);
            Assert.AreEqual(p.num[0], (uint)1);
            Assert.AreEqual(p.num[1], (uint)2);
            Assert.AreEqual(p.channels.Count, 2);
            Assert.AreEqual(p.channels[0], (uint)3);
            Assert.AreEqual(p.channels[1], (uint)4);
            Assert.AreEqual(p.height.Count, 2);
            Assert.AreEqual(p.height[0], (uint)5);
            Assert.AreEqual(p.height[1], (uint)6);
            Assert.AreEqual(p.width.Count, 2);
            Assert.AreEqual(p.width[0], (uint)7);
            Assert.AreEqual(p.width[1], (uint)8);
            Assert.AreEqual(p.force_refill, true); // we now refill on each forward pass because of layer memory sharing.

            RawProto proto2 = p.ToProto("dummydata_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestEltwiseParameter()
        {
            string str = "eltwise_param { operation: PROD coeff: -0.982 }";
            RawProto proto = RawProto.Parse(str).FindChild("eltwise_param");
            EltwiseParameter p = EltwiseParameter.FromProto(proto);

            Assert.AreEqual(p.operation, EltwiseParameter.EltwiseOp.PROD);
            Assert.AreEqual(p.coeff.Count, 1);
            Assert.AreEqual(p.coeff[0], -0.982);
            Assert.AreEqual(p.stable_prod_grad, true);

            RawProto proto2 = p.ToProto("eltwise_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestEluParameter()
        {
            string str = "elu_param { alpha: 0.667 }";
            RawProto proto = RawProto.Parse(str).FindChild("elu_param");
            EluParameter p = EluParameter.FromProto(proto);

            Assert.AreEqual(p.alpha, 0.667);

            RawProto proto2 = p.ToProto("elu_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestEmbedParameter()
        {
            string str = "embed_param { num_output: 3 input_dim: 5 weight_filler { type: \"mrsa\" variance_norm: FAN_OUT } bias_filler { type: \"positive_unitball\" } }";
            RawProto proto = RawProto.Parse(str).FindChild("embed_param");
            EmbedParameter p = EmbedParameter.FromProto(proto);

            Assert.AreEqual(p.num_output, (uint)3);
            Assert.AreEqual(p.input_dim, (uint)5);
            Assert.AreEqual(p.bias_term, true);
            Assert.AreNotEqual(p.weight_filler, null);
            Assert.AreEqual(p.weight_filler.type, "mrsa");
            Assert.AreEqual(p.weight_filler.variance_norm, FillerParameter.VarianceNorm.FAN_OUT);
            Assert.AreNotEqual(p.bias_filler, null);
            Assert.AreEqual(p.bias_filler.type, "positive_unitball");

            RawProto proto2 = p.ToProto("embed_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestExpParameter()
        {
            string str = "exp_param { base: 0.667 scale: -2.1 shift: 0.003 }";
            RawProto proto = RawProto.Parse(str).FindChild("exp_param");
            ExpParameter p = ExpParameter.FromProto(proto);

            Assert.AreEqual(p.base_val, 0.667);
            Assert.AreEqual(p.scale, -2.1);
            Assert.AreEqual(p.shift, 0.003);

            RawProto proto2 = p.ToProto("exp_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLabelMappingParameter()
        {
            string str = "labelmapping_param  { mapping: \"0->3\" mapping: \"1->3\" update_database: true reset_database_labels: true }";
            RawProto proto = RawProto.Parse(str).FindChild("labelmapping_param");
            LabelMappingParameter p = LabelMappingParameter.FromProto(proto);

            Assert.AreEqual(p.mapping.Count, 2);
            Assert.AreEqual(p.mapping[0].OriginalLabel, 0);
            Assert.AreEqual(p.mapping[0].NewLabel, 3);
            Assert.AreEqual(p.mapping[0].ConditionBoostEquals, null);
            Assert.AreEqual(p.mapping[1].OriginalLabel, 1);
            Assert.AreEqual(p.mapping[1].NewLabel, 3);
            Assert.AreEqual(p.mapping[1].ConditionBoostEquals, null);

            RawProto proto2 = p.ToProto("labelmapping_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLabelMappingParameterWithConditions()
        {
            string str = "labelmapping_param  { mapping: \"0->3?boost=1\" mapping: \"1->3?boost=0\" update_database: true reset_database_labels: true }";
            RawProto proto = RawProto.Parse(str).FindChild("labelmapping_param");
            LabelMappingParameter p = LabelMappingParameter.FromProto(proto);

            Assert.AreEqual(p.mapping.Count, 2);
            Assert.AreEqual(p.mapping[0].OriginalLabel, 0);
            Assert.AreEqual(p.mapping[0].NewLabel, 3);
            Assert.AreEqual(p.mapping[0].NewLabelConditionFalse, null);
            Assert.AreEqual(p.mapping[0].ConditionBoostEquals, 1);
            Assert.AreEqual(p.mapping[1].OriginalLabel, 1);
            Assert.AreEqual(p.mapping[1].NewLabel, 3);
            Assert.AreEqual(p.mapping[1].NewLabelConditionFalse, null);
            Assert.AreEqual(p.mapping[1].ConditionBoostEquals, 0);

            RawProto proto2 = p.ToProto("labelmapping_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLabelMappingParameterWithConditionsTrueAndFalse()
        {
            string str = "labelmapping_param  { mapping: \"0->3?0,boost=1\" mapping: \"1->3?1,boost=0\" update_database: true reset_database_labels: true }";
            RawProto proto = RawProto.Parse(str).FindChild("labelmapping_param");
            LabelMappingParameter p = LabelMappingParameter.FromProto(proto);

            Assert.AreEqual(p.mapping.Count, 2);
            Assert.AreEqual(p.mapping[0].OriginalLabel, 0);
            Assert.AreEqual(p.mapping[0].NewLabel, 3);
            Assert.AreEqual(p.mapping[0].NewLabelConditionFalse, 0);
            Assert.AreEqual(p.mapping[0].ConditionBoostEquals, 1);
            Assert.AreEqual(p.mapping[1].OriginalLabel, 1);
            Assert.AreEqual(p.mapping[1].NewLabel, 3);
            Assert.AreEqual(p.mapping[1].NewLabelConditionFalse, 1);
            Assert.AreEqual(p.mapping[1].ConditionBoostEquals, 0);

            RawProto proto2 = p.ToProto("labelmapping_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLogParameter()
        {
            string str = "log_param { base: 0.667 scale: -2.1 shift: 0.003 }";
            RawProto proto = RawProto.Parse(str).FindChild("log_param");
            LogParameter p = LogParameter.FromProto(proto);

            Assert.AreEqual(p.base_val, 0.667);
            Assert.AreEqual(p.scale, -2.1);
            Assert.AreEqual(p.shift, 0.003);

            RawProto proto2 = p.ToProto("log_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestPowerParameter()
        {
            string str = "power_param { power: 0.667 scale: -2.1 shift: 0.003 }";
            RawProto proto = RawProto.Parse(str).FindChild("power_param");
            PowerParameter p = PowerParameter.FromProto(proto);

            Assert.AreEqual(p.power, 0.667);
            Assert.AreEqual(p.scale, -2.1);
            Assert.AreEqual(p.shift, 0.003);

            RawProto proto2 = p.ToProto("power_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestFlattenParameter()
        {
            string str = "flatten_param { axis: -2 end_axis: -4 }";
            RawProto proto = RawProto.Parse(str).FindChild("flatten_param");
            FlattenParameter p = FlattenParameter.FromProto(proto);

            Assert.AreEqual(p.axis, -2);
            Assert.AreEqual(p.end_axis, -4);

            RawProto proto2 = p.ToProto("flatten_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestHingeLossParameter()
        {
            string str = "hinge_loss_param { norm: L2 }";
            RawProto proto = RawProto.Parse(str).FindChild("hinge_loss_param");
            HingeLossParameter p = HingeLossParameter.FromProto(proto);

            Assert.AreEqual(p.norm, HingeLossParameter.Norm.L2);

            RawProto proto2 = p.ToProto("hinge_loss_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestInfogainLossParameter()
        {
            string str = "infogain_loss_param { source: \"foo\\boo\" }";
            RawProto proto = RawProto.Parse(str).FindChild("infogain_loss_param");
            InfogainLossParameter p = InfogainLossParameter.FromProto(proto);

            Assert.AreEqual(p.source, "foo\\boo");

            RawProto proto2 = p.ToProto("infogain_loss_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestInnerProductParameter()
        {
            string str = "inner_product_param { num_output: 44 bias_term: True weight_filler { type: \"uniform\" min: -2 max: 4 } bias_filler { type: \"gaussian\" mean: 0.345 std: 0.987 } axis: -3 }";
            RawProto proto = RawProto.Parse(str).FindChild("inner_product_param");
            InnerProductParameter p = InnerProductParameter.FromProto(proto);

            Assert.AreEqual(p.num_output, (uint)44);
            Assert.AreEqual(p.bias_term, true);
            Assert.AreNotEqual(p.weight_filler, null);
            Assert.AreNotEqual(p.bias_filler, null);
            Assert.AreEqual(p.weight_filler.type, "uniform");
            Assert.AreEqual(p.weight_filler.min, -2);
            Assert.AreEqual(p.weight_filler.max, 4);
            Assert.AreEqual(p.bias_filler.type, "gaussian");
            Assert.AreEqual(p.bias_filler.mean, 0.345);
            Assert.AreEqual(p.bias_filler.std, 0.987);
            Assert.AreEqual(p.axis, -3);

            RawProto proto2 = p.ToProto("inner_product_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLossParameter1()
        {
            string str = "loss_param { ignore_label: 3 normalization: BATCH_SIZE }";
            RawProto proto = RawProto.Parse(str).FindChild("loss_param");
            LossParameter p = LossParameter.FromProto(proto);

            Assert.AreEqual(p.ignore_label.HasValue, true);
            Assert.AreEqual(p.ignore_label.Value, 3);
            Assert.AreEqual(p.normalization, LossParameter.NormalizationMode.BATCH_SIZE);

            RawProto proto2 = p.ToProto("loss_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLossParameter2()
        {
            string str = "loss_param { }";
            RawProto proto = RawProto.Parse(str).FindChild("loss_param");
            LossParameter p = LossParameter.FromProto(proto);

            Assert.AreEqual(p.ignore_label.HasValue, false);
            Assert.AreEqual(p.normalization, LossParameter.NormalizationMode.VALID);
            Assert.AreEqual(p.normalize, false);
        }

        [TestMethod]
        public void TestLRNParameter()
        {
            string str = "lrn_param { engine: CAFFE local_size: 20 alpha: 2.3 beta: -3.2 norm_region: WITHIN_CHANNEL k: 33.2 }";
            RawProto proto = RawProto.Parse(str).FindChild("lrn_param");
            LRNParameter p = LRNParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CAFFE);
            Assert.AreEqual(p.local_size, (uint)20);
            Assert.AreEqual(p.alpha, 2.3);
            Assert.AreEqual(p.beta, -3.2);
            Assert.AreEqual(p.norm_region, LRNParameter.NormRegion.WITHIN_CHANNEL);
            Assert.AreEqual(p.k, 33.2);

            RawProto proto2 = p.ToProto("lrn_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestMVNParameter()
        {
            string str = "mvn_param { normalize_variance: False across_channels: True eps: 1E-07 }";
            RawProto proto = RawProto.Parse(str).FindChild("mvn_param");
            MVNParameter p = MVNParameter.FromProto(proto);

            Assert.AreEqual(p.normalize_variance, false);
            Assert.AreEqual(p.across_channels, true);
            Assert.AreEqual(p.eps, 1E-07);

            RawProto proto2 = p.ToProto("mvn_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestOneHotParameter()
        {
            string str = "onehot_param { axis: 1 num_output: 13 min: -1.01 max: 2.02 min_axes: 4 }";
            RawProto proto = RawProto.Parse(str).FindChild("onehot_param");
            OneHotParameter p = OneHotParameter.FromProto(proto);

            Assert.AreEqual(p.axis, 1);
            Assert.AreEqual(p.min, -1.01);
            Assert.AreEqual(p.max, 2.02);
            Assert.AreEqual((int)p.num_output, 13);

            RawProto proto2 = p.ToProto("onehot_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestNetStateParameter()
        {
            string str = "netstate { phase: TRAIN level: 2 stage: \"foo\" }";
            RawProto proto = RawProto.Parse(str).FindChild("netstate");
            NetState p = NetState.FromProto(proto);

            Assert.AreEqual(p.phase, Phase.TRAIN);
            Assert.AreEqual(p.level, 2);
            Assert.AreEqual(p.stage.Count, 1);
            Assert.AreEqual(p.stage[0], "foo");

            RawProto proto2 = p.ToProto("netstate");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestNetStateRuleParameter()
        {
            string str = "netstaterule { phase: TRAIN min_level: 1 max_level: 3 stage: \"foo\" stage: \"boo\" not_stage: \"moo\" }";
            RawProto proto = RawProto.Parse(str).FindChild("netstaterule");
            NetStateRule p = NetStateRule.FromProto(proto);

            Assert.AreEqual(p.phase, Phase.TRAIN);
            Assert.AreEqual(p.min_level.HasValue, true);
            Assert.AreEqual(p.min_level.Value, 1);
            Assert.AreEqual(p.max_level.HasValue, true);
            Assert.AreEqual(p.max_level.Value, 3);
            Assert.AreEqual(p.stage.Count, 2);
            Assert.AreEqual(p.stage[0], "foo");
            Assert.AreEqual(p.stage[1], "boo");
            Assert.AreEqual(p.not_stage.Count, 1);
            Assert.AreEqual(p.not_stage[0], "moo");

            RawProto proto2 = p.ToProto("netstaterule");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestParamSpecParameter()
        {
            string str = "paramspec { name: \"foo\" share_mode: PERMISSIVE lr_mult: 0.99 decay_mult: 3.33 }";
            RawProto proto = RawProto.Parse(str).FindChild("paramspec");
            ParamSpec p = ParamSpec.FromProto(proto);

            Assert.AreEqual(p.name, "foo");
            Assert.AreEqual(p.share_mode, ParamSpec.DimCheckMode.PERMISSIVE);
            Assert.AreEqual(p.lr_mult, 0.99);
            Assert.AreEqual(p.decay_mult, 3.33);

            RawProto proto2 = p.ToProto("paramspec");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestPoolingParameter1()
        {
            string str = "pooling_param { engine: CUDNN kernel_size: 4 stride: 2 pad: 1 pool: AVE global_pooling: True }";
            RawProto proto = RawProto.Parse(str).FindChild("pooling_param");
            PoolingParameter p = PoolingParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CUDNN);
            Assert.AreEqual(p.pad.Count, 1);
            Assert.AreEqual(p.pad[0], (uint)1);
            Assert.AreEqual(p.stride.Count, 1);
            Assert.AreEqual(p.stride[0], (uint)2);
            Assert.AreEqual(p.kernel_size.Count, 1);
            Assert.AreEqual(p.kernel_size[0], (uint)4);
            Assert.AreEqual(p.dilation.Count, 0);
            Assert.AreEqual(p.pool, PoolingParameter.PoolingMethod.AVE);
            Assert.AreEqual(p.global_pooling, true);

            RawProto proto2 = p.ToProto("pooling_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestPoolingParameter2()
        {
            string str = "pooling_param { engine: CUDNN kernel_h: 5 kernel_w: 6 stride_h: 3 stride_w: 4 pad_h: 1 pad_w: 2 pool: STOCHASTIC }";
            RawProto proto = RawProto.Parse(str).FindChild("pooling_param");
            PoolingParameter p = PoolingParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CUDNN);
            Assert.AreEqual(p.pad.Count, 0);
            Assert.AreEqual(p.stride.Count, 0);
            Assert.AreEqual(p.kernel_size.Count, 0);
            Assert.AreEqual(p.pad_h.HasValue, true);
            Assert.AreEqual(p.pad_h.Value, (uint)1);
            Assert.AreEqual(p.pad_w.HasValue, true);
            Assert.AreEqual(p.pad_w.Value, (uint)2);
            Assert.AreEqual(p.stride_h.HasValue, true);
            Assert.AreEqual(p.stride_h.Value, (uint)3);
            Assert.AreEqual(p.stride_w.HasValue, true);
            Assert.AreEqual(p.stride_w.Value, (uint)4);
            Assert.AreEqual(p.kernel_h.HasValue, true);
            Assert.AreEqual(p.kernel_h.Value, (uint)5);
            Assert.AreEqual(p.kernel_w.HasValue, true);
            Assert.AreEqual(p.kernel_w.Value, (uint)6);
            Assert.AreEqual(p.dilation.Count, 0);
            Assert.AreEqual(p.pool, PoolingParameter.PoolingMethod.STOCHASTIC);
            Assert.AreEqual(p.global_pooling, false);

            RawProto proto2 = p.ToProto("pooling_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestPReLUParameter()
        {
            string str = "prelu_param { filler { type: \"bilinear\" } channel_shared: True }";
            RawProto proto = RawProto.Parse(str).FindChild("prelu_param");
            PReLUParameter p = PReLUParameter.FromProto(proto);

            Assert.AreNotEqual(p.filler, null);
            Assert.AreEqual(p.filler.type, "bilinear");
            Assert.AreEqual(p.channel_shared, true);

            RawProto proto2 = p.ToProto("prelu_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestReductionParameter()
        {
            string str = "reduction_param { operation: ASUM axis: -3 coeff: 9.87 }";
            RawProto proto = RawProto.Parse(str).FindChild("reduction_param");
            ReductionParameter p = ReductionParameter.FromProto(proto);

            Assert.AreEqual(p.operation, ReductionParameter.ReductionOp.ASUM);
            Assert.AreEqual(p.axis, -3);
            Assert.AreEqual(p.coeff, 9.87);

            RawProto proto2 = p.ToProto("reduction_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestReLUParameter()
        {
            string str = "relu_param { negative_slope: 2.89 }";
            RawProto proto = RawProto.Parse(str).FindChild("relu_param");
            ReLUParameter p = ReLUParameter.FromProto(proto);

            Assert.AreEqual(p.negative_slope, 2.89);

            RawProto proto2 = p.ToProto("relu_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestReshapeParameter()
        {
            string str = "reshape_param { shape { dim: 2 dim: 4 } axis: -3 num_axes: 3 }";
            RawProto proto = RawProto.Parse(str).FindChild("reshape_param");
            ReshapeParameter p = ReshapeParameter.FromProto(proto);

            Assert.AreEqual(p.shape.dim.Count, 2);
            Assert.AreEqual(p.shape.dim[0], 2);
            Assert.AreEqual(p.shape.dim[1], 4);
            Assert.AreEqual(p.axis, -3);
            Assert.AreEqual(p.num_axes, 3);

            RawProto proto2 = p.ToProto("reshape_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestScaleParameter()
        {
            string str = "scale_param { bias_term: True bias_filler { type: \"gaussian\" mean: 3 std: 1.43 } }";
            RawProto proto = RawProto.Parse(str).FindChild("scale_param");
            ScaleParameter p = ScaleParameter.FromProto(proto);

            Assert.AreEqual(p.bias_term, true);
            Assert.AreNotEqual(p.bias_filler, null);
            Assert.AreEqual(p.bias_filler.type, "gaussian");
            Assert.AreEqual(p.bias_filler.mean, 3);
            Assert.AreEqual(p.bias_filler.std, 1.43);

            RawProto proto2 = p.ToProto("scale_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestSigmoidParameter()
        {
            string str = "sigmoid_param { engine: CUDNN }";
            RawProto proto = RawProto.Parse(str).FindChild("sigmoid_param");
            SigmoidParameter p = SigmoidParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CUDNN);

            RawProto proto2 = p.ToProto("sigmoid_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestTanhParameter()
        {
            string str = "tanh_param { engine: CUDNN }";
            RawProto proto = RawProto.Parse(str).FindChild("tanh_param");
            TanhParameter p = TanhParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CUDNN);

            RawProto proto2 = p.ToProto("tanh_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestSliceParameter()
        {
            string str = "slice_param { axis: -2 slice_point: 2 slice_point: 3 slice_point: 4 slice_dim: 5 }";
            RawProto proto = RawProto.Parse(str).FindChild("slice_param");
            SliceParameter p = SliceParameter.FromProto(proto);

            Assert.AreEqual(p.axis, -2);
            Assert.AreEqual(p.slice_point.Count, 3);
            Assert.AreEqual(p.slice_point[0], (uint)2);
            Assert.AreEqual(p.slice_point[1], (uint)3);
            Assert.AreEqual(p.slice_point[2], (uint)4);
            Assert.AreEqual(p.slice_dim, (uint)5);

            RawProto proto2 = p.ToProto("slice_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestSoftmaxParameter()
        {
            string str = "softmax_param { engine: CAFFE axis: -4 }";
            RawProto proto = RawProto.Parse(str).FindChild("softmax_param");
            SoftmaxParameter p = SoftmaxParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CAFFE);
            Assert.AreEqual(p.axis, -4);

            RawProto proto2 = p.ToProto("softmax_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestSPPParameter()
        {
            string str = "spp_param { engine: CAFFE method: STOCHASTIC pyramid_height: 20 }";
            RawProto proto = RawProto.Parse(str).FindChild("spp_param");
            SPPParameter p = SPPParameter.FromProto(proto);

            Assert.AreEqual(p.engine, EngineParameter.Engine.CAFFE);
            Assert.AreEqual(p.pool, PoolingParameter.PoolingMethod.STOCHASTIC);
            Assert.AreEqual(p.pyramid_height, (uint)20);

            RawProto proto2 = p.ToProto("spp_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestThresholdParameter()
        {
            string str = "threshold_param { threshold: 33.21 }";
            RawProto proto = RawProto.Parse(str).FindChild("threshold_param");
            ThresholdParameter p = ThresholdParameter.FromProto(proto);

            Assert.AreEqual(p.threshold, 33.21);

            RawProto proto2 = p.ToProto("threshold_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestTileParameter()
        {
            string str = "tile_param { axis: 4 tiles: 2 }";
            RawProto proto = RawProto.Parse(str).FindChild("tile_param");
            TileParameter p = TileParameter.FromProto(proto);

            Assert.AreEqual(p.axis, 4);
            Assert.AreEqual(p.tiles, 2);

            RawProto proto2 = p.ToProto("tile_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestTransformParameter1()
        {
            string str = "transform_param { scale: 0.5 mirror: True crop_size: 3 use_imagedb_mean: True force_color: True color_order: RGB }";
            RawProto proto = RawProto.Parse(str).FindChild("transform_param");
            TransformationParameter p = TransformationParameter.FromProto(proto);

            Assert.AreEqual(p.scale, 0.5);
            Assert.AreEqual(p.mirror, true);
            Assert.AreEqual(p.crop_size, (uint)3);
            Assert.AreEqual(p.use_imagedb_mean, true);
            Assert.AreEqual(p.mean_value.Count, 0);
            Assert.AreEqual(p.force_color, true);
            Assert.AreEqual(p.force_gray, false);

            RawProto proto2 = p.ToProto("transform_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestTransformParameter2()
        {
            string str = "transform_param { scale: 3.5 mirror: True crop_size: 3 mean_value: 2 mean_value: 3 mean_value: 4 force_gray: True color_order: RGB }";
            RawProto proto = RawProto.Parse(str).FindChild("transform_param");
            TransformationParameter p = TransformationParameter.FromProto(proto);

            Assert.AreEqual(p.scale, 3.5);
            Assert.AreEqual(p.mirror, true);
            Assert.AreEqual(p.crop_size, (uint)3);
            Assert.AreEqual(p.use_imagedb_mean, false);
            Assert.AreEqual(p.mean_value.Count, 3);
            Assert.AreEqual(p.mean_value[0], 2);
            Assert.AreEqual(p.mean_value[1], 3);
            Assert.AreEqual(p.mean_value[2], 4);
            Assert.AreEqual(p.force_color, false);
            Assert.AreEqual(p.force_gray, true);

            RawProto proto2 = p.ToProto("transform_param");
            string strProto2 = proto2.ToString();
            string strProto1 = proto.ToString();

            Assert.AreEqual(strProto1, strProto2);
        }

        [TestMethod]
        public void TestLayerParameter_Accuracy()
        {
            string str = "layer { name: \"accuracy\" type: \"Accuracy\" bottom: \"fc8\" bottom: \"label\" top: \"accuracy\" include { phase: TEST } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "accuracy");
            Assert.AreEqual(p.type, LayerParameter.LayerType.ACCURACY);
            Assert.AreEqual(p.bottom.Count, 2);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "fc8");
            Assert.AreEqual(p.bottom[1], "label");
            Assert.AreEqual(p.top[0], "accuracy");
            Assert.AreEqual(p.include.Count, 1);
            Assert.AreEqual(p.include[0].phase, Phase.TEST);
            Assert.AreEqual(p.freeze_learning, false);

            Assert.AreNotEqual(p.accuracy_param, null);
        }

        [TestMethod]
        public void TestLayerParameter_Concat()
        {
            string str = "layer { " + Environment.NewLine +
                         "  name: \"inception_3a/output\" " + Environment.NewLine +
                         "  type: \"Concat\" " + Environment.NewLine +
                         "  bottom: \"inception_3a/1x1\" " + Environment.NewLine +
                         "  bottom: \"inception_3a/3x3\" " + Environment.NewLine +
                         "  bottom: \"inception_3a/5x5\" " + Environment.NewLine +
                         "  bottom: \"inception_3a/pool_proj\" " + Environment.NewLine +
                         "  top: \"inception_3a/output\" " + Environment.NewLine +
                         "  freeze_learning: False " + Environment.NewLine +
                         "}";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "inception_3a/output");
            Assert.AreEqual(p.type, LayerParameter.LayerType.CONCAT);
            Assert.AreEqual(p.bottom.Count, 4);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "inception_3a/1x1");
            Assert.AreEqual(p.bottom[1], "inception_3a/3x3");
            Assert.AreEqual(p.bottom[2], "inception_3a/5x5");
            Assert.AreEqual(p.bottom[3], "inception_3a/pool_proj");
            Assert.AreEqual(p.top[0], "inception_3a/output");
            Assert.AreEqual(p.freeze_learning, false);

            Assert.AreNotEqual(p.concat_param, null);
        }

        [TestMethod]
        public void TestLayerParameter_Convolution()
        {
            string str = "layer { name: \"conv1\" type: \"Convolution\" bottom: \"data\" top: \"conv1\" param { lr_mult: 1 decay_mult: 1 } convolution_param { num_output: 96 kernel_size: 11 stride: 4 weight_filler { type: \"gaussian\" std: 0.01 } bias_filler { type: \"constant\" value: 0 } } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "conv1");
            Assert.AreEqual(p.type, LayerParameter.LayerType.CONVOLUTION);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "data");
            Assert.AreEqual(p.top[0], "conv1");

            Assert.AreEqual(p.parameters.Count, 1);
            Assert.AreEqual(p.parameters[0].lr_mult, 1);
            Assert.AreEqual(p.parameters[0].decay_mult, 1);

            Assert.AreNotEqual(p.convolution_param, null);
            Assert.AreEqual(p.convolution_param.num_output, (uint)96);
            Assert.AreEqual(p.convolution_param.kernel_size.Count, 1);
            Assert.AreEqual(p.convolution_param.kernel_size[0], (uint)11);
            Assert.AreEqual(p.convolution_param.stride.Count, 1);
            Assert.AreEqual(p.convolution_param.stride[0], (uint)4);
            Assert.AreNotEqual(p.convolution_param.weight_filler, null);
            Assert.AreEqual(p.convolution_param.weight_filler.type, "gaussian");
            Assert.AreEqual(p.convolution_param.weight_filler.std, 0.01);
            Assert.AreNotEqual(p.convolution_param.bias_filler, null);
            Assert.AreEqual(p.convolution_param.bias_filler.type, "constant");
            Assert.AreEqual(p.convolution_param.bias_filler.value, 0);
        }

        [TestMethod]
        public void TestLayerParameter_Data()
        {
            string str = "layer { name: \"data\" type: \"Data\" top: \"data\" top: \"label\" include { phase: TRAIN } transform_param { mirror: true crop_size: 227 mean_file: \"data/ilsvc12/imagenet_mean.binaryproto\" } data_param { source: \"examples/imagenet/ilsvrc12_train_lmdb\" batch_size: 256 backend: LMDB } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "data");
            Assert.AreEqual(p.type, LayerParameter.LayerType.DATA);
            Assert.AreEqual(p.top.Count, 2);
            Assert.AreEqual(p.top[0], "data");
            Assert.AreEqual(p.top[1], "label");
            Assert.AreEqual(p.include.Count, 1);
            Assert.AreEqual(p.include[0].phase, Phase.TRAIN);

            Assert.AreNotEqual(p.transform_param, null);
            Assert.AreEqual(p.transform_param.mirror, true);
            Assert.AreEqual(p.transform_param.crop_size, (uint)227);
            Assert.AreEqual(p.transform_param.use_imagedb_mean, true);

            Assert.AreNotEqual(p.data_param, null);
            Assert.AreEqual(p.data_param.source, "examples/imagenet/ilsvrc12_train_lmdb");
            Assert.AreEqual(p.data_param.batch_size, (uint)256);
            Assert.AreEqual(p.data_param.backend, DataParameter.DB.IMAGEDB);
        }

        [TestMethod]
        public void TestLayerParameter_Data2()
        {
            string str = "layer { " + Environment.NewLine +
                         "  name: \"data\" " + Environment.NewLine + 
                         "  type: \"Data\" " + Environment.NewLine + 
                         "  top: \"data\" " + Environment.NewLine + 
                         "  top: \"label\" " + Environment.NewLine + 
                         "  include { " + Environment.NewLine + 
                         "    phase: TRAIN " + Environment.NewLine + 
                         "  } " + Environment.NewLine + 
                         "  transform_param { " + Environment.NewLine +
                         "    mirror: false " + Environment.NewLine +
                         "    crop_size: 224 " + Environment.NewLine + 
                         "    mean_value: 104 " + Environment.NewLine +
                         "    mean_value: 117 " + Environment.NewLine +
                         "    mean_value: 123 " + Environment.NewLine +
                         "  } " + Environment.NewLine + 
                         "  data_param { " + Environment.NewLine +
                         "    source: \"examples/imagenet/ilsvrc12_train_lmdb\" " + Environment.NewLine +
                         "    batch_size: 50 " + Environment.NewLine +
                         "    backend: LMDB " + Environment.NewLine +
                         "  } " + Environment.NewLine +
                         "}";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "data");
            Assert.AreEqual(p.type, LayerParameter.LayerType.DATA);
            Assert.AreEqual(p.top.Count, 2);
            Assert.AreEqual(p.top[0], "data");
            Assert.AreEqual(p.top[1], "label");
            Assert.AreEqual(p.include.Count, 1);
            Assert.AreEqual(p.include[0].phase, Phase.TRAIN);

            Assert.AreNotEqual(p.transform_param, null);
            Assert.AreEqual(p.transform_param.mirror, false);
            Assert.AreEqual(p.transform_param.crop_size, (uint)224);
            Assert.AreEqual(p.transform_param.use_imagedb_mean, false);
            Assert.AreEqual(p.transform_param.mean_value.Count, 3);
            Assert.AreEqual(p.transform_param.mean_value[0], 104);
            Assert.AreEqual(p.transform_param.mean_value[1], 117);
            Assert.AreEqual(p.transform_param.mean_value[2], 123);

            Assert.AreNotEqual(p.data_param, null);
            Assert.AreEqual(p.data_param.source, "examples/imagenet/ilsvrc12_train_lmdb");
            Assert.AreEqual(p.data_param.batch_size, (uint)50);
            Assert.AreEqual(p.data_param.backend, DataParameter.DB.IMAGEDB);
        }

        [TestMethod]
        public void TestLayerParameter_Dropout()
        {
            string str = "layer { name: \"drop6\" type: \"Dropout\" bottom: \"fc6\" top: \"fc6\" dropout_param { dropout_ratio: 0.5 } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "drop6");
            Assert.AreEqual(p.type, LayerParameter.LayerType.DROPOUT);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "fc6");
            Assert.AreEqual(p.top[0], "fc6");

            Assert.AreNotEqual(p.dropout_param, null);
            Assert.AreEqual(p.dropout_param.dropout_ratio, 0.5);
        }

        [TestMethod]
        public void TestLayerParameter_InnerProduct()
        {
            string str = "layer { name: \"fc6\" type: \"InnerProduct\" bottom: \"pool1\" top: \"fc6\" param { lr_mult: 1 decay_mult: 1 } param { lr_mult: 1 decay_mult: 0 } inner_product_param { num_output: 4096 weight_filler { type: \"gaussian\" std: 0.005 } bias_filler { type: \"constant\" value: 0.1 } } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "fc6");
            Assert.AreEqual(p.type, LayerParameter.LayerType.INNERPRODUCT);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "pool1");
            Assert.AreEqual(p.top[0], "fc6");
            Assert.AreEqual(p.freeze_learning, false);

            Assert.AreNotEqual(p.inner_product_param, null);
            Assert.AreEqual(p.inner_product_param.num_output, (uint)4096);
            Assert.AreNotEqual(p.inner_product_param.weight_filler, null);
            Assert.AreEqual(p.inner_product_param.weight_filler.type, "gaussian");
            Assert.AreEqual(p.inner_product_param.weight_filler.std, 0.005);
            Assert.AreNotEqual(p.inner_product_param.bias_filler, null);
            Assert.AreEqual(p.inner_product_param.bias_filler.type, "constant");
            Assert.AreEqual(p.inner_product_param.bias_filler.value, 0.1);
        }

        [TestMethod]
        public void TestLayerParameter_InnerProduct_Freeze()
        {
            string str = "layer { name: \"fc6\" type: \"InnerProduct\" bottom: \"pool1\" top: \"fc6\" freeze_learning: True param { lr_mult: 1 decay_mult: 1 } param { lr_mult: 1 decay_mult: 0 } inner_product_param { num_output: 4096 weight_filler { type: \"gaussian\" std: 0.005 } bias_filler { type: \"constant\" value: 0.1 } } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "fc6");
            Assert.AreEqual(p.type, LayerParameter.LayerType.INNERPRODUCT);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "pool1");
            Assert.AreEqual(p.top[0], "fc6");
            Assert.AreEqual(p.freeze_learning, true);

            Assert.AreNotEqual(p.inner_product_param, null);
            Assert.AreEqual(p.inner_product_param.num_output, (uint)4096);
            Assert.AreNotEqual(p.inner_product_param.weight_filler, null);
            Assert.AreEqual(p.inner_product_param.weight_filler.type, "gaussian");
            Assert.AreEqual(p.inner_product_param.weight_filler.std, 0.005);
            Assert.AreNotEqual(p.inner_product_param.bias_filler, null);
            Assert.AreEqual(p.inner_product_param.bias_filler.type, "constant");
            Assert.AreEqual(p.inner_product_param.bias_filler.value, 0.1);
        }

        [TestMethod]
        public void TestLayerParameter_LRN()
        {
            string str = "layer { name: \"norm1\" type: \"LRN\" bottom: \"conv1\" top: \"norm1\" lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "norm1");
            Assert.AreEqual(p.type, LayerParameter.LayerType.LRN);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "conv1");
            Assert.AreEqual(p.top[0], "norm1");

            Assert.AreNotEqual(p.lrn_param, null);
            Assert.AreEqual(p.lrn_param.local_size, (uint)5);
            Assert.AreEqual(p.lrn_param.alpha, 0.0001);
            Assert.AreEqual(p.lrn_param.beta, 0.75);
        }

        [TestMethod]
        public void TestLayerParameter_Pooling()
        {
            string str = "layer { name: \"pool1\" type: \"Pooling\" bottom: \"norm1\" top: \"pool1\" pooling_param { pool: MAX kernel_size: 3 stride: 2 } }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "pool1");
            Assert.AreEqual(p.type, LayerParameter.LayerType.POOLING);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "norm1");
            Assert.AreEqual(p.top[0], "pool1");

            Assert.AreNotEqual(p.pooling_param, null);
            Assert.AreEqual(p.pooling_param.pool, PoolingParameter.PoolingMethod.MAX);
            Assert.AreEqual(p.pooling_param.kernel_size.Count, 1);
            Assert.AreEqual(p.pooling_param.kernel_size[0], (uint)3);
            Assert.AreEqual(p.pooling_param.stride.Count, 1);
            Assert.AreEqual(p.pooling_param.stride[0], (uint)2);
        }

        [TestMethod]
        public void TestLayerParameter_ReLU()
        {
            string str = "layer { name: \"relu1\" type: \"ReLU\" bottom: \"conv1\" top: \"conv1\" }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "relu1");
            Assert.AreEqual(p.type, LayerParameter.LayerType.RELU);
            Assert.AreEqual(p.bottom.Count, 1);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "conv1");
            Assert.AreEqual(p.top[0], "conv1");

            Assert.AreNotEqual(p.relu_param, null);
        }

        [TestMethod]
        public void TestLayerParameter_SoftmaxWithLoss()
        {
            string str = "layer { name: \"loss\" type: \"SoftmaxWithLoss\" bottom: \"fc8\" bottom: \"label\" top: \"loss\" }";
            RawProto proto = RawProto.Parse(str).FindChild("layer");
            LayerParameter p = LayerParameter.FromProto(proto);

            Assert.AreEqual(p.name, "loss");
            Assert.AreEqual(p.type, LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            Assert.AreEqual(p.bottom.Count, 2);
            Assert.AreEqual(p.top.Count, 1);
            Assert.AreEqual(p.bottom[0], "fc8");
            Assert.AreEqual(p.bottom[1], "label");
            Assert.AreEqual(p.top[0], "loss");

            Assert.AreNotEqual(p.loss_param, null);
            Assert.AreNotEqual(p.softmax_param, null);
        }

        [TestMethod]
        public void TestNetParameter()
        {
            string str =    "name: \"LeNet\" " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"mnist\" " + Environment.NewLine +
                            "  type: \"Data\" " + Environment.NewLine +
                            "  top: \"data\" " + Environment.NewLine +
                            "  top: \"label\" " + Environment.NewLine +
                            "  include { " + Environment.NewLine +
                            "    phase: TRAIN " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  transform_param { " + Environment.NewLine +
                            "    scale: 0.00390625 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  data_param { " + Environment.NewLine +
                            "    source: \"examples/mnist/mnist_train_lmdb\" " + Environment.NewLine +
                            "    batch_size: 64 " + Environment.NewLine +
                            "    backend: LMDB " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"mnist\" " + Environment.NewLine +
                            "  type: \"Data\" " + Environment.NewLine +
                            "  top: \"data\" " + Environment.NewLine +
                            "  top: \"label\" " + Environment.NewLine +
                            "  include { " + Environment.NewLine +
                            "    phase: TEST " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  transform_param { " + Environment.NewLine +
                            "    scale: 0.00390625 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  data_param {" + Environment.NewLine +
                            "    source: \"examples/mnist/mnist_test_lmdb\" " + Environment.NewLine +
                            "    batch_size: 100 " + Environment.NewLine +
                            "    backend: LMDB " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"conv1\" " + Environment.NewLine +
                            "  type: \"Convolution\" " + Environment.NewLine +
                            "  bottom: \"data\" " + Environment.NewLine +
                            "  top: \"conv1\" " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 1 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 2 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  convolution_param { " + Environment.NewLine +
                            "    num_output: 20 " + Environment.NewLine +
                            "    kernel_size: 5 " + Environment.NewLine +
                            "    stride: 1 " + Environment.NewLine +
                            "    weight_filler { " + Environment.NewLine +
                            "      type: \"xavier\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "    bias_filler { " + Environment.NewLine +
                            "      type: \"constant\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"pool1\" " + Environment.NewLine +
                            "  type: \"Pooling\" " + Environment.NewLine +
                            "  bottom: \"conv1\" " + Environment.NewLine +
                            "  top: \"pool1\" " + Environment.NewLine +
                            "  pooling_param { " + Environment.NewLine +
                            "    pool: MAX " + Environment.NewLine +
                            "    kernel_size: 2 " + Environment.NewLine +
                            "    stride: 2 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"conv2\" " + Environment.NewLine +
                            "  type: \"Convolution\" " + Environment.NewLine +
                            "  bottom: \"pool1\" " + Environment.NewLine +
                            "  top: \"conv2\" " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 1 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 2 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  convolution_param { " + Environment.NewLine +
                            "    num_output: 50 " + Environment.NewLine +
                            "    kernel_size: 5 " + Environment.NewLine +
                            "    stride: 1 " + Environment.NewLine +
                            "    weight_filler { " + Environment.NewLine +
                            "      type: \"xavier\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "    bias_filler { " + Environment.NewLine +
                            "      type: \"constant\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "    name: \"pool2\" " + Environment.NewLine +
                            "    type: \"Pooling\" " + Environment.NewLine +
                            "    bottom: \"conv2\" " + Environment.NewLine +
                            "    top: \"pool2\" " + Environment.NewLine +
                            "    pooling_param { " + Environment.NewLine +
                            "      pool: MAX " + Environment.NewLine +
                            "      kernel_size: 2 " + Environment.NewLine +
                            "      stride: 2 " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"ip1\" " + Environment.NewLine +
                            "  type: \"InnerProduct\" " + Environment.NewLine +
                            "  bottom: \"pool2\" " + Environment.NewLine +
                            "  top: \"ip1\" " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 1 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 2 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  inner_product_param { " + Environment.NewLine +
                            "    num_output: 500 " + Environment.NewLine +
                            "    weight_filler { " + Environment.NewLine +
                            "      type: \"xavier\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "    bias_filler { " + Environment.NewLine +
                            "      type: \"constant\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"relu1\" " + Environment.NewLine +
                            "  type: \"ReLU\" " + Environment.NewLine +
                            "  bottom: \"ip1\" " + Environment.NewLine +
                            "  top: \"ip1\" " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"ip2\" " + Environment.NewLine +
                            "  type: \"InnerProduct\" " + Environment.NewLine +
                            "  bottom: \"ip1\" " + Environment.NewLine +
                            "  top: \"ip2\" " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 1 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  param { " + Environment.NewLine +
                            "    lr_mult: 2 " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "  inner_product_param { " + Environment.NewLine +
                            "    num_output: 10 " + Environment.NewLine +
                            "    weight_filler { " + Environment.NewLine +
                            "      type: \"xavier\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "    bias_filler { " + Environment.NewLine +
                            "      type: \"constant\" " + Environment.NewLine +
                            "    } " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"accuracy\" " + Environment.NewLine +
                            "  type: \"Accuracy\" " + Environment.NewLine +
                            "  bottom: \"ip2\" " + Environment.NewLine +
                            "  bottom: \"label\" " + Environment.NewLine +
                            "  top: \"accuracy\" " + Environment.NewLine +
                            "  include { " + Environment.NewLine +
                            "    phase: TEST " + Environment.NewLine +
                            "  } " + Environment.NewLine +
                            "} " + Environment.NewLine +
                            "layer { " + Environment.NewLine +
                            "  name: \"loss\" " + Environment.NewLine +
                            "  type: \"SoftmaxWithLoss\" " + Environment.NewLine +
                            "  bottom: \"ip2\" " + Environment.NewLine +
                            "  bottom: \"label\" " + Environment.NewLine +
                            "  top: \"loss\" " + Environment.NewLine +
                            "}";
            RawProto proto = RawProto.Parse(str);
            NetParameter p = NetParameter.FromProto(proto);
            int nIdx = 0;

            Assert.AreEqual(p.name, "LeNet");
            Assert.AreEqual(p.layer.Count, 11);

            //--- layer 0: Data ---

            Assert.AreEqual(p.layer[nIdx].name, "mnist");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.DATA);
            Assert.AreEqual(p.layer[nIdx].top.Count, 2);
            Assert.AreEqual(p.layer[nIdx].top[0], "data");
            Assert.AreEqual(p.layer[nIdx].top[1], "label");
            Assert.AreEqual(p.layer[nIdx].include.Count, 1);
            Assert.AreEqual(p.layer[nIdx].include[0].phase, Phase.TRAIN);
            
            Assert.AreNotEqual(p.layer[nIdx].transform_param, null);
            Assert.AreEqual(p.layer[nIdx].transform_param.scale, 0.00390625);

            Assert.AreNotEqual(p.layer[nIdx].data_param, null);
            Assert.AreEqual(p.layer[nIdx].data_param.source, "examples/mnist/mnist_train_lmdb");
            Assert.AreEqual(p.layer[nIdx].data_param.batch_size, (uint)64);
            Assert.AreEqual(p.layer[nIdx].data_param.backend, DataParameter.DB.IMAGEDB);

            //--- layer 1: Data ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "mnist");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.DATA);
            Assert.AreEqual(p.layer[nIdx].top.Count, 2);
            Assert.AreEqual(p.layer[nIdx].top[0], "data");
            Assert.AreEqual(p.layer[nIdx].top[1], "label");
            Assert.AreEqual(p.layer[nIdx].include.Count, 1);
            Assert.AreEqual(p.layer[nIdx].include[0].phase, Phase.TEST);

            Assert.AreNotEqual(p.layer[nIdx].transform_param, null);
            Assert.AreEqual(p.layer[nIdx].transform_param.scale, 0.00390625);

            Assert.AreNotEqual(p.layer[nIdx].data_param, null);
            Assert.AreEqual(p.layer[nIdx].data_param.source, "examples/mnist/mnist_test_lmdb");
            Assert.AreEqual(p.layer[nIdx].data_param.batch_size, (uint)100);
            Assert.AreEqual(p.layer[nIdx].data_param.backend, DataParameter.DB.IMAGEDB);

            //--- layer 2: Convolution ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "conv1");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.CONVOLUTION);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "data");
            Assert.AreEqual(p.layer[nIdx].top[0], "conv1");

            Assert.AreEqual(p.layer[nIdx].parameters.Count, 2);
            Assert.AreEqual(p.layer[nIdx].parameters[0].lr_mult, 1);
            Assert.AreEqual(p.layer[nIdx].parameters[1].lr_mult, 2);

            Assert.AreNotEqual(p.layer[nIdx].convolution_param, null);
            Assert.AreEqual(p.layer[nIdx].convolution_param.num_output, (uint)20);
            Assert.AreEqual(p.layer[nIdx].convolution_param.kernel_size.Count, 1);
            Assert.AreEqual(p.layer[nIdx].convolution_param.kernel_size[0], (uint)5);
            Assert.AreEqual(p.layer[nIdx].convolution_param.stride.Count, 1);
            Assert.AreEqual(p.layer[nIdx].convolution_param.stride[0], (uint)1);
            Assert.AreNotEqual(p.layer[nIdx].convolution_param.weight_filler, null);
            Assert.AreEqual(p.layer[nIdx].convolution_param.weight_filler.type, "xavier");
            Assert.AreNotEqual(p.layer[nIdx].convolution_param.bias_filler, null);
            Assert.AreEqual(p.layer[nIdx].convolution_param.bias_filler.type, "constant");

            //--- layer 3: Pooling ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "pool1");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.POOLING);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "conv1");
            Assert.AreEqual(p.layer[nIdx].top[0], "pool1");

            Assert.AreNotEqual(p.layer[nIdx].pooling_param, null);
            Assert.AreEqual(p.layer[nIdx].pooling_param.pool, PoolingParameter.PoolingMethod.MAX);
            Assert.AreEqual(p.layer[nIdx].pooling_param.kernel_size.Count, 1);
            Assert.AreEqual(p.layer[nIdx].pooling_param.kernel_size[0], (uint)2);
            Assert.AreEqual(p.layer[nIdx].pooling_param.stride.Count, 1);
            Assert.AreEqual(p.layer[nIdx].pooling_param.stride[0], (uint)2);

            //--- layer 4: Convolution ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "conv2");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.CONVOLUTION);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "pool1");
            Assert.AreEqual(p.layer[nIdx].top[0], "conv2");

            Assert.AreEqual(p.layer[nIdx].parameters.Count, 2);
            Assert.AreEqual(p.layer[nIdx].parameters[0].lr_mult, 1);
            Assert.AreEqual(p.layer[nIdx].parameters[1].lr_mult, 2);

            Assert.AreNotEqual(p.layer[nIdx].convolution_param, null);
            Assert.AreEqual(p.layer[nIdx].convolution_param.num_output, (uint)50);
            Assert.AreEqual(p.layer[nIdx].convolution_param.kernel_size.Count, 1);
            Assert.AreEqual(p.layer[nIdx].convolution_param.kernel_size[0], (uint)5);
            Assert.AreEqual(p.layer[nIdx].convolution_param.stride.Count, 1);
            Assert.AreEqual(p.layer[nIdx].convolution_param.stride[0], (uint)1);
            Assert.AreNotEqual(p.layer[nIdx].convolution_param.weight_filler, null);
            Assert.AreEqual(p.layer[nIdx].convolution_param.weight_filler.type, "xavier");
            Assert.AreNotEqual(p.layer[nIdx].convolution_param.bias_filler, null);
            Assert.AreEqual(p.layer[nIdx].convolution_param.bias_filler.type, "constant");

            //--- layer 5: Pooling ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "pool2");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.POOLING);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "conv2");
            Assert.AreEqual(p.layer[nIdx].top[0], "pool2");

            Assert.AreNotEqual(p.layer[nIdx].pooling_param, null);
            Assert.AreEqual(p.layer[nIdx].pooling_param.pool, PoolingParameter.PoolingMethod.MAX);
            Assert.AreEqual(p.layer[nIdx].pooling_param.kernel_size.Count, 1);
            Assert.AreEqual(p.layer[nIdx].pooling_param.kernel_size[0], (uint)2);
            Assert.AreEqual(p.layer[nIdx].pooling_param.stride.Count, 1);
            Assert.AreEqual(p.layer[nIdx].pooling_param.stride[0], (uint)2);

            //--- layer 6: InnerProduct ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "ip1");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.INNERPRODUCT);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "pool2");
            Assert.AreEqual(p.layer[nIdx].top[0], "ip1");

            Assert.AreEqual(p.layer[nIdx].parameters.Count, 2);
            Assert.AreEqual(p.layer[nIdx].parameters[0].lr_mult, 1);
            Assert.AreEqual(p.layer[nIdx].parameters[1].lr_mult, 2);

            Assert.AreNotEqual(p.layer[nIdx].inner_product_param, null);
            Assert.AreEqual(p.layer[nIdx].inner_product_param.num_output, (uint)500);
            Assert.AreNotEqual(p.layer[nIdx].inner_product_param.weight_filler, null);
            Assert.AreEqual(p.layer[nIdx].inner_product_param.weight_filler.type, "xavier");
            Assert.AreNotEqual(p.layer[nIdx].inner_product_param.bias_filler, null);
            Assert.AreEqual(p.layer[nIdx].inner_product_param.bias_filler.type, "constant");

            //--- layer 7: ReLU ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "relu1");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.RELU);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "ip1");
            Assert.AreEqual(p.layer[nIdx].top[0], "ip1");

            //--- layer 8: InnerProduct ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "ip2");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.INNERPRODUCT);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 1);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "ip1");
            Assert.AreEqual(p.layer[nIdx].top[0], "ip2");

            Assert.AreEqual(p.layer[nIdx].parameters.Count, 2);
            Assert.AreEqual(p.layer[nIdx].parameters[0].lr_mult, 1);
            Assert.AreEqual(p.layer[nIdx].parameters[1].lr_mult, 2);

            Assert.AreNotEqual(p.layer[nIdx].inner_product_param, null);
            Assert.AreEqual(p.layer[nIdx].inner_product_param.num_output, (uint)10);
            Assert.AreNotEqual(p.layer[nIdx].inner_product_param.weight_filler, null);
            Assert.AreEqual(p.layer[nIdx].inner_product_param.weight_filler.type, "xavier");
            Assert.AreNotEqual(p.layer[nIdx].inner_product_param.bias_filler, null);
            Assert.AreEqual(p.layer[nIdx].inner_product_param.bias_filler.type, "constant");

            //--- layer 9: Accuracy ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "accuracy");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.ACCURACY);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 2);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "ip2");
            Assert.AreEqual(p.layer[nIdx].bottom[1], "label");
            Assert.AreEqual(p.layer[nIdx].top[0], "accuracy");
            Assert.AreEqual(p.layer[nIdx].include.Count, 1);
            Assert.AreEqual(p.layer[nIdx].include[0].phase, Phase.TEST);

            //--- layer 9: Loss ---

            nIdx++;
            Assert.AreEqual(p.layer[nIdx].name, "loss");
            Assert.AreEqual(p.layer[nIdx].type, LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            Assert.AreEqual(p.layer[nIdx].bottom.Count, 2);
            Assert.AreEqual(p.layer[nIdx].top.Count, 1);
            Assert.AreEqual(p.layer[nIdx].bottom[0], "ip2");
            Assert.AreEqual(p.layer[nIdx].bottom[1], "label");
            Assert.AreEqual(p.layer[nIdx].top[0], "loss");

            Assert.AreNotEqual(p.layer[nIdx].loss_param, null);
            Assert.AreNotEqual(p.layer[nIdx].softmax_param, null);
        }

        [TestMethod]
        public void TestSolverParameter()
        {
            string str = "# The train/test net protocol buffer definition" + Environment.NewLine +
                            "net: \"examples/mnist/lenet_train_test.prototxt\"" + Environment.NewLine +
                            "# test_iter specifies how many forward passes the test should carry out." + Environment.NewLine +
                            "# In the case of MNIST, we have test batch size 100 and 100 test iterations," + Environment.NewLine +
                            "# covering the full 10,000 testing images." + Environment.NewLine +
                            "test_iter: 100" + Environment.NewLine +
                            "# Carry out testing every 500 training iterations." + Environment.NewLine +
                            "test_interval: 500" + Environment.NewLine +
                            "# The base learning rate, momentum and the weight decay of the network." + Environment.NewLine +
                            "base_lr: 0.01" + Environment.NewLine +
                            "momentum: 0.9" + Environment.NewLine +
                            "weight_decay: 0.0005" + Environment.NewLine +
                            "# The learning rate policy" + Environment.NewLine +
                            "lr_policy: \"inv\"" + Environment.NewLine +
                            "gamma: 0.0001" + Environment.NewLine +
                            "power: 0.75" + Environment.NewLine +
                            "# Display every 100 iterations" + Environment.NewLine +
                            "display: 100" + Environment.NewLine +
                            "# The maximum number of iterations" + Environment.NewLine +
                            "max_iter: 10000" + Environment.NewLine +
                            "# snapshot intermediate results" + Environment.NewLine +
                            "snapshot: 5000" + Environment.NewLine +
                            "snapshot_prefix: \"examples/mnist/lenet\"" + Environment.NewLine +
                            "# solver mode: CPU or GPU" + Environment.NewLine +
                            "solver_mode: GPU";
            RawProto proto = RawProto.Parse(str);
            SolverParameter p = SolverParameter.FromProto(proto);

            Assert.AreEqual(p.test_iter.Count, 1);
            Assert.AreEqual(p.test_iter[0], 100);
            Assert.AreEqual(p.test_interval, 500);
            Assert.AreEqual(p.base_lr, 0.01);
            Assert.AreEqual(p.momentum, 0.9);
            Assert.AreEqual(p.weight_decay, 0.0005);
            Assert.AreEqual(p.lr_policy, "inv");
            Assert.AreEqual(p.gamma, 0.0001);
            Assert.AreEqual(p.power, 0.75);
            Assert.AreEqual(p.display, 100);
            Assert.AreEqual(p.max_iter, 10000);
            Assert.AreEqual(p.snapshot, 5000);
            Assert.AreEqual(p.snapshot_prefix, "examples/mnist/lenet");
        }
    }
}
