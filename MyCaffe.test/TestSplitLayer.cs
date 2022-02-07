using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using System.Threading;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSplitLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            SplitLayerTest test = new SplitLayerTest();

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward()
        {
            SplitLayerTest test = new SplitLayerTest();

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient()
        {
            SplitLayerTest test = new SplitLayerTest();

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNoInsertion1()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'SoftmaxWithLoss' " +
                          "  bottom: 'innerprod' " +
                          "  bottom: 'label' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strInput);   
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNoInsertion2()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'data_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'data' " +
                          "  top: 'data_split_0' " +
                          "  top: 'data_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_split_0' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_split_1' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1' " +
                          "  bottom: 'innerprod2' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strInput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNoInsertionImageNet()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'CaffeNet' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  data_param { " +
                          "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' " +
                          "    batch_size: 256 " +
                          "  } " +
                          "  transform_param { " +
                          "    crop_size: 227 " +
                          "    mirror: true " +
                          "    mean_file: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' " +
                          "  } " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'conv1' " +
                          "  type: 'Convolution' " +
                          "  convolution_param { " +
                          "    num_output: 96 " +
                          "    kernel_size: 11 " +
                          "    stride: 4 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 0. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'data' " +
                          "  top: 'conv1' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu1' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'conv1' " +
                          "  top: 'conv1' " +
                          "} " +
                          "layer { " +
                          "  name: 'pool1' " +
                          "  type: 'Pooling' " +
                          "  pooling_param { " +
                          "    pool: MAX " +
                          "    kernel_size: 3 " +
                          "    stride: 2 " +
                          "  } " +
                          "  bottom: 'conv1' " +
                          "  top: 'pool1' " +
                          "} " +
                          "layer { " +
                          "  name: 'norm1' " +
                          "  type: 'LRN' " +
                          "  lrn_param { " +
                          "    local_size: 5 " +
                          "    alpha: 0.0001 " +
                          "    beta: 0.75 " +
                          "  } " +
                          "  bottom: 'pool1' " +
                          "  top: 'norm1' " +
                          "} " +
                          "layer { " +
                          "  name: 'conv2' " +
                          "  type: 'Convolution' " +
                          "  convolution_param { " +
                          "    num_output: 256 " +
                          "    group: 2 " +
                          "    kernel_size: 5 " +
                          "    pad: 2 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 1. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'norm1' " +
                          "  top: 'conv2' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu2' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'conv2' " +
                          "  top: 'conv2' " +
                          "} " +
                          "layer { " +
                          "  name: 'pool2' " +
                          "  type: 'Pooling' " +
                          "  pooling_param { " +
                          "    pool: MAX " +
                          "    kernel_size: 3 " +
                          "    stride: 2 " +
                          "  } " +
                          "  bottom: 'conv2' " +
                          "  top: 'pool2' " +
                          "} " +
                          "layer { " +
                          "  name: 'norm2' " +
                          "  type: 'LRN' " +
                          "  lrn_param { " +
                          "    local_size: 5 " +
                          "    alpha: 0.0001 " +
                          "    beta: 0.75 " +
                          "  } " +
                          "  bottom: 'pool2' " +
                          "  top: 'norm2' " +
                          "} " +
                          "layer { " +
                          "  name: 'conv3' " +
                          "  type: 'Convolution' " +
                          "  convolution_param { " +
                          "    num_output: 384 " +
                          "    kernel_size: 3 " +
                          "    pad: 1 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 0. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'norm2' " +
                          "  top: 'conv3' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu3' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'conv3' " +
                          "  top: 'conv3' " +
                          "} " +
                          "layer { " +
                          "  name: 'conv4' " +
                          "  type: 'Convolution' " +
                          "  convolution_param { " +
                          "    num_output: 384 " +
                          "    group: 2 " +
                          "    kernel_size: 3 " +
                          "    pad: 1 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 1. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'conv3' " +
                          "  top: 'conv4' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu4' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'conv4' " +
                          "  top: 'conv4' " +
                          "} " +
                          "layer { " +
                          "  name: 'conv5' " +
                          "  type: 'Convolution' " +
                          "  convolution_param { " +
                          "    num_output: 256 " +
                          "    group: 2 " +
                          "    kernel_size: 3 " +
                          "    pad: 1 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 1. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'conv4' " +
                          "  top: 'conv5' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu5' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'conv5' " +
                          "  top: 'conv5' " +
                          "} " +
                          "layer { " +
                          "  name: 'pool5' " +
                          "  type: 'Pooling' " +
                          "  pooling_param { " +
                          "    kernel_size: 3 " +
                          "    pool: MAX " +
                          "    stride: 2 " +
                          "  } " +
                          "  bottom: 'conv5' " +
                          "  top: 'pool5' " +
                          "} " +
                          "layer { " +
                          "  name: 'fc6' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 4096 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.005 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 1. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'pool5' " +
                          "  top: 'fc6' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu6' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'fc6' " +
                          "  top: 'fc6' " +
                          "} " +
                          "layer { " +
                          "  name: 'drop6' " +
                          "  type: 'Dropout' " +
                          "  dropout_param { " +
                          "    dropout_ratio: 0.5 " +
                          "  } " +
                          "  bottom: 'fc6' " +
                          "  top: 'fc6' " +
                          "} " +
                          "layer { " +
                          "  name: 'fc7' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 4096 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.005 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 1. " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'fc6' " +
                          "  top: 'fc7' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu7' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'fc7' " +
                          "  top: 'fc7' " +
                          "} " +
                          "layer { " +
                          "  name: 'drop7' " +
                          "  type: 'Dropout' " +
                          "  dropout_param { " +
                          "    dropout_ratio: 0.5 " +
                          "  } " +
                          "  bottom: 'fc7' " +
                          "  top: 'fc7' " +
                          "} " +
                          "layer { " +
                          "  name: 'fc8' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 1000 " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "    bias_filler { " +
                          "      type: 'constant' " +
                          "      value: 0 " +
                          "    } " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 1 " +
                          "    decay_mult: 1 " +
                          "  } " +
                          "  param { " +
                          "    lr_mult: 2 " +
                          "    decay_mult: 0 " +
                          "  } " +
                          "  bottom: 'fc7' " +
                          "  top: 'fc8' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'SoftmaxWithLoss' " +
                          "  bottom: 'fc8' " +
                          "  bottom: 'label' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strInput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestNoInsertionWithInPlace()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'innerprod' " +
                          "  top: 'innerprod' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'SoftmaxWithLoss' " +
                          "  bottom: 'innerprod' " +
                          "  bottom: 'label' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strInput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLossInsertion()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'UnsharedWeightsNetwork' " +
                          "force_backward: true " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'DummyData' " +
                          "  dummy_data_param { " +
                          "    num: 5 " +
                          "    channels: 2 " +
                          "    height: 3 " +
                          "    width: 4 " +
                          "    data_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "  } " +
                          "  top: 'data' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerproduct1' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 10 " +
                          "    bias_term: false " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 10 " +
                          "    } " +
                          "  } " +
                          "  param { name: 'unsharedweights1' } " +
                          "  bottom: 'data' " +
                          "  top: 'innerproduct1' " +
                          "  loss_weight: 2.5 " +
                          "} " +
                          "layer { " +
                          "  name: 'innerproduct2' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 10 " +
                          "    bias_term: false " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 10 " +
                          "    } " +
                          "  } " +
                          "  param { name: 'unsharedweights2' } " +
                          "  bottom: 'data' " +
                          "  top: 'innerproduct2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerproduct1' " +
                          "  bottom: 'innerproduct2' " +
                          "} ";
            string strExpectedOutput =
                          "name: 'UnsharedWeightsNetwork' " +
                          "force_backward: true " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'DummyData' " +
                          "  dummy_data_param { " +
                          "    num: 5 " +
                          "    channels: 2 " +
                          "    height: 3 " +
                          "    width: 4 " +
                          "    data_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 0.01 " +
                          "    } " +
                          "  } " +
                          "  top: 'data' " +
                          "} " +
                          "layer { " +
                          "  name: 'data_data_0_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'data' " +
                          "  top: 'data_data_0_split_0' " +
                          "  top: 'data_data_0_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerproduct1' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 10 " +
                          "    bias_term: false " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 10 " +
                          "    } " +
                          "  } " +
                          "  param { name: 'unsharedweights1' } " +
                          "  bottom: 'data_data_0_split_0' " +
                          "  top: 'innerproduct1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerproduct1_innerproduct1_0_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'innerproduct1' " +
                          "  top: 'innerproduct1_innerproduct1_0_split_0' " +
                          "  top: 'innerproduct1_innerproduct1_0_split_1' " +
                          "  loss_weight: 2.5 " +
                          "  loss_weight: 0 " +
                          "} " +
                          "layer { " +
                          "  name: 'innerproduct2' " +
                          "  type: 'InnerProduct' " +
                          "  inner_product_param { " +
                          "    num_output: 10 " +
                          "    bias_term: false " +
                          "    weight_filler { " +
                          "      type: 'gaussian' " +
                          "      std: 10 " +
                          "    } " +
                          "  } " +
                          "  param { name: 'unsharedweights2' } " +
                          "  bottom: 'data_data_0_split_1' " +
                          "  top: 'innerproduct2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerproduct1_innerproduct1_0_split_1' " +
                          "  bottom: 'innerproduct2' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strExpectedOutput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInsertion()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod1' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod1' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod2' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod2' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod3' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data' " +
                  "  top: 'innerprod3' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss1' " +
                  "  type: 'EuclideanLoss' " +
                  "  bottom: 'innerprod1' " +
                  "  bottom: 'innerprod2' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss2' " +
                  "  type: 'EuclideanLoss' " +
                  "  bottom: 'innerprod2' " +
                  "  bottom: 'innerprod3' " +
                  "} ";
            string strExpectedOutput =
                  "name: 'TestNetwork' " +
                  "layer { " +
                  "  name: 'data' " +
                  "  type: 'Data' " +
                  "  top: 'data' " +
                  "  top: 'label' " +
                  "} " +
                  "layer { " +
                  "  name: 'data_data_0_split' " +
                  "  type: 'Split' " +
                  "  bottom: 'data' " +
                  "  top: 'data_data_0_split_0' " +
                  "  top: 'data_data_0_split_1' " +
                  "  top: 'data_data_0_split_2' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod1' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data_data_0_split_0' " +
                  "  top: 'innerprod1' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod2' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data_data_0_split_1' " +
                  "  top: 'innerprod2' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod2_innerprod2_0_split' " +
                  "  type: 'Split' " +
                  "  bottom: 'innerprod2' " +
                  "  top: 'innerprod2_innerprod2_0_split_0' " +
                  "  top: 'innerprod2_innerprod2_0_split_1' " +
                  "} " +
                  "layer { " +
                  "  name: 'innerprod3' " +
                  "  type: 'InnerProduct' " +
                  "  bottom: 'data_data_0_split_2' " +
                  "  top: 'innerprod3' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss1' " +
                  "  type: 'EuclideanLoss' " +
                  "  bottom: 'innerprod1' " +
                  "  bottom: 'innerprod2_innerprod2_0_split_0' " +
                  "} " +
                  "layer { " +
                  "  name: 'loss2' " +
                  "  type: 'EuclideanLoss' " +
                  "  bottom: 'innerprod2_innerprod2_0_split_1' " +
                  "  bottom: 'innerprod3' " +
                  "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strExpectedOutput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInsertionTwoTop()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'label' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod3' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod3' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod4' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'label' " +
                          "  top: 'innerprod4' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss1' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1' " +
                          "  bottom: 'innerprod3' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss2' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod2' " +
                          "  bottom: 'innerprod4' " +
                          "} ";
            string strExpectedOutput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'data_data_0_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'data' " +
                          "  top: 'data_data_0_split_0' " +
                          "  top: 'data_data_0_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'label_data_1_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'label' " +
                          "  top: 'label_data_1_split_0' " +
                          "  top: 'label_data_1_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_data_0_split_0' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'label_data_1_split_0' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod3' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_data_0_split_1' " +
                          "  top: 'innerprod3' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod4' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'label_data_1_split_1' " +
                          "  top: 'innerprod4' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss1' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1' " +
                          "  bottom: 'innerprod3' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss2' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod2' " +
                          "  bottom: 'innerprod4' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strExpectedOutput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInputInsertion()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'TestNetwork' " +
                          "input: 'data' " +
                          "input_dim: 10 " +
                          "input_dim: 3 " +
                          "input_dim: 227 " +
                          "input_dim: 227 " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1' " +
                          "  bottom: 'innerprod2' " +
                          "} ";
            string strExpectedOutput =
                          "name: 'TestNetwork' " +
                          "input: 'data' " +
                          "input_dim: 10 " +
                          "input_dim: 3 " +
                          "input_dim: 227 " +
                          "input_dim: 227 " +
                          "layer { " +
                          "  name: 'data_input_0_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'data' " +
                          "  top: 'data_input_0_split_0' " +
                          "  top: 'data_input_0_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_input_0_split_0' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_input_0_split_1' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1' " +
                          "  bottom: 'innerprod2' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strExpectedOutput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestWithInPlace()
        {
            SplitLayerTest test = new SplitLayerTest();
            string strInput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu1' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'innerprod1' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'innerprod1' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss1' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1' " +
                          "  bottom: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss2' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod2' " +
                          "  bottom: 'data' " +
                          "} ";
            string strExpectedOutput =
                          "name: 'TestNetwork' " +
                          "layer { " +
                          "  name: 'data' " +
                          "  type: 'Data' " +
                          "  top: 'data' " +
                          "  top: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'data_data_0_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'data' " +
                          "  top: 'data_data_0_split_0' " +
                          "  top: 'data_data_0_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'data_data_0_split_0' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'relu1' " +
                          "  type: 'ReLU' " +
                          "  bottom: 'innerprod1' " +
                          "  top: 'innerprod1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod1_relu1_0_split' " +
                          "  type: 'Split' " +
                          "  bottom: 'innerprod1' " +
                          "  top: 'innerprod1_relu1_0_split_0' " +
                          "  top: 'innerprod1_relu1_0_split_1' " +
                          "} " +
                          "layer { " +
                          "  name: 'innerprod2' " +
                          "  type: 'InnerProduct' " +
                          "  bottom: 'innerprod1_relu1_0_split_0' " +
                          "  top: 'innerprod2' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss1' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod1_relu1_0_split_1' " +
                          "  bottom: 'label' " +
                          "} " +
                          "layer { " +
                          "  name: 'loss2' " +
                          "  type: 'EuclideanLoss' " +
                          "  bottom: 'innerprod2' " +
                          "  bottom: 'data_data_0_split_1' " +
                          "} ";

            try
            {
                foreach (ISplitLayerTest t in test.Tests)
                {
                    t.RunInsertionTest(strInput, strExpectedOutput);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ISplitLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestGradient();
        void RunInsertionTest(string strInput, string strOutput);
    }

    class SplitLayerTest : TestBase
    {
        public SplitLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Split Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SplitLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SplitLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SplitLayerTest<T> : TestEx<T>, ISplitLayerTest
    {
        Blob<T> m_blob_top_b;
        
        public SplitLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_top_b = new Blob<T>(m_cuda, m_log);

            TopVec.Add(m_blob_top_b);
        }

        protected override void dispose()
        {
            m_blob_top_b.Dispose();
            base.dispose();
        }

        public Blob<T> TopB
        {
            get { return m_blob_top_b; }
        }

        public void RunInsertionTest(string input_param_string, string output_param_string)
        {
            // Test that InsertSplits called on the proto specified by 
            // input_param_string results in the proto specified by
            // output_param_string.
            Net<T> net = new Net<T>(m_cuda, m_log, new NetParameter(), new CancelEvent(), null);

            try
            {
                RawProto proto1 = RawProto.Parse(input_param_string);
                NetParameter input_param = NetParameter.FromProto(proto1);
                RawProto proto2 = RawProto.Parse(output_param_string);
                NetParameter expected_output_param = NetParameter.FromProto(proto2);
                NetParameter actual_output_param = net.InsertSplits(input_param);

                m_log.CHECK(expected_output_param.Compare(actual_output_param), "The expected and actual protos are not equal!");

                // Also test idempotence.
                NetParameter double_split_insert_param = net.InsertSplits(actual_output_param);
                m_log.CHECK(actual_output_param.Compare(double_split_insert_param), "The double and acutal protos are not equal!");
            }
            finally
            {
                net.Dispose();
            }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPLIT);
            SplitLayer<T> layer = new SplitLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num, "Top num should be 2.");
                m_log.CHECK_EQ(3, Top.channels, "Top channels should be 3.");
                m_log.CHECK_EQ(6, Top.height, "Top height should be 6.");
                m_log.CHECK_EQ(5, Top.width, "Top width should be 5.");
                m_log.CHECK_EQ(2, TopB.num, "TopB num should be 2.");
                m_log.CHECK_EQ(3, TopB.channels, "TopB channels should be 3.");
                m_log.CHECK_EQ(6, TopB.height, "TopB height should be 6.");
                m_log.CHECK_EQ(5, TopB.width, "TopB width should be 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPLIT);
            SplitLayer<T> layer = new SplitLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgBottom = convert(Bottom.update_cpu_data());
                double[] rgTopA = convert(Top.update_cpu_data());
                double[] rgTopB = convert(TopB.update_cpu_data());

                for (int i = 0; i < Bottom.count(); i++)
                {
                    double dfBottom = rgBottom[i];
                    m_log.CHECK_EQ(dfBottom, rgTopA[i], "The bottom value should equal the TopA value at " + i.ToString());
                    m_log.CHECK_EQ(dfBottom, rgTopB[i], "The bottom value should equal the TopB value at " + i.ToString());
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPLIT);
            SplitLayer<T> layer = new SplitLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
