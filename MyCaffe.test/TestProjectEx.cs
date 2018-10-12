using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.alpha;

/// <summary>
/// Testing the ProjectEx class.
/// </summary>
/// <remarks>
/// Each project manages, the dataset, model description, solver description and model results that make up a given project.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestProjectEx
    {
        [TestMethod]
        public void TestFindLayerParameter()
        {
            ProjectExTest test = new ProjectExTest();

            try
            {
                foreach (IProjectExTest t in test.Tests)
                {
                    t.TestFindLayerParameter();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IProjectExTest : ITest
    {
        void TestFindLayerParameter();
    }

    class ProjectExTest : TestBase
    {
        public ProjectExTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ProjectEx Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ProjectExTest<double>(strName, nDeviceID, engine);
            else
                return new ProjectExTest<float>(strName, nDeviceID, engine);
        }
    }

    class ProjectExTest<T> : TestEx<T>, IProjectExTest
    {
        public ProjectExTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestFindLayerParameter()
        {
            string strModel = "name: \"test\" " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"data\"                     " +
                                "   type: \"Data\"                     " +
                                "   top: \"data\"                      " +
                                "   top: \"label\"                     " +
                                "   include                            " +
                                "   {                                  " +
                                "      phase: TRAIN                    " +
                                "   }                                  " +
                                "   transform_param                    " +
                                "   {                                  " +
                                "      scale: 0.00390625               " +
                                "      mirror: True                    " +
                                "      use_image_mean: True            " +
                                "   }                                  " +
                                "   data_param                         " +
                                "   {                                  " +
                                "      source: \"CIFAR-10.training\"   " +
                                "      batch_size: 256                 " +
                                "      backend: IMAGEDB                " +
                                "      enable_random_selection: True   " +
                                "   }                                  " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"data\"                     " +
                                "   type: \"Data\"                     " +
                                "   top: \"data\"                      " +
                                "   top: \"label\"                     " +
                                "   include                            " +
                                "   {                                  " +
                                "      phase: TEST                     " +
                                "   }                                  " +
                                "   transform_param                    " +
                                "   {                                  " +
                                "      scale: 0.00390625               " +
                                "      use_image_mean: True            " +
                                "   }                                  " +
                                "   data_param                         " +
                                "   {                                  " +
                                "      source: \"CIFAR-10.testing\"    " +
                                "      batch_size: 50                  " +
                                "      backend: IMAGEDB                " +
                                "      enable_random_selection: True   " +
                                "   }                                  " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"conv1\"                    " +
                                "   type: \"Convolution\"              " +
                                "   bottom: \"data\"                   " +
                                "   top: \"conv1\"                     " +
                                "   param                              " +
                                "   {                                  " +
                                "       lr_mult: 1                     " +
                                "   }                                  " +
                                "   param                              " +
                                "   {                                  " +
                                "      lr_mult: 2                      " +
                                "      decay_mult: 0                   " +
                                "   }                                  " +
                                "   convolution_param                  " +
                                "   {                                  " +
                                "      kernel_size: 5                  " +
                                "      stride: 1                       " +
                                "      pad: 2                          " +
                                "      dilation: 1                     " +
                                "      num_output: 192                 " +
                                "      weight_filler                   " +
                                "      {                               " +
                                "         type: \"xavier\"             " +
                                "         variance_norm: FAN_IN        " +
                                "      }                               " +
                                "      bias_filler                     " +
                                "      {                               " +
                                "         type: \"constant\"           " +
                                "         value: 0.1                   " +
                                "      }                               " +
                                "   }                                  " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"relu1\"                    " +
                                "   type: \"ReLU\"                     " +
                                "   bottom: \"conv1\"                  " +
                                "   top: \"conv1\"                     " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"norm1\"                    " +
                                "   type: \"LRN\"                      " +
                                "   bottom: \"conv1\"                  " +
                                "   top: \"norm1\"                     " +
                                "   lrn_param                          " +
                                "   {                                  " +
                                "      local_size: 5                   " +
                                "      alpha: 0.0001                   " +
                                "      beta: 0.75                      " +
                                "      norm_region: ACROSS_CHANNELS    " +
                                "      k: 1                            " +
                                "   }                                  " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"pool1\"                    " +
                                "   type: \"Pooling\"                  " +
                                "   bottom: \"norm1\"                  " +
                                "   top: \"pool1\"                     " +
                                "   pooling_param                      " +
                                "   {                                  " +
                                "      kernel_size: 3                  " +
                                "      stride: 2                       " +
                                "      pad: 1                          " +
                                "      pool: MAX                       " +
                                "   }                                  " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"dropout1\"                 " +
                                "   type: \"Dropout\"                  " +
                                "   bottom: \"pool1\"                  " +
                                "   top: \"pool1\"                     " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"ip1\"                      " +
                                "   type: \"InnerProduct\"             " +
                                "   bottom: \"pool1\"                  " +
                                "   top: \"ip1\"                       " +
                                "   param                              " +
                                "   {                                  " +
                                "      lr_mult: 1                      " +
                                "      decay_mult: 2                   " +
                                "   }                                  " +
                                "   param                              " +
                                "   {                                  " +
                                "      lr_mult: 1                      " +
                                "      decay_mult: 0                   " +
                                "   }                                  " +
                                "   inner_product_param                " +
                                "   {                                  " +
                                "      num_output: 10                  " +
                                "      bias_term: True                 " +
                                "      weight_filler                   " +
                                "      {                               " +
                                "         type: \"xavier\"             " +
                                "         variance_norm: FAN_IN        " +
                                "      }                               " +
                                "      bias_filler                     " +
                                "      {                               " +
                                "         type: \"constant\"           " +
                                "         value: 0.1                   " +
                                "      }                               " +
                                "      axis: 1                         " +
                                "   }                                  " +
                                "}                                     " +
                                "layer                                 " +
                                "{                                     " +
                                "   name: \"loss\"                     " +
                                "   type: \"SoftmaxWithLoss\"          " +
                                "   bottom: \"ip1\"                    " +
                                "   bottom: \"label\"                  " +
                                "   top: \"loss\"                      " +
                                "}";

            string strVar;

            strVar = ProjectEx.FindLayerParameter(strModel, "data", "Data", "data_param", "source");
            m_log.CHECK(strVar == "CIFAR-10.training", "The first found is not as expected.");

            strVar = ProjectEx.FindLayerParameter(strModel, "data", "Data", "data_param", "source", Phase.TRAIN);
            m_log.CHECK(strVar == "CIFAR-10.training", "The first found is not as expected.");

            strVar = ProjectEx.FindLayerParameter(strModel, "data", "Data", "data_param", "source", Phase.TEST);
            m_log.CHECK(strVar == "CIFAR-10.testing", "The first found is not as expected.");

            strVar = ProjectEx.FindLayerParameter(strModel, null, "Data", "data_param", "source");
            m_log.CHECK(strVar == "CIFAR-10.training", "The first found is not as expected.");

            strVar = ProjectEx.FindLayerParameter(strModel, null, "Data", "data_param", "source", Phase.TRAIN);
            m_log.CHECK(strVar == "CIFAR-10.training", "The first found is not as expected.");

            strVar = ProjectEx.FindLayerParameter(strModel, null, "Data", "data_param", "source", Phase.TEST);
            m_log.CHECK(strVar == "CIFAR-10.testing", "The first found is not as expected.");
        }
    }
}
