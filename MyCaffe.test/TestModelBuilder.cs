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
using MyCaffe.layers.beta;
using MyCaffe.model;

/// <summary>
/// Testing the Model Builders.
/// </summary> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestModelBuilder
    {
        [TestMethod]
        public void TestCreateSolver_SSDPascal()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateSolver();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCreateDeployModel_SSDPascal()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateDeployModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCreateTrainingModel_SSDPascal()
        {
            SSDPascalModelBuilderTest test = new SSDPascalModelBuilderTest();

            try
            {
                foreach (IModelBuilderTest t in test.Tests)
                {
                    t.TestCreateTrainingModel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IModelBuilderTest : ITest
    {
        void TestCreateSolver();
        void TestCreateTrainingModel();
        void TestCreateDeployModel();
    }

    class SSDPascalModelBuilderTest : TestBase
    {
        public SSDPascalModelBuilderTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SSD Pascal Model Builder Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ModelBuilderTest<double>(strName, nDeviceID, engine);
            else
                return new ModelBuilderTest<float>(strName, nDeviceID, engine);
        }
    }

    class ModelBuilderTest<T> : TestEx<T>, IModelBuilderTest
    {
        string m_strBaseDir;

        public ModelBuilderTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_strBaseDir = TestBase.GetTestPath("\\MyCaffe\\test_data\\modelbuilder", true, true);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestCreateSolver()
        {
            SsdPascalModelBuilder builder = new SsdPascalModelBuilder(m_strBaseDir);

            SolverParameter solverParam = builder.CreateSolver();
            RawProto proto = solverParam.ToProto("root");
            string strSolver = proto.ToString();

            RawProto proto2 = RawProto.Parse(strSolver);
            SolverParameter solverParam2 = SolverParameter.FromProto(proto2);

            m_log.CHECK(solverParam2.Compare(solverParam), "The two solver parameters should be the same!");
        }

        public void TestCreateTrainingModel()
        {
            SsdPascalModelBuilder builder = new SsdPascalModelBuilder(m_strBaseDir);
            NetParameter net_param = builder.CreateModel();
            RawProto proto = net_param.ToProto("root");
            string strNet = proto.ToString();

            RawProto proto2 = RawProto.Parse(strNet);
            NetParameter net_param2 = NetParameter.FromProto(proto2);

            m_log.CHECK(net_param2.Compare(net_param), "The two net parameters should be the same!");

            // verify creating the model.
            SolverParameter solver = builder.CreateSolver();
            RawProto protoSolver = solver.ToProto("root");
            string strSolver = protoSolver.ToString();

            SettingsCaffe settings = new SettingsCaffe();
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, m_log, evtCancel);

//            mycaffe.LoadLite(Phase.TRAIN, strSolver, strNet, null);
            mycaffe.Dispose();
        }

        public void TestCreateDeployModel()
        {
            SsdPascalModelBuilder builder = new SsdPascalModelBuilder(m_strBaseDir);
            NetParameter net_param = builder.CreateDeployModel();
            RawProto proto = net_param.ToProto("root");
            string strNet = proto.ToString();

            RawProto proto2 = RawProto.Parse(strNet);
            NetParameter net_param2 = NetParameter.FromProto(proto2);

            m_log.CHECK(net_param2.Compare(net_param), "The two net parameters should be the same!");

            // verify creating the model.
            SettingsCaffe settings = new SettingsCaffe();
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(settings, m_log, evtCancel);

//            mycaffe.LoadToRun(strNet, null, new BlobShape(1, 3, 300, 300));
            mycaffe.Dispose();
        }
    }
}
