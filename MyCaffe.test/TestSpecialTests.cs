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
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSpecialTests
    {
        [TestMethod]
        public void TestAlexNetCiFar()
        {
            SpecialTestsTest test = new SpecialTestsTest();

            try
            {
                foreach (ISpecialTests t in test.Tests)
                {
                    t.TestAlexNetCiFar();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ISpecialTests : ITest
    {
        void TestAlexNetCiFar();
    }

    class SpecialTestsTest : TestBase
    {
        public SpecialTestsTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Special Test Cases", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SpecialTests<double>(strName, nDeviceID, engine);
            else
                return new SpecialTests<float>(strName, nDeviceID, engine);
        }
    }

    class SpecialTests<T> : TestEx<T>, ISpecialTests
    {
        public SpecialTests(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private ProjectEx getProject()
        {
            ProjectEx p = new ProjectEx("AlexNet Project");

            DatasetFactory factory = new DatasetFactory();
            DatasetDescriptor ds = factory.LoadDataset("CIFAR-10");

            p.SetDataset(ds);

            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\alexnet\\cifar\\alexnet_cifar_train_val.prototxt");
            string strSolverFile = getTestPath("\\MyCaffe\\test_data\\models\\alexnet\\cifar\\alexnet_cifar_solver.prototxt");

            p.LoadModelFile(strModelFile);
            RawProto proto = RawProtoFile.LoadFromFile(strSolverFile);

            RawProto iter = proto.FindChild("max_iter");
            iter.Value = "100";

            p.SolverDescription = proto.ToString();

            return p;
        }

        private string getGpuIds()
        {
            return m_cuda.GetDeviceID().ToString();
        }

        public void TestAlexNetCiFar()
        {
            CancelEvent evtCancel = new CancelEvent();
            SettingsCaffe settings = new SettingsCaffe();
            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND;
            settings.EnableRandomInputSelection = true;
            settings.GpuIds = getGpuIds();

            Trace.WriteLine("Running TestAlexNetCiFar on GPU " + settings.GpuIds);

            ProjectEx p = getProject();
            MyCaffeControl<T> ctrl = new MyCaffeControl<T>(settings, m_log, evtCancel);

            try
            {
                ctrl.Load(Phase.TRAIN, p);
                ctrl.Train();
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                ctrl.Dispose();
            }
        }
    }
}
