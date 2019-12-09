using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.solvers;
using System.Threading;
using MyCaffe.db.image;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSolver
    {
        [TestMethod]
        public void TestInitTrainTestNets()
        {
            SolverTest test = new SolverTest();

            try
            {
                foreach (ISolverTest t in test.Tests)
                {
                    t.TestInitTrainTestNets();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface ISolverTest : ITest
    {
        void TestInitTrainTestNets();
    }

    class SolverTest : TestBase
    {
        public SolverTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Accuracy Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SolverTest<double>(strName, nDeviceID, engine);
            else
                return new SolverTest<float>(strName, nDeviceID, engine);
        }
    }

    class SolverTest<T> : TestEx<T>, ISolverTest
    {
        Solver<T> m_solver;
        IXImageDatabaseBase m_db;
        IXPersist<T> m_persist;

        public SolverTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
            m_persist = new PersistCaffe<T>(m_log, false);
        }

        public override void initialize()
        {
            base.initialize();
            m_db = createImageDb(m_log);
            m_db.InitializeWithDsName1(new SettingsCaffe(), "MNIST");
        }


        protected override void dispose()
        {
            base.dispose();
        }

        public void InitSolverFromProtoString(string strProto)
        {
            RawProto proto = RawProto.Parse(strProto);
            SolverParameter p = SolverParameter.FromProto(proto);

            m_solver = new SGDSolver<T>(m_cuda, m_log, p, new CancelEvent(), new AutoResetEvent(false), new AutoResetEvent(false), m_db, m_persist);
        }

        public void TestInitTrainTestNets()
        {
            string proto =
                 "test_interval: 10 " +
                 "test_iter: 10 " +
                 "test_state: { stage: 'with-softmax' } " +
                 "test_iter: 10 " +
                 "test_state: { } " +
                 "net_param { " +
                 "  name: 'TestNetwork' " +
                 "  layer { " +
                 "    name: 'data' " +
                 "    type: 'DummyData' " +
                 "    dummy_data_param { " +
                 "      shape { " +
                 "        dim: 5 " +
                 "        dim: 2 " +
                 "        dim: 3 " +
                 "        dim: 4 " +
                 "      } " +
                 "      shape { " +
                 "        dim: 5 " +
                 "      } " +
                 "    } " +
                 "    top: 'data' " +
                 "    top: 'label' " +
                 "  } " +
                 "  layer { " +
                 "    name: 'innerprod' " +
                 "    type: 'InnerProduct' " +
                 "    inner_product_param { " +
                 "      num_output: 10 " +
                 "    } " +
                 "    bottom: 'data' " +
                 "    top: 'innerprod' " +
                 "  } " +
                 "  layer { " +
                 "    name: 'accuracy' " +
                 "    type: 'Accuracy' " +
                 "    bottom: 'innerprod' " +
                 "    bottom: 'label' " +
                 "    top: 'accuracy' " +
                 "    exclude: { phase: TRAIN } " +
                 "  } " +
                 "  layer { " +
                 "    name: 'loss' " +
                 "    type: 'SoftmaxWithLoss' " +
                 "    bottom: 'innerprod' " +
                 "    bottom: 'label' " +
                 "    include: { phase: TRAIN } " +
                 "    include: { phase: TEST stage: 'with-softmax' } " +
                 "  } " +
                 "} ";

            InitSolverFromProtoString(proto);

            m_log.CHECK(m_solver.net != null, "The net should not be null.");
            m_log.CHECK(m_solver.net.has_layer("loss"), "The net should have a 'loss' layer.");
            m_log.CHECK(!m_solver.net.has_layer("accuracy"), "The net should have an 'accuracy' layer.");
            m_log.CHECK_EQ(2, m_solver.test_nets.Count, "The solver should have 2 test nets.");
            m_log.CHECK(m_solver.test_nets[0].has_layer("loss"), "The solver test_net[0] should have a 'loss' layer.");
            m_log.CHECK(m_solver.test_nets[0].has_layer("accuracy"), "The solver test_net[0] should have a 'accuracy' layer.");
            m_log.CHECK(!m_solver.test_nets[1].has_layer("loss"), "The solver test_net[1] should have a 'loss' layer.");
            m_log.CHECK(m_solver.test_nets[1].has_layer("accuracy"), "The solver test_net[1] should have a 'accuracy' layer.");
        }
    }
}
