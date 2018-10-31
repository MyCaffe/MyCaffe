using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using System.Threading;
using MyCaffe.common;
using System.Drawing;
using System.Diagnostics;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.gym;
using MyCaffe.db.stream;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffeGym
    {
        [TestMethod]
        public void TestCartPoleWithOutUi()
        {
            MyCaffeGymTest test = new MyCaffeGymTest();

            try
            {
                foreach (IMyCaffeGymTest t in test.Tests)
                {
                    t.TestCartPole(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
        [TestMethod]
        public void TestDataGym_General()
        {
            MyCaffeGymTest test = new MyCaffeGymTest();

            try
            {
                foreach (IMyCaffeGymTest t in test.Tests)
                {
                    t.TestDataGymGeneral(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IMyCaffeGymTest : ITest
    {
        void TestCartPole(bool bShowUi);
        void TestDataGymGeneral(bool bShowUi);
    }

    class MyCaffeGymTest : TestBase
    {
        public MyCaffeGymTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MyCaffe Gym Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MyCaffeGymTest<double>(strName, nDeviceID, engine);
            else
                return new MyCaffeGymTest<float>(strName, nDeviceID, engine);
        }
    }

    class MyCaffeGymTest<T> : TestEx<T>, IMyCaffeGymTest
    {
        public MyCaffeGymTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestCartPole(bool bShowUi)
        {
            m_log.WriteHeader("Test Gym - Open");
            GymCollection col = new GymCollection();
            col.Load();

            string strName = "Cart-Pole";
            IXMyCaffeGym igym = col.Find(strName);
            Assert.AreEqual(igym != null, true);

            igym = igym.Clone();
            Assert.AreEqual(igym != null, true);

            igym.Initialize(m_log, null);

            Dictionary<string, int> rgActions = igym.GetActionSpace();
            Assert.AreEqual(rgActions.Count, 2);
            Assert.AreEqual(rgActions.ContainsKey("MoveLeft"), true);
            Assert.AreEqual(rgActions.ContainsKey("MoveRight"), true);
            Assert.AreEqual(rgActions["MoveLeft"], 0);
            Assert.AreEqual(rgActions["MoveRight"], 1);

            igym.Reset();
            igym.Step(0);
            igym.Step(1);

            Thread.Sleep(5000);

            igym.Close();
        }

        public void TestDataGymGeneral(bool bShowUi)
        {
            m_log.WriteHeader("Test Gym - DataGeneral");
            GymCollection col = new GymCollection();
            col.Load();

            string strName = "DataGeneral";
            IXMyCaffeGym igym = col.Find(strName);
            Assert.AreEqual(igym != null, true);

            igym = igym.Clone();
            Assert.AreEqual(igym != null, true);

            string strSchema = "ConnectionCount=1;";
            string strDataPath = getTestPath("\\MyCaffe\\test_data\\data\\char-rnn", true);
            string strParam = "FilePath=" + strDataPath + ";";

            strParam = ParamPacker.Pack(strParam);
            strSchema += "Connection0_CustomQueryName=StdTextFileQuery;";
            strSchema += "Connection0_CustomQueryParam=" + strParam + ";";

            PropertySet ps = new PropertySet(strSchema);
            igym.Initialize(m_log, ps);

            Dictionary<string, int> rgActions = igym.GetActionSpace();
            Assert.AreEqual(rgActions.Count, 0);

            Tuple<State, double, bool> data0 = igym.Reset();
            m_log.CHECK(data0.Item3 != true, "We should not be done just yet.");
            m_log.CHECK_EQ(data0.Item2, 0, "The general data gyms do not have reward values.");

            int nDataLen;
            SimpleDatum sd = data0.Item1.GetData(false, out nDataLen);
            m_log.CHECK(sd != null, "The data should not be null.");
            m_log.CHECK_EQ(sd.ItemCount, nDataLen, "The data length should be the SimpleDatum ItemCount.");
            m_log.CHECK_GT(sd.ItemCount, 0, "There should be data in the SimpleDatum.");

            Tuple<State, double, bool> data1 = igym.Step(1);
            m_log.CHECK(data1.Item3 == true, "We should now be done.");
            m_log.CHECK_EQ(data1.Item2, 0, "The general data gyms do not have reward values.");

            int nDataLen1;
            SimpleDatum sd1 = data1.Item1.GetData(false, out nDataLen1);
            m_log.CHECK(sd1 == null, "The data should be null.");
            m_log.CHECK_EQ(nDataLen1, 0, "The data length should be zero.");

            igym.Close();
        }
    }
}
