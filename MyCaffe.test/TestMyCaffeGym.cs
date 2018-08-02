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
using MyCaffe.imagedb;
using MyCaffe.basecode.descriptors;
using MyCaffe.gym;

namespace MyCaffe.test
{
    [TestClass]
    public class TestMyCaffeGym
    {
        [TestMethod]
        public void TestCartPoleWithUi()
        {
            MyCaffeGymTest test = new MyCaffeGymTest();

            try
            {
                foreach (IMyCaffeGymTest t in test.Tests)
                {
                    t.TestCartPole(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

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
    }


    interface IMyCaffeGymTest : ITest
    {
        void TestCartPole(bool bShowUi);
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
            MyCaffeGymClient gym = new MyCaffeGymClient();

            gym.Open("Cart Pole", true, bShowUi);

            Dictionary<string, int> rgActions = gym.ActionSpace;
            Assert.AreEqual(rgActions.Count, 3);
            Assert.AreEqual(rgActions.ContainsKey("Nothing"), true);
            Assert.AreEqual(rgActions.ContainsKey("MoveLeft"), true);
            Assert.AreEqual(rgActions.ContainsKey("MoveRight"), true);
            Assert.AreEqual(rgActions["Nothing"], 0);
            Assert.AreEqual(rgActions["MoveLeft"], 1);
            Assert.AreEqual(rgActions["MoveRight"], 2);

            gym.Run(0);
            gym.Run(1);
            gym.Run(2);

            Thread.Sleep(5000);

            gym.Close();
        }
    }
}
