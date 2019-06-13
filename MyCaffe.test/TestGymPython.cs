using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.gym.python;

namespace MyCaffe.test
{
    [TestClass]
    public class TestGymPython
    {
        [TestMethod]
        public void TestGym()
        {
            MyCaffePythonGym gym = new MyCaffePythonGym();
            Random random = new Random();

            gym.Initialize("ATARI", "FrameSkip=4;GameROM=C:\\Program~Files\\SignalPop\\AI~Designer\\roms\\pong.bin");

            string strName = gym.Name;
            Assert.AreEqual(strName, "ATARI");

            int[] rgActions = gym.Actions;
            Assert.AreEqual(rgActions.Length, 2);
            Assert.AreEqual(rgActions[0], 0);
            Assert.AreEqual(rgActions[1], 1);

            gym.OpenUi();

            for (int i = 0; i < 100; i++)
            {
                if (i == 0 || gym.IsTerminal)
                    gym.Reset();

                int nActionIdx = random.Next(rgActions.Length);
                gym.Step(rgActions[nActionIdx], 1);

                Assert.AreNotEqual(gym.Data, null);
            }
           
            gym.CloseUi();
        }
    }
}
