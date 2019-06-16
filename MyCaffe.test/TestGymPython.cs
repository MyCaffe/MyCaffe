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
using System.Diagnostics;

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

            gym.Initialize("ATARI", "Preprocess=False;AllowNegativeRewards=True;TerminateOnRallyEnd=True;FrameSkip=4;GameROM=C:\\ProgramData\\MyCaffe\\test_data\\roms\\breakout.bin");

            string strName = gym.Name;
            Assert.AreEqual(strName, "MyCaffe ATARI");

            int[] rgActions = gym.Actions;
            Assert.AreEqual(rgActions.Length, 3);
            Assert.AreEqual(rgActions[0], 0);
            Assert.AreEqual(rgActions[1], 1);
            Assert.AreEqual(rgActions[2], 2);

            gym.OpenUi();

            for (int i = 0; i < 100; i++)
            {
                if (i == 0 || gym.IsTerminal)
                    gym.Reset();

                int nActionIdx = random.Next(rgActions.Length);
                gym.Step(rgActions[nActionIdx], 1);

                Assert.AreNotEqual(gym.Data, null);

                List<List<List<double>>> rgrgrgData = gym.GetDataAsImage();

                Assert.AreNotEqual(rgrgrgData, null);
                Assert.AreEqual(rgrgrgData.Count, 80);
                Assert.AreEqual(rgrgrgData[0].Count, 80);
                Assert.AreEqual(rgrgrgData[0][0].Count, 3);

                rgrgrgData = gym.GetDataAsImage(true);

                Assert.AreNotEqual(rgrgrgData, null);
                Assert.AreEqual(rgrgrgData.Count, 80);
                Assert.AreEqual(rgrgrgData[0].Count, 80);
                Assert.AreEqual(rgrgrgData[0][0].Count, 1);

                Trace.WriteLine("Reward = " + gym.Reward);

                if (gym.IsTerminal)
                    Trace.WriteLine("TERMINAL = TRUE");
            }

            gym.CloseUi();
        }

        [TestMethod]
        public void TestGymGray()
        {
            MyCaffePythonGym gym = new MyCaffePythonGym();
            Random random = new Random();

            gym.Initialize("ATARI", "Preprocess=False;ForceGray=True;AllowNegativeRewards=True;TerminateOnRallyEnd=True;FrameSkip=4;GameROM=C:\\ProgramData\\MyCaffe\\test_data\\roms\\breakout.bin");

            string strName = gym.Name;
            Assert.AreEqual(strName, "MyCaffe ATARI");

            int[] rgActions = gym.Actions;
            Assert.AreEqual(rgActions.Length, 3);
            Assert.AreEqual(rgActions[0], 0);
            Assert.AreEqual(rgActions[1], 1);
            Assert.AreEqual(rgActions[2], 2);

            gym.OpenUi();

            for (int i = 0; i < 100; i++)
            {
                if (i == 0 || gym.IsTerminal)
                    gym.Reset();

                int nActionIdx = random.Next(rgActions.Length);
                gym.Step(rgActions[nActionIdx], 1);

                Assert.AreNotEqual(gym.Data, null);

                List<List<List<double>>> rgrgrgData = gym.GetDataAsImage();

                Assert.AreNotEqual(rgrgrgData, null);
                Assert.AreEqual(rgrgrgData.Count, 80);
                Assert.AreEqual(rgrgrgData[0].Count, 80);
                Assert.AreEqual(rgrgrgData[0][0].Count, 1);

                Trace.WriteLine("Reward = " + gym.Reward);

                if (gym.IsTerminal)
                    Trace.WriteLine("TERMINAL = TRUE");
            }

            gym.CloseUi();
        }
    }
}
