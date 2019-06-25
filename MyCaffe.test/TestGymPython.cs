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
        public void TestGymCartPole()
        {
            MyCaffePythonGym gym = new MyCaffePythonGym();
            Random random = new Random();

            gym.Initialize("Cart-Pole", "");

            string strName = gym.Name;
            Assert.AreEqual(strName, "MyCaffe Cart-Pole");

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

                List<double> rgData = gym.Data;

                Assert.AreNotEqual(rgData, null);
                Assert.AreEqual(rgData.Count, 4);

                Trace.WriteLine("Reward = " + gym.Reward);

                if (gym.IsTerminal)
                    Trace.WriteLine("TERMINAL = TRUE");
            }

            gym.CloseUi();
        }

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

                List<List<List<double>>> rgrgrgData = gym.GetDataAs3D();

                Assert.AreNotEqual(rgrgrgData, null);
                Assert.AreEqual(rgrgrgData.Count, 80);
                Assert.AreEqual(rgrgrgData[0].Count, 80);
                Assert.AreEqual(rgrgrgData[0][0].Count, 3);

                rgrgrgData = gym.GetDataAs3D(true);

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

                List<List<List<double>>> rgrgrgData = gym.GetDataAs3D();

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
        public void TestGymGrayStacked()
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

            List<double[]> rgrgActions = new List<double[]>();
            double[] rgProb = new double[51];

            for (int i = 0; i < 51; i++)
            {
                rgProb[i] = 1.0 / Math.PI * Math.Exp(-(i - 25) * (i - 25) / (2 * 1));
            }

            rgrgActions.Add(rgProb);
            rgrgActions.Add(rgProb);
            rgrgActions.Add(rgProb);

            for (int i = 0; i < 100; i++)
            {
                if (i == 0 || gym.IsTerminal)
                    gym.Reset();

                int nActionIdx = random.Next(rgActions.Length);
                gym.Step(rgActions[nActionIdx], 1);

                Assert.AreNotEqual(gym.Data, null);

                List<List<List<double>>> rgrgrgData = gym.GetDataAsStacked3D((i == 0) ? true : false, 4, 4, true, 1.0);

                Assert.AreNotEqual(rgrgrgData, null);
                Assert.AreEqual(rgrgrgData.Count, 80);
                Assert.AreEqual(rgrgrgData[0].Count, 80);
                Assert.AreEqual(rgrgrgData[0][0].Count, 4);

                Trace.WriteLine("Reward = " + gym.Reward);

                if (gym.IsTerminal)
                    Trace.WriteLine("TERMINAL = TRUE");

                for (int j = 0; j < rgrgActions.Count; j++)
                {
                    for (int k = 0; k < rgrgActions[j].Length; k++)
                    {
                        rgrgActions[j][k] += 0.01 * random.NextDouble();
                    }
                }

                gym.SetActionDistributions(rgrgActions.ToArray());
            }

            gym.CloseUi();
        }
    }
}
