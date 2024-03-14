﻿using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;
using System.IO;
using MyCaffe.db.image;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using System.Drawing;
using System.Net;
using MyCaffe.converter.pytorch;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPersistPytorch
    {
        [TestMethod]
        public void TestConvertLeNettoPytorch()
        {
            PersistPytorchTest test = new PersistPytorchTest();

            try
            {
                foreach (IPersistPytorchTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestConvertLeNettoPytorch();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConvertTFTtoPytorch()
        {
            PersistPytorchTest test = new PersistPytorchTest();

            try
            {
                foreach (IPersistPytorchTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestConvertTFTtoPytorch();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IPersistPytorchTest : ITest
    {
        void TestConvertLeNettoPytorch();
        void TestConvertTFTtoPytorch();
    }

    class PersistPytorchTest : TestBase
    {
        public PersistPytorchTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Test Persist Pytorch", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PersistPytorchTest<double>(strName, nDeviceID, engine);
            else
                return new PersistPytorchTest<float>(strName, nDeviceID, engine);
        }
    }

    class PersistPytorchTest<T> : TestEx<T>, IPersistPytorchTest
    {
        public PersistPytorchTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private string loadTextFile(string strFile)
        {
            using (StreamReader sr = new StreamReader(strFile))
            {
                return sr.ReadToEnd();
            }
        }

        private byte[] loadBinaryFile(string strFile)
        {
            using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
            {
                return br.ReadBytes((int)fs.Length);
            }
        }

        private void saveTextToFile(string str, string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (StreamWriter sw = new StreamWriter(strFile))
            {
                sw.WriteLine(str);
            }
        }

        private void saveBinaryToFile(byte[] rg, string strFile)
        {
            if (File.Exists(strFile))
                File.Delete(strFile);

            using (FileStream fs = new FileStream(strFile, FileMode.CreateNew, FileAccess.Write))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                bw.Write(rg);
            }
        }

        public void TestConvertLeNettoPytorch()
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\Pytorch";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\mnist";
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            SettingsCaffe s = new SettingsCaffe();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            try
            {
                ProjectEx prj = new ProjectEx("LeNet");
                prj.SolverDescription = loadTextFile(strModelPath + "\\lenet_solver.prototxt");
                prj.ModelDescription = loadTextFile(strModelPath + "\\lenet_train_test.prototxt");
                prj.WeightsState = loadBinaryFile(strModelPath + "\\my_weights.mycaffemodel");

                DatasetFactory factory = new DatasetFactory();
                prj.SetDataset(factory.LoadDataset("MNIST"));

                mycaffe.Load(Phase.TRAIN, prj);

                string strPytorchModelFile = strTestPath + "\\lenet_from_mycaffe_model.py";
                string strPytorchSolverFile = strTestPath + "\\lenet_from_mycaffe_solver.py";

                if (File.Exists(strPytorchModelFile))
                    File.Delete(strPytorchModelFile);

                if (File.Exists(strPytorchSolverFile))
                    File.Delete(strPytorchSolverFile);

                MyCaffeModelData data = new MyCaffeModelData(prj.ModelDescription, null, null, null, prj.SolverDescription, new List<int>() {  100, 3, 28, 28 });
                convert.ConvertMyCaffeToPyTorch(mycaffe.Cuda, mycaffe.Log, data, strPytorchModelFile, strPytorchSolverFile, true);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                mycaffe.Dispose();
                convert.Dispose();
            }
        }

        public void TestConvertTFTtoPytorch()
        {
            string strTestPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\Pytorch";
            if (!Directory.Exists(strTestPath))
                Directory.CreateDirectory(strTestPath);

            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\tft\\TFT_electricity";
            MyCaffeConversionControl<T> convert = new MyCaffeConversionControl<T>();

            SettingsCaffe s = new SettingsCaffe();
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            try
            {
                ProjectEx prj = new ProjectEx("TFT");
                prj.SolverDescription = loadTextFile(strModelPath + "\\tft_electricity_solver.prototxt");
                prj.ModelDescription = loadTextFile(strModelPath + "\\tft_electricity_train_test.prototxt");

                DatasetFactory factory = new DatasetFactory();
                prj.SetDataset(factory.LoadDataset("TFT.electricity"));

                mycaffe.Load(Phase.TRAIN, prj);

                string strPytorchModelFile = strTestPath + "\\tft_electricity_from_mycaffe_model.py";
                string strPytorchSolverFile = strTestPath + "\\tft_electricity_from_mycaffe_solver.py";

                if (File.Exists(strPytorchModelFile))
                    File.Delete(strPytorchModelFile);

                if (File.Exists(strPytorchSolverFile))
                    File.Delete(strPytorchSolverFile);

                MyCaffeModelData data = new MyCaffeModelData(prj.ModelDescription, null, null, null, prj.SolverDescription, new List<int>() { 100, 3, 28, 28 });
                convert.ConvertMyCaffeToPyTorch(mycaffe.Cuda, mycaffe.Log, data, strPytorchModelFile, strPytorchSolverFile, true);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                mycaffe.Dispose();
                convert.Dispose();
            }
        }
    }
}
