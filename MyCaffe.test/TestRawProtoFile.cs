using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Diagnostics;
using MyCaffe.basecode;

namespace MyCaffe.test
{
    [TestClass]
    public class TestRawProtoFile
    {
        [TestMethod]
        public void TestLoadingFiles()
        {
            string strPath = TestBase.GetTestPath("\\MyCaffe\\test_data\\models");
            string[] rgstrDir = Directory.GetDirectories(strPath);

            foreach (string strDir in rgstrDir)
            {
                string[] rgstrFiles = Directory.GetFiles(strDir);

                foreach (string strFile in rgstrFiles)
                {
                    FileInfo fi = new FileInfo(strFile);

                    if (fi.Extension == ".prototxt")
                    {
                        Trace.WriteLine("Loading '" + fi.FullName + "...");

                        RawProto proto1 = RawProtoFile.LoadFromFile(fi.FullName);
                        string strProto1 = proto1.ToString();

                        RawProto proto2 = RawProto.Parse(strProto1);
                        string strProto2 = proto2.ToString();

                        Assert.AreEqual(strProto1, strProto2);
                    }
                }
            }
        }
    }
}
