using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe;
using System.Drawing;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestDatasetFactory
    {
        [TestMethod]
        public void TestIndexQuery()
        {
            PreTest.Init();
            Log log = new Log("Test Dataset Factory");
            log.EnableTrace = true;

            string strDs = "MNIST";
            DatasetFactory factory = new DatasetFactory();
            Stopwatch sw = new Stopwatch();

            try
            {
                DatasetDescriptor ds = factory.LoadDataset(strDs);
                factory.Open(ds.TrainingSource.ID);

                sw.Start();
                List<DbItem> rgItems = factory.LoadImageIndexes(false);
                sw.Stop();

                log.CHECK_EQ(rgItems.Count, ds.TrainingSource.ImageCount, "The query count should match the image count!");
                factory.Close();

                log.WriteLine("Query time = " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");

                sw.Restart();

                int nMin = int.MaxValue;
                int nMax = -int.MaxValue;
                for (int i = 0; i < rgItems.Count; i++)
                {
                    nMin = Math.Min(rgItems[i].Label, nMin);
                    nMax = Math.Max(rgItems[i].Label, nMax);
                }

                List<DbItem> rgBoosted = rgItems.Where(p => p.Boost > 0).ToList();

                for (int nLabel = nMin; nLabel <= nMax; nLabel++)
                {
                    List<DbItem> rgLabel = rgItems.Where(p => p.Label == nLabel).ToList();
                }

                sw.Stop();

                log.WriteLine("Query time (profile) = " + sw.Elapsed.TotalMilliseconds.ToString("N5") + " ms.");
            }
            finally
            {
                factory.Dispose();
            }
        }
    }
}
