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
    public class TestColorMapper
    {
        [TestMethod]
        public void TestColorMapping()
        {
            PreTest.Init();

            Log log = new Log("Test ColorMapper");
            log.EnableTrace = true;

            int nCount = 300;
            ColorMapper clrMap = new ColorMapper(0, 1, Color.Black, Color.Red);
            double dfVal = 0;
            double dfInc = 1.0 / (double)nCount;

            for (int i = 0; i < nCount; i++)
            {
                Color clr = clrMap.GetColor(dfVal);
                double dfVal1 = clrMap.GetValue(clr);

                log.EXPECT_EQUAL<float>(dfVal1, dfVal, "The value at i = " + i.ToString() + " value = " + dfVal.ToString() + " does not equal value1 = " + dfVal1.ToString());

                dfVal += dfInc;
            }

            dfVal = clrMap.GetValue(Color.FromArgb(0, 130, 124));
            log.CHECK_NE(0, dfVal, "The value should not be zero.");

            dfVal = clrMap.GetValue(Color.FromArgb(255, 252, 0));
            log.CHECK_NE(0, dfVal, "The value should not be zero.");

            log.WriteLine("DONE.");
        }
    }
}
