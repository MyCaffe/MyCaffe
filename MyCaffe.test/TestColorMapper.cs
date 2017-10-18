using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.imagedb;
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
            Log log = new Log("Test ColorMapper");
            log.EnableTrace = true;

            ColorMapper clrMap = new ColorMapper(0, 1, Color.Black, Color.Red);
            double dfVal = 0;
            double dfInc = 1.0 / 10.0;

            for (int i = 0; i < 10; i++)
            {
                Color clr = clrMap.GetColor(dfVal);
                double dfVal1 = clrMap.GetValue(clr);

                dfVal1 = Math.Round(dfVal1, 1);

                log.EXPECT_EQUAL<float>(dfVal1, dfVal, "The value at i = " + i.ToString() + " value = " + dfVal.ToString() + " does not equal value1 = " + dfVal1.ToString());

                dfVal += dfInc;
            }

            log.WriteLine("DONE.");
        }
    }
}
