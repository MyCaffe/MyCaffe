using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.data;
using MyCaffe.param.ssd;

namespace MyCaffe.test
{
    [TestClass]
    public class TestAnnotatedDataLayer
    {
        [TestMethod]
        public void TestRead()
        {
            AnnotatedDataLayerTest test = new AnnotatedDataLayerTest();

            try
            {
                foreach (IAnnotatedDataLayerTest t in test.Tests)
                {
                    t.TestRead();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IAnnotatedDataLayerTest : ITest
    {
        void TestRead();
    }

    class AnnotatedDataLayerTest : TestBase
    {
        public AnnotatedDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Annotated Data Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new AnnotatedDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new AnnotatedDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class AnnotatedDataLayerTest<T> : TestEx<T>, IAnnotatedDataLayerTest
    {
        public AnnotatedDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestRead()
        {
        }
    }
}
