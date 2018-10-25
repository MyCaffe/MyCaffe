using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.fillers;
using MyCaffe.basecode;
using MyCaffe.db.image;

namespace MyCaffe.test
{
    [TestClass]
    public class TestNeuronLayer
    {
        [TestMethod]
        public void TestAbsVal()
        {
            AbsValLayerTest test = new AbsValLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAbsValGradient()
        {
            AbsValLayerTest test = new AbsValLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSigmoid()
        {
            SigmoidLayerTest test = new SigmoidLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSigmoidGradient()
        {
            SigmoidLayerTest test = new SigmoidLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSigmoidCuDnn()
        {
            SigmoidLayerTest test = new SigmoidLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSigmoidGradientCuDnn()
        {
            SigmoidLayerTest test = new SigmoidLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestClip()
        {
            ClipLayerTest test = new ClipLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestClipGradient()
        {
            ClipLayerTest test = new ClipLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSwish()
        {
            SwishLayerTest test = new SwishLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSwishGradient()
        {
            SwishLayerTest test = new SwishLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSwishWithBeta()
        {
            SwishLayerWithBetaTest test = new SwishLayerWithBetaTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSwishWithBetaGradient()
        {
            SwishLayerWithBetaTest test = new SwishLayerWithBetaTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSwishAsLinear()
        {
            SwishLayerAsLinearTest test = new SwishLayerAsLinearTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSwishAsLinearGradient()
        {
            SwishLayerAsLinearTest test = new SwishLayerAsLinearTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestTanh()
        {
            TanhLayerTest test = new TanhLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTanhGradient()
        {
            TanhLayerTest test = new TanhLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTanhCuDnn()
        {
            TanhLayerTest test = new TanhLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTanhGradientCuDnn()
        {
            TanhLayerTest test = new TanhLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRelu()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluGradient()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluCuDnn()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluGradientCuDnn()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluWithNegativeSlope()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IReluLayerTest t in test.Tests)
                {
                    t.TestForwardWithNegativeSlope();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluGradientWithNegativeSlope()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IReluLayerTest t in test.Tests)
                {
                    t.TestGradientWithNegativeSlope();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluWithNegativeSlopeCuDnn()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IReluLayerTest t in test.Tests)
                {
                    t.TestForwardWithNegativeSlope();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReluGradientWithNegativeSlopeCuDnn()
        {
            ReluLayerTest test = new ReluLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IReluLayerTest t in test.Tests)
                {
                    t.TestGradientWithNegativeSlope();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestELU()
        {
            ELULayerTest test = new ELULayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IELULayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestELUasReLU()
        {
            ELULayerTest test = new ELULayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IELULayerTest t in test.Tests)
                {
                    t.TestForwardAsReLU();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestELUGradient()
        {
            ELULayerTest test = new ELULayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IELULayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestELUasReLUGradient()
        {
            ELULayerTest test = new ELULayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IELULayerTest t in test.Tests)
                {
                    t.TestGradientAsReLU();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestELUCuDnn()
        {
            ELULayerTest test = new ELULayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IELULayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestELUGradientCuDnn()
        {
            ELULayerTest test = new ELULayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IELULayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExp()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpGradient()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpBase2()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 1, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpGradientBase2()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 1, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpBase2Shift1()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 1, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpGradientBase2Shift1()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 1, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpBase2Scale3()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 3, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpGradientBase2Scale3()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 3, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpBase2Shift1Scale3()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestExpGradientBase2Shift1Scale3()
        {
            ExpLayerTest test = new ExpLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLabelMapping()
        {
            LabelMappingLayerTest test = new LabelMappingLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLog()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogGradient()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogBase2()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 1, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogGradientBase2()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 1, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogBase2Shift1()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 1, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogGradientBase2Shift1()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 1, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogBase2Scale3()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 3, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogGradientBase2Scale3()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 3, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogBase2Shift1Scale3()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestForward(2, 3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestLogGradientBase2Shift1Scale3()
        {
            LogLayerTest test = new LogLayerTest();

            try
            {
                foreach (IExpLogLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropout()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestDropoutGradient()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropoutThreeQuarters()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDropoutLayerTest t in test.Tests)
                {
                    t.TestDropoutThreeQuarters();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropoutTestPhase()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDropoutLayerTest t in test.Tests)
                {
                    t.TestDropoutTestPhase();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropoutGradientTest()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDropoutLayerTest t in test.Tests)
                {
                    t.TestDropoutGradientTest();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestDropoutCuDnn()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestDropoutGradientCuDnn()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropoutThreeQuartersCuDnn()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IDropoutLayerTest t in test.Tests)
                {
                    t.TestDropoutThreeQuarters();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropoutTestPhaseCuDnn()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IDropoutLayerTest t in test.Tests)
                {
                    t.TestDropoutTestPhase();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDropoutGradientTestCuDnn()
        {
            DropoutLayerTest test = new DropoutLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IDropoutLayerTest t in test.Tests)
                {
                    t.TestDropoutGradientTest();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBNLL()
        {
            BNLLLayerTest test = new BNLLLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBNLLGradient()
        {
            BNLLLayerTest test = new BNLLLayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPReLU()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPReLUGradient()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPReLUParam()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (IPReLULayerTest t in test.Tests)
                {
                    t.TestPReLUParam();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardChannelShared()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (IPReLULayerTest t in test.Tests)
                {
                    t.TestForwardChannelShared();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientChannelShared()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (IPReLULayerTest t in test.Tests)
                {
                    t.TestGradientChannelShared();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestConsistencyReLU()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (IPReLULayerTest t in test.Tests)
                {
                    t.TestConsistencyReLU();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPReLUInPlace()
        {
            PReLULayerTest test = new PReLULayerTest();

            try
            {
                foreach (IPReLULayerTest t in test.Tests)
                {
                    t.TestPReLUInPlace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientScaleSetup()
        {
            GradientScaleLayerTest test = new GradientScaleLayerTest();

            try
            {
                foreach (IGradientScaleLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientScale()
        {
            GradientScaleLayerTest test = new GradientScaleLayerTest();

            try
            {
                foreach (IGradientScaleLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientScaleGradient()
        {
            GradientScaleLayerTest test = new GradientScaleLayerTest();

            try
            {
                foreach (IGradientScaleLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface INeuronLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class SigmoidLayerTest : TestBase
    {
        public SigmoidLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Sigmoid Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SigmoidLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SigmoidLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SigmoidLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public SigmoidLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOID);
            SigmoidLayer<T> layer = new SigmoidLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfBottom = rgBottom[i];
                double dfExpected = 1.0 / (1.0 + Math.Exp(-dfBottom));

                m_log.EXPECT_NEAR(dfTop, dfExpected, (m_dt == common.DataType.DOUBLE) ? 1e-15 : 1e-7);

                // check that we squashed the value between 0 and 1.
                m_log.CHECK(dfTop >= 0.0, "The top value at " + i.ToString() + " should be greater than or equal to 0.");
                m_log.CHECK(dfTop <= 1.0, "The top value at " + i.ToString() + " should be less than or equal to 1.");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOID);
            SigmoidLayer<T> layer = new SigmoidLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    class SwishLayerTest : TestBase
    {
        public SwishLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Swish Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SwishLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SwishLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SwishLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public SwishLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SWISH);
            SwishLayer<T> layer = new SwishLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfBottom = rgBottom[i];
                double dfExpected = dfBottom / (1.0 + Math.Exp(-dfBottom));

                m_log.EXPECT_EQUAL<float>(dfTop, dfExpected);
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SWISH);
            SwishLayer<T> layer = new SwishLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    class SwishLayerWithBetaTest : TestBase
    {
        public SwishLayerWithBetaTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Swish Layer With Beta Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SwishLayerWithBetaTest<double>(strName, nDeviceID, engine);
            else
                return new SwishLayerWithBetaTest<float>(strName, nDeviceID, engine);
        }
    }

    class SwishLayerWithBetaTest<T> : TestEx<T>, INeuronLayerTest
    {
        public SwishLayerWithBetaTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            RawProto rp = RawProto.Parse("type: \"Swish\" swish_param { beta: 1.5 }");
            LayerParameter p = LayerParameter.FromProto(rp);
            SwishLayer<T> layer = new SwishLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfBottom = rgBottom[i];
                double dfExpected = dfBottom / (1.0 + Math.Exp(-1.5 * dfBottom));

                m_log.EXPECT_EQUAL<float>(dfTop, dfExpected);
            }
        }

        public void TestGradient()
        {
            RawProto rp = RawProto.Parse("type: \"Swish\" swish_param { beta: 1.5 }");
            LayerParameter p = LayerParameter.FromProto(rp);
            SwishLayer<T> layer = new SwishLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    class SwishLayerAsLinearTest : TestBase
    {
        public SwishLayerAsLinearTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Swish Layer As Linear Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SwishLayerAsLinearTest<double>(strName, nDeviceID, engine);
            else
                return new SwishLayerAsLinearTest<float>(strName, nDeviceID, engine);
        }
    }

    class SwishLayerAsLinearTest<T> : TestEx<T>, INeuronLayerTest
    {
        public SwishLayerAsLinearTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            RawProto rp = RawProto.Parse("type: \"Swish\" swish_param { beta: 0.0 }");
            LayerParameter p = LayerParameter.FromProto(rp);
            SwishLayer<T> layer = new SwishLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfBottom = rgBottom[i];
                double dfExpected = dfBottom / 2.0;

                m_log.EXPECT_EQUAL<float>(dfTop, dfExpected);
            }
        }

        public void TestGradient()
        {
            RawProto rp = RawProto.Parse("type: \"Swish\" swish_param { beta: 0.0 }");
            LayerParameter p = LayerParameter.FromProto(rp);
            SwishLayer<T> layer = new SwishLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    class ClipLayerTest : TestBase
    {
        public ClipLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Clip Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ClipLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ClipLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ClipLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public ClipLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CLIP);
            p.clip_param.min = -1;
            p.clip_param.max = 2;
            ClipLayer<T> layer = new ClipLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                m_log.CHECK_GE(rgTopData[i], -1, "The top data should be >= -1");
                m_log.CHECK_LE(rgTopData[i], 2, "The top data should be <= 2");
                m_log.CHECK(rgBottomData[i] > -1 || rgTopData[i] == -1, "The data is incorrect.");
                m_log.CHECK(rgBottomData[i] < 2 || rgTopData[i] == 2, "The data is incorrect.");
                m_log.CHECK(!(rgBottomData[i] >= -1 && rgBottomData[i] <= 2) || rgTopData[i] == rgBottomData[i], "The data is incorrect.");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CLIP);
            p.clip_param.min = -1;
            p.clip_param.max = 2;
            ClipLayer<T> layer = new ClipLayer<T>(m_cuda, m_log, p);

            // Unfortunately, it might happen that an input value lands exactly within
            // the discontinuity region of the Clip function.  In this case the numeric
            // gradient is likely to differ significantly (i.e. by a value larger than
            // checker tolerance) from the computed gradient.  To handle such cases, we
            // eliminate such values from the input blob before the gradient check.
            double dfEpsilon = 1e-2;
            double dfMinRangeStart = p.clip_param.min - dfEpsilon;
            double dfMinRangeEnd = p.clip_param.min + dfEpsilon;
            double dfMaxRangeStart = p.clip_param.max - dfEpsilon;
            double dfMaxRangeEnd = p.clip_param.max + dfEpsilon;

            // The input blob is owned by the TestBase object, so we begin with
            // creating a temporary blob and copying the input data there.
            Blob<T> temp_bottom = new Blob<T>(m_cuda, m_log);
            temp_bottom.CopyFrom(Bottom, false, true);
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTempData = convert(temp_bottom.mutable_cpu_data);

            for (int i = 0; i < Bottom.count(); i++)
            {
                if (rgBottomData[i] >= dfMinRangeStart && rgBottomData[i] <= dfMinRangeEnd)
                {
                    rgTempData[i] = rgBottomData[i] - dfEpsilon;
                }
                else if (rgBottomData[i] >= dfMaxRangeStart && rgBottomData[i] <= dfMaxRangeEnd)
                {
                    rgTempData[i] = rgBottomData[i] + dfEpsilon;
                }
            }
            temp_bottom.mutable_cpu_data = convert(rgTempData);

            BlobCollection<T> TempBottomVec = new BlobCollection<T>();
            TempBottomVec.Add(temp_bottom);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, dfEpsilon, 1e-3);
            checker.CheckGradientEltwise(layer, TempBottomVec, TopVec);
        }
    }


    class TanhLayerTest : TestBase
    {
        public TanhLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Tanh Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TanhLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TanhLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TanhLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public TanhLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TANH);
            TanhLayer<T> layer = new TanhLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test exact values
            for (int i = 0; i < Bottom.num; i++)
            {
                for (int j = 0; j < Bottom.channels; j++)
                {
                    for (int k = 0; k < Bottom.height; k++)
                    {
                        for (int l = 0; l < Bottom.width; l++)
                        {
                            double dfTop = convert(Top.data_at(i, j, k, l));
                            double dfBottom = convert(Bottom.data_at(i, j, k, l));
                            double dfVal = Math.Exp(2 * dfBottom);
                            double dfExpected = (dfVal - 1) / (dfVal + 1);

                            m_log.CHECK(dfTop + 1e-4 >= dfExpected, "The top value should be greater than or equal to " + dfExpected.ToString());
                            m_log.CHECK(dfTop - 1e-4 <= dfExpected, "The top value should be less than or equal to " + dfExpected.ToString());
                        }
                    }
                }
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TANH);
            TanhLayer<T> layer = new TanhLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }


    class ReluLayerTest : TestBase
    {
        public ReluLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Relu Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ReluLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ReluLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IReluLayerTest : INeuronLayerTest
    {
        void TestForwardWithNegativeSlope();
        void TestGradientWithNegativeSlope();
    }

    class ReluLayerTest<T> : TestEx<T>, IReluLayerTest
    {
        public ReluLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RELU);
            p.relu_param.engine = m_engine;
            ReLULayer<T> layer = new ReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTopVal = rgTopData[i];
                double dfBottomVal = rgBottomData[i];

                m_log.CHECK_GE(dfTopVal, 0.0, "The top value at " + i.ToString() + " should be greater than or equal to 0.");
                m_log.CHECK(dfTopVal == 0 || dfTopVal == dfBottomVal, "The top value should either be equal to zero or to the bottom value at " + i.ToString());
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RELU);
            p.relu_param.engine = m_engine;
            ReLULayer<T> layer = new ReLULayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestForwardWithNegativeSlope()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RELU);
            p.relu_param.engine = m_engine;
            p.relu_param.negative_slope = 0.01;
            ReLULayer<T> layer = new ReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTopVal = rgTopData[i];
                double dfBottomVal = rgBottomData[i];

                if (dfTopVal >= 0)
                    m_log.CHECK_EQ(dfTopVal, dfBottomVal, "Top and bottom should be equal at index " + i.ToString() + ".");
                else
                    m_log.CHECK_EQ(dfTopVal, dfBottomVal * 0.01, "Top value should equal Bottom value * 0.01, or " + (dfBottomVal * 0.01).ToString());
            }
        }

        public void TestGradientWithNegativeSlope()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RELU);
            p.relu_param.engine = m_engine;
            p.relu_param.negative_slope = 0.01;
            ReLULayer<T> layer = new ReLULayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    class ELULayerTest : TestBase
    {
        public ELULayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ELU Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ELULayerTest<double>(strName, nDeviceID, engine);
            else
                return new ELULayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IELULayerTest : INeuronLayerTest
    {
        void TestForwardAsReLU();
        void TestGradientAsReLU();
    }

    class ELULayerTest<T> : TestEx<T>, IELULayerTest
    {
        public ELULayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            RawProto proto = RawProto.Parse("name: \"elu\" type: \"ELU\" elu_param { alpha: 0.5 }");
            LayerParameter p = LayerParameter.FromProto(proto);
            ELULayer<T> layer = new ELULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double kDelta = 2e-4;
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTopVal = rgTopData[i];
                double dfBottomVal = rgBottomData[i];

                if (dfBottomVal > 0)
                    m_log.EXPECT_EQUAL<float>(dfTopVal, dfBottomVal);
                else
                    m_log.EXPECT_NEAR(dfTopVal, 0.5 * (Math.Exp(dfBottomVal) - 1), kDelta);
            }
        }

        public void TestForwardAsReLU()
        {
            RawProto proto = RawProto.Parse("name: \"elu\" type: \"ELU\" elu_param { alpha: 0.0 }");
            LayerParameter p = LayerParameter.FromProto(proto);
            ELULayer<T> layer = new ELULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTopVal = rgTopData[i];
                double dfBottomVal = rgBottomData[i];

                m_log.CHECK_GE(dfTopVal, 0.0, "The top value should be >= 0.0");
                m_log.CHECK(dfTopVal == 0 || dfTopVal == dfBottomVal, "The top should be 0 or equal to the bottom val");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELU);
            ELULayer<T> layer = new ELULayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestGradientAsReLU()
        {
            RawProto proto = RawProto.Parse("name: \"elu\" type: \"ELU\" elu_param { alpha: 0.0 }");
            LayerParameter p = LayerParameter.FromProto(proto);
            ELULayer<T> layer = new ELULayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    class AbsValLayerTest : TestBase
    {
        public AbsValLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("AbsVal Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new AbsValLayerTest<double>(strName, nDeviceID, engine);
            else
                return new AbsValLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class AbsValLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public AbsValLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ABSVAL);
            AbsValLayer<T> layer = new AbsValLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTopVal = rgTopData[i];
                double dfBottomVal = rgBottomData[i];

                m_log.CHECK_EQ(dfTopVal, Math.Abs(dfBottomVal), "The top value does not equal the abs val of the bottom value at " + i.ToString());
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ABSVAL);
            AbsValLayer<T> layer = new AbsValLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }

    interface IExpLogLayerTest : INeuronLayerTest
    {
        void TestForward(double dfBase, double dfScale, double dfShift);
        void TestGradient(double dfBase, double dfScale, double dfShift);
    }

    class ExpLayerTest : TestBase
    {
        public ExpLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Exp Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ExpLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ExpLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ExpLayerTest<T> : TestEx<T>, IExpLogLayerTest
    {
        public ExpLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward(double dfBase, double dfScale, double dfShift)
        {
            LayerParameter layer_param = new LayerParameter(LayerParameter.LayerType.EXP);
            layer_param.exp_param.base_val = dfBase;
            layer_param.exp_param.scale = dfScale;
            layer_param.exp_param.shift = dfShift;
            ExpLayer<T> layer = new ExpLayer<T>(m_cuda, m_log, layer_param);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfDelta = 2e-4;
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfBottomVal = rgBottomData[i];
                double dfTopVal = rgTopData[i];

                if (dfBase == -1)
                    m_log.EXPECT_NEAR(dfTopVal, Math.Exp(dfShift + dfScale * dfBottomVal), dfDelta);
                else
                    m_log.EXPECT_NEAR(dfTopVal, Math.Pow(dfBase, dfShift + dfScale * dfBottomVal), dfDelta);
            }
        }

        public void TestGradient(double dfBase, double dfScale, double dfShift)
        {
            LayerParameter layer_param = new LayerParameter(LayerParameter.LayerType.EXP);
            layer_param.exp_param.base_val = dfBase;
            layer_param.exp_param.scale = dfScale;
            layer_param.exp_param.shift = dfShift;
            ExpLayer<T> layer = new ExpLayer<T>(m_cuda, m_log, layer_param);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestForward()
        {
            // Test default base of '-1' -- should actually set base := e
            TestForward(-1, 1, 0);
        }

        public void TestGradient()
        {
            // Test default base of '-1' -- should actually set base := e
            TestGradient(-1, 1, 0);
        }
    }

    class LabelMappingLayerTest : TestBase
    {
        public LabelMappingLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Label Mapping Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LabelMappingLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LabelMappingLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LabelMappingLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public LabelMappingLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;

            double[] rgData = new double[] { 0, 1, 2 };
            Bottom.mutable_cpu_data = convert(rgData);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter layer_param = new LayerParameter(LayerParameter.LayerType.LABELMAPPING);
            layer_param.labelmapping_param.mapping.Add(new LabelMapping(0, 10, null, null));
            layer_param.labelmapping_param.mapping.Add(new LabelMapping(2, 30, null, null));
            LabelMappingLayer<T> layer = new LabelMappingLayer<T>(m_cuda, m_log, layer_param, null);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(Top.update_cpu_data());

            m_log.CHECK_EQ(10, (int)rgTopData[0], "The top[0] item should be 10.");
            m_log.CHECK_EQ(1, (int)rgTopData[1], "The top[1] item should be 1.");
            m_log.CHECK_EQ(30, (int)rgTopData[2], "The top[2] item should be 30.");
        }

        public void TestGradient()
        {
        }
    }

    class LogLayerTest : TestBase
    {
        public LogLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Log Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LogLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LogLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LogLayerTest<T> : TestEx<T>, IExpLogLayerTest
    {
        public LogLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void LogBottomInit()
        {
            FillerParameter filler_param = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);

            filler.Fill(Bottom);
            m_cuda.exp(Bottom.count(), Bottom.gpu_data, Bottom.mutable_gpu_data);
        }

        public void TestForward(double dfBase, double dfScale, double dfShift)
        {
            LogBottomInit();
            LayerParameter layer_param = new LayerParameter(LayerParameter.LayerType.LOG);
            layer_param.log_param.base_val = dfBase;
            layer_param.log_param.scale = dfScale;
            layer_param.log_param.shift = dfShift;
            LogLayer<T> layer = new LogLayer<T>(m_cuda, m_log, layer_param);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfDelta = 2e-4;
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfBottomVal = rgBottomData[i];
                double dfTopVal = rgTopData[i];

                if (dfBase == -1)
                    m_log.EXPECT_NEAR(dfTopVal, Math.Log(dfShift + dfScale * dfBottomVal), dfDelta);
                else
                    m_log.EXPECT_NEAR(dfTopVal, Math.Log(dfShift + dfScale * dfBottomVal) / Math.Log(dfBase), dfDelta);
            }
        }

        public void TestGradient(double dfBase, double dfScale, double dfShift)
        {
            LogBottomInit();
            LayerParameter layer_param = new LayerParameter(LayerParameter.LayerType.LOG);
            layer_param.log_param.base_val = dfBase;
            layer_param.log_param.scale = dfScale;
            layer_param.log_param.shift = dfShift;
            LogLayer<T> layer = new LogLayer<T>(m_cuda, m_log, layer_param);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 0.012);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestForward()
        {
            // Test default base of '-1' -- should actually set base := e
            TestForward(-1, 1, 0);
        }

        public void TestGradient()
        {
            // Test default base of '-1' -- should actually set base := e
            TestGradient(-1, 1, 0);
        }
    }


    class DropoutLayerTest : TestBase
    {
        public DropoutLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Dropout Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DropoutLayerTest<double>(strName, nDeviceID, engine);
            else
                return new DropoutLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IDropoutLayerTest : INeuronLayerTest
    {
        void TestDropoutThreeQuarters();
        void TestDropoutTestPhase();
        void TestDropoutGradientTest();
    }

    class DropoutLayerTest<T> : TestEx<T>, IDropoutLayerTest
    {
        public DropoutLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward(double dfDropoutRatio)
        {
            LayerParameter layer_param = new LayerParameter(LayerParameter.LayerType.DROPOUT);

            layer_param.dropout_param.engine = m_engine;

            // Fill in the given dropout_ratio, unless its 0.5, in which case we don't
            // set it explicitly to test that 0.5 is the default.
            if (dfDropoutRatio != 0.5)
                layer_param.dropout_param.dropout_ratio = dfDropoutRatio;

            layer_param.phase = Phase.TRAIN;
            DropoutLayer<T> layer = new DropoutLayer<T>(m_cuda, m_log, layer_param);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());
            double dfScale = 1.0 / (1.0 - layer_param.dropout_param.dropout_ratio);
            int nCount = Bottom.count();

            // Initialize num_kept to count the number of inputs NOT dropped out.
            int nNumKept = 0;

            for (int i = 0; i < nCount; i++)
            {
                if (rgTopData[i] != 0)
                {
                    nNumKept++;
                    m_log.CHECK_EQ(rgTopData[i], rgBottomData[i] * dfScale, "The top data must equal the bottom * dfScale");
                }
            }

            double dfStdError = Math.Sqrt(dfDropoutRatio * (1 - dfDropoutRatio) / nCount);

            // Fail if the number dropped was more than 1.96 * dfStdError away from the
            // expected number -- requires 95% confidence that the dropout layer is not
            // obeying the given dropout ratio for test failure.
            double dfEmpericalDroputRatio = 1 - nNumKept / (double)nCount;
          
            m_log.EXPECT_NEAR(dfEmpericalDroputRatio, dfDropoutRatio, (m_dt == common.DataType.DOUBLE) ? (1.96 * dfStdError) : ((1.96 * 1.2) * dfStdError));
        }

        public void TestForward()
        {
            TestForward(0.5);
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            p.dropout_param.engine = m_engine;
            p.phase = Phase.TRAIN;
            DropoutLayer<T> layer = new DropoutLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
#warning DropoutLayer.TestGradient test fails with ENGINE = cuDnn.
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestDropoutThreeQuarters()
        {
            TestForward(0.75);
        }

        public void TestDropoutTestPhase()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            p.dropout_param.engine = m_engine;
            p.phase = Phase.TEST;
            DropoutLayer<T> layer = new DropoutLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < Top.count(); i++)
            {
                double dfBottom = rgBottom[i];
                double dfTop = rgTop[i];

                if (dfTop != 0)
                    m_log.CHECK_EQ(dfTop, dfBottom, "If the values are not zero, they should be equal and they are not at " + i.ToString());
            }
        }

        public void TestDropoutGradientTest()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            p.dropout_param.engine = m_engine;
            p.phase = Phase.TEST;
            DropoutLayer<T> layer = new DropoutLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }


    class BNLLLayerTest : TestBase
    {
        public BNLLLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("BNLL Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BNLLLayerTest<double>(strName, nDeviceID, engine);
            else
                return new BNLLLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BNLLLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public BNLLLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BNLL);
            BNLLLayer<T> layer = new BNLLLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottom = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());

            for (int i = 0; i < Top.count(); i++)
            {
                double dfBottom = rgBottom[i];
                double dfTop = rgTop[i];

                m_log.CHECK_GE(dfTop, 0.0, "The top value at " + i.ToString() + " must be greater than or equal to zero.");
                m_log.CHECK_GE(dfTop, dfBottom, "The top value at " + i.ToString() + " must be greater than or equal to the bottom value.");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BNLL);
            BNLLLayer<T> layer = new BNLLLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }


    class PReLULayerTest : TestBase
    {
        public PReLULayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PReLU Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PReLULayerTest<double>(strName, nDeviceID, engine);
            else
                return new PReLULayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IPReLULayerTest : INeuronLayerTest
    {
        void TestPReLUParam();
        void TestForwardChannelShared();
        void TestGradientChannelShared();
        void TestConsistencyReLU();
        void TestPReLUInPlace();
    }

    class PReLULayerTest<T> : TestEx<T>, IPReLULayerTest
    {
        public PReLULayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestPReLU(PReLULayer<T> layer)
        {
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());
            double[] rgSlopeData = convert(layer.blobs[0].update_cpu_data());
            int nHw = Bottom.height * Bottom.width;
            int nChannels = Bottom.channels;
            bool bChannelShared = layer.layer_param.prelu_param.channel_shared;

            for (int i = 0; i < Bottom.count(); i++)
            {
                int c = (bChannelShared) ? 0 : (i / nHw) % nChannels;
                double dfExpected = Math.Max(rgBottomData[i], 0) + rgSlopeData[c] * Math.Min(rgBottomData[i], 0);

                m_log.EXPECT_EQUAL<float>(rgTopData[i], dfExpected, "expected top[" + i.ToString() + "] to equal " + dfExpected.ToString() + "!");
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRELU);
            PReLULayer<T> layer = new PReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            FillerParameter filler_param = new FillerParameter();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);
            filler.Fill(layer.blobs[0]);

            TestPReLU(layer);
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRELU);
            PReLULayer<T> layer = new PReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            FillerParameter filler_param = new FillerParameter();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, filler_param);
            filler.Fill(layer.blobs[0]);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestPReLUParam()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRELU);
            PReLULayer<T> layer = new PReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            double[] rgSlopeData = convert(layer.blobs[0].update_cpu_data());
            int nCount = layer.blobs[0].count();

            for (int i = 0; i < nCount; i++)
            {
                double dfVal = rgSlopeData[i];

                m_log.CHECK_EQ(dfVal, 0.25, "The slope value at " + i.ToString() + " is expected to be 0.25.");
            }
        }

        public void TestForwardChannelShared()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRELU);
            p.prelu_param.channel_shared = true;
            PReLULayer<T> layer = new PReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            TestPReLU(layer);
        }

        public void TestGradientChannelShared()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRELU);
            p.prelu_param.channel_shared = true;
            PReLULayer<T> layer = new PReLULayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestConsistencyReLU()
        {
            LayerParameter prelu_layer_param = new LayerParameter(LayerParameter.LayerType.PRELU);
            LayerParameter relu_layer_param = new LayerParameter(LayerParameter.LayerType.RELU);
            relu_layer_param.relu_param.negative_slope = 0.25;

            PReLULayer<T> prelu = new PReLULayer<T>(m_cuda, m_log, prelu_layer_param);
            ReLULayer<T> relu = new ReLULayer<T>(m_cuda, m_log, relu_layer_param);

            // Setup blobs
            BlobCollection<T> colBottomVec2 = new BlobCollection<T>();
            BlobCollection<T> colTopVec2 = new BlobCollection<T>();
            Blob<T> blob_bottom2 = new Blob<T>(m_cuda, m_log);
            Blob<T> blob_top2 = new Blob<T>(m_cuda, m_log);
            colBottomVec2.Add(blob_bottom2);
            colTopVec2.Add(blob_top2);
            blob_bottom2.CopyFrom(Bottom, false, true);

            // Setup layers
            prelu.Setup(BottomVec, TopVec);
            relu.Setup(colBottomVec2, colTopVec2);

            // Check forward
            prelu.Forward(BottomVec, TopVec);
            relu.Forward(colBottomVec2, colTopVec2);

            double[] rgTopData1 = convert(Top.update_cpu_data());
            double[] rgTopData2 = convert(blob_top2.update_cpu_data());

            for (int i = 0; i < blob_top2.count(); i++)
            {
                double dfTop1 = rgTopData1[i];
                double dfTop2 = rgTopData2[i];

                m_log.CHECK_EQ(dfTop1, dfTop2, "The top values do not match at " + i.ToString());
            }

            // Check backward
            Blob<T> tmp_blob = new Blob<T>(m_cuda, m_log, blob_top2);
            FillerParameter fillerParam = new FillerParameter();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fillerParam);

            filler.Fill(tmp_blob);
            m_cuda.copy(blob_top2.count(), tmp_blob.gpu_data, Top.mutable_gpu_diff);
            m_cuda.copy(blob_top2.count(), tmp_blob.gpu_data, blob_top2.mutable_gpu_diff);

            List<bool> rgPropagateDown = new List<bool>() { true };

            prelu.Backward(TopVec, rgPropagateDown, BottomVec);
            relu.Backward(colTopVec2, rgPropagateDown, colBottomVec2);

            double[] rgBottomDiff1 = convert(Bottom.update_cpu_diff());
            double[] rgBottomDiff2 = convert(blob_bottom2.update_cpu_diff());

            for (int i = 0; i < blob_bottom2.count(); i++)
            {
                double dfBottom1 = rgBottomDiff1[i];
                double dfBottom2 = rgBottomDiff2[i];

                m_log.CHECK_EQ(dfBottom1, dfBottom2, "The bottom diff values do not match at " + i.ToString());
            }
        }

        public void TestPReLUInPlace()
        {
            // Set layer parameters;
            LayerParameter ip_layer_param = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            LayerParameter prelu_layer_param = new LayerParameter(LayerParameter.LayerType.PRELU);

            InnerProductParameter ip_param = ip_layer_param.inner_product_param;
            ip_param.weight_filler = new FillerParameter();
            ip_param.weight_filler.type = "gaussian";
            ip_param.num_output = 3;

            InnerProductLayer<T> ip = new InnerProductLayer<T>(m_cuda, m_log, ip_layer_param);
            PReLULayer<T> prelu = new PReLULayer<T>(m_cuda, m_log, prelu_layer_param);
            InnerProductLayer<T> ip2 = new InnerProductLayer<T>(m_cuda, m_log, ip_layer_param);
            PReLULayer<T> prelu2 = new PReLULayer<T>(m_cuda, m_log, prelu_layer_param);

            // Setup blobs
            BlobCollection<T> colBottomVec2 = new BlobCollection<T>();
            BlobCollection<T> colMiddleVec2 = new BlobCollection<T>();
            BlobCollection<T> colTopVec2 = new BlobCollection<T>();
            Blob<T> blob_bottom2 = new Blob<T>(m_cuda, m_log);
            Blob<T> blob_middle2 = new Blob<T>(m_cuda, m_log);
            Blob<T> blob_top2 = new Blob<T>(m_cuda, m_log);

            colBottomVec2.Add(blob_bottom2);
            colMiddleVec2.Add(blob_middle2);
            colTopVec2.Add(blob_top2);
            blob_bottom2.CopyFrom(Bottom, false, true);

            // Setup layers
            ip.Setup(BottomVec, TopVec);
            prelu.Setup(BottomVec, TopVec);
            ip2.Setup(colBottomVec2, colMiddleVec2);
            prelu2.Setup(colMiddleVec2, colTopVec2);
            m_cuda.copy(ip2.blobs[0].count(), ip.blobs[0].gpu_data, ip2.blobs[0].mutable_gpu_data);

            // Forward in-place
            ip.Forward(BottomVec, TopVec);
            prelu.Forward(TopVec, TopVec);

            // Forward non-in-place
            ip2.Forward(colBottomVec2, colMiddleVec2);
            prelu2.Forward(colMiddleVec2, colTopVec2);

            // Check numbers
            double[] rgTopData1 = convert(Top.update_cpu_data());
            double[] rgTopData2 = convert(blob_top2.update_cpu_data());

            for (int i = 0; i < blob_top2.count(); i++)
            {
                double dfTop1 = rgTopData1[i];
                double dfTop2 = rgTopData2[i];

                m_log.CHECK_EQ(dfTop1, dfTop2, "The two top values at " + i.ToString() + " do not match!");
            }

            // Fill top diff with random numbers.
            Blob<T> tmp_blob = new Blob<T>(m_cuda, m_log, blob_top2);
            FillerParameter fillerParam = new FillerParameter();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fillerParam);

            filler.Fill(tmp_blob);
            m_cuda.copy(blob_top2.count(), tmp_blob.gpu_data, Top.mutable_gpu_diff);
            m_cuda.copy(blob_top2.count(), tmp_blob.gpu_data, blob_top2.mutable_gpu_diff);

            // Backward in-place;
            List<bool> rgPropagateDown = new List<bool>() { true };
            prelu.Backward(TopVec, rgPropagateDown, TopVec);
            ip.Backward(TopVec, rgPropagateDown, BottomVec);

            // Backward non-in-place
            prelu2.Backward(colTopVec2, rgPropagateDown, colMiddleVec2);
            ip2.Backward(colMiddleVec2, rgPropagateDown, colBottomVec2);

            // Check numbers
            double[] rgBottomDiff1 = convert(Bottom.update_cpu_diff());
            double[] rgBottomDiff2 = convert(blob_bottom2.update_cpu_diff());

            for (int i = 0; i < blob_bottom2.count(); i++)
            {
                double dfBottomDiff1 = rgBottomDiff1[i];
                double dfBottomDiff2 = rgBottomDiff2[i];

                m_log.EXPECT_EQUAL<float>(dfBottomDiff1, dfBottomDiff2, "The bottom diffs do not match at " + i.ToString());
            }

            for (int j = 0; j < 2; j++)
            {
                double[] rgIPDiff1 = convert(ip.blobs[j].update_cpu_diff());
                double[] rgIPDiff2 = convert(ip2.blobs[j].update_cpu_diff());

                for (int i = 0; i < ip.blobs[j].count(); i++)
                {
                    double dfIPDiff1 = rgIPDiff1[i];
                    double dfIPDiff2 = rgIPDiff2[i];

                    m_log.EXPECT_EQUAL<float>(dfIPDiff1, dfIPDiff2, "The IP diffs at blob[" + j.ToString() + "] do not match at " + i.ToString());
                }
            }

            double[] rgPreluDiff1 = convert(prelu.blobs[0].update_cpu_diff());
            double[] rgPreluDiff2 = convert(prelu2.blobs[0].update_cpu_diff());

            for (int i = 0; i < prelu.blobs[0].count(); i++)
            {
                double dfPreluDiff1 = rgPreluDiff1[i];
                double dfPreluDiff2 = rgPreluDiff2[i];

                m_log.EXPECT_EQUAL<float>(dfPreluDiff1, dfPreluDiff2, "The prelu diffs do not match at " + i.ToString());
            }
        }
    }

    class GradientScaleLayerTest : TestBase
    {
        public GradientScaleLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("GradientScale Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GradientScaleLayerTest<double>(strName, nDeviceID, engine);
            else
                return new GradientScaleLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    interface IGradientScaleLayerTest : INeuronLayerTest
    {
        void TestSetup();
    }

    class GradientScaleLayerTest<T> : TestEx<T>, IGradientScaleLayerTest
    {
        CancelEvent m_evtCancel = new CancelEvent();
        Blob<T> m_blobTemp;
        Blob<T> m_blobCompare;
        int m_nIteration = 2;
        bool m_bGetIterationHit = false;

        public GradientScaleLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int> { 2, 3, 4, 5 }, nDeviceID)
        {
            m_blobTemp = new Blob<T>(m_cuda, m_log, m_blob_bottom);
            m_blobCompare = new Blob<T>(m_cuda, m_log, m_blob_bottom);
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_blobTemp.Dispose();
            m_blobCompare.Dispose();
            m_evtCancel.Dispose();
            base.dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRADIENTSCALER);

            m_log.CHECK(p.type == LayerParameter.LayerType.GRADIENTSCALER, "Wrong type in parameter, expected GRADIENTSCALER");
            m_log.CHECK(p.gradient_scale_param != null, "The gradient_scale_param should not be null.");
            m_log.CHECK_EQ(p.gradient_scale_param.lower_bound, 0, "The gradient scale layer lower bound should be 0");
            m_log.CHECK_EQ(p.gradient_scale_param.upper_bound, 1, "The gradient scale layer upper bound should be 1");
            m_log.CHECK_EQ(p.gradient_scale_param.alpha, 10, "The gradient scale layer alpha should be 10");
            m_log.CHECK_EQ(p.gradient_scale_param.max_iter, 1, "The gradient scale layer max iter should be 1");

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_evtCancel);
            layer.OnGetIteration += Layer_OnGetIteration;

            m_log.CHECK(layer.type == LayerParameter.LayerType.GRADIENTSCALER, "Wrong type in layer, expected GRADIENTSCALER");
            p = layer.layer_param;

            m_log.CHECK(p.type == LayerParameter.LayerType.GRADIENTSCALER, "Wrong type in parameter, expected GRADIENTSCALER");
            m_log.CHECK(p.gradient_scale_param != null, "The gradient_scale_param should not be null.");
            m_log.CHECK_EQ(p.gradient_scale_param.lower_bound, 0, "The gradient scale layer lower bound should be 0");
            m_log.CHECK_EQ(p.gradient_scale_param.upper_bound, 1, "The gradient scale layer upper bound should be 1");
            m_log.CHECK_EQ(p.gradient_scale_param.alpha, 10, "The gradient scale layer alpha should be 10");
            m_log.CHECK_EQ(p.gradient_scale_param.max_iter, 1, "The gradient scale layer max iter should be 1");

            m_bGetIterationHit = false;
            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(m_bGetIterationHit == true, "The OnGetIteration event was not fired!");
        }

        public void TestForward()
        {
            m_blobTemp.CopyFrom(m_blob_bottom, false, true);
            m_blobTemp.CopyFrom(m_blob_bottom, true, true);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRADIENTSCALER);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_evtCancel);
            layer.OnGetIteration += Layer_OnGetIteration;

            m_bGetIterationHit = false;
            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(m_bGetIterationHit == true, "The OnGetIteration event was not fired!");

            layer.Forward(BottomVec, TopVec);

            int nCount = m_blobTemp.count();
            m_cuda.sub(nCount, m_blob_top.gpu_data, m_blobTemp.gpu_data, m_blobCompare.mutable_gpu_data);
            m_cuda.sub(nCount, m_blob_top.gpu_diff, m_blobTemp.gpu_diff, m_blobCompare.mutable_gpu_diff);

            // There should be no change as the gradient scalar performs an 
            // identity transform on the forward pass.
            double dfDiff1 = convert(m_blobCompare.asum_data());
            m_log.CHECK_EQ(dfDiff1, 0, "The data asum should be 0.0");

            double dfDiff2 = convert(m_blobCompare.asum_diff());
            m_log.CHECK_EQ(dfDiff2, 0, "The data asum should be 0.0");
        }

        private void Layer_OnGetIteration(object sender, GetIterationArgs e)
        {
            e.SetIteration(Phase.TRAIN, m_nIteration);
            m_bGetIterationHit = true;
        }

        public void TestGradient()
        {
            Top.ReshapeLike(Bottom);
            m_filler.Fill(Top);
            m_cuda.copy(Top.count(), Top.gpu_data, Top.mutable_gpu_diff);
            m_blobTemp.CopyFrom(Top, false, true);
            m_blobTemp.CopyFrom(Top, true, true);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRADIENTSCALER);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_evtCancel);
            layer.OnGetIteration += Layer_OnGetIteration;

            m_bGetIterationHit = false;
            layer.Setup(BottomVec, TopVec);
            m_log.CHECK(m_bGetIterationHit == true, "The OnGetIteration event was not fired!");

            layer.Forward(BottomVec, TopVec);
            layer.Backward(TopVec, new List<bool> { true }, BottomVec);

            // Check values.
            double dfLowerBound = p.gradient_scale_param.lower_bound;
            double dfHeight = p.gradient_scale_param.upper_bound - dfLowerBound;
            double dfAlpha = p.gradient_scale_param.alpha;
            double dfProgress = Math.Min(1, (double)m_nIteration / p.gradient_scale_param.max_iter);
            double dfCoeff = 2.0 * dfHeight / (1.0 + Math.Exp(-dfAlpha * dfProgress)) - dfHeight + dfLowerBound;

            int nCount = m_blobTemp.count();
            // The diff should have been scaled by the -dfCoeff value.
            m_blobTemp.scale_diff(-dfCoeff);
            m_cuda.sub(nCount, m_blob_bottom.gpu_diff, m_blobTemp.gpu_diff, m_blobCompare.mutable_gpu_diff);
            double dfDiff2 = convert(m_blobCompare.asum_diff());
            m_log.CHECK_EQ(dfDiff2, 0, "The data asum should be 0.0");
        }
    }
}
